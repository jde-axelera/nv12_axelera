// YOLOv5s RGBA pipeline with simulated FPGA DMA integration.
//
// Demonstrates the full Option-A DMA-BUF flow:
//
//  Startup (once):
//    Two RGBA DMA-BUF source buffers (rgba_dma[0/1]) are allocated from
//    /dev/dma_heap/system. Their physical addresses are queried via
//    /proc/self/pagemap and printed. In production these addresses are
//    passed to the FPGA driver so the DMA engine knows where to write.
//
//  Per frame (double-buffered):
//    An async thread simulates the FPGA DMA engine by memcpy-ing a test
//    RGBA frame into rgba_dma[nxt].ptr (the CPU-visible window onto the
//    same physical memory the FPGA would DMA-write into). The CPU
//    preprocessing step then reads directly from that pointer — zero
//    copies from the FPGA's perspective.
//
//  Production delta (what changes when a real FPGA is connected):
//    Replace the memcpy block with:
//      fpga_trigger_frame(slot=nxt);          // tell FPGA to fill slot
//      fpga_wait_frame_done(slot=nxt);        // wait for DMA complete IRQ
//    Everything else (rgba_to_tensor, in_dma, axruntime) stays identical.
//
// Copyright Axelera AI, 2026

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <numeric>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "axruntime/axruntime.hpp"

#include "dmabuf.hpp"
#include "preprocess.hpp"
#include "yolo_decode.hpp"
#include "annotate.hpp"
#include "timer.hpp"

static constexpr float NMS_IOU_THR = 0.45f;
static constexpr int   MODEL_WH    = 640;

// ── Physical address helper ───────────────────────────────────────────────────
// Reads /proc/self/pagemap to translate a mmap'd virtual address to the
// underlying physical address. Requires the pages to be faulted in (call
// memset before this). Returns 0 if the page is not present or the kernel
// restricts pagemap PFN access (needs CAP_SYS_ADMIN / root on >= 4.0).
static uint64_t virt_to_phys(const void* vaddr)
{
    const uintptr_t virt = reinterpret_cast<uintptr_t>(vaddr);
    const off_t     off  = static_cast<off_t>((virt / 4096) * 8);

    int fd = ::open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) return 0;

    uint64_t entry = 0;
    bool ok = (::pread(fd, &entry, sizeof(entry), off) == 8);
    ::close(fd);

    if (!ok || !(entry & (1ULL << 63))) return 0;   // not present
    const uint64_t pfn = entry & ((1ULL << 55) - 1);
    return pfn * 4096 + (virt % 4096);
}

// ── FPGA simulator ────────────────────────────────────────────────────────────
// Encapsulates a single "DMA write" operation: copies src_rgba into the
// DMA-BUF pointer, returning the elapsed time in ms.
// In production, replace write_frame() body with:
//     fpga_trigger_frame(slot);
//     fpga_wait_frame_done(slot);   // blocks until DMA complete IRQ
struct FpgaSim {
    const uint8_t* src;   // test RGBA frame (repeated every call)
    size_t         bytes; // W*H*4

    double write_frame(void* dma_ptr) const {
        const auto t0 = Clock::now();
        std::memcpy(dma_ptr, src, bytes);
        return Ms(Clock::now() - t0).count();
    }
};

// ── Latency table ─────────────────────────────────────────────────────────────
static void print_latency_table(
    const SectionTimer& t_dma,
    const SectionTimer& t_pre,
    const SectionTimer& t_inf,
    const SectionTimer& t_dec,
    const SectionTimer& t_wall,
    int runs, bool use_dmabuf)
{
    const double fps = 1000.0 / t_wall.avg();

    std::printf("\n+--------------------------------------------------------------------+\n");
    std::printf("| LATENCY BREAKDOWN  (%d runs, %s, FPGA-sim DMA source)\n",
                runs, use_dmabuf ? "DMA-BUF" : "host-mem");
    std::printf("| NOTE: FPGA DMA sim runs in the async thread (overlaps inference).\n");
    std::printf("| In production the DMA is hardware-triggered; this row → ~0 ms.\n");
    std::printf("+------------------+----------+----------+----------+----------+\n");
    std::printf("| %-16s | %8s | %8s | %8s | %8s |\n",
                "Section", "avg ms", "min ms", "max ms", "p95 ms");
    std::printf("+------------------+----------+----------+----------+----------+\n");

    auto row = [](const SectionTimer& t) {
        std::printf("| %-16s | %8.3f | %8.3f | %8.3f | %8.3f |\n",
                    t.name.c_str(), t.avg(), t.min(), t.max(), t.p95());
    };
    row(t_dma);
    row(t_pre);
    row(t_inf);
    row(t_dec);
    row(t_wall);

    std::printf("+------------------+----------+----------+----------+----------+\n");
    std::printf("| Throughput (pipelined):  %.1f FPS\n", fps);
    std::printf("| Async thread total:      %.3f ms  (DMA sim + preprocess)\n",
                t_dma.avg() + t_pre.avg());
    std::printf("| Sequential latency:      %.3f ms  (all sections, non-overlapped)\n",
                t_dma.avg() + t_pre.avg() + t_inf.avg() + t_dec.avg());
    std::printf("+--------------------------------------------------------------------+\n\n");
}

// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    std::string model_path, image_path, labels_path;
    int warmup = 5, bench = 20;
    int rgba_w = 0, rgba_h = 0;
    std::string out_path;

    for (int i = 1; i < argc; ++i) {
        std::string s(argv[i]);
        if      (s.ends_with(".json"))             model_path  = s;
        else if (s.ends_with(".names") ||
                 s.ends_with(".txt"))              labels_path = s;
        else if (s.starts_with("--warmup="))       warmup   = std::stoi(s.substr(9));
        else if (s.starts_with("--runs="))         bench    = std::stoi(s.substr(7));
        else if (s.starts_with("--size="))
            std::sscanf(s.c_str() + 7, "%dx%d", &rgba_w, &rgba_h);
        else if (s.starts_with("--output="))       out_path = s.substr(9);
        else                                        image_path  = s;
    }

    if (model_path.empty()) {
        std::fprintf(stderr,
            "Usage: %s model.json [image] [labels.names] [--size=WxH]"
            " [--output=path] [--warmup=N] [--runs=N]\n"
            "  image    : JPEG/PNG or raw .rgba file (4 bytes/pixel)\n"
            "  --size   : raw RGBA frame dimensions\n", argv[0]);
        return 1;
    }

    std::vector<std::string> labels;
    if (!labels_path.empty()) {
        std::ifstream f(labels_path);
        for (std::string l; std::getline(f, l);) labels.push_back(l);
    }

    // ── Runtime setup ─────────────────────────────────────────────────────────
    auto ctx = axr::to_ptr(axr_create_context());
    axr_set_logger(ctx.get(), AXR_LOG_ERROR,
        [](void*, axrLogLevel, const char* m){ fputs(m, stderr); fputc('\n', stderr); },
        nullptr);

    auto* model = axr_load_model(ctx.get(), model_path.c_str());
    if (!model) {
        std::fprintf(stderr, "[ERROR] %s\n", axr_last_error_string(AXR_OBJECT(ctx.get())));
        return 1;
    }

    const size_t n_in  = axr_num_model_inputs(model);
    const size_t n_out = axr_num_model_outputs(model);
    std::vector<axrTensorInfo> in_info(n_in), out_info(n_out);
    for (size_t i = 0; i < n_in;  ++i) in_info[i]  = axr_get_model_input(model, i);
    for (size_t i = 0; i < n_out; ++i) out_info[i] = axr_get_model_output(model, i);

    std::printf("[INFO] Inputs=%zu Outputs=%zu\n", n_in, n_out);
    for (size_t i = 0; i < n_in; ++i) {
        auto& t = in_info[i];
        std::printf("[INFO] Input[%zu] shape=[", i);
        for (size_t d = 0; d < t.ndims; ++d)
            std::printf("%lu%s", (unsigned long)t.dims[d], d+1 < t.ndims ? "," : "");
        std::printf("] scale=%g zp=%d\n", t.scale, t.zero_point);
    }

    // ── Load image, determine frame dimensions ─────────────────────────────────
    cv::Mat vis_bgr;
    int src_w = 0, src_h = 0;
    std::string out_jpg;
    std::vector<uint8_t> src_rgba_data;   // canonical RGBA frame for FPGA sim

    if (!image_path.empty()) {
        const auto ext = image_path.substr(image_path.rfind('.') + 1);

        if (ext == "rgba") {
            src_w = rgba_w > 0 ? rgba_w : 1920;
            src_h = rgba_h > 0 ? rgba_h : 1080;
            if (rgba_w == 0) {
                auto us = image_path.rfind('_');
                if (us != std::string::npos)
                    std::sscanf(image_path.c_str() + us + 1, "%dx%d", &src_w, &src_h);
            }
            src_rgba_data.resize(static_cast<size_t>(src_w) * src_h * 4);
            std::ifstream f(image_path, std::ios::binary);
            if (!f) { std::fprintf(stderr, "[ERROR] Cannot open %s\n", image_path.c_str()); return 1; }
            f.read(reinterpret_cast<char*>(src_rgba_data.data()),
                   static_cast<std::streamsize>(src_rgba_data.size()));
            cv::Mat rm(src_h, src_w, CV_8UC4, src_rgba_data.data());
            cv::cvtColor(rm, vis_bgr, cv::COLOR_RGBA2BGR);
            out_jpg = out_path.empty()
                ? image_path.substr(0, image_path.rfind('.')) + "_detections.jpg" : out_path;
        } else {
            cv::Mat img = cv::imread(image_path);
            if (img.empty()) { std::fprintf(stderr, "[ERROR] Cannot read %s\n", image_path.c_str()); return 1; }
            src_w = img.cols; src_h = img.rows;
            vis_bgr = img.clone();
            cv::Mat rm; cv::cvtColor(img, rm, cv::COLOR_BGR2RGBA);
            src_rgba_data.assign(rm.data, rm.data + static_cast<size_t>(src_w) * src_h * 4);
            out_jpg = out_path.empty()
                ? image_path.substr(0, image_path.rfind('.')) + "_detections.jpg" : out_path;
        }
    } else {
        src_w = 640; src_h = 640;
        src_rgba_data.assign(static_cast<size_t>(src_w) * src_h * 4, 114);
        vis_bgr = cv::Mat(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));
        out_jpg = out_path.empty() ? "synthetic_detections.jpg" : out_path;
        std::printf("[INFO] No image -- using synthetic 640x640 RGBA test pattern\n");
    }
    std::printf("[INFO] Source frame: %dx%d RGBA (%zu KB)\n",
                src_w, src_h, src_rgba_data.size() / 1024);

    const size_t rgba_frame_bytes = static_cast<size_t>(src_w) * src_h * 4;

    // ── DMA heap setup ────────────────────────────────────────────────────────
    int  heap_fd    = ::open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
    bool use_dmabuf = (heap_fd >= 0);

    // ── rgba_dma[2]: FPGA writes HERE (Option-A source buffers) ───────────────
    // These are separate from the tensor buffers. In production the physical
    // addresses below are given to the FPGA driver once at startup.
    std::array<DmaBuf, 2>                    rgba_dma;
    std::array<void*, 2>                     rgba_ptrs{nullptr, nullptr};
    std::array<std::unique_ptr<uint8_t[]>,2> rgba_heap;

    std::printf("\n[FPGA-SIM] Allocating 2x RGBA source buffers (%zu KB each)...\n",
                rgba_frame_bytes / 1024);

    for (int b = 0; b < 2; ++b) {
        if (use_dmabuf) {
            rgba_dma[b] = DmaBuf::alloc(heap_fd, rgba_frame_bytes);
            if (rgba_dma[b].valid()) {
                rgba_ptrs[b] = rgba_dma[b].ptr;
            } else {
                use_dmabuf = false;
            }
        }
        if (!use_dmabuf || rgba_ptrs[b] == nullptr) {
            rgba_heap[b] = std::make_unique<uint8_t[]>(rgba_frame_bytes);
            rgba_ptrs[b] = rgba_heap[b].get();
        }
        // Fault all pages in before querying physical address
        std::memset(rgba_ptrs[b], 0, rgba_frame_bytes);

        const uint64_t phys = virt_to_phys(rgba_ptrs[b]);
        if (phys) {
            std::printf("[FPGA-SIM] Slot %d:  virt=%-18p  phys=0x%012lx  (%zu KB)\n",
                        b, rgba_ptrs[b], (unsigned long)phys, rgba_frame_bytes / 1024);
            std::printf("[FPGA-SIM]          → production: fpga_set_dma_target("
                        "slot=%d, phys=0x%lx, stride=%d)\n",
                        b, (unsigned long)phys, src_w * 4);
        } else {
            std::printf("[FPGA-SIM] Slot %d:  virt=%-18p  phys=N/A "
                        "(needs CAP_SYS_ADMIN / root to read PFN)\n",
                        b, rgba_ptrs[b]);
            std::printf("[FPGA-SIM]          → run as root to obtain physical address "
                        "for FPGA driver\n");
        }
    }

    // ── in_dma[2]: tensor buffers — these go to axruntime ────────────────────
    std::array<std::vector<DmaBuf>, 2>                    in_dma;
    std::array<std::vector<std::unique_ptr<int8_t[]>>, 2> in_host_bufs;
    std::array<std::vector<axrArgument>, 2>               in_args;
    std::array<int8_t*, 2>                                in_ptrs{nullptr, nullptr};

    for (int b = 0; b < 2; ++b) {
        in_dma[b].resize(n_in);
        in_args[b].resize(n_in);
    }

    std::vector<std::unique_ptr<int8_t[]>> out_host;
    std::vector<axrArgument>               out_args(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        out_host.emplace_back(new int8_t[axr_tensor_size(&out_info[i])]);
        out_args[i] = {out_host[i].get(), 0, 0, 0};
    }

    bool tensor_dmabuf = use_dmabuf;
    if (tensor_dmabuf) {
        for (int b = 0; b < 2 && tensor_dmabuf; ++b) {
            for (size_t i = 0; i < n_in; ++i) {
                in_dma[b][i] = DmaBuf::alloc(heap_fd, axr_tensor_size(&in_info[i]));
                if (!in_dma[b][i].valid()) { tensor_dmabuf = false; break; }
                in_args[b][i] = {nullptr, in_dma[b][i].fd, 0, 0};
            }
            if (tensor_dmabuf)
                in_ptrs[b] = static_cast<int8_t*>(in_dma[b][0].ptr);
        }
        if (!tensor_dmabuf)
            for (int b = 0; b < 2; ++b) for (auto& d : in_dma[b]) d.release();
    }
    if (!tensor_dmabuf) {
        std::fprintf(stderr, "[WARN] Tensor DMA-BUF unavailable, using host memory\n");
        for (int b = 0; b < 2; ++b) {
            for (size_t i = 0; i < n_in; ++i) {
                in_host_bufs[b].emplace_back(new int8_t[axr_tensor_size(&in_info[i])]);
                in_args[b][i] = {in_host_bufs[b][i].get(), 0, 0, 0};
            }
            in_ptrs[b] = in_host_bufs[b][0].get();
        }
    }

    std::printf("\n[INFO] Buffer layout:\n");
    std::printf("  rgba_dma[0] ptr=%-18p  ← FPGA DMA target, slot 0\n", rgba_ptrs[0]);
    std::printf("  rgba_dma[1] ptr=%-18p  ← FPGA DMA target, slot 1\n", rgba_ptrs[1]);
    std::printf("  in_dma[0]   ptr=%-18p  ← tensor input to axruntime, slot 0\n", (void*)in_ptrs[0]);
    std::printf("  in_dma[1]   ptr=%-18p  ← tensor input to axruntime, slot 1\n", (void*)in_ptrs[1]);

    // ── Device + instance ──────────────────────────────────────────────────────
    auto* conn = axr_device_connect(ctx.get(), nullptr, 1, nullptr);
    if (!conn) {
        std::fprintf(stderr, "[ERROR] %s\n", axr_last_error_string(AXR_OBJECT(ctx.get())));
        return 1;
    }
    const std::string prop_str =
        std::string(tensor_dmabuf ? "input_dmabuf=1" : "input_dmabuf=0")
        + ";output_dmabuf=0;num_sub_devices=1;aipu_cores=1;double_buffer=0;elf_in_ddr=1";
    auto* props    = axr_create_properties(ctx.get(), prop_str.c_str());
    auto* instance = axr_load_model_instance(conn, model, props);
    if (!instance) {
        std::fprintf(stderr, "[ERROR] %s\n", axr_last_error_string(AXR_OBJECT(ctx.get())));
        return 1;
    }
    std::printf("[INFO] Properties: %s\n\n", prop_str.c_str());

    // ── FPGA simulator ────────────────────────────────────────────────────────
    FpgaSim fpga{src_rgba_data.data(), rgba_frame_bytes};

    // ── Warmup: prime slot 0 synchronously ────────────────────────────────────
    // Simulate: FPGA fills rgba_dma[0] → CPU preprocesses → tensor in in_dma[0]
    fpga.write_frame(rgba_ptrs[0]);
    rgba_to_tensor(static_cast<const uint8_t*>(rgba_ptrs[0]),
                   src_w, src_h, in_ptrs[0], in_info[0]);

    std::printf("[INFO] Warming up (%d runs)...\n", warmup);
    for (int i = 0; i < warmup; ++i)
        axr_run_model_instance(instance, in_args[0].data(), n_in, out_args.data(), n_out);

    // ── Benchmark: double-buffered FPGA-sim + preprocess + inference ──────────
    //
    //  Iteration i:
    //    async thread:  fpga.write_frame(rgba_ptrs[nxt])   ← simulates DMA write
    //                   rgba_to_tensor(rgba_ptrs[nxt], ..., in_ptrs[nxt])
    //    main thread:   axr_run_model_instance(in_args[cur])
    //    main thread:   future.get() + decode + NMS
    //
    //  In production the async thread becomes:
    //    fpga_trigger_frame(nxt);          // non-blocking ioctl
    //    fpga_wait_frame_done(nxt);        // blocks until DMA IRQ
    //    rgba_to_tensor(rgba_ptrs[nxt], ..., in_ptrs[nxt])
    //
    std::printf("[INFO] Benchmarking (%d runs, FPGA-sim DMA + double-buffered pipeline)...\n\n", bench);

    SectionTimer t_dma {"FPGA DMA sim"};
    SectionTimer t_pre {"Preprocess RGBA"};
    SectionTimer t_inf {"Inference (AIPU)"};
    SectionTimer t_dec {"Decode + NMS"};
    SectionTimer t_wall{"Frame wall time"};

    for (int i = 0; i < bench; ++i) {
        const int cur = i & 1;
        const int nxt = cur ^ 1;

        const auto iter_t0 = Clock::now();

        // Capture by value — safe across iterations
        void*                nxt_rgba   = rgba_ptrs[nxt];
        int8_t*              nxt_tensor = in_ptrs[nxt];
        const axrTensorInfo& tinfo      = in_info[0];
        const FpgaSim&       sim        = fpga;

        // Async: FPGA sim writes nxt slot, then CPU preprocesses nxt
        auto pre_fut = std::async(std::launch::async,
            [&sim, nxt_rgba, src_w, src_h, nxt_tensor, &tinfo]
            () -> std::pair<double, double>
        {
            // ── [1] Simulate FPGA DMA write ─────────────────────────────────
            // Production replacement:
            //   fpga_trigger_frame(nxt);
            //   fpga_wait_frame_done(nxt);
            const double dma_ms = sim.write_frame(nxt_rgba);

            // ── [2] CPU preprocess: reads from DMA-BUF, writes to tensor buf ─
            const auto t0 = Clock::now();
            rgba_to_tensor(static_cast<const uint8_t*>(nxt_rgba),
                           src_w, src_h, nxt_tensor, tinfo);
            const double pre_ms = Ms(Clock::now() - t0).count();

            return {dma_ms, pre_ms};
        });

        // Main thread: inference on cur (overlaps the async thread above)
        {
            ScopeTimer st(t_inf);
            if (axr_run_model_instance(instance,
                    in_args[cur].data(), n_in, out_args.data(), n_out) != AXR_SUCCESS) {
                std::fprintf(stderr, "[ERROR] %s\n",
                    axr_last_error_string(AXR_OBJECT(ctx.get())));
                return 1;
            }
        }

        auto [dma_ms, pre_ms] = pre_fut.get();
        t_dma.record(dma_ms);
        t_pre.record(pre_ms);

        {
            ScopeTimer st(t_dec);
            auto grid_to_sid = [](size_t h) -> int {
                if (h >= 70) return 0;
                if (h >= 35) return 1;
                return 2;
            };
            std::vector<Det> tmp;
            for (size_t j = 0; j < n_out; ++j)
                decode_head(out_host[j].get(), out_info[j],
                            grid_to_sid(out_info[j].dims[1]), tmp);
            nms(std::move(tmp), NMS_IOU_THR);
        }

        t_wall.record(Ms(Clock::now() - iter_t0).count());
    }

    print_latency_table(t_dma, t_pre, t_inf, t_dec, t_wall, bench, tensor_dmabuf);

    // ── Final inference for annotation ─────────────────────────────────────────
    fpga.write_frame(rgba_ptrs[0]);
    rgba_to_tensor(static_cast<const uint8_t*>(rgba_ptrs[0]),
                   src_w, src_h, in_ptrs[0], in_info[0]);
    axr_run_model_instance(instance, in_args[0].data(), n_in, out_args.data(), n_out);

    auto grid_to_sid = [](size_t h) -> int {
        if (h >= 70) return 0;
        if (h >= 35) return 1;
        return 2;
    };
    std::vector<Det> all_dets;
    for (size_t i = 0; i < n_out; ++i)
        decode_head(out_host[i].get(), out_info[i],
                    grid_to_sid(out_info[i].dims[1]), all_dets);
    std::printf("[PRE-NMS] %zu candidates\n", all_dets.size());
    auto final_dets = nms(std::move(all_dets), NMS_IOU_THR);

    std::printf("[DETECTIONS] %zu object(s):\n", final_dets.size());
    for (auto& d : final_dets) {
        const std::string name = d.cls < static_cast<int>(labels.size())
            ? labels[d.cls] : "cls" + std::to_string(d.cls);
        std::printf("  %-20s  conf=%.3f  box(640sp)=[%d,%d,%d,%d]\n",
            name.c_str(), d.conf,
            static_cast<int>(d.x1), static_cast<int>(d.y1),
            static_cast<int>(d.x2), static_cast<int>(d.y2));
    }

    if (!vis_bgr.empty() && !out_jpg.empty())
        save_annotated(vis_bgr, final_dets, labels, out_jpg, MODEL_WH);

    // ── Cleanup ────────────────────────────────────────────────────────────────
    for (int b = 0; b < 2; ++b) {
        rgba_dma[b].release();
        if (tensor_dmabuf) for (auto& d : in_dma[b]) d.release();
    }
    if (heap_fd >= 0) ::close(heap_fd);
    return 0;
}
