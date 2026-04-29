// EVS — YOLOv5s object detection with NV12 input and DMA-BUF passthrough
// Targets the Axelera Metis device via axruntime.
// Copyright EVS / Axelera AI, 2026

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <numeric>
#include <span>
#include <string>
#include <vector>

#include "axruntime/axruntime.hpp"

#include "dmabuf.hpp"
#include "preprocess.hpp"
#include "yolo_decode.hpp"
#include "annotate.hpp"
#include "timer.hpp"

static constexpr float NMS_IOU_THR = 0.45f;
static constexpr int   MODEL_WH    = 640;

static void print_latency_table(
    const SectionTimer& t_pre,
    const SectionTimer& t_inf,
    const SectionTimer& t_dec,
    const SectionTimer& t_wall,
    int runs,
    bool use_dmabuf)
{
    const double fps = 1000.0 / t_wall.avg();

    std::printf("\n+--------------------------------------------------------------------+\n");
    std::printf("| LATENCY BREAKDOWN  (%d runs, %s, double-buffered pipeline)\n",
                runs, use_dmabuf ? "DMA-BUF input" : "host-mem input");
    std::printf("+------------------+----------+----------+----------+----------+\n");
    std::printf("| %-16s | %8s | %8s | %8s | %8s |\n",
                "Section", "avg ms", "min ms", "max ms", "p95 ms");
    std::printf("+------------------+----------+----------+----------+----------+\n");

    auto row = [](const SectionTimer& t) {
        std::printf("| %-16s | %8.3f | %8.3f | %8.3f | %8.3f |\n",
                    t.name.c_str(), t.avg(), t.min(), t.max(), t.p95());
    };
    row(t_pre);
    row(t_inf);
    row(t_dec);
    row(t_wall);

    std::printf("+------------------+----------+----------+----------+----------+\n");
    std::printf("| Throughput (pipelined):  %.1f FPS\n", fps);
    std::printf("| Sequential latency:      %.3f ms  (pre+inf+dec, non-overlapped)\n",
                t_pre.avg() + t_inf.avg() + t_dec.avg());
    std::printf("+--------------------------------------------------------------------+\n\n");
}

int main(int argc, char** argv)
{
    std::string model_path, image_path, labels_path;
    int warmup = 5, bench = 20;
    int nv12_w = 0, nv12_h = 0;
    std::string out_path;

    for (int i = 1; i < argc; ++i) {
        std::string s(argv[i]);
        if (s.ends_with(".json"))
            model_path  = s;
        else if (s.ends_with(".names") || s.ends_with(".txt"))
            labels_path = s;
        else if (s.starts_with("--warmup="))
            warmup = std::stoi(s.substr(9));
        else if (s.starts_with("--runs="))
            bench  = std::stoi(s.substr(7));
        else if (s.starts_with("--size="))
            std::sscanf(s.c_str() + 7, "%dx%d", &nv12_w, &nv12_h);
        else if (s.starts_with("--output="))
            out_path = s.substr(9);
        else
            image_path  = s;
    }

    if (model_path.empty()) {
        std::fprintf(stderr,
            "Usage: %s model.json [image] [labels.names] [--size=WxH]"
            " [--output=path] [--warmup=N] [--runs=N]\n"
            "  image       : JPEG/PNG or raw .yuv/.nv12 file\n"
            "  --size=WxH  : NV12/YUV dimensions, e.g. --size=176x144\n"
            "  --output    : output JPEG path\n"
            "  --warmup=N  : warmup iterations (default 5)\n"
            "  --runs=N    : benchmark iterations (default 20)\n",
            argv[0]);
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
            std::printf("%lu%s", (unsigned long)t.dims[d], d + 1 < t.ndims ? "," : "");
        std::printf("] scale=%g zp=%d\n", t.scale, t.zero_point);
    }
    for (size_t i = 0; i < n_out; ++i) {
        auto& t = out_info[i];
        std::printf("[INFO] Output[%zu] %s shape=[", i, t.name);
        for (size_t d = 0; d < t.ndims; ++d)
            std::printf("%lu%s", (unsigned long)t.dims[d], d + 1 < t.ndims ? "," : "");
        std::printf("] scale=%g zp=%d\n", t.scale, t.zero_point);
    }

    // ── Double-buffered DMA-BUF / host memory ─────────────────────────────────
    // Two input buffers: while AIPU processes buf[cur], CPU preprocesses into
    // buf[nxt], hiding the ~2ms preprocess behind the ~6ms inference.
    int heap_fd = ::open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
    bool use_dmabuf = (heap_fd >= 0);

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

    if (use_dmabuf) {
        for (int b = 0; b < 2 && use_dmabuf; ++b) {
            for (size_t i = 0; i < n_in; ++i) {
                in_dma[b][i] = DmaBuf::alloc(heap_fd, axr_tensor_size(&in_info[i]));
                if (!in_dma[b][i].valid()) { use_dmabuf = false; break; }
                in_args[b][i] = {nullptr, in_dma[b][i].fd, 0, 0};
            }
            if (use_dmabuf)
                in_ptrs[b] = static_cast<int8_t*>(in_dma[b][0].ptr);
        }
        if (!use_dmabuf)
            for (int b = 0; b < 2; ++b) for (auto& d : in_dma[b]) d.release();
    }
    if (!use_dmabuf) {
        std::fprintf(stderr, "[WARN] DMA-BUF heap unavailable, using host memory\n");
        for (int b = 0; b < 2; ++b) {
            for (size_t i = 0; i < n_in; ++i) {
                in_host_bufs[b].emplace_back(new int8_t[axr_tensor_size(&in_info[i])]);
                in_args[b][i] = {in_host_bufs[b][i].get(), 0, 0, 0};
            }
            in_ptrs[b] = in_host_bufs[b][0].get();
        }
    } else {
        std::printf("[DMA-BUF] 2x input buffers via /dev/dma_heap/system (pipelined)\n");
    }

    // ── Device + instance ──────────────────────────────────────────────────────
    auto* conn = axr_device_connect(ctx.get(), nullptr, 1, nullptr);
    if (!conn) {
        std::fprintf(stderr, "[ERROR] %s\n", axr_last_error_string(AXR_OBJECT(ctx.get())));
        return 1;
    }
    const std::string prop_str =
        std::string(use_dmabuf ? "input_dmabuf=1" : "input_dmabuf=0")
        + ";output_dmabuf=0;num_sub_devices=1;aipu_cores=1;double_buffer=0;elf_in_ddr=1";
    auto* props    = axr_create_properties(ctx.get(), prop_str.c_str());
    auto* instance = axr_load_model_instance(conn, model, props);
    if (!instance) {
        std::fprintf(stderr, "[ERROR] %s\n", axr_last_error_string(AXR_OBJECT(ctx.get())));
        return 1;
    }
    std::printf("[INFO] Properties: %s\n", prop_str.c_str());

    // ── Load input image into buf[0] ───────────────────────────────────────────
    int8_t* in_ptr0 = in_ptrs[0];
    cv::Mat vis_bgr;
    int orig_w = 0, orig_h = 0;
    std::string out_jpg;

    if (!image_path.empty()) {
        const auto ext = image_path.substr(image_path.rfind('.') + 1);

        if (ext == "nv12" || ext == "yuv") {
            int sw = nv12_w > 0 ? nv12_w : 1920;
            int sh = nv12_h > 0 ? nv12_h : 1080;
            if (nv12_w == 0) {
                auto us = image_path.rfind('_');
                if (us != std::string::npos)
                    std::sscanf(image_path.c_str() + us + 1, "%dx%d", &sw, &sh);
            }
            const size_t frame_bytes = static_cast<size_t>(sw) * sh * 3 / 2;
            std::vector<uint8_t> nv12(frame_bytes);
            std::ifstream f(image_path, std::ios::binary);
            if (!f) {
                std::fprintf(stderr, "[ERROR] Cannot open: %s\n", image_path.c_str());
                return 1;
            }
            f.read(reinterpret_cast<char*>(nv12.data()),
                   static_cast<std::streamsize>(frame_bytes));
            std::printf("[INFO] NV12 input: %dx%d (%zu KB, first frame)\n",
                        sw, sh, frame_bytes / 1024);

            cv::Mat yuv(sh + sh / 2, sw, CV_8UC1, nv12.data());
            cv::cvtColor(yuv, vis_bgr, cv::COLOR_YUV2BGR_NV12);
            orig_w = sw; orig_h = sh;

            nv12_to_tensor(nv12.data(), sw, sh, in_ptr0, in_info[0]);
            out_jpg = out_path.empty()
                ? image_path.substr(0, image_path.rfind('.')) + "_detections.jpg"
                : out_path;

        } else {
            cv::Mat img = cv::imread(image_path);
            if (img.empty()) {
                std::fprintf(stderr, "[ERROR] Cannot read: %s\n", image_path.c_str());
                return 1;
            }
            if (img.cols % 2 != 0 || img.rows % 2 != 0)
                cv::resize(img, img,
                    cv::Size(img.cols + img.cols % 2, img.rows + img.rows % 2));
            orig_w = img.cols; orig_h = img.rows;
            vis_bgr = img.clone();
            std::printf("[INFO] Image %dx%d decoded and converted via NV12 pipeline\n",
                        orig_w, orig_h);

            cv::Mat i420;
            cv::cvtColor(img, i420, cv::COLOR_BGR2YUV_I420);
            std::vector<uint8_t> nv12(static_cast<size_t>(orig_w) * orig_h * 3 / 2);
            std::memcpy(nv12.data(), i420.data,
                        static_cast<size_t>(orig_w) * orig_h);
            const uint8_t* u = i420.data + orig_w * orig_h;
            const uint8_t* v = u + orig_w * orig_h / 4;
            uint8_t* uv = nv12.data() + orig_w * orig_h;
            for (int k = 0; k < orig_w * orig_h / 4; ++k) {
                uv[2*k] = u[k]; uv[2*k+1] = v[k];
            }
            nv12_to_tensor(nv12.data(), orig_w, orig_h, in_ptr0, in_info[0]);
            out_jpg = out_path.empty()
                ? image_path.substr(0, image_path.rfind('.')) + "_detections.jpg"
                : out_path;
        }
    } else {
        std::printf("[INFO] No image -- using synthetic NV12 640x640 test pattern\n");
        orig_w = 640; orig_h = 640;
        std::vector<uint8_t> nv12(640 * 640 * 3 / 2, 114);
        nv12_to_tensor(nv12.data(), 640, 640, in_ptr0, in_info[0]);
        vis_bgr = cv::Mat(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));
        out_jpg = out_path.empty() ? "synthetic_detections.jpg" : out_path;
    }

    // ── Warmup ─────────────────────────────────────────────────────────────────
    std::printf("[INFO] Warming up (%d runs)...\n", warmup);
    for (int i = 0; i < warmup; ++i)
        axr_run_model_instance(instance,
            in_args[0].data(), n_in, out_args.data(), n_out);

    // ── Double-buffer pipelined benchmark ─────────────────────────────────────
    // Pattern per iteration i:
    //   Thread: preprocess(nv12) -> buf[nxt]     (~2 ms, overlaps inference)
    //   Main:   axr_run_model_instance(buf[cur]) (~6 ms, blocking)
    //   Main:   future.get()  (thread already done since 2ms < 6ms)
    //   Main:   decode + NMS                     (~0.1 ms)
    //
    // Steady-state frame time = max(preprocess, inference) ~= 6 ms -> ~160 FPS
    std::printf("[INFO] Benchmarking (%d runs, pipelined preprocess+inference)...\n", bench);

    SectionTimer t_pre{"Preprocess NV12"};
    SectionTimer t_inf{"Inference (AIPU)"};
    SectionTimer t_dec{"Decode + NMS"};
    SectionTimer t_wall{"Frame wall time"};

    std::vector<uint8_t> nv12_bench;
    int bench_w = orig_w, bench_h = orig_h;
    {
        cv::Mat src = vis_bgr.empty()
            ? cv::Mat(640, 640, CV_8UC3, cv::Scalar(114,114,114))
            : vis_bgr;
        if (src.cols % 2 != 0 || src.rows % 2 != 0)
            cv::resize(src, src, cv::Size(src.cols + src.cols%2, src.rows + src.rows%2));
        bench_w = src.cols; bench_h = src.rows;
        cv::Mat i420;
        cv::cvtColor(src, i420, cv::COLOR_BGR2YUV_I420);
        nv12_bench.resize(static_cast<size_t>(bench_w) * bench_h * 3 / 2);
        std::memcpy(nv12_bench.data(), i420.data,
                    static_cast<size_t>(bench_w) * bench_h);
        const uint8_t* u = i420.data + bench_w * bench_h;
        const uint8_t* v = u + bench_w * bench_h / 4;
        uint8_t* uv = nv12_bench.data() + bench_w * bench_h;
        for (int k = 0; k < bench_w * bench_h / 4; ++k) {
            uv[2*k] = u[k]; uv[2*k+1] = v[k];
        }
    }

    // Prime buf[0] before the loop so iteration 0 can launch preprocess into buf[1]
    nv12_to_tensor(nv12_bench.data(), bench_w, bench_h, in_ptrs[0], in_info[0]);

    for (int i = 0; i < bench; ++i) {
        const int cur = i & 1;
        const int nxt = cur ^ 1;

        const auto iter_t0 = Clock::now();

        // Launch preprocess of next frame on a background thread
        const uint8_t*       src_data = nv12_bench.data();
        int8_t*              nxt_ptr  = in_ptrs[nxt];
        const axrTensorInfo& tinfo    = in_info[0];
        auto pre_fut = std::async(std::launch::async,
            [src_data, bench_w, bench_h, nxt_ptr, &tinfo]() -> double {
                const auto t0 = Clock::now();
                nv12_to_tensor(src_data, bench_w, bench_h, nxt_ptr, tinfo);
                return Ms(Clock::now() - t0).count();
            });

        // Run inference on buf[cur] -- overlaps with preprocess thread
        {
            ScopeTimer st(t_inf);
            if (axr_run_model_instance(instance,
                    in_args[cur].data(), n_in, out_args.data(), n_out) != AXR_SUCCESS) {
                std::fprintf(stderr, "[ERROR] %s\n",
                    axr_last_error_string(AXR_OBJECT(ctx.get())));
                return 1;
            }
        }

        t_pre.record(pre_fut.get());  // get() waits and returns thread duration

        // Decode + NMS
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

    print_latency_table(t_pre, t_inf, t_dec, t_wall, bench, use_dmabuf);

    // ── Final inference for annotation ─────────────────────────────────────────
    nv12_to_tensor(nv12_bench.data(), bench_w, bench_h, in_ptrs[0], in_info[0]);
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
    if (use_dmabuf) {
        for (int b = 0; b < 2; ++b) for (auto& d : in_dma[b]) d.release();
        ::close(heap_fd);
    }
    return 0;
}
