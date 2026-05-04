// YOLOv5s → ResNet50 cascaded detection + embedding pipeline.
//
// Stage 1 — Detection (YOLOv5s, 2 AIPU cores via version-2 model):
//   RGBA frame → preprocess → AIPU inference → bounding boxes
//
// Stage 2 — Embedding (ResNet50, 2 AIPU cores via version-2 model):
//   For each detected crop: resize → ImageNet normalise → AIPU inference
//   → 1024-dimensional embedding vector
//   Crops are batched in pairs; ResNet50 v2 processes 2 crops per call.
//
// Double-buffer optimisation (YOLO only):
//   Preprocessing frame N+1 into buf[nxt] overlaps YOLO inference on buf[cur].
//   ResNet50 runs after YOLO completes (cascade dependency).
//
// Copyright Axelera AI, 2026

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
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
static constexpr int   YOLO_WH     = 640;

// ── Model handle ─────────────────────────────────────────────────────────────
struct Model {
    axrModel*         model    = nullptr;
    axrModelInstance* instance = nullptr;
    size_t n_in = 0, n_out = 0;
    std::vector<axrTensorInfo>             in_info, out_info;
    std::vector<std::unique_ptr<int8_t[]>> out_host;
    std::vector<axrArgument>               out_args;

    bool load(axrContext* ctx, axrConnection* conn,
              const char* path, const std::string& props)
    {
        model = axr_load_model(ctx, path);
        if (!model) return false;
        n_in  = axr_num_model_inputs(model);
        n_out = axr_num_model_outputs(model);
        in_info.resize(n_in);  out_info.resize(n_out);
        for (size_t i = 0; i < n_in;  ++i) in_info[i]  = axr_get_model_input(model, i);
        for (size_t i = 0; i < n_out; ++i) out_info[i] = axr_get_model_output(model, i);
        auto* p = axr_create_properties(ctx, props.c_str());
        instance = axr_load_model_instance(conn, model, p);
        if (!instance) return false;
        for (size_t i = 0; i < n_out; ++i) {
            out_host.emplace_back(new int8_t[axr_tensor_size(&out_info[i])]);
            out_args.push_back({out_host[i].get(), 0, 0, 0});
        }
        return true;
    }

    void print_shapes(const char* tag) const {
        for (size_t i = 0; i < n_in; ++i) {
            auto& t = in_info[i];
            std::printf("[%s] Input[%zu]  shape=[", tag, i);
            for (size_t d = 0; d < t.ndims; ++d)
                std::printf("%lu%s", (unsigned long)t.dims[d], d+1<t.ndims?",":"");
            std::printf("] scale=%g zp=%d\n", t.scale, t.zero_point);
        }
        for (size_t i = 0; i < n_out; ++i) {
            auto& t = out_info[i];
            std::printf("[%s] Output[%zu] %s shape=[", tag, i, t.name);
            for (size_t d = 0; d < t.ndims; ++d)
                std::printf("%lu%s", (unsigned long)t.dims[d], d+1<t.ndims?",":"");
            std::printf("] scale=%g zp=%d\n", t.scale, t.zero_point);
        }
    }
};

// ── Latency table ─────────────────────────────────────────────────────────────
static void print_latency_table(
    const SectionTimer& t_pre,
    const SectionTimer& t_yolo,
    const SectionTimer& t_dec,
    const SectionTimer& t_resnet,
    const SectionTimer& t_wall,
    int runs, bool use_dmabuf, int yolo_cores, int resnet_cores)
{
    std::printf("\n+----------------------------------------------------------------------+\n");
    std::printf("| LATENCY BREAKDOWN  (%d runs, %s, YOLOv5s+ResNet50 cascade)\n",
                runs, use_dmabuf ? "DMA-BUF" : "host-mem");
    std::printf("| YOLOv5s: %d AIPU core(s) (v%d)    ResNet50: %d AIPU core(s) (v%d)\n",
                yolo_cores, yolo_cores, resnet_cores, resnet_cores);
    std::printf("+------------------+----------+----------+----------+----------+\n");
    std::printf("| %-16s | %8s | %8s | %8s | %8s |\n",
                "Section", "avg ms", "min ms", "max ms", "p95 ms");
    std::printf("+------------------+----------+----------+----------+----------+\n");

    auto row = [](const SectionTimer& t) {
        std::printf("| %-16s | %8.3f | %8.3f | %8.3f | %8.3f |\n",
                    t.name.c_str(), t.avg(), t.min(), t.max(), t.p95());
    };
    row(t_pre);
    row(t_yolo);
    row(t_dec);
    row(t_resnet);
    row(t_wall);

    std::printf("+------------------+----------+----------+----------+----------+\n");
    std::printf("| Throughput:  %.1f FPS   (1/wall)  YOLO alone: %.1f FPS\n",
                1000.0 / t_wall.avg(), 1000.0 / (t_yolo.avg() + t_dec.avg()));
    std::printf("| YOLO sequential: %.3f ms   YOLO+ResNet sequential: %.3f ms\n",
                t_pre.avg() + t_yolo.avg() + t_dec.avg(),
                t_pre.avg() + t_yolo.avg() + t_dec.avg() + t_resnet.avg());
    std::printf("+----------------------------------------------------------------------+\n\n");
}

// ── Main ──────────────────────────────────────────────────────────────────────
// Extract AIPU core count from model path (.../N/model.json → N, clamped 1–4)
static int cores_from_path(const std::string& p)
{
    auto slash = p.rfind('/');
    if (slash == std::string::npos || slash == 0) return 1;
    auto prev = p.rfind('/', slash - 1);
    const std::string ver = p.substr(prev + 1, slash - prev - 1);
    int n = 1;
    try { n = std::stoi(ver); } catch (...) {}
    return std::clamp(n, 1, 4);
}

int main(int argc, char** argv)
{
    std::string yolo_path, resnet_path, image_path, labels_path, out_path;
    int warmup = 5, bench = 20, rgba_w = 0, rgba_h = 0;

    for (int i = 1; i < argc; ++i) {
        std::string s(argv[i]);
        if      (s.starts_with("--yolo="))    yolo_path   = s.substr(7);
        else if (s.starts_with("--resnet="))  resnet_path = s.substr(9);
        else if (s.starts_with("--warmup="))  warmup      = std::stoi(s.substr(9));
        else if (s.starts_with("--runs="))    bench       = std::stoi(s.substr(7));
        else if (s.starts_with("--size="))
            std::sscanf(s.c_str() + 7, "%dx%d", &rgba_w, &rgba_h);
        else if (s.starts_with("--output="))  out_path    = s.substr(9);
        else if (s.ends_with(".names") ||
                 s.ends_with(".txt"))          labels_path = s;
        else if (!s.starts_with("--"))         image_path  = s;
    }

    if (yolo_path.empty() || resnet_path.empty()) {
        std::fprintf(stderr,
            "Usage: %s --yolo=model.json --resnet=model.json [image] [labels]\n"
            "          [--size=WxH] [--output=path.jpg] [--warmup=N] [--runs=N]\n"
            "\n"
            "  --yolo    : YOLOv5s model.json (version dir = number of AIPU cores)\n"
            "  --resnet  : ResNet50 model.json (version dir = number of AIPU cores)\n"
            "  e.g.  .../yolov5s-v7-coco/1/model.json  → 1 core\n"
            "        .../yolov5s-v7-coco/2/model.json  → 2 cores\n"
            "  image     : .rgba file or JPEG/PNG\n",
            argv[0]);
        return 1;
    }

    const int yolo_cores   = cores_from_path(yolo_path);
    const int resnet_cores = cores_from_path(resnet_path);

    std::vector<std::string> labels;
    if (!labels_path.empty()) {
        std::ifstream f(labels_path);
        for (std::string l; std::getline(f, l);) labels.push_back(l);
    }

    // ── Runtime ───────────────────────────────────────────────────────────────
    auto ctx = axr::to_ptr(axr_create_context());
    axr_set_logger(ctx.get(), AXR_LOG_ERROR,
        [](void*, axrLogLevel, const char* m){ fputs(m, stderr); fputc('\n', stderr); },
        nullptr);

    // Connect to enough sub-devices (cores) to fit both models simultaneously.
    // Each model occupies its own core's L2 memory; sharing a single sub-device
    // causes both models to compete for one core's 8 MB budget.
    const int total_cores = yolo_cores + resnet_cores;
    auto* conn = axr_device_connect(ctx.get(), nullptr,
                                    static_cast<size_t>(total_cores), nullptr);
    if (!conn) {
        std::fprintf(stderr, "[ERROR] %s\n", axr_last_error_string(AXR_OBJECT(ctx.get())));
        return 1;
    }
    std::printf("[INFO] Connected to %d AIPU sub-device(s) (YOLO:%d + ResNet50:%d)\n\n",
                total_cores, yolo_cores, resnet_cores);

    // ── Load models ───────────────────────────────────────────────────────────
    // Open DMA heap first so we know whether to use DMA-BUF for YOLO
    int heap_fd = ::open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
    bool use_dmabuf = (heap_fd >= 0);

    const std::string yolo_props =
        std::string(use_dmabuf ? "input_dmabuf=1" : "input_dmabuf=0")
        + ";output_dmabuf=0;num_sub_devices=" + std::to_string(yolo_cores)
        + ";aipu_cores=" + std::to_string(yolo_cores)
        + ";double_buffer=0;elf_in_ddr=1";
    const std::string resnet_props =
        "input_dmabuf=0;output_dmabuf=0;num_sub_devices=" + std::to_string(resnet_cores)
        + ";aipu_cores=" + std::to_string(resnet_cores)
        + ";double_buffer=0;elf_in_ddr=1";

    Model yolo, resnet;
    if (!yolo.load(ctx.get(), conn, yolo_path.c_str(), yolo_props)) {
        std::fprintf(stderr, "[ERROR] YOLOv5s: %s\n", axr_last_error_string(AXR_OBJECT(ctx.get())));
        return 1;
    }
    if (!resnet.load(ctx.get(), conn, resnet_path.c_str(), resnet_props)) {
        std::fprintf(stderr, "[ERROR] ResNet50: %s\n", axr_last_error_string(AXR_OBJECT(ctx.get())));
        return 1;
    }

    yolo.print_shapes("YOLOv5s");
    resnet.print_shapes("ResNet50");
    std::printf("[INFO] YOLO props:   %s\n", yolo_props.c_str());
    std::printf("[INFO] ResNet props: %s\n\n", resnet_props.c_str());

    // ── Single-batch info helpers (batch dim = 1 from each v2 model) ──────────
    // YOLOv5s v2: in_info[0].dims = [2, 644, 656, 4]
    // We fill both batch slots (duplicate frame) and decode only slot 0.
    axrTensorInfo yolo_single = yolo.in_info[0];
    yolo_single.dims[0] = 1;
    const size_t yolo_single_sz = axr_tensor_size(&yolo_single);

    // ResNet50 v2: in_info[0].dims = [2, 230, 240, 4]
    axrTensorInfo resnet_single = resnet.in_info[0];
    resnet_single.dims[0] = 1;
    const size_t resnet_single_sz = axr_tensor_size(&resnet_single);

    // ResNet50 output: [2, 1, 1, 1024] → 1024 features per crop
    const size_t embed_dim = resnet.out_info[0].dims[3];  // 1024
    std::printf("[INFO] Embedding dim: %zu\n", embed_dim);

    // ── YOLO input buffers — DMA-BUF, double-buffered ─────────────────────────
    std::array<DmaBuf, 2>                    yolo_dma;
    std::array<std::unique_ptr<int8_t[]>, 2> yolo_heap;
    std::array<int8_t*, 2>                   yolo_ptrs{nullptr, nullptr};
    std::array<std::array<axrArgument,1>, 2> yolo_args;

    const size_t yolo_total_sz = axr_tensor_size(&yolo.in_info[0]);
    for (int b = 0; b < 2; ++b) {
        if (use_dmabuf) {
            yolo_dma[b] = DmaBuf::alloc(heap_fd, yolo_total_sz);
            if (yolo_dma[b].valid()) {
                yolo_ptrs[b]     = static_cast<int8_t*>(yolo_dma[b].ptr);
                yolo_args[b][0]  = {nullptr, yolo_dma[b].fd, 0, 0};
                continue;
            }
            use_dmabuf = false;
        }
        yolo_heap[b]    = std::make_unique<int8_t[]>(yolo_total_sz);
        yolo_ptrs[b]    = yolo_heap[b].get();
        yolo_args[b][0] = {yolo_ptrs[b], 0, 0, 0};
    }
    if (!use_dmabuf) std::fprintf(stderr, "[WARN] DMA-BUF unavailable, using host memory\n");
    else             std::printf("[DMA-BUF] YOLO: 2x %zu KB via /dev/dma_heap/system\n",
                                 yolo_total_sz / 1024);

    // ── ResNet50 input buffer — host memory, batch=2, single (no double-buf) ──
    auto resnet_in = std::make_unique<int8_t[]>(axr_tensor_size(&resnet.in_info[0]));
    axrArgument resnet_arg = {resnet_in.get(), 0, 0, 0};

    // ── Load RGBA source and build benchmark frame ─────────────────────────────
    cv::Mat vis_bgr;
    int src_w = 0, src_h = 0;
    std::string out_jpg;
    std::vector<uint8_t> rgba_bench;

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
            rgba_bench.resize(static_cast<size_t>(src_w) * src_h * 4);
            std::ifstream f(image_path, std::ios::binary);
            if (!f) { std::fprintf(stderr, "[ERROR] Cannot open %s\n", image_path.c_str()); return 1; }
            f.read(reinterpret_cast<char*>(rgba_bench.data()),
                   static_cast<std::streamsize>(rgba_bench.size()));
            cv::Mat rm(src_h, src_w, CV_8UC4, rgba_bench.data());
            cv::cvtColor(rm, vis_bgr, cv::COLOR_RGBA2BGR);
        } else {
            cv::Mat img = cv::imread(image_path);
            if (img.empty()) { std::fprintf(stderr, "[ERROR] Cannot read %s\n", image_path.c_str()); return 1; }
            src_w = img.cols; src_h = img.rows;
            vis_bgr = img.clone();
            cv::Mat rm; cv::cvtColor(img, rm, cv::COLOR_BGR2RGBA);
            rgba_bench.assign(rm.data, rm.data + static_cast<size_t>(src_w) * src_h * 4);
        }
        out_jpg = out_path.empty()
            ? image_path.substr(0, image_path.rfind('.')) + "_cascade.jpg" : out_path;
    } else {
        src_w = 640; src_h = 640;
        rgba_bench.assign(static_cast<size_t>(src_w) * src_h * 4, 114);
        vis_bgr = cv::Mat(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));
        out_jpg = out_path.empty() ? "cascade_detections.jpg" : out_path;
        std::printf("[INFO] No image — using synthetic 640×640 RGBA pattern\n");
    }
    std::printf("[INFO] Source: %dx%d RGBA  Embed dim: %zu\n\n", src_w, src_h, embed_dim);

    const int resnet_batch = static_cast<int>(resnet.in_info[0].dims[0]);

    // Helper: fill YOLO input; duplicate frame across all batch slots
    auto preprocess_yolo = [&](int8_t* ptr) {
        rgba_to_tensor(rgba_bench.data(), src_w, src_h, ptr, yolo_single);
        for (int s = 1; s < yolo_cores; ++s)
            std::memcpy(ptr + s * yolo_single_sz, ptr, yolo_single_sz);
    };

    // Helper: decode detections from batch slot 0 of YOLO output
    auto decode_yolo = [&]() -> std::vector<Det> {
        auto grid_to_sid = [](size_t h) -> int {
            if (h >= 70) return 0;
            if (h >= 35) return 1;
            return 2;
        };
        std::vector<Det> cands;
        for (size_t j = 0; j < yolo.n_out; ++j)
            decode_head(yolo.out_host[j].get(), yolo.out_info[j],
                        grid_to_sid(yolo.out_info[j].dims[1]), cands);
        return nms(std::move(cands), NMS_IOU_THR);
    };

    // Helper: run ResNet50 on a batch of detections starting at dets[start]
    // Fills up to resnet_batch slots; caller must pad dets to a multiple of resnet_batch.
    auto embed_batch = [&](const std::vector<Det>& dets, size_t start) {
        for (int s = 0; s < resnet_batch; ++s) {
            const Det& d = dets[start + s];
            crop_to_tensor(rgba_bench.data(), src_w, src_h,
                           d.x1, d.y1, d.x2, d.y2, YOLO_WH,
                           resnet_in.get() + s * resnet_single_sz, resnet_single);
        }
        axr_run_model_instance(resnet.instance, &resnet_arg, 1,
                               resnet.out_args.data(), resnet.n_out);
    };

    // ── Prime buf[0] before benchmark loop ───────────────────────────────────
    preprocess_yolo(yolo_ptrs[0]);

    // ── Warmup ────────────────────────────────────────────────────────────────
    std::printf("[INFO] Warming up (%d runs)...\n", warmup);
    for (int i = 0; i < warmup; ++i)
        axr_run_model_instance(yolo.instance,
            yolo_args[0].data(), yolo.n_in, yolo.out_args.data(), yolo.n_out);

    // ── Benchmark ─────────────────────────────────────────────────────────────
    //
    //  Async thread:  preprocess frame into yolo_in[nxt]   (~6 ms)
    //  Main thread:   yolo inference on yolo_in[cur]       (~5 ms with 2 cores)
    //  Main thread:   decode + NMS                         (~0.1 ms)
    //  Main thread:   ResNet50 crop pairs                  (~? ms × ceil(N/2) calls)
    //
    std::printf("[INFO] Benchmarking (%d runs)...\n\n", bench);

    SectionTimer t_pre   {"Preprocess RGBA"};
    SectionTimer t_yolo  {"YOLO v" + std::to_string(yolo_cores)};
    SectionTimer t_dec   {"Decode + NMS"};
    SectionTimer t_resnet{"ResNet50/call"};
    SectionTimer t_wall  {"Frame wall time"};

    std::vector<Det> last_dets;

    for (int i = 0; i < bench; ++i) {
        const int cur = i & 1;
        const int nxt = cur ^ 1;
        const auto t0 = Clock::now();

        int8_t* nxt_ptr = yolo_ptrs[nxt];
        auto pre_fut = std::async(std::launch::async, [&, nxt_ptr]() -> double {
            const auto tp = Clock::now();
            preprocess_yolo(nxt_ptr);
            return Ms(Clock::now() - tp).count();
        });

        // YOLO inference on cur (overlaps preprocess above)
        {
            ScopeTimer st(t_yolo);
            axr_run_model_instance(yolo.instance,
                yolo_args[cur].data(), yolo.n_in, yolo.out_args.data(), yolo.n_out);
        }

        t_pre.record(pre_fut.get());

        // Decode
        std::vector<Det> dets;
        { ScopeTimer st(t_dec); dets = decode_yolo(); }
        last_dets = dets;

        // ResNet50 — process detected crops in batches of resnet_batch
        if (!dets.empty()) {
            while (static_cast<int>(dets.size()) % resnet_batch != 0)
                dets.push_back(dets.back());
            for (size_t d = 0; d < dets.size(); d += resnet_batch) {
                const auto tr = Clock::now();
                embed_batch(dets, d);
                t_resnet.record(Ms(Clock::now() - tr).count());
            }
        }

        t_wall.record(Ms(Clock::now() - t0).count());
    }

    print_latency_table(t_pre, t_yolo, t_dec, t_resnet, t_wall,
                        bench, use_dmabuf, yolo_cores, resnet_cores);

    // ── Final pass: annotate image and print embeddings ───────────────────────
    preprocess_yolo(yolo_ptrs[0]);
    axr_run_model_instance(yolo.instance,
        yolo_args[0].data(), yolo.n_in, yolo.out_args.data(), yolo.n_out);
    last_dets = decode_yolo();

    const size_t real_dets = last_dets.size();
    while (static_cast<int>(last_dets.size()) % resnet_batch != 0)
        last_dets.push_back(last_dets.back());

    std::printf("[DETECTIONS + EMBEDDINGS]  %zu object(s)\n", real_dets);

    const auto& ro  = resnet.out_info[0];
    const float rsc = static_cast<float>(ro.scale);
    const int   rzp = ro.zero_point;

    for (size_t d = 0; d < last_dets.size(); d += resnet_batch) {
        embed_batch(last_dets, d);

        for (int slot = 0; slot < resnet_batch; ++slot) {
            if (d + slot >= real_dets) break;  // skip padding slots

            const Det& det = last_dets[d + slot];
            const std::string cls = det.cls < (int)labels.size()
                ? labels[det.cls] : "cls" + std::to_string(det.cls);

            const int8_t* emb = resnet.out_host[0].get() + slot * embed_dim;
            std::printf("  %-20s  conf=%.3f  embed[0..3]=[%.4f, %.4f, %.4f, %.4f]\n",
                cls.c_str(), det.conf,
                (emb[0] - rzp) * rsc,
                (emb[1] - rzp) * rsc,
                (emb[2] - rzp) * rsc,
                (emb[3] - rzp) * rsc);
        }
    }

    if (!vis_bgr.empty() && !out_jpg.empty())
        save_annotated(vis_bgr, last_dets, labels, out_jpg, YOLO_WH);

    // ── Cleanup ───────────────────────────────────────────────────────────────
    for (int b = 0; b < 2; ++b) yolo_dma[b].release();
    if (heap_fd >= 0) ::close(heap_fd);
    return 0;
}
