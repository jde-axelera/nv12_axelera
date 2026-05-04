// YOLOv5s → ResNet50 cascaded detection + embedding pipeline.
//
// Stage 1 — Detection (YOLOv5s):
//   RGBA frame → preprocess → AIPU inference → bounding boxes
//   Double-buffered: frame N+1 preprocessing overlaps frame N inference.
//
// Stage 2 — Embedding (ResNet50):
//   For each detected crop: resize → ImageNet normalise → AIPU inference
//   → 1024-dimensional embedding vector
//   Double-buffered: crop batch N+1 preprocessing overlaps batch N AIPU run.
//   DMA-BUF input: crops written directly to pinned memory; no extra copy.
//
// Core assignment: yolo_cores + resnet_cores sub-devices are connected so each
// model gets its own core's L2 memory (~8 MB per core on Metis).
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
    const SectionTimer& t_rprep,
    const SectionTimer& t_raipu,
    const SectionTimer& t_wall,
    int runs, bool use_dmabuf, int yolo_cores, int resnet_cores)
{
    std::printf("\n+----------------------------------------------------------------------+\n");
    std::printf("| LATENCY BREAKDOWN  (%d runs, %s, YOLOv5s+ResNet50 cascade)\n",
                runs, use_dmabuf ? "DMA-BUF" : "host-mem");
    std::printf("| YOLOv5s: %d AIPU core(s) (v%d)    ResNet50: %d AIPU core(s) (v%d)\n",
                yolo_cores, yolo_cores, resnet_cores, resnet_cores);
    std::printf("+--------------------+----------+----------+----------+----------+\n");
    std::printf("| %-18s | %8s | %8s | %8s | %8s |\n",
                "Section", "avg ms", "min ms", "max ms", "p95 ms");
    std::printf("+--------------------+----------+----------+----------+----------+\n");

    auto row = [](const SectionTimer& t) {
        std::printf("| %-18s | %8.3f | %8.3f | %8.3f | %8.3f |\n",
                    t.name.c_str(), t.avg(), t.min(), t.max(), t.p95());
    };
    row(t_pre);
    row(t_yolo);
    row(t_dec);
    row(t_rprep);
    row(t_raipu);
    row(t_wall);

    std::printf("+--------------------+----------+----------+----------+----------+\n");
    const double rprep = t_rprep.avg();
    const double raipu = t_raipu.avg();
    std::printf("| Throughput: %.1f FPS (1/wall)   YOLO alone: %.1f FPS\n",
                1000.0 / t_wall.avg(),
                1000.0 / (t_yolo.avg() + t_dec.avg()));
    std::printf("| ResNet50 prep hidden behind AIPU: %.3f ms overlap per call\n",
                std::max(0.0, rprep - raipu));   // positive → prep is bottleneck
    std::printf("+----------------------------------------------------------------------+\n\n");
}

// ── Core count from model path (.../N/model.json → N) ────────────────────────
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

// ── Main ──────────────────────────────────────────────────────────────────────
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
            "  Version dir in path = AIPU core count = batch size:\n"
            "    .../yolov5s-v7-coco/1/model.json   → 1 core, batch 1\n"
            "    .../resnet50-imagenet/2/model.json  → 2 cores, batch 2\n"
            "  image : .rgba file or JPEG/PNG\n",
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

    // Connect to yolo_cores + resnet_cores sub-devices so each model gets its
    // own core's L2 memory.
    const int total_cores = yolo_cores + resnet_cores;
    auto* conn = axr_device_connect(ctx.get(), nullptr,
                                    static_cast<size_t>(total_cores), nullptr);
    if (!conn) {
        std::fprintf(stderr, "[ERROR] %s\n", axr_last_error_string(AXR_OBJECT(ctx.get())));
        return 1;
    }
    std::printf("[INFO] Connected to %d AIPU sub-device(s) "
                "(YOLO:%d + ResNet50:%d)\n\n", total_cores, yolo_cores, resnet_cores);

    // ── DMA heap ─────────────────────────────────────────────────────────────
    int heap_fd = ::open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
    bool use_dmabuf = (heap_fd >= 0);

    // ── Load models ───────────────────────────────────────────────────────────
    const std::string yolo_props =
        std::string(use_dmabuf ? "input_dmabuf=1" : "input_dmabuf=0")
        + ";output_dmabuf=0;num_sub_devices=" + std::to_string(yolo_cores)
        + ";aipu_cores=" + std::to_string(yolo_cores)
        + ";double_buffer=0;elf_in_ddr=1";
    const std::string resnet_props =
        std::string(use_dmabuf ? "input_dmabuf=1" : "input_dmabuf=0")
        + ";output_dmabuf=0;num_sub_devices=" + std::to_string(resnet_cores)
        + ";aipu_cores=" + std::to_string(resnet_cores)
        + ";double_buffer=0;elf_in_ddr=1";

    Model yolo, resnet;
    if (!yolo.load(ctx.get(), conn, yolo_path.c_str(), yolo_props)) {
        std::fprintf(stderr, "[ERROR] YOLOv5s: %s\n",
                     axr_last_error_string(AXR_OBJECT(ctx.get())));
        return 1;
    }
    if (!resnet.load(ctx.get(), conn, resnet_path.c_str(), resnet_props)) {
        std::fprintf(stderr, "[ERROR] ResNet50: %s\n",
                     axr_last_error_string(AXR_OBJECT(ctx.get())));
        return 1;
    }

    yolo.print_shapes("YOLOv5s");
    resnet.print_shapes("ResNet50");
    std::printf("[INFO] YOLO   props: %s\n", yolo_props.c_str());
    std::printf("[INFO] ResNet props: %s\n\n", resnet_props.c_str());

    // ── Tensor info helpers ───────────────────────────────────────────────────
    axrTensorInfo yolo_single = yolo.in_info[0];
    yolo_single.dims[0] = 1;
    const size_t yolo_single_sz = axr_tensor_size(&yolo_single);

    axrTensorInfo resnet_single = resnet.in_info[0];
    resnet_single.dims[0] = 1;
    const size_t resnet_single_sz = axr_tensor_size(&resnet_single);

    const int    resnet_batch = static_cast<int>(resnet.in_info[0].dims[0]);
    const size_t embed_dim    = resnet.out_info[0].dims[3];  // 1024
    std::printf("[INFO] Embedding dim: %zu   ResNet50 batch: %d\n",
                embed_dim, resnet_batch);

    // ResNet50 dequantisation constants
    const float rsc = static_cast<float>(resnet.out_info[0].scale);
    const int   rzp = resnet.out_info[0].zero_point;

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
                yolo_ptrs[b]    = static_cast<int8_t*>(yolo_dma[b].ptr);
                yolo_args[b][0] = {nullptr, yolo_dma[b].fd, 0, 0};
                continue;
            }
            use_dmabuf = false;
        }
        yolo_heap[b]    = std::make_unique<int8_t[]>(yolo_total_sz);
        yolo_ptrs[b]    = yolo_heap[b].get();
        yolo_args[b][0] = {yolo_ptrs[b], 0, 0, 0};
    }

    // ── ResNet50 input buffers — DMA-BUF, double-buffered ─────────────────────
    // Two slots: while AIPU runs slot[cur], CPU fills slot[nxt] with the next
    // batch of crops. This hides crop preprocessing behind AIPU inference.
    std::array<DmaBuf, 2>                    resnet_dma;
    std::array<std::unique_ptr<int8_t[]>, 2> resnet_heap;
    std::array<int8_t*, 2>                   resnet_ptrs{nullptr, nullptr};
    std::array<axrArgument, 2>               resnet_in_args;
    bool use_resnet_dmabuf = use_dmabuf;

    const size_t resnet_total_sz = axr_tensor_size(&resnet.in_info[0]);
    for (int b = 0; b < 2; ++b) {
        if (use_resnet_dmabuf) {
            resnet_dma[b] = DmaBuf::alloc(heap_fd, resnet_total_sz);
            if (resnet_dma[b].valid()) {
                resnet_ptrs[b]    = static_cast<int8_t*>(resnet_dma[b].ptr);
                resnet_in_args[b] = {nullptr, resnet_dma[b].fd, 0, 0};
                continue;
            }
            use_resnet_dmabuf = false;
        }
        resnet_heap[b]    = std::make_unique<int8_t[]>(resnet_total_sz);
        resnet_ptrs[b]    = resnet_heap[b].get();
        resnet_in_args[b] = {resnet_ptrs[b], 0, 0, 0};
    }

    if (!use_dmabuf)
        std::fprintf(stderr, "[WARN] DMA-BUF unavailable, using host memory\n");
    else {
        std::printf("[DMA-BUF] YOLO:    2x %4zu KB via /dev/dma_heap/system\n",
                    yolo_total_sz / 1024);
        std::printf("[DMA-BUF] ResNet50:2x %4zu KB via /dev/dma_heap/system\n",
                    resnet_total_sz / 1024);
    }

    // ── Load RGBA source ──────────────────────────────────────────────────────
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
            if (!f) {
                std::fprintf(stderr, "[ERROR] Cannot open %s\n", image_path.c_str());
                return 1;
            }
            f.read(reinterpret_cast<char*>(rgba_bench.data()),
                   static_cast<std::streamsize>(rgba_bench.size()));
            cv::Mat rm(src_h, src_w, CV_8UC4, rgba_bench.data());
            cv::cvtColor(rm, vis_bgr, cv::COLOR_RGBA2BGR);
        } else {
            cv::Mat img = cv::imread(image_path);
            if (img.empty()) {
                std::fprintf(stderr, "[ERROR] Cannot read %s\n", image_path.c_str());
                return 1;
            }
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
    std::printf("[INFO] Source: %dx%d RGBA\n\n", src_w, src_h);

    // ── Helpers ───────────────────────────────────────────────────────────────

    // Fill YOLO input slot (all batch elements = same frame)
    auto preprocess_yolo = [&](int8_t* ptr) {
        rgba_to_tensor(rgba_bench.data(), src_w, src_h, ptr, yolo_single);
        for (int s = 1; s < yolo_cores; ++s)
            std::memcpy(ptr + s * yolo_single_sz, ptr, yolo_single_sz);
    };

    // Decode YOLOv5s output (batch slot 0 only)
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

    // Fill one ResNet50 input slot with resnet_batch crops starting at dets[start].
    // dets must already be padded to a multiple of resnet_batch.
    auto fill_resnet_slot = [&](int8_t* ptr, const std::vector<Det>& dets, size_t start) {
        for (int k = 0; k < resnet_batch; ++k) {
            const Det& d = dets[start + k];
            crop_to_tensor(rgba_bench.data(), src_w, src_h,
                           d.x1, d.y1, d.x2, d.y2, YOLO_WH,
                           ptr + k * resnet_single_sz, resnet_single);
        }
    };

    // Read AIPU output for one batch into the flat embedding store.
    // Only writes entries [det_start .. det_start+count), skipping padding slots.
    auto read_embed_batch = [&](std::vector<float>& store,
                                size_t det_start, int count) {
        const int8_t* out = resnet.out_host[0].get();
        for (int k = 0; k < count; ++k) {
            const int8_t* src = out + k * embed_dim;
            float*        dst = store.data() + (det_start + k) * embed_dim;
            for (size_t j = 0; j < embed_dim; ++j)
                dst[j] = (src[j] - rzp) * rsc;
        }
    };

    // Double-buffered ResNet50 embedding extraction.
    //
    //  Timeline for N batches:
    //    Prime:        [fill slot 0]
    //    b=0:          [AIPU slot 0] | [fill slot 1]  ← overlap
    //    b=1:          [AIPU slot 1] | [fill slot 0]  ← overlap
    //    ...
    //    b=N-1:        [AIPU slot N-1]                ← no fill needed
    //
    //  t_rprep records each fill time (hidden when fill < AIPU).
    //  t_raipu records each pure AIPU time.
    //  store is resized to n_real * embed_dim and filled in-order.
    auto embed_all = [&](const std::vector<Det>& raw_dets,
                         std::vector<float>&     store,
                         SectionTimer&           t_rprep,
                         SectionTimer&           t_raipu)
    {
        const int n_real = static_cast<int>(raw_dets.size());
        if (n_real == 0) { store.clear(); return; }

        store.resize(static_cast<size_t>(n_real) * embed_dim);

        // Pad dets to a multiple of resnet_batch
        std::vector<Det> dets = raw_dets;
        while (static_cast<int>(dets.size()) % resnet_batch != 0)
            dets.push_back(dets.back());
        const int n_batches = static_cast<int>(dets.size()) / resnet_batch;

        // Prime slot 0 (no overlap for first batch)
        {
            const auto tp = Clock::now();
            fill_resnet_slot(resnet_ptrs[0], dets, 0);
            t_rprep.record(Ms(Clock::now() - tp).count());
        }

        for (int b = 0; b < n_batches; ++b) {
            const int  cur      = b & 1;
            const int  nxt      = cur ^ 1;
            const bool has_next = (b + 1 < n_batches);

            // Async: fill next slot while AIPU runs current
            std::future<double> fill_fut;
            if (has_next) {
                int8_t*      nxt_ptr   = resnet_ptrs[nxt];
                const size_t nxt_start = static_cast<size_t>(b + 1) * resnet_batch;
                fill_fut = std::async(std::launch::async,
                    [&, nxt_ptr, nxt_start]() -> double {
                        const auto tp = Clock::now();
                        fill_resnet_slot(nxt_ptr, dets, nxt_start);
                        return Ms(Clock::now() - tp).count();
                    });
            }

            // AIPU inference on current slot
            {
                ScopeTimer st(t_raipu);
                axr_run_model_instance(resnet.instance, &resnet_in_args[cur], 1,
                                       resnet.out_args.data(), resnet.n_out);
            }

            // Collect fill time (should be ~0 if fill was hidden behind AIPU)
            if (has_next)
                t_rprep.record(fill_fut.get());

            // Dequantise output for this batch into the embedding store
            const size_t cur_start = static_cast<size_t>(b) * resnet_batch;
            const int    cur_count = std::min(resnet_batch, n_real - static_cast<int>(cur_start));
            if (cur_count > 0)
                read_embed_batch(store, cur_start, cur_count);
        }
    };

    // ── Prime YOLO buf[0] ─────────────────────────────────────────────────────
    preprocess_yolo(yolo_ptrs[0]);

    // ── Warmup ────────────────────────────────────────────────────────────────
    std::printf("[INFO] Warming up (%d runs)...\n", warmup);
    for (int i = 0; i < warmup; ++i)
        axr_run_model_instance(yolo.instance,
            yolo_args[0].data(), yolo.n_in, yolo.out_args.data(), yolo.n_out);

    // ── Benchmark ─────────────────────────────────────────────────────────────
    //
    //  YOLO stage (double-buffered across frames):
    //    Async thread:  preprocess frame N+1 into yolo_ptrs[nxt]  (~6 ms)
    //    Main thread:   AIPU inference on yolo_ptrs[cur]           (~9 ms)
    //
    //  ResNet50 stage (double-buffered across batches within each frame):
    //    Async thread:  crop + ImageNet-normalise next batch        (~1 ms)
    //    Main thread:   AIPU inference on current batch             (~6 ms)
    //
    std::printf("[INFO] Benchmarking (%d runs)...\n\n", bench);

    SectionTimer t_pre   {"Preprocess RGBA"};
    SectionTimer t_yolo  {"YOLO v" + std::to_string(yolo_cores)};
    SectionTimer t_dec   {"Decode + NMS"};
    SectionTimer t_rprep {"ResNet50 prep"};
    SectionTimer t_raipu {"ResNet50 AIPU"};
    SectionTimer t_wall  {"Frame wall time"};

    std::vector<Det>   last_dets;
    std::vector<float> embeds;

    for (int i = 0; i < bench; ++i) {
        const int cur = i & 1;
        const int nxt = cur ^ 1;
        const auto t0 = Clock::now();

        // Async YOLO preprocess of next frame
        int8_t* nxt_ptr = yolo_ptrs[nxt];
        auto pre_fut = std::async(std::launch::async, [&, nxt_ptr]() -> double {
            const auto tp = Clock::now();
            preprocess_yolo(nxt_ptr);
            return Ms(Clock::now() - tp).count();
        });

        // YOLO inference on current frame
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

        // ResNet50 — double-buffered across all crop batches
        embed_all(dets, embeds, t_rprep, t_raipu);

        t_wall.record(Ms(Clock::now() - t0).count());
    }

    print_latency_table(t_pre, t_yolo, t_dec, t_rprep, t_raipu, t_wall,
                        bench, use_dmabuf, yolo_cores, resnet_cores);

    // ── Final pass: annotate + print embeddings ───────────────────────────────
    preprocess_yolo(yolo_ptrs[0]);
    axr_run_model_instance(yolo.instance,
        yolo_args[0].data(), yolo.n_in, yolo.out_args.data(), yolo.n_out);
    last_dets = decode_yolo();

    SectionTimer dummy_prep{"_"}, dummy_aipu{"_"};
    embed_all(last_dets, embeds, dummy_prep, dummy_aipu);

    std::printf("[DETECTIONS + EMBEDDINGS]  %zu object(s)\n", last_dets.size());
    for (size_t d = 0; d < last_dets.size(); ++d) {
        const Det& det = last_dets[d];
        const std::string cls = det.cls < (int)labels.size()
            ? labels[det.cls] : "cls" + std::to_string(det.cls);
        const float* emb = embeds.data() + d * embed_dim;
        std::printf("  %-20s  conf=%.3f  embed[0..3]=[%.4f, %.4f, %.4f, %.4f]\n",
            cls.c_str(), det.conf, emb[0], emb[1], emb[2], emb[3]);
    }

    if (!vis_bgr.empty() && !out_jpg.empty())
        save_annotated(vis_bgr, last_dets, labels, out_jpg, YOLO_WH);

    // ── Cleanup ───────────────────────────────────────────────────────────────
    for (int b = 0; b < 2; ++b) {
        yolo_dma[b].release();
        resnet_dma[b].release();
    }
    if (heap_fd >= 0) ::close(heap_fd);
    return 0;
}
