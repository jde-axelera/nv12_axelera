// Generic embedding feature extraction on Axelera Metis AIPU.
// Accepts any compiled embedding model (ResNet18, ResNet50 backbone, etc.).
// Copyright Axelera AI, 2026

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "opencv2/opencv.hpp"
#include "axruntime/axruntime.hpp"
#include "dmabuf.hpp"
#include "preprocess.hpp"
#include "timer.hpp"

// ── Latency table ─────────────────────────────────────────────────────────────
static void print_latency_table(
    const SectionTimer& t_pre,
    const SectionTimer& t_inf,
    const SectionTimer& t_wall,
    int runs, bool use_dmabuf, size_t embed_dim, int n_cores)
{
    std::printf("\n+--------------------------------------------------------------------+\n");
    std::printf("| LATENCY BREAKDOWN  (%d runs, %s, double-buffered pipeline)\n",
                runs, use_dmabuf ? "DMA-BUF input" : "host-mem input");
    std::printf("| Embedding dim: %zu    AIPU: %d core(s)\n", embed_dim, n_cores);
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
    row(t_wall);
    std::printf("+------------------+----------+----------+----------+----------+\n");
    std::printf("| Throughput (pipelined):  %.1f FPS\n", 1000.0 / t_wall.avg());
    std::printf("| Sequential latency:      %.3f ms  (pre+inf, non-overlapped)\n",
                t_pre.avg() + t_inf.avg());
    std::printf("+--------------------------------------------------------------------+\n\n");
}

// ── Core count from model path (.../N/model.json → N) ─────────────────────────
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
    std::string model_path, image_path, out_emb_path;
    int warmup = 5, bench = 30, rgba_w = 0, rgba_h = 0;

    for (int i = 1; i < argc; ++i) {
        std::string s(argv[i]);
        if      (s.starts_with("--model="))      model_path   = s.substr(8);
        else if (s.starts_with("--warmup="))     warmup       = std::stoi(s.substr(9));
        else if (s.starts_with("--runs="))       bench        = std::stoi(s.substr(7));
        else if (s.starts_with("--size="))
            std::sscanf(s.c_str() + 7, "%dx%d", &rgba_w, &rgba_h);
        else if (s.starts_with("--output-emb=")) out_emb_path = s.substr(13);
        else if (!s.starts_with("--"))           image_path   = s;
    }

    if (model_path.empty()) {
        std::fprintf(stderr,
            "Usage: %s --model=model.json [image] [--size=WxH]\n"
            "          [--output-emb=emb.txt] [--warmup=N] [--runs=N]\n"
            "  image          : JPEG/PNG or raw .rgba file\n"
            "  --size=WxH     : required for .rgba files (e.g. --size=768x576)\n"
            "  --output-emb   : save float embeddings for ONNX comparison\n",
            argv[0]);
        return 1;
    }

    const int n_cores = cores_from_path(model_path);

    // ── Runtime ───────────────────────────────────────────────────────────────
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

    std::printf("[INFO] Model: %s  (%d core(s))\n", model_path.c_str(), n_cores);
    for (size_t i = 0; i < n_in; ++i) {
        auto& t = in_info[i];
        std::printf("[INFO] Input[%zu]  shape=[", i);
        for (size_t d = 0; d < t.ndims; ++d)
            std::printf("%lu%s", (unsigned long)t.dims[d], d+1<t.ndims?",":"");
        std::printf("] scale=%g zp=%d\n", t.scale, t.zero_point);
    }
    for (size_t i = 0; i < n_out; ++i) {
        auto& t = out_info[i];
        std::printf("[INFO] Output[%zu] shape=[", i);
        for (size_t d = 0; d < t.ndims; ++d)
            std::printf("%lu%s", (unsigned long)t.dims[d], d+1<t.ndims?",":"");
        std::printf("] scale=%g zp=%d\n", t.scale, t.zero_point);
    }

    // AIPU output is NHWC: [N, H, W, C].
    // axcompile may split avgpool+flatten to CPU postprocess, so the AIPU
    // outputs the spatial feature map [N, H, W, C] rather than [N, C].
    // We apply global average pooling over H×W here to get the C-dim embedding.
    // If H=W=1 (e.g. ResNet50 compiled with avgpool on AIPU), this is a no-op.
    const size_t out_H     = out_info[0].ndims >= 2 ? out_info[0].dims[1] : 1;
    const size_t out_W     = out_info[0].ndims >= 3 ? out_info[0].dims[2] : 1;
    const size_t embed_dim = out_info[0].ndims >= 4 ? out_info[0].dims[3]
                           : out_info[0].dims[out_info[0].ndims - 1];
    std::printf("[INFO] Embedding dim: %zu  (AIPU output: %zux%zu spatial → global avg pool)\n\n",
                embed_dim, out_H, out_W);

    // ── DMA-BUF double-buffer input ───────────────────────────────────────────
    int heap_fd    = ::open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
    bool use_dmabuf = (heap_fd >= 0);

    const size_t in_sz = axr_tensor_size(&in_info[0]);
    std::array<DmaBuf, 2>                    in_dma;
    std::array<std::unique_ptr<int8_t[]>, 2> in_heap;
    std::array<int8_t*, 2>                   in_ptrs{nullptr, nullptr};
    std::array<std::array<axrArgument, 1>, 2> in_args;

    for (int b = 0; b < 2; ++b) {
        if (use_dmabuf) {
            in_dma[b] = DmaBuf::alloc(heap_fd, in_sz);
            if (in_dma[b].valid()) {
                in_ptrs[b]    = static_cast<int8_t*>(in_dma[b].ptr);
                in_args[b][0] = {nullptr, in_dma[b].fd, 0, 0};
                continue;
            }
            use_dmabuf = false;
        }
        in_heap[b]    = std::make_unique<int8_t[]>(in_sz);
        in_ptrs[b]    = in_heap[b].get();
        in_args[b][0] = {in_ptrs[b], 0, 0, 0};
    }

    if (!use_dmabuf)
        std::fprintf(stderr, "[WARN] DMA-BUF unavailable, using host memory\n");
    else
        std::printf("[DMA-BUF] 2x %zu KB double-buffered input\n", in_sz / 1024);

    // Output buffers — MMIO outputs, DMA-BUF not supported
    std::vector<std::unique_ptr<int8_t[]>> out_host(n_out);
    std::vector<axrArgument>               out_args(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        out_host[i] = std::make_unique<int8_t[]>(axr_tensor_size(&out_info[i]));
        out_args[i] = {out_host[i].get(), 0, 0, 0};
    }

    // ── Device + instance ─────────────────────────────────────────────────────
    auto* conn = axr_device_connect(ctx.get(), nullptr,
                                    static_cast<size_t>(n_cores), nullptr);
    if (!conn) {
        std::fprintf(stderr, "[ERROR] %s\n", axr_last_error_string(AXR_OBJECT(ctx.get())));
        return 1;
    }
    const std::string props_str =
        std::string(use_dmabuf ? "input_dmabuf=1" : "input_dmabuf=0")
        + ";output_dmabuf=0;num_sub_devices=" + std::to_string(n_cores)
        + ";aipu_cores=" + std::to_string(n_cores)
        + ";double_buffer=0;elf_in_ddr=1";
    auto* props    = axr_create_properties(ctx.get(), props_str.c_str());
    auto* instance = axr_load_model_instance(conn, model, props);
    if (!instance) {
        std::fprintf(stderr, "[ERROR] %s\n", axr_last_error_string(AXR_OBJECT(ctx.get())));
        return 1;
    }
    std::printf("[INFO] Props: %s\n\n", props_str.c_str());

    // ── Load image ────────────────────────────────────────────────────────────
    int src_w = 640, src_h = 640;
    std::vector<uint8_t> rgba_data;

    if (!image_path.empty()) {
        const auto ext = image_path.substr(image_path.rfind('.') + 1);
        if (ext == "rgba") {
            src_w = rgba_w > 0 ? rgba_w : 640;
            src_h = rgba_h > 0 ? rgba_h : 640;
            if (rgba_w == 0) {
                auto us = image_path.rfind('_');
                if (us != std::string::npos)
                    std::sscanf(image_path.c_str() + us + 1, "%dx%d", &src_w, &src_h);
            }
            rgba_data.resize(static_cast<size_t>(src_w) * src_h * 4);
            std::ifstream f(image_path, std::ios::binary);
            if (!f) {
                std::fprintf(stderr, "[ERROR] Cannot open %s\n", image_path.c_str());
                return 1;
            }
            f.read(reinterpret_cast<char*>(rgba_data.data()),
                   static_cast<std::streamsize>(rgba_data.size()));
        } else {
            cv::Mat img = cv::imread(image_path);
            if (img.empty()) {
                std::fprintf(stderr, "[ERROR] Cannot read %s\n", image_path.c_str());
                return 1;
            }
            src_w = img.cols; src_h = img.rows;
            cv::Mat rgba_mat;
            cv::cvtColor(img, rgba_mat, cv::COLOR_BGR2RGBA);
            rgba_data.assign(rgba_mat.data,
                             rgba_mat.data + static_cast<size_t>(src_w) * src_h * 4);
        }
        std::printf("[INFO] Source image: %s  (%dx%d)\n\n",
                    image_path.c_str(), src_w, src_h);
    } else {
        std::printf("[INFO] No image — using synthetic 640x640 grey RGBA\n\n");
        rgba_data.assign(static_cast<size_t>(640) * 640 * 4, 114);
    }

    // ── Preprocess lambda ─────────────────────────────────────────────────────
    auto preprocess = [&](int8_t* ptr) {
        rgba_to_tensor(rgba_data.data(), src_w, src_h, ptr, in_info[0]);
    };

    // ── Prime buf[0] + warmup ─────────────────────────────────────────────────
    preprocess(in_ptrs[0]);

    std::printf("[INFO] Warming up (%d runs)...\n", warmup);
    for (int i = 0; i < warmup; ++i)
        axr_run_model_instance(instance, in_args[0].data(), n_in, out_args.data(), n_out);

    // ── Double-buffered benchmark ──────────────────────────────────────────────
    //
    //  Each iteration:
    //    Async thread: preprocess frame N+1 into slot[nxt]  (~1 ms for 224x224)
    //    Main thread:  axr_run_model_instance slot[cur]      (~3-5 ms)
    //  Preprocess fully hidden behind AIPU → wall ≈ AIPU time.
    //
    std::printf("[INFO] Benchmarking (%d runs, double-buffered)...\n\n", bench);

    SectionTimer t_pre {"Preprocess"};
    SectionTimer t_inf {"Inference (AIPU)"};
    SectionTimer t_wall{"Frame wall time"};

    for (int i = 0; i < bench; ++i) {
        const int cur = i & 1;
        const int nxt = cur ^ 1;
        const auto t0 = Clock::now();

        int8_t* nxt_ptr = in_ptrs[nxt];
        auto pre_fut = std::async(std::launch::async, [&, nxt_ptr]() -> double {
            const auto tp = Clock::now();
            preprocess(nxt_ptr);
            return Ms(Clock::now() - tp).count();
        });

        {
            ScopeTimer st(t_inf);
            axr_run_model_instance(instance,
                in_args[cur].data(), n_in, out_args.data(), n_out);
        }
        t_pre.record(pre_fut.get());
        t_wall.record(Ms(Clock::now() - t0).count());
    }

    print_latency_table(t_pre, t_inf, t_wall, bench, use_dmabuf, embed_dim, n_cores);

    // ── Final inference + dequantise ──────────────────────────────────────────
    preprocess(in_ptrs[0]);
    axr_run_model_instance(instance, in_args[0].data(), n_in, out_args.data(), n_out);

    const float out_scale = static_cast<float>(out_info[0].scale);
    const int   out_zp    = out_info[0].zero_point;

    // Global average pool: dequantise and average over H×W spatial dims.
    // NHWC layout: element [y,x,c] = raw[(y*out_W + x)*embed_dim + c]
    std::vector<float> embedding(embed_dim, 0.0f);
    const int8_t* raw     = out_host[0].get();
    const float   inv_hw  = 1.0f / static_cast<float>(out_H * out_W);
    for (size_t y = 0; y < out_H; ++y)
        for (size_t x = 0; x < out_W; ++x)
            for (size_t c = 0; c < embed_dim; ++c)
                embedding[c] += (static_cast<float>(
                    raw[(y * out_W + x) * embed_dim + c]) - out_zp) * out_scale;
    for (float& v : embedding) v *= inv_hw;

    float norm_sq = 0.0f;
    for (float v : embedding) norm_sq += v * v;

    std::printf("[EMBEDDING]  dim=%zu   norm=%.4f\n", embed_dim, std::sqrt(norm_sq));
    std::printf("  First 6 values: [%.5f, %.5f, %.5f, %.5f, %.5f, %.5f]\n\n",
                embedding[0], embedding[1], embedding[2],
                embedding[3], embedding[4], embedding[5]);

    if (!out_emb_path.empty()) {
        std::ofstream f(out_emb_path);
        for (float v : embedding) f << v << "\n";
        std::printf("[SAVED]  embedding → %s  (use verify_onnx_embedding.py --aipu)\n",
                    out_emb_path.c_str());
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    for (int b = 0; b < 2; ++b) in_dma[b].release();
    if (heap_fd >= 0) ::close(heap_fd);
    return 0;
}
