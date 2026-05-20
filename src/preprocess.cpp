#include "preprocess.hpp"
#include <algorithm>
#include <cmath>
#include "opencv2/opencv.hpp"

namespace {

// Extracts NHWC dims and padding from an axrTensorInfo.
// uH / uW are the active (unpadded) spatial dimensions.
struct TensorLayout {
    int H, W, C, uH, uW;
    int ypl, ypr, xpl, xpr, cpl, cpr;
    int8_t pval;

    explicit TensorLayout(const axrTensorInfo& t) {
        H    = static_cast<int>(t.dims[1]);
        W    = static_cast<int>(t.dims[2]);
        C    = static_cast<int>(t.dims[3]);
        ypl  = static_cast<int>(t.padding[1][0]);
        ypr  = static_cast<int>(t.padding[1][1]);
        xpl  = static_cast<int>(t.padding[2][0]);
        xpr  = static_cast<int>(t.padding[2][1]);
        cpl  = static_cast<int>(t.padding[3][0]);
        cpr  = static_cast<int>(t.padding[3][1]);
        uH   = H - ypl - ypr;
        uW   = W - xpl - xpr;
        pval = static_cast<int8_t>(std::clamp(t.zero_point, -128, 127));
    }
};

// Writes the quantised tensor for a resized BGR image.
// mul = 1 / (scale * per_channel_divisor), add = zero_point.
void write_nhwc(const cv::Mat& resized, const TensorLayout& l,
                float mul, float add, int8_t* out)
{
    out = std::fill_n(out, l.ypl * l.W * l.C, l.pval);
    for (int y = 0; y < l.uH; ++y) {
        out = std::fill_n(out, l.xpl * l.C, l.pval);
        for (int x = 0; x < l.uW; ++x) {
            out = std::fill_n(out, l.cpl, l.pval);
            const uint8_t* px = resized.data + (y * l.uW + x) * 3;
            // BGR pixel: px[0]=B, px[1]=G, px[2]=R → output RGB order
            for (int c = 0; c < 3; ++c) {
                float v = static_cast<float>(px[2 - c]) * mul + add;
                *out++  = static_cast<int8_t>(std::clamp(v, -128.0f, 127.0f));
            }
            out = std::fill_n(out, l.cpr, l.pval);
        }
        out = std::fill_n(out, l.xpr * l.C, l.pval);
    }
    std::fill_n(out, l.ypr * l.W * l.C, l.pval);
}

} // namespace

void nv12_to_tensor(const uint8_t* nv12, int src_w, int src_h,
                    int8_t* out, const axrTensorInfo& info)
{
    TensorLayout l(info);

    cv::Mat yuv(src_h + src_h / 2, src_w, CV_8UC1, const_cast<uint8_t*>(nv12));
    cv::Mat bgr;
    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(l.uW, l.uH), 0, 0, cv::INTER_LINEAR);

    const float mul = 1.0f / (static_cast<float>(info.scale) * 255.0f);
    const float add = static_cast<float>(info.zero_point);
    write_nhwc(resized, l, mul, add, out);
}

void rgba_to_tensor(const uint8_t* rgba, int src_w, int src_h,
                    int8_t* out, const axrTensorInfo& info)
{
    TensorLayout l(info);

    cv::Mat src(src_h, src_w, CV_8UC4, const_cast<uint8_t*>(rgba));
    cv::Mat bgr;
    cv::cvtColor(src, bgr, cv::COLOR_RGBA2BGR);
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(l.uW, l.uH), 0, 0, cv::INTER_LINEAR);

    const float mul = 1.0f / (static_cast<float>(info.scale) * 255.0f);
    const float add = static_cast<float>(info.zero_point);
    write_nhwc(resized, l, mul, add, out);
}

void rgba_to_tensor_imagenet(const uint8_t* rgba, int src_w, int src_h,
                              int8_t* out, const axrTensorInfo& info)
{
    static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};

    TensorLayout l(info);

    cv::Mat src(src_h, src_w, CV_8UC4, const_cast<uint8_t*>(rgba));
    cv::Mat bgr;
    cv::cvtColor(src, bgr, cv::COLOR_RGBA2BGR);
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(l.uW, l.uH), 0, 0, cv::INTER_LINEAR);

    const float  scale = static_cast<float>(info.scale);
    const float  zp    = static_cast<float>(info.zero_point);

    out = std::fill_n(out, l.ypl * l.W * l.C, l.pval);
    for (int y = 0; y < l.uH; ++y) {
        out = std::fill_n(out, l.xpl * l.C, l.pval);
        for (int x = 0; x < l.uW; ++x) {
            out = std::fill_n(out, l.cpl, l.pval);
            const uint8_t* px = resized.data + (y * l.uW + x) * 3;
            for (int c = 0; c < 3; ++c) {
                float nv = (px[2 - c] / 255.0f - MEAN[c]) / STD[c];
                float qf = nv / scale + zp;
                *out++   = static_cast<int8_t>(
                    std::clamp(std::round(qf), -128.0f, 127.0f));
            }
            out = std::fill_n(out, l.cpr, l.pval);
        }
        out = std::fill_n(out, l.xpr * l.C, l.pval);
    }
    std::fill_n(out, l.ypr * l.W * l.C, l.pval);
}
