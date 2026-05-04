#include "preprocess.hpp"
#include <algorithm>
#include "opencv2/opencv.hpp"

void nv12_to_tensor(const uint8_t* nv12, int src_w, int src_h,
                    int8_t* out, const axrTensorInfo& info)
{
    cv::Mat yuv(src_h + src_h / 2, src_w, CV_8UC1, const_cast<uint8_t*>(nv12));
    cv::Mat bgr;
    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);

    const int H   = static_cast<int>(info.dims[1]);
    const int W   = static_cast<int>(info.dims[2]);
    const int C   = static_cast<int>(info.dims[3]);
    const int ypl = static_cast<int>(info.padding[1][0]);
    const int ypr = static_cast<int>(info.padding[1][1]);
    const int xpl = static_cast<int>(info.padding[2][0]);
    const int xpr = static_cast<int>(info.padding[2][1]);
    const int cpl = static_cast<int>(info.padding[3][0]);
    const int cpr = static_cast<int>(info.padding[3][1]);
    const int uH  = H - ypl - ypr;
    const int uW  = W - xpl - xpr;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(uW, uH), 0, 0, cv::INTER_LINEAR);

    const auto*  in_bgr = resized.data;
    const float  scale  = static_cast<float>(info.scale);
    const int    zp     = info.zero_point;
    const float  mul    = 1.0f / (scale * 255.0f);
    const float  add    = static_cast<float>(zp);
    const int8_t pval   = static_cast<int8_t>(std::clamp(zp, -128, 127));

    out = std::fill_n(out, ypl * W * C, pval);
    for (int y = 0; y < uH; ++y) {
        out = std::fill_n(out, xpl * C, pval);
        for (int x = 0; x < uW; ++x) {
            out = std::fill_n(out, cpl, pval);
            for (int c = 0; c < 3; ++c) {
                float v = static_cast<float>(in_bgr[2 - c]) * mul + add;
                *out++  = static_cast<int8_t>(std::clamp(v, -128.0f, 127.0f));
            }
            in_bgr += 3;
            out = std::fill_n(out, cpr, pval);
        }
        out = std::fill_n(out, xpr * C, pval);
    }
    std::fill_n(out, ypr * W * C, pval);
}

void rgba_to_tensor(const uint8_t* rgba, int src_w, int src_h,
                    int8_t* out, const axrTensorInfo& info)
{
    cv::Mat rgba_mat(src_h, src_w, CV_8UC4, const_cast<uint8_t*>(rgba));

    const int H   = static_cast<int>(info.dims[1]);
    const int W   = static_cast<int>(info.dims[2]);
    const int C   = static_cast<int>(info.dims[3]);
    const int ypl = static_cast<int>(info.padding[1][0]);
    const int ypr = static_cast<int>(info.padding[1][1]);
    const int xpl = static_cast<int>(info.padding[2][0]);
    const int xpr = static_cast<int>(info.padding[2][1]);
    const int cpl = static_cast<int>(info.padding[3][0]);
    const int cpr = static_cast<int>(info.padding[3][1]);
    const int uH  = H - ypl - ypr;
    const int uW  = W - xpl - xpr;

    cv::Mat resized;
    cv::resize(rgba_mat, resized, cv::Size(uW, uH), 0, 0, cv::INTER_LINEAR);

    const auto*  in_rgba = resized.data;
    const float  scale   = static_cast<float>(info.scale);
    const int    zp      = info.zero_point;
    const float  mul     = 1.0f / (scale * 255.0f);
    const float  add     = static_cast<float>(zp);
    const int8_t pval    = static_cast<int8_t>(std::clamp(zp, -128, 127));

    out = std::fill_n(out, ypl * W * C, pval);
    for (int y = 0; y < uH; ++y) {
        out = std::fill_n(out, xpl * C, pval);
        for (int x = 0; x < uW; ++x) {
            out = std::fill_n(out, cpl, pval);
            for (int c = 0; c < 3; ++c) {
                // RGBA layout: [R,G,B,A]; model expects R,G,B order
                float v = static_cast<float>(in_rgba[c]) * mul + add;
                *out++  = static_cast<int8_t>(std::clamp(v, -128.0f, 127.0f));
            }
            in_rgba += 4;  // stride over A channel
            out = std::fill_n(out, cpr, pval);
        }
        out = std::fill_n(out, xpr * C, pval);
    }
    std::fill_n(out, ypr * W * C, pval);
}
