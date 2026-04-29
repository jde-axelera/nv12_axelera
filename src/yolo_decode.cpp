#include "yolo_decode.hpp"
#include <algorithm>
#include <cmath>

// YOLOv5s COCO anchors [stride_idx][anchor_idx][w,h]
static constexpr float ANCHORS[3][3][2] = {
    {{10, 13}, {16, 30},  {33,  23}},   // stride  8
    {{30, 61}, {62, 45},  {59, 119}},   // stride 16
    {{116, 90},{156,198}, {373, 326}}    // stride 32
};
static constexpr int STRIDES[3]   = {8, 16, 32};
static constexpr int NUM_CLASSES  = 80;
static constexpr int NUM_ANCHORS  = 3;
static constexpr float CONF_THR   = 0.25f;

void decode_head(const int8_t* raw, const axrTensorInfo& info,
                 int sid, std::vector<Det>& out)
{
    const int Hg    = static_cast<int>(info.dims[1]);
    const int Wg    = static_cast<int>(info.dims[2]);
    const int C     = static_cast<int>(info.dims[3]);
    const int hpl   = static_cast<int>(info.padding[1][0]);
    const int hpr   = static_cast<int>(info.padding[1][1]);
    const int wpl   = static_cast<int>(info.padding[2][0]);
    const int wpr   = static_cast<int>(info.padding[2][1]);
    const int cpl   = static_cast<int>(info.padding[3][0]);
    const int realH = Hg - hpl - hpr;
    const int realW = Wg - wpl - wpr;

    const float sc     = static_cast<float>(info.scale);
    const int   zp     = info.zero_point;
    const int   stride = STRIDES[sid];
    const float (&anch)[3][2] = ANCHORS[sid];

    auto dq = [sc, zp](int8_t v) -> float {
        return (static_cast<float>(v) - zp) * sc;
    };

    const int per_anch = 5 + NUM_CLASSES;   // 85

    for (int gy = 0; gy < realH; ++gy) {
        for (int gx = 0; gx < realW; ++gx) {
            const int base = ((gy + hpl) * Wg + (gx + wpl)) * C + cpl;
            for (int a = 0; a < NUM_ANCHORS; ++a) {
                const int off = base + a * per_anch;
                if (off + per_anch > Hg * Wg * C) continue;

                const float obj = dq(raw[off + 4]);
                if (obj < CONF_THR) continue;

                int   best_c = 0;
                float best_s = dq(raw[off + 5]);
                for (int c = 1; c < NUM_CLASSES; ++c) {
                    float cs = dq(raw[off + 5 + c]);
                    if (cs > best_s) { best_s = cs; best_c = c; }
                }
                const float conf = obj * best_s;
                if (conf < CONF_THR) continue;

                const float tx = dq(raw[off + 0]);
                const float ty = dq(raw[off + 1]);
                const float tw = dq(raw[off + 2]);
                const float th = dq(raw[off + 3]);

                const float cx = (tx * 2.0f - 0.5f + static_cast<float>(gx)) * stride;
                const float cy = (ty * 2.0f - 0.5f + static_cast<float>(gy)) * stride;
                float bw = tw * 2.0f; bw = bw * bw * anch[a][0];
                float bh = th * 2.0f; bh = bh * bh * anch[a][1];

                out.push_back({cx - bw * 0.5f, cy - bh * 0.5f,
                               cx + bw * 0.5f, cy + bh * 0.5f,
                               conf, best_c});
            }
        }
    }
}

static float iou(const Det& a, const Det& b) {
    float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    float ua = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter;
    return inter / (ua + 1e-6f);
}

std::vector<Det> nms(std::vector<Det> dets, float iou_thr) {
    std::sort(dets.begin(), dets.end(), [](auto& a, auto& b){ return a.conf > b.conf; });
    std::vector<bool> dead(dets.size(), false);
    std::vector<Det>  out;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (dead[i]) continue;
        out.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j)
            if (!dead[j] && dets[i].cls == dets[j].cls && iou(dets[i], dets[j]) > iou_thr)
                dead[j] = true;
    }
    return out;
}
