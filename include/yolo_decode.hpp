#pragma once
#include <cstdint>
#include <vector>
#include "axruntime/axruntime.h"

struct Det {
    float x1, y1, x2, y2, conf;
    int   cls;
};

// Decode one YOLOv5 head output tensor.
// sid: stride index -- 0=stride8(80x80), 1=stride16(40x40), 2=stride32(20x20)
// Outputs are already post-sigmoid (applied on-chip); do NOT sigmoid again.
void decode_head(const int8_t* raw, const axrTensorInfo& info,
                 int sid, std::vector<Det>& out);

// Per-class greedy NMS.
std::vector<Det> nms(std::vector<Det> dets, float iou_thr);
