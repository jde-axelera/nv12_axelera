#pragma once
#include <cstdint>
#include "axruntime/axruntime.h"

// Convert first frame of an NV12 buffer (src_w x src_h) into the model's
// quantised int8 NHWC input tensor. Handles padding and BGR->RGB reorder.
// YOLOv5 normalisation: pixel / 255  (no ImageNet mean/std).
void nv12_to_tensor(const uint8_t* nv12, int src_w, int src_h,
                    int8_t* out, const axrTensorInfo& info);
