#pragma once
#include <cstdint>
#include "axruntime/axruntime.h"

void nv12_to_tensor(const uint8_t* nv12, int src_w, int src_h,
                    int8_t* out, const axrTensorInfo& info);

// Convert an RGBA buffer (src_w x src_h, 4 bytes/pixel) into the model's
// quantised int8 NHWC input tensor. Alpha channel is discarded; RGB channels
// map directly to the model's expected R,G,B order (no cvtColor required).
void rgba_to_tensor(const uint8_t* rgba, int src_w, int src_h,
                    int8_t* out, const axrTensorInfo& info);
