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

// Crop a detected region from an RGBA frame, resize it to the ResNet50
// model's input dimensions, and quantise with ImageNet normalisation.
// bbox coords (bx1,by1,bx2,by2) are in the detection model's coordinate
// space (det_model_wh × det_model_wh, e.g. 640 for YOLOv5s). They are
// scaled back to src_w × src_h before cropping.
void crop_to_tensor(const uint8_t* rgba, int src_w, int src_h,
                    float bx1, float by1, float bx2, float by2,
                    int det_model_wh,
                    int8_t* out, const axrTensorInfo& info);
