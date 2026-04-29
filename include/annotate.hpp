#pragma once
#include <string>
#include <vector>
#include "yolo_decode.hpp"
#include "opencv2/opencv.hpp"

// Draw bounding boxes on img (original resolution) and save as JPEG.
// Detections are in MODEL_WH x MODEL_WH coordinate space and are scaled
// back to the image resolution before drawing.
void save_annotated(cv::Mat img,
                    const std::vector<Det>& dets,
                    const std::vector<std::string>& labels,
                    const std::string& out_path,
                    int model_wh = 640);
