#include "annotate.hpp"
#include <algorithm>
#include <filesystem>

void save_annotated(cv::Mat img,
                    const std::vector<Det>& dets,
                    const std::vector<std::string>& labels,
                    const std::string& out_path,
                    int model_wh)
{
    static const cv::Scalar PALETTE[] = {
        {255, 56,  56},  {255,157,151}, {255,112, 31}, {255,178, 29},
        {207,210, 49},   { 72,249, 10}, {146,204, 23}, { 61,219,134},
        { 26,147, 52},   {  0,212,187}, { 44,153,168}, {  0,194,255},
        { 52, 69,147},   {100,115,255}, {  0, 24,236}, {132, 56,255},
        { 82,  0,133},   {203, 56,255}, {255,149,200}, {255, 55,199},
    };
    const int np = static_cast<int>(sizeof(PALETTE) / sizeof(PALETTE[0]));

    const float sx = static_cast<float>(img.cols) / static_cast<float>(model_wh);
    const float sy = static_cast<float>(img.rows) / static_cast<float>(model_wh);
    const int thick = std::max(1, img.cols / 400);
    const double fscale = std::max(0.4, img.cols / 1200.0);

    for (const auto& d : dets) {
        const cv::Scalar col = PALETTE[d.cls % np];

        const int x1 = std::clamp(static_cast<int>(d.x1 * sx), 0, img.cols - 1);
        const int y1 = std::clamp(static_cast<int>(d.y1 * sy), 0, img.rows - 1);
        const int x2 = std::clamp(static_cast<int>(d.x2 * sx), 0, img.cols - 1);
        const int y2 = std::clamp(static_cast<int>(d.y2 * sy), 0, img.rows - 1);

        cv::rectangle(img, {x1, y1}, {x2, y2}, col, thick);

        const std::string name = d.cls < static_cast<int>(labels.size())
            ? labels[d.cls] : "cls" + std::to_string(d.cls);
        const std::string lbl = name + " " +
            std::to_string(static_cast<int>(d.conf * 100)) + "%";

        int baseline = 0;
        const auto ts = cv::getTextSize(lbl, cv::FONT_HERSHEY_SIMPLEX,
                                        fscale, thick, &baseline);
        const int ty = std::max(y1 - 2, ts.height + 2);
        cv::rectangle(img, {x1, ty - ts.height - 2}, {x1 + ts.width + 2, ty + 2},
                      col, cv::FILLED);
        cv::putText(img, lbl, {x1 + 1, ty}, cv::FONT_HERSHEY_SIMPLEX,
                    fscale, {255,255,255}, thick, cv::LINE_AA);
    }

    std::filesystem::create_directories(std::filesystem::path(out_path).parent_path());
    cv::imwrite(out_path, img, {cv::IMWRITE_JPEG_QUALITY, 92});
    std::printf("[SAVED]  %s\n", out_path.c_str());
}
