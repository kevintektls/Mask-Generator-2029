#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace autopilot {

cv::Mat detect_lines(const cv::Mat& frame_gray);
std::vector<float> mask_to_input(const cv::Mat& mask);
cv::Mat build_display_frame(const cv::Mat& mask, float servo, float duty);
std::vector<uint8_t> encode_jpeg(const cv::Mat& bgr, int quality = 85);

}  // namespace autopilot
