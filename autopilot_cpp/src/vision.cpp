#include "autopilot/vision.hpp"

#include "autopilot/config.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <sstream>
#include <vector>

namespace autopilot {

cv::Mat detect_lines(const cv::Mat& frame_gray) {
    CV_Assert(frame_gray.type() == CV_8UC1);
    const int w = frame_gray.cols;
    const int h = frame_gray.rows;
    const int start_y = static_cast<int>(h * CROP_TOP_RATIO);
    const int roi_h = h - start_y;

    cv::Mat roi = frame_gray(cv::Rect(0, start_y, w, roi_h));
    cv::Mat blurred;
    cv::GaussianBlur(roi, blurred, cv::Size(0, 0), 1.1);

    cv::Mat binary_roi;
    cv::threshold(blurred, binary_roi, ULTRA_BINARY_THRESH, 255, cv::THRESH_BINARY);

    cv::Mat clean_mask = cv::Mat::zeros(h, w, CV_8UC1);
    binary_roi.copyTo(clean_mask(cv::Rect(0, start_y, w, roi_h)));

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat opened;
    cv::morphologyEx(clean_mask, opened, cv::MORPH_OPEN, kernel);
    return opened;
}

std::vector<float> mask_to_input(const cv::Mat& mask) {
    cv::Mat resized;
    cv::resize(mask, resized, cv::Size(MASK_W, MASK_H), 0, 0, cv::INTER_LINEAR);
    std::vector<float> out(static_cast<size_t>(MASK_W * MASK_H));
    for (int y = 0; y < MASK_H; ++y) {
        const uint8_t* row = resized.ptr<uint8_t>(y);
        for (int x = 0; x < MASK_W; ++x) {
            out[static_cast<size_t>(y * MASK_W + x)] = row[x] / 255.0f;
        }
    }
    return out;
}

cv::Mat build_display_frame(const cv::Mat& mask, float servo, float duty) {
    cv::Mat resized;
    cv::resize(mask, resized, cv::Size(DISPLAY_W, DISPLAY_H), 0, 0, cv::INTER_LINEAR);
    cv::Mat bgr;
    cv::cvtColor(resized, bgr, cv::COLOR_GRAY2BGR);

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(2);
    oss << "Servo: " << servo << " | Duty: ";
    oss.precision(3);
    oss << duty;
    cv::putText(bgr, oss.str(), cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    return bgr;
}

std::vector<uint8_t> encode_jpeg(const cv::Mat& bgr, int quality) {
    std::vector<uint8_t> buf;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, quality};
    cv::imencode(".jpg", bgr, buf, params);
    return buf;
}

}  // namespace autopilot
