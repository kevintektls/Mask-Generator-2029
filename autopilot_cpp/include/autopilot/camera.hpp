#pragma once

#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <vector>

namespace autopilot {

class MonoCamera {
public:
    static MonoCamera connect(const std::string& addr);

    std::optional<cv::Mat> try_get_gray();

private:
    explicit MonoCamera(int fd);

    int fd_ = -1;
    std::vector<uint8_t> rx_buf_;
    std::vector<uint8_t> scratch_;
};

}  // namespace autopilot
