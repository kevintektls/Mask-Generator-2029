#pragma once

#include <opencv2/core.hpp>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace autopilot {

class MonoCamera {
public:
    using ShouldContinue = std::function<bool()>;

    static MonoCamera connect(
        const std::string& addr,
        ShouldContinue should_continue = [] { return true; });

    std::optional<cv::Mat> try_get_mask();

private:
    explicit MonoCamera(int fd);

    int fd_ = -1;
    std::vector<uint8_t> rx_buf_;
    std::vector<uint8_t> scratch_;
};

}  // namespace autopilot
