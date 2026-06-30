#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace autopilot {

class MjpegServer {
public:
    explicit MjpegServer(uint16_t port);
    ~MjpegServer();

    MjpegServer(const MjpegServer&) = delete;
    MjpegServer& operator=(const MjpegServer&) = delete;

    void publish_frame(std::vector<uint8_t> jpeg);
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace autopilot
