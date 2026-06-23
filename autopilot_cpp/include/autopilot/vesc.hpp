#pragma once

#include "autopilot/config.hpp"

#include <string>

namespace autopilot {

class VescClient {
public:
    static VescClient connect(const std::string& port_path);
    ~VescClient();

    VescClient(const VescClient&) = delete;
    VescClient& operator=(const VescClient&) = delete;
    VescClient(VescClient&&) noexcept;
    VescClient& operator=(VescClient&&) noexcept;

    void set_duty(float duty);
    void set_brake(float amps);
    void set_servo(float pos);
    void servo_center();
    void safe_stop();

private:
    explicit VescClient(int fd);

    int fd_ = -1;
    float shutdown_brake_a_ = SHUTDOWN_BRAKE_A;
};

}  // namespace autopilot
