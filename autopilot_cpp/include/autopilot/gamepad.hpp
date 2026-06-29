#pragma once

#include "autopilot/config.hpp"

#include <atomic>
#include <memory>
#include <thread>

namespace autopilot {

class GamepadMonitor {
public:
    static std::unique_ptr<GamepadMonitor> try_start();

    bool manual_mode() const { return manual_mode_.load(std::memory_order_seq_cst); }
    float manual_duty() const { return manual_duty_.load(std::memory_order_seq_cst); }
    float manual_servo() const { return manual_servo_.load(std::memory_order_seq_cst); }
    float duty_offset() const { return duty_offset_.load(std::memory_order_seq_cst); }

    GamepadMonitor(const GamepadMonitor&) = delete;
    GamepadMonitor& operator=(const GamepadMonitor&) = delete;
    ~GamepadMonitor();

private:
    GamepadMonitor();

    std::atomic<bool> manual_mode_{false};
    std::atomic<float> manual_duty_{0.0f};
    std::atomic<float> manual_servo_{SERVO_CENTER};
    std::atomic<float> duty_offset_{0.0f};
    std::atomic<bool> running_{true};
    std::thread thread_;
};

}  // namespace autopilot
