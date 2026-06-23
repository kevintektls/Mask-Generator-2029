#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <thread>

namespace autopilot {

class GamepadMonitor {
public:
    static std::optional<GamepadMonitor> try_start();

    bool is_emergency() const { return emergency_.load(std::memory_order_seq_cst); }

    GamepadMonitor(const GamepadMonitor&) = delete;
    GamepadMonitor& operator=(const GamepadMonitor&) = delete;
    ~GamepadMonitor();

private:
    GamepadMonitor();

    std::atomic<bool> emergency_{false};
    std::atomic<bool> running_{true};
    std::thread thread_;
};

}  // namespace autopilot
