#include "autopilot/gamepad.hpp"

#include "autopilot/config.hpp"

#include <SDL2/SDL.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

namespace autopilot {

namespace {

constexpr int POLL_INTERVAL_MS = 50;

float apply_deadzone(float value) {
    if (std::fabs(value) < GAMEPAD_DEADZONE) {
        return 0.0f;
    }
    const float sign = value > 0.0f ? 1.0f : -1.0f;
    return sign * (std::fabs(value) - GAMEPAD_DEADZONE) / (1.0f - GAMEPAD_DEADZONE);
}

float triggers_to_duty(float forward_raw, float backward_raw) {
    float throttle = std::clamp(forward_raw - backward_raw, -1.0f, 1.0f);
    throttle = apply_deadzone(throttle);
    return std::clamp(throttle * MANUAL_MAX_DUTY, -MANUAL_MAX_DUTY, MANUAL_MAX_DUTY);
}

float axis_to_servo(float axis_value) {
    const float v = apply_deadzone(axis_value);
    return std::clamp(SERVO_CENTER + v * MANUAL_SERVO_RANGE, 0.0f, 1.0f);
}

float normalize_axis(Sint16 raw) {
    return static_cast<float>(raw) / 32767.0f;
}

struct ControllerState {
    std::vector<SDL_GameController*> controllers;

    ~ControllerState() {
        for (SDL_GameController* c : controllers) {
            if (c) {
                SDL_GameControllerClose(c);
            }
        }
    }
};

}  // namespace

GamepadMonitor::GamepadMonitor() {
    auto* state = new ControllerState();
    for (int i = 0; i < SDL_NumJoysticks(); ++i) {
        if (SDL_IsGameController(i)) {
            SDL_GameController* ctrl = SDL_GameControllerOpen(i);
            if (ctrl) {
                state->controllers.push_back(ctrl);
            }
        }
    }

    thread_ = std::thread([this, state]() {
        SDL_GameController* primary =
            state->controllers.empty() ? nullptr : state->controllers.front();
        bool lb_prev = false;

        while (running_.load()) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
            }

            if (primary != nullptr) {
                const bool lb_now =
                    SDL_GameControllerGetButton(primary, SDL_CONTROLLER_BUTTON_LEFTSHOULDER) != 0;
                if (lb_now && !lb_prev) {
                    const bool manual = !manual_mode_.load(std::memory_order_seq_cst);
                    manual_mode_.store(manual, std::memory_order_seq_cst);
                    std::cout << "[gamepad] mode: " << (manual ? "manual" : "autonomous") << '\n';
                }
                lb_prev = lb_now;

                const float forward =
                    normalize_axis(SDL_GameControllerGetAxis(primary, SDL_CONTROLLER_AXIS_TRIGGERRIGHT));
                const float backward =
                    normalize_axis(SDL_GameControllerGetAxis(primary, SDL_CONTROLLER_AXIS_TRIGGERLEFT));
                const float steering =
                    normalize_axis(SDL_GameControllerGetAxis(primary, SDL_CONTROLLER_AXIS_LEFTX));

                manual_duty_.store(triggers_to_duty(forward, backward), std::memory_order_seq_cst);
                manual_servo_.store(axis_to_servo(steering), std::memory_order_seq_cst);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
        }
        delete state;
    });
}

GamepadMonitor::~GamepadMonitor() {
    running_.store(false);
    if (thread_.joinable()) {
        thread_.join();
    }
    SDL_QuitSubSystem(SDL_INIT_GAMECONTROLLER);
}

std::unique_ptr<GamepadMonitor> GamepadMonitor::try_start() {
    SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");
    if (SDL_InitSubSystem(SDL_INIT_GAMECONTROLLER) != 0) {
        std::cerr << "[gamepad] SDL init failed: " << SDL_GetError() << '\n';
        return nullptr;
    }

    if (SDL_NumJoysticks() <= 0) {
        std::cout << "[gamepad] no gamepad detected — manual mode disabled\n";
        SDL_QuitSubSystem(SDL_INIT_GAMECONTROLLER);
        return nullptr;
    }

    bool has_controller = false;
    for (int i = 0; i < SDL_NumJoysticks(); ++i) {
        if (SDL_IsGameController(i)) {
            has_controller = true;
            break;
        }
    }
    if (!has_controller) {
        std::cout << "[gamepad] no compatible gamepad — manual mode disabled\n";
        SDL_QuitSubSystem(SDL_INIT_GAMECONTROLLER);
        return nullptr;
    }

    std::cout << "[gamepad] connected — LB toggles manual/autonomous\n";
    return std::unique_ptr<GamepadMonitor>(new GamepadMonitor());
}

}  // namespace autopilot
