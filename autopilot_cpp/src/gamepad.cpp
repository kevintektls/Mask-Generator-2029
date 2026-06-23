#include "autopilot/gamepad.hpp"

#include <SDL2/SDL.h>

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

namespace autopilot {

namespace {

constexpr int POLL_INTERVAL_MS = 20;

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
        while (running_.load()) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_CONTROLLERBUTTONDOWN &&
                    event.cbutton.button == SDL_CONTROLLER_BUTTON_LEFTSHOULDER) {
                    emergency_.store(true, std::memory_order_seq_cst);
                }
            }

            for (SDL_GameController* ctrl : state->controllers) {
                if (SDL_GameControllerGetButton(ctrl, SDL_CONTROLLER_BUTTON_LEFTSHOULDER)) {
                    emergency_.store(true, std::memory_order_seq_cst);
                }
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

std::optional<GamepadMonitor> GamepadMonitor::try_start() {
    if (SDL_InitSubSystem(SDL_INIT_GAMECONTROLLER) != 0) {
        std::cerr << "[gamepad] SDL init failed: " << SDL_GetError() << '\n';
        return std::nullopt;
    }

    if (SDL_NumJoysticks() <= 0) {
        std::cout << "[gamepad] no gamepad detected — emergency LB disabled\n";
        SDL_QuitSubSystem(SDL_INIT_GAMECONTROLLER);
        return std::nullopt;
    }

    bool has_controller = false;
    for (int i = 0; i < SDL_NumJoysticks(); ++i) {
        if (SDL_IsGameController(i)) {
            has_controller = true;
            break;
        }
    }
    if (!has_controller) {
        std::cout << "[gamepad] no compatible gamepad — emergency LB disabled\n";
        SDL_QuitSubSystem(SDL_INIT_GAMECONTROLLER);
        return std::nullopt;
    }

    std::cout << "[gamepad] connected for emergency stop (LB)\n";
    return GamepadMonitor();
}

}  // namespace autopilot
