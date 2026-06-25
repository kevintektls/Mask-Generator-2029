#include "autopilot/camera.hpp"
#include "autopilot/config.hpp"
#include "autopilot/gamepad.hpp"
#include "autopilot/model.hpp"
#include "autopilot/stream.hpp"
#include "autopilot/vision.hpp"
#include "autopilot/vesc.hpp"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>

namespace {

std::atomic<bool> g_run{true};

void on_shutdown_signal(int) {
    std::cout << "[autopilot] interrupt received — stopping\n";
    std::cout.flush();
    g_run.store(false);
}

void install_shutdown_handlers() {
    std::signal(SIGINT, on_shutdown_signal);
    std::signal(SIGTERM, on_shutdown_signal);
}

}  // namespace

int main(int argc, char* argv[]) {
    install_shutdown_handlers();

    try {
        const autopilot::Cli cli = autopilot::parse_cli(argc, argv);

        std::cout << "[autopilot] initializing autopilot IA++ (C++)\n";

        autopilot::OnnxModel model(cli.model);
        std::cout << "[autopilot] model loaded: " << cli.model << '\n';

        autopilot::VescClient vesc = autopilot::VescClient::connect(cli.vesc_port);
        std::cout << "[autopilot] VESC connected\n";

        auto gamepad = autopilot::GamepadMonitor::try_start();

        autopilot::MjpegServer stream(cli.stream_port);
        std::cout << "[autopilot] video stream http://localhost:" << cli.stream_port
                  << " (or Jetson IP)\n";

        autopilot::MonoCamera camera = autopilot::MonoCamera::connect(
            cli.camera_addr, [] { return g_run.load(); });
        std::cout << "[autopilot] camera bridge connected at " << cli.camera_addr << '\n';

        vesc.servo_center();
        vesc.set_duty(0.0f);
        std::this_thread::sleep_for(std::chrono::seconds(1));

        std::cout << "[autopilot] operational — LB = emergency brake\n";

        while (g_run.load()) {
            if (gamepad && gamepad->is_emergency()) {
                std::cout << "[autopilot] EMERGENCY: LB pressed — applying motor brake\n";
                vesc.set_brake(autopilot::EMERGENCY_BRAKE_A);
                vesc.servo_center();
                break;
            }

            auto mask = camera.try_get_mask();
            if (!mask) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            const std::vector<float> input = autopilot::mask_to_input(*mask);
            float servo_pos = model.predict_flat(input);
            servo_pos = std::clamp(servo_pos, 0.0f, 1.0f);
            const float current_duty = autopilot::adaptive_duty(servo_pos);

            vesc.set_servo(servo_pos);
            vesc.set_duty(current_duty);

            const cv::Mat display = autopilot::build_display_frame(*mask, servo_pos, current_duty);
            stream.publish_frame(autopilot::encode_jpeg(display));
        }

        std::cout << "[autopilot] shutting down systems\n";
        vesc.safe_stop();
        stream.stop();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[autopilot] fatal: " << e.what() << '\n';
        return 1;
    }
}
