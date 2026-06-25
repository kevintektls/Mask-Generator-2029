#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace autopilot {

// Display / stream
constexpr int DISPLAY_W = 640;
constexpr int DISPLAY_H = 480;
constexpr int CAM_FPS = 60;
constexpr uint16_t STREAM_PORT = 8080;

// Camera bridge (Python depthai 2.x → vision_preprocess masks)
constexpr int MASK_W = 160;
constexpr int MASK_H = 120;
constexpr const char* CAMERA_BRIDGE_ADDR = "127.0.0.1:9000";
constexpr char CAMERA_BRIDGE_MAGIC[4] = {'M', 'A', 'S', 'K'};

// Vision
constexpr double CROP_TOP_RATIO = 0.20;
constexpr uint8_t ULTRA_BINARY_THRESH = 220;

// VESC
constexpr const char* VESC_PORT = "/dev/ttyACM0";
constexpr int VESC_BAUDRATE = 115200;
constexpr int VESC_TIMEOUT_MS = 1000;
constexpr int VESC_CONNECT_RETRIES = 8;
constexpr int VESC_CONNECT_SETTLE_MS = 1000;

// Driving
constexpr float SERVO_CENTER = 0.5f;
constexpr float SERVO_RANGE = 0.48f;
constexpr float DUTY_MIN = 0.050f;
constexpr float DUTY_MAX = 0.070f;
constexpr float STEER_THRESHOLD = 0.08f;

constexpr float EMERGENCY_BRAKE_A = 15.0f;
constexpr float SHUTDOWN_BRAKE_A = 10.0f;

constexpr const char* DEFAULT_MODEL_PATH = "../model/pilot_model.onnx";

inline float adaptive_duty(float servo_pos) {
    const float steering_intensity = std::fabs(servo_pos - SERVO_CENTER);
    if (steering_intensity < STEER_THRESHOLD) {
        return DUTY_MAX;
    }
    const float factor = std::clamp(
        (steering_intensity - STEER_THRESHOLD) / (SERVO_RANGE - STEER_THRESHOLD),
        0.0f, 1.0f);
    return DUTY_MAX - factor * (DUTY_MAX - DUTY_MIN);
}

struct Cli {
    std::string model = DEFAULT_MODEL_PATH;
    std::string vesc_port = VESC_PORT;
    uint16_t stream_port = STREAM_PORT;
    uint32_t cam_fps = CAM_FPS;
    std::string camera_addr = CAMERA_BRIDGE_ADDR;
};

Cli parse_cli(int argc, char* argv[]);

}  // namespace autopilot
