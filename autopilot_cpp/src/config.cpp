#include "autopilot/config.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace autopilot {

namespace {

void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "  --model <path>         ONNX weights (default: " << DEFAULT_MODEL_PATH << ")\n"
        << "  --vesc-port <path>     VESC serial (default: " << VESC_PORT << ")\n"
        << "  --stream-port <port>   MJPEG HTTP (default: " << STREAM_PORT << ")\n"
        << "  --cam-fps <fps>        Camera bridge FPS hint (default: " << CAM_FPS << ")\n"
        << "  --camera-addr <host:port>  Bridge TCP (default: " << CAMERA_BRIDGE_ADDR << ")\n";
}

bool next_arg(int argc, char* argv[], int& i, const char*& out) {
    if (i + 1 >= argc) {
        return false;
    }
    out = argv[++i];
    return true;
}

}  // namespace

Cli parse_cli(int argc, char* argv[]) {
    Cli cli;
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        }
        if (std::strcmp(arg, "--model") == 0) {
            const char* v = nullptr;
            if (!next_arg(argc, argv, i, v)) {
                throw std::runtime_error("--model requires a path");
            }
            cli.model = v;
        } else if (std::strcmp(arg, "--vesc-port") == 0) {
            const char* v = nullptr;
            if (!next_arg(argc, argv, i, v)) {
                throw std::runtime_error("--vesc-port requires a path");
            }
            cli.vesc_port = v;
        } else if (std::strcmp(arg, "--stream-port") == 0) {
            const char* v = nullptr;
            if (!next_arg(argc, argv, i, v)) {
                throw std::runtime_error("--stream-port requires a port");
            }
            cli.stream_port = static_cast<uint16_t>(std::stoi(v));
        } else if (std::strcmp(arg, "--cam-fps") == 0) {
            const char* v = nullptr;
            if (!next_arg(argc, argv, i, v)) {
                throw std::runtime_error("--cam-fps requires a value");
            }
            cli.cam_fps = static_cast<uint32_t>(std::stoi(v));
        } else if (std::strcmp(arg, "--camera-addr") == 0) {
            const char* v = nullptr;
            if (!next_arg(argc, argv, i, v)) {
                throw std::runtime_error("--camera-addr requires host:port");
            }
            cli.camera_addr = v;
        } else {
            throw std::runtime_error(std::string("unknown argument: ") + arg);
        }
    }
    return cli;
}

}  // namespace autopilot
