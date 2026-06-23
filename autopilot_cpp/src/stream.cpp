#include "autopilot/stream.hpp"

#include "autopilot/config.hpp"

#define CPPHTTPLIB_OPENSSL_SUPPORT 0
#include "httplib.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace autopilot {

struct MjpegServer::Impl {
    std::mutex frame_mutex;
    std::vector<uint8_t> latest_jpeg;
    httplib::Server server;
    std::thread server_thread;
    std::atomic<bool> running{true};

    explicit Impl(uint16_t port) {
        server.Get("/", [this](const httplib::Request&, httplib::Response& res) {
            const std::string boundary = "frame";
            const std::string content_type = "multipart/x-mixed-replace; boundary=" + boundary;
            res.set_chunked_content_provider(
                content_type,
                [this, boundary](size_t, httplib::DataSink& sink) {
                    const auto interval = std::chrono::duration<double>(1.0 / CAM_FPS);
                    while (running.load()) {
                        std::vector<uint8_t> jpeg;
                        {
                            std::lock_guard<std::mutex> lock(frame_mutex);
                            jpeg = latest_jpeg;
                        }
                        if (!jpeg.empty()) {
                            std::string part;
                            part += "--" + boundary + "\r\n";
                            part += "Content-Type: image/jpeg\r\n";
                            part += "Content-Length: " + std::to_string(jpeg.size()) + "\r\n\r\n";
                            if (!sink.write(part.data(), part.size())) {
                                return false;
                            }
                            if (!sink.write(reinterpret_cast<const char*>(jpeg.data()), jpeg.size())) {
                                return false;
                            }
                            if (!sink.write("\r\n", 2)) {
                                return false;
                            }
                        }
                        std::this_thread::sleep_for(
                            std::chrono::duration_cast<std::chrono::milliseconds>(interval));
                    }
                    return false;
                });
        });

        server_thread = std::thread([this, port]() {
            server.listen("0.0.0.0", port);
        });
    }

    void stop() {
        running.store(false);
        server.stop();
        if (server_thread.joinable()) {
            server_thread.join();
        }
    }
};

MjpegServer::MjpegServer(uint16_t port) : impl_(std::make_unique<Impl>(port)) {}

MjpegServer::~MjpegServer() {
    stop();
}

void MjpegServer::publish_frame(std::vector<uint8_t> jpeg) {
    std::lock_guard<std::mutex> lock(impl_->frame_mutex);
    impl_->latest_jpeg = std::move(jpeg);
}

void MjpegServer::stop() {
    if (impl_) {
        impl_->stop();
    }
}

}  // namespace autopilot
