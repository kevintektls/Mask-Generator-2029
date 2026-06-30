#include "autopilot/camera.hpp"

#include "autopilot/config.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace autopilot {

namespace {

constexpr size_t HEADER_LEN = 12;
constexpr size_t FRAME_BYTES = static_cast<size_t>(MASK_W * MASK_H);

int connect_tcp_once(const std::string& addr, int timeout_ms) {
    const auto colon = addr.rfind(':');
    if (colon == std::string::npos) {
        throw std::runtime_error("invalid camera address (expected host:port)");
    }
    const std::string host = addr.substr(0, colon);
    const int port = std::stoi(addr.substr(colon + 1));

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* res = nullptr;
    const std::string port_str = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res) != 0) {
        throw std::runtime_error("camera bridge DNS resolve failed for " + host);
    }

    int fd = -1;
    for (addrinfo* p = res; p != nullptr; p = p->ai_next) {
        fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) {
            continue;
        }

        const int flags = fcntl(fd, F_GETFL, 0);
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);

        const int rc = ::connect(fd, p->ai_addr, p->ai_addrlen);
        if (rc == 0) {
            break;
        }
        if (rc < 0 && errno != EINPROGRESS) {
            ::close(fd);
            fd = -1;
            continue;
        }

        pollfd pfd{fd, POLLOUT, 0};
        if (poll(&pfd, 1, timeout_ms) <= 0) {
            ::close(fd);
            fd = -1;
            continue;
        }

        int err = 0;
        socklen_t err_len = sizeof(err);
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &err_len) != 0 || err != 0) {
            ::close(fd);
            fd = -1;
            continue;
        }
        break;
    }
    freeaddrinfo(res);

    if (fd < 0) {
        return -1;
    }

    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);

    int yes = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
    return fd;
}

}  // namespace

MonoCamera::MonoCamera(int fd) : fd_(fd) {
    rx_buf_.reserve(FRAME_BYTES + HEADER_LEN + 4096);
    scratch_.resize(64 * 1024);
}

MonoCamera MonoCamera::connect(const std::string& addr, ShouldContinue should_continue) {
    while (should_continue()) {
        const int fd = connect_tcp_once(addr, 1000);
        if (fd >= 0) {
            return MonoCamera(fd);
        }
        std::cerr << "[camera] waiting for camera bridge at " << addr
                  << " (start camera_bridge.py first)\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    throw std::runtime_error("camera connect cancelled");
}

std::optional<cv::Mat> MonoCamera::try_get_mask() {
    while (true) {
        const ssize_t n = ::read(fd_, scratch_.data(), scratch_.size());
        if (n > 0) {
            rx_buf_.insert(rx_buf_.end(), scratch_.begin(), scratch_.begin() + n);
            continue;
        }
        if (n == 0) {
            throw std::runtime_error("camera bridge closed the connection");
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            break;
        }
        throw std::runtime_error(std::string("camera read: ") + std::strerror(errno));
    }

    std::optional<cv::Mat> latest;
    while (true) {
        if (rx_buf_.size() < HEADER_LEN) {
            break;
        }

        if (std::memcmp(rx_buf_.data(), CAMERA_BRIDGE_MAGIC, 4) != 0) {
            auto it = std::search(
                rx_buf_.begin(), rx_buf_.end(),
                std::begin(CAMERA_BRIDGE_MAGIC), std::end(CAMERA_BRIDGE_MAGIC));
            if (it != rx_buf_.end()) {
                rx_buf_.erase(rx_buf_.begin(), it);
                continue;
            }
            if (rx_buf_.size() > 256 * 1024) {
                rx_buf_.clear();
            }
            break;
        }

        uint32_t w = 0;
        uint32_t h = 0;
        std::memcpy(&w, rx_buf_.data() + 4, 4);
        std::memcpy(&h, rx_buf_.data() + 8, 4);

        if (w != static_cast<uint32_t>(MASK_W) || h != static_cast<uint32_t>(MASK_H)) {
            rx_buf_.erase(rx_buf_.begin(), rx_buf_.begin() + 4);
            continue;
        }

        const size_t total = HEADER_LEN + FRAME_BYTES;
        if (rx_buf_.size() < total) {
            break;
        }

        cv::Mat mask(MASK_H, MASK_W, CV_8UC1);
        std::memcpy(mask.data, rx_buf_.data() + HEADER_LEN, FRAME_BYTES);
        rx_buf_.erase(rx_buf_.begin(), rx_buf_.begin() + static_cast<std::ptrdiff_t>(total));
        latest = std::move(mask);
    }

    return latest;
}

}  // namespace autopilot
