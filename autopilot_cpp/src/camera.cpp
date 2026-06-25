#include "autopilot/camera.hpp"

#include "autopilot/config.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

namespace autopilot {

namespace {

constexpr size_t HEADER_LEN = 12;

int connect_tcp(const std::string& addr) {
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
        if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0) {
            break;
        }
        ::close(fd);
        fd = -1;
    }
    freeaddrinfo(res);

    if (fd < 0) {
        throw std::runtime_error("connecting to camera bridge at " + addr);
    }

    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    int yes = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
    return fd;
}

}  // namespace

MonoCamera::MonoCamera(int fd) : fd_(fd) {
    rx_buf_.reserve(static_cast<size_t>(MONO_W * MONO_H + HEADER_LEN));
    scratch_.resize(4096);
}

MonoCamera MonoCamera::connect(const std::string& addr) {
    return MonoCamera(connect_tcp(addr));
}

std::optional<cv::Mat> MonoCamera::try_get_gray() {
    while (true) {
        if (rx_buf_.size() >= HEADER_LEN) {
            if (std::memcmp(rx_buf_.data(), CAMERA_BRIDGE_MAGIC, 4) != 0) {
                auto it = std::search(
                    rx_buf_.begin(), rx_buf_.end(),
                    std::begin(CAMERA_BRIDGE_MAGIC), std::end(CAMERA_BRIDGE_MAGIC));
                if (it != rx_buf_.end()) {
                    rx_buf_.erase(rx_buf_.begin(), it);
                    continue;
                }
                rx_buf_.clear();
                throw std::runtime_error("lost sync with camera bridge stream");
            }

            uint32_t w = 0, h = 0;
            std::memcpy(&w, rx_buf_.data() + 4, 4);
            std::memcpy(&h, rx_buf_.data() + 8, 4);

            const size_t payload = static_cast<size_t>(w) * static_cast<size_t>(h);
            const size_t total = HEADER_LEN + payload;
            if (rx_buf_.size() < total) {
                break;
            }

            cv::Mat gray(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
            std::memcpy(gray.data, rx_buf_.data() + HEADER_LEN, payload);
            rx_buf_.erase(rx_buf_.begin(), rx_buf_.begin() + static_cast<std::ptrdiff_t>(total));
            return gray;
        }

        const ssize_t n = ::read(fd_, scratch_.data(), scratch_.size());
        if (n > 0) {
            rx_buf_.insert(rx_buf_.end(), scratch_.begin(), scratch_.begin() + n);
            continue;
        }
        if (n == 0) {
            throw std::runtime_error("camera bridge closed the connection");
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return std::nullopt;
        }
        throw std::runtime_error(std::string("camera read: ") + std::strerror(errno));
    }
    return std::nullopt;
}

}  // namespace autopilot
