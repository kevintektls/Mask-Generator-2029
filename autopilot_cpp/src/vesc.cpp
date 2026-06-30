#include "autopilot/vesc.hpp"

#include "autopilot/config.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <termios.h>

namespace autopilot {

namespace {

constexpr uint8_t COMM_SET_DUTY = 5;
constexpr uint8_t COMM_SET_CURRENT_BRAKE = 7;
constexpr uint8_t COMM_SET_SERVO_POS = 12;

constexpr float DUTY_SCALE = 100000.0f;
constexpr float CURRENT_SCALE = 1000.0f;
constexpr float SERVO_SCALE = 1000.0f;

uint16_t crc16(const uint8_t* data, size_t len) {
    uint16_t crc = 0;
    for (size_t i = 0; i < len; ++i) {
        crc ^= static_cast<uint16_t>(data[i]) << 8;
        for (int b = 0; b < 8; ++b) {
            if (crc & 0x8000) {
                crc = static_cast<uint16_t>((crc << 1) ^ 0x1021);
            } else {
                crc = static_cast<uint16_t>(crc << 1);
            }
        }
    }
    return crc;
}

std::vector<uint8_t> encode_packet(const std::vector<uint8_t>& payload) {
    if (payload.empty() || payload.size() > 255) {
        throw std::runtime_error("invalid VESC payload size");
    }
    std::vector<uint8_t> out;
    out.reserve(payload.size() + 6);
    out.push_back(2);
    out.push_back(static_cast<uint8_t>(payload.size()));
    out.insert(out.end(), payload.begin(), payload.end());
    const uint16_t c = crc16(out.data() + 2, payload.size());
    out.push_back(static_cast<uint8_t>(c >> 8));
    out.push_back(static_cast<uint8_t>(c & 0xFF));
    out.push_back(3);
    return out;
}

void append_be32(std::vector<uint8_t>& payload, int32_t value) {
    payload.push_back(static_cast<uint8_t>((value >> 24) & 0xFF));
    payload.push_back(static_cast<uint8_t>((value >> 16) & 0xFF));
    payload.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
    payload.push_back(static_cast<uint8_t>(value & 0xFF));
}

void append_be16(std::vector<uint8_t>& payload, int16_t value) {
    payload.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
    payload.push_back(static_cast<uint8_t>(value & 0xFF));
}

std::vector<uint8_t> encode_set_duty(float duty) {
    const int32_t scaled = static_cast<int32_t>(duty * DUTY_SCALE);
    std::vector<uint8_t> payload = {COMM_SET_DUTY};
    append_be32(payload, scaled);
    return encode_packet(payload);
}

std::vector<uint8_t> encode_set_brake(float amps) {
    const int32_t scaled = static_cast<int32_t>(amps * CURRENT_SCALE);
    std::vector<uint8_t> payload = {COMM_SET_CURRENT_BRAKE};
    append_be32(payload, scaled);
    return encode_packet(payload);
}

std::vector<uint8_t> encode_set_servo(float pos) {
    const float clamped = std::clamp(pos, 0.0f, 1.0f);
    const int16_t scaled = static_cast<int16_t>(clamped * SERVO_SCALE);
    std::vector<uint8_t> payload = {COMM_SET_SERVO_POS};
    append_be16(payload, scaled);
    return encode_packet(payload);
}

speed_t baud_to_flag(int baud) {
    switch (baud) {
        case 9600: return B9600;
        case 19200: return B19200;
        case 38400: return B38400;
        case 57600: return B57600;
        case 115200: return B115200;
        default: throw std::runtime_error("unsupported baud rate");
    }
}

int open_serial(const std::string& port_path) {
    int fd = ::open(port_path.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) {
        throw std::runtime_error(std::string("opening VESC port ") + port_path + ": " + std::strerror(errno));
    }

    termios tty{};
    if (tcgetattr(fd, &tty) != 0) {
        ::close(fd);
        throw std::runtime_error("tcgetattr failed");
    }

    cfsetospeed(&tty, baud_to_flag(VESC_BAUDRATE));
    cfsetispeed(&tty, baud_to_flag(VESC_BAUDRATE));

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_cflag |= CLOCAL | CREAD;
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON);
    tty.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
    tty.c_oflag &= ~OPOST;

    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = VESC_TIMEOUT_MS / 100;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        ::close(fd);
        throw std::runtime_error("tcsetattr failed");
    }
    return fd;
}

}  // namespace

VescClient::VescClient(int fd) : fd_(fd) {}

VescClient VescClient::connect(const std::string& port_path) {
    std::string last_err;
    for (int attempt = 1; attempt <= VESC_CONNECT_RETRIES; ++attempt) {
        try {
            return VescClient(open_serial(port_path));
        } catch (const std::exception& e) {
            last_err = e.what();
            std::this_thread::sleep_for(std::chrono::milliseconds(VESC_CONNECT_SETTLE_MS));
        }
    }
    throw std::runtime_error(last_err.empty() ? "VESC connect failed" : last_err);
}

VescClient::~VescClient() {
    if (fd_ >= 0) {
        safe_stop();
        ::close(fd_);
    }
}

VescClient::VescClient(VescClient&& other) noexcept : fd_(other.fd_), shutdown_brake_a_(other.shutdown_brake_a_) {
    other.fd_ = -1;
}

VescClient& VescClient::operator=(VescClient&& other) noexcept {
    if (this != &other) {
        if (fd_ >= 0) {
            ::close(fd_);
        }
        fd_ = other.fd_;
        shutdown_brake_a_ = other.shutdown_brake_a_;
        other.fd_ = -1;
    }
    return *this;
}

void VescClient::set_duty(float duty) {
    const auto pkt = encode_set_duty(duty);
    if (::write(fd_, pkt.data(), pkt.size()) < 0) {
        throw std::runtime_error(std::string("VESC write duty: ") + std::strerror(errno));
    }
}

void VescClient::set_brake(float amps) {
    const auto pkt = encode_set_brake(amps);
    if (::write(fd_, pkt.data(), pkt.size()) < 0) {
        throw std::runtime_error(std::string("VESC write brake: ") + std::strerror(errno));
    }
}

void VescClient::set_servo(float pos) {
    const auto pkt = encode_set_servo(pos);
    if (::write(fd_, pkt.data(), pkt.size()) < 0) {
        throw std::runtime_error(std::string("VESC write servo: ") + std::strerror(errno));
    }
}

void VescClient::servo_center() {
    set_servo(SERVO_CENTER);
}

void VescClient::safe_stop() {
    try {
        set_duty(0.0f);
    } catch (...) {
    }
    try {
        set_brake(shutdown_brake_a_);
    } catch (...) {
    }
    try {
        servo_center();
    } catch (...) {
    }
}

}  // namespace autopilot
