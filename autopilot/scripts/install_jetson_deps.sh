#!/usr/bin/env bash
# Install system packages required to build autopilot on Jetson Nano (Ubuntu/Debian).
set -euo pipefail

echo "[autopilot] Installing Jetson build dependencies..."

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  ninja-build \
  git \
  pkg-config \
  python3 \
  autoconf \
  automake \
  autoconf-archive \
  libtool \
  clang \
  libclang-dev \
  libudev-dev \
  libssl-dev \
  libusb-1.0-0-dev \
  libopencv-dev \
  nasm \
  libdw-dev \
  libelf-dev \
  fonts-dejavu-core

echo "[autopilot] Done. Build with:"
echo "  bash scripts/build_jetson.sh"
