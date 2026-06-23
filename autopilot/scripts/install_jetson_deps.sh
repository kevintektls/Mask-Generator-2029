#!/usr/bin/env bash
# Install system packages required to build autopilot on Jetson Nano (Ubuntu/Debian).
set -euo pipefail

echo "[autopilot] Installing Jetson build dependencies..."

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
  clang \
  libclang-dev \
  libudev-dev \
  libssl-dev \
  fonts-dejavu-core

echo "[autopilot] Done. Rebuild with:"
echo "  cd autopilot && cargo build --release -p autopilot"
