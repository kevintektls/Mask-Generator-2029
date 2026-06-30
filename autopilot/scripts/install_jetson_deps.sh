#!/usr/bin/env bash
# Install system packages required to build autopilot on Jetson Nano (Ubuntu/Debian).
set -euo pipefail

echo "[autopilot] Installing Jetson build dependencies..."

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  pkg-config \
  libudev-dev \
  fonts-dejavu-core

echo "[autopilot] Python camera bridge (install if missing):"
echo "  pip3 install --user depthai==2.29.0 opencv-python numpy"
echo "[autopilot] Build with:"
echo "  bash scripts/build_jetson.sh"
echo "[autopilot] If you hit depthai-sys / CMake errors from an old checkout:"
echo "  bash scripts/fix_jetson_depthai_build.sh"
