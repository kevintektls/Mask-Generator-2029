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

echo "[autopilot] Done. Build with:"
echo "  bash scripts/build_jetson.sh"
