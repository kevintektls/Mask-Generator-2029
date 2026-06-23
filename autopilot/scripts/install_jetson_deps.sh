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
  libopencv-dev \
  libudev-dev \
  libssl-dev

# opencv-rust build script must find an executable named `clang`.
if ! command -v clang >/dev/null 2>&1; then
  echo "[autopilot] 'clang' not in PATH — searching for versioned binary..."
  for candidate in /usr/bin/clang-* /usr/lib/llvm-*/bin/clang; do
    if [[ -x "$candidate" ]]; then
      echo "[autopilot] Linking $candidate -> /usr/bin/clang"
      sudo ln -sf "$candidate" /usr/bin/clang
      break
    fi
  done
fi

if ! command -v clang >/dev/null 2>&1; then
  echo "[autopilot] ERROR: clang still not found. Install manually:"
  echo "  sudo apt-get install -y clang libclang-dev"
  exit 1
fi

echo "[autopilot] clang: $(clang --version | head -1)"
echo "[autopilot] opencv: $(pkg-config --modversion opencv4 2>/dev/null || echo 'not found')"
echo "[autopilot] Done. Rebuild with:"
echo "  cd autopilot && cargo build --release -p autopilot"
