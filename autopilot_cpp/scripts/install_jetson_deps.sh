#!/usr/bin/env bash
# Install system packages required to build autopilot_cpp on Jetson Nano.
set -euo pipefail

ORT_VERSION="1.20.1"
ORT_DIR="/opt/onnxruntime"
ORT_TGZ="onnxruntime-linux-aarch64-${ORT_VERSION}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_TGZ}"

echo "[autopilot_cpp] Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  pkg-config \
  libopencv-dev \
  libsdl2-dev \
  libjpeg-dev \
  fonts-dejavu-core \
  curl \
  python3-pip

# Ubuntu 18.04 (Jetson) ships cmake 3.10; autopilot_cpp requires >= 3.16.
cmake_ver() {
  cmake --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' || true
}
CMAKE_VER="$(cmake_ver)"
CMAKE_MAJOR="${CMAKE_VER%%.*}"
CMAKE_MINOR="$(echo "${CMAKE_VER}" | cut -d. -f2)"
if [[ -z "${CMAKE_VER}" ]] || [[ "${CMAKE_MAJOR}" -lt 3 ]] || [[ "${CMAKE_MAJOR}" -eq 3 && "${CMAKE_MINOR}" -lt 16 ]]; then
  echo "[autopilot_cpp] Upgrading cmake (found ${CMAKE_VER:-none}, need >= 3.16)..."
  sudo pip3 install --upgrade cmake
  hash -r 2>/dev/null || true
  CMAKE_VER="$(cmake_ver)"
  echo "[autopilot_cpp] cmake now: ${CMAKE_VER}"
fi

if [[ ! -f "${ORT_DIR}/lib/libonnxruntime.so" ]]; then
  echo "[autopilot_cpp] Downloading ONNX Runtime ${ORT_VERSION} (aarch64)..."
  tmp=$(mktemp -d)
  curl -fsSL "${ORT_URL}" -o "${tmp}/${ORT_TGZ}"
  sudo mkdir -p "${ORT_DIR}"
  sudo tar -xzf "${tmp}/${ORT_TGZ}" -C "${ORT_DIR}" --strip-components=1
  rm -rf "${tmp}"
  echo "[autopilot_cpp] ONNX Runtime installed to ${ORT_DIR}"
else
  echo "[autopilot_cpp] ONNX Runtime already present at ${ORT_DIR}"
fi

echo
echo "[autopilot_cpp] Export model (once, needs torch+onnx):"
echo "  pip install torch onnx"
echo "  python3 tools/export_onnx.py"
echo
echo "[autopilot_cpp] Build:"
echo "  export ONNXRUNTIME_ROOT=${ORT_DIR}"
echo "  bash scripts/build_jetson.sh"
