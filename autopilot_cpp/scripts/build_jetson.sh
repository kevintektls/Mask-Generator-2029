#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

ORT_ROOT="${ONNXRUNTIME_ROOT:-/opt/onnxruntime}"
if [[ ! -d "${ORT_ROOT}/lib" ]]; then
  echo "[autopilot_cpp] ERROR: ONNXRUNTIME_ROOT not set or invalid (${ORT_ROOT})"
  echo "  Run: bash scripts/install_jetson_deps.sh"
  exit 1
fi

CMAKE_VER="$(cmake --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' || true)"
CMAKE_MAJOR="${CMAKE_VER%%.*}"
CMAKE_MINOR="$(echo "${CMAKE_VER}" | cut -d. -f2)"
if [[ -z "${CMAKE_VER}" ]] || [[ "${CMAKE_MAJOR}" -lt 3 ]] || [[ "${CMAKE_MAJOR}" -eq 3 && "${CMAKE_MINOR}" -lt 16 ]]; then
  echo "[autopilot_cpp] ERROR: cmake >= 3.16 required (found: ${CMAKE_VER:-none})"
  echo "  Run: bash scripts/install_jetson_deps.sh"
  exit 1
fi

mkdir -p build
cmake -S "${PROJECT_ROOT}" -B "${PROJECT_ROOT}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT="${ORT_ROOT}"
cmake --build "${PROJECT_ROOT}/build" -j"$(nproc)"
echo
echo "[autopilot_cpp] Done: build/autopilot_cpp"
echo "[autopilot_cpp] Run:"
echo "  python3 ../autopilot/tools/camera_bridge.py --fps 60 &"
echo "  ./build/autopilot_cpp --model ../model/pilot_model.onnx"
