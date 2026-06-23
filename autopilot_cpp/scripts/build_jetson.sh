#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

ORT_ROOT="${ONNXRUNTIME_ROOT:-/opt/onnxruntime}"
if [[ ! -d "${ORT_ROOT}/lib" ]]; then
  echo "[autopilot_cpp] ERROR: ONNXRUNTIME_ROOT not set or invalid (${ORT_ROOT})"
  echo "  Run: bash scripts/install_jetson_deps.sh"
  exit 1
fi

cmake -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT="${ORT_ROOT}"
cmake --build build -j"$(nproc)"
echo
echo "[autopilot_cpp] Done: build/autopilot_cpp"
echo "[autopilot_cpp] Run:"
echo "  python3 ../autopilot/tools/camera_bridge.py --fps 60 &"
echo "  ./build/autopilot_cpp --model ../model/pilot_model.onnx"
