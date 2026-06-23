#!/usr/bin/env bash
# Build autopilot on Jetson Nano (no depthai-sys — camera via Python bridge).
set -euo pipefail
cd "$(dirname "$0")/.."

# Stale DEPTHAI_CORE_ROOT or target/dai-build/ from old depthai-sys builds breaks CMake.
if [[ -n "${DEPTHAI_CORE_ROOT:-}" ]] && [[ ! -f "${DEPTHAI_CORE_ROOT}/CMakeLists.txt" ]]; then
  echo "[autopilot] Unsetting invalid DEPTHAI_CORE_ROOT=$DEPTHAI_CORE_ROOT"
  unset DEPTHAI_CORE_ROOT
fi
DAI_BUILD="target/dai-build/v3.6.1"
if [[ -d "$DAI_BUILD" ]] && [[ ! -f "$DAI_BUILD/depthai-core/CMakeLists.txt" ]]; then
  echo "[autopilot] Removing stale depthai-core checkout in $DAI_BUILD"
  rm -rf target/dai-build
fi

echo "[autopilot] cargo build --release -p autopilot"
cargo build --release -p autopilot "$@"
echo "[autopilot] Done: target/release/autopilot"
echo "[autopilot] Run:"
echo "  python3 tools/camera_bridge.py &"
echo "  ./target/release/autopilot --model ../model/pilot_model.pth"
