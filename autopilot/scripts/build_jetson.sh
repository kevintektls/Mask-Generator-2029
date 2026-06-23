#!/usr/bin/env bash
# Build autopilot on Jetson Nano (handles depthai-sys / depthai-core CMake setup).
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[autopilot] Jetson release build"

# A bad DEPTHAI_CORE_ROOT (parent folder without CMakeLists.txt) causes the exact CMake error you saw.
if [[ -n "${DEPTHAI_CORE_ROOT:-}" ]]; then
  echo "[autopilot] WARNING: DEPTHAI_CORE_ROOT=$DEPTHAI_CORE_ROOT"
  if [[ ! -f "${DEPTHAI_CORE_ROOT}/CMakeLists.txt" ]]; then
    echo "[autopilot] Unsetting invalid DEPTHAI_CORE_ROOT"
    unset DEPTHAI_CORE_ROOT
  fi
fi

DAI_BUILD="target/dai-build/v3.6.1"
if [[ -d "$DAI_BUILD" ]] && [[ ! -f "$DAI_BUILD/depthai-core/CMakeLists.txt" ]]; then
  echo "[autopilot] Removing corrupted depthai-core checkout in $DAI_BUILD"
  rm -rf target/dai-build
fi

# Help CMake find system OpenCV on Jetson (4.1.x in /usr)
for candidate in \
  /usr/lib/aarch64-linux-gnu/cmake/opencv4 \
  /usr/lib/cmake/opencv4; do
  if [[ -d "$candidate" ]]; then
    export OpenCV_DIR="$candidate"
    echo "[autopilot] OpenCV_DIR=$OpenCV_DIR"
    break
  fi
done

export DEPTHAI_SYS_LINK_SHARED=1
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"

echo "[autopilot] Building (depthai-core first compile can take 30-60+ min on Nano)..."
cargo build --release -p autopilot "$@"

echo "[autopilot] Done: target/release/autopilot"
