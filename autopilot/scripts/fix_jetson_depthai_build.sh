#!/usr/bin/env bash
# One-shot fix for depthai-sys / CMake failures on Jetson.
# Migrates to the Python camera_bridge.py architecture (no native depthai-sys).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
AUTOPILOT_DIR="$REPO_ROOT/autopilot"

echo "[autopilot] Fixing depthai-sys CMake build failure"
echo "[autopilot] Repo: $REPO_ROOT"

cd "$REPO_ROOT"

if [[ -d .git ]]; then
  echo "[autopilot] Syncing latest dev branch..."
  git fetch origin
  git checkout dev
  git pull origin dev
else
  echo "[autopilot] WARNING: not a git repo — ensure you have commit 1f222ce+ (no depthai in Cargo.toml)"
fi

if rg -n 'depthai' "$AUTOPILOT_DIR/Cargo.toml" "$AUTOPILOT_DIR"/crates/*/Cargo.toml 2>/dev/null; then
  echo "[autopilot] ERROR: depthai still in Cargo.toml — git pull may have failed."
  exit 1
fi

cd "$AUTOPILOT_DIR"

echo "[autopilot] Cleaning stale DepthAI-Core build artifacts..."
if [[ -n "${DEPTHAI_CORE_ROOT:-}" ]]; then
  echo "[autopilot] Unsetting DEPTHAI_CORE_ROOT=$DEPTHAI_CORE_ROOT"
  unset DEPTHAI_CORE_ROOT
fi
rm -rf target/dai-build
cargo clean

echo "[autopilot] Installing system build dependencies..."
bash scripts/install_jetson_deps.sh

echo "[autopilot] Installing Python camera bridge dependencies..."
if command -v pip3 >/dev/null 2>&1; then
  pip3 install --user depthai==2.29.0 opencv-python numpy
elif command -v pip >/dev/null 2>&1; then
  pip install --user depthai==2.29.0 opencv-python numpy
else
  echo "[autopilot] WARNING: pip not found — run: pip install depthai==2.29.0 opencv-python numpy"
fi

bash scripts/ensure_no_opencv.sh

echo "[autopilot] Building release binary (no depthai-sys)..."
bash scripts/build_jetson.sh

echo ""
echo "[autopilot] Done. Run in two terminals:"
echo "  cd $AUTOPILOT_DIR"
echo "  python3 tools/camera_bridge.py --fps 30"
echo "  ./target/release/autopilot --model ../model/pilot_model.pth"
