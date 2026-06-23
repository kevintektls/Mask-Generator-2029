#!/usr/bin/env bash
# Build autopilot on Jetson Nano.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[autopilot] cargo build --release -p autopilot"
cargo build --release -p autopilot "$@"
echo "[autopilot] Done: target/release/autopilot"
echo "[autopilot] Run:"
echo "  python3 tools/camera_bridge.py &"
echo "  ./target/release/autopilot --model ../model/pilot_model.pth"
