#!/usr/bin/env bash
# Start autopilot on Jetson (camera bridge must run first).
set -euo pipefail
cd "$(dirname "$0")/.."

if ! python3 -c "import depthai" 2>/dev/null; then
  echo "[autopilot] depthai not installed. Run: pip3 install --user depthai==2.29.0"
  exit 1
fi

if [[ ! -x target/release/autopilot ]]; then
  echo "[autopilot] Binary missing. Run: bash scripts/build_jetson.sh"
  exit 1
fi

echo "[autopilot] Start camera bridge in another terminal:"
echo "  python3 tools/camera_bridge.py --fps 30"
echo ""
echo "[autopilot] Waiting 3s for bridge (Ctrl+C to abort)..."
sleep 3

exec ./target/release/autopilot --model ../model/pilot_model.pth "$@"
