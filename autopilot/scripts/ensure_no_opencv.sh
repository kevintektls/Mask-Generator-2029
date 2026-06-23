#!/usr/bin/env bash
# Fail fast if the tree still pulls native opencv/depthai (use image/imageproc + camera_bridge.py).
set -euo pipefail

if rg -n "opencv|depthai" Cargo.toml crates/*/Cargo.toml autopilot/Cargo.toml 2>/dev/null; then
  echo
  echo "[autopilot] ERROR: opencv/depthai crate still referenced — run 'git pull' for the latest code."
  exit 1
fi

if cargo tree -p autopilot 2>/dev/null | rg -q "opencv v|depthai"; then
  echo "[autopilot] ERROR: cargo tree still resolves opencv/depthai. Run:"
  echo "  cargo clean"
  echo "  cargo build --release -p autopilot"
  exit 1
fi

echo "[autopilot] OK: no opencv/depthai native deps"
