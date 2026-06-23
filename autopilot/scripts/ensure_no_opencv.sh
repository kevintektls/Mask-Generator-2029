#!/usr/bin/env bash
# Fail fast if the tree still depends on opencv-rust (removed in favor of image/imageproc).
set -euo pipefail

if rg -n "opencv" Cargo.toml crates/*/Cargo.toml autopilot/Cargo.toml 2>/dev/null; then
  echo
  echo "[autopilot] ERROR: opencv crate still referenced — run 'git pull' for the latest code."
  exit 1
fi

if cargo tree -p autopilot 2>/dev/null | rg -q "opencv v"; then
  echo "[autopilot] ERROR: cargo tree still resolves opencv. Run:"
  echo "  cargo clean"
  echo "  cargo build --release -p autopilot"
  exit 1
fi

echo "[autopilot] OK: no opencv dependency"
