#!/usr/bin/env bash
# Cross-compile autopilot for Jetson Nano from macOS (or any non-aarch64 Linux host).
#
# Preferred: Docker via `cross` (no pkg-config/sysroot setup on the host).
# Fallback: set JETSON_SYSROOT after running scripts/sync_jetson_sysroot.sh
set -euo pipefail

cd "$(dirname "$0")/.."
TARGET=aarch64-unknown-linux-gnu
BIN="target/${TARGET}/release/autopilot"

rustup target add "${TARGET}" 2>/dev/null || true

if command -v cross >/dev/null 2>&1 && command -v docker >/dev/null 2>&1; then
  if ! docker info >/dev/null 2>&1; then
    echo "[autopilot] Docker is installed but not running. Start Docker Desktop, then retry."
    exit 1
  fi
  echo "[autopilot] cross build --target ${TARGET} --release -p autopilot"
  cross build --target "${TARGET}" --release -p autopilot "$@"
  echo "[autopilot] Done: ${BIN}"
  echo "[autopilot] Copy to Jetson:"
  echo "  scp ${BIN} robotcar@<jetson-ip>:~/Mask-Generator-2029/autopilot/target/release/"
  exit 0
fi

if [[ -n "${JETSON_SYSROOT:-}" ]] && [[ -f "${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig/libudev.pc" ]]; then
  export PKG_CONFIG_ALLOW_CROSS=1
  export PKG_CONFIG_SYSROOT_DIR="${JETSON_SYSROOT}"
  export PKG_CONFIG_PATH="${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig:${JETSON_SYSROOT}/usr/share/pkgconfig"
  echo "[autopilot] Using JETSON_SYSROOT=${JETSON_SYSROOT}"
  echo "[autopilot] cargo build --target ${TARGET} --release -p autopilot"
  cargo build --target "${TARGET}" --release -p autopilot "$@"
  echo "[autopilot] Done: ${BIN}"
  exit 0
fi

cat <<'EOF'
[autopilot] Cannot cross-compile: libudev-sys needs Linux headers for aarch64.

Option A — Docker + cross (recommended on Mac):
  brew install --cask docker          # start Docker Desktop
  cargo install cross --git https://github.com/cross-rs/cross
  bash scripts/build_jetson_cross.sh

Option B — sysroot from your Jetson (no Docker):
  export JETSON=robotcar@<jetson-ip>
  bash scripts/sync_jetson_sysroot.sh
  export JETSON_SYSROOT=$HOME/jetson-sysroot
  bash scripts/build_jetson_cross.sh

Option C — build on the Jetson (simplest):
  ssh robotcar@<jetson-ip>
  cd Mask-Generator-2029/autopilot && bash scripts/build_jetson.sh

Why `cargo build --target aarch64-unknown-linux-gnu` fails on Mac:
  serialport/gilrs link libudev. pkg-config on macOS cannot find aarch64 Linux
  libraries without a sysroot or a Linux build container.
EOF
exit 1
