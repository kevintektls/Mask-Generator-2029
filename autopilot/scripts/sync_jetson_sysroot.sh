#!/usr/bin/env bash
# Pull libudev dev files from Jetson into a local sysroot for Mac cross-compilation.
set -euo pipefail

JETSON="${JETSON:-}"
SYSROOT="${JETSON_SYSROOT:-${HOME}/jetson-sysroot}"

if [[ -z "${JETSON}" ]]; then
  echo "Usage: JETSON=robotcar@<jetson-ip> bash scripts/sync_jetson_sysroot.sh"
  echo "Optional: JETSON_SYSROOT=~/jetson-sysroot"
  exit 1
fi

echo "[autopilot] Syncing libudev sysroot from ${JETSON} → ${SYSROOT}"
mkdir -p "${SYSROOT}"

# Package libudev-dev paths from Jetson (small, enough for libudev-sys).
ssh "${JETSON}" 'dpkg -L libudev-dev | tar czf - -T -' | tar xzf - -C "${SYSROOT}"

# Runtime .so (may live outside -dev file list on some images).
ssh "${JETSON}" 'tar czf - usr/lib/aarch64-linux-gnu/libudev.so.1 usr/lib/aarch64-linux-gnu/libudev.so 2>/dev/null || true' \
  | tar xzf - -C "${SYSROOT}" 2>/dev/null || true

if [[ ! -f "${SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig/libudev.pc" ]]; then
  echo "[autopilot] ERROR: libudev.pc not found. On Jetson run: sudo apt install libudev-dev"
  exit 1
fi

echo "[autopilot] Sysroot ready at ${SYSROOT}"
echo "[autopilot] Next:"
echo "  export JETSON_SYSROOT=${SYSROOT}"
echo "  bash scripts/build_jetson_cross.sh"
