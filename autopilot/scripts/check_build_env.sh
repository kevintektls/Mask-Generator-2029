#!/usr/bin/env bash
# Quick pre-flight check before `cargo build` on Jetson.
set -euo pipefail

ok=true

check() {
  local name="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    echo "[OK]   $name"
  else
    echo "[FAIL] $name"
    ok=false
  fi
}

check "pkg-config" pkg-config --version
check "libudev (pkg-config)" pkg-config --exists libudev
check "cmake" cmake --version
check "git" git --version
check "rustc" rustc --version
check "cargo" cargo --version

if [[ "$ok" != true ]]; then
  echo
  echo "Run: bash scripts/install_jetson_deps.sh"
  exit 1
fi

echo
echo "Environment looks good. Build with:"
echo "  cargo build --release -p autopilot"
