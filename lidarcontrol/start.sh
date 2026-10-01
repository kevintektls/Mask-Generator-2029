#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(dirname -- "$SCRIPT_DIR")
cd "$REPO_ROOT"
if ! command -v cargo >/dev/null 2>&1 && [ -f "${HOME:-}/.cargo/env" ]; then
  # SSH non-interactive shells often skip the user's Cargo PATH setup.
  . "$HOME/.cargo/env"
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "Cargo introuvable sur la Jetson. Installe Rust pour l'utilisateur $(id -un) ou ajoute ~/.cargo/bin au PATH." >&2
  exit 127
fi
exec cargo run --release --manifest-path "$SCRIPT_DIR/Cargo.toml" -- start "$@"
