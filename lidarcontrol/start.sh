#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(dirname -- "$SCRIPT_DIR")
cd "$REPO_ROOT"
exec cargo run --release --manifest-path "$SCRIPT_DIR/Cargo.toml" -- start "$@"
