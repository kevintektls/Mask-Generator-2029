#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
CLIENT_SOURCE="$SCRIPT_DIR/tools/xbox360_remote_client.c"
CLIENT_BINARY="$SCRIPT_DIR/target/xbox360_remote_client"
CONTROL_PORT="${LIDAR_CONTROL_PORT:-5010}"
JETSON_SSH_TARGET="${JETSON_SSH_TARGET:-}"
JETSON_REPO_PATH="${JETSON_REPO_PATH:-/home/robotcar/Mask-Generator-2029}"

if [[ -z "$JETSON_SSH_TARGET" ]]; then
  echo "Définis JETSON_SSH_TARGET avec ta cible SSH, par exemple robotcar@192.168.1.42." >&2
  echo "Tu peux aussi utiliser un alias déjà présent dans ~/.ssh/config." >&2
  exit 2
fi

if [[ ! "$CONTROL_PORT" =~ ^[0-9]+$ ]] || (( CONTROL_PORT < 1 || CONTROL_PORT > 65535 )); then
  echo "LIDAR_CONTROL_PORT doit être un port entre 1 et 65535." >&2
  exit 2
fi

if ! command -v pkg-config >/dev/null || ! pkg-config --exists libusb-1.0; then
  echo "libusb et pkg-config sont requis. Sur Mac : brew install libusb pkg-config" >&2
  exit 2
fi
if ! command -v cc >/dev/null; then
  echo "Un compilateur C est requis. Installe les Xcode Command Line Tools." >&2
  exit 2
fi

mkdir -p "$SCRIPT_DIR/target"
if [[ ! -x "$CLIENT_BINARY" || "$CLIENT_SOURCE" -nt "$CLIENT_BINARY" ]]; then
  # pkg-config supplies the installed libusb include and link flags.
  LIBUSB_PREFIX="$(pkg-config --variable=prefix libusb-1.0)"
  cc -std=c11 -O2 -Wall -Wextra -o "$CLIENT_BINARY" "$CLIENT_SOURCE" \
    -Wl,-rpath,"$LIBUSB_PREFIX/lib" \
    $(pkg-config --cflags --libs libusb-1.0)
fi

REMOTE_REPO_PATH="$(printf '%q' "$JETSON_REPO_PATH")"
REMOTE_COMMAND="cd $REMOTE_REPO_PATH && ./lidarcontrol/start.sh --remote-control --control-bind 127.0.0.1:$CONTROL_PORT"
SSH_PID=""
cleanup() {
  if [[ -n "$SSH_PID" ]] && kill -0 "$SSH_PID" 2>/dev/null; then
    for _ in {1..30}; do
      kill -0 "$SSH_PID" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$SSH_PID" 2>/dev/null; then
      kill -TERM "$SSH_PID" 2>/dev/null || true
    fi
    wait "$SSH_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Démarrage lidarcontrol sur la Jetson via SSH…"
ssh -tt \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=3 \
  -o ServerAliveCountMax=2 \
  -L "127.0.0.1:$CONTROL_PORT:127.0.0.1:$CONTROL_PORT" \
  -L 127.0.0.1:5001:127.0.0.1:5001 \
  -L 127.0.0.1:9011:127.0.0.1:9011 \
  "$JETSON_SSH_TARGET" "$REMOTE_COMMAND" </dev/null &
SSH_PID=$!

echo "L'aperçu LiDAR + caméra sera disponible sur http://127.0.0.1:5001/"
"$CLIENT_BINARY" "$CONTROL_PORT"
