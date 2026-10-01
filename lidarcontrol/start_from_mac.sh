#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
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

if ! command -v cargo >/dev/null 2>&1; then
  echo "Cargo est requis sur le Mac. Installe Rust avec rustup, puis relance." >&2
  exit 2
fi

REMOTE_REPO_PATH="$(printf '%q' "$JETSON_REPO_PATH")"
REMOTE_MIDDLECAM=""
for argument in "$@"; do
  case "$argument" in
    --midlecam) REMOTE_MIDDLECAM=" --midlecam" ;;
    *)
      echo "Option inconnue : $argument (option disponible : --midlecam)." >&2
      exit 2
      ;;
  esac
done
REMOTE_COMMAND="cd $REMOTE_REPO_PATH && ./lidarcontrol/start.sh --remote-control --control-bind 127.0.0.1:$CONTROL_PORT$REMOTE_MIDDLECAM"
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
echo "L'aperçu LiDAR + caméra sera disponible sur http://127.0.0.1:5001/"
ssh -tt \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=3 \
  -o ServerAliveCountMax=2 \
  -L "127.0.0.1:$CONTROL_PORT:127.0.0.1:$CONTROL_PORT" \
  -L 127.0.0.1:5001:127.0.0.1:5001 \
  -L 127.0.0.1:9011:127.0.0.1:9011 \
  "$JETSON_SSH_TARGET" "$REMOTE_COMMAND" </dev/null &
SSH_PID=$!

# SSH can fail immediately when a previous tunnel already owns one of the
# local ports. Do not start the gamepad client against that unrelated tunnel.
for _ in {1..20}; do
  if ! kill -0 "$SSH_PID" 2>/dev/null; then
    wait "$SSH_PID" 2>/dev/null || true
    echo "Le tunnel SSH n'a pas démarré. Vérifie les ports locaux 5010, 5001 et 9011 avec :" >&2
    echo "lsof -nP -iTCP:5010 -iTCP:5001 -iTCP:9011 -sTCP:LISTEN" >&2
    exit 1
  fi
  sleep 0.1
done

cargo run --release --manifest-path "$SCRIPT_DIR/Cargo.toml" -- remote-client --port "$CONTROL_PORT"
