# Autopilot IA++ (Rust)

Rust port of [`scripts/Autopilot_IA++.py`](../scripts/Autopilot_IA++.py) for **Jetson Nano + OAK-D Lite + VESC + Xbox gamepad**.

Hybrid autonomous driving: DepthAI mono camera → lane mask → behavioral-cloning CNN (Candle) → adaptive throttle + steering, with MJPEG debug stream and LB emergency brake.

## Architecture

```
OAK-D CAM_B (640×480 gray)
    → detect_lines (OpenCV)
    → BehavioralCloningCNN (Candle, pilot_model.pth)
    → VESC (servo + duty)
    ↔ MJPEG :8080
    ↔ Gamepad LB (e-stop)
```

Workspace crates:

| Crate | Role |
|---|---|
| `autopilot-config` | Constants + CLI |
| `autopilot-vision` | `detect_lines`, mask resize |
| `autopilot-model` | CNN inference (`.pth` via Candle) |
| `autopilot-vesc` | Serial VESC (duty, brake, servo) |
| `autopilot-input` | Gamepad LB monitor (gilrs) |
| `autopilot-camera` | DepthAI Core v3 mono pipeline |
| `autopilot-stream` | Axum MJPEG server |
| `autopilot` | Main binary |

## Jetson prerequisites

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# System deps (gamepad/evdev + DepthAI build toolchain; OpenCV not required for Rust build)
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git pkg-config \
  clang libclang-dev \
  libudev-dev libssl-dev \
  fonts-dejavu-core

# Or use the helper script:
# bash scripts/install_jetson_deps.sh

# Serial access
sudo usermod -aG dialout "$USER"
# re-login after usermod

# DepthAI Core v3 is built automatically by depthai-sys on first compile (long build).
```

## Build

On Jetson, use the helper script (installs correct env + cleans bad depthai checkout):

```bash
cd autopilot
bash scripts/install_jetson_deps.sh   # once
bash scripts/build_jetson.sh
```

Or manually:

```bash
unset DEPTHAI_CORE_ROOT   # must NOT point to target/dai-build/v3.6.1 (parent folder)
rm -rf target/dai-build   # if a previous depthai-core clone failed
cargo build --release -p autopilot
```

`depthai-sys` clones and compiles **DepthAI-Core v3.6.1** on first build (30–60+ min on Nano, needs ~4 GB swap recommended). Vision uses pure Rust (`image`/`imageproc`) — no `opencv-rust`.

First build can take **30–60+ minutes** on Jetson Nano while `depthai-sys` compiles DepthAI-Core. Subsequent builds reuse `target/dai-build/`.

Release profile uses LTO (`Cargo.toml` workspace `[profile.release]`).

## Run

```bash
cd autopilot
./target/release/autopilot --model ../model/pilot_model.pth

# RL-trained weights (same architecture):
./target/release/autopilot --model ../model/pilot_model_rl.pth
```

CLI options:

```
--model <path>       Weights file (default: ../model/pilot_model.pth)
--vesc-port <path>   Serial port (default: /dev/ttyACM0)
--stream-port <port> MJPEG HTTP port (default: 8080)
--cam-fps <fps>      Camera FPS (default: 60)
```

Open `http://<jetson-ip>:8080` in a browser for the live mask overlay.

**Emergency stop:** hold **LB** on the Xbox-style gamepad.

## Validate inference (Python vs Rust)

On a machine with PyTorch:

```bash
cd autopilot
cargo build --release -p autopilot-model --bin autopilot-verify
python3 tools/verify_model.py
```

Expect `Delta < 1e-4` between Python and Rust predictions.

## Hardware smoke test

1. OAK-D Lite connected; stream visible at `:8080`
2. Car on stand: wheels respond to mask / steering
3. LB triggers brake and exits cleanly
4. Ctrl+C or exit → duty 0, brake 10 A, servo centered

## DepthAI v2 → v3 note

The Python script uses `depthai==2.29.0`. This Rust project uses **DepthAI-Core v3** via the [`depthai`](https://crates.io/crates/depthai) crate. The mono CAM_B 480p pipeline is equivalent but requires Core v3 on the Jetson. Keep the Python script as fallback during migration.

## Troubleshooting

| Issue | Fix |
|---|---|
| `CMakeLists.txt` not found in `target/dai-build/v3.6.1` | `unset DEPTHAI_CORE_ROOT`, `rm -rf target/dai-build`, run `bash scripts/build_jetson.sh` |
| depthai-core build OOM on Nano | Add 4G swap; `export CMAKE_BUILD_PARALLEL_LEVEL=2` |
| `libudev` / `libudev-sys` build error | `bash scripts/install_jetson_deps.sh` |
| `Permission denied` on `/dev/ttyACM0` | `sudo usermod -aG dialout $USER`, re-login |
| DepthAI build fails | Ensure `cmake`, `git`, enough disk in `target/dai-build/` |
| Old `opencv` build errors | `git pull` — vision no longer uses `opencv-rust` |
| No gamepad | Autopilot runs without LB e-stop; connect Xbox-compatible pad |
