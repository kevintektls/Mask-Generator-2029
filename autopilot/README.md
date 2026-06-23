# Autopilot IA++ (Rust)

Rust port of [`scripts/Autopilot_IA++.py`](../scripts/Autopilot_IA++.py) for **Jetson Nano + OAK-D Lite + VESC + Xbox gamepad**.

Hybrid autonomous driving: DepthAI mono camera → lane mask → behavioral-cloning CNN (Candle) → adaptive throttle + steering, with MJPEG debug stream and LB emergency brake.

## Architecture

```
Python camera_bridge.py (depthai stereo + vision_preprocess OpenCV)
    → TCP :9000 160x120 masks
    → BehavioralCloningCNN (Candle) — vision algo runs in Python
    → VESC (servo + duty)
    ↔ MJPEG :8080
    ↔ Gamepad LB (e-stop)
```

The Rust build **does not compile DepthAI-Core**. Camera + lane mask use the same Python stack as [`scripts/Autopilot_IA++.py`](../scripts/Autopilot_IA++.py) via [`tools/camera_bridge.py`](tools/camera_bridge.py) (`make_mask_stereo` from `vision_preprocess.py`).

## Jetson prerequisites

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Build deps (no OpenCV / no depthai-sys)
bash scripts/install_jetson_deps.sh

pip install depthai==2.29.0 opencv-python numpy

sudo usermod -aG dialout "$USER"   # VESC serial
```

## Build

```bash
cd autopilot
cargo build --release -p autopilot
```

Fast build — no `depthai-sys`, no `opencv-rust`.

### Cross-compile from Mac (optional)

`cargo build --target aarch64-unknown-linux-gnu` fails on macOS because `serialport` / `gilrs` need **Linux `libudev`** and host `pkg-config` cannot cross-resolve it.

**Recommended** — Docker + [cross](https://github.com/cross-rs/cross):

```bash
# Install Docker Desktop, then:
cargo install cross --git https://github.com/cross-rs/cross
cd autopilot
bash scripts/build_jetson_cross.sh
scp target/aarch64-unknown-linux-gnu/release/autopilot robotcar@<jetson-ip>:~/Mask-Generator-2029/autopilot/target/release/
```

**Without Docker** — sync a minimal sysroot from the Jetson:

```bash
JETSON=robotcar@<jetson-ip> bash scripts/sync_jetson_sysroot.sh
export JETSON_SYSROOT=$HOME/jetson-sysroot
bash scripts/build_jetson_cross.sh
```

**Simplest** — build directly on the Jetson: `bash scripts/build_jetson.sh`

## Run

**Terminal 1** — camera bridge (keep running):

```bash
cd autopilot
python3 tools/camera_bridge.py --fps 30
```

**Terminal 2** — autopilot:

```bash
cd autopilot
./target/release/autopilot --model ../model/pilot_model.pth
```

Open `http://<jetson-ip>:8080` for the live mask stream. Hold **LB** for emergency brake.

CLI options:

```
--model <path>         Weights (default: ../model/pilot_model.pth)
--camera-addr <host:port>  Bridge TCP (default: 127.0.0.1:9000)
--vesc-port <path>     Serial port (default: /dev/ttyACM0)
--stream-port <port>   MJPEG HTTP (default: 8080)
--cam-fps <fps>        Passed to camera_bridge.py (default: 60)
```

## Validate inference (Python vs Rust)

```bash
cargo build --release -p autopilot-model --bin autopilot-verify
python3 tools/verify_model.py
```

## Troubleshooting

| Issue | Fix |
|---|---|
| `depthai-sys` / `CMakeLists.txt` not found in `target/dai-build` | `bash scripts/fix_jetson_depthai_build.sh` (or `git pull`, `unset DEPTHAI_CORE_ROOT`, `rm -rf target/dai-build`, `cargo clean`) |
| Build still mentions `depthai-sys` | Code not updated; `git pull origin dev`, then `cargo clean` |
| `connecting to camera bridge` failed | Start `python3 tools/camera_bridge.py` first |
| `depthai not installed` | `pip install depthai==2.29.0` |
| `libudev` build error | `sudo apt install libudev-dev` (Jetson) |
| Mac cross-compile `pkg-config has not been configured to support cross-compilation` | Use `bash scripts/build_jetson_cross.sh` (Docker + `cross`) or `sync_jetson_sysroot.sh` — see README |
| VESC permission denied | `sudo usermod -aG dialout $USER`, re-login |
| No gamepad e-stop | Connect Xbox pad; autopilot runs without it |
