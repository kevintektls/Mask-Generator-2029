# Autopilot IA++ (Rust)

Rust port of [`scripts/Autopilot_IA++.py`](../scripts/Autopilot_IA++.py) for **Jetson Nano + OAK-D Lite + VESC + Xbox gamepad**.

Hybrid autonomous driving: DepthAI mono camera → lane mask → behavioral-cloning CNN (Candle) → adaptive throttle + steering, with MJPEG debug stream and LB emergency brake.

## Architecture

```
Python camera_bridge.py (depthai 2.29, CAM_B 480p)
    → TCP :9000 gray frames
    → detect_lines (image/imageproc)
    → BehavioralCloningCNN (Candle)
    → VESC (servo + duty)
    ↔ MJPEG :8080
    ↔ Gamepad LB (e-stop)
```

The Rust build **does not compile DepthAI-Core** (too heavy / CMake ≥ 3.20 required on Jetson). Camera uses the existing Python `depthai==2.29.0` stack via [`tools/camera_bridge.py`](tools/camera_bridge.py).

## Jetson prerequisites

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Build deps (no OpenCV / no depthai-sys)
bash scripts/install_jetson_deps.sh

# Python camera (already on your Jetson for Autopilot_IA++.py)
pip install depthai==2.29.0 opencv-python numpy

sudo usermod -aG dialout "$USER"   # VESC serial
```

## Build

```bash
cd autopilot
cargo build --release -p autopilot
```

Fast build — no `depthai-sys`, no `opencv-rust`.

## Run

**Terminal 1** — camera bridge (keep running):

```bash
cd autopilot
python3 tools/camera_bridge.py --fps 60
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
| `connecting to camera bridge` failed | Start `python3 tools/camera_bridge.py` first |
| `libudev` build error | `sudo apt install libudev-dev` |
| VESC permission denied | `sudo usermod -aG dialout $USER`, re-login |
| No gamepad e-stop | Connect Xbox pad; autopilot runs without it |
