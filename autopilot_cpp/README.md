# Autopilot IA++ (C++)

C++ port of [`scripts/Autopilot_IA++.py`](../scripts/Autopilot_IA++.py) for **Jetson Nano + OAK-D Lite + VESC + Xbox gamepad**.

Designed for **fast builds on Jetson** (~1–3 min clean vs ~1h Rust): single CMake target, system OpenCV/SDL2, prebuilt ONNX Runtime — no `depthai-sys`, no Candle, no tokio.

## Architecture

```
Python camera_bridge.py (depthai 2.29, CAM_B 480p)
    → TCP :9000 gray frames
    → OpenCV vision (detect_lines)
    → ONNX Runtime (BehavioralCloningCNN)
    → VESC serial (duty / brake / servo)
    ↔ MJPEG :8080
    ↔ SDL2 gamepad LB e-stop
```

Reuses [`autopilot/tools/camera_bridge.py`](../autopilot/tools/camera_bridge.py) — start it before the C++ binary.

## Jetson setup (one-time)

```bash
# Build tools
bash autopilot_cpp/scripts/install_jetson_deps.sh

# Export ONNX weights (needs torch + onnx, run once)
pip install torch onnx onnxruntime
python3 autopilot_cpp/tools/export_onnx.py   # → model/pilot_model.onnx

# VESC serial access
sudo usermod -aG dialout "$USER"   # re-login after
```

## Build

```bash
cd autopilot_cpp
export ONNXRUNTIME_ROOT=/opt/onnxruntime
bash scripts/build_jetson.sh
```

Incremental rebuilds are typically **10–30 seconds**.

## Run

**Terminal 1** — camera bridge:

```bash
cd autopilot
python3 tools/camera_bridge.py --fps 60
```

**Terminal 2** — autopilot:

```bash
cd autopilot_cpp
./build/autopilot_cpp --model ../model/pilot_model.onnx
```

Open `http://<jetson-ip>:8080` for the live mask stream. Hold **LB** for emergency brake.

### CLI options

```
--model <path>              ONNX weights (default: ../model/pilot_model.onnx)
--vesc-port <path>          Serial port (default: /dev/ttyACM0)
--stream-port <port>        MJPEG HTTP (default: 8080)
--camera-addr <host:port>   Bridge TCP (default: 127.0.0.1:9000)
--cam-fps <fps>             Hint for camera_bridge.py (default: 60)
```

## Troubleshooting

| Issue | Fix |
|---|---|
| `ONNXRUNTIME_ROOT not set` | Run `install_jetson_deps.sh` or set `export ONNXRUNTIME_ROOT=/opt/onnxruntime` |
| `connecting to camera bridge` failed | Start `python3 autopilot/tools/camera_bridge.py` first |
| Model file missing | Run `python3 tools/export_onnx.py` |
| VESC permission denied | `sudo usermod -aG dialout $USER`, re-login |
| No gamepad e-stop | Connect Xbox pad; autopilot runs without it |

## Project layout

```
autopilot_cpp/
├── CMakeLists.txt
├── include/autopilot/   # headers
├── src/                 # implementation
├── third_party/httplib.h
├── tools/export_onnx.py
└── scripts/
```

The Rust [`autopilot/`](../autopilot/) project remains available as an alternative.
