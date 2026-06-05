#!/usr/bin/env python3
"""
OAK-D Lite Camera Viewer  —  DepthAI v2.29 API with MJPEG Streaming
Streams left mono, right mono, and optionally depth to a local HTTP server.

Requirements:
    pip install depthai==2.29.0 opencv-python numpy

Usage:
    python camera_website.py [FPS]
    python camera_website.py 20
    python camera_website.py 20 --depth

Controls:
    Ctrl+C   - Quit/Stop the server safely
    Browser  - Open http://localhost:8080 or http://<your-ip>:8080
"""

import argparse
import http.server
import socketserver
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import depthai as dai
except ImportError:
    print("DepthAI not installed.  Run:  pip install depthai==2.29.0")
    sys.exit(1)

try:
    import cv2
    import numpy as np
except ImportError:
    print("OpenCV / NumPy not installed.  Run:  pip install opencv-python numpy")
    sys.exit(1)


# ── Configuration ──────────────────────────────────────────────────────────────
PREVIEW_SIZE = (640, 480)  # (width, height) for every displayed window
DEPTH_MAX_MM = 8_000  # Initial max depth shown (mm)
SNAPSHOT_DIR = Path("snapshots")
PORT = 8080
RECONNECT_DELAY_S = 5.0
USB2_MAX_FPS = 20
USB2_PREVIEW_SIZE = (640, 480)
USB2_DEPTH_PREVIEW_SIZE = (480, 360)
JPEG_QUALITY = 92

FPS = 20
ENABLE_DEPTH = False
STREAM_SIZE = PREVIEW_SIZE

COLORMAPS = [cv2.COLORMAP_TURBO, cv2.COLORMAP_BONE, cv2.COLORMAP_HSV]
COLORMAP_NAMES = ["TURBO", "BONE", "HSV"]


# ── MJPEG Stream Server State & Logic ─────────────────────────────────────────
_stream_frames: dict[str, bytes] = {}
_stream_lock = threading.Lock()


def _encode_frame(frame: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes()


def _update_stream(name: str, frame: np.ndarray):
    with _stream_lock:
        _stream_frames[name] = _encode_frame(frame)


class MJPEGHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Silence noisy request logs in the terminal

    def do_GET(self):
        name = self.path.strip("/").upper() or None
        allowed = ("LEFT", "RIGHT", "DEPTH") if ENABLE_DEPTH else ("LEFT", "RIGHT")

        if name == "DEPTH" and not ENABLE_DEPTH:
            self.send_error(404)
            return

        if name and name in allowed:
            self.send_response(200)
            # Single MJPEG Stream
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            try:
                while True:
                    with _stream_lock:
                        frame = _stream_frames.get(name)
                    if frame:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                            + frame
                            + b"\r\n"
                        )
                    # Throttles stream output close to the device's delivery rate
                    time.sleep(1 / FPS)
            except (ConnectionResetError, BrokenPipeError):
                pass  # Client disconnected gracefully
            except Exception as e:
                print(f"Stream error: {e}")
        elif name:
            self.send_error(404)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            depth_panel = (
                f'<div style="text-align:center;"><h3 style="margin:4px;">Depth Map</h3>'
                f'<img src="/depth" style="width:{STREAM_SIZE[0]}px; height:{STREAM_SIZE[1]}px; border:2px solid #333;"></div>'
                if ENABLE_DEPTH
                else ""
            )
            html = f"""
            <!DOCTYPE html>
            <html>
            <head><title>OAK-D Lite MJPEG Streams</title></head>
            <body style="background:#111; display:flex; gap:8px; flex-wrap:wrap; margin:20px; font-family:sans-serif; color:white;">
                <div style="text-align:center;"><h3 style="margin:4px;">Left Eye</h3><img src="/left" style="width:{STREAM_SIZE[0]}px; height:{STREAM_SIZE[1]}px; border:2px solid #333; image-rendering:auto;"></div>
                <div style="text-align:center;"><h3 style="margin:4px;">Right Eye</h3><img src="/right" style="width:{STREAM_SIZE[0]}px; height:{STREAM_SIZE[1]}px; border:2px solid #333; image-rendering:auto;"></div>
                {depth_panel}
            </body>
            </html>
            """
            self.wfile.write(html.encode())


def start_mjpeg_server(port: int = 8080):
    # Allow quick port reuse if restarting frequently
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    server = socketserver.ThreadingTCPServer(("0.0.0.0", port), MJPEGHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"\n🚀 MJPEG stream launched successfully!")
    print(f"👉 Main Dashboard:  http://localhost:{port}/")
    streams = f"http://localhost:{port}/left | /right"
    if ENABLE_DEPTH:
        streams += " | /depth"
    print(f"👉 Direct Streams: {streams}\n")
    return server


# ── Pipeline ───────────────────────────────────────────────────────────────────


def _is_usb2(speed) -> bool:
    return speed in (
        dai.UsbSpeed.HIGH,
        dai.UsbSpeed.FULL,
        dai.UsbSpeed.LOW,
    )


def _stream_settings(
    usb2_mode: bool, enable_depth: bool
) -> tuple[tuple[int, int], int]:
    if usb2_mode:
        if enable_depth:
            return USB2_DEPTH_PREVIEW_SIZE, min(FPS, 15)
        return USB2_PREVIEW_SIZE, min(FPS, USB2_MAX_FPS)
    return PREVIEW_SIZE, FPS


NATIVE_MONO_SIZE = (640, 480)


def build_pipeline(
    preview_size: tuple[int, int], stream_fps: int, enable_depth: bool
):
    pipeline = dai.Pipeline()

    cam_left = pipeline.create(dai.node.MonoCamera)
    cam_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    cam_left.setFps(stream_fps)

    cam_right = pipeline.create(dai.node.MonoCamera)
    cam_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    cam_right.setFps(stream_fps)

    def _xlink_out(name: str):
        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName(name)
        xout.input.setBlocking(False)
        xout.input.setQueueSize(2)
        return xout

    def _resize_before_xlink(source, name: str, gray: bool = True):
        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setResize(*preview_size)
        if gray:
            manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
        manip.setMaxOutputFrameSize(preview_size[0] * preview_size[1])
        source.link(manip.inputImage)
        xout = _xlink_out(name)
        manip.out.link(xout.input)

    def _mono_to_xlink(source, name: str):
        if preview_size == NATIVE_MONO_SIZE:
            xout = _xlink_out(name)
            source.link(xout.input)
        else:
            _resize_before_xlink(source, name)

    if enable_depth:
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(False)
        stereo.setExtendedDisparity(False)
        stereo.setOutputSize(*preview_size)

        cam_left.out.link(stereo.left)
        cam_right.out.link(stereo.right)

        _resize_before_xlink(stereo.rectifiedLeft, "left")
        _resize_before_xlink(stereo.rectifiedRight, "right")

        xout_depth = _xlink_out("depth")
        stereo.depth.link(xout_depth.input)
    else:
        _mono_to_xlink(cam_left.out, "left")
        _mono_to_xlink(cam_right.out, "right")

    return pipeline


# ── Helpers ────────────────────────────────────────────────────────────────────


def colorize_depth(depth_frame: np.ndarray, colormap: int, max_mm: int) -> np.ndarray:
    clipped = np.clip(depth_frame, 0, max_mm).astype(np.float32)
    normed = (clipped / max_mm * 255).astype(np.uint8)
    return cv2.applyColorMap(normed, colormap)


def draw_fps(frame: np.ndarray, fps: float, label: str = "") -> np.ndarray:
    text = f"{label}  {fps:.1f} FPS" if label else f"{fps:.1f} FPS"
    cv2.putText(frame, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return frame


def placeholder_frame(
    label: str, message: str, size: tuple[int, int] = PREVIEW_SIZE
) -> np.ndarray:
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.putText(
        frame, label, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )
    cv2.putText(
        frame, message, (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1
    )
    return frame


def show_reconnecting_streams(
    preview_size: tuple[int, int] = PREVIEW_SIZE, enable_depth: bool = False
):
    msg = "Reconnecting to device..."
    names = ["LEFT", "RIGHT"]
    if enable_depth:
        names.append("DEPTH")
    for name in names:
        _update_stream(name, placeholder_frame(name, msg, preview_size))


def safe_try_get(queue):
    try:
        return queue.tryGet()
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Queue read failed: {e}") from e


def stream_from_device(
    device,
    fps_data,
    colormap_idx,
    depth_max,
    preview_size: tuple[int, int],
    enable_depth: bool,
):
    q_left = device.getOutputQueue(name="left", maxSize=2, blocking=False)
    q_right = device.getOutputQueue(name="right", maxSize=2, blocking=False)
    q_depth = (
        device.getOutputQueue(name="depth", maxSize=2, blocking=False)
        if enable_depth
        else None
    )

    latest = {k: None for k in fps_data}

    queues = [("left", q_left), ("right", q_right)]
    if enable_depth:
        queues.append(("depth", q_depth))

    while True:
        for name, q in queues:
            pkt = safe_try_get(q)
            if pkt is not None:
                latest[name] = pkt
                tick_fps(fps_data, name)

        if latest["left"] is not None:
            f = latest["left"].getCvFrame()
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            if f.shape[1::-1] != preview_size:
                f = cv2.resize(f, preview_size)
            f = draw_fps(f, fps_data["left"]["fps"], "LEFT")
            _update_stream("LEFT", f)

        if latest["right"] is not None:
            f = latest["right"].getCvFrame()
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            if f.shape[1::-1] != preview_size:
                f = cv2.resize(f, preview_size)
            f = draw_fps(f, fps_data["right"]["fps"], "RIGHT")
            _update_stream("RIGHT", f)

        if enable_depth and latest["depth"] is not None:
            raw = latest["depth"].getFrame()
            f = colorize_depth(raw, COLORMAPS[colormap_idx], depth_max)
            if f.shape[1::-1] != preview_size:
                f = cv2.resize(f, preview_size)
            lbl = f"DEPTH [{COLORMAP_NAMES[colormap_idx]}] max={depth_max // 1000}m"
            f = draw_fps(f, fps_data["depth"]["fps"], lbl)
            _update_stream("DEPTH", f)

        time.sleep(0.001)


def tick_fps(fps_data, name: str) -> float:
    d = fps_data[name]
    d["count"] += 1
    now = time.monotonic()
    if now - d["t"] >= 1.0:
        d["fps"] = d["count"] / (now - d["t"])
        d["count"] = 0
        d["t"] = now
    return d["fps"]


def connect_and_stream(
    fps_data, colormap_idx, depth_max, enable_depth: bool, usb2_mode: bool = False
):
    global STREAM_SIZE

    preview_size, stream_fps = _stream_settings(usb2_mode, enable_depth)
    STREAM_SIZE = preview_size
    print("Building DepthAI v2.29 pipeline …")
    mode = "left + right + depth" if enable_depth else "left + right"
    print(f"📷 Streams: {mode}")
    print(
        f"🖼️  Preview: {preview_size[0]}x{preview_size[1]} @ {stream_fps} FPS "
        f"(JPEG q={JPEG_QUALITY})"
    )
    if usb2_mode:
        print("⚙️  USB2 bandwidth limit active")
    pipeline = build_pipeline(preview_size, stream_fps, enable_depth)
    print("Starting pipeline …")

    with dai.Device(pipeline) as device:
        speed = device.getUsbSpeed()
        print("🔌 Vitesse USB détectée :", speed)
        if _is_usb2(speed) and not usb2_mode:
            usb2_preview, usb2_fps = _stream_settings(True, enable_depth)
            if usb2_preview != preview_size or usb2_fps != stream_fps:
                print("⚠️  USB2 détecté — redémarrage avec bande passante réduite …")
                return True
            print("⚠️  USB2 détecté — qualité conservée (left/right uniquement)")

        print("Connected! Streaming online. Press Ctrl+C in terminal to stop.")
        stream_from_device(
            device, fps_data, colormap_idx, depth_max, preview_size, enable_depth
        )
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="OAK-D Lite MJPEG camera streams")
    parser.add_argument(
        "fps",
        nargs="?",
        type=int,
        default=20,
        help="Target FPS (default: 20)",
    )
    parser.add_argument(
        "--depth",
        action="store_true",
        help="Enable depth stream (StereoDepth, heavier on USB bandwidth)",
    )
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    global FPS, ENABLE_DEPTH

    args = parse_args()
    FPS = args.fps
    ENABLE_DEPTH = args.depth

    colormap_idx = 0
    depth_max = DEPTH_MAX_MM

    stream_names = ["left", "right"]
    if ENABLE_DEPTH:
        stream_names.append("depth")
    fps_data = {
        name: {"count": 0, "t": time.monotonic(), "fps": 0.0}
        for name in stream_names
    }

    start_mjpeg_server(port=PORT)

    usb2_mode = False
    preview_size = PREVIEW_SIZE

    while True:
        try:
            need_usb2 = connect_and_stream(
                fps_data,
                colormap_idx,
                depth_max,
                ENABLE_DEPTH,
                usb2_mode=usb2_mode,
            )
            if need_usb2:
                usb2_mode = True
                preview_size = _stream_settings(True, ENABLE_DEPTH)[0]
                continue
        except KeyboardInterrupt:
            print("\nShutting down pipeline and server gracefully …")
            break
        except RuntimeError as e:
            print(f"\n⚠️  Device communication error: {e}")
            show_reconnecting_streams(preview_size, ENABLE_DEPTH)
            print(f"Retrying in {RECONNECT_DELAY_S:.0f}s … (Ctrl+C to stop)")
            time.sleep(RECONNECT_DELAY_S)
            usb2_mode = True
            preview_size = _stream_settings(True, ENABLE_DEPTH)[0]
        except Exception as e:
            print(f"\n⚠️  Unexpected device error: {e}")
            show_reconnecting_streams(preview_size, ENABLE_DEPTH)
            print(f"Retrying in {RECONNECT_DELAY_S:.0f}s … (Ctrl+C to stop)")
            time.sleep(RECONNECT_DELAY_S)
            usb2_mode = True
            preview_size = _stream_settings(True, ENABLE_DEPTH)[0]


if __name__ == "__main__":
    main()
