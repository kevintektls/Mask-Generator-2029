#!/usr/bin/env python3
"""
OAK-D Lite Camera Viewer  —  DepthAI v2.29 API with MJPEG Streaming
Streams left mono, right mono, and depth streams to a local HTTP server.

Requirements:
    pip install depthai==2.29.0 opencv-python numpy

Usage:
    python oak_d_lite_mjpeg.py

Controls:
    Ctrl+C   - Quit/Stop the server safely
    Browser  - Open http://localhost:8080 or http://<your-ip>:8080
"""

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
FPS = int(sys.argv[1])
DEPTH_MAX_MM = 8_000  # Initial max depth shown (mm)
SNAPSHOT_DIR = Path("snapshots")
PORT = 8080

COLORMAPS = [cv2.COLORMAP_TURBO, cv2.COLORMAP_BONE, cv2.COLORMAP_HSV]
COLORMAP_NAMES = ["TURBO", "BONE", "HSV"]


# ── MJPEG Stream Server State & Logic ─────────────────────────────────────────
_stream_frames: dict[str, bytes] = {}
_stream_lock = threading.Lock()


def _encode_frame(frame: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


def _update_stream(name: str, frame: np.ndarray):
    with _stream_lock:
        _stream_frames[name] = _encode_frame(frame)


class MJPEGHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Silence noisy request logs in the terminal

    def do_GET(self):
        # Pick stream from URL: /left  /right  /depth  or / for composite
        name = self.path.strip("/").upper() or None
        self.send_response(200)

        if name and name in ("LEFT", "RIGHT", "DEPTH"):
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
        else:
            # Simple HTML grid view showing all three streams simultaneously
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head><title>OAK-D Lite MJPEG Streams</title></head>
            <body style="background:#111; display:flex; gap:8px; flex-wrap:wrap; margin:20px; font-family:sans-serif; color:white;">
                <div style="text-align:center;"><h3 style="margin:4px;">Left Eye</h3><img src="/left" style="height:480px; border:2px solid #333;"></div>
                <div style="text-align:center;"><h3 style="margin:4px;">Right Eye</h3><img src="/right" style="height:480px; border:2px solid #333;"></div>
                <div style="text-align:center;"><h3 style="margin:4px;">Depth Map</h3><img src="/depth" style="height:480px; border:2px solid #333;"></div>
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
    print(f"👉 Direct Streams: http://localhost:{port}/left | /right | /depth\n")
    return server


# ── Pipeline ───────────────────────────────────────────────────────────────────


def build_pipeline():
    pipeline = dai.Pipeline()

    # ── Left mono ──────────────────────────────────────────────────────────
    cam_left = pipeline.create(dai.node.MonoCamera)
    cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    cam_left.setFps(FPS)

    # XLinkOut for left display
    xout_left = pipeline.create(dai.node.XLinkOut)
    xout_left.setStreamName("left")
    xout_left.input.setBlocking(False)
    xout_left.input.setQueueSize(4)
    cam_left.out.link(xout_left.input)

    # ── Right mono ─────────────────────────────────────────────────────────
    cam_right = pipeline.create(dai.node.MonoCamera)
    cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    cam_right.setFps(FPS)

    # XLinkOut for right display
    xout_right = pipeline.create(dai.node.XLinkOut)
    xout_right.setStreamName("right")
    xout_right.input.setBlocking(False)
    xout_right.input.setQueueSize(4)
    cam_right.out.link(xout_right.input)

    # ── Stereo depth ───────────────────────────────────────────────────────
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)
    stereo.setOutputSize(*PREVIEW_SIZE)

    cam_left.out.link(stereo.left)
    cam_right.out.link(stereo.right)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    xout_depth.input.setBlocking(False)
    xout_depth.input.setQueueSize(4)
    stereo.depth.link(xout_depth.input)

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


def resize_to_preview(frame: np.ndarray) -> np.ndarray:
    """Resize a frame to PREVIEW_SIZE if it doesn't already match."""
    h, w = frame.shape[:2]
    if (w, h) != PREVIEW_SIZE:
        frame = cv2.resize(frame, PREVIEW_SIZE)
    return frame


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    print("Building DepthAI v2.29 pipeline …")
    pipeline = build_pipeline()

    colormap_idx = 0
    depth_max = DEPTH_MAX_MM

    fps_data = {
        name: {"count": 0, "t": time.monotonic(), "fps": 0.0}
        for name in ("left", "right", "depth")
    }

    def tick(name: str) -> float:
        d = fps_data[name]
        d["count"] += 1
        now = time.monotonic()
        if now - d["t"] >= 1.0:
            d["fps"] = d["count"] / (now - d["t"])
            d["count"] = 0
            d["t"] = now
        return d["fps"]

    # Spin up the background webserver
    start_mjpeg_server(port=PORT)

    print("Starting pipeline …")

    try:
        with dai.Device(pipeline) as device:
            print("🔌 Vitesse USB détectée :", device.getUsbSpeed())
            q_left = device.getOutputQueue(name="left", maxSize=4, blocking=False)
            q_right = device.getOutputQueue(name="right", maxSize=4, blocking=False)
            q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

            print("Connected! Streaming online. Press Ctrl+C in terminal to stop.")

            latest = {k: None for k in ("left", "right", "depth")}

            while True:
                # Grab latest frames (non-blocking)
                for name, q in (
                    ("left", q_left),
                    ("right", q_right),
                    ("depth", q_depth),
                ):
                    pkt = q.tryGet()
                    if pkt is not None:
                        latest[name] = pkt
                        tick(name)

                if latest["left"] is not None:
                    f = latest["left"].getCvFrame()
                    f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                    f = resize_to_preview(f)
                    f = draw_fps(f, fps_data["left"]["fps"], "LEFT")
                    _update_stream("LEFT", f)

                if latest["right"] is not None:
                    f = latest["right"].getCvFrame()
                    f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                    f = resize_to_preview(f)
                    f = draw_fps(f, fps_data["right"]["fps"], "RIGHT")
                    _update_stream("RIGHT", f)

                if latest["depth"] is not None:
                    raw = latest["depth"].getFrame()
                    f = colorize_depth(raw, COLORMAPS[colormap_idx], depth_max)
                    f = resize_to_preview(f)
                    lbl = f"DEPTH [{COLORMAP_NAMES[colormap_idx]}] max={depth_max // 1000}m"
                    f = draw_fps(f, fps_data["depth"]["fps"], lbl)
                    _update_stream("DEPTH", f)

                # Small sleep to prevent burning high CPU resources in an unthrottled loop
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nShutting down pipeline and server gracefully …")


if __name__ == "__main__":
    main()
