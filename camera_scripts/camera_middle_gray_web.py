#!/usr/bin/env python3
"""
OAK-D Lite Camera Viewer  —  DepthAI v2.29 API with MJPEG Streaming
Streams the center (RGB) camera in grayscale to a local HTTP server.

Requirements:
    pip install depthai==2.29.0 opencv-python numpy

Usage:
    python oak_d_lite_center_gray.py [FPS]
    python oak_d_lite_center_gray.py 30

Controls:
    Ctrl+C   - Quit/Stop the server safely
    Browser  - Open http://localhost:8080 or http://<your-ip>:8080
"""

import http.server
import socketserver
import sys
import threading
import time
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
PREVIEW_SIZE = (640, 480)  # (width, height) for the displayed stream
FPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
PORT = 8080


# ── MJPEG Stream Server State ──────────────────────────────────────────────────
_latest_frame: bytes = b""
_frame_lock = threading.Lock()


def _encode_frame(frame: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


def _update_stream(frame: np.ndarray):
    global _latest_frame
    with _frame_lock:
        _latest_frame = _encode_frame(frame)


class MJPEGHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Silence noisy request logs in the terminal

    def do_GET(self):
        path = self.path.strip("/").lower()

        if path == "stream":
            # Raw MJPEG stream endpoint
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            try:
                while True:
                    with _frame_lock:
                        frame = _latest_frame
                    if frame:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                            + frame
                            + b"\r\n"
                        )
                    time.sleep(1 / FPS)
            except (ConnectionResetError, BrokenPipeError):
                pass  # Client disconnected gracefully
            except Exception as e:
                print(f"Stream error: {e}")

        else:
            # Simple HTML page with the single stream
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>OAK-D Lite — Center Camera</title>
    <style>
        body {{
            margin: 0;
            background: #111;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            font-family: monospace;
            color: #ccc;
        }}
        h2 {{ margin-bottom: 12px; letter-spacing: 2px; }}
        img {{
            border: 2px solid #333;
            max-width: 100%;
        }}
        p {{ margin-top: 10px; font-size: 0.8em; color: #555; }}
    </style>
</head>
<body>
    <h2>OAK-D Lite — Center Camera (Grayscale)</h2>
    <img src="/stream" alt="Center Camera Stream">
    <p>Direct stream: <a href="/stream" style="color:#888;">/stream</a> &nbsp;|&nbsp; {FPS} FPS target</p>
</body>
</html>"""
            self.wfile.write(html.encode())


def start_mjpeg_server(port: int = PORT):
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    server = socketserver.ThreadingTCPServer(("0.0.0.0", port), MJPEGHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"\n🚀 MJPEG stream launched!")
    print(f"👉 Dashboard:    http://localhost:{port}/")
    print(f"👉 Direct stream: http://localhost:{port}/stream\n")
    return server


# ── Pipeline ───────────────────────────────────────────────────────────────────


def build_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()

    # Center RGB camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setPreviewSize(*PREVIEW_SIZE)
    cam_rgb.setFps(FPS)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # XLinkOut — use the low-resolution preview output for efficiency
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("center")
    xout.input.setBlocking(False)
    xout.input.setQueueSize(4)
    cam_rgb.preview.link(xout.input)

    return pipeline


# ── Helpers ────────────────────────────────────────────────────────────────────


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(
        frame,
        f"{fps:.1f} FPS",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (200, 200, 200),
        2,
    )
    return frame


def resize_to_preview(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if (w, h) != PREVIEW_SIZE:
        frame = cv2.resize(frame, PREVIEW_SIZE)
    return frame


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    print(f"Building DepthAI v2.29 pipeline (center camera, {FPS} FPS) …")
    pipeline = build_pipeline()

    fps_count, fps_t, fps_val = 0, time.monotonic(), 0.0

    def tick() -> float:
        nonlocal fps_count, fps_t, fps_val
        fps_count += 1
        now = time.monotonic()
        if now - fps_t >= 1.0:
            fps_val = fps_count / (now - fps_t)
            fps_count = 0
            fps_t = now
        return fps_val

    start_mjpeg_server(port=PORT)

    print("Starting pipeline …")
    try:
        with dai.Device(pipeline) as device:
            q = device.getOutputQueue(name="center", maxSize=4, blocking=False)
            print("Connected! Streaming online. Press Ctrl+C in terminal to stop.")

            while True:
                pkt = q.tryGet()
                if pkt is not None:
                    frame = pkt.getCvFrame()  # BGR from the color cam
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # → grayscale
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # 3-ch for JPEG
                    frame = resize_to_preview(frame)
                    frame = draw_fps(frame, tick())
                    _update_stream(frame)

                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nShutting down gracefully …")


if __name__ == "__main__":
    main()
