#!/usr/bin/env python3
"""OAK-D Lite mono-frame bridge for lidarcontrol (DepthAI v2.29)."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import struct
import threading
import time

print("[oak_bridge] loading OpenCV (cv2)...", flush=True)
import cv2
print("[oak_bridge] loading DepthAI...", flush=True)
import depthai as dai


MAGIC = b"OAK1"
HEADER = struct.Struct("<4sQHHI")
latest_frame = {"jpeg": None, "timestamp_unix_ns": 0}
latest_lock = threading.Lock()


class PreviewHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/frame.jpg":
            with latest_lock:
                jpeg = latest_frame["jpeg"]
            if jpeg is None:
                self.send_error(503, "camera frame not available yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)
            return
        if self.path != "/":
            self.send_error(404)
            return
        page = """<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>
<title>OAK-D Lite preview</title><style>body{background:#111820;color:#e8eef2;font:16px system-ui;margin:2rem}img{width:min(94vw,800px);background:#222;border-radius:8px}</style>
<h2>OAK-D Lite · CAM_B mono</h2><img id=frame><p id=status>Waiting for camera…</p>
<script>const f=document.querySelector('#frame'),s=document.querySelector('#status');function poll(){f.src='/frame.jpg?t='+Date.now();f.onload=()=>s.textContent='Live · '+new Date().toLocaleTimeString();f.onerror=()=>s.textContent='Waiting for camera frame…'}setInterval(poll,150);poll()</script>""".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)


def build_pipeline(fps: int) -> dai.Pipeline:
    pipeline = dai.Pipeline()
    camera = pipeline.create(dai.node.MonoCamera)
    camera.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    camera.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    camera.setFps(fps)

    output = pipeline.create(dai.node.XLinkOut)
    output.setStreamName("mono")
    output.input.setBlocking(False)
    output.input.setQueueSize(2)
    camera.out.link(output.input)
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--preview-host", default="0.0.0.0")
    parser.add_argument("--preview-port", type=int, default=9011)
    args = parser.parse_args()
    if not 1 <= args.fps <= 60:
        parser.error("--fps must be in [1, 60]")
    if not 30 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [30, 100]")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    server.setblocking(False)
    preview = ThreadingHTTPServer((args.preview_host, args.preview_port), PreviewHandler)
    threading.Thread(target=preview.serve_forever, daemon=True).start()
    print(f"[oak_bridge] camera preview at http://<jetson-ip>:{args.preview_port}/")
    print(f"[oak_bridge] waiting for optional Rust recorder on {args.host}:{args.port}")

    pipeline = build_pipeline(args.fps)
    conn = None
    with dai.Device(pipeline) as device:
        queue = device.getOutputQueue(name="mono", maxSize=2, blocking=False)
        print(f"[oak_bridge] CAM_B mono 640x480 @ {args.fps} fps")
        while True:
            if conn is None:
                try:
                    conn, address = server.accept()
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    print(f"[oak_bridge] Rust recorder connected from {address}")
                except BlockingIOError:
                    pass

            packet = queue.tryGet()
            if packet is None:
                time.sleep(0.002)
                continue

            frame = packet.getCvFrame()
            timestamp_unix_ns = time.time_ns()
            ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            )
            if not ok:
                continue
            height, width = frame.shape[:2]
            payload = encoded.tobytes()
            with latest_lock:
                latest_frame["jpeg"] = payload
                latest_frame["timestamp_unix_ns"] = timestamp_unix_ns
            if conn is not None:
                try:
                    conn.sendall(HEADER.pack(MAGIC, timestamp_unix_ns, width, height, len(payload)))
                    conn.sendall(payload)
                except OSError:
                    print("[oak_bridge] Rust recorder disconnected; preview continues")
                    conn.close()
                    conn = None


if __name__ == "__main__":
    try:
        main()
    except (BrokenPipeError, ConnectionResetError, OSError) as error:
        print(f"[oak_bridge] connection closed: {error}")
