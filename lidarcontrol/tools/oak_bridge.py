#!/usr/bin/env python3
"""Synchronized OAK-D Lite stereo-pair bridge for lidarcontrol (DepthAI v2.29)."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
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
SYNC_THRESHOLD_MS = 5.0
latest_frame = {
    "jpeg": None,
    "timestamp_unix_ns": 0,
    "sequence": 0,
    "left_right_delta_ms": None,
}
latest_lock = threading.Lock()
frame_condition = threading.Condition(latest_lock)


class PreviewHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/status":
            with latest_lock:
                payload = json.dumps(
                    {
                        "ready": latest_frame["jpeg"] is not None,
                        "sequence": latest_frame["sequence"],
                        "left_right_delta_ms": latest_frame["left_right_delta_ms"],
                        "sync_threshold_ms": SYNC_THRESHOLD_MS,
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
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
        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            sequence = -1
            try:
                while True:
                    with frame_condition:
                        frame_condition.wait_for(
                            lambda: latest_frame["jpeg"] is not None
                            and latest_frame["sequence"] != sequence,
                            timeout=5,
                        )
                        jpeg = latest_frame["jpeg"]
                        current_sequence = latest_frame["sequence"]
                    if jpeg is None:
                        continue
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                        + jpeg
                        + b"\r\n"
                    )
                    sequence = current_sequence
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                return
        if self.path != "/":
            self.send_error(404)
            return
        page = """<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>
<title>OAK-D Lite stereo preview</title><style>body{background:#111820;color:#e8eef2;font:16px system-ui;margin:2rem}img{width:min(98vw,1200px);background:#222;border-radius:8px}</style>
<h2>OAK-D Lite · CAM_B gauche | CAM_C droite</h2><img id=frame src='/stream.mjpg'><p id=status>En attente d’une paire synchronisée…</p>
<script>const s=document.querySelector('#status');async function poll(){try{const d=await (await fetch('/status',{cache:'no-store'})).json();if(d.ready)s.textContent=`Paire synchronisée · séquence ${d.sequence} · écart ${Number(d.left_right_delta_ms).toFixed(3)} ms (seuil ${d.sync_threshold_ms} ms)`}catch(e){s.textContent='Flux caméra déconnecté'}}setInterval(poll,500);poll()</script>""".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)


def build_pipeline(fps: int) -> dai.Pipeline:
    pipeline = dai.Pipeline()
    left = pipeline.create(dai.node.MonoCamera)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    left.setFps(fps)

    right = pipeline.create(dai.node.MonoCamera)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    right.setFps(fps)

    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(milliseconds=SYNC_THRESHOLD_MS))
    sync.setSyncAttempts(-1)
    left.out.link(sync.inputs["left"])
    right.out.link(sync.inputs["right"])

    output = pipeline.create(dai.node.XLinkOut)
    output.setStreamName("stereo")
    output.input.setBlocking(False)
    output.input.setQueueSize(2)
    sync.out.link(output.input)
    return pipeline


def main() -> None:
    global SYNC_THRESHOLD_MS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--sync-threshold-ms", type=float, default=SYNC_THRESHOLD_MS)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--preview-host", default="0.0.0.0")
    parser.add_argument("--preview-port", type=int, default=9011)
    args = parser.parse_args()
    if not 1 <= args.fps <= 24:
        parser.error("--fps must be in [1, 24]")
    if not 0.5 <= args.sync_threshold_ms <= 20.0:
        parser.error("--sync-threshold-ms must be in [0.5, 20]")
    if not 30 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [30, 100]")
    SYNC_THRESHOLD_MS = args.sync_threshold_ms

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
        queue = device.getOutputQueue(name="stereo", maxSize=2, blocking=False)
        print(
            f"[oak_bridge] synchronized CAM_B/CAM_C 640x480 @ {args.fps} fps; "
            f"threshold={SYNC_THRESHOLD_MS:g} ms"
        )
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

            left_packet = packet["left"]
            right_packet = packet["right"]
            left_timestamp = left_packet.getTimestampDevice()
            right_timestamp = right_packet.getTimestampDevice()
            delta_ms = (right_timestamp - left_timestamp).total_seconds() * 1000.0
            if abs(delta_ms) > SYNC_THRESHOLD_MS:
                continue

            left_frame = left_packet.getCvFrame()
            right_frame = right_packet.getCvFrame()
            stereo_frame = cv2.hconcat((left_frame, right_frame))
            timestamp_unix_ns = time.time_ns()
            ok, encoded = cv2.imencode(
                ".jpg", stereo_frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            )
            if not ok:
                continue
            height, width = stereo_frame.shape[:2]
            payload = encoded.tobytes()
            with frame_condition:
                latest_frame["jpeg"] = payload
                latest_frame["timestamp_unix_ns"] = timestamp_unix_ns
                latest_frame["sequence"] = latest_frame.get("sequence", 0) + 1
                latest_frame["left_right_delta_ms"] = delta_ms
                frame_condition.notify_all()
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
