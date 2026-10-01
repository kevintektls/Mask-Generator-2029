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
MIDDLE_CAMERA_ENABLED = False
latest_frame = {
    "jpeg": None,
    "left_jpeg": None,
    "middle_jpeg": None,
    "right_jpeg": None,
    "timestamp_unix_ns": 0,
    "sequence": 0,
    "left_right_delta_ms": None,
    "middle_delta_ms": None,
}
latest_lock = threading.Lock()
frame_condition = threading.Condition(latest_lock)


class PreviewHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        pass

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/status":
            with latest_lock:
                payload = json.dumps(
                    {
                        "ready": latest_frame["jpeg"] is not None,
                        "sequence": latest_frame["sequence"],
                        "left_right_delta_ms": latest_frame["left_right_delta_ms"],
                        "middle_delta_ms": latest_frame["middle_delta_ms"],
                        "middle_camera_enabled": MIDDLE_CAMERA_ENABLED,
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
        if path == "/frame.jpg":
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
        stream_name = {
            "/stream.mjpg": "jpeg",
            "/left.mjpg": "left_jpeg",
            "/middle.mjpg": "middle_jpeg",
            "/right.mjpg": "right_jpeg",
        }.get(path)
        if stream_name is not None:
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
                            lambda: latest_frame[stream_name] is not None
                            and latest_frame["sequence"] != sequence,
                            timeout=5,
                        )
                        jpeg = latest_frame[stream_name]
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
        if path != "/":
            self.send_error(404)
            return
        page = """<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>
<title>OAK-D Lite camera preview</title><style>body{background:#111820;color:#e8eef2;font:16px system-ui;margin:2rem}img{width:min(98vw,1600px);background:#222;border-radius:8px}</style>
<h2 id=title>OAK-D Lite · CAM_B gauche | CAM_C droite</h2><img id=frame src='/stream.mjpg'><p id=status>En attente d’une paire synchronisée…</p>
<script>const s=document.querySelector('#status'),t=document.querySelector('#title');async function poll(){try{const d=await (await fetch('/status',{cache:'no-store'})).json();t.textContent=d.middle_camera_enabled?'OAK-D Lite · CAM_B gauche | CAM_A couleur | CAM_C droite':'OAK-D Lite · CAM_B gauche | CAM_C droite';if(d.ready)s.textContent=d.middle_camera_enabled?`Flux 3 caméras · séq. ${d.sequence} · CAM_B/CAM_C ${Number(d.left_right_delta_ms).toFixed(2)} ms · CAM_A/CAM_B ${Number(d.middle_delta_ms).toFixed(2)} ms`:`Paire synchronisée · séquence ${d.sequence} · écart ${Number(d.left_right_delta_ms).toFixed(3)} ms (seuil ${d.sync_threshold_ms} ms)`}catch(e){s.textContent='Flux caméra déconnecté'}}setInterval(poll,500);poll()</script>""".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)


def build_pipeline(fps: int, midlecam: bool = False) -> dai.Pipeline:
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

    if midlecam:
        middle = pipeline.create(dai.node.ColorCamera)
        middle.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        middle.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        middle.setPreviewSize(640, 360)
        middle.setFps(fps)

        middle_output = pipeline.create(dai.node.XLinkOut)
        middle_output.setStreamName("middle")
        middle_output.input.setBlocking(False)
        middle_output.input.setQueueSize(2)
        middle.preview.link(middle_output.input)
    return pipeline


def as_bgr(frame):
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def fit_frame(frame, width: int, height: int):
    frame = as_bgr(frame)
    frame_height, frame_width = frame.shape[:2]
    scale = min(width / frame_width, height / frame_height)
    resized_width = max(1, round(frame_width * scale))
    resized_height = max(1, round(frame_height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height))
    horizontal = width - resized_width
    vertical = height - resized_height
    return cv2.copyMakeBorder(
        resized,
        vertical // 2,
        vertical - vertical // 2,
        horizontal // 2,
        horizontal - horizontal // 2,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def main() -> None:
    global SYNC_THRESHOLD_MS, MIDDLE_CAMERA_ENABLED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--sync-threshold-ms", type=float, default=SYNC_THRESHOLD_MS)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--preview-host", default="0.0.0.0")
    parser.add_argument("--preview-port", type=int, default=9011)
    parser.add_argument(
        "--midlecam",
        action="store_true",
        help="inclut la caméra couleur centrale CAM_A dans l'aperçu",
    )
    args = parser.parse_args()
    if not 1 <= args.fps <= 24:
        parser.error("--fps must be in [1, 24]")
    if not 0.5 <= args.sync_threshold_ms <= 20.0:
        parser.error("--sync-threshold-ms must be in [0.5, 20]")
    if not 30 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [30, 100]")
    SYNC_THRESHOLD_MS = args.sync_threshold_ms
    MIDDLE_CAMERA_ENABLED = args.midlecam

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    server.setblocking(False)
    preview = ThreadingHTTPServer((args.preview_host, args.preview_port), PreviewHandler)
    threading.Thread(target=preview.serve_forever, daemon=True).start()
    print(f"[oak_bridge] camera preview at http://<jetson-ip>:{args.preview_port}/")
    print(f"[oak_bridge] waiting for optional Rust recorder on {args.host}:{args.port}")

    pipeline = build_pipeline(args.fps, args.midlecam)
    conn = None
    with dai.Device(pipeline) as device:
        queue = device.getOutputQueue(name="stereo", maxSize=2, blocking=False)
        middle_queue = (
            device.getOutputQueue(name="middle", maxSize=2, blocking=False)
            if args.midlecam
            else None
        )
        middle_frame = None
        middle_timestamp = None
        if args.midlecam:
            camera_layout = "CAM_B/CAM_A couleur/CAM_C"
        else:
            camera_layout = "CAM_B/CAM_C"
        print(
            f"[oak_bridge] {camera_layout} @ {args.fps} fps; "
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
            left_bgr = as_bgr(left_frame)
            right_bgr = as_bgr(right_frame)
            middle_delta_ms = None
            middle_preview = None
            if middle_queue is not None:
                next_middle = middle_queue.tryGet()
                while next_middle is not None:
                    middle_frame = next_middle.getCvFrame()
                    middle_timestamp = next_middle.getTimestampDevice()
                    next_middle = middle_queue.tryGet()
                if middle_frame is None or middle_timestamp is None:
                    continue
                middle_preview = as_bgr(middle_frame)
                middle_delta_ms = (
                    middle_timestamp - left_timestamp
                ).total_seconds() * 1000.0
                frame_height = left_frame.shape[0]
                frame_width = left_frame.shape[1]
                stereo_frame = cv2.hconcat(
                    (
                        left_bgr,
                        fit_frame(middle_preview, frame_width, frame_height),
                        right_bgr,
                    )
                )
            else:
                stereo_frame = cv2.hconcat((left_bgr, right_bgr))
            timestamp_unix_ns = time.time_ns()
            ok, encoded = cv2.imencode(
                ".jpg", stereo_frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            )
            if not ok:
                continue
            height, width = stereo_frame.shape[:2]
            payload = encoded.tobytes()
            camera_payloads = {}
            for name, frame in (("left_jpeg", left_bgr), ("right_jpeg", right_bgr)):
                ok, camera_encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
                )
                if ok:
                    camera_payloads[name] = camera_encoded.tobytes()
            if middle_preview is not None:
                ok, camera_encoded = cv2.imencode(
                    ".jpg", middle_preview, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
                )
                if ok:
                    camera_payloads["middle_jpeg"] = camera_encoded.tobytes()
            with frame_condition:
                latest_frame["jpeg"] = payload
                latest_frame.update(camera_payloads)
                latest_frame["timestamp_unix_ns"] = timestamp_unix_ns
                latest_frame["sequence"] = latest_frame.get("sequence", 0) + 1
                latest_frame["left_right_delta_ms"] = delta_ms
                latest_frame["middle_delta_ms"] = middle_delta_ms
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
