#!/usr/bin/env python3
"""
DepthAI 2.x camera bridge for the Rust autopilot.

Uses the same mono CAM_B 480p pipeline as scripts/Autopilot_IA++.py and streams
grayscale frames over TCP to the Rust binary.

Protocol per frame:
  magic "OAK1" (4 bytes) + width u32 LE + height u32 LE + gray pixels (w*h)

Usage (Jetson, from autopilot/):
  python3 tools/camera_bridge.py
  python3 tools/camera_bridge.py --fps 60 --port 9000

Start this BEFORE ./target/release/autopilot
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time

try:
    import depthai as dai
except ImportError:
    print("depthai not installed. Run: pip install depthai==2.29.0")
    sys.exit(1)

MAGIC = b"OAK1"
MONO_W, MONO_H = 640, 480


def build_pipeline(fps: int) -> dai.Pipeline:
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.MonoCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    cam.setFps(fps)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("left")
    xout.input.setBlocking(False)
    xout.input.setQueueSize(2)
    cam.out.link(xout.input)
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"[camera_bridge] waiting for autopilot on {args.host}:{args.port} ...")
    conn, addr = server.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[camera_bridge] autopilot connected from {addr}")

    pipeline = build_pipeline(args.fps)
    with dai.Device(pipeline) as device:
        q = device.getOutputQueue(name="left", maxSize=2, blocking=False)
        print(f"[camera_bridge] OAK-D CAM_B streaming {MONO_W}x{MONO_H} @ {args.fps} fps")
        while True:
            pkt = q.tryGet()
            if pkt is None:
                time.sleep(0.001)
                continue
            frame = pkt.getCvFrame()
            if frame.ndim != 2:
                continue
            h, w = frame.shape
            header = MAGIC + struct.pack("<II", w, h)
            try:
                conn.sendall(header + frame.tobytes())
            except (BrokenPipeError, ConnectionResetError):
                print("[camera_bridge] client disconnected")
                break


if __name__ == "__main__":
    main()
