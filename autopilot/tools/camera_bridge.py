#!/usr/bin/env python3
"""
DepthAI camera bridge — same pipeline + vision algo as scripts/Autopilot_IA++.py.

Runs make_mask_stereo() in Python (OpenCV, fast on Jetson) and streams the
160x120 model mask to Rust. Rust only runs CNN inference (no slow detect_lines).

Protocol per frame:
  magic "MASK" (4 bytes) + width u32 LE + height u32 LE + gray pixels (160*120)

Usage (Jetson, from autopilot/):
  python3 tools/camera_bridge.py
  python3 tools/camera_bridge.py --fps 30 --stats

Start this BEFORE ./target/release/autopilot
"""

from __future__ import annotations

import argparse
import errno
import socket
import struct
import sys
import time
from pathlib import Path

try:
    import depthai as dai
except ImportError:
    print("depthai not installed. Run: pip install depthai==2.29.0")
    sys.exit(1)

# vision_preprocess.py — same module as Autopilot_IA++.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
try:
    from vision_preprocess import make_mask_stereo, resize_for_model, MODEL_W, MODEL_H
except ImportError as e:
    print(f"Cannot import vision_preprocess: {e}")
    print(f"Expected scripts/ under {_REPO_ROOT}")
    sys.exit(1)

MAGIC = b"MASK"
DEFAULT_FPS = 30
HEADER = struct.Struct("<4sII")
FRAME_BYTES = MODEL_W * MODEL_H


def build_pipeline(fps: int) -> dai.Pipeline:
    """Identical stereo mono setup to scripts/Autopilot_IA++.py."""
    pipeline = dai.Pipeline()

    cam_left = pipeline.create(dai.node.MonoCamera)
    cam_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    cam_left.setFps(fps)

    xout_left = pipeline.create(dai.node.XLinkOut)
    xout_left.setStreamName("left")
    xout_left.input.setBlocking(False)
    xout_left.input.setQueueSize(2)
    cam_left.out.link(xout_left.input)

    cam_right = pipeline.create(dai.node.MonoCamera)
    cam_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    cam_right.setFps(fps)

    xout_right = pipeline.create(dai.node.XLinkOut)
    xout_right.setStreamName("right")
    xout_right.input.setBlocking(False)
    xout_right.input.setQueueSize(2)
    cam_right.out.link(xout_right.input)

    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--stats", action="store_true", help="print FPS every second")
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
    packet = HEADER.pack(MAGIC, MODEL_W, MODEL_H)

    sent_frames = 0
    t0 = time.monotonic()

    with dai.Device(pipeline) as device:
        q_left = device.getOutputQueue(name="left", maxSize=2, blocking=False)
        q_right = device.getOutputQueue(name="right", maxSize=2, blocking=False)
        print(
            f"[camera_bridge] stereo CAM_B+CAM_C → mask {MODEL_W}x{MODEL_H} "
            f"@ {args.fps} fps (vision_preprocess algo)"
        )

        while True:
            # Same grab pattern as Autopilot_IA++.py
            pkt_left = q_left.tryGet()
            pkt_right = q_right.tryGet()

            if pkt_left is None or pkt_right is None:
                time.sleep(0.002)
                continue

            raw_left = pkt_left.getCvFrame()
            raw_right = pkt_right.getCvFrame()

            mask = make_mask_stereo(raw_left, raw_right)
            mask_model = resize_for_model(mask)

            conn.sendall(packet + mask_model.tobytes())

            sent_frames += 1
            if args.stats:
                now = time.monotonic()
                if now - t0 >= 1.0:
                    print(f"[camera_bridge] tx {sent_frames / (now - t0):.1f} fps")
                    sent_frames = 0
                    t0 = now


if __name__ == "__main__":
    try:
        main()
    except (BrokenPipeError, ConnectionResetError, OSError) as e:
        if isinstance(e, OSError) and e.errno not in (
            errno.EPIPE,
            errno.ECONNRESET,
            errno.ENOTCONN,
        ):
            raise
        print("[camera_bridge] client disconnected")
