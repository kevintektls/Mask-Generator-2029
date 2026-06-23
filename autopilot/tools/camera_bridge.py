#!/usr/bin/env python3
"""
DepthAI 2.x camera bridge for the Rust autopilot.

Uses the same mono CAM_B 480p pipeline as scripts/Autopilot_IA++.py and streams
grayscale frames over TCP to the Rust binary.

Protocol per frame:
  magic "OAK1" (4 bytes) + width u32 LE + height u32 LE + gray pixels (w*h)

Usage (Jetson, from autopilot/):
  python3 tools/camera_bridge.py
  python3 tools/camera_bridge.py --fps 60 --port 9000 --stats

Start this BEFORE ./target/release/autopilot
"""

from __future__ import annotations

import argparse
import errno
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
FRAME_BYTES = MONO_W * MONO_H
HEADER = struct.Struct("<4sII")


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


def latest_packet(q: dai.DataOutputQueue) -> dai.ImgFrame | None:
    """Drain the DepthAI queue and return only the freshest frame."""
    pkt: dai.ImgFrame | None = None
    while True:
        nxt = q.tryGet()
        if nxt is None:
            break
        pkt = nxt
    return pkt


def frame_payload(pkt: dai.ImgFrame) -> bytes | None:
    """GRAY8 payload bytes for a 640x480 mono frame."""
    data = pkt.getData()
    if data is not None and int(data.size) == FRAME_BYTES:
        return data.tobytes()

    # Fallback — some depthai builds expose only OpenCV frames.
    frame = pkt.getCvFrame()
    if frame is not None and frame.ndim == 2 and int(frame.size) == FRAME_BYTES:
        return frame.tobytes()
    return None


def flush_pending(conn: socket.socket, pending: bytearray) -> None:
    """Send all of `pending`; never return with a partial frame on the wire."""
    while pending:
        try:
            n = conn.send(pending)
        except BlockingIOError:
            time.sleep(0.0002)
            continue
        except InterruptedError:
            continue
        if n == 0:
            raise ConnectionError("camera bridge: peer closed during send")
        del pending[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--stats", action="store_true", help="print FPS every second")
    args = parser.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"[camera_bridge] waiting for autopilot on {args.host}:{args.port} ...")
    conn, addr = server.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 << 20)
    print(f"[camera_bridge] autopilot connected from {addr}")

    pipeline = build_pipeline(args.fps)
    pending = bytearray()

    sent_frames = 0
    dropped_cam = 0
    t0 = time.monotonic()

    with dai.Device(pipeline) as device:
        q = device.getOutputQueue(name="left", maxSize=2, blocking=False)
        print(f"[camera_bridge] OAK-D CAM_B streaming {MONO_W}x{MONO_H} @ {args.fps} fps target")
        while True:
            # Finish the current TCP frame before starting another (avoids stream corruption).
            if pending:
                flush_pending(conn, pending)
                sent_frames += 1
                if args.stats:
                    now = time.monotonic()
                    if now - t0 >= 1.0:
                        fps = sent_frames / (now - t0)
                        print(
                            f"[camera_bridge] tx {fps:.1f} fps "
                            f"(bad_cam={dropped_cam})"
                        )
                        sent_frames = 0
                        dropped_cam = 0
                        t0 = now
                continue

            pkt = latest_packet(q)
            if pkt is None:
                time.sleep(0.001)
                continue

            payload = frame_payload(pkt)
            if payload is None:
                dropped_cam += 1
                continue

            pending.extend(HEADER.pack(MAGIC, MONO_W, MONO_H))
            pending.extend(payload)


if __name__ == "__main__":
    try:
        main()
    except (BrokenPipeError, ConnectionResetError, ConnectionError, OSError) as e:
        if isinstance(e, OSError) and e.errno not in (
            errno.EPIPE,
            errno.ECONNRESET,
            errno.ENOTCONN,
        ):
            raise
        print("[camera_bridge] client disconnected")
