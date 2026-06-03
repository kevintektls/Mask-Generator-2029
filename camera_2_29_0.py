#!/usr/bin/env python3
"""
OAK-D Lite Camera Viewer  —  DepthAI v2.29 API
Displays left mono, right mono, and depth streams via USB.

Requirements:
    pip install depthai==2.29.0 opencv-python numpy

Usage:
    python oak_d_lite_viewer.py

Controls:
    q        - Quit
    s        - Save a snapshot of all streams into ./snapshots/
    d        - Cycle depth colormap (TURBO → BONE → HSV)
    +/-      - Increase / decrease max depth range (±1 m steps)
"""

import sys
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
FPS = 60
DEPTH_MAX_MM = 8_000  # Initial max depth shown (mm)
SNAPSHOT_DIR = Path("snapshots")

COLORMAPS = [cv2.COLORMAP_TURBO, cv2.COLORMAP_BONE, cv2.COLORMAP_HSV]
COLORMAP_NAMES = ["TURBO", "BONE", "HSV"]


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


def save_snapshot(frames: dict):
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name, frame in frames.items():
        path = SNAPSHOT_DIR / f"{ts}_{name}.png"
        cv2.imwrite(str(path), frame)
        print(f"  Saved: {path}")


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

    print("Starting pipeline …")

    # v2 API: use Device as a context manager, then start the pipeline
    with dai.Device(pipeline) as device:
        q_left = device.getOutputQueue(name="left", maxSize=4, blocking=False)
        q_right = device.getOutputQueue(name="right", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        print("Connected!  Press  q=quit  s=snapshot  d=colormap  +/-=depth range\n")

        latest = {k: None for k in ("left", "right", "depth")}

        while True:
            # Grab latest frames (non-blocking)
            for name, q in (("left", q_left), ("right", q_right), ("depth", q_depth)):
                pkt = q.tryGet()
                if pkt is not None:
                    latest[name] = pkt
                    tick(name)

            show = {}

            if latest["left"] is not None:
                f = latest["left"].getCvFrame()
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                f = resize_to_preview(f)
                show["LEFT"] = draw_fps(f, fps_data["left"]["fps"], "LEFT")

            if latest["right"] is not None:
                f = latest["right"].getCvFrame()
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                f = resize_to_preview(f)
                show["RIGHT"] = draw_fps(f, fps_data["right"]["fps"], "RIGHT")

            if latest["depth"] is not None:
                raw = latest["depth"].getFrame()
                f = colorize_depth(raw, COLORMAPS[colormap_idx], depth_max)
                f = resize_to_preview(f)
                lbl = f"DEPTH [{COLORMAP_NAMES[colormap_idx]}] max={depth_max // 1000}m"
                show["DEPTH"] = draw_fps(f, fps_data["depth"]["fps"], lbl)

            for win, frame in show.items():
                cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Quitting …")
                break
            elif key == ord("s"):
                print("Saving snapshot …")
                save_snapshot(show)
            elif key == ord("d"):
                colormap_idx = (colormap_idx + 1) % len(COLORMAPS)
                print(f"Depth colormap → {COLORMAP_NAMES[colormap_idx]}")
            elif key in (ord("+"), ord("=")):
                depth_max = min(depth_max + 1000, 20_000)
                print(f"Depth max → {depth_max} mm")
            elif key == ord("-"):
                depth_max = max(depth_max - 1000, 1_000)
                print(f"Depth max → {depth_max} mm")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
