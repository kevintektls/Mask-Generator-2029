#!/usr/bin/env python3
"""
OAK-D Lane Detection — MJPEG HTTP Stream
Capture mono gauche, détecte les lignes blanches (OpenCV), stream HTTP MJPEG.

Requirements:
    pip install depthai opencv-python numpy

Usage:
    python oak_d_lane_stream.py

Controls HTTP:
    http://<ip>:5000/         - Vue navigateur
    http://<ip>:5000/stream   - Flux MJPEG brut

Controls clavier (fenêtre locale si dispo, sinon ignoré) :
    q        - Quitter
    s        - Snapshot dans ./snapshots/
    t        - Toggle seuillage adaptatif / classique
    +/-      - Ajuster le seuil classique (±5)
"""

import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

try:
    import depthai as dai
except ImportError:
    print("DepthAI not installed.  Run:  pip install depthai")
    sys.exit(1)

try:
    import cv2
    import numpy as np
except ImportError:
    print("OpenCV / NumPy not installed.  Run:  pip install opencv-python numpy")
    sys.exit(1)


# ── Configuration ──────────────────────────────────────────────────────────────
DISPLAY_W         = 640
DISPLAY_H         = 480
FPS               = 60
HTTP_PORT         = 5000
SNAPSHOT_DIR      = Path("snapshots")

# Détection de lignes
USE_ADAPTIVE_THRESH = False   # toggle avec 't'
BINARY_THRESHOLD    = 180     # seuil classique (ajustable avec +/-)


# ── Détection de lignes ────────────────────────────────────────────────────────
def detect_lines(frame_gray: np.ndarray, adaptive: bool, threshold: int) -> np.ndarray:
    """
    Prend une image en niveaux de gris, retourne un masque binaire BGR (640×480).
    """
    blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)

    if adaptive:
        mask = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 4
        )
    else:
        _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = cv2.resize(mask, (DISPLAY_W, DISPLAY_H))
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


# ── Frame partagée thread-safe ────────────────────────────────────────────────
latest_frame: np.ndarray | None = None
frame_lock = threading.Lock()


# ── Serveur MJPEG ──────────────────────────────────────────────────────────────
class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def do_GET(self):
        if self.path == "/":
            html = (
                b"<!DOCTYPE html><html><head><title>Lane Detection</title>"
                b"<style>body{background:#111;margin:0;display:flex;"
                b"justify-content:center;align-items:center;height:100vh;}"
                b"img{max-width:100%;border:2px solid #0f0;}</style></head>"
                b"<body><img src='/stream'></body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        frame = latest_frame
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    ok, jpg = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
                    )
                    if not ok:
                        continue
                    data = jpg.tobytes()
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(data)).encode()
                        + b"\r\n\r\n" + data + b"\r\n"
                    )
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()


def start_http_server():
    HTTPServer(("0.0.0.0", HTTP_PORT), MJPEGHandler).serve_forever()


# ── Pipeline OAK-D ────────────────────────────────────────────────────────────
def build_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.MonoCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    cam.setFps(FPS)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("left")
    xout.input.setBlocking(False)
    xout.input.setQueueSize(2)
    cam.out.link(xout.input)

    return pipeline


# ── Snapshot ──────────────────────────────────────────────────────────────────
def save_snapshot(frame: np.ndarray):
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SNAPSHOT_DIR / f"{ts}_mask.png"
    cv2.imwrite(str(path), frame)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global latest_frame

    threading.Thread(target=start_http_server, daemon=True).start()
    print(f"✓ Stream sur http://localhost:{HTTP_PORT}  (ou l'IP de ta Jetson)")

    pipeline = build_pipeline()

    adaptive  = USE_ADAPTIVE_THRESH
    threshold = BINARY_THRESHOLD

    count, t0, fps_val = 0, time.monotonic(), 0.0

    with dai.Device(pipeline) as device:
        q = device.getOutputQueue(name="left", maxSize=2, blocking=False)
        print("✓ Camera connectée — Ctrl+C pour quitter")
        print("  Clavier (fenêtre locale) : q=quit  s=snapshot  t=toggle seuil  +/-=seuil\n")

        while True:
            pkt = q.tryGet()
            if pkt is None:
                time.sleep(0.001)
                continue

            raw = pkt.getCvFrame()   # uint8 grayscale

            # FPS
            count += 1
            now = time.monotonic()
            if now - t0 >= 1.0:
                fps_val = count / (now - t0)
                count = 0
                t0 = now

            # Détection
            display = detect_lines(raw, adaptive, threshold)

            # OSD
            mode_str = "ADAPTIVE" if adaptive else f"BINARY thr={threshold}"
            label = f"{fps_val:.1f} FPS | {mode_str}"
            cv2.putText(display, label, (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            with frame_lock:
                latest_frame = display

            # Fenêtre locale (optionnelle, échoue silencieusement si pas de display)
            try:
                cv2.imshow("Lane Mask", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Quitting …")
                    break
                elif key == ord("s"):
                    save_snapshot(display)
                elif key == ord("t"):
                    adaptive = not adaptive
                    print(f"Seuillage → {'ADAPTIVE' if adaptive else 'CLASSIQUE'}")
                elif key in (ord("+"), ord("=")):
                    threshold = min(threshold + 5, 250)
                    print(f"Seuil → {threshold}")
                elif key == ord("-"):
                    threshold = max(threshold - 5, 10)
                    print(f"Seuil → {threshold}")
            except cv2.error:
                pass   # Pas de display dispo (Jetson headless)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nQuitting.")
