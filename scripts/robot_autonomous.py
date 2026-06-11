#!/usr/bin/env python3
"""
Robot Car — Pilotage autonome par détection de lignes
OAK-D (DepthAI) → OpenCV lane detection → VESC (direction + vitesse)
Stream MJPEG HTTP du masque sur :5000

Requirements:
    pip install depthai opencv-python numpy pyvesc
    + Gamepad lib dans /home/robotcar/Gamepad

Usage:
    python robot_autonomous.py

Contrôles clavier (fenêtre locale si dispo) :
    q   - Quitter
    s   - Snapshot dans ./snapshots/
    t   - Toggle seuillage adaptatif / classique
    +/- - Ajuster le seuil classique (±5)

Sécurité :
    - Bouton LB du gamepad = arrêt d'urgence (coupe le VESC)
    - Ctrl+C = arrêt propre
    - Si aucune ligne détectée → vitesse réduite à AUTO_DUTY_SLOW
"""

from __future__ import annotations

import sys
import time
import gc
import threading
from datetime import datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, '/home/robotcar/Gamepad')

try:
    import Gamepad
except ImportError:
    print("Gamepad lib introuvable dans /home/robotcar/Gamepad")
    sys.exit(1)

try:
    from pyvesc import VESC
except ImportError:
    print("pyvesc not installed.  Run:  pip install pyvesc")
    sys.exit(1)

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

# Caméra
DISPLAY_W = 640
DISPLAY_H = 480
CAM_FPS   = 60

# HTTP
HTTP_PORT = 5000

# Détection de lignes
USE_ADAPTIVE_THRESH = False
BINARY_THRESHOLD    = 180
# ROI : on ne traite que la moitié basse de l'image (zone pertinente pour les lignes au sol)
ROI_TOP_RATIO = 0.5

# VESC
VESC_PORT            = '/dev/ttyACM0'
VESC_BAUDRATE        = 115200
VESC_TIMEOUT         = 1.0
VESC_CONNECT_RETRIES = 8
VESC_CONNECT_SETTLE  = 1.0

# Pilotage autonome
SERVO_CENTER    = 0.5
SERVO_RANGE     = 0.45   # amplitude max de braquage depuis le centre
AUTO_DUTY       = 0.08   # vitesse normale en autonome
AUTO_DUTY_SLOW  = 0.04   # vitesse réduite si aucune ligne détectée
MAX_DUTY_CYCLE  = 0.10

# Gamepad (arrêt d'urgence uniquement)
GAMEPAD_TYPE    = Gamepad.Xbox360
POLL_INTERVAL   = 0.05

SNAPSHOT_DIR = Path("snapshots")


# ── Détection de lignes ────────────────────────────────────────────────────────

def detect_lines(frame_gray: np.ndarray, adaptive: bool, threshold: int) -> np.ndarray:
    """Retourne un masque binaire uint8 (même taille que frame_gray)."""
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
    return mask


def compute_steering(mask: np.ndarray) -> tuple[float, bool]:
    """
    Calcule la consigne de servo à partir du masque binaire.

    Stratégie : centroïde horizontal des pixels blancs dans la ROI basse.
    Retourne (servo_position 0.0-1.0, ligne_détectée).

    - Centre image  → servo 0.5 (tout droit)
    - Décalage gauche → servo < 0.5 (virer à gauche)
    - Décalage droite → servo > 0.5 (virer à droite)
    """
    h, w = mask.shape
    roi_y = int(h * ROI_TOP_RATIO)
    roi = mask[roi_y:, :]  # moitié basse

    white_pixels = cv2.findNonZero(roi)
    if white_pixels is None or len(white_pixels) < 50:
        return SERVO_CENTER, False  # pas de ligne → tout droit par défaut

    cx = float(np.mean(white_pixels[:, 0, 0]))  # x moyen des pixels blancs
    # Normalise : 0.0 (extrême gauche) → 1.0 (extrême droite)
    normalized = cx / w
    # Erreur par rapport au centre
    error = normalized - 0.5
    servo = SERVO_CENTER + error * SERVO_RANGE
    servo = max(0.0, min(1.0, servo))
    return servo, True


# ── Frame partagée (thread-safe) ──────────────────────────────────────────────
latest_frame: np.ndarray | None = None
frame_lock = threading.Lock()

# Signal d'arrêt global
stop_event = threading.Event()


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
                while not stop_event.is_set():
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


# ── VESC ───────────────────────────────────────────────────────────────────────
def vesc_connect() -> VESC:
    last_exc = None
    for attempt in range(VESC_CONNECT_RETRIES):
        try:
            vesc = VESC(serial_port=VESC_PORT, baudrate=VESC_BAUDRATE, timeout=VESC_TIMEOUT)
            print(f"[INFO] VESC connecté (tentative {attempt + 1})")
            return vesc
        except Exception as e:
            last_exc = e
            print(f"[WARNING] Tentative {attempt + 1} échouée : {e}")
            time.sleep(VESC_CONNECT_SETTLE)
    raise Exception(f"Impossible de connecter le VESC après {VESC_CONNECT_RETRIES} tentatives : {last_exc}")


# ── Pipeline OAK-D ────────────────────────────────────────────────────────────
def build_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.MonoCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    cam.setFps(CAM_FPS)
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

    # HTTP stream
    threading.Thread(target=start_http_server, daemon=True).start()
    print(f"✓ Stream sur http://localhost:{HTTP_PORT}")

    # Gamepad (arrêt d'urgence)
    if not Gamepad.available():
        print("[INFO] En attente du gamepad…")
        while not Gamepad.available():
            time.sleep(1)
    gamepad = GAMEPAD_TYPE()
    gamepad.startBackgroundUpdates()
    print("✓ Gamepad connecté  (LB = arrêt d'urgence)")

    # VESC
    print(f"[INFO] Connexion VESC sur {VESC_PORT}…")
    vesc = vesc_connect()

    # Caméra
    pipeline = build_pipeline()

    adaptive  = USE_ADAPTIVE_THRESH
    threshold = BINARY_THRESHOLD
    count, t0, fps_val = 0, time.monotonic(), 0.0
    emergency_stop = False

    with vesc:
        vesc.set_servo(SERVO_CENTER)
        vesc.set_duty_cycle(0)
        time.sleep(0.5)
        print("✓ VESC prêt — démarrage du pilotage autonome")
        print("  Clavier local : q=quit  s=snapshot  t=toggle seuil  +/-=seuil\n")

        try:
            with dai.Device(pipeline) as device:
                q = device.getOutputQueue(name="left", maxSize=2, blocking=False)

                while not stop_event.is_set():

                    # ── Arrêt d'urgence gamepad ────────────────────────────
                    if gamepad.isConnected() and gamepad.isPressed("LB"):
                        if not emergency_stop:
                            print("[URGENCE] LB pressé → arrêt moteur")
                            emergency_stop = True
                        vesc.set_duty_cycle(0)
                        vesc.set_servo(SERVO_CENTER)
                        time.sleep(POLL_INTERVAL)
                        continue
                    else:
                        emergency_stop = False

                    # ── Frame caméra ───────────────────────────────────────
                    pkt = q.tryGet()
                    if pkt is None:
                        time.sleep(0.001)
                        continue

                    raw = pkt.getCvFrame()  # uint8 grayscale

                    # FPS
                    count += 1
                    now = time.monotonic()
                    if now - t0 >= 1.0:
                        fps_val = count / (now - t0)
                        count = 0
                        t0 = now

                    # ── Détection + pilotage ───────────────────────────────
                    mask = detect_lines(raw, adaptive, threshold)
                    servo_pos, line_found = compute_steering(mask)

                    duty = AUTO_DUTY if line_found else AUTO_DUTY_SLOW

                    vesc.set_servo(servo_pos)
                    vesc.set_duty_cycle(duty)

                    # ── Affichage ──────────────────────────────────────────
                    display = cv2.resize(mask, (DISPLAY_W, DISPLAY_H))
                    display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

                    # Ligne de centroïde visuelle
                    roi_y_px = int(DISPLAY_H * ROI_TOP_RATIO)
                    cx_px = int(servo_pos * DISPLAY_W)
                    cv2.line(display, (cx_px, roi_y_px), (cx_px, DISPLAY_H),
                             (0, 0, 255), 2)
                    cv2.line(display, (DISPLAY_W // 2, roi_y_px),
                             (DISPLAY_W // 2, DISPLAY_H), (255, 0, 0), 1)

                    status = "LINE OK" if line_found else "NO LINE"
                    mode_str = "ADAPTIVE" if adaptive else f"BINARY thr={threshold}"
                    label = (f"{fps_val:.1f} FPS | {mode_str} | "
                             f"servo={servo_pos:.2f} duty={duty:.2f} | {status}")
                    cv2.putText(display, label, (8, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    with frame_lock:
                        latest_frame = display

                    # ── Clavier local ──────────────────────────────────────
                    try:
                        cv2.imshow("Lane Mask", display)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            print("Quitting…")
                            stop_event.set()
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
                        pass  # headless

        except KeyboardInterrupt:
            print("\n[INFO] Ctrl+C reçu")

        finally:
            print("[INFO] Arrêt propre…")
            vesc.set_duty_cycle(0)
            vesc.set_servo(SERVO_CENTER)
            gamepad.stopBackgroundUpdates()
            cv2.destroyAllWindows()
            gc.collect()


if __name__ == "__main__":
    main()
