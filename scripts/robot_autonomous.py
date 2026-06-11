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

import os
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
# ROI : bande étroite juste devant la voiture (on ignore l'horizon qui est bruité)
ROI_TOP_RATIO    = 0.78
ROI_BOTTOM_RATIO = 0.98
# Largeur typique de voie en pixels (utilisée quand on ne voit qu'une seule ligne)
LANE_WIDTH_PX = 320
# Tolérance autour de LANE_WIDTH_PX : si l'écart entre les 2 pics est hors [min,max],
# on retombe en mode "une seule ligne" (plus robuste face au bruit)
LANE_WIDTH_MIN = 180
LANE_WIDTH_MAX = 520

# VESC
VESC_PORT            = '/dev/ttyACM0'
VESC_BAUDRATE        = 115200
VESC_TIMEOUT         = 1.0
VESC_CONNECT_RETRIES = 8
VESC_CONNECT_SETTLE  = 1.0

# Pilotage autonome
SERVO_CENTER    = 0.5
SERVO_RANGE     = 0.45   # amplitude max de braquage depuis le centre
AUTO_DUTY       = 0.08   # vitesse max en ligne droite
AUTO_DUTY_MIN   = 0.04   # vitesse min (gros virage / aucune ligne)
TURN_SLOWDOWN   = 0.7    # quel % de AUTO_DUTY on coupe au braquage max (0..1)
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


def compute_steering(mask: np.ndarray):
    """
    Reste ENTRE deux lignes blanches sur fond noir.

    Algo :
      1. Bande étroite devant la voiture → histogramme par colonne.
      2. On part du centre image et on scanne vers l'extérieur de chaque côté.
         La PREMIÈRE colonne franchissant le seuil = frontière de la voie.
         Tout ce qu'il y a au-delà (bruit, horizon, autres lignes) est ignoré.
      3. Cible = milieu entre les deux frontières.
      4. Si une seule vue : on suppose l'autre à LANE_WIDTH_PX.

    Retourne (servo, found, target_x, left_x, right_x, lane_status)
      lane_status ∈ {"BOTH", "LEFT", "RIGHT", "NONE"}
    """
    h, w = mask.shape
    y0 = int(h * ROI_TOP_RATIO)
    y1 = int(h * ROI_BOTTOM_RATIO)
    band = mask[y0:y1, :]
    band_h = max(1, y1 - y0)

    # Histogramme par colonne, lissé pour gommer les spikes
    hist = np.sum(band > 0, axis=0).astype(np.float32)
    k = 15
    hist = np.convolve(hist, np.ones(k, dtype=np.float32) / k, mode="same")

    mid = w // 2
    # Seuil : une vraie ligne remplit au moins 25 % de la hauteur de la bande
    threshold = band_h * 0.25

    # Scan depuis le centre vers la gauche → 1ère colonne au-dessus du seuil
    left_hits = np.where(hist[:mid][::-1] >= threshold)[0]
    left_x = (mid - 1 - int(left_hits[0])) if left_hits.size else -1

    # Scan depuis le centre vers la droite
    right_hits = np.where(hist[mid:] >= threshold)[0]
    right_x = (mid + int(right_hits[0])) if right_hits.size else -1

    # Sanity check : voie trop étroite (probable artefact)
    if left_x >= 0 and right_x >= 0 and (right_x - left_x) < LANE_WIDTH_MIN:
        if hist[left_x] >= hist[right_x]:
            right_x = -1
        else:
            left_x = -1

    if left_x >= 0 and right_x >= 0:
        target = (left_x + right_x) / 2.0
        status = "BOTH"
    elif left_x >= 0:
        target = left_x + LANE_WIDTH_PX / 2.0
        status = "LEFT"
    elif right_x >= 0:
        target = right_x - LANE_WIDTH_PX / 2.0
        status = "RIGHT"
    else:
        return SERVO_CENTER, False, mid, -1, -1, "NONE"

    error = (target - mid) / (w / 2.0)
    error = max(-1.0, min(1.0, error))
    servo = SERVO_CENTER + error * SERVO_RANGE
    servo = max(0.0, min(1.0, servo))
    return servo, True, int(target), left_x, right_x, status


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
    stopped = False         # état RUN/STOP toggle via LB
    prev_lb = False         # détection front montant
    has_display = bool(os.environ.get("DISPLAY"))
    if not has_display:
        print("[INFO] Pas de DISPLAY → fenêtre locale désactivée (stream HTTP uniquement)")

    with vesc:
        vesc.set_servo(SERVO_CENTER)
        vesc.set_duty_cycle(0)
        time.sleep(0.5)
        print("✓ VESC prêt — démarrage du pilotage autonome")
        print("  Manette : LB = toggle RUN/STOP  (caméra continue dans tous les cas)\n")

        try:
            with dai.Device(pipeline) as device:
                q = device.getOutputQueue(name="left", maxSize=2, blocking=False)

                while not stop_event.is_set():

                    # ── Toggle RUN/STOP (front montant sur LB) ─────────────
                    lb_now = gamepad.isConnected() and gamepad.isPressed("LB")
                    if lb_now and not prev_lb:
                        stopped = not stopped
                        print(f"[GAMEPAD] {'⏸  STOP' if stopped else '▶  RUN'}")
                    prev_lb = lb_now

                    # ── Frame caméra (toujours, même à l'arrêt) ────────────
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

                    # ── Détection (toujours) ───────────────────────────────
                    mask = detect_lines(raw, adaptive, threshold)
                    servo_pos, line_found, target_x, left_x, right_x, lane_status = compute_steering(mask)

                    # ── Décision moteur ────────────────────────────────────
                    if stopped:
                        duty = 0.0
                        vesc.set_duty_cycle(0.0)
                        vesc.set_servo(SERVO_CENTER)
                        direction = "STOP"
                    else:
                        # Vitesse = AUTO_DUTY en ligne droite, ↘ vers AUTO_DUTY_MIN
                        # quand on braque à fond. Plus le servo s'éloigne du centre,
                        # plus on ralentit pour ne pas survirer.
                        turn = min(1.0, abs(servo_pos - SERVO_CENTER) / SERVO_RANGE)
                        if line_found:
                            duty = AUTO_DUTY * (1.0 - TURN_SLOWDOWN * turn)
                            duty = max(AUTO_DUTY_MIN, duty)
                        else:
                            duty = AUTO_DUTY_MIN
                        vesc.set_servo(servo_pos)
                        vesc.set_duty_cycle(duty)
                        if servo_pos < SERVO_CENTER - 0.05:
                            direction = "LEFT"
                        elif servo_pos > SERVO_CENTER + 0.05:
                            direction = "RIGHT"
                        else:
                            direction = "STRAIGHT"

                    # ── Affichage ──────────────────────────────────────────
                    src_w = mask.shape[1]
                    display = cv2.resize(mask, (DISPLAY_W, DISPLAY_H))
                    display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

                    sx = DISPLAY_W / src_w  # facteur d'échelle X
                    roi_y0_px = int(DISPLAY_H * ROI_TOP_RATIO)
                    roi_y1_px = int(DISPLAY_H * ROI_BOTTOM_RATIO)

                    # Bande analysée (cadre jaune)
                    cv2.rectangle(display, (0, roi_y0_px), (DISPLAY_W - 1, roi_y1_px),
                                  (0, 255, 255), 1)

                    # Centre image (bleu)
                    cv2.line(display, (DISPLAY_W // 2, roi_y0_px),
                             (DISPLAY_W // 2, roi_y1_px), (255, 0, 0), 1)

                    # Lignes détectées (vert)
                    if left_x >= 0:
                        x = int(left_x * sx)
                        cv2.line(display, (x, roi_y0_px), (x, roi_y1_px), (0, 255, 0), 2)
                    if right_x >= 0:
                        x = int(right_x * sx)
                        cv2.line(display, (x, roi_y0_px), (x, roi_y1_px), (0, 255, 0), 2)

                    # Cible (rouge)
                    tx = int(target_x * sx)
                    cv2.line(display, (tx, roi_y0_px), (tx, DISPLAY_H), (0, 0, 255), 2)
                    cv2.circle(display, (tx, (roi_y0_px + roi_y1_px) // 2), 6, (0, 0, 255), -1)

                    # Bandeau d'état (haut)
                    state_color = (0, 0, 255) if stopped else (0, 200, 0)
                    state_text = "⏸ STOPPED (LB pour RUN)" if stopped else "▶ RUNNING (LB pour STOP)"
                    cv2.rectangle(display, (0, 0), (DISPLAY_W, 28), (0, 0, 0), -1)
                    cv2.putText(display, state_text, (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

                    # Bandeau décisions (bas)
                    cv2.rectangle(display, (0, DISPLAY_H - 56), (DISPLAY_W, DISPLAY_H), (0, 0, 0), -1)
                    line1 = f"DIR: {direction:<8}  LANES: {lane_status:<5}  servo={servo_pos:.2f}  duty={duty:.2f}"
                    line2 = f"{fps_val:4.1f} FPS  |  thr={threshold}  |  target_x={target_x}  ({'ADAPT' if adaptive else 'BIN'})"
                    cv2.putText(display, line1, (8, DISPLAY_H - 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(display, line2, (8, DISPLAY_H - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                    with frame_lock:
                        latest_frame = display

                    # ── Clavier local ──────────────────────────────────────
                    if has_display:
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
