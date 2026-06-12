#!/usr/bin/env python3
"""
Robot Car — Pilotage & Enregistrement pour Entraînement IA (Behavioral Cloning)
Plateforme : Jetson Nano 4Go (Optimisé RAM & Stockage via Masque Binaire)
Version : Optimisation Stéréo (Left + Right) & Filtrage Avancé du Masque
"""

from __future__ import annotations
import os
import sys
import time
import gc
import csv
import threading
from datetime import datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Queue, Empty

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


# ── Configuration Système ──────────────────────────────────────────────────────

DISPLAY_W = 640
DISPLAY_H = 480
CAM_FPS   = 30
HTTP_PORT = 5000

# ✂️ Rognage horizon & Seuil ultra-binaire
CROP_TOP_RATIO      = 0.45  # Légèrement descendu pour éviter les bruits lointains
ULTRA_BINARY_THRESH = 215   # Ajusté pour être un poil plus tolérant avec la fusion

# Zones de vision pour l'algo géométrique classique
ROI_FAR_TOP     = 0.15
ROI_FAR_BOT     = 0.45
ROI_NEAR_TOP    = 0.50
ROI_NEAR_BOT    = 0.85

LANE_WIDTH_PX    = 340  
LANE_WIDTH_MIN   = 160
SMOOTHING_ALPHA  = 0.25  

# VESC Connection
VESC_PORT            = '/dev/ttyACM0'
VESC_BAUDRATE        = 115200
VESC_TIMEOUT         = 1.0
VESC_CONNECT_RETRIES = 8
VESC_CONNECT_SETTLE  = 1.0

# 🎮 Mapping Manette Logitech F710 (Mode X)
GAMEPAD_TYPE   = Gamepad.Xbox360
AXIS_FORWARD   = "RT"
AXIS_BACKWARD  = "LT"
AXIS_STEERING  = "LEFT-X"
DEADZONE       = 0.08

# Paramètres Physiques Pilotage
SERVO_CENTER    = 0.5
SERVO_RANGE     = 0.48   
AUTO_DUTY       = 0.1
AUTO_DUTY_MIN   = 0.010  
TURN_SLOWDOWN   = 0.90   

# 📂 Configuration du l'Enregistrement IA
DATASET_DIR = Path("dataset")
IMAGES_DIR  = DATASET_DIR / "images"
CSV_FILE    = DATASET_DIR / "driving_log.csv"


# ── Variables d'état globales ──────────────────────────────────────────────────
prev_servo_pos = SERVO_CENTER
is_recording   = False
csv_writer     = None
csv_file_handle = None
record_lock    = threading.Lock()
write_queue    = Queue(maxsize=60) 

# ── Fonctions Utilitaires Manette ─────────────────────────────────────────────

def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))

def apply_deadzone(value: float) -> float:
    if abs(value) < DEADZONE:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - DEADZONE) / (1.0 - DEADZONE)


# ── Vision Stéréo Ultra-Binaire Nettoyée ──────────────────────────────────────

def detect_lines_stereo(frame_left: np.ndarray, frame_right: np.ndarray) -> np.ndarray:
    """
    Version corrigée : Filtre le bruit hors piste avant de lisser les lignes.
    """
    # 1. Fusion des deux vues
    merged = cv2.max(frame_left, frame_right)
    
    h, w = merged.shape
    clean_mask = np.zeros_like(merged)
    
    # 2. Rognage horizon
    start_y = int(h * CROP_TOP_RATIO)
    roi_sol = merged[start_y:h, :]
    
    # 3. Filtre bilatéral pour lisser le grain de la piste
    filtered = cv2.bilateralFilter(roi_sol, d=5, sigmaColor=40, sigmaSpace=40)
    
    # 4. Seuillage TRÈS strict (on passe à 230 pour tuer le gris/imperfections)
    _, binary_sol = cv2.threshold(filtered, 230, 255, cv2.THRESH_BINARY)
    
    # 5. Nettoyage Morphologique Intelligent
    # On utilise une forme d'ellipse (plus naturelle pour les perspectives)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Étape A : OPEN (Erosion puis Dilatation) -> Supprime les petits points isolés hors piste
    binary_sol = cv2.morphologyEx(binary_sol, cv2.MORPH_OPEN, kernel)
    
    # Étape B : CLOSE (Dilatation puis Erosion) -> Bouche les petits trous DANS les lignes
    binary_sol = cv2.morphologyEx(binary_sol, cv2.MORPH_CLOSE, kernel)
    
    # Remise dans le masque global
    clean_mask[start_y:h, :] = binary_sol
    
    return clean_mask


def _get_band_center(mask: np.ndarray, y_top: int, y_bot: int) -> tuple[float | None, int, int, str]:
    h, w = mask.shape
    mid = w // 2
    band = mask[y_top:y_bot, :]
    
    hist = np.sum(band > 0, axis=0).astype(np.float32)
    if np.max(hist) == 0: return None, -1, -1, "NONE"
    
    hist = np.convolve(hist, np.ones(21, dtype=np.float32) / 21, mode="same")
    threshold = (y_bot - y_top) * 0.20  
    
    left_hits = np.where(hist[:mid][::-1] >= threshold)[0]
    left_x = (mid - 1 - int(left_hits[0])) if left_hits.size else -1

    right_hits = np.where(hist[mid:] >= threshold)[0]
    right_x = (mid + int(right_hits[0])) if right_hits.size else -1

    if left_x >= 0 and right_x >= 0:
        if (right_x - left_x) < LANE_WIDTH_MIN:
            if hist[left_x] >= hist[right_x]: right_x = -1
            else: left_x = -1

    if left_x >= 0 and right_x >= 0: return (left_x + right_x) / 2.0, left_x, right_x, "BOTH"
    elif left_x >= 0: return left_x + (LANE_WIDTH_PX / 2.0), left_x, -1, "LEFT"
    elif right_x >= 0: return right_x - (LANE_WIDTH_PX / 2.0), -1, right_x, "RIGHT"
    return None, -1, -1, "NONE"


def compute_steering(mask: np.ndarray):
    global prev_servo_pos
    h, w = mask.shape
    mid = w // 2

    n_top, n_bot = int(h * ROI_NEAR_TOP), int(h * ROI_NEAR_BOT)
    f_top, f_bot = int(h * ROI_FAR_TOP), int(h * ROI_FAR_BOT)

    target_near, left_x, right_x, status_near = _get_band_center(mask, n_top, n_bot)
    target_far, _, _, status_far = _get_band_center(mask, f_top, f_bot)

    if target_near is None and target_far is None:
        if abs(prev_servo_pos - SERVO_CENTER) > 0.15: return prev_servo_pos, False, mid, -1, -1, "NONE", "NONE"
        return SERVO_CENTER, False, mid, -1, -1, "NONE", "NONE"

    if target_near is not None and target_far is not None:
        far_error = abs(target_far - mid) / mid
        near_error = abs(target_near - mid) / mid
        if far_error > 0.30 or near_error > 0.30:
            target = target_far if far_error > near_error else target_near
        else:
            target = (0.60 * target_near) + (0.40 * target_far)
    elif target_far is not None: target = target_far
    else: target = target_near

    error = (target - mid) / mid
    if abs(error) > 0.35: error_smoothed = 1.0 if error > 0 else -1.0
    else:
        sign = 1.0 if error >= 0 else -1.0
        error_smoothed = sign * (abs(error) ** 1.1)

    raw_servo_pos = SERVO_CENTER + (error_smoothed * SERVO_RANGE)
    raw_servo_pos = max(0.0, min(1.0, raw_servo_pos))
    
    alpha = 0.45 if abs(raw_servo_pos - SERVO_CENTER) > abs(prev_servo_pos - SERVO_CENTER) else 0.12
    actual_servo = (1.0 - alpha) * prev_servo_pos + alpha * raw_servo_pos
    actual_servo = max(0.0, min(1.0, actual_servo))
    
    prev_servo_pos = actual_servo
    return actual_servo, True, int(target), left_x, right_x, status_near, status_far


# ── Initialisation du Dataset ─────────────────────────────────────────────────

def init_dataset():
    global csv_writer, csv_file_handle
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = CSV_FILE.exists()
    
    csv_file_handle = open(CSV_FILE, mode="a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file_handle)
    
    if not file_exists:
        csv_writer.writerow(["image_path", "servo", "duty"])
        csv_file_handle.flush()


# ── Serveur Vidéo HTTP ────────────────────────────────────────────────────────
latest_frame: np.ndarray | None = None
frame_lock = threading.Lock()
stop_event = threading.Event()

class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        if self.path == "/":
            html = (
                b"<!DOCTYPE html><html><head><title>Robot Car Training</title>"
                b"<style>body{background:#111;margin:0;display:flex;justify-content:center;align-items:center;height:100vh;}"
                b"img{max-width:100%;border:2px solid #ff0055;box-shadow: 0 0 25px #ff0055;}</style></head>"
                b"<body><img src='/stream'></body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not stop_event.is_set():
                    with frame_lock: frame = latest_frame
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    if not ok: continue
                    data = jpg.tobytes()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
            except (BrokenPipeError, ConnectionResetError): pass
        else:
            self.send_response(404)
            self.end_headers()

def start_http_server():
    try: HTTPServer(("0.0.0.0", HTTP_PORT), MJPEGHandler).serve_forever()
    except Exception: pass

def vesc_connect() -> VESC:
    for attempt in range(VESC_CONNECT_RETRIES):
        try:
            vesc = VESC(serial_port=VESC_PORT, baudrate=VESC_BAUDRATE, timeout=VESC_TIMEOUT)
            print("[INFO] VESC Connecté")
            return vesc
        except Exception: time.sleep(VESC_CONNECT_SETTLE)
    raise Exception("Erreur : VESC introuvable.")


# ── Configuration Pipeline Stéréo DepthAI ─────────────────────────────────────

def build_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()
    
    # Caméra Gauche
    cam_left = pipeline.create(dai.node.MonoCamera)
    cam_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    cam_left.setFps(CAM_FPS)
    
    xout_left = pipeline.create(dai.node.XLinkOut)
    xout_left.setStreamName("left")
    xout_left.input.setBlocking(False)
    xout_left.input.setQueueSize(2)
    cam_left.out.link(xout_left.input)
    
    # Caméra Droite
    cam_right = pipeline.create(dai.node.MonoCamera)
    cam_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    cam_right.setFps(CAM_FPS)
    
    xout_right = pipeline.create(dai.node.XLinkOut)
    xout_right.setStreamName("right")
    xout_right.input.setBlocking(False)
    xout_right.input.setQueueSize(2)
    cam_right.out.link(xout_right.input)
    
    return pipeline

def disk_writer():
    while True:
        try:
            img_path, data, row = write_queue.get(timeout=1.0)
            cv2.imwrite(str(img_path), data)
            with record_lock:
                csv_writer.writerow(row)
                csv_file_handle.flush()
        except Empty:
            continue

# ── Boucle Principale de Contrôle ─────────────────────────────────────────────

def main():
    global latest_frame, is_recording, csv_writer, csv_file_handle
    threading.Thread(target=start_http_server, daemon=True).start()
    
    if not Gamepad.available():
        while not Gamepad.available(): time.sleep(0.5)
    gamepad = GAMEPAD_TYPE()
    gamepad.startBackgroundUpdates()

    init_dataset()
    vesc = vesc_connect()
    pipeline = build_pipeline()

    count, t0, fps_val = 0, time.monotonic(), 0.0
    autonomous_mode = False  
    prev_lb = False
    prev_a  = False
    has_display = bool(os.environ.get("DISPLAY"))

    threading.Thread(target=disk_writer, daemon=True).start()
    with vesc:
        vesc.set_servo(SERVO_CENTER)
        vesc.set_duty_cycle(0)
        print("\n=== SYSTEM DATA LOGGER READY ===")
        print(" -> Mode courant : 🎮 MANUEL")
        print(" -> Bouton A   : ÉCRIRE / STOPPER le Dataset [REC]")
        print(" -> Bouton LB  : Basculer en mode autonome classique géométrique\n")
        
        try:
            with dai.Device(pipeline) as device:
                # Récupération des deux files d'attente
                q_left = device.getOutputQueue(name="left", maxSize=2, blocking=False)
                q_right = device.getOutputQueue(name="right", maxSize=2, blocking=False)

                while not stop_event.is_set() and gamepad.isConnected():
                    lb_now = gamepad.isPressed("LB")
                    a_now  = gamepad.isPressed("A")
                    
                    if lb_now and not prev_lb:
                        autonomous_mode = not autonomous_mode
                        if autonomous_mode:
                            with record_lock:
                                is_recording = False
                        print(f"[MODE] {'🏎️ AUTONOME GÉOMÉTRIQUE' if autonomous_mode else '🎮 CONDUITE MANUELLE'}")
                    prev_lb = lb_now

                    if a_now and not prev_a and not autonomous_mode:
                        with record_lock:
                            is_recording = not is_recording
                        print(f"[DATASET] {'🔴 ENREGISTREMENT EN COURS...' if is_recording else '⏹️ ENREGISTREMENT STOPPÉ'}")
                    prev_a = a_now

                    # Récupération synchrone des frames OAK-D
                    pkt_left = q_left.tryGet()
                    pkt_right = q_right.tryGet()
                    
                    if pkt_left is None or pkt_right is None:
                        time.sleep(0.002)
                        continue

                    raw_left = pkt_left.getCvFrame()
                    raw_right = pkt_right.getCvFrame()
                    
                    h, w = raw_left.shape
                    count += 1
                    now = time.monotonic()
                    if now - t0 >= 1.0:
                        fps_val = count / (now - t0)
                        count = 0
                        t0 = now

                    # Appel de notre fonction de traitement stéréo ultra-propre
                    mask = detect_lines_stereo(raw_left, raw_right)

                    if autonomous_mode:
                        (servo_pos, line_found, target_x, left_x, right_x, _, _) = compute_steering(mask)
                        turn = min(1.0, abs(servo_pos - SERVO_CENTER) / SERVO_RANGE)
                        duty = max(AUTO_DUTY_MIN, AUTO_DUTY * (1.0 - TURN_SLOWDOWN * turn)) if line_found else AUTO_DUTY_MIN
                    else:
                        forward_raw  = gamepad.axis(AXIS_FORWARD)
                        backward_raw = gamepad.axis(AXIS_BACKWARD)
                        steering_raw = gamepad.axis(AXIS_STEERING)

                        throttle = clamp(forward_raw - backward_raw, -1.0, 1.0)
                        throttle = apply_deadzone(throttle)
                        duty = clamp(throttle * AUTO_DUTY, -AUTO_DUTY, AUTO_DUTY)

                        steer_v = apply_deadzone(steering_raw)
                        servo_pos = clamp(SERVO_CENTER + steer_v * SERVO_RANGE, 0.0, 1.0)
                        
                        target_x, left_x, right_x = w // 2, -1, -1

                    vesc.set_servo(servo_pos)
                    vesc.set_duty_cycle(duty)

                    if is_recording and abs(duty) > 0.005:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        img_name = f"line_{timestamp}.png"
                        img_path = IMAGES_DIR / img_name
                        row = [f"images/{img_name}", f"{servo_pos:.4f}", f"{duty:.4f}"]
                        try:
                            write_queue.put_nowait((img_path, mask.copy(), row))
                        except:
                            pass

                    # ── Rendu Visuel HUD ──────────────────────────────────────
                    src_w = mask.shape[1]
                    display = cv2.resize(mask, (DISPLAY_W, DISPLAY_H))
                    display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
                    sx = DISPLAY_W / src_w

                    cv2.line(display, (0, int(DISPLAY_H*CROP_TOP_RATIO)), (DISPLAY_W, int(DISPLAY_H*CROP_TOP_RATIO)), (0, 0, 150), 1)

                    if autonomous_mode:
                        cv2.rectangle(display, (0, int(DISPLAY_H*ROI_NEAR_TOP)), (DISPLAY_W, int(DISPLAY_H*ROI_NEAR_BOT)), (0, 255, 255), 1)
                        cv2.rectangle(display, (0, int(DISPLAY_H*ROI_FAR_TOP)), (DISPLAY_W, int(DISPLAY_H*ROI_FAR_BOT)), (255, 255, 0), 1)
                        tx = int(target_x * sx)
                        cv2.circle(display, (tx, int(DISPLAY_H * ROI_FAR_TOP)), 8, (0, 0, 255), -1)
                        cv2.line(display, (DISPLAY_W // 2, DISPLAY_H), (tx, int(DISPLAY_H * ROI_FAR_TOP)), (0, 0, 255), 2)

                    cv2.rectangle(display, (0, 0), (DISPLAY_W, 30), (0, 0, 0), -1)
                    mode_str = "[AUTO]" if autonomous_mode else "[MANUAL]"
                    rec_str = " | 🔴 REC" if is_recording else ""
                    status_str = f"{fps_val:.1f} FPS | {mode_str}{rec_str} | Servo: {servo_pos:.2f} | Duty: {duty:.3f}"
                    hud_color = (0, 0, 255) if is_recording else (0, 255, 202)
                    cv2.putText(display, status_str, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1)

                    with frame_lock: latest_frame = display

                    if has_display:
                        cv2.imshow("Lane Mask Output", display)
                        if cv2.waitKey(1) & 0xFF == ord("q"): stop_event.set()

        except KeyboardInterrupt: pass
        finally:
            with record_lock:
                if csv_file_handle: csv_file_handle.close()
            vesc.set_duty_cycle(0)
            vesc.set_servo(SERVO_CENTER)
            gamepad.stopBackgroundUpdates()
            cv2.destroyAllWindows()
            gc.collect()

if __name__ == "__main__":
    main()