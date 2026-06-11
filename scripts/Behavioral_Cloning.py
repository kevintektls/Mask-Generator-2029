#!/usr/bin/env python3
"""
Robot Car — Pilotage & Enregistrement pour Entraînement IA (Behavioral Cloning)
Plateforme : Jetson Nano 4Go (Optimisé RAM & Stockage via Masque Binaire)
Version corrigée : Mapping Manuel Manette Alternatif
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
CAM_FPS   = 60
HTTP_PORT = 5000

# ✂️ Rognage horizon & Seuil ultra-binaire
CROP_TOP_RATIO      = 0.40  
ULTRA_BINARY_THRESH = 220  

# Zones de vision pour l'algo géométrique classique
ROI_FAR_TOP     = 0.15
ROI_FAR_BOT     = 0.45
ROI_NEAR_TOP    = 0.50
ROI_NEAR_BOT    = 0.85

LANE_WIDTH_PX    = 340  
LANE_WIDTH_MIN   = 160
SMOOTHING_ALPHA  = 0.25  

# VESC
VESC_PORT            = '/dev/ttyACM0'
VESC_BAUDRATE        = 115200
VESC_TIMEOUT         = 1.0
VESC_CONNECT_RETRIES = 8
VESC_CONNECT_SETTLE  = 1.0

# Pilotage Auto Géométrique
SERVO_CENTER    = 0.5
SERVO_RANGE     = 0.48   
AUTO_DUTY       = 0.045  
AUTO_DUTY_MIN   = 0.010  
TURN_SLOWDOWN   = 0.90   

# 📂 Configuration de l'Enregistrement IA
DATASET_DIR = Path("dataset")
IMAGES_DIR  = DATASET_DIR / "images"
CSV_FILE    = DATASET_DIR / "driving_log.csv"

GAMEPAD_TYPE = Gamepad.Xbox360


# ── Variables d'état globales ──────────────────────────────────────────────────
prev_servo_pos = SERVO_CENTER
is_recording   = False
csv_writer     = None
csv_file_handle = None
record_lock    = threading.Lock()


# ── Vision Ultra-Binaire Rognée ───────────────────────────────────────────────

def detect_lines(frame_gray: np.ndarray) -> np.ndarray:
    h, w = frame_gray.shape
    clean_mask = np.zeros_like(frame_gray)
    
    start_y = int(h * CROP_TOP_RATIO)
    roi_sol = frame_gray[start_y:h, :]
    
    blurred = cv2.GaussianBlur(roi_sol, (5, 5), 0)
    _, binary_sol = cv2.threshold(blurred, ULTRA_BINARY_THRESH, 255, cv2.THRESH_BINARY)
    
    clean_mask[start_y:h, :] = binary_sol
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
    
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

    with vesc:
        vesc.set_servo(SERVO_CENTER)
        vesc.set_duty_cycle(0)
        print("\n=== SYSTEM DATA LOGGER READY ===")
        print(" -> Mode courant : 🎮 MANUEL")
        print(" -> Bouton A   : ÉCRIRE / STOPPER le Dataset [REC]")
        print(" -> Bouton LB  : Basculer en mode autonome classique géométrique\n")
        
        try:
            with dai.Device(pipeline) as device:
                q = device.getOutputQueue(name="left", maxSize=2, blocking=False)

                while not stop_event.is_set():
                    lb_now = gamepad.isConnected() and gamepad.isPressed("LB")
                    a_now  = gamepad.isConnected() and gamepad.isPressed("A")
                    
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

                    pkt = q.tryGet()
                    if pkt is None:
                        time.sleep(0.002)
                        continue

                    raw = pkt.getCvFrame()
                    h, w = raw.shape
                    count += 1
                    now = time.monotonic()
                    if now - t0 >= 1.0:
                        fps_val = count / (now - t0)
                        count = 0
                        t0 = now

                    mask = detect_lines(raw)

                    if autonomous_mode:
                        (servo_pos, line_found, target_x, left_x, right_x, _, _) = compute_steering(mask)
                        turn = min(1.0, abs(servo_pos - SERVO_CENTER) / SERVO_RANGE)
                        duty = max(AUTO_DUTY_MIN, AUTO_DUTY * (1.0 - TURN_SLOWDOWN * turn)) if line_found else AUTO_DUTY_MIN
                    else:
                        # ── 🕹️ Lecture Robuste via Dictionnaire Interne de la Lib ──
                        steer_input = 0.0
                        gas_input = 0.0
                        
                        if hasattr(gamepad, '_getAxis'):
                            # Essai avec la méthode interne brute de l'API
                            steer_input = gamepad._getAxis(0) # Généralement l'axe X du stick gauche
                            gas_input = gamepad._getAxis(5)  # Généralement l'axe RT
                        elif hasattr(gamepad, 'axisNames'):
                            # Fallback si les axes sont mappés en chaînes brutes dans l'instance
                            for name in gamepad.axisNames:
                                if "X" in name or "STICK" in name: steer_input = getattr(gamepad, name, 0.0)
                                if "TR" in name or "RT" in name: gas_input = getattr(gamepad, name, 0.0)
                        
                        # Si l'API renvoie des valeurs entre 0 et 255 au lieu de 0.0/1.0 pour la gâchette
                        if abs(gas_input) > 1.0: gas_input = gas_input / 255.0
                        if abs(steer_input) > 1.0: steer_input = steer_input / 255.0

                        servo_pos = SERVO_CENTER + (steer_input * SERVO_RANGE)
                        servo_pos = max(0.0, min(1.0, servo_pos))
                        
                        duty = abs(gas_input) * AUTO_DUTY
                        target_x, left_x, right_x = w // 2, -1, -1

                    # Application VESC
                    vesc.set_servo(servo_pos)
                    vesc.set_duty_cycle(duty)

                    # Sauvegarde
                    if is_recording and duty > 0.005:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        img_name = f"line_{timestamp}.png"
                        img_path = IMAGES_DIR / img_name
                        
                        threading.Thread(target=cv2.imwrite, args=(str(img_path), mask)).start()
                        
                        with record_lock:
                            csv_writer.writerow([f"images/{img_name}", f"{servo_pos:.4f}", f"{duty:.4f}"])
                            csv_file_handle.flush()

                    # HUD
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