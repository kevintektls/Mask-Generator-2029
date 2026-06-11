#!/usr/bin/env python3
"""
Robot Car — Pilotage autonome avec Mémoire Temporelle et Vitesse Adaptative
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


# ── Configuration Optimisée ────────────────────────────────────────────────────

DISPLAY_W = 640
DISPLAY_H = 480
CAM_FPS   = 60
HTTP_PORT = 5000

# Détection de lignes
USE_ADAPTIVE_THRESH = False
BINARY_THRESHOLD    = 170  

# Découpage des zones de vision
ROI_FAR_TOP     = 0.48
ROI_FAR_BOT     = 0.65
ROI_NEAR_TOP    = 0.68
ROI_NEAR_BOT    = 0.85

LANE_WIDTH_PX    = 340  
LANE_WIDTH_MIN   = 160

# 🧠 Paramètre de mémoire (Lissage temporel)
# Plus la valeur est petite (ex: 0.15), plus la voiture a de la mémoire et est fluide.
# Plus elle est grande (ex: 0.8), plus elle réagit vite mais risque d'osciller.
SMOOTHING_ALPHA = 0.25  

# 🔌 VESC
VESC_PORT            = '/dev/ttyACM0'
VESC_BAUDRATE        = 115200
VESC_TIMEOUT         = 1.0
VESC_CONNECT_RETRIES = 8
VESC_CONNECT_SETTLE  = 1.0

# 🏎️ Pilotage & Vitesse (Modérés pour la stabilité)
SERVO_CENTER    = 0.5
SERVO_RANGE     = 0.48   

AUTO_DUTY       = 0.045  # Vitesse max baissée (était à 0.07) pour garder le contrôle
AUTO_DUTY_MIN   = 0.010  # Vitesse plancher très basse pour forcer à ramper si besoin
TURN_SLOWDOWN   = 0.90   # Freine plus agressivement en virage (75% de la vitesse coupée au max)

GAMEPAD_TYPE    = Gamepad.Xbox360


# ── Variable d'état globale pour la mémoire du servo ──────────────────────────
prev_servo_pos = SERVO_CENTER


# ── Traitement d'Image ────────────────────────────────────────────────────────

def detect_lines(frame_gray: np.ndarray, adaptive: bool, threshold: int) -> np.ndarray:
    blurred = cv2.bilateralFilter(frame_gray, 7, 50, 50)
    if adaptive:
        mask = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 3
        )
    else:
        _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)   
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close) 
    return mask


def _get_band_center(mask: np.ndarray, y_top: int, y_bot: int) -> tuple[float | None, int, int, str]:
    h, w = mask.shape
    mid = w // 2
    band = mask[y_top:y_bot, :]
    
    hist = np.sum(band > 0, axis=0).astype(np.float32)
    if np.max(hist) == 0:
        return None, -1, -1, "NONE"
    
    hist = np.convolve(hist, np.ones(21, dtype=np.float32) / 21, mode="same")
    threshold = (y_bot - y_top) * 0.20  
    
    left_hits = np.where(hist[:mid][::-1] >= threshold)[0]
    left_x = (mid - 1 - int(left_hits[0])) if left_hits.size else -1

    right_hits = np.where(hist[mid:] >= threshold)[0]
    right_x = (mid + int(right_hits[0])) if right_hits.size else -1

    if left_x >= 0 and right_x >= 0:
        actual_width = right_x - left_x
        if actual_width < LANE_WIDTH_MIN:
            if hist[left_x] >= hist[right_x]: right_x = -1
            else: left_x = -1

    if left_x >= 0 and right_x >= 0:
        return (left_x + right_x) / 2.0, left_x, right_x, "BOTH"
    elif left_x >= 0:
        return left_x + (LANE_WIDTH_PX / 2.0), left_x, -1, "LEFT"
    elif right_x >= 0:
        return right_x - (LANE_WIDTH_PX / 2.0), -1, right_x, "RIGHT"
        
    return None, -1, -1, "NONE"


def compute_steering(mask: np.ndarray):
    """Calcule l'ordre servo avec Lookahead dynamique et filtre de mémoire."""
    global prev_servo_pos
    h, w = mask.shape
    mid = w // 2

    n_top, n_bot = int(h * ROI_NEAR_TOP), int(h * ROI_NEAR_BOT)
    f_top, f_bot = int(h * ROI_FAR_TOP), int(h * ROI_FAR_BOT)

    target_near, left_x, right_x, status_near = _get_band_center(mask, n_top, n_bot)
    target_far, _, _, status_far = _get_band_center(mask, f_top, f_bot)

    if target_near is None and target_far is None:
        return prev_servo_pos, False, mid, -1, -1, "NONE", "NONE"

    # ── Lookahead Dynamique Adaptatif ──────────────────────────────────────────
    if target_near is not None and target_far is not None:
        far_error = abs(target_far - mid) / (w / 2.0)
        
        if far_error > 0.30:  
            # Virage détecté au loin : priorité absolue à l'anticipation (75%)
            target = (0.25 * target_near) + (0.75 * target_far)
        else:
            # Ligne droite ou courbe très légère : on donne la priorité au NEAR (60%) 
            # pour éviter que le bruit lointain ne fasse louvoyer la voiture.
            target = (0.60 * target_near) + (0.40 * target_far)
            
    elif target_far is not None:
        target = target_far
    else:
        target = target_near

    # Calcul de l'erreur brute (-1.0 à 1.0)
    error = (target - mid) / (w / 2.0)
    sign = 1.0 if error >= 0 else -1.0
    error_smoothed = sign * (abs(error) ** 1.1)

    # Cible brute instantanée du servo
    raw_servo_pos = SERVO_CENTER + (error_smoothed * SERVO_RANGE)
    raw_servo_pos = max(0.0, min(1.0, raw_servo_pos))
    
    # 🧠 Application de la mémoire (Filtre IIR)
    # Lisse la trajectoire en combinant l'état précédent et la nouvelle vision
    actual_servo = (1.0 - SMOOTHING_ALPHA) * prev_servo_pos + SMOOTHING_ALPHA * raw_servo_pos
    actual_servo = max(0.0, min(1.0, actual_servo))
    
    # Sauvegarde pour la prochaine frame
    prev_servo_pos = actual_servo
    
    return actual_servo, True, int(target), left_x, right_x, status_near, status_far


# ── Reste de l'architecture système ───────────────────────────────────────────
latest_frame: np.ndarray | None = None
frame_lock = threading.Lock()
stop_event = threading.Event()

class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        if self.path == "/":
            html = (
                b"<!DOCTYPE html><html><head><title>Autonomous Car Mask</title>"
                b"<style>body{background:#111;margin:0;display:flex;justify-content:center;align-items:center;height:100vh;}"
                b"img{max-width:100%;border:2px solid #00ffca;box-shadow: 0 0 20px #00ffca;}</style></head>"
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
                    with frame_lock:
                        frame = latest_frame
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
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
            print(f"[INFO] VESC connecté")
            return vesc
        except Exception: time.sleep(VESC_CONNECT_SETTLE)
    raise Exception("VESC introuvable")

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

def main():
    global latest_frame
    threading.Thread(target=start_http_server, daemon=True).start()
    
    if not Gamepad.available():
        while not Gamepad.available(): time.sleep(0.5)
    gamepad = GAMEPAD_TYPE()
    gamepad.startBackgroundUpdates()

    vesc = vesc_connect()
    pipeline = build_pipeline()

    adaptive  = USE_ADAPTIVE_THRESH
    threshold = BINARY_THRESHOLD
    count, t0, fps_val = 0, time.monotonic(), 0.0
    stopped = True  
    prev_lb = False
    has_display = bool(os.environ.get("DISPLAY"))

    with vesc:
        vesc.set_servo(SERVO_CENTER)
        vesc.set_duty_cycle(0)
        
        try:
            with dai.Device(pipeline) as device:
                q = device.getOutputQueue(name="left", maxSize=2, blocking=False)

                while not stop_event.is_set():
                    lb_now = gamepad.isConnected() and gamepad.isPressed("LB")
                    if lb_now and not prev_lb:
                        stopped = not stopped
                        print(f"[GAMEPAD] {'⏸ STOP' if stopped else '▶ RUN'}")
                    prev_lb = lb_now

                    pkt = q.tryGet()
                    if pkt is None:
                        time.sleep(0.002)
                        continue

                    raw = pkt.getCvFrame()
                    count += 1
                    now = time.monotonic()
                    if now - t0 >= 1.0:
                        fps_val = count / (now - t0)
                        count = 0
                        t0 = now

                    mask = detect_lines(raw, adaptive, threshold)
                    (servo_pos, line_found, target_x,
                     left_x, right_x, lane_near, lane_far) = compute_steering(mask)

                    if stopped:
                        duty = 0.0
                        vesc.set_duty_cycle(0.0)
                        vesc.set_servo(SERVO_CENTER)
                        direction = "STOP"
                    else:
                        # Vitesse proportionnelle au braquage
                        turn = min(1.0, abs(servo_pos - SERVO_CENTER) / SERVO_RANGE)
                        if line_found:
                            duty = AUTO_DUTY * (1.0 - TURN_SLOWDOWN * turn)
                            duty = max(AUTO_DUTY_MIN, duty)
                        else:
                            duty = AUTO_DUTY_MIN
                        
                        vesc.set_servo(servo_pos)
                        vesc.set_duty_cycle(duty)
                        
                        if servo_pos < SERVO_CENTER - 0.04: direction = "LEFT"
                        elif servo_pos > SERVO_CENTER + 0.04: direction = "RIGHT"
                        else: direction = "STRAIGHT"

                    # Rendu visuel 
                    src_w = mask.shape[1]
                    display = cv2.resize(mask, (DISPLAY_W, DISPLAY_H))
                    display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
                    sx = DISPLAY_W / src_w

                    cv2.rectangle(display, (0, int(DISPLAY_H*ROI_NEAR_TOP)), (DISPLAY_W, int(DISPLAY_H*ROI_NEAR_BOT)), (0, 255, 255), 1)
                    cv2.rectangle(display, (0, int(DISPLAY_H*ROI_FAR_TOP)), (DISPLAY_W, int(DISPLAY_H*ROI_FAR_BOT)), (255, 255, 0), 1)

                    if left_x >= 0: cv2.circle(display, (int(left_x*sx), int(DISPLAY_H*ROI_NEAR_TOP)), 5, (0, 255, 0), -1)
                    if right_x >= 0: cv2.circle(display, (int(right_x*sx), int(DISPLAY_H*ROI_NEAR_TOP)), 5, (0, 255, 0), -1)

                    tx = int(target_x * sx)
                    cv2.circle(display, (tx, int(DISPLAY_H * ROI_FAR_TOP)), 8, (0, 0, 255), -1)
                    cv2.line(display, (DISPLAY_W // 2, DISPLAY_H), (tx, int(DISPLAY_H * ROI_FAR_TOP)), (0, 0, 255), 2)

                    cv2.rectangle(display, (0, 0), (DISPLAY_W, 30), (0, 0, 0), -1)
                    status_str = f"FPS: {fps_val:.1f} | {'[STOP]' if stopped else '[AUTONOMOUS]'} | Dir: {direction} | Servo: {servo_pos:.2f} | Duty: {duty:.3f}"
                    cv2.putText(display, status_str, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 202), 1)

                    with frame_lock:
                        latest_frame = display

                    if has_display:
                        cv2.imshow("Lane Mask Output", display)
                        if cv2.waitKey(1) & 0xFF == ord("q"): stop_event.set()

        except KeyboardInterrupt: pass
        finally:
            vesc.set_duty_cycle(0)
            vesc.set_servo(SERVO_CENTER)
            gamepad.stopBackgroundUpdates()
            cv2.destroyAllWindows()
            gc.collect()

if __name__ == "__main__":
    main()