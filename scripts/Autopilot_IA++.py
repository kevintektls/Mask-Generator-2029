#!/usr/bin/env python3
"""
Robot Car — Pilotage Autonome Hybride (Vision Streamée & Arrêt d'Urgence Instantané)
Plateforme : Jetson Nano 4Go (Optimisé CPU)
"""

from __future__ import annotations
import os
import sys
import time
import gc
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import torch
import torch.nn as nn
import depthai as dai
import cv2
import numpy as np
from pyvesc import VESC

sys.path.insert(0, '/home/robotcar/Gamepad')
import Gamepad

# ── CONFIGURATION SYSTÈME ──────────────────────────────────────────────────────
DISPLAY_W = 640
DISPLAY_H = 480
CAM_FPS   = 60
STREAM_PORT = 8080  # Port du serveur web pour voir la caméra (http://<IP>:8080)

CROP_TOP_RATIO      = 0.20
ULTRA_BINARY_THRESH = 220  

VESC_PORT     = '/dev/ttyACM0'
VESC_BAUDRATE = 115200
VESC_TIMEOUT  = 1.0

SERVO_CENTER    = 0.5
SERVO_RANGE     = 0.48   
MODEL_PATH      = "../model/pilot_model.pth"

# 🏎️ PARAMÈTRES DE VITESSE DYNAMIQUE
DUTY_MIN        = 0.050  
DUTY_MAX        = 0.070  
STEER_THRESHOLD = 0.08   

GAMEPAD_TYPE = Gamepad.Xbox360

# Variable globale partagée pour le streaming vidéo
output_frame = None
frame_lock = threading.Lock()


# ── 🌐 SERVEUR DE STREAMING VIDÉO HTTP (MJPEG) ──────────────────────────────────
class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global output_frame
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        if output_frame is None:
                            time.sleep(0.01)
                            continue
                        _, encoded_img = cv2.imencode('.jpg', output_frame)
                        buffer = encoded_img.tobytes()
                    
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(buffer)))
                    self.end_headers()
                    self.wfile.write(buffer)
                    self.wfile.write(b'\r\n')
                    time.sleep(1 / CAM_FPS)
            except Exception as e:
                pass

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Serveur HTTP supportant le multi-threading pour éviter de bloquer l'IA."""
    allow_reuse_address = True


# ── ARCHITECTURE DU RÉSEAU CNN ────────────────────────────────────────────────
class BehavioralCloningCNN(nn.Module):
    def __init__(self):
        super(BehavioralCloningCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.flatten = nn.Flatten()
        self.regressor = nn.Sequential(
            nn.Linear(64 * 10 * 15, 100), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.regressor(x).squeeze(1)


# ── TRAITEMENT DE VISION ──────────────────────────────────────────────────────
def detect_lines(frame_gray: np.ndarray) -> np.ndarray:
    h, w = frame_gray.shape
    clean_mask = np.zeros_like(frame_gray)
    start_y = int(h * CROP_TOP_RATIO)
    roi_sol = frame_gray[start_y:h, :]
    blurred = cv2.GaussianBlur(roi_sol, (5, 5), 0)
    _, binary_sol = cv2.threshold(blurred, ULTRA_BINARY_THRESH, 255, cv2.THRESH_BINARY)
    clean_mask[start_y:h, :] = binary_sol
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)


def main():
    global output_frame
    print("[INFO] Initialisation de l'Autopilote IA Avancé...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(2)

    # 1. Modèle
    model = BehavioralCloningCNN().to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Modèle introuvable : {MODEL_PATH}")
        sys.exit(1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # 2. Sécurités & Hardware
    gamepad = GAMEPAD_TYPE() if Gamepad.available() else None
    if gamepad: 
        gamepad.startBackgroundUpdates()
        print("[INFO] Manette connectée.")
    
    try:
        vesc = VESC(serial_port=VESC_PORT, baudrate=VESC_BAUDRATE, timeout=VESC_TIMEOUT)
        print("[INFO] VESC Connecté.")
    except Exception as e:
        print(f"[ERROR] Impossible de joindre le VESC : {e}")
        sys.exit(1)

    # 3. Lancement du serveur Web de streaming en tâche de fond
    server = ThreadedHTTPServer(('0.0.0.0', STREAM_PORT), StreamingHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[LIVE] Flux vidéo disponible sur http://localhost:{STREAM_PORT} (ou l'IP de la Jetson)")

    # 4. Pipeline DepthAI
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.MonoCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    cam.setFps(CAM_FPS)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("left")
    cam.out.link(xout.input)

    print("\n=== 🤖 AUTOPILOTE OPÉRATIONNEL ===")
    print(" -> LB enfoncé : FREINAGE D'URGÈNCE ÉLECTRIQUE IMMÉDIAT.")

    with vesc:
        vesc.set_servo(SERVO_CENTER)
        vesc.set_duty_cycle(0)
        time.sleep(1.0)

        try:
            with dai.Device(pipeline) as device_dai:
                q = device_dai.getOutputQueue(name="left", maxSize=2, blocking=False)

                while True:
                    # 🚨 ARRÊT D'URGENCE CRITIQUE INTERNE ET FORCE
                    if gamepad and gamepad.isConnected() and gamepad.isPressed("LB"):
                        print("\n[🚨 URGENCE KRITIK] Bouton LB détecté ! Application du frein moteur direct !")
                        # Ordre de freinage immédiat au VESC (Courant de freinage inverse à 15A pour bloquer les roues)
                        vesc.set_brake(15.0) 
                        vesc.set_servo(SERVO_CENTER)
                        break

                    pkt = q.tryGet()
                    if pkt is None:
                        time.sleep(0.002)
                        continue

                    raw = pkt.getCvFrame()
                    mask = detect_lines(raw)

                    # Inférence IA
                    mask_resized = cv2.resize(mask, (160, 120))
                    img_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

                    with torch.no_grad():
                        prediction = model(img_tensor).item()

                    servo_pos = max(0.0, min(1.0, prediction))

                    # Vitesse adaptative
                    steering_intensity = abs(servo_pos - SERVO_CENTER)
                    if steering_intensity < STEER_THRESHOLD:
                        current_duty = DUTY_MAX
                    else:
                        factor = (steering_intensity - STEER_THRESHOLD) / (SERVO_RANGE - STEER_THRESHOLD)
                        factor = min(1.0, max(0.0, factor))
                        current_duty = DUTY_MAX - factor * (DUTY_MAX - DUTY_MIN)

                    # Envoi des consignes physiques standard
                    vesc.set_servo(servo_pos)
                    vesc.set_duty_cycle(current_duty)

                    # 🌐 PRÉPARATION DU FLUX STREAMING (Met à jour l'image partagée)
                    display = cv2.resize(mask, (DISPLAY_W, DISPLAY_H))
                    display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
                    status_str = f"Servo: {servo_pos:.2f} | Duty: {current_duty:.3f}"
                    cv2.putText(display, status_str, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    with frame_lock:
                        output_frame = display.copy()

        except KeyboardInterrupt:
            print("\n[INFO] Interruption manuelle demandée.")
        finally:
            print("[INFO] Extinction propre des systèmes...")
            # On force la coupure complète en sortie
            try:
                vesc.set_duty_cycle(0)
                vesc.set_brake(10.0)
                vesc.set_servo(SERVO_CENTER)
            except:
                pass
            if gamepad:
                gamepad.stopBackgroundUpdates()
            server.shutdown()
            cv2.destroyAllWindows()
            gc.collect()

if __name__ == "__main__":
    main()