#!/usr/bin/env python3
"""
Robot Car — Pilotage Autonome Hybride (Vision Stéréo Streamée & Arrêt d'Urgence Instantané)
Plateforme : Jetson Nano 4Go + OAK-D Lite + VESC

IMPORTANT :
Ce fichier importe directement l'architecture et les pré-traitements depuis 'model_def'
et 'vision_preprocess' pour conserver une structure de fichiers propre.
"""

from __future__ import annotations
import os
import sys
import time
import gc
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np
import torch
import depthai as dai
from pyvesc import VESC

sys.path.insert(0, "/home/robotcar/Gamepad")
import Gamepad

# Importation directe depuis ton architecture de fichiers d'origine
from model_def import BehavioralCloningCNN
from vision_preprocess import make_mask_stereo, resize_for_model, CROP_TOP_RATIO


# ── CONFIGURATION SYSTÈME ──────────────────────────────────────────────────────
DISPLAY_W = 640
DISPLAY_H = 480
CAM_FPS = 30
STREAM_PORT = 8080  # Port du serveur web (http://<IP>:8080)

VESC_PORT = "/dev/ttyACM0"
VESC_BAUDRATE = 115200
VESC_TIMEOUT = 1.0

SERVO_CENTER = 0.5
SERVO_RANGE = 0.48   
MODEL_PATH = "../model/pilot_model.pth"

# 🏎️ PARAMÈTRES DE VITESSE DYNAMIQUE ADAPTATIVE
DUTY_MIN = 0.050  
DUTY_MAX = 0.11
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


# ── OUTILS CHASSIS ────────────────────────────────────────────────────────────
def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def build_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()

    cam_left = pipeline.create(dai.node.MonoCamera)
    cam_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    cam_left.setFps(CAM_FPS)

    xout_left = pipeline.create(dai.node.XLinkOut)
    xout_left.setStreamName("left")
    xout_left.input.setBlocking(False)
    xout_left.input.setQueueSize(2)
    cam_left.out.link(xout_left.input)

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


def load_model(device: torch.device) -> BehavioralCloningCNN:
    model = BehavioralCloningCNN().to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Modèle introuvable : {MODEL_PATH}")
        sys.exit(1)

    try:
        state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(MODEL_PATH, map_location=device)

    model.load_state_dict(state)
    model.eval()
    return model


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    global output_frame
    print("[INFO] Initialisation de l'Autopilote IA Avancé (Imports Externes)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Inférence sur : {device}")

    if device.type == "cpu":
        torch.set_num_threads(2)
        print("[OPTIMISATION] PyTorch limité à 2 threads CPU.")

    model = load_model(device)
    print("[INFO] Modèle chargé.")

    if Gamepad.available():
        gamepad = GAMEPAD_TYPE()
        gamepad.startBackgroundUpdates()
        print("[INFO] Manette connectée. LB = Freinage d'urgence direct.")
    else:
        gamepad = None
        print("[WARNING] Aucune manette détectée.")

    try:
        vesc = VESC(serial_port=VESC_PORT, baudrate=VESC_BAUDRATE, timeout=VESC_TIMEOUT)
        print("[INFO] VESC connecté.")
    except Exception as e:
        print(f"[ERROR] VESC impossible à joindre : {e}")
        if gamepad:
            gamepad.stopBackgroundUpdates()
        sys.exit(1)

    # Lancement du serveur de streaming HTTP
    server = ThreadedHTTPServer(('0.0.0.0', STREAM_PORT), StreamingHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[LIVE] Flux vidéo disponible sur http://localhost:{STREAM_PORT}")

    pipeline = build_pipeline()

    print("\n=== 🤖 AUTOPILOTE OPÉRATIONNEL ===")
    print(" -> LB ou CTRL+C = FREINAGE D'URGENCE IMMÉDIAT\n")

    with vesc:
        vesc.set_servo(SERVO_CENTER)
        vesc.set_duty_cycle(0)
        time.sleep(1.0)

        try:
            with dai.Device(pipeline) as device_dai:
                q_left = device_dai.getOutputQueue(name="left", maxSize=2, blocking=False)
                q_right = device_dai.getOutputQueue(name="right", maxSize=2, blocking=False)

                while True:
                    # 🚨 FREINAGE ET ARRÊT D'URGENCE MANETTE (LB)
                    if gamepad and gamepad.isConnected() and gamepad.isPressed("LB"):
                        print("\n[🚨 URGENCE] LB enfoncé ! Injection du frein électrique !")
                        vesc.set_brake(15.0)
                        vesc.set_servo(SERVO_CENTER)
                        break

                    pkt_left = q_left.tryGet()
                    pkt_right = q_right.tryGet()

                    if pkt_left is None or pkt_right is None:
                        time.sleep(0.002)
                        continue

                    raw_left = pkt_left.getCvFrame()
                    raw_right = pkt_right.getCvFrame()

                    # Utilisation stricte de tes fichiers externes importés
                    mask = make_mask_stereo(raw_left, raw_right)
                    mask_resized = resize_for_model(mask)

                    img_tensor = (
                        torch.from_numpy(mask_resized)
                        .float()
                        .unsqueeze(0)
                        .unsqueeze(0)
                        / 255.0
                    ).to(device)

                    with torch.no_grad():
                        prediction = model(img_tensor).item()

                    servo_pos = clamp(prediction, 0.0, 1.0)

                    # Gestion de la vitesse dynamique selon le braquage
                    steering_intensity = abs(servo_pos - SERVO_CENTER)
                    if steering_intensity < STEER_THRESHOLD:
                        current_duty = DUTY_MAX
                    else:
                        factor = (steering_intensity - STEER_THRESHOLD) / (SERVO_RANGE - STEER_THRESHOLD)
                        factor = clamp(factor, 0.0, 1.0)
                        current_duty = DUTY_MAX - factor * (DUTY_MAX - DUTY_MIN)

                    vesc.set_servo(servo_pos)
                    vesc.set_duty_cycle(current_duty)

                    # 🌐 PRÉPARATION DU CORPS DE L'IMAGE POUR LE FLUX MJPEG
                    display = cv2.resize(mask, (DISPLAY_W, DISPLAY_H))
                    display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

                    # Tracé de la ligne de crop importée
                    cv2.line(
                        display,
                        (0, int(DISPLAY_H * CROP_TOP_RATIO)),
                        (DISPLAY_W, int(DISPLAY_H * CROP_TOP_RATIO)),
                        (0, 0, 150),
                        1,
                    )

                    text = f"IA | Servo: {servo_pos:.2f} | Duty: {current_duty:.3f}"
                    cv2.putText(display, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    with frame_lock:
                        output_frame = display.copy()

        except KeyboardInterrupt:
            print("\n[INFO] Interruption clavier.")
        finally:
            print("[INFO] Extinction propre des systèmes...")
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