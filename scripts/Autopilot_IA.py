#!/usr/bin/env python3
"""
Robot Car — Pilotage Autonome par Intelligence Artificielle (Behavioral Cloning)
Plateforme : Jetson Nano 4Go (Inférence PyTorch en temps réel - Optimisé CPU)
"""

from __future__ import annotations
import os
import sys
import time
import gc
import threading
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

# Même traitement d'image que lors de l'enregistrement
CROP_TOP_RATIO      = 0.40  
ULTRA_BINARY_THRESH = 220  

# Configuration VESC
VESC_PORT     = '/dev/ttyACM0'
VESC_BAUDRATE = 115200
VESC_TIMEOUT  = 1.0

# Paramètres de conduite de l'IA
SERVO_CENTER    = 0.5
SERVO_RANGE     = 0.48   
AUTO_DUTY       = 0.040  # Vitesse de croisière sécurisée pour l'IA
MODEL_PATH      = "../model/best_autopilot.pth "

# Configuration Manette (Logitech F710 / Xbox360) pour la reprise de contrôle urgente
GAMEPAD_TYPE = Gamepad.Xbox360


# ── ARCHITECTURE DU RÉSEAU CNN (Doit être STRICTEMENT IDENTIQUE au PC) ────────
class BehavioralCloningCNN(nn.Module):
    def __init__(self):
        super(BehavioralCloningCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2), 
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), 
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), 
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), 
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 10 * 15, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1) 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(1)


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
    print("[INFO] Initialisation de l'Autopilote IA...")
    
    # 1. Sélection du hardware (Ici, forcera le CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Inférence exécutée sur : {device}")

    # 🚀 OPTIMISATION CPU : Empêche PyTorch de saturer tous les cœurs et de faire laguer l'OS
    if device.type == "cpu":
        torch.set_num_threads(2) # Laisse 2 cœurs libres pour DepthAI et la gestion système
        print("[OPTIMISATION] Nombre de threads PyTorch restreint à 2 pour préserver la Jetson.")

    # 2. Chargement du modèle entraîné
    model = BehavioralCloningCNN().to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Le fichier modèle '{MODEL_PATH}' est introuvable au chemin : {MODEL_PATH}")
        sys.exit(1)
        
    # Le map_location=device force la conversion des tenseurs du fichier vers le CPU sans planter
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Mode évaluation obligatoire
    print("[INFO] Modèle de Behavioral Cloning chargé avec succès.")

    # 3. Connexion au Gamepad (Sécurité)
    if Gamepad.available():
        gamepad = GAMEPAD_TYPE()
        gamepad.startBackgroundUpdates()
        print("[INFO] Manette connectée pour sécurité (Bouton LB = Arrêt d'urgence).")
    else:
        gamepad = None
        print("[WARNING] Aucune manette détectée. Arrêt d'urgence clavier uniquement.")

    # 4. Connexion VESC
    try:
        vesc = VESC(serial_port=VESC_PORT, baudrate=VESC_BAUDRATE, timeout=VESC_TIMEOUT)
        print("[INFO] VESC Connecté.")
    except Exception as e:
        print(f"[ERROR] Impossible de joindre le VESC : {e}")
        sys.exit(1)

    # 5. Pipeline Caméra DepthAI
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

    has_display = bool(os.environ.get("DISPLAY"))
    ai_active = True

    print("\n=== 🤖 AUTOPILOTE IA PRÊT (MODE CPU) ===")
    print(" -> Appuie sur LB sur la manette ou CTRL+C pour stopper la voiture immédiatement.\n")

    with vesc:
        vesc.set_servo(SERVO_CENTER)
        vesc.set_duty_cycle(0)
        time.sleep(1.0)

        try:
            with dai.Device(pipeline) as device_dai:
                q = device_dai.getOutputQueue(name="left", maxSize=2, blocking=False)

                while ai_active:
                    # 🚨 Sécurité : Arrêt si le bouton LB est pressé
                    if gamepad and gamepad.isConnected() and gamepad.isPressed("LB"):
                        print("[URGENCE] Bouton LB enfoncé ! Coupure immédiate.")
                        break

                    pkt = q.tryGet()
                    if pkt is None:
                        time.sleep(0.002)
                        continue

                    raw = pkt.getCvFrame()
                    
                    # Génération du masque binaire
                    mask = detect_lines(raw)

                    # 🧠 Étape IA : Préparation de l'image pour le réseau de neurones
                    mask_resized = cv2.resize(mask, (160, 120))
                    
                    # Transformation propre en tenseur PyTorch Float32 [1, 1, 120, 160]
                    img_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
                    img_tensor = img_tensor.to(device)

                    # Inférence ultra rapide (sans calcul de gradient)
                    with torch.no_grad():
                        prediction = model(img_tensor).item()

                    # Contrainte de sécurité de la sortie du servo (0.0 à 1.0)
                    servo_pos = max(0.0, min(1.0, prediction))

                    # Envoi des ordres physiques au châssis
                    vesc.set_servo(servo_pos)
                    vesc.set_duty_cycle(AUTO_DUTY)

                    # Rendu HUD si un écran est branché via SSH -X ou HDMI
                    if has_display:
                        display = cv2.resize(mask, (DISPLAY_W, DISPLAY_H))
                        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
                        status_str = f"IA CPU | Servo: {servo_pos:.2f} | Duty: {AUTO_DUTY}"
                        cv2.putText(display, status_str, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.imshow("IA Autopilot Output", display)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

        except KeyboardInterrupt:
            print("\n[INFO] Interruption reçue.")
        finally:
            print("[INFO] Nettoyage et arrêt du véhicule...")
            vesc.set_duty_cycle(0)
            vesc.set_servo(SERVO_CENTER)
            if gamepad:
                gamepad.stopBackgroundUpdates()
            cv2.destroyAllWindows()
            gc.collect()

if __name__ == "__main__":
    main()