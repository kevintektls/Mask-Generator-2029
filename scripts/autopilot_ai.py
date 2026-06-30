#!/usr/bin/env python3
"""
Robot Car — Autopilot IA Behavioral Cloning
Plateforme : Jetson Nano 4Go + OAK-D Lite + VESC

IMPORTANT :
Ce fichier utilise le même prétraitement stéréo que le dataset.
"""

from __future__ import annotations

import os
import sys
import time
import gc

import cv2
import numpy as np
import torch
import depthai as dai
from pyvesc import VESC

sys.path.insert(0, "/home/robotcar/Gamepad")
import Gamepad

from model_def import BehavioralCloningCNN
from vision_preprocess import make_mask_stereo, resize_for_model, CROP_TOP_RATIO
from collections import deque


# ── CONFIG ────────────────────────────────────────────────────────────────────

DISPLAY_W = 640
DISPLAY_H = 480
CAM_FPS = 30

VESC_PORT = "/dev/ttyACM0"
VESC_BAUDRATE = 115200
VESC_TIMEOUT = 1.0

SERVO_CENTER = 0.5
AUTO_DUTY = 0.045
MODEL_PATH = "../model/pilot_model.pth"

GAMEPAD_TYPE = Gamepad.Xbox360


# ── OUTILS ────────────────────────────────────────────────────────────────────

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
        # Compatibilité anciennes versions PyTorch
        state = torch.load(MODEL_PATH, map_location=device)

    model.load_state_dict(state)
    model.eval()

    return model


def emergency_stop(vesc):
    try:
        vesc.set_duty_cycle(0)
        vesc.set_servo(SERVO_CENTER)
        time.sleep(0.05)
        vesc.set_duty_cycle(0)
    except Exception as e:
        print(f"[ERROR] Emergency stop failed: {e}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("[INFO] Initialisation autopilot IA...")

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
        print("[INFO] Manette connectée. LB = arrêt urgence.")
    else:
        gamepad = None
        print("[WARNING] Aucune manette détectée.")

    try:
        vesc = VESC(serial_port=VESC_PORT, baudrate=VESC_BAUDRATE, timeout=VESC_TIMEOUT)
        print("[INFO] VESC connecté.")
    except Exception as e:
        print(f"[ERROR] VESC impossible à joindre : {e}")
        sys.exit(1)

    pipeline = build_pipeline()
    has_display = bool(os.environ.get("DISPLAY"))

    print("\n=== AUTOPILOTE IA PRÊT ===")
    print("LB ou CTRL+C = arrêt immédiat\n")

    with vesc:
        vesc.set_servo(SERVO_CENTER)
        vesc.set_duty_cycle(0)
        time.sleep(1.0)

        try:
            frame_buffer = deque(maxlen=3)
            with dai.Device(pipeline) as device_dai:
                q_left = device_dai.getOutputQueue(name="left", maxSize=2, blocking=False)
                q_right = device_dai.getOutputQueue(name="right", maxSize=2, blocking=False)

                while True:
                    if gamepad and gamepad.isConnected() and gamepad.isPressed("LB"):
                        print("[URGENCE] LB pressé. Coupure immédiate.")
                        emergency_stop(vesc)
                        break
                
                    pkt_left = q_left.tryGet()
                    pkt_right = q_right.tryGet()

                    if pkt_left is None or pkt_right is None:
                        time.sleep(0.002)
                        continue

                    raw_left = pkt_left.getCvFrame()
                    raw_right = pkt_right.getCvFrame()

                    mask = make_mask_stereo(raw_left, raw_right)
                    mask_resized = resize_for_model(mask)

                    # ── AJOUT : Gestion de la file d'attente temporelle en direct ──
                    if len(frame_buffer) == 0:
                        for _ in range(3):
                            frame_buffer.append(mask_resized.copy())
                    else:
                        frame_buffer.append(mask_resized.copy())

                    # On convertit le buffer (3 masques) en un array numpy de dimension (3, H, W)
                    stacked_input = np.stack(list(frame_buffer), axis=0)

                    # Transformation pour PyTorch : ajout de la dimension Batch -> (1, 3, H, W)
                    img_tensor = (
                        torch.from_numpy(stacked_input)
                        .float()
                        .unsqueeze(0) 
                        / 255.0
                    ).to(device)

                    with torch.no_grad():
                        prediction = model(img_tensor).item()

                    servo_pos = clamp(prediction, 0.0, 1.0)

                    vesc.set_servo(servo_pos)
                    vesc.set_duty_cycle(AUTO_DUTY)

                    if has_display:
                        display = cv2.resize(mask, (DISPLAY_W, DISPLAY_H))
                        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

                        cv2.line(
                            display,
                            (0, int(DISPLAY_H * CROP_TOP_RATIO)),
                            (DISPLAY_W, int(DISPLAY_H * CROP_TOP_RATIO)),
                            (0, 0, 150),
                            1,
                        )

                        text = f"IA | Servo: {servo_pos:.3f} | Duty: {AUTO_DUTY:.3f}"
                        cv2.putText(
                            display,
                            text,
                            (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

                        cv2.imshow("IA Autopilot Mask", display)

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

        except KeyboardInterrupt:
            print("\n[INFO] Interruption clavier.")
        finally:
            print("[INFO] Arrêt véhicule...")
            emergency_stop(vesc)

            if gamepad:
                gamepad.stopBackgroundUpdates()

            cv2.destroyAllWindows()
            gc.collect()


if __name__ == "__main__":
    main()
