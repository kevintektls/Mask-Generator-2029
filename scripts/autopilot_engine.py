#!/usr/bin/env python3
"""
Robot Car — Autopilot IA Behavioral Cloning (Version Ultra-Optimisée)
Plateforme : Jetson Nano 4Go + OAK-D Lite + VESC + TensorRT

Optimisations appliquées :
1. Historique temporel (Frame Stacking) sur 3 frames pour capter la trajectoire.
2. Inférence matérielle pure via TensorRT (.engine en FP16) sans surcoût PyTorch.
"""

from __future__ import annotations

import os
import sys
import time
import gc
from collections import deque

import cv2
import numpy as np
import depthai as dai
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Initialise automatiquement le contexte CUDA
from pyvesc import VESC

sys.path.insert(0, "/home/robotcar/Gamepad")
try:
    import Gamepad
except ImportError:
    print("[WARNING] Gamepad lib introuvable.")

from vision_preprocess import make_mask_stereo, resize_for_model, CROP_TOP_RATIO


# ── CONFIG ────────────────────────────────────────────────────────────────────

DISPLAY_W = 640
DISPLAY_H = 480
CAM_FPS = 30

# Dimensions attendues par ton modèle d'IA (à ajuster selon ton modèle)
MODEL_H = 120 
MODEL_WIDTH = 160

VESC_PORT = "/dev/ttyACM0"
VESC_BAUDRATE = 115200
VESC_TIMEOUT = 1.0

SERVO_CENTER = 0.5
AUTO_DUTY = 0.040

# Ton fichier compilé avec trtexec (.engine)
ENGINE_PATH = "robot_model.engine"

GAMEPAD_TYPE = Gamepad.Xbox360 if 'Gamepad' in sys.modules else None


# ── TENSORRT TOOLKIT ──────────────────────────────────────────────────────────

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTInferenceWrapper:
    """Gère le cycle de vie, les buffers GPU et l'exécution du moteur TensorRT."""
    def __init__(self, engine_path: str, channels=3, h=120, w=160):
        if not os.path.exists(engine_path):
            print(f"[ERROR] Moteur TensorRT introuvable : {engine_path}")
            sys.exit(1)
            
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        # Allocation des buffers mémoires (Host RAM & Device GPU)
        self.h_input = cuda.pagelocked_empty(channels * h * w, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(1, dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.stream = cuda.Stream()

    def infer(self, stacked_frames: np.ndarray) -> float:
        # Normalisation identique au dataset et aplatissement (ravel)
        normalized = (stacked_frames.astype(np.float32) / 255.0).ravel()
        
        # Copie RAM vers GPU
        np.copyto(self.h_input, normalized)
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        
        # Inférence sur les Coeurs CUDA/Tensor
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)], 
            stream_handle=self.stream.handle
        )
        
        # Récupération GPU vers RAM
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        
        return float(self.h_output[0])


# ── OUTILS VÉHICULE ───────────────────────────────────────────────────────────

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
    print("[INFO] Initialisation autopilot IA avec accélération matérielle...")

    # Chargement du wrapper TensorRT (Canaux=3, H=120, W=160)
    trt_agent = TRTInferenceWrapper(ENGINE_PATH, channels=3, h=MODEL_H, w=MODEL_WIDTH)
    print("[INFO] Moteur TensorRT (.engine) initialisé avec succès.")

    if GAMEPAD_TYPE and Gamepad.available():
        gamepad = GAMEPAD_TYPE()
        gamepad.startBackgroundUpdates()
        print("[INFO] Manette connectée. LB = arrêt urgence.")
    else:
        gamepad = None
        print("[WARNING] Aucune manette détectée ou librairie absente.")

    try:
        vesc = VESC(serial_port=VESC_PORT, baudrate=VESC_BAUDRATE, timeout=VESC_TIMEOUT)
        print("[INFO] VESC connecté.")
    except Exception as e:
        print(f"[ERROR] VESC impossible à joindre : {e}")
        sys.exit(1)

    pipeline = build_pipeline()
    has_display = bool(os.environ.get("DISPLAY"))

    # Initialisation du buffer d'historique (3 masques max)
    frame_buffer = deque(maxlen=3)

    print("\n=== AUTOPILOTE IA TENSORRT PRÊT ===")
    print("LB ou CTRL+C = arrêt immédiat\n")

    with vesc:
        vesc.set_servo(SERVO_CENTER)
        vesc.set_duty_cycle(0)
        time.sleep(1.0)

        try:
            with dai.Device(pipeline) as device_dai:
                q_left = device_dai.getOutputQueue(name="left", maxSize=2, blocking=False)
                q_right = device_dai.getOutputQueue(name="right", maxSize=2, blocking=False)

                while True:
                    # Sécurité coupure manette
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

                    # Prétraitement géométrique stéréo identique au Dataset
                    mask = make_mask_stereo(raw_left, raw_right)
                    mask_resized = resize_for_model(mask)

                    # ── GESTION DE L'HISTORIQUE TEMPOREL (STACKING) ────────────────
                    if len(frame_buffer) == 0:
                        # Remplissage initial si le buffer est vide
                        for _ in range(3):
                            frame_buffer.append(mask_resized.copy())
                    else:
                        frame_buffer.append(mask_resized.copy())

                    # Transformation du buffer en un bloc NumPy (3, H, W)
                    stacked_input = np.stack(list(frame_buffer), axis=0)

                    # ── INFÉRENCE TENSORRT CHRONOMÉTRÉE ───────────────────────────
                    prediction = trt_agent.infer(stacked_input)

                    # Application des commandes moteurs
                    servo_pos = clamp(prediction, 0.0, 1.0)
                    vesc.set_servo(servo_pos)
                    vesc.set_duty_cycle(AUTO_DUTY)

                    # Rendu HUD
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

                        text = f"IA TRT | Servo: {servo_pos:.3f} | Duty: {AUTO_DUTY:.3f}"
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