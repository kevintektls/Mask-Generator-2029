#!/usr/bin/env python3
"""Autopilot behavioral cloning caméra + LiDAR (checkpoint de train_lidar.py)."""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
import torch
from pyvesc import VESC

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))
from vision_preprocess import make_mask_stereo, resize_for_model
from Behavioral_Cloning_Lidar import D500Reader, LIDAR_PORT, LIDAR_MAX_RANGE_M
from model.camera_lidar_model import CameraLidarBehavioralCloning

MODEL_PATH = PROJECT_DIR / "model" / "camera_lidar_model.pth"
VESC_PORT = "/dev/ttyACM0"
VESC_BAUDRATE = 115200
VESC_TIMEOUT = 1.0
SERVO_CENTER = 0.5
AUTO_DUTY = 0.045
CAM_FPS = 30
LIDAR_STALE_AFTER_S = 0.5
GAMEPAD_TYPE = None
try:
    sys.path.insert(0, "/home/robotcar/Gamepad")
    import Gamepad
    GAMEPAD_TYPE = Gamepad.Xbox360
except ImportError:
    Gamepad = None


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def build_pipeline():
    pipeline = dai.Pipeline()
    for socket, stream in ((dai.CameraBoardSocket.CAM_B, "left"),
                           (dai.CameraBoardSocket.CAM_C, "right")):
        camera = pipeline.create(dai.node.MonoCamera)
        camera.setBoardSocket(socket)
        camera.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        camera.setFps(CAM_FPS)
        output = pipeline.create(dai.node.XLinkOut)
        output.setStreamName(stream)
        output.input.setBlocking(False)
        output.input.setQueueSize(2)
        camera.out.link(output.input)
    return pipeline


def load_model(device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle caméra + LiDAR introuvable : {MODEL_PATH}. Entraîne-le avec model/train_lidar.py.")
    try:
        checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(MODEL_PATH), map_location=device)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_type") != "camera_lidar":
        raise ValueError(f"{MODEL_PATH} n'est pas un checkpoint caméra + LiDAR. Le modèle LiDAR seul ne convient pas.")
    ray_count = int(checkpoint.get("ray_count", 0))
    history_scans = int(checkpoint.get("history_scans", 0))
    history_images = int(checkpoint.get("history_images", 0))
    max_range = float(checkpoint.get("max_range", 0))
    if ray_count != 180 or history_scans != 3 or history_images != 3 or max_range <= 0:
        raise ValueError("Métadonnées caméra/LiDAR absentes ou incompatibles dans le checkpoint.")
    model = CameraLidarBehavioralCloning(ray_count, history_scans).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, ray_count, history_scans, max_range


def lidar_worker(reader, state, lock, stop_event):
    while not stop_event.is_set():
        try:
            scan = reader.read_scan()
            with lock:
                state["scan"] = scan
                state["time"] = time.monotonic()
                state["error"] = None
        except Exception as exc:
            with lock:
                state["error"] = str(exc)
            if not stop_event.is_set():
                time.sleep(0.1)


def emergency_stop(vesc):
    try:
        vesc.set_duty_cycle(0.0)
        vesc.set_servo(SERVO_CENTER)
    except Exception as exc:
        print(f"[ERROR] Arrêt VESC impossible : {exc}")


def main() -> int:
    print("[INFO] Initialisation autopilot caméra + LiDAR...")
    device = torch.device("cpu")
    torch.set_num_threads(2)
    print("[INFO] Inférence sur : cpu (2 threads)")
    try:
        model, ray_count, history_size, max_range = load_model(device)
    except Exception as exc:
        print(f"[ERROR] Chargement du modèle : {exc}")
        return 1
    print(f"[INFO] Modèle fusionné chargé : caméra stéréo + {ray_count} rayons LiDAR × {history_size}")

    gamepad = None
    if Gamepad is not None and Gamepad.available():
        gamepad = GAMEPAD_TYPE()
        gamepad.startBackgroundUpdates()
        print("[INFO] Manette connectée ; LB ou déconnexion = arrêt.")
    else:
        print("[WARNING] Aucune manette détectée ; Ctrl+C reste disponible pour arrêter.")

    try:
        vesc = VESC(serial_port=VESC_PORT, baudrate=VESC_BAUDRATE, timeout=VESC_TIMEOUT)
        lidar = D500Reader(LIDAR_PORT)
    except Exception as exc:
        if gamepad:
            gamepad.stopBackgroundUpdates()
        print(f"[ERROR] Connexion matériel impossible : {exc}")
        return 1

    lidar_state = {"scan": None, "time": 0.0, "error": None}
    lidar_lock = threading.Lock()
    stop_event = threading.Event()
    scan_thread = threading.Thread(target=lidar_worker, args=(lidar, lidar_state, lidar_lock, stop_event), daemon=True)
    scan_thread.start()
    image_history = deque(maxlen=history_size)
    scan_history = deque(maxlen=history_size)

    try:
        with vesc:
            vesc.set_servo(SERVO_CENTER)
            vesc.set_duty_cycle(0.0)
            try:
                with dai.Device(build_pipeline()) as camera:
                    q_left = camera.getOutputQueue(name="left", maxSize=2, blocking=False)
                    q_right = camera.getOutputQueue(name="right", maxSize=2, blocking=False)
                    print("[INFO] Caméras OAK-D et D500 actifs. LB ou Ctrl+C arrête la voiture.")
                    while True:
                        if gamepad and (not gamepad.isConnected() or gamepad.isPressed("LB")):
                            print("\n[STOP] Arrêt demandé par la manette.")
                            break
                        packet_left, packet_right = q_left.tryGet(), q_right.tryGet()
                        if packet_left is None or packet_right is None:
                            with lidar_lock:
                                fresh = lidar_state["scan"] is not None and time.monotonic() - lidar_state["time"] <= LIDAR_STALE_AFTER_S
                            if not fresh:
                                emergency_stop(vesc)
                            time.sleep(0.005)
                            continue

                        with lidar_lock:
                            raw_scan = lidar_state["scan"]
                            scan_age = time.monotonic() - lidar_state["time"] if raw_scan is not None else float("inf")
                            lidar_error = lidar_state["error"]
                        if raw_scan is None or scan_age > LIDAR_STALE_AFTER_S:
                            emergency_stop(vesc)
                            if lidar_error:
                                print(f"\r[WARNING] Scan LiDAR indisponible : {lidar_error}   ", end="", flush=True)
                            continue

                        mask = make_mask_stereo(packet_left.getCvFrame(), packet_right.getCvFrame())
                        image_history.append(resize_for_model(mask).astype(np.float32) / 255.0)
                        normalized_scan = np.nan_to_num(np.asarray(raw_scan, dtype=np.float32), nan=max_range, posinf=max_range, neginf=0.0)
                        if normalized_scan.size != ray_count:
                            emergency_stop(vesc)
                            raise ValueError(f"Scan reçu avec {normalized_scan.size} rayons au lieu de {ray_count}.")
                        normalized_scan = np.clip(normalized_scan, 0.0, max_range) / max_range
                        scan_history.append(normalized_scan)
                        while len(image_history) < history_size:
                            image_history.appendleft(image_history[0])
                        while len(scan_history) < history_size:
                            scan_history.appendleft(scan_history[0])
                        images = torch.from_numpy(np.stack(image_history)).unsqueeze(0).to(device)
                        scans = torch.from_numpy(np.stack(scan_history).reshape(1, -1)).to(device)
                        with torch.no_grad():
                            servo = clamp(float(model(images, scans).item()), 0.0, 1.0)
                        vesc.set_servo(servo)
                        vesc.set_duty_cycle(AUTO_DUTY)
                        print(f"\rservo={servo:.3f} duty={AUTO_DUTY:.3f} lidar={scan_age * 1000:.0f} ms   ", end="", flush=True)
            except KeyboardInterrupt:
                print("\n[INFO] Interruption clavier.")
            finally:
                print("\n[INFO] Arrêt du véhicule...")
                emergency_stop(vesc)
    except Exception as exc:
        print(f"\n[ERROR] Autopilot caméra + LiDAR : {exc}")
        emergency_stop(vesc)
        return 1
    finally:
        stop_event.set()
        lidar.close()
        if gamepad:
            gamepad.stopBackgroundUpdates()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
