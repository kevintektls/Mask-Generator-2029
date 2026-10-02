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
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Queue, Full
from vision_preprocess import make_mask_stereo
from collections import deque

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
    import serial
except ImportError:
    print("pyserial not installed. Run: pip install pyserial")
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
CROP_TOP_RATIO      = 0.35  # Légèrement descendu pour éviter les bruits lointains
ULTRA_BINARY_THRESH = 215   # Ajusté pour être un poil plus tolérant avec la fusion

# Zones de vision pour l'algo géométrique classique
ROI_FAR_TOP     = 0.15
ROI_FAR_BOT     = 0.45
ROI_NEAR_TOP    = 0.50
ROI_NEAR_BOT    = 0.85

LANE_WIDTH_PX    = 340  
LANE_WIDTH_MIN   = 160
SMOOTHING_ALPHA  = 0.25  
frame_buffer = deque(maxlen=3)

# VESC Connection
VESC_PORT            = '/dev/ttyACM0'
VESC_BAUDRATE        = 115200
VESC_TIMEOUT         = 1.0
VESC_CONNECT_RETRIES = 8
VESC_CONNECT_SETTLE  = 1.0

# LiDAR LDROBOT D500 / STL-19P
LIDAR_PORT = "/dev/ttyTHS1"
LIDAR_BAUDRATE = 230400
LIDAR_MAX_RANGE_M = 12.0
LIDAR_BINS = 180
LIDAR_STALE_AFTER_S = 0.5
LIDAR_PACKET_SIZE = 47

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
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
IMAGES_DIR  = DATASET_DIR / "images"
CSV_FILE    = DATASET_DIR / "driving_log.csv"
CSV_HEADER  = ["timestamp", "image_path", "servo", "duty", "lidar_timestamp", "lidar"]


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


def lidar_crc8(data: bytes) -> int:
    """CRC-8 LDROBOT utilisé par les paquets STL-19P."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (((crc << 1) ^ 0x4D) if crc & 0x80 else (crc << 1)) & 0xFF
    return crc


class D500Reader:
    """Lit des scans avant de 180 bins, de -90° à +90°, en mètres."""

    def __init__(self, port: str):
        self.serial = serial.Serial(
            port=port,
            baudrate=LIDAR_BAUDRATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.2,
        )
        self.buffer = bytearray()
        self.points = deque()
        self.pending_point = None

    def close(self):
        self.serial.close()

    def _read_packet(self) -> bytes:
        while not stop_event.is_set():
            waiting = self.serial.in_waiting
            chunk = self.serial.read(waiting if waiting else 1)
            if chunk:
                self.buffer.extend(chunk)
            while self.buffer:
                if self.buffer[0] != 0x54:
                    del self.buffer[0]
                    continue
                if len(self.buffer) < 2:
                    break
                if self.buffer[1] != 0x2C:
                    del self.buffer[0]
                    continue
                if len(self.buffer) < LIDAR_PACKET_SIZE:
                    break
                packet = bytes(self.buffer[:LIDAR_PACKET_SIZE])
                if lidar_crc8(packet[:-1]) != packet[-1]:
                    del self.buffer[0]
                    continue
                del self.buffer[:LIDAR_PACKET_SIZE]
                return packet
        raise RuntimeError("Arrêt demandé pendant la lecture du LiDAR.")

    @staticmethod
    def _decode_points(packet: bytes):
        start = int.from_bytes(packet[4:6], "little") / 100.0
        end = int.from_bytes(packet[42:44], "little") / 100.0
        delta = (end - start) % 360.0
        for i in range(12):
            offset = 6 + i * 3
            distance_m = int.from_bytes(packet[offset:offset + 2], "little") / 1000.0
            angle = (start + delta * i / 11.0) % 360.0
            yield angle, distance_m

    def _next_point(self):
        while not self.points:
            self.points.extend(self._decode_points(self._read_packet()))
        return self.points.popleft()

    def read_scan(self) -> list[float]:
        ranges = [LIDAR_MAX_RANGE_M] * LIDAR_BINS
        last_angle = None
        point_count = 0
        while not stop_event.is_set():
            angle, distance_m = self.pending_point or self._next_point()
            self.pending_point = None
            if last_angle is not None and angle < last_angle - 180.0:
                if point_count >= 100:
                    self.pending_point = (angle, distance_m)
                    return ranges
                ranges = [LIDAR_MAX_RANGE_M] * LIDAR_BINS
                point_count = 0

            signed_angle = angle if angle <= 180.0 else angle - 360.0
            if -90.0 <= signed_angle <= 90.0 and 0.0 < distance_m <= LIDAR_MAX_RANGE_M:
                index = min(LIDAR_BINS - 1, int(signed_angle + 90.0))
                ranges[index] = min(ranges[index], distance_m)
                point_count += 1
            last_angle = angle
        raise RuntimeError("Arrêt demandé pendant la lecture du LiDAR.")


def lidar_worker():
    global latest_lidar, lidar_error
    reader = None
    try:
        reader = D500Reader(LIDAR_PORT)
        print(f"[LiDAR] D500 connecté sur {LIDAR_PORT} @ {LIDAR_BAUDRATE}")
        while not stop_event.is_set():
            ranges = reader.read_scan()
            stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            with lidar_lock:
                latest_lidar = (ranges, stamp, time.monotonic())
    except Exception as exc:
        if not stop_event.is_set():
            lidar_error = str(exc)
            print(f"[LiDAR] Erreur : {exc}")
            stop_event.set()
    finally:
        if reader is not None:
            reader.close()


# ── Vision Stéréo Ultra-Binaire Nettoyée ──────────────────────────────────────




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
    file_exists = CSV_FILE.exists() and CSV_FILE.stat().st_size > 0
    if file_exists:
        with CSV_FILE.open("r", newline="", encoding="utf-8") as existing:
            header = next(csv.reader(existing), [])
        if header != CSV_HEADER:
            raise ValueError(
                f"Le CSV {CSV_FILE} a un ancien format. Renomme-le ou déplace-le "
                "avant de commencer un dataset avec LiDAR."
            )
    
    csv_file_handle = open(CSV_FILE, mode="a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file_handle)
    
    if not file_exists:
        csv_writer.writerow(CSV_HEADER)
        csv_file_handle.flush()


# ── Serveur Vidéo HTTP ────────────────────────────────────────────────────────
latest_frame: np.ndarray | None = None
frame_lock = threading.Lock()
stop_event = threading.Event()
lidar_lock = threading.Lock()
latest_lidar = None  # (ranges_m, timestamp_utc, received_monotonic)
lidar_error = None


_default_thread_excepthook = threading.excepthook


def _thread_excepthook(args):
    """Turn the Gamepad library's uncaught unplug error into a safe shutdown."""
    error_text = str(args.exc_value).lower()
    if isinstance(args.exc_value, OSError) and "gamepad" in error_text and "disconnect" in error_text:
        print("\n[WARN] Manette déconnectée. Arrêt sécurisé du robot.")
        stop_event.set()
        return
    _default_thread_excepthook(args)


threading.excepthook = _thread_excepthook


def get_frame_or_stop(queue):
    """Poll DepthAI without blocking forever if another thread requests stop."""
    while not stop_event.is_set():
        packet = queue.tryGet()
        if packet is not None:
            return packet
        time.sleep(0.005)
    return None


def watch_gamepad_connection(gamepad):
    """Request a safe exit if Gamepad reports that its device disappeared."""
    while not stop_event.wait(0.2):
        try:
            connected = gamepad.isConnected()
        except Exception:
            connected = False
        if not connected:
            print("\n[WARN] Manette déconnectée. Arrêt sécurisé du robot.")
            stop_event.set()
            return

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
        item = write_queue.get()
        try:
            if item is None:
                return
            img_path, data, row = item
            if not cv2.imwrite(str(img_path), data):
                print(f"\n[WARN] Image impossible à écrire : {img_path}")
                continue
            with record_lock:
                csv_writer.writerow(row)
                csv_file_handle.flush()
        except Exception as exc:
            print(f"\n[WARN] Écriture du dataset impossible : {exc}")
        finally:
            write_queue.task_done()

# ── Boucle Principale de Contrôle ─────────────────────────────────────────────

def main():
    global latest_frame, is_recording, csv_writer, csv_file_handle
    threading.Thread(target=start_http_server, daemon=True).start()
    
    if not Gamepad.available():
        while not Gamepad.available(): time.sleep(0.5)
    gamepad = GAMEPAD_TYPE()
    gamepad.startBackgroundUpdates()
    threading.Thread(
        target=watch_gamepad_connection,
        args=(gamepad,),
        daemon=True,
        name="gamepad-watchdog",
    ).start()

    init_dataset()
    vesc = vesc_connect()
    pipeline = build_pipeline()

    count, t0, fps_val = 0, time.monotonic(), 0.0
    autonomous_mode = False  
    prev_lb = False
    prev_a  = False
    lb_pressed_since = None
    has_display = bool(os.environ.get("DISPLAY"))

    writer_thread = threading.Thread(target=disk_writer, daemon=True)
    writer_thread.start()
    lidar_thread = threading.Thread(target=lidar_worker, daemon=True, name="d500-reader")
    lidar_thread.start()
    print(f"[LiDAR] Connexion au D500 sur {LIDAR_PORT} @ {LIDAR_BAUDRATE} bauds…")
    with vesc:
        vesc.set_servo(SERVO_CENTER)
        vesc.set_duty_cycle(0)
        print("\n=== SYSTEM DATA LOGGER READY ===")
        print(" -> Mode courant : 🎮 MANUEL")
        print(" -> Bouton A   : ÉCRIRE / STOPPER le Dataset [REC]")
        print(" -> Appui bref LB : basculer en mode autonome géométrique")
        print(" -> Maintenir LB 2 s : arrêter le programme\n")
        
        try:
            with dai.Device(pipeline) as device:
                # Récupération des deux files d'attente
                q_left = device.getOutputQueue(name="left", maxSize=2, blocking=False)
                q_right = device.getOutputQueue(name="right", maxSize=2, blocking=False)

                while not stop_event.is_set() and gamepad.isConnected():
                    lb_now = gamepad.isPressed("LB")
                    a_now  = gamepad.isPressed("A")

                    if lb_now:
                        if lb_pressed_since is None:
                            lb_pressed_since = time.monotonic()
                        elif time.monotonic() - lb_pressed_since >= 2.0:
                            print("\n[STOP] LB maintenu 2 s : arrêt du programme.")
                            stop_event.set()
                            break
                    else:
                        lb_pressed_since = None
                    
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

                    # Poll au lieu d'attendre sans fin, pour que déconnexion manette/LiDAR
                    # puisse interrompre la boucle et couper les commandes du VESC.
                    pkt_left = get_frame_or_stop(q_left)
                    if pkt_left is None:
                        break
                    pkt_right = get_frame_or_stop(q_right)
                    if pkt_right is None:
                        break

                    raw_left = pkt_left.getCvFrame()
                    raw_right = pkt_right.getCvFrame()

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
                    mask = make_mask_stereo(raw_left, raw_right)
                    # ── AJOUT : Gestion de l'historique temporel ────────────────────────────────────
                    if len(frame_buffer) == 0:
                        # Au démarrage, on remplit le buffer avec 3 copies du premier masque
                        for _ in range(3):
                            frame_buffer.append(mask.copy())
                    else:
                        frame_buffer.append(mask.copy())
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

                    if is_recording:
                        with lidar_lock:
                            lidar_snapshot = latest_lidar
                        if (lidar_snapshot is None or
                                time.monotonic() - lidar_snapshot[2] > LIDAR_STALE_AFTER_S):
                            print("\n[WARN] Aucun scan LiDAR récent : image ignorée pour garder le CSV synchronisé.")
                        else:
                            ranges, lidar_timestamp, _received_at = lidar_snapshot
                            timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                            image_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                            img_name = f"line_{image_stamp}.png"
                            img_path = IMAGES_DIR / img_name
                            row = [
                                timestamp,
                                f"images/{img_name}",
                                f"{servo_pos:.4f}",
                                f"{duty:.4f}",
                                lidar_timestamp,
                                json.dumps(ranges, separators=(",", ":")),
                            ]
                            try:
                                write_queue.put_nowait((img_path, mask.copy(), row))
                            except Full:
                                print("\n[WARN] File d'écriture pleine : image ignorée.")

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
            stop_event.set()
            write_queue.put(None)
            write_queue.join()
            writer_thread.join(timeout=2.0)
            lidar_thread.join(timeout=1.0)
            with record_lock:
                if csv_file_handle: csv_file_handle.close()
            try:
                vesc.set_duty_cycle(0)
                vesc.set_servo(SERVO_CENTER)
            except Exception as exc:
                print(f"[WARN] Impossible d'envoyer l'arrêt au VESC : {exc}")
            try:
                gamepad.stopBackgroundUpdates()
            except Exception:
                pass
            cv2.destroyAllWindows()
            gc.collect()

if __name__ == "__main__":
    main()
