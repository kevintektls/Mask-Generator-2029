#!/usr/bin/env python3
"""Conduite manuelle et enregistrement caméra stéréo + LiDAR D500.

Le CSV produit contient une ligne par masque enregistré :
image_path,servo,duty,lidar_timestamp,lidar
``lidar`` contient les 180 distances en mètres au format JSON. Le format est
également lisible par le collecteur/entraîneur LiDAR du dépôt.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Full, Queue

import depthai as dai
import numpy as np
import cv2
import serial
from pyvesc import VESC

PROJECT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_DIR.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, "/home/robotcar/Gamepad")

try:
    import Gamepad
except ImportError as exc:
    raise SystemExit("Gamepad introuvable. Ajoute sa bibliothèque au PYTHONPATH.") from exc

try:
    from vision_preprocess import make_mask_stereo
except ImportError as exc:
    raise SystemExit(f"Import vision_preprocess impossible depuis {SCRIPTS_DIR}") from exc


# Matériel et acquisition
CAM_FPS = 30
LIDAR_PORT = "/dev/ttyTHS1"
LIDAR_BAUDRATE = 230400
LIDAR_MAX_RANGE_M = 12.0
LIDAR_BINS = 180
VESC_PORT = "/dev/ttyACM0"
VESC_BAUDRATE = 115200
VESC_TIMEOUT = 1.0
VESC_CONNECT_RETRIES = 8
VESC_CONNECT_SETTLE = 1.0
GAMEPAD_TYPE = Gamepad.Xbox360
AXIS_FORWARD = "RT"
AXIS_BACKWARD = "LT"
AXIS_STEERING = "LEFT-X"
DEADZONE = 0.08

# Commandes moteur et mode autonome géométrique conservés du script original
SERVO_CENTER = 0.5
SERVO_RANGE = 0.48
AUTO_DUTY = 0.10
AUTO_DUTY_MIN = 0.010
TURN_SLOWDOWN = 0.90
ROI_FAR_TOP, ROI_FAR_BOT = 0.15, 0.45
ROI_NEAR_TOP, ROI_NEAR_BOT = 0.50, 0.85
LANE_WIDTH_PX = 340
LANE_WIDTH_MIN = 160
HTTP_PORT = 5000

DATASET_DIR = PROJECT_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
CSV_FILE = DATASET_DIR / "driving_log.csv"
CSV_HEADER = ["image_path", "servo", "duty", "lidar_timestamp", "lidar"]

stop_event = threading.Event()
frame_lock = threading.Lock()
lidar_lock = threading.Lock()
latest_frame: np.ndarray | None = None
latest_lidar: tuple[list[float], str, float] | None = None
lidar_error: str | None = None
frame_buffer: deque[np.ndarray] = deque(maxlen=3)
prev_servo_pos = SERVO_CENTER
write_queue: Queue = Queue(maxsize=60)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def apply_deadzone(value: float) -> float:
    if abs(value) < DEADZONE:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - DEADZONE) / (1.0 - DEADZONE)


def _get_band_center(mask: np.ndarray, y_top: int, y_bot: int):
    _height, width = mask.shape
    mid = width // 2
    band = mask[y_top:y_bot, :]
    hist = np.sum(band > 0, axis=0).astype(np.float32)
    if np.max(hist) == 0:
        return None, -1, -1
    hist = np.convolve(hist, np.ones(21, dtype=np.float32) / 21, mode="same")
    threshold = (y_bot - y_top) * 0.20
    left_hits = np.where(hist[:mid][::-1] >= threshold)[0]
    right_hits = np.where(hist[mid:] >= threshold)[0]
    left_x = mid - 1 - int(left_hits[0]) if left_hits.size else -1
    right_x = mid + int(right_hits[0]) if right_hits.size else -1
    if left_x >= 0 and right_x >= 0 and right_x - left_x < LANE_WIDTH_MIN:
        if hist[left_x] >= hist[right_x]:
            right_x = -1
        else:
            left_x = -1
    if left_x >= 0 and right_x >= 0:
        return (left_x + right_x) / 2.0, left_x, right_x
    if left_x >= 0:
        return left_x + LANE_WIDTH_PX / 2.0, left_x, -1
    if right_x >= 0:
        return right_x - LANE_WIDTH_PX / 2.0, -1, right_x
    return None, -1, -1


def compute_steering(mask: np.ndarray):
    global prev_servo_pos
    height, width = mask.shape
    mid = width // 2
    near, left_x, right_x = _get_band_center(
        mask, int(height * ROI_NEAR_TOP), int(height * ROI_NEAR_BOT)
    )
    far, _, _ = _get_band_center(
        mask, int(height * ROI_FAR_TOP), int(height * ROI_FAR_BOT)
    )
    if near is None and far is None:
        if abs(prev_servo_pos - SERVO_CENTER) > 0.15:
            return prev_servo_pos, False, mid, left_x, right_x
        return SERVO_CENTER, False, mid, left_x, right_x
    if near is not None and far is not None:
        near_error, far_error = abs(near - mid) / mid, abs(far - mid) / mid
        if far_error > 0.30 or near_error > 0.30:
            target = far if far_error > near_error else near
        else:
            target = 0.60 * near + 0.40 * far
    else:
        target = far if near is None else near
    error = (target - mid) / mid
    shaped_error = math.copysign(abs(error) ** 1.1, error)
    if abs(error) > 0.35:
        shaped_error = math.copysign(1.0, error)
    raw = clamp(SERVO_CENTER + shaped_error * SERVO_RANGE, 0.0, 1.0)
    alpha = 0.45 if abs(raw - SERVO_CENTER) > abs(prev_servo_pos - SERVO_CENTER) else 0.12
    prev_servo_pos = clamp((1.0 - alpha) * prev_servo_pos + alpha * raw, 0.0, 1.0)
    return prev_servo_pos, True, int(target), left_x, right_x


def crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (((crc << 1) ^ 0x4D) if crc & 0x80 else (crc << 1)) & 0xFF
    return crc


class D500Reader:
    """Décode le D500 et regroupe les retours avant en 180 bins de 1 degré."""

    PACKET_SIZE = 47

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
        self.points: deque[tuple[float, float]] = deque()
        self.pending_point: tuple[float, float] | None = None

    def close(self):
        self.serial.close()

    def _read_packet(self) -> bytes:
        while not stop_event.is_set():
            waiting = self.serial.in_waiting
            self.buffer.extend(self.serial.read(waiting if waiting else 1))
            while self.buffer:
                if self.buffer[0] != 0x54:
                    del self.buffer[0]
                    continue
                if len(self.buffer) < 2:
                    break
                if self.buffer[1] != 0x2C:
                    del self.buffer[0]
                    continue
                if len(self.buffer) < self.PACKET_SIZE:
                    break
                packet = bytes(self.buffer[: self.PACKET_SIZE])
                if crc8(packet[:-1]) != packet[-1]:
                    del self.buffer[0]
                    continue
                del self.buffer[: self.PACKET_SIZE]
                return packet
        raise RuntimeError("arrêt demandé pendant la lecture LiDAR")

    @staticmethod
    def _decode(packet: bytes):
        start = int.from_bytes(packet[4:6], "little") / 100.0
        end = int.from_bytes(packet[42:44], "little") / 100.0
        delta = (end - start) % 360.0
        for i in range(12):
            offset = 6 + i * 3
            distance_m = int.from_bytes(packet[offset : offset + 2], "little") / 1000.0
            angle = (start + delta * i / 11.0) % 360.0
            yield angle, distance_m

    def _next_point(self):
        while not self.points:
            self.points.extend(self._decode(self._read_packet()))
        return self.points.popleft()

    def read_scan(self) -> list[float]:
        ranges = [LIDAR_MAX_RANGE_M] * LIDAR_BINS
        last_angle = None
        point_count = 0
        while not stop_event.is_set():
            angle, distance = self.pending_point or self._next_point()
            self.pending_point = None
            if last_angle is not None and angle < last_angle - 180.0:
                if point_count >= 100:
                    self.pending_point = (angle, distance)
                    return ranges
                ranges = [LIDAR_MAX_RANGE_M] * LIDAR_BINS
                point_count = 0
            signed = angle if angle <= 180.0 else angle - 360.0
            if -90.0 <= signed <= 90.0 and 0.0 < distance <= LIDAR_MAX_RANGE_M:
                index = min(LIDAR_BINS - 1, int(signed + 90.0))
                ranges[index] = min(ranges[index], distance)
                point_count += 1
            last_angle = angle
        raise RuntimeError("arrêt demandé pendant la lecture LiDAR")


def lidar_worker():
    global latest_lidar, lidar_error
    reader = None
    try:
        reader = D500Reader(LIDAR_PORT)
        print(f"[LiDAR] D500 connecté sur {LIDAR_PORT} @ {LIDAR_BAUDRATE}")
        while not stop_event.is_set():
            scan = reader.read_scan()
            stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            with lidar_lock:
                latest_lidar = (scan, stamp, time.monotonic())
    except Exception as exc:
        if not stop_event.is_set():
            lidar_error = str(exc)
            print(f"[LiDAR] erreur : {exc}")
            stop_event.set()
    finally:
        if reader is not None:
            reader.close()


def init_dataset():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    exists = CSV_FILE.exists() and CSV_FILE.stat().st_size > 0
    if exists:
        with CSV_FILE.open("r", newline="", encoding="utf-8") as existing:
            header = next(csv.reader(existing), [])
        if header != CSV_HEADER:
            raise ValueError(
                f"{CSV_FILE} existe avec un autre format CSV. Choisis un nouveau dossier dataset."
            )
    handle = CSV_FILE.open("a", newline="", encoding="utf-8", buffering=1)
    writer = csv.writer(handle)
    if not exists:
        writer.writerow(CSV_HEADER)
        handle.flush()
    return handle, writer


def disk_writer(csv_writer, csv_handle):
    while True:
        item = write_queue.get()
        try:
            if item is None:
                return
            image_path, image, row = item
            if not cv2.imwrite(str(image_path), image):
                print(f"\n[WARN] impossible d'écrire {image_path}")
                continue
            csv_writer.writerow(row)
            csv_handle.flush()
        except Exception as exc:
            print(f"\n[WARN] écriture dataset échouée : {exc}")
        finally:
            write_queue.task_done()


class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path == "/":
            page = b"<!doctype html><meta charset='utf-8'><title>Robot camera</title><body style='margin:0;background:#111'><img style='width:100%;height:100%;object-fit:contain' src='/stream'></body>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return
        if self.path != "/stream":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while not stop_event.is_set():
                with frame_lock:
                    frame = latest_frame
                if frame is None:
                    time.sleep(0.02)
                    continue
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                if not ok:
                    continue
                data = jpg.tobytes()
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(data)).encode()
                    + b"\r\n\r\n"
                    + data
                    + b"\r\n"
                )
        except (BrokenPipeError, ConnectionResetError):
            pass


def build_pipeline():
    pipeline = dai.Pipeline()
    for socket, name in ((dai.CameraBoardSocket.CAM_B, "left"), (dai.CameraBoardSocket.CAM_C, "right")):
        camera = pipeline.create(dai.node.MonoCamera)
        camera.setBoardSocket(socket)
        camera.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        camera.setFps(CAM_FPS)
        output = pipeline.create(dai.node.XLinkOut)
        output.setStreamName(name)
        output.input.setBlocking(False)
        output.input.setQueueSize(2)
        camera.out.link(output.input)
    return pipeline


def vesc_connect():
    last_error = None
    for attempt in range(VESC_CONNECT_RETRIES):
        try:
            vesc = VESC(serial_port=VESC_PORT, baudrate=VESC_BAUDRATE, timeout=VESC_TIMEOUT)
            print(f"[VESC] connecté sur {VESC_PORT}")
            return vesc
        except Exception as exc:
            last_error = exc
            print(f"[VESC] tentative {attempt + 1}/{VESC_CONNECT_RETRIES} échouée : {exc}")
            time.sleep(VESC_CONNECT_SETTLE)
    raise RuntimeError(f"VESC introuvable : {last_error}")


def main():
    global latest_frame
    try:
        csv_handle, csv_writer = init_dataset()
    except (OSError, ValueError) as exc:
        print(f"[ERROR] dataset : {exc}")
        return 1

    writer_thread = threading.Thread(
        target=disk_writer, args=(csv_writer, csv_handle), daemon=True, name="dataset-writer"
    )
    writer_thread.start()
    lidar_thread = threading.Thread(target=lidar_worker, daemon=True, name="d500-reader")
    lidar_thread.start()
    server = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), MJPEGHandler)
    threading.Thread(target=server.serve_forever, daemon=True, name="mjpeg-preview").start()
    print(f"[INFO] Aperçu vidéo : http://<adresse-du-robot>:{HTTP_PORT}/")

    gamepad = None
    vesc = None
    try:
        if not Gamepad.available():
            print("[INFO] en attente de la manette…")
            while not Gamepad.available() and not stop_event.is_set():
                time.sleep(0.5)
        if stop_event.is_set():
            return 1
        gamepad = GAMEPAD_TYPE()
        gamepad.startBackgroundUpdates()
        vesc = vesc_connect()

        print("[INFO] en attente du premier tour LiDAR valide…")
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not stop_event.is_set():
            with lidar_lock:
                scan_ready = latest_lidar is not None
            if scan_ready:
                break
            if lidar_error:
                raise RuntimeError(f"LiDAR indisponible : {lidar_error}")
            time.sleep(0.1)
        else:
            raise RuntimeError("aucun scan reçu du D500 après 30 secondes")

        vesc.set_servo(SERVO_CENTER)
        vesc.set_duty_cycle(0.0)
        print("RT : avancer | LT : reculer | joystick gauche : tourner | A : REC/pause | LB : mode manuel/auto")
        pipeline = build_pipeline()
        frame_count = 0
        started = time.monotonic()
        fps = 0.0
        recording = False
        autonomous = False
        previous_lb = previous_a = False
        has_display = bool(__import__("os").environ.get("DISPLAY"))

        with dai.Device(pipeline) as device:
            left_queue = device.getOutputQueue(name="left", maxSize=2, blocking=False)
            right_queue = device.getOutputQueue(name="right", maxSize=2, blocking=False)
            while not stop_event.is_set() and gamepad.isConnected():
                lb_now = gamepad.isPressed("LB")
                a_now = gamepad.isPressed("A")
                if lb_now and not previous_lb:
                    autonomous = not autonomous
                    if autonomous:
                        recording = False
                    print("[MODE] " + ("AUTO géométrique" if autonomous else "conduite manuelle"))
                previous_lb = lb_now
                if a_now and not previous_a and not autonomous:
                    recording = not recording
                    print("[DATASET] " + ("REC" if recording else "pause"))
                previous_a = a_now

                raw_left = left_queue.get().getCvFrame()
                raw_right = right_queue.get().getCvFrame()
                mask = make_mask_stereo(raw_left, raw_right)
                frame_buffer.append(mask.copy())
                frame_count += 1
                elapsed = time.monotonic() - started
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count, started = 0, time.monotonic()

                if autonomous:
                    servo, line_found, target_x, _left, _right = compute_steering(mask)
                    turn = min(1.0, abs(servo - SERVO_CENTER) / SERVO_RANGE)
                    duty = max(AUTO_DUTY_MIN, AUTO_DUTY * (1.0 - TURN_SLOWDOWN * turn)) if line_found else AUTO_DUTY_MIN
                else:
                    throttle = clamp(gamepad.axis(AXIS_FORWARD) - gamepad.axis(AXIS_BACKWARD), -1.0, 1.0)
                    duty = clamp(apply_deadzone(throttle) * AUTO_DUTY, -AUTO_DUTY, AUTO_DUTY)
                    steer = apply_deadzone(gamepad.axis(AXIS_STEERING))
                    servo = clamp(SERVO_CENTER + steer * SERVO_RANGE, 0.0, 1.0)
                    target_x = mask.shape[1] // 2

                vesc.set_servo(servo)
                vesc.set_duty_cycle(duty)

                with lidar_lock:
                    lidar_snapshot = latest_lidar
                if recording:
                    if lidar_snapshot is None or time.monotonic() - lidar_snapshot[2] > 0.5:
                        print("\n[WARN] scan LiDAR trop ancien : image non enregistrée")
                    else:
                        ranges, lidar_timestamp, _received_at = lidar_snapshot
                        image_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
                        image_name = f"frame_{image_stamp}.png"
                        image_path = IMAGES_DIR / image_name
                        relative_image = f"images/{image_name}"
                        row = [relative_image, f"{servo:.4f}", f"{duty:.4f}", lidar_timestamp,
                               json.dumps(ranges, separators=(",", ":"))]
                        try:
                            write_queue.put_nowait((image_path, mask.copy(), row))
                        except Full:
                            print("\n[WARN] file d'écriture saturée : frame ignorée")

                display = cv2.resize(mask, (640, 480))
                display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
                if autonomous:
                    tx = int(target_x * display.shape[1] / mask.shape[1])
                    cv2.line(display, (display.shape[1] // 2, display.shape[0]), (tx, int(display.shape[0] * ROI_FAR_TOP)), (0, 0, 255), 2)
                mode = "AUTO" if autonomous else "MANUAL"
                rec = " | REC" if recording else ""
                cv2.putText(display, f"{fps:.1f} FPS | {mode}{rec} | Servo {servo:.2f} | Duty {duty:.3f}",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if recording else (0, 255, 202), 1)
                with frame_lock:
                    latest_frame = display
                if has_display:
                    cv2.imshow("Lane mask + LiDAR dataset", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop_event.set()
    except KeyboardInterrupt:
        print("\n[INFO] arrêt demandé")
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        return 1
    finally:
        stop_event.set()
        if vesc is not None:
            try:
                vesc.set_duty_cycle(0.0)
                vesc.set_servo(SERVO_CENTER)
            except Exception:
                pass
            close_vesc = getattr(vesc, "close", None)
            if callable(close_vesc):
                try:
                    close_vesc()
                except Exception:
                    pass
        if gamepad is not None:
            try:
                gamepad.stopBackgroundUpdates()
            except Exception:
                pass
        try:
            write_queue.put(None, timeout=2.0)
            writer_thread.join(timeout=5.0)
        except Full:
            print("[WARN] arrêt du writer avant vidage complet de la file")
        csv_handle.close()
        server.shutdown()
        server.server_close()
        cv2.destroyAllWindows()
        print("[INFO] arrêt moteur demandé, ressources fermées")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
