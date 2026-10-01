#!/usr/bin/env python3
"""Conduite manuelle et collecte d'un dataset LiDAR pour le LDROBOT D500.

Le D500 (STL-19P) est lu sur l'UART de la Jetson Nano à 230400 bauds.
Chaque tour complet devient une ligne CSV avec les 180 distances du champ
avant (de -90° à +90°), en mètres, et les commandes servo/duty de la voiture.

Branchement UART Jetson Nano : TX LiDAR -> broche 10 (UART RX), GND commun.
La broche 8 (UART TX) n'est pas utilisée. Le D500 demande une alimentation
5 V et sa broche PWM doit être à GND si elle n'est pas pilotée.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import Gamepad
except ImportError:
    Gamepad = None

try:
    from pyvesc import VESC
except ImportError:
    print("[ERROR] pyvesc absent. Installe-le avec : pip install pyvesc")
    sys.exit(1)

try:
    import serial
except ImportError:
    print("[ERROR] pyserial absent. Installe-le avec : pip install pyserial")
    sys.exit(1)


# Configuration matériel et dataset
AXIS_FORWARD = "RT"
AXIS_BACKWARD = "LT"
AXIS_STEERING = "LEFT-X"
DEADZONE = 0.08

VESC_PORT = "/dev/ttyACM0"
VESC_BAUDRATE = 115200
VESC_TIMEOUT = 1.0
VESC_CONNECT_RETRIES = 8
VESC_CONNECT_SETTLE = 1.0

LIDAR_PORT = "/dev/ttyTHS1"
LIDAR_BAUDRATE = 230400
LIDAR_MAX_RANGE_M = 12.0
LIDAR_BINS = 180
PREVIEW_HOST = "0.0.0.0"
PREVIEW_PORT = 5001
REMOTE_CONTROL_BIND = "127.0.0.1:5010"
REMOTE_COMMAND_TIMEOUT = 0.24
REMOTE_COMMAND_RATE = 0.01

MAX_DUTY_CYCLE = 0.10
SERVO_CENTER = 0.5
SERVO_RANGE = 0.48
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_CSV = PROJECT_DIR / "dataset_lidar/driving_log.csv"

PACKET_HEADER = 0x54
PACKET_VERLEN = 0x2C
PACKET_POINT_COUNT = 12
PACKET_SIZE = 47


def crc8(data: bytes) -> int:
    """CRC-8 LDROBOT (table officielle, polynôme 0x4D, valeur initiale 0)."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (((crc << 1) ^ 0x4D) if crc & 0x80 else (crc << 1)) & 0xFF
    return crc


class D500Reader:
    """Décode les trames STL-19P et renvoie des scans avant à 1° de résolution."""

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
        while True:
            if self.serial.in_waiting:
                self.buffer.extend(self.serial.read(self.serial.in_waiting))
            else:
                chunk = self.serial.read(1)
                if chunk:
                    self.buffer.extend(chunk)

            while self.buffer:
                if self.buffer[0] != PACKET_HEADER:
                    del self.buffer[0]
                    continue
                if len(self.buffer) < 2:
                    break
                if self.buffer[1] != PACKET_VERLEN:
                    del self.buffer[0]
                    continue
                if len(self.buffer) < PACKET_SIZE:
                    break

                packet = bytes(self.buffer[:PACKET_SIZE])
                if crc8(packet[:-1]) != packet[-1]:
                    del self.buffer[0]
                    continue
                del self.buffer[:PACKET_SIZE]
                return packet

    @staticmethod
    def _decode_points(packet: bytes):
        start = int.from_bytes(packet[4:6], "little") / 100.0
        end = int.from_bytes(packet[42:44], "little") / 100.0
        angle_delta = (end - start) % 360.0
        for i in range(PACKET_POINT_COUNT):
            offset = 6 + i * 3
            distance_mm = int.from_bytes(packet[offset:offset + 2], "little")
            angle_deg = (start + angle_delta * i / (PACKET_POINT_COUNT - 1)) % 360.0
            yield angle_deg, distance_mm / 1000.0

    def _next_point(self):
        while not self.points:
            packet = self._read_packet()
            self.points.extend(self._decode_points(packet))
        return self.points.popleft()

    @staticmethod
    def _bin_for_angle(angle_deg: float):
        # STL-19P angles increase clockwise from forward: positive is right.
        signed = angle_deg if angle_deg <= 180.0 else angle_deg - 360.0
        if not -90.0 <= signed <= 90.0:
            return None
        return min(LIDAR_BINS - 1, int(signed + 90.0))

    def read_scan(self) -> list[float]:
        ranges = [LIDAR_MAX_RANGE_M] * LIDAR_BINS
        last_angle = None
        point_count = 0

        while True:
            angle, distance_m = self.pending_point or self._next_point()
            self.pending_point = None

            # La baisse de 0/360 degrés marque la fin d'un tour complet.
            if last_angle is not None and angle < last_angle - 180.0:
                if point_count >= 100:
                    self.pending_point = (angle, distance_m)
                    return ranges
                ranges = [LIDAR_MAX_RANGE_M] * LIDAR_BINS
                point_count = 0

            bin_index = self._bin_for_angle(angle)
            if bin_index is not None and 0.0 < distance_m <= LIDAR_MAX_RANGE_M:
                ranges[bin_index] = min(ranges[bin_index], distance_m)
                point_count += 1
            last_angle = angle


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def apply_deadzone(value: float) -> float:
    if abs(value) < DEADZONE:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - DEADZONE) / (1.0 - DEADZONE)


def connect_vesc():
    last_error = None
    for attempt in range(VESC_CONNECT_RETRIES):
        try:
            vesc = VESC(serial_port=VESC_PORT, baudrate=VESC_BAUDRATE, timeout=VESC_TIMEOUT)
            print(f"[INFO] VESC connecté ({attempt + 1}/{VESC_CONNECT_RETRIES}).")
            return vesc
        except Exception as exc:
            last_error = exc
            print(f"[WARNING] Connexion VESC {attempt + 1} échouée : {exc}")
            time.sleep(VESC_CONNECT_SETTLE)
    raise RuntimeError(f"Connexion VESC impossible : {last_error}")


class RemoteControl:
    """Receive Mac gamepad state; the VESC is still written only on the Jetson."""

    def __init__(self, address: str):
        host, separator, port_text = address.rpartition(":")
        if not separator or not host:
            raise ValueError("--control-bind must be HOST:PORT")
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("le contrôle distant doit rester lié à localhost")
        self.address = (host, int(port_text))
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.last_frame = 0.0
        self.rt = 0
        self.lt = 0
        self.lx = 0
        self.a = False
        self.lb = False
        self.previous_a = False
        self.recording = False
        self.armed = False
        self.connected = False
        self.listener_thread = threading.Thread(target=self._listen, daemon=True)

    def start(self):
        self.listener_thread.start()

    def _listen(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(self.address)
            listener.listen(1)
            listener.settimeout(0.5)
            print(f"[INFO] En attente du client manette sur {self.address[0]}:{self.address[1]}")
            while not self.stop_event.is_set():
                try:
                    connection, peer = listener.accept()
                except socket.timeout:
                    continue
                try:
                    connection.sendall(b'{"type":"ready"}\n')
                except OSError as exc:
                    print(f"[WARNING] Client manette fermé avant l'initialisation : {exc}")
                    connection.close()
                    continue
                print(f"[INFO] Client manette connecté depuis {peer[0]}")
                with connection:
                    connection.settimeout(0.5)
                    buffer = bytearray()
                    with self.lock:
                        self.connected = True
                        self.armed = False
                        self.last_frame = 0.0
                    while not self.stop_event.is_set():
                        try:
                            chunk = connection.recv(256)
                        except socket.timeout:
                            continue
                        if not chunk:
                            break
                        buffer.extend(chunk)
                        if len(buffer) > 4096:
                            raise ValueError("commande manette trop longue")
                        while b"\n" in buffer:
                            line, _, remainder = buffer.partition(b"\n")
                            buffer = bytearray(remainder)
                            if not line:
                                continue
                            self._apply_frame(json.loads(line))
                with self.lock:
                    self.connected = False
                    self.armed = False
                    self.last_frame = 0.0
                    self.rt = self.lt = self.lx = 0
                    self.a = self.lb = False
                self.stop_event.set()
                break
        except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
            print(f"[ERROR] Réception manette interrompue : {exc}")
            self.stop_event.set()
        finally:
            listener.close()

    def _apply_frame(self, frame):
        if frame.get("type") == "stop":
            self.stop_event.set()
            return
        rt = int(frame["rt"])
        lt = int(frame["lt"])
        lx = int(frame["lx"])
        a = bool(frame["a"])
        lb = bool(frame["lb"])
        if not (0 <= rt <= 255 and 0 <= lt <= 255 and -32768 <= lx <= 32767):
            raise ValueError("valeurs manette hors limites")
        with self.lock:
            self.rt, self.lt, self.lx = rt, lt, lx
            self.a, self.lb = a, lb
            self.last_frame = time.monotonic()
            if a and not self.previous_a:
                self.recording = not self.recording
                print("[DATASET] " + ("ENREGISTREMENT ACTIF" if self.recording else "PAUSE"))
            self.previous_a = a
            if lb:
                self.recording = False
                self.stop_event.set()
            if rt <= 5 and lt <= 5 and abs(lx) < 6000 and not lb:
                self.armed = True

    def snapshot(self):
        with self.lock:
            fresh = self.connected and (time.monotonic() - self.last_frame) <= REMOTE_COMMAND_TIMEOUT
            if not fresh:
                self.armed = False
            armed = self.armed and fresh and not self.lb
            throttle = (self.rt - self.lt) / 255.0 if armed else 0.0
            steering = self.lx / 32767.0 if armed else 0.0
            recording = self.recording
        duty = apply_deadzone(clamp(throttle, -1.0, 1.0)) * MAX_DUTY_CYCLE
        servo = clamp(SERVO_CENTER + apply_deadzone(steering) * SERVO_RANGE, 0.0, 1.0)
        return duty, servo, recording


def remote_motor_loop(vesc, remote: RemoteControl):
    try:
        while not remote.stop_event.is_set():
            duty, servo, _ = remote.snapshot()
            vesc.set_duty_cycle(duty)
            vesc.set_servo(servo)
            time.sleep(REMOTE_COMMAND_RATE)
    except Exception as exc:
        print(f"[ERROR] Écriture VESC interrompue : {exc}")
        remote.stop_event.set()
    finally:
        try:
            vesc.set_duty_cycle(0.0)
            vesc.set_servo(SERVO_CENTER)
        except Exception as exc:
            print(f"[WARNING] Arrêt VESC distant incomplet : {exc}")


def open_dataset(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    handle = path.open("a", newline="", encoding="utf-8", buffering=1)
    writer = csv.writer(handle)
    if not exists:
        writer.writerow(["timestamp", "servo", "duty", "lidar"])
        handle.flush()
    else:
        with path.open("r", newline="", encoding="utf-8") as existing:
            header = next(csv.reader(existing), [])
        if header != ["timestamp", "servo", "duty", "lidar"]:
            handle.close()
            raise ValueError(
                f"Le CSV {path} existe avec un format différent. "
                "Choisis un nouveau chemin avec --csv."
            )
    return handle, writer


preview_state = {"scan": [], "servo": SERVO_CENTER, "duty": 0.0, "recording": False,
                 "updated_at": None}
preview_lock = threading.Lock()


class LidarPreviewHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path == "/scan":
            with preview_lock:
                payload = json.dumps(preview_state, separators=(",", ":")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path != "/":
            self.send_error(404)
            return

        page = """<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Preview LiDAR + caméra</title>
<style>body{margin:0;background:#111820;color:#e8eef2;font:15px system-ui;padding:16px;box-sizing:border-box}
main{width:min(98vw,1400px);margin:auto}h1{font-size:1.2rem;font-weight:600;margin:0 0 10px}.sensors{display:grid;grid-template-columns:1.15fr 1fr;gap:14px;align-items:start}.panel{min-width:0;background:#151e27;border:1px solid #34414c;border-radius:12px;padding:10px;box-sizing:border-box}h2{font-size:1rem;margin:0 0 8px}canvas{display:block;width:100%;background:#151e27;border-radius:8px}#camera{display:block;width:100%;height:auto;max-height:70vh;object-fit:contain;background:#0b0f13;border-radius:8px}p{color:#aab7c1;font-variant-numeric:tabular-nums;margin:8px 0}.help{font-size:.9rem}@media(max-width:760px){.sensors{grid-template-columns:1fr}}</style><main>
<h1>Preview LiDAR + caméra · contrôle manette distant</h1><section class="sensors"><div class="panel"><h2>LiDAR D500</h2>
<canvas id="radar" width="760" height="520"></canvas><p id="status">Connexion LiDAR…</p></div><div class="panel"><h2 id="camera-title">OAK-D Lite · CAM_B gauche / CAM_C droite synchronisées</h2>
<img id="camera" alt="Flux caméra en attente"><p id="camera-status">Connexion caméra…</p></div></section>
<p class="help">Manette : RT = avancer, LT = reculer, joystick gauche = direction, A = enregistrer/pause, LB = arrêt.</p></main>
<script>const c=document.querySelector('#radar'),x=c.getContext('2d'),status=document.querySelector('#status');
function draw(s){const w=c.width,h=c.height,cx=w/2,cy=h-42,R=Math.min(w/2-30,h-80);x.clearRect(0,0,w,h);x.fillStyle='#151e27';x.fillRect(0,0,w,h);
for(let m=1;m<=6;m++){let r=R*m/6;x.beginPath();x.arc(cx,cy,r,Math.PI,2*Math.PI);x.strokeStyle='#35424d';x.stroke();x.fillStyle='#aab7c1';x.font='13px system-ui';x.fillText(m+'m',cx+8,cy-r-4)}
for(let a=-90;a<=90;a+=30){let t=a*Math.PI/180;x.beginPath();x.moveTo(cx,cy);x.lineTo(cx+Math.sin(t)*R,cy-Math.cos(t)*R);x.strokeStyle='#26333d';x.stroke()}
(s.scan||[]).forEach((d,i)=>{if(!(d>0&&d<12))return;let a=(i-90)*Math.PI/180,r=R*Math.min(d,6)/6;x.beginPath();x.arc(cx+Math.sin(a)*r,cy-Math.cos(a)*r,3,0,Math.PI*2);x.fillStyle=d<1?'#ff645e':'#56d891';x.fill()});
x.fillStyle='#8fbbe0';x.fillRect(cx-1,cy-R,2,R);status.textContent=`${s.recording?'REC':'MANUAL'} · servo ${Number(s.servo).toFixed(2)} · duty ${Number(s.duty).toFixed(3)} · updated ${s.updated_at||'waiting for scan'}`}
async function poll(){try{const r=await fetch('/scan',{cache:'no-store'});draw(await r.json())}catch(e){status.textContent='Connexion LiDAR perdue ; reconnexion…'}setTimeout(poll,150)}poll();
const camera=document.querySelector('#camera'),cameraTitle=document.querySelector('#camera-title'),cameraStatus=document.querySelector('#camera-status');camera.src=`http://${location.hostname}:9011/stream.mjpg`;camera.onerror=()=>cameraStatus.textContent='Caméra indisponible — vérifier oak_bridge.py';async function pollCameraStatus(){try{const d=await (await fetch(`http://${location.hostname}:9011/status`,{cache:'no-store'})).json();cameraTitle.textContent=d.middle_camera_enabled?'OAK-D Lite · CAM_B gauche / CAM_A couleur / CAM_C droite':'OAK-D Lite · CAM_B gauche / CAM_C droite synchronisées';cameraStatus.textContent=d.ready?(d.middle_camera_enabled?`3 caméras · séq. ${d.sequence} · synchronisation CAM_B/CAM_C ${Number(d.left_right_delta_ms).toFixed(2)} ms · CAM_A/CAM_B ${Number(d.middle_delta_ms).toFixed(2)} ms`:`Paire synchronisée · séq. ${d.sequence} · écart ${Number(d.left_right_delta_ms).toFixed(3)} ms / seuil ${d.sync_threshold_ms} ms`):'En attente des caméras'}catch(e){cameraStatus.textContent='Caméra indisponible — vérifier oak_bridge.py'}}setInterval(pollCameraStatus,500);pollCameraStatus();</script></html>""".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lidar-port", default=LIDAR_PORT,
                        help=f"UART du LiDAR (défaut : {LIDAR_PORT})")
    parser.add_argument("--csv", type=Path, default=DATASET_CSV,
                        help=f"CSV de sortie (défaut : {DATASET_CSV})")
    parser.add_argument("--vesc-port", default=VESC_PORT,
                        help=f"Port série VESC (défaut : {VESC_PORT})")
    parser.add_argument("--preview", action="store_true",
                        help="Démarre l'aperçu LiDAR accessible dans un navigateur (utile en SSH)")
    parser.add_argument("--preview-host", default=PREVIEW_HOST,
                        help=f"Adresse d'écoute de l'aperçu (défaut : {PREVIEW_HOST})")
    parser.add_argument("--preview-port", type=int, default=PREVIEW_PORT,
                        help=f"Port HTTP de l'aperçu (défaut : {PREVIEW_PORT})")
    parser.add_argument("--remote-control", action="store_true",
                        help="Reçoit la manette depuis un client distant via TCP")
    parser.add_argument("--control-bind", default=REMOTE_CONTROL_BIND,
                        help=f"Adresse du client distant (défaut : {REMOTE_CONTROL_BIND})")
    parser.add_argument("--check-dependencies", action="store_true",
                        help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    global VESC_PORT
    args = parse_args()
    VESC_PORT = args.vesc_port

    if args.check_dependencies:
        if not args.remote_control and Gamepad is None:
            sys.path.insert(0, "/home/robotcar/Gamepad")
            try:
                import Gamepad as gamepad_api
            except ImportError:
                print("[ERROR] Bibliothèque Gamepad introuvable.")
                return 1
        return 0

    preview_server = None
    if args.preview:
        try:
            preview_server = ThreadingHTTPServer((args.preview_host, args.preview_port), LidarPreviewHandler)
        except OSError as exc:
            print(f"[ERROR] Impossible de démarrer le serveur d'aperçu : {exc}")
            return 1
        threading.Thread(target=preview_server.serve_forever, daemon=True).start()
        print(f"[INFO] Aperçu web : http://<adresse-du-robot>:{args.preview_port}/")

    gamepad = None
    if not args.remote_control:
        if Gamepad is None:
            sys.path.insert(0, "/home/robotcar/Gamepad")
            try:
                import Gamepad as gamepad_api
            except ImportError:
                print("[ERROR] Bibliothèque Gamepad introuvable.")
                return 1
        else:
            gamepad_api = Gamepad
        if not gamepad_api.available():
            print("[INFO] En attente de la manette...")
            while not gamepad_api.available():
                time.sleep(0.5)
        gamepad = gamepad_api.Xbox360()
        gamepad.startBackgroundUpdates()
    dataset_handle = None
    lidar = None
    vesc = None
    remote = RemoteControl(args.control_bind) if args.remote_control else None
    remote_motor_thread = None
    recording = False
    previous_a = False

    try:
        try:
            lidar = D500Reader(args.lidar_port)
        except serial.SerialException as exc:
            raise RuntimeError(
                f"Impossible d'ouvrir le LiDAR sur {args.lidar_port}: {exc}. "
                "Vérifie le port, les permissions et le branchement UART."
            ) from exc

        dataset_handle, csv_writer = open_dataset(args.csv)
        vesc = connect_vesc()
        with vesc:
            vesc.set_servo(SERVO_CENTER)
            vesc.set_duty_cycle(0.0)
            if remote is not None:
                remote.start()
                remote_motor_thread = threading.Thread(
                    target=remote_motor_loop, args=(vesc, remote), daemon=True
                )
                remote_motor_thread.start()
            print("\n=== COLLECTEUR DATASET LiDAR D500 ===")
            print("RT : avancer | LT : reculer | joystick gauche : direction")
            print("A : démarrer/arrêter l'enregistrement | LB : arrêt immédiat")
            if args.preview:
                print(f"Aperçu web : http://<adresse-du-robot>:{args.preview_port}/")
            print(f"CSV : {args.csv} | UART LiDAR : {args.lidar_port} @ {LIDAR_BAUDRATE}")

            try:
                while ((remote is None and gamepad.isConnected()) or
                       (remote is not None and not remote.stop_event.is_set())):
                    if remote is not None:
                        duty, servo, recording = remote.snapshot()
                    else:
                        duty = servo = 0.0
                        recording = False

                    if remote is None and gamepad.isPressed("LB"):
                        recording = False
                        print("[STOP] LB pressé : arrêt immédiat.")
                        break

                    if remote is None:
                        a_now = gamepad.isPressed("A")
                        if a_now and not previous_a:
                            recording = not recording
                            print("[DATASET] " + ("ENREGISTREMENT ACTIF" if recording else "PAUSE"))
                        previous_a = a_now

                        throttle = clamp(
                            gamepad.axis(AXIS_FORWARD) - gamepad.axis(AXIS_BACKWARD), -1.0, 1.0
                        )
                        duty = apply_deadzone(throttle) * MAX_DUTY_CYCLE
                        steering = apply_deadzone(gamepad.axis(AXIS_STEERING))
                        servo = clamp(SERVO_CENTER + steering * SERVO_RANGE, 0.0, 1.0)
                        vesc.set_duty_cycle(duty)
                        vesc.set_servo(servo)

                    # Les commandes restent stables pendant l'acquisition du tour,
                    # elles décrivent donc bien le scan associé dans le CSV.
                    scan = lidar.read_scan()

                    if args.preview:
                        with preview_lock:
                            preview_state.update({
                                "scan": scan, "servo": servo, "duty": duty,
                                "recording": recording,
                                "updated_at": datetime.now().isoformat(timespec="seconds"),
                            })

                    if recording:
                        timestamp = datetime.now().isoformat(timespec="milliseconds")
                        csv_writer.writerow([
                            timestamp,
                            f"{servo:.4f}",
                            f"{duty:.4f}",
                            json.dumps(scan, separators=(",", ":")),
                        ])
                        dataset_handle.flush()

                    valid_ranges = [distance for distance in scan if distance < LIDAR_MAX_RANGE_M]
                    nearest = min(valid_ranges) if valid_ranges else math.inf
                    nearest_text = f"{nearest:.2f} m" if math.isfinite(nearest) else "aucun retour"
                    mode = "REC" if recording else "MANUEL"
                    print(
                        f"\r[{mode}] servo={servo:.2f} duty={duty:.3f} "
                        f"obstacle avant le plus proche={nearest_text}   ",
                        end="",
                        flush=True,
                    )
            finally:
                if remote is not None:
                    remote.stop_event.set()
                    if remote_motor_thread is not None:
                        remote_motor_thread.join()
                # Couper le moteur avant de quitter le contexte pyvesc.
                vesc.set_duty_cycle(0.0)
                vesc.set_servo(SERVO_CENTER)

    except KeyboardInterrupt:
        print("\n[INFO] Arrêt demandé.")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"\n[ERROR] {exc}")
        return 1
    finally:
        if lidar is not None:
            lidar.close()
        if dataset_handle is not None:
            dataset_handle.close()
        if preview_server is not None:
            preview_server.shutdown()
            preview_server.server_close()
        if gamepad is not None:
            gamepad.stopBackgroundUpdates()
        print("\n[INFO] Commandes moteur coupées, ressources fermées.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
