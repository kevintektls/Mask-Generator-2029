#!/usr/bin/env python3
import sys, time, threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

try:
    import depthai as dai
except ImportError:
    print("DepthAI not installed."); sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
DISPLAY_W         = 640
DISPLAY_H         = 480
HTTP_PORT         = 5000

# Paramètres de détection (Ajustables selon ton éclairage)
# Si tes lignes sont très nettes, un seuillage binaire classique (cv2.threshold) suffit.
# Si la luminosité change bcp, le seuillage adaptatif est préférable.
USE_ADAPTIVE_THRESH = True

# ── Traitement d'image classique (Remplace U-Net) ──────────────────────────────
def detect_lines(frame_gray: np.ndarray) -> np.ndarray:
    """
    Prend une image en niveaux de gris et isole les lignes blanches.
    Retourne un masque binaire (0 ou 255) de la taille d'origine.
    """
    # 1. Réduire le bruit (indispensable pour éviter les faux positifs)
    blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    
    if USE_ADAPTIVE_THRESH:
        # S'adapte aux changements de lumière locaux
        # 255 = blanc pour la ligne, 21 = taille du bloc, 4 = constante soustraite
        mask = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 4
        )
    else:
        # Seuillage classique : tout ce qui est au-dessus de 180 devient blanc
        _, mask = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
        
    # Optionnel : Opération morphologique pour nettoyer les petits points noirs/blancs isolés
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

# ── Frame partagée thread-safe ────────────────────────────────────────────────
latest_frame: np.ndarray | None = None
frame_lock = threading.Lock()

# ── Serveur MJPEG ─────────────────────────────────────────────────────────────
class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def do_GET(self):
        if self.path == "/":
            html = (
                b"<!DOCTYPE html><html><head><title>OAK-D Stream</title>"
                b"<style>body{background:#111;margin:0;display:flex;"
                b"justify-content:center;align-items:center;height:100vh;}"
                b"img{max-width:100%;border:2px solid #0f0;}</style></head>"
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
                while True:
                    with frame_lock:
                        frame = latest_frame
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if not ok:
                        continue
                    data = jpg.tobytes()
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
                    )
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404); self.end_headers()

def start_http_server():
    HTTPServer(("0.0.0.0", HTTP_PORT), MJPEGHandler).serve_forever()

# ── Pipeline OAK-D ───────────────────────────────────────────────────────────
def build_pipeline():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.MonoCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    
    # On peut repasser à 20 ou 30 FPS sans problème maintenant !
    cam.setFps(60) 
    
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("left")
    xout.input.setBlocking(False)
    xout.input.setQueueSize(2)
    cam.out.link(xout.input)
    return pipeline

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    global latest_frame

    threading.Thread(target=start_http_server, daemon=True).start()
    print(f"✓ Stream sur http://localhost:{HTTP_PORT} (ou l'IP de ta Jetson)")

    pipeline = build_pipeline()
    with dai.Device(pipeline) as device:
        q = device.getOutputQueue(name="left", maxSize=2, blocking=False)
        print("✓ Camera connectee — OpenCV pipeline actif — Ctrl+C pour quitter\n")

        count, t0, fps = 0, time.monotonic(), 0.0

        while True:
            pkt = q.tryGet()
            if pkt is None:
                time.sleep(0.001)
                continue

            raw = pkt.getCvFrame() # Image en niveaux de gris directe de l'OAK-D

            # FPS display
            count += 1
            now = time.monotonic()
            if now - t0 >= 1.0:
                fps = count / (now - t0); count = 0; t0 = now

            # Traitement ultra rapide (Quelques microsecondes)
            mask = detect_lines(raw)

            # Redimensionnement et passage en BGR pour l'affichage de la box de texte
            mask_resized = cv2.resize(mask, (DISPLAY_W, DISPLAY_H))
            mask_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

            label = f"{fps:.1f} FPS (OpenCV Pipeline)"
            cv2.putText(mask_bgr, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            with frame_lock:
                latest_frame = mask_bgr

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nQuitting.")