#!/usr/bin/env python3
import sys, time, threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    import depthai as dai
except ImportError:
    print("DepthAI not installed."); sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH        = Path("../model/unet.pth")
DISPLAY_W         = 640
DISPLAY_H         = 480
UNET_SIZE         = 128          # 128 au lieu de 256 → ×4 plus rapide, suffisant pour lignes épaisses
INFER_EVERY_N     = 2            # n'inférer qu'1 frame sur N (skip frames)
HTTP_PORT         = 5000
torch.set_num_threads(4)         # Jetson Nano = 4 cœurs ARM Cortex-A57

# ── Archi exacte du train.py ──────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, features=(16, 32, 64, 128)):
        super().__init__()
        self.enc1 = ConvBlock(3,           features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(features[3], features[3] * 2)
        self.up4  = nn.ConvTranspose2d(features[3] * 2, features[3], 2, 2)
        self.dec4 = ConvBlock(features[3] * 2, features[3])
        self.up3  = nn.ConvTranspose2d(features[3], features[2], 2, 2)
        self.dec3 = ConvBlock(features[2] * 2, features[2])
        self.up2  = nn.ConvTranspose2d(features[2], features[1], 2, 2)
        self.dec2 = ConvBlock(features[1] * 2, features[1])
        self.up1  = nn.ConvTranspose2d(features[1], features[0], 2, 2)
        self.dec1 = ConvBlock(features[0] * 2, features[0])
        self.final = nn.Conv2d(features[0], 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))

# ── Chargement + optimisations CPU ───────────────────────────────────────────
print(f"Loading U-Net from {MODEL_PATH}...")
_model = UNet()
_model.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu", weights_only=True))
_model.eval()
model = torch.jit.script(_model)   # compile le graph → ~20-30% plus rapide sur CPU
print("✓ Model loaded + JIT compiled\n")

# Normalisation ImageNet en numpy (évite PIL + torchvision.transforms à chaque frame)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(frame_gray: np.ndarray) -> torch.Tensor:
    """Gray uint8 → tensor (1,3,UNET_SIZE,UNET_SIZE), tout en numpy, zéro PIL."""
    rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
    rgb = cv2.resize(rgb, (UNET_SIZE, UNET_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,H,W)

def run_inference(frame_gray: np.ndarray) -> np.ndarray:
    tensor = preprocess(frame_gray)
    with torch.no_grad():
        pred = model(tensor)
    return ((pred > 0.5).squeeze().cpu().numpy() * 255).astype(np.uint8)

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
    cam.setFps(30)
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
    print(f"✓ Stream sur http://$(hostname -I | awk '{{print $1}}'):{HTTP_PORT}")

    pipeline = build_pipeline()
    with dai.Device(pipeline) as device:
        q = device.getOutputQueue(name="left", maxSize=2, blocking=False)
        print("✓ Camera connected — Ctrl+C pour quitter\n")

        frame_count = 0
        count, t0, fps = 0, time.monotonic(), 0.0
        last_mask = np.zeros((DISPLAY_H, DISPLAY_W), dtype=np.uint8)

        while True:
            pkt = q.tryGet()
            if pkt is None:
                time.sleep(0.001)
                continue

            raw = pkt.getCvFrame()
            frame_count += 1

            # FPS display
            count += 1
            now = time.monotonic()
            if now - t0 >= 1.0:
                fps = count / (now - t0); count = 0; t0 = now

            # Inférence seulement 1 frame sur INFER_EVERY_N
            if frame_count % INFER_EVERY_N == 0:
                last_mask = run_inference(raw)

            # Panels affichage
            frame_bgr = cv2.resize(
                cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR),
                (DISPLAY_W, DISPLAY_H)
            )
            mask_bgr = cv2.resize(
                cv2.cvtColor(last_mask, cv2.COLOR_GRAY2BGR),
                (DISPLAY_W, DISPLAY_H)
            )

            label = f"{fps:.1f} FPS  (infer 1/{INFER_EVERY_N})"
            cv2.putText(frame_bgr, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(mask_bgr,  label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            with frame_lock:
                latest_frame = np.hstack([frame_bgr, mask_bgr])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nQuitting.")