#!/usr/bin/env python3
"""
Capture OAK-D + inférence U-Net → stream MJPEG sur http://<IP>:5000
"""
import sys
import time
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

try:
    import depthai as dai
except ImportError:
    print("DepthAI not installed."); sys.exit(1)

# Config
MODEL_PATH  = Path("../model/unet.pth")
DISPLAY_W, DISPLAY_H = 640, 480
UNET_SIZE   = (256, 256)
FPS         = 30
HTTP_PORT   = 5000

# ── U-Net (archi identique au train.py) ──────────────────────────────────────
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
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading U-Net from {MODEL_PATH}...")
model = UNet()
model.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu", weights_only=True))
model.eval()
print("✓ Model loaded")

transform = transforms.Compose([
    transforms.Resize(UNET_SIZE),
    transforms.ToTensor(),
])

# Frame partagée entre le thread caméra et le serveur HTTP
latest_frame = None
frame_lock   = threading.Lock()

# ── Serveur MJPEG ─────────────────────────────────────────────────────────────
class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass  # silence les logs HTTP

    def do_GET(self):
        if self.path == "/":
            # Page HTML minimale avec auto-refresh
            html = b"""<!DOCTYPE html><html><head>
            <title>OAK-D U-Net Stream</title>
            <style>body{background:#111;display:flex;justify-content:center;
            align-items:center;height:100vh;margin:0;}
            img{max-width:100%;border:2px solid #0f0;}</style>
            </head><body>
            <img src="/stream" />
            </body></html>"""
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
                        time.sleep(0.033)
                        continue
                    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ok:
                        continue
                    data = jpg.tobytes()
                    self.wfile.write(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                        + data + b"\r\n"
                    )
            except (BrokenPipeError, ConnectionResetError):
                pass  # client déconnecté
        else:
            self.send_response(404); self.end_headers()

def start_http_server():
    server = HTTPServer(("0.0.0.0", HTTP_PORT), MJPEGHandler)
    print(f"✓ Stream dispo sur http://<IP_JETSON>:{HTTP_PORT}")
    server.serve_forever()

# ── Pipeline OAK-D ───────────────────────────────────────────────────────────
def build_pipeline():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.MonoCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    cam.setFps(FPS)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("left")
    xout.input.setBlocking(False)
    xout.input.setQueueSize(2)
    cam.out.link(xout.input)
    return pipeline

def run_inference(frame_gray):
    pil    = Image.fromarray(cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB))
    tensor = transform(pil).unsqueeze(0)
    with torch.no_grad():
        pred = model(tensor)
    mask = (pred > 0.5).squeeze().cpu().numpy()
    return (mask * 255).astype(np.uint8)

def main():
    global latest_frame

    # Lance le serveur HTTP dans un thread daemon
    t = threading.Thread(target=start_http_server, daemon=True)
    t.start()

    print("Building OAK-D pipeline...")
    pipeline = build_pipeline()

    with dai.Device(pipeline) as device:
        q = device.getOutputQueue(name="left", maxSize=2, blocking=False)
        print("✓ Camera connected!\nCtrl+C pour quitter.\n")

        count, t0, fps = 0, time.monotonic(), 0.0

        while True:
            pkt = q.tryGet()
            if pkt is None:
                time.sleep(0.001)
                continue

            count += 1
            now = time.monotonic()
            if now - t0 >= 1.0:
                fps = count / (now - t0); count = 0; t0 = now

            raw = pkt.getCvFrame()

            frame_bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            frame_bgr = cv2.resize(frame_bgr, (DISPLAY_W, DISPLAY_H))

            mask_u8  = run_inference(raw)
            mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
            mask_bgr = cv2.resize(mask_bgr, (DISPLAY_W, DISPLAY_H))

            label = f"{fps:.1f} FPS"
            for img in (frame_bgr, mask_bgr):
                cv2.putText(img, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

            combined = np.hstack([frame_bgr, mask_bgr])
            with frame_lock:
                latest_frame = combined

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nQuitting.")