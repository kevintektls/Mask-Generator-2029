#!/usr/bin/env python3
"""
Capture caméra OAK-D + inférence U-Net en temps réel.
Affiche image + masque côte à côte.
"""
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

try:
    import depthai as dai
except ImportError:
    print("DepthAI not installed. Run: pip install depthai")
    sys.exit(1)

# Config
MODEL_PATH = Path("../model/unet.pth")
DISPLAY_W, DISPLAY_H = 640, 480
UNET_SIZE = (256, 256)  # le modèle attend du carré 256x256
FPS = 30

# ── Architecture exacte du train.py ──────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
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
        self.final = nn.Conv2d(features[0], 1, kernel_size=1)

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
print("✓ Model loaded\n")

transform = transforms.Compose([
    transforms.Resize(UNET_SIZE),
    transforms.ToTensor(),
])

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
    pil = Image.fromarray(cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB))
    tensor = transform(pil).unsqueeze(0)  # (1, 3, 256, 256)
    with torch.no_grad():
        pred = model(tensor)              # sigmoid déjà appliqué dans forward()
    mask = (pred > 0.5).squeeze().cpu().numpy()
    return (mask * 255).astype(np.uint8)

def main():
    print("Building OAK-D pipeline...")
    pipeline = build_pipeline()

    with dai.Device(pipeline) as device:
        q = device.getOutputQueue(name="left", maxSize=2, blocking=False)
        print("✓ Camera connected!\nPress 'q' to quit, 's' to save.\n")

        count, t0, fps = 0, time.monotonic(), 0.0

        while True:
            pkt = q.tryGet()
            if pkt is None:
                continue

            count += 1
            now = time.monotonic()
            if now - t0 >= 1.0:
                fps = count / (now - t0); count = 0; t0 = now

            raw = pkt.getCvFrame()

            # Panel gauche : caméra
            frame_bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            frame_bgr = cv2.resize(frame_bgr, (DISPLAY_W, DISPLAY_H))

            # Panel droit : masque
            mask_u8  = run_inference(raw)
            mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
            mask_bgr = cv2.resize(mask_bgr, (DISPLAY_W, DISPLAY_H))

            label = f"{fps:.1f} FPS"
            for img in (frame_bgr, mask_bgr):
                cv2.putText(img, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Camera | U-Net Mask", np.hstack([frame_bgr, mask_bgr]))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting.")
                break
            elif key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"camera_{ts}.png", frame_bgr)
                cv2.imwrite(f"mask_{ts}.png", mask_bgr)
                print(f"✓ Saved {ts}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()