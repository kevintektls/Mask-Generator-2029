#!/usr/bin/env python3
"""
Capture caméra OAK-D + inférence U-Net en temps réel.
Affiche image + masque (lignes blanches) côte à côte.
"""

import sys
import time
from pathlib import Path
from PIL import Image

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

try:
    import depthai as dai
except ImportError:
    print("DepthAI not installed. Run: pip install depthai")
    sys.exit(1)

# Config
MODEL_PATH = Path("../model/unet.pth")
PREVIEW_SIZE = (640, 480)
FPS = 30
MODEL_INPUT_SIZE = (256, 256)  # (width, height) - model trained on 256x256

# ── U-NET Architecture (same as train.py) ──────────────────────────────────
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
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, features=(16, 32, 64, 128)):
        super().__init__()
        self.enc1 = ConvBlock(3, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(features[3], features[3] * 2)
        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], 2, 2)
        self.dec4 = ConvBlock(features[3] * 2, features[3])
        self.up3 = nn.ConvTranspose2d(features[3], features[2], 2, 2)
        self.dec3 = ConvBlock(features[2] * 2, features[2])
        self.up2 = nn.ConvTranspose2d(features[2], features[1], 2, 2)
        self.dec2 = ConvBlock(features[1] * 2, features[1])
        self.up1 = nn.ConvTranspose2d(features[1], features[0], 2, 2)
        self.dec1 = ConvBlock(features[0] * 2, features[0])
        self.final = nn.Conv2d(features[0], 1, kernel_size=1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))

# Charge modèle
print(f"Loading U-Net from {MODEL_PATH}...")
if not MODEL_PATH.exists():
    print(f"ERROR: Model not found at {MODEL_PATH}")
    sys.exit(1)

model = UNet()
model.load_state_dict(torch.load(str(MODEL_PATH), map_location='cpu'))
model.eval()
print("✓ Model loaded\n")

# Transforms U-Net - exact size without Resize conflicts
transform = transforms.Compose([
    transforms.ToTensor(),
])

def build_pipeline():
    pipeline = dai.Pipeline()
    
    cam_left = pipeline.create(dai.node.MonoCamera)
    cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    cam_left.setFps(FPS)
    
    xout_left = pipeline.create(dai.node.XLinkOut)
    xout_left.setStreamName("left")
    xout_left.input.setBlocking(False)
    xout_left.input.setQueueSize(4)
    cam_left.out.link(xout_left.input)
    
    return pipeline

def main():
    print("Building OAK-D pipeline...")
    pipeline = build_pipeline()
    
    with dai.Device(pipeline) as device:
        q_left = device.getOutputQueue(name="left", maxSize=4, blocking=False)
        
        print("✓ Camera connected!\nPress 'q' to quit, 's' to save.\n")
        
        fps_counter = {"count": 0, "t": time.monotonic(), "fps": 0.0}
        
        while True:
            pkt = q_left.tryGet()
            
            if pkt is None:
                continue
            
            # Calcul FPS
            fps_counter["count"] += 1
            now = time.monotonic()
            if now - fps_counter["t"] >= 1.0:
                fps_counter["fps"] = fps_counter["count"] / (now - fps_counter["t"])
                fps_counter["count"] = 0
                fps_counter["t"] = now
            
            # Image caméra
            frame = pkt.getCvFrame()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame_rgb = cv2.resize(frame_rgb, PREVIEW_SIZE)
            
            # Inférence U-Net
            # Redimensionne exactement à (347, 256) sans double resize
            frame_resized = cv2.resize(frame, MODEL_INPUT_SIZE)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            tensor = transform(frame_pil).unsqueeze(0)
            
            with torch.no_grad():
                mask_logits = model(tensor)
                mask = (torch.sigmoid(mask_logits) > 0.5).float().squeeze(0).cpu().numpy()
            
            # Masque en blanc (255) sur fond noir (0)
            mask_white = (mask[0] * 255).astype(np.uint8)
            mask_white = cv2.resize(mask_white, PREVIEW_SIZE)
            mask_display = cv2.cvtColor(mask_white, cv2.COLOR_GRAY2BGR)
            
            # FPS
            fps_text = f"{fps_counter['fps']:.1f} FPS"
            cv2.putText(frame_rgb, fps_text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(mask_display, fps_text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Affiche côte à côte
            display = np.hstack([frame_rgb, mask_display])
            cv2.imshow("Camera | U-Net Mask (Lines)", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"camera_{ts}.png", frame_rgb)
                cv2.imwrite(f"mask_{ts}.png", mask_display)
                print(f"✓ Saved: camera_{ts}.png, mask_{ts}.png")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()