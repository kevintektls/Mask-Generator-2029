#!/usr/bin/env python3
"""
Quantize U-Net model to int8 for Jetson Nano.
Gain expected: 2-3x FPS improvement.

Usage:
    python quantize_model.py
"""

import torch
import torch.nn as nn
from pathlib import Path
import os

MODEL_PATH = Path("model/unet.pth")
OUTPUT_PATH = Path("model/unet_quantized.pth")

# ── U-Net Architecture (same as train.py) ──────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
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

# ── Quantization ───────────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH}...")
if not MODEL_PATH.exists():
    print(f"ERROR: {MODEL_PATH} not found")
    exit(1)

model = UNet()
model.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu", weights_only=True))
model.eval()

print("✓ Model loaded\n")

# Original size
orig_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"Original model size: {orig_size:.1f} MB")

# Dynamic quantization (int8)
print("Quantizing to int8...")
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

# Save quantized model
print(f"Saving to {OUTPUT_PATH}...")
torch.save(quantized_model.state_dict(), str(OUTPUT_PATH))

# Check size
quant_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
reduction = (1 - quant_size / orig_size) * 100

print(f"\n✓ Quantized model size: {quant_size:.1f} MB")
print(f"✓ Size reduction: {reduction:.1f}% ({orig_size:.1f} → {quant_size:.1f} MB)")
print(f"\n✓ Quantized model saved: {OUTPUT_PATH}")
print(f"\nExpected FPS improvement on Jetson Nano: 2-3x")
print(f"\nUpdate mask_inference/inference_camera.py:")
print(f"  MODEL_PATH = Path('../model/unet_quantized.pth')")
