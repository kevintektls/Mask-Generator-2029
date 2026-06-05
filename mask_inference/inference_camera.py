#!/usr/bin/env python3
"""
Capture caméra OAK-D + inférence U-Net en temps réel.
Affiche image + masque (lignes blanches) côte à côte.
"""
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

try:
    import depthai as dai
except ImportError:
    print("DepthAI not installed. Run: pip install depthai")
    sys.exit(1)

# Config
MODEL_PATH = Path("../model/unet.pth")
DISPLAY_W, DISPLAY_H = 640, 480   # taille d'affichage finale (les deux panels)
UNET_W, UNET_H = 347, 256         # taille d'entrée du U-Net
FPS = 30

# --- Définition minimale U-Net (à remplacer par ton architecture si différente) ---
# Si tu as la classe UNet dans un module séparé, importe-la ici à la place.
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs, self.ups, self.pool = nn.ModuleList(), nn.ModuleList(), nn.MaxPool2d(2, 2)
        for f in features:
            self.downs.append(DoubleConv(in_channels, f)); in_channels = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))
        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x); skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[i // 2]
            if x.shape != s.shape:
                x = torch.nn.functional.interpolate(x, size=s.shape[2:])
            x = self.ups[i + 1](torch.cat([s, x], dim=1))
        return self.final(x)
# ---------------------------------------------------------------------------------

print(f"Loading U-Net from {MODEL_PATH}...")
model = UNet()
model.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu", weights_only=True))
model.eval()
print("✓ Model loaded\n")

transform = transforms.Compose([
    transforms.Resize((UNET_H, UNET_W)),
    transforms.ToTensor(),
])

def build_pipeline():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.MonoCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)   # pas de DeprecationWarning
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    cam.setFps(FPS)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("left")
    xout.input.setBlocking(False)
    xout.input.setQueueSize(2)   # 2 suffit sur Jetson, évite la RAM inutile
    cam.out.link(xout.input)
    return pipeline

def run_inference(frame_gray):
    """frame_gray : np.uint8 H×W (mono). Retourne masque np.uint8 H×W (0/255)."""
    pil = Image.fromarray(cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB))
    tensor = transform(pil).unsqueeze(0)   # (1, 3, 256, 347)
    with torch.no_grad():
        logits = model(tensor)
    mask = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy()
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

            # FPS
            count += 1
            now = time.monotonic()
            if now - t0 >= 1.0:
                fps = count / (now - t0); count = 0; t0 = now

            # Frame caméra → BGR affiché
            raw = pkt.getCvFrame()                          # uint8, H×W (mono)
            frame_bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            frame_bgr = cv2.resize(frame_bgr, (DISPLAY_W, DISPLAY_H))  # 640×480

            # Inférence
            mask_u8 = run_inference(raw)                    # 256×347, uint8
            mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
            mask_bgr = cv2.resize(mask_bgr, (DISPLAY_W, DISPLAY_H))    # 640×480

            # OSD FPS
            label = f"{fps:.1f} FPS"
            for img in (frame_bgr, mask_bgr):
                cv2.putText(img, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Affichage côte à côte — même shape garantie
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