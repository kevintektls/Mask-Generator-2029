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
from torchvision import transforms

try:
    import depthai as dai
except ImportError:
    print("DepthAI not installed. Run: pip install depthai")
    sys.exit(1)

# Config
MODEL_PATH = Path("../model/unet_scripted.pt")
PREVIEW_SIZE = (640, 480)
FPS = 30

# Charge modèle
print(f"Loading U-Net from {MODEL_PATH}...")
model = torch.jit.load(str(MODEL_PATH))
model.eval()
print("✓ Model loaded\n")

# Transforms U-Net (347x256)
transform = transforms.Compose([
    transforms.Resize((256, 347)),
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
            frame_resized = cv2.resize(frame, (347, 256))
            tensor = transform(cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2RGB)).unsqueeze(0)
            
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