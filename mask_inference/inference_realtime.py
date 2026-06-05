#!/usr/bin/env python3
"""
Charge images depuis un dossier local + inférence U-Net en temps réel.
Affiche image + masque côte à côte (sans DepthAI).

Requirements:
    pip install opencv-python numpy torch torchvision

Usage:
    python inference_realtime.py --input /path/to/images
    ou
    python inference_realtime.py  (utilise ./test_images par défaut)

Controls:
    q        - Quit
    s        - Save snapshot (image + mask)
    n        - Next image
    p        - Previous image
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import argparse

import cv2
import numpy as np
import torch
from torchvision import transforms

# ── Configuration ──────────────────────────────────────────────────────────────
PREVIEW_SIZE = (640, 480)
SNAPSHOT_DIR = Path("snapshots")

# ── Load Model ─────────────────────────────────────────────────────────────────
def load_model(model_path):
    print(f"Loading model from {model_path}...")
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)
    
    model = torch.jit.load(str(model_path))
    model.eval()
    print("✓ Model loaded\n")
    return model

# Transforms pour U-Net (347x256)
transform = transforms.Compose([
    transforms.Resize((256, 347)),
    transforms.ToTensor(),
])

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_images(image_dir):
    """Charge toutes les images du dossier"""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"ERROR: Directory {image_dir} not found")
        sys.exit(1)
    
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    images = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in extensions])
    
    if not images:
        print(f"ERROR: No images found in {image_dir}")
        sys.exit(1)
    
    print(f"✓ Loaded {len(images)} images from {image_dir}\n")
    return images

def infer_mask(model, frame_gray):
    """Inférence U-Net sur une image"""
    # Redimensionne à 347x256 pour le modèle
    frame_model = cv2.resize(frame_gray, (347, 256))
    frame_model_rgb = cv2.cvtColor(frame_model, cv2.COLOR_GRAY2RGB)
    
    # Applique transforms
    tensor = transform(frame_model_rgb).unsqueeze(0)
    
    with torch.no_grad():
        mask_logits = model(tensor)
        mask = (torch.sigmoid(mask_logits) > 0.5).float().squeeze(0).cpu().numpy()
    
    # Redimensionne masque à taille preview (640x480)
    mask_single = mask[0]  # Prend premier channel
    mask_3ch = np.stack([mask_single, mask_single, mask_single], axis=-1) * 255
    mask_3ch = cv2.resize(mask_3ch.astype(np.uint8), PREVIEW_SIZE)
    
    return mask_3ch

def save_snapshot(frame_rgb, mask_img):
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    frame_path = SNAPSHOT_DIR / f"{ts}_image.png"
    mask_path = SNAPSHOT_DIR / f"{ts}_mask.png"
    
    cv2.imwrite(str(frame_path), frame_rgb)
    cv2.imwrite(str(mask_path), mask_img)
    
    print(f"✓ Saved: {frame_path}")
    print(f"✓ Saved: {mask_path}")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="U-Net inference on local images")
    parser.add_argument("--input", type=str, default="./test_images", help="Image directory")
    parser.add_argument("--model", type=str, default="../model/unet_scripted.pt", help="Model path")
    args = parser.parse_args()
    
    # Charge modèle
    model_path = Path(args.model)
    model = load_model(model_path)
    
    # Charge images
    images = load_images(args.input)
    
    current_idx = 0
    fps_counter = {"count": 0, "t": time.monotonic(), "fps": 0.0}
    
    print(f"Press  q=quit  s=snapshot  n=next  p=previous\n")
    
    while True:
        # Charge image courante
        img_path = images[current_idx]
        frame_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if frame_gray is None:
            print(f"ERROR: Could not load {img_path}")
            current_idx = (current_idx + 1) % len(images)
            continue
        
        # ── FPS counter ────
        fps_counter["count"] += 1
        now = time.monotonic()
        if now - fps_counter["t"] >= 1.0:
            fps_counter["fps"] = fps_counter["count"] / (now - fps_counter["t"])
            fps_counter["count"] = 0
            fps_counter["t"] = now
        
        # Prépare image RGB
        frame_rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        frame_rgb = cv2.resize(frame_rgb, PREVIEW_SIZE)
        
        # ── Inférence U-Net ────
        mask_3ch = infer_mask(model, frame_gray)
        
        # ── Draw FPS et filename ────
        fps_text = f"{fps_counter['fps']:.1f} FPS"
        filename = img_path.name
        cv2.putText(frame_rgb, fps_text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame_rgb, filename, (8, PREVIEW_SIZE[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(mask_3ch, fps_text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(mask_3ch, f"{current_idx+1}/{len(images)}", (8, PREVIEW_SIZE[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # ── Display ────
        display = np.hstack([frame_rgb, mask_3ch])
        cv2.imshow("Image (LEFT) | U-Net Mask (RIGHT)", display)
        
        # ── Keyboard input ────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            print("\nSaving snapshot...")
            save_snapshot(frame_rgb, mask_3ch)
        elif key == ord('n'):
            current_idx = (current_idx + 1) % len(images)
            print(f"→ Image {current_idx+1}/{len(images)}: {images[current_idx].name}")
        elif key == ord('p'):
            current_idx = (current_idx - 1) % len(images)
            print(f"← Image {current_idx+1}/{len(images)}: {images[current_idx].name}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
