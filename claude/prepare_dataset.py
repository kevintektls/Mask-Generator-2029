"""
Étape 1 : Préparation du dataset
- Charge les images capturées par DatasetCapture.cs (masques déjà binarisés)
- Complète avec les anciennes withFilter/withoutFilter si disponibles
- Applique de la data augmentation pour atteindre TARGET_PAIRS
- Sauvegarde dans dataset_final/images/ et dataset_final/masks/
"""

import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random

# ── CONFIG ──────────────────────────────────────────────────────────────────
UNITY_DATASET_DIR  = "Dataset"                # dossier Assets/Dataset/ copié ici
RAW_DIR    = "unitySimulator/withFilter"    # anciennes images (optionnel)
FILTER_DIR = "unitySimulator/withFilter"          # anciens masques  (optionnel)
OUT_DIR          = "dataset_final"
BINARIZE_THRESHOLD = 200
TARGET_PAIRS     = 5000
SEED             = 42
# ────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)

os.makedirs(f"{OUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUT_DIR}/masks",  exist_ok=True)


def load_unity_dataset(base_dir):
    images_dir = os.path.join(base_dir, "images")
    masks_dir  = os.path.join(base_dir, "masks")
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"[SKIP] '{base_dir}' introuvable.")
        return []
    pairs = []
    for fname in sorted(os.listdir(images_dir)):
        img_path  = os.path.join(images_dir, fname)
        mask_path = os.path.join(masks_dir, fname)
        if not os.path.exists(mask_path):
            continue
        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path).convert("L"))
        mask  = np.where(mask > BINARIZE_THRESHOLD, 255, 0).astype(np.uint8)
        pairs.append((image, mask, fname))
    print(f"  -> {len(pairs)} paires depuis '{base_dir}'")
    return pairs


def load_old_dataset(raw_dir, filter_dir):
    if not os.path.exists(raw_dir) or not os.path.exists(filter_dir):
        print(f"[SKIP] Anciens dossiers introuvables.")
        return []
    pairs = []
    for fname in sorted(os.listdir(raw_dir)):
        raw_path    = os.path.join(raw_dir, fname)
        filter_path = os.path.join(filter_dir, fname)
        if not os.path.exists(filter_path):
            continue
        image    = np.array(Image.open(raw_path).convert("RGB"))
        mask_raw = np.array(Image.open(filter_path).convert("L"))
        mask     = np.where(mask_raw > BINARIZE_THRESHOLD, 255, 0).astype(np.uint8)
        pairs.append((image, mask, f"old_{fname}"))
    print(f"  -> {len(pairs)} paires depuis les anciens dossiers")
    return pairs


def augment_pair(image, mask):
    img_pil  = Image.fromarray(image)
    mask_pil = Image.fromarray(mask)

    if random.random() < 0.5:
        img_pil  = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
        mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)

    if random.random() < 0.7:
        angle    = random.uniform(-15, 15)
        img_pil  = img_pil.rotate(angle, fillcolor=0)
        mask_pil = mask_pil.rotate(angle, fillcolor=0)

    if random.random() < 0.5:
        w, h  = img_pil.size
        ratio = random.uniform(0.8, 1.0)
        nw, nh = int(w * ratio), int(h * ratio)
        left  = random.randint(0, w - nw)
        top   = random.randint(0, h - nh)
        box   = (left, top, left + nw, top + nh)
        img_pil  = img_pil.crop(box).resize((w, h), Image.BILINEAR)
        mask_pil = mask_pil.crop(box).resize((w, h), Image.NEAREST)

    if random.random() < 0.6:
        img_pil = ImageEnhance.Brightness(img_pil).enhance(random.uniform(0.6, 1.4))

    if random.random() < 0.5:
        img_pil = ImageEnhance.Contrast(img_pil).enhance(random.uniform(0.7, 1.3))

    if random.random() < 0.4:
        w, h = img_pil.size
        dx, dy = random.randint(-20, 20), random.randint(-10, 10)
        img_pil  = img_pil.transform((w, h), Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=0)
        mask_pil = mask_pil.transform((w, h), Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=0)

    if random.random() < 0.3:
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    return np.array(img_pil), np.array(mask_pil)


def main():
    print("Chargement des données...")
    pairs  = load_unity_dataset(UNITY_DATASET_DIR)
    pairs += load_old_dataset(RAW_DIR, FILTER_DIR)

    if not pairs:
        print("Aucune paire trouvée. Vérifie les chemins dans la config.")
        return

    print(f"Total paires originales : {len(pairs)}")

    saved = 0
    for image, mask, fname in pairs:
        stem = os.path.splitext(fname)[0]
        Image.fromarray(image).save(f"{OUT_DIR}/images/{stem}_orig.png")
        Image.fromarray(mask).save(f"{OUT_DIR}/masks/{stem}_orig.png")
        saved += 1
    print(f"Originaux sauvegardés : {saved}")

    aug_id = 0
    while saved < TARGET_PAIRS:
        image, mask, fname = random.choice(pairs)
        stem = os.path.splitext(fname)[0]
        aug_image, aug_mask = augment_pair(image.copy(), mask.copy())
        Image.fromarray(aug_image).save(f"{OUT_DIR}/images/{stem}_aug{aug_id:05d}.png")
        Image.fromarray(aug_mask).save(f"{OUT_DIR}/masks/{stem}_aug{aug_id:05d}.png")
        saved  += 1
        aug_id += 1
        if saved % 500 == 0:
            print(f"  ... {saved}/{TARGET_PAIRS}")

    print(f"\nDataset pret : {saved} paires dans '{OUT_DIR}/'")


if __name__ == "__main__":
    main()