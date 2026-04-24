"""
train.py - Entraînement U-Net pour la segmentation de lignes de piste
Optimisé pour CPU (léger, batch size petit, epochs raisonnables)

Usage : python3 train.py
Le modèle entraîné est sauvegardé dans : model/unet.pth
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_DIR = "dataset_final"   # dossier produit par prepare_dataset.py
MODEL_DIR   = "model"
IMG_SIZE    = (256, 256)         # redimensionnement (carré, plus simple pour le CNN)
BATCH_SIZE  = 4                  # petit batch → moins de RAM
EPOCHS      = 20                 # 20 epochs = bon compromis qualité/temps sur CPU
LR          = 1e-3               # learning rate
TRAIN_SPLIT = 0.85               # 85% train, 15% validation
SEED        = 42
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
os.makedirs(MODEL_DIR, exist_ok=True)
DEVICE = torch.device("cpu")
print(f"Entraînement sur : {DEVICE}")


# ── DATASET ───────────────────────────────────────────────────────────────────

class SegmentationDataset(Dataset):
    """
    Charge les paires image/masque depuis dataset_final/.
    - image  : RGB normalisé [0, 1], shape (3, H, W)
    - masque : binaire [0 ou 1], shape (1, H, W)
    """
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Charge et redimensionne l'image RGB
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = image.resize(IMG_SIZE, Image.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0  # normalise [0,1]
        image = torch.from_numpy(image).permute(2, 0, 1)   # (H,W,3) → (3,H,W)

        # Charge et redimensionne le masque
        mask = Image.open(self.mask_paths[idx]).convert("L")
        mask = mask.resize(IMG_SIZE, Image.NEAREST)
        mask = np.array(mask, dtype=np.float32) / 255.0    # 0.0 ou 1.0
        mask = torch.from_numpy(mask).unsqueeze(0)          # (H,W) → (1,H,W)

        return image, mask


def build_dataloaders():
    images_dir = os.path.join(DATASET_DIR, "images")
    masks_dir  = os.path.join(DATASET_DIR, "masks")

    fnames = sorted(os.listdir(images_dir))
    image_paths = [os.path.join(images_dir, f) for f in fnames]
    mask_paths  = [os.path.join(masks_dir,  f) for f in fnames]

    # Mélange et split train/val
    indices = list(range(len(fnames)))
    np.random.seed(SEED)
    np.random.shuffle(indices)
    split = int(len(indices) * TRAIN_SPLIT)
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = SegmentationDataset(
        [image_paths[i] for i in train_idx],
        [mask_paths[i]  for i in train_idx]
    )
    val_ds = SegmentationDataset(
        [image_paths[i] for i in val_idx],
        [mask_paths[i]  for i in val_idx]
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train : {len(train_ds)} images | Val : {len(val_ds)} images")
    return train_loader, val_loader


# ── U-NET ─────────────────────────────────────────────────────────────────────
# Architecture en forme de U :
#   Encoder (descend) : extrait des features de plus en plus abstraites
#   Bottleneck        : couche la plus compressée
#   Decoder (remonte) : reconstruit le masque pixel par pixel
#   Skip connections  : relie encoder et decoder pour garder les détails fins

class ConvBlock(nn.Module):
    """Deux convolutions 3x3 + BatchNorm + ReLU — brique de base du U-Net"""
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
    """
    U-Net léger (features réduits pour CPU).
    Entrée  : (batch, 3, 256, 256)   — image RGB
    Sortie  : (batch, 1, 256, 256)   — masque binaire (sigmoid → [0,1])
    """
    def __init__(self, features=(16, 32, 64, 128)):
        super().__init__()

        # ── Encoder (descend, MaxPool divise par 2 à chaque étage) ──
        self.enc1 = ConvBlock(3,           features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])

        self.pool = nn.MaxPool2d(2)

        # ── Bottleneck ──
        self.bottleneck = ConvBlock(features[3], features[3] * 2)

        # ── Decoder (remonte, ConvTranspose double la résolution) ──
        self.up4   = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.dec4  = ConvBlock(features[3] * 2, features[3])  # *2 car skip connection

        self.up3   = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3  = ConvBlock(features[2] * 2, features[2])

        self.up2   = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2  = ConvBlock(features[1] * 2, features[1])

        self.up1   = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1  = ConvBlock(features[0] * 2, features[0])

        # ── Sortie : convolution 1x1 → 1 canal, sigmoid pour [0,1] ──
        self.final = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder avec skip connections (concat encoder + decoder)
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.final(d1))


# ── LOSS ──────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Dice Loss — meilleure que BCE pour la segmentation car robuste
    aux déséquilibres (peu de pixels blancs vs beaucoup de pixels noirs).
    Dice = 2 * intersection / union → Loss = 1 - Dice
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred   = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """BCE + Dice : combine les avantages des deux"""
    def __init__(self):
        super().__init__()
        self.bce  = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return 0.5 * self.bce(pred, target) + 0.5 * self.dice(pred, target)


# ── MÉTRIQUES ─────────────────────────────────────────────────────────────────

def dice_score(pred, target, threshold=0.5):
    """Calcule le Dice Score (0 = nul, 1 = parfait)"""
    pred   = (pred > threshold).float().view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return ((2.0 * intersection + 1.0) / (pred.sum() + target.sum() + 1.0)).item()


# ── ENTRAÎNEMENT ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_dice = 0.0, 0.0
    for images, masks in loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_dice += dice_score(preds.detach(), masks)
    n = len(loader)
    return total_loss / n, total_dice / n


def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, total_dice = 0.0, 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            preds = model(images)
            total_loss += criterion(preds, masks).item()
            total_dice += dice_score(preds, masks)
    n = len(loader)
    return total_loss / n, total_dice / n


def main():
    train_loader, val_loader = build_dataloaders()

    model     = UNet().to(DEVICE)
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Affiche le nombre de paramètres du modèle
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modèle U-Net : {n_params:,} paramètres")

    best_dice = 0.0
    print(f"\nDébut entraînement ({EPOCHS} epochs)...\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_dice = train_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_dice   = val_epoch(model, val_loader, criterion)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train loss: {train_loss:.4f}  Dice: {train_dice:.4f} | "
              f"Val loss: {val_loss:.4f}  Dice: {val_dice:.4f} | "
              f"{elapsed:.0f}s")

        # Sauvegarde le meilleur modèle
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "unet.pth"))
            print(f"  --> Meilleur modèle sauvegardé (Dice: {best_dice:.4f})")

    print(f"\nEntraînement terminé. Meilleur Dice val : {best_dice:.4f}")
    print(f"Modèle sauvegardé dans : {MODEL_DIR}/unet.pth")


if __name__ == "__main__":
    main()