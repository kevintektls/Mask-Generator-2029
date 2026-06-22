#!/usr/bin/env python3
"""
Entraînement du modèle de pilotage par Behavioral Cloning.

Entrée dataset :
dataset/
├── driving_log.csv
└── images/
    ├── line_xxx.png
    └── ...

CSV attendu :
image_path,servo,duty

Le modèle prédit uniquement le servo.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from model_def import BehavioralCloningCNN
from vision_preprocess import resize_for_model


# ── CONFIG ────────────────────────────────────────────────────────────────────

DATASET_DIR = Path("dataset")
CSV_FILE = DATASET_DIR / "driving_log.csv"

BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 2e-3

MODEL_OUT = Path("pilot_model.pth")
LOSS_PLOT_OUT = Path("loss_plot.png")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── DATASET ───────────────────────────────────────────────────────────────────

class RobotCarDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, base_dir: Path):
        self.df = dataframe.reset_index(drop=True)
        self.base_dir = Path(base_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        rel_path = str(row["image_path"]).replace("\\", "/")
        img_path = self.base_dir / rel_path

        mask = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Impossible de charger l'image : {img_path}")

        mask_resized = resize_for_model(mask)

        img_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0) / 255.0
        servo = torch.tensor(float(row["servo"]), dtype=torch.float32)

        return img_tensor, servo


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[INFO] Entraînement sur : {DEVICE}")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    if not CSV_FILE.exists():
        print(f"[ERROR] Fichier introuvable : {CSV_FILE}")
        sys.exit(1)

    df = pd.read_csv(CSV_FILE)

    required_cols = {"image_path", "servo", "duty"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ERROR] Colonnes manquantes dans le CSV : {missing}")
        sys.exit(1)

    # On garde seulement les moments où la voiture avance/recul vraiment
    df = df[df["duty"].abs() > 0.01].reset_index(drop=True)

    # Sécurité : servo doit être dans [0, 1]
    df = df[(df["servo"] >= 0.0) & (df["servo"] <= 1.0)].reset_index(drop=True)

    if len(df) < 100:
        print(f"[WARNING] Dataset très petit : {len(df)} images. Le modèle risque d'être nul.")

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    print(f"[DATA] Train: {len(train_df)} images | Val: {len(val_df)} images")

    train_dataset = RobotCarDataset(train_df, DATASET_DIR)
    val_dataset = RobotCarDataset(val_df, DATASET_DIR)

    num_workers = 0 if os.name == "nt" else 4
    pin_memory = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = BehavioralCloningCNN().to(DEVICE)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.2,
        patience=5,
    )

    best_val_loss = float("inf")
    patience_early_stopping = 15
    patience_counter = 0

    train_losses = []
    val_losses = []

    print("\n--- DÉBUT ENTRAÎNEMENT ---")

    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0

        for images, targets in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, targets)

                running_val_loss += loss.item() * images.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
            f"| LR: {current_lr:.2e} "
            f"| Train: {epoch_train_loss:.6f} "
            f"| Val: {epoch_val_loss:.6f}"
        )

        scheduler.step(epoch_val_loss)

        # Sauvegarde UNIQUEMENT du meilleur modèle
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"   ↳ Nouveau meilleur modèle sauvegardé : {MODEL_OUT}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_early_stopping:
            print(f"\n[STOP] Early stopping à l'époque {epoch + 1}")
            break

    print(f"\n[OK] Meilleure Val Loss : {best_val_loss:.6f}")
    print(f"[OK] Meilleur modèle : {MODEL_OUT}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Évolution de la perte")
    plt.savefig(LOSS_PLOT_OUT)
    print(f"[OK] Courbe sauvegardée : {LOSS_PLOT_OUT}")


if __name__ == "__main__":
    main()
