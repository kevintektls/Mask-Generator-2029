#!/usr/bin/env python3
"""Entraîne le modèle comportemental multimodal caméra + LiDAR."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_DIR / "scripts"
sys.path.insert(0, str(PROJECT_DIR / "model"))
sys.path.insert(0, str(SCRIPT_DIR))
from camera_lidar_model import CameraLidarBehavioralCloning
from vision_preprocess import resize_for_model

DEFAULT_CSV = SCRIPT_DIR / "dataset" / "driving_log.csv"
HISTORY = 3

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    plt = None
    print(f"[WARNING] Matplotlib indisponible ({exc}); le graphique sera remplacé par un CSV.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="CSV caméra + LiDAR produit par scripts/Behavioral_Cloning.py")
    parser.add_argument("--scan-column", default=None)
    parser.add_argument("--max-range", type=float, default=12.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model-out", type=Path, default=PROJECT_DIR / "model/camera_lidar_model.pth")
    parser.add_argument("--loss-plot-out", type=Path, default=PROJECT_DIR / "model/camera_lidar_loss_plot.png")
    return parser.parse_args()


def select_device(preference: str) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if preference == "cuda":
            raise RuntimeError("CUDA demandé, mais PyTorch ne détecte aucun GPU CUDA disponible.")
        print("[WARNING] CUDA indisponible; entraînement sur CPU.")
        return torch.device("cpu")
    try:
        torch.empty(1, device="cuda")
        torch.cuda.synchronize()
        return torch.device("cuda")
    except Exception as exc:
        if preference == "cuda":
            raise RuntimeError(f"CUDA demandé mais inutilisable : {exc}") from exc
        print(f"[WARNING] CUDA détecté mais inutilisable ({exc}); entraînement sur CPU.")
        return torch.device("cpu")


def find_scan_columns(headers, requested):
    if requested:
        if requested not in headers:
            raise ValueError(f"Colonne LiDAR introuvable : {requested}")
        return requested, []
    for candidate in ("lidar", "lidar_data", "scan", "ranges"):
        if candidate in headers:
            return candidate, []
    indexed = []
    for column in headers:
        match = re.fullmatch(r"lidar_(\d+)", str(column))
        if match:
            indexed.append((int(match.group(1)), column))
    if indexed:
        indexed.sort()
        if [index for index, _ in indexed] != list(range(len(indexed))):
            raise ValueError("Les colonnes lidar_* doivent être indexées de lidar_0 sans trous.")
        return None, [column for _, column in indexed]
    raise ValueError("Aucune colonne LiDAR trouvée dans le CSV.")


def parse_scan(value, row_number):
    try:
        scan = np.asarray(json.loads(value), dtype=np.float32).reshape(-1)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Scan LiDAR invalide ligne {row_number}: {exc}") from exc
    if scan.size == 0:
        raise ValueError(f"Scan LiDAR vide ligne {row_number}.")
    return scan


def load_csv_dataset(csv_path: Path, requested_scan_column: str | None):
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        missing = {"image_path", "servo", "duty"} - set(headers)
        if missing:
            raise ValueError(f"Colonnes manquantes dans le CSV caméra + LiDAR : {sorted(missing)}")
        scan_column, scan_columns = find_scan_columns(headers, requested_scan_column)
        for row_number, row in enumerate(reader, start=2):
            try:
                servo, duty = float(row.get("servo", "")), float(row.get("duty", ""))
            except (TypeError, ValueError):
                continue
            image_path = (row.get("image_path") or "").strip()
            if not math.isfinite(servo) or not math.isfinite(duty) or abs(duty) <= 0.01 or not 0 <= servo <= 1 or not image_path:
                continue
            if scan_column is not None:
                scan = parse_scan(row.get(scan_column, ""), row_number)
            else:
                try:
                    scan = np.asarray([float(row.get(column, "nan")) for column in scan_columns], dtype=np.float32)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Scan LiDAR invalide ligne {row_number}: {exc}") from exc
            image = Path(image_path)
            if not image.is_absolute():
                image = csv_path.parent / image
            rows.append((image, scan, servo))
    if len(rows) < 2:
        raise ValueError("Il faut au moins deux lignes valides avec image, LiDAR, servo et duty actif.")
    ray_count = rows[0][1].size
    for index, (image, scan, _servo) in enumerate(rows, start=2):
        if scan.size != ray_count:
            raise ValueError(f"Scan à l'échantillon {index} : {scan.size} rayons au lieu de {ray_count}.")
        if cv2.imread(str(image), cv2.IMREAD_GRAYSCALE) is None:
            raise ValueError(f"Image introuvable ou illisible : {image}")
    return rows, ray_count


class CameraLidarDataset(Dataset):
    def __init__(self, rows, max_range):
        self.rows = rows
        self.max_range = max_range

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        first = max(0, idx - HISTORY + 1)
        history = self.rows[first:idx + 1]
        if len(history) < HISTORY:
            history = [self.rows[0]] * (HISTORY - len(history)) + history
        images, scans = [], []
        for image_path, scan, _servo in history:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"Image devenue illisible pendant l'entraînement : {image_path}")
            images.append(resize_for_model(image).astype(np.float32) / 255.0)
            clean = np.nan_to_num(scan, nan=self.max_range, posinf=self.max_range, neginf=0.0)
            scans.append(np.clip(clean, 0.0, self.max_range) / self.max_range)
        image_tensor = torch.from_numpy(np.stack(images))
        scan_tensor = torch.from_numpy(np.stack(scans).astype(np.float32).reshape(-1))
        target = torch.tensor(self.rows[idx][2], dtype=torch.float32)
        return image_tensor, scan_tensor, target


def main():
    args = parse_args()
    if args.max_range <= 0:
        print("[ERROR] --max-range doit être supérieur à zéro.")
        return 1
    try:
        device = select_device(args.device)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1
    print(f"[INFO] Entraînement caméra + LiDAR sur : {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    if not args.csv.exists():
        print(f"[ERROR] Fichier introuvable : {args.csv}")
        return 1
    try:
        rows, ray_count = load_csv_dataset(args.csv, args.scan_column)
    except (OSError, ValueError) as exc:
        print(f"[ERROR] Dataset invalide : {exc}")
        return 1

    split_index = int(len(rows) * 0.8)
    if split_index == 0 or split_index >= len(rows):
        print("[ERROR] Dataset trop petit pour séparer entraînement et validation.")
        return 1
    train_ds = CameraLidarDataset(rows[:split_index], args.max_range)
    val_ds = CameraLidarDataset(rows[split_index:], args.max_range)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CameraLidarBehavioralCloning(ray_count, HISTORY)
    try:
        model = model.to(device)
    except Exception as exc:
        if device.type != "cuda" or args.device == "cuda":
            print(f"[ERROR] Impossible de déplacer le modèle sur {device}: {exc}")
            return 1
        print(f"[WARNING] CUDA est devenu indisponible ({exc}); reprise sur CPU.")
        device = torch.device("cpu")
        model = CameraLidarBehavioralCloning(ray_count, HISTORY)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5)
    best_val_loss, patience, patience_counter = math.inf, 15, 0
    train_losses, val_losses = [], []
    print(f"[DATA] {len(rows)} échantillons | {len(train_ds)} train | {len(val_ds)} validation | {ray_count} rayons/scan | {HISTORY} images + scans")
    print("\n--- DÉBUT ENTRAÎNEMENT ---")
    for epoch in range(args.epochs):
        model.train()
        train_total = 0.0
        for images, scans, targets in train_loader:
            images, scans, targets = images.to(device), scans.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images, scans), targets)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * images.size(0)
        train_loss = train_total / len(train_ds)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for images, scans, targets in val_loader:
                images, scans, targets = images.to(device), scans.to(device), targets.to(device)
                val_total += criterion(model(images, scans), targets).item() * images.size(0)
        val_loss = val_total / len(val_ds)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch + 1:03d}/{args.epochs}] | LR: {optimizer.param_groups[0]['lr']:.2e} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            args.model_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_type": "camera_lidar",
                "model_state_dict": model.state_dict(),
                "ray_count": ray_count,
                "history_scans": HISTORY,
                "history_images": HISTORY,
                "max_range": args.max_range,
            }, args.model_out)
            print(f"   ↳ Meilleur modèle sauvegardé : {args.model_out}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[STOP] Early stopping à l'époque {epoch + 1}")
                break

    args.loss_plot_out.parent.mkdir(parents=True, exist_ok=True)
    if plt is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.title("Perte — caméra + LiDAR")
        plt.savefig(args.loss_plot_out)
        print(f"[OK] Courbe : {args.loss_plot_out}")
    else:
        loss_csv = args.loss_plot_out.with_suffix(".csv")
        with loss_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "train_mse", "validation_mse"])
            writer.writerows((epoch, train_loss, val_loss) for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1))
        print(f"[OK] Historique des pertes : {loss_csv}")
    print(f"[OK] Meilleure Val Loss : {best_val_loss:.6f}")
    print(f"[OK] Modèle caméra + LiDAR : {args.model_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
