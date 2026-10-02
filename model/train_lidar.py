#!/usr/bin/env python3
"""Entraîne un modèle de pilotage par clonage comportemental à partir du LiDAR.

Format CSV attendu :
    timestamp,image_path,servo,duty,lidar_timestamp,lidar
    ...,images/frame.png,0.52,0.12,...,"[0.8, 0.7, 0.6, ...]"

La colonne ``lidar`` peut aussi s'appeler ``lidar_data`` ou ``scan``. À la
place d'un vecteur JSON, le CSV peut contenir une colonne par rayon nommée
``lidar_0``, ``lidar_1``, ... Les colonnes servo et duty restent nécessaires.

Les trois scans successifs sont concaténés en entrée du réseau. Les distances
sont ramenées dans [0, 1] avec --max-range (12 mètres par défaut pour le D500).
Les colonnes image et timestamps sont conservées dans le CSV mais ne sont pas
utilisées par ce modèle LiDAR.

Exécution depuis la racine du dépôt :
    python3 model/train_lidar.py
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_DIR / "scripts/dataset/driving_log.csv"

# Matplotlib système peut être installé sans ses extensions natives compilées.
# Il ne doit pas empêcher l'entraînement ; dans ce cas on exporte les pertes en CSV.
try:
    import matplotlib.pyplot as plt
except Exception as exc:
    plt = None
    print(f"[WARNING] Matplotlib indisponible ({exc}); le graphique sera remplacé par un CSV.")


class LidarBehavioralCloningMLP(nn.Module):
    """Petit réseau dense compatible avec un nombre de rayons variable."""

    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, scans: torch.Tensor) -> torch.Tensor:
        return self.network(scans).squeeze(1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="CSV caméra + LiDAR produit par scripts/Behavioral_Cloning.py")
    parser.add_argument("--scan-column", default=None,
                        help="Colonne contenant le vecteur JSON (auto-détection par défaut)")
    parser.add_argument("--max-range", type=float, default=12.0,
                        help="Portée LiDAR maximale en mètres pour normaliser les distances")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--model-out", type=Path, default=PROJECT_DIR / "model/lidar_model.pth")
    parser.add_argument("--loss-plot-out", type=Path, default=PROJECT_DIR / "model/lidar_loss_plot.png")
    return parser.parse_args()


def find_scan_columns(df: pd.DataFrame, requested: str | None) -> tuple[str | None, list[str]]:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Colonne LiDAR introuvable : {requested}")
        return requested, []

    for candidate in ("lidar", "lidar_data", "scan", "ranges"):
        if candidate in df.columns:
            return candidate, []

    indexed = []
    for column in df.columns:
        match = re.fullmatch(r"lidar_(\d+)", str(column))
        if match:
            indexed.append((int(match.group(1)), column))
    if indexed:
        indexed.sort()
        indexes = [index for index, _ in indexed]
        if indexes != list(range(len(indexes))):
            raise ValueError("Les colonnes lidar_* doivent être indexées sans trous depuis lidar_0.")
        return None, [column for _, column in indexed]

    raise ValueError(
        "Aucune mesure LiDAR trouvée. Utilise une colonne lidar (vecteur JSON) "
        "ou des colonnes lidar_0, lidar_1, ..."
    )


def parse_scan(value, row_number: int) -> np.ndarray:
    try:
        values = json.loads(value) if isinstance(value, str) else value
        scan = np.asarray(values, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Scan LiDAR invalide à la ligne CSV {row_number}: {exc}") from exc
    if scan.size == 0:
        raise ValueError(f"Scan LiDAR vide à la ligne CSV {row_number}.")
    return scan


class LidarDataset(Dataset):
    def __init__(self, scans: np.ndarray, servos: np.ndarray, max_range: float):
        self.scans = torch.from_numpy(scans.astype(np.float32))
        self.servos = torch.from_numpy(servos.astype(np.float32))
        self.max_range = max_range

    def __len__(self):
        return len(self.servos)

    def __getitem__(self, idx):
        # En début de séquence, répéter le premier scan disponible.
        first = max(0, idx - 2)
        history = self.scans[first:idx + 1]
        if history.shape[0] < 3:
            history = torch.cat((self.scans[0:1].expand(3 - history.shape[0], -1), history), dim=0)
        # Les mesures invalides/infini représentent une absence de retour : portée max.
        history = torch.nan_to_num(history, nan=self.max_range,
                                   posinf=self.max_range, neginf=0.0)
        history = torch.clamp(history, min=0.0, max=self.max_range) / self.max_range
        return history.flatten(), self.servos[idx]


def main():
    args = parse_args()
    if args.max_range <= 0:
        print("[ERROR] --max-range doit être supérieur à zéro.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Entraînement LiDAR sur : {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    if not args.csv.exists():
        print(f"[ERROR] Fichier introuvable : {args.csv}")
        return 1

    df = pd.read_csv(args.csv)
    required = {"servo", "duty"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] Colonnes manquantes dans le CSV : {sorted(missing)}")
        return 1
    scan_column, scan_columns = find_scan_columns(df, args.scan_column)

    # Préserve l'ordre des lignes du CSV : les lignes doivent être chronologiques.
    df = df.copy()
    df["servo"] = pd.to_numeric(df["servo"], errors="coerce")
    df["duty"] = pd.to_numeric(df["duty"], errors="coerce")
    df = df.dropna(subset=["servo", "duty"])
    df = df[(df["duty"].abs() > 0.01) & df["servo"].between(0.0, 1.0)].reset_index(drop=True)

    if len(df) < 2:
        print("[ERROR] Il faut au moins deux lignes valides (servo dans [0,1], duty non nul).")
        return 1

    if scan_column is not None:
        scans = [parse_scan(value, i + 2) for i, value in enumerate(df[scan_column])]
    else:
        scans = df[scan_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        if scans.shape[1] == 0:
            print("[ERROR] Aucune colonne de rayon LiDAR.")
            return 1

    ray_count = len(scans[0]) if isinstance(scans, list) else scans.shape[1]
    if isinstance(scans, list):
        if any(len(scan) != ray_count for scan in scans):
            print("[ERROR] Tous les scans doivent contenir le même nombre de rayons.")
            return 1
        scans = np.stack(scans)
    scans = np.nan_to_num(scans, nan=args.max_range, posinf=args.max_range, neginf=0.0)
    servos = df["servo"].to_numpy(dtype=np.float32)

    train_idx, val_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, shuffle=False
    )
    if len(train_idx) == 0 or len(val_idx) == 0:
        print("[ERROR] Dataset trop petit pour séparer entraînement et validation.")
        return 1

    # Les contextes temporels de validation démarrent au premier scan de validation.
    train_ds = LidarDataset(scans[train_idx], servos[train_idx], args.max_range)
    val_ds = LidarDataset(scans[val_idx], servos[val_idx], args.max_range)
    workers = 0 if os.name == "nt" else 4
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=workers, pin_memory=pin_memory)

    model = LidarBehavioralCloningMLP(input_size=ray_count * 3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=5
    )

    best_val_loss = math.inf
    patience, patience_counter = 15, 0
    train_losses, val_losses = [], []
    print(f"[DATA] {len(train_ds)} train | {len(val_ds)} validation | {ray_count} rayons/scan")
    print("\n--- DÉBUT ENTRAÎNEMENT ---")

    for epoch in range(args.epochs):
        model.train()
        train_total = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * features.size(0)
        train_loss = train_total / len(train_ds)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                val_total += criterion(model(features), targets).item() * features.size(0)
        val_loss = val_total / len(val_ds)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch + 1:03d}/{args.epochs}] | LR: {optimizer.param_groups[0]['lr']:.2e} "
              f"| Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            args.model_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "ray_count": ray_count,
                "history_scans": 3,
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
        plt.title("Évolution de la perte — LiDAR")
        plt.savefig(args.loss_plot_out)
        print(f"[OK] Courbe : {args.loss_plot_out}")
    else:
        loss_csv = args.loss_plot_out.with_suffix(".csv")
        with loss_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "train_mse", "validation_mse"])
            writer.writerows(
                (epoch, train_loss, val_loss)
                for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1)
            )
        print(f"[OK] Historique des pertes (Matplotlib indisponible) : {loss_csv}")
    print(f"\n[OK] Meilleure Val Loss : {best_val_loss:.6f}")
    print(f"[OK] Modèle : {args.model_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
