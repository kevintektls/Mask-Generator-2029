"""
Architecture CNN partagée avec Behavioral Cloning (scripts/Autopilot_IA.py).
Entrée : masque binaire (1, 120, 160) — pas de lidar, uniquement caméra.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BehavioralCloningCNN(nn.Module):
    """Doit rester strictement identique au réseau utilisé sur la Jetson."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.flatten = nn.Flatten()
        self.regressor = nn.Sequential(
            nn.Linear(64 * 10 * 15, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        return self.regressor(x).squeeze(-1)


def load_bc_weights(model: BehavioralCloningCNN, path: str, device: torch.device) -> None:
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"[RL] Poids BC chargés depuis {path}")
