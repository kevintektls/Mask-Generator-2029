"""Architecture MLP utilisée par l'entraînement et l'inférence LiDAR."""

import torch
from torch import nn


class LidarBehavioralCloningMLP(nn.Module):
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
