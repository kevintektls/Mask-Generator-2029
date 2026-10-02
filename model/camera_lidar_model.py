"""Réseau de behavioral cloning qui fusionne masque caméra et scans LiDAR."""

import torch
from torch import nn


class CameraLidarBehavioralCloning(nn.Module):
    """Prédit le servo depuis 3 masques consécutifs et 3 scans LiDAR."""

    def __init__(self, ray_count: int, history_scans: int = 3):
        super().__init__()
        self.ray_count = ray_count
        self.history_scans = history_scans

        self.image_cnn = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Flatten(),
        )
        self.image_encoder = nn.Sequential(
            nn.Linear(64 * 10 * 15, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 50),
            nn.ReLU(),
        )
        self.lidar_encoder = nn.Sequential(
            nn.Linear(ray_count * history_scans, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.regressor = nn.Sequential(
            nn.Linear(50 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, images: torch.Tensor, scans: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(self.image_cnn(images))
        lidar_features = self.lidar_encoder(scans)
        return self.regressor(torch.cat((image_features, lidar_features), dim=1)).squeeze(1)
