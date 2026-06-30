#!/usr/bin/env python3
"""
Définition unique du modèle CNN.
À importer dans train_model.py et autopilot_ai.py.
"""

import torch
import torch.nn as nn


class BehavioralCloningCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), # Attend maintenant tes 3 canaux (t, t-1, t-2)
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

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.regressor(x)
        return x.squeeze(1)
