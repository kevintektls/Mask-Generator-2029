"""Policy Stable-Baselines3 initialisée depuis pilot_model.pth (Behavioral Cloning)."""

from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rl.models import BehavioralCloningCNN


class MaskCNNExtractor(BaseFeaturesExtractor):
    """Feature extractor identique au CNN de Behavioral Cloning."""

    def __init__(self, observation_space, features_dim: int = 50):
        super().__init__(observation_space, features_dim)
        self.cnn = BehavioralCloningCNN()
        self._latent = nn.Sequential(
            nn.Linear(64 * 10 * 15, 100),
            nn.ReLU(),
            nn.Linear(100, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn.features(observations)
        x = self.cnn.flatten(x)
        x = self.cnn.regressor[0](x)  # Linear → 100
        x = self.cnn.regressor[1](x)  # ReLU
        x = self.cnn.regressor[3](x)  # Linear → 50
        x = self.cnn.regressor[4](x)  # ReLU
        return x


def init_policy_from_bc(model, bc_path: str, device: torch.device) -> None:
    """
    Charge pilot_model.pth dans le feature extractor + initialise
    la tête actor proche de la sortie BC (steering).
    """
    bc = BehavioralCloningCNN().to(device)
    state = torch.load(bc_path, map_location=device, weights_only=True)
    bc.load_state_dict(state)

    policy = model.policy
    fe: MaskCNNExtractor = policy.features_extractor
    fe.cnn.load_state_dict(bc.state_dict())

    with torch.no_grad():
        dummy = torch.zeros(1, 1, 120, 160, device=device)
        latent = fe(dummy)
        bc_out = bc(dummy).item()
        steer_norm = bc_out * 2.0 - 1.0
        if hasattr(policy, "action_net"):
            policy.action_net.weight.fill_(0.0)
            policy.action_net.bias.zero_()
            policy.action_net.bias[0] = steer_norm

    print(f"[RL] Policy initialisée depuis {bc_path} (steer≈{steer_norm:.3f})")
