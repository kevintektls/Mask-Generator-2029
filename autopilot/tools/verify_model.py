#!/usr/bin/env python3
"""
Compare Python vs Rust behavioral-cloning inference on a synthetic mask.

Usage (from repo root, with torch installed):
    python3 autopilot/tools/verify_model.py
    python3 autopilot/tools/verify_model.py --rust-binary autopilot/target/release/autopilot-verify

Requires the Rust verify binary built on the Jetson or dev machine:
    cargo build --release -p autopilot-model --bin autopilot-verify
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("torch required: pip install torch")
    sys.exit(1)

from rl.vision import mask_to_tensor, detect_lines  # noqa: E402

MASK_H, MASK_W = 120, 160
MODEL_PATH = REPO / "model" / "pilot_model.pth"


class BehavioralCloningCNN(nn.Module):
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

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.regressor(x).squeeze(1)


def python_predict(model: BehavioralCloningCNN, obs: np.ndarray) -> float:
    with torch.no_grad():
        t = torch.from_numpy(obs).unsqueeze(0)
        return float(model(t).item())


def rust_predict(binary: Path, obs: np.ndarray) -> float:
    payload = struct.pack(f"<{obs.size}f", *obs.flatten().tolist())
    proc = subprocess.run(
        [str(binary), str(MODEL_PATH)],
        input=payload,
        capture_output=True,
        check=True,
    )
    return float(proc.stdout.decode().strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rust-binary",
        type=Path,
        default=REPO / "autopilot" / "target" / "release" / "autopilot-verify",
    )
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"model missing: {MODEL_PATH}")
        sys.exit(1)

    model = BehavioralCloningCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    rng = np.random.default_rng(42)
    gray = (rng.random((480, 640)) * 255).astype(np.uint8)
    mask = detect_lines(gray)
    obs = mask_to_tensor(mask)

    py_out = python_predict(model, obs)
    print(f"Python servo prediction: {py_out:.8f}")

    if not args.rust_binary.exists():
        print(f"Rust verify binary not found: {args.rust_binary}")
        print("Build with: cd autopilot && cargo build --release -p autopilot-model --bin autopilot-verify")
        sys.exit(0)

    rs_out = rust_predict(args.rust_binary, obs)
    print(f"Rust servo prediction:   {rs_out:.8f}")
    delta = abs(py_out - rs_out)
    print(f"Delta: {delta:.2e} (tolerance {args.tolerance})")

    if delta > args.tolerance:
        print("FAIL: predictions diverge")
        sys.exit(1)
    print("OK: Python and Rust agree")


if __name__ == "__main__":
    main()
