#!/usr/bin/env python3
"""
Export pilot_model.pth to ONNX for autopilot_cpp.

Usage (from repo root or autopilot_cpp/):
  pip install torch onnx
  python3 autopilot_cpp/tools/export_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from model_def import BehavioralCloningCNN  # noqa: E402

PTH_PATH = REPO / "model" / "pilot_model.pth"
ONNX_PATH = REPO / "model" / "pilot_model.onnx"


def main() -> None:
    if not PTH_PATH.is_file():
        print(f"ERROR: weights not found: {PTH_PATH}")
        sys.exit(1)

    model = BehavioralCloningCNN()
    state = torch.load(PTH_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 1, 120, 160)
    with torch.no_grad():
        torch_out = model(dummy).item()

    torch.onnx.export(
        model,
        dummy,
        str(ONNX_PATH),
        input_names=["input"],
        output_names=["servo"],
        dynamic_axes=None,
        opset_version=17,
    )

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
        onnx_out = sess.run(None, {"input": dummy.numpy()})[0].item()
        print(f"PyTorch sample output: {torch_out:.6f}")
        print(f"ONNX     sample output: {onnx_out:.6f}")
        print(f"Delta: {abs(torch_out - onnx_out):.2e}")
    except ImportError:
        print("onnxruntime not installed — skipped parity check")

    print(f"Exported: {ONNX_PATH}")


if __name__ == "__main__":
    main()
