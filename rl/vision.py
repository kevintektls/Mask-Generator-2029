"""
Traitement vision identique au robot réel (Autopilot_IA++.py).
Caméra → masque binaire des bandes blanches → tensor (1, 120, 160).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

CROP_TOP_RATIO = 0.20
ULTRA_BINARY_THRESH = 220
MASK_W = 160
MASK_H = 120
SERVO_CENTER = 0.5
LANE_WIDTH_PX = 340.0
LANE_WIDTH_MIN = 160.0


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def detect_lines(frame_gray: np.ndarray) -> np.ndarray:
    """Extrait le masque binaire des lignes blanches (même algo que la Jetson)."""
    h, w = frame_gray.shape
    clean_mask = np.zeros_like(frame_gray)
    start_y = int(h * CROP_TOP_RATIO)
    roi_sol = frame_gray[start_y:h, :]
    blurred = cv2.GaussianBlur(roi_sol, (5, 5), 0)
    _, binary_sol = cv2.threshold(blurred, ULTRA_BINARY_THRESH, 255, cv2.THRESH_BINARY)
    clean_mask[start_y:h, :] = binary_sol
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)


def mask_to_tensor(mask: np.ndarray) -> np.ndarray:
    """Masque uint8 → observation float32 (1, H, W) normalisée [0, 1]."""
    resized = cv2.resize(mask, (MASK_W, MASK_H), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)[np.newaxis, ...]


def rgb_to_observation(rgb: np.ndarray) -> np.ndarray:
    gray = rgb_to_gray(rgb)
    mask = detect_lines(gray)
    return mask_to_tensor(mask)


def lateral_error_from_mask(mask: np.ndarray) -> float:
    """
    Erreur latérale normalisée depuis le masque [-1, 1].
    0 = centré entre les deux bandes, négatif = trop à gauche, positif = trop à droite.
    """
    h, w = mask.shape
    mid = w // 2
    y_top = int(h * 0.50)
    y_bot = int(h * 0.85)
    band = mask[y_top:y_bot, :]
    hist = np.sum(band > 0, axis=0).astype(np.float32)
    if np.max(hist) == 0:
        return 0.0

    hist = np.convolve(hist, np.ones(21, dtype=np.float32) / 21.0, mode="same")
    threshold = (y_bot - y_top) * 0.20

    left_hits = np.where(hist[:mid][::-1] >= threshold)[0]
    left_x = (mid - 1 - int(left_hits[0])) if left_hits.size else -1

    right_hits = np.where(hist[mid:] >= threshold)[0]
    right_x = (mid + int(right_hits[0])) if right_hits.size else -1

    if left_x >= 0 and right_x >= 0:
        target = (left_x + right_x) / 2.0
    elif left_x >= 0:
        target = left_x + (LANE_WIDTH_PX / 2.0)
    elif right_x >= 0:
        target = right_x - (LANE_WIDTH_PX / 2.0)
    else:
        return 0.0

    error = (target - mid) / mid
    return float(np.clip(error, -1.0, 1.0))


def tensor_from_rgb_bytes(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    obs = rgb_to_observation(rgb)
    return torch.from_numpy(obs).unsqueeze(0).to(device)
