#!/usr/bin/env python3
"""
Prétraitement vision partagé pour :
- enregistrement dataset
- entraînement
- autopilot IA

Objectif : garantir que le modèle voit EXACTEMENT le même type d'image
pendant l'entraînement et pendant l'inférence réelle.
"""

import cv2
import numpy as np

# IMPORTANT : garder les mêmes paramètres partout
CROP_TOP_RATIO = 0.35

# Alignement manuel caméra droite -> caméra gauche
STEREO_DX = -15.0
STEREO_DY = -5.0
STEREO_ANGLE = 0.0

# Image réseau
MODEL_W = 160
MODEL_H = 120


def make_mask_stereo(frame_left: np.ndarray, frame_right: np.ndarray) -> np.ndarray:
    """
    Transforme les deux images mono de l'OAK-D Lite en masque binaire propre.
    Entrée :
        frame_left  : image grayscale CAM_B
        frame_right : image grayscale CAM_C
    Sortie :
        masque binaire uint8, même résolution que l'entrée
    """
    if frame_left is None or frame_right is None:
        raise ValueError("frame_left/frame_right ne doivent pas être None")

    if len(frame_left.shape) != 2 or len(frame_right.shape) != 2:
        raise ValueError("make_mask_stereo attend des images grayscale 1 canal")

    h, w = frame_left.shape

    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, STEREO_ANGLE, 1.0)
    M[0, 2] += STEREO_DX
    M[1, 2] += STEREO_DY

    frame_right_aligned = cv2.warpAffine(
        frame_right,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Fusion des deux yeux après alignement
    merged = cv2.max(frame_left, frame_right_aligned)

    clean_mask = np.zeros_like(merged)
    start_y = int(h * CROP_TOP_RATIO)

    roi_sol = merged[start_y:h, :]

    # Même traitement que dataset
    filtered = cv2.bilateralFilter(roi_sol, d=5, sigmaColor=40, sigmaSpace=40)
    _, binary_sol = cv2.threshold(filtered, 225, 255, cv2.THRESH_BINARY)

    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))

    binary_sol = cv2.morphologyEx(binary_sol, cv2.MORPH_OPEN, kernel_clean)
    binary_sol = cv2.morphologyEx(binary_sol, cv2.MORPH_CLOSE, kernel_vertical)

    clean_mask[start_y:h, :] = binary_sol
    return clean_mask


def resize_for_model(mask: np.ndarray) -> np.ndarray:
    """
    Redimensionne le masque au format exact attendu par le CNN : 160x120.
    """
    return cv2.resize(mask, (MODEL_W, MODEL_H))


def mask_to_tensor_numpy(mask: np.ndarray) -> np.ndarray:
    """
    Prépare un tableau numpy normalisé [1, 1, 120, 160].
    Utile si tu veux ensuite convertir vers torch.
    """
    resized = resize_for_model(mask)
    tensor = resized.astype(np.float32) / 255.0
    return tensor[None, None, :, :]
