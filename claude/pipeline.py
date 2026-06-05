"""
pipeline.py - Pipeline complet : image → masque → raycast → vecteur
Interface socket TCP pour Unity (envoie le vecteur de distances en JSON)

Usage : python3 pipeline.py
Unity se connecte sur localhost:9999 et envoie des images JPEG brutes,
reçoit en retour un JSON avec les distances des rayons.
"""

import torch
import numpy as np
from PIL import Image
import json
import socket
import struct
import io
import time

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH = "model/unet_scripted.pt"
HOST       = "localhost"
PORT       = 9999
IMG_SIZE   = (256, 256)
NUM_RAYS   = 32
FOV_DEG    = 180
THRESHOLD  = 0.5
# ─────────────────────────────────────────────────────────────────────────────


def load_model():
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    print(f"Modèle chargé : {MODEL_PATH}")
    return model


def preprocess(image_bytes):
    """Bytes JPEG → tensor (1, 3, 256, 256)"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE, Image.BILINEAR)
    arr   = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def predict_mask(model, tensor):
    """Tensor → masque numpy (256, 256)"""
    with torch.no_grad():
        pred = model(tensor)
    return (pred.squeeze().numpy() > THRESHOLD).astype(np.float32)


def raycast(mask_np, num_rays=NUM_RAYS, fov_deg=FOV_DEG):
    """Masque → vecteur de distances normalisées"""
    H, W  = mask_np.shape
    diag  = np.sqrt(H**2 + W**2)
    ox, oy = W / 2, H - 1

    angles_deg = np.linspace(-fov_deg / 2, fov_deg / 2, num_rays)
    angles_rad = np.deg2rad(angles_deg - 90)
    distances  = np.ones(num_rays)

    for i, angle in enumerate(angles_rad):
        dx = np.cos(angle)
        dy = np.sin(angle)
        for step in range(1, int(diag)):
            x = int(ox + dx * step)
            y = int(oy + dy * step)
            if x < 0 or x >= W or y < 0 or y >= H:
                distances[i] = step / diag
                break
            if mask_np[y, x] >= THRESHOLD:
                distances[i] = step / diag
                break

    return distances, angles_deg


def process_frame(model, image_bytes):
    """Pipeline complet pour une frame"""
    t0 = time.time()

    tensor    = preprocess(image_bytes)
    mask      = predict_mask(model, tensor)
    distances, angles = raycast(mask)

    elapsed = (time.time() - t0) * 1000

    return {
        "distances": distances.tolist(),   # liste de NUM_RAYS floats [0, 1]
        "angles_deg": angles.tolist(),     # angle correspondant à chaque rayon
        "num_rays": NUM_RAYS,
        "fov_deg": FOV_DEG,
        "inference_ms": round(elapsed, 1)
    }


def recv_exactly(conn, n):
    """Reçoit exactement n bytes (TCP fragmente les données)"""
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connexion fermée")
        data += chunk
    return data


def run_server():
    """
    Protocole TCP avec Unity :
    - Unity envoie : [4 bytes uint32 big-endian = taille] + [N bytes JPEG]
    - Python répond : [4 bytes uint32 big-endian = taille] + [N bytes JSON]
    """
    model = load_model()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"En attente de connexion Unity sur {HOST}:{PORT}...")

    while True:
        conn, addr = server.accept()
        print(f"Unity connecté : {addr}")
        try:
            while True:
                # Reçoit la taille de l'image
                size_bytes = recv_exactly(conn, 4)
                size = struct.unpack(">I", size_bytes)[0]

                # Reçoit l'image
                image_bytes = recv_exactly(conn, size)

                # Traite
                result = process_frame(model, image_bytes)

                # Répond
                response = json.dumps(result).encode("utf-8")
                conn.sendall(struct.pack(">I", len(response)))
                conn.sendall(response)

                print(f"Frame traitée en {result['inference_ms']} ms")

        except (ConnectionError, struct.error):
            print("Unity déconnecté, en attente...")
            conn.close()


if __name__ == "__main__":
    run_server()
