"""
inference_server.py - Serveur TCP Python
- Reçoit une image depuis Unity (raw bytes RGB)
- La passe dans le U-Net entraîné
- Renvoie le masque binaire à Unity

Usage : python3 inference_server.py
        (lancer AVANT de démarrer Unity)
"""

import socket
import struct
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import io

# ── CONFIG ────────────────────────────────────────────────────────────────────
HOST       = "127.0.0.1"
PORT       = 65432
MODEL_PATH = "model/unet.pth"
IMG_SIZE   = (256, 256)
THRESHOLD  = 0.5        # seuil de binarisation du masque CNN
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────────────────────────


# ── U-NET (même architecture que train.py) ────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, features=(16, 32, 64, 128)):
        super().__init__()
        self.enc1 = ConvBlock(3,           features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(features[3], features[3] * 2)
        self.up4  = nn.ConvTranspose2d(features[3] * 2, features[3], 2, 2)
        self.dec4 = ConvBlock(features[3] * 2, features[3])
        self.up3  = nn.ConvTranspose2d(features[3], features[2], 2, 2)
        self.dec3 = ConvBlock(features[2] * 2, features[2])
        self.up2  = nn.ConvTranspose2d(features[2], features[1], 2, 2)
        self.dec2 = ConvBlock(features[1] * 2, features[1])
        self.up1  = nn.ConvTranspose2d(features[1], features[0], 2, 2)
        self.dec1 = ConvBlock(features[0] * 2, features[0])
        self.final = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))


# ── CHARGEMENT DU MODÈLE ──────────────────────────────────────────────────────
def load_model():
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Modèle chargé depuis {MODEL_PATH} sur {DEVICE}")
    return model


# ── INFÉRENCE ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_mask(model, image_bytes: bytes, orig_w: int, orig_h: int) -> bytes:
    """
    Reçoit les bytes RGB de l'image, retourne les bytes du masque binaire (L).
    """
    # Décode l'image
    image = Image.frombytes("RGB", (orig_w, orig_h), image_bytes)

    # Prépare le tensor
    img_resized = image.resize(IMG_SIZE, Image.BILINEAR)
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    tensor  = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Inférence
    pred = model(tensor)                          # (1, 1, 256, 256)
    pred = pred.squeeze().cpu().numpy()           # (256, 256)

    # Binarise et redimensionne à la taille originale
    mask = (pred > THRESHOLD).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, mode="L").resize((orig_w, orig_h), Image.NEAREST)

    # Convertit en RGB (Unity attend du RGB)
    mask_rgb = mask_img.convert("RGB")
    return mask_rgb.tobytes()


# ── PROTOCOLE TCP ─────────────────────────────────────────────────────────────
# Format des messages :
#   Envoi Unity → Python  : [4 bytes width][4 bytes height][width*height*3 bytes RGB]
#   Réponse Python → Unity : [width*height*3 bytes RGB mask]

def recv_all(conn, n: int) -> bytes:
    """Reçoit exactement n bytes depuis la connexion."""
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connexion fermée par Unity")
        data += chunk
    return data


def handle_client(conn, model):
    """Traite une connexion Unity : reçoit image, envoie masque."""
    try:
        while True:
            # Lit width et height (2 x 4 bytes little-endian)
            header = recv_all(conn, 8)
            w, h   = struct.unpack("<II", header)

            # Lit les pixels RGB
            n_bytes    = w * h * 3
            image_bytes = recv_all(conn, n_bytes)

            # Inférence CNN
            mask_bytes = predict_mask(model, image_bytes, w, h)

            # Envoie le masque
            conn.sendall(mask_bytes)

    except (ConnectionError, OSError):
        print("Client déconnecté.")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    model = load_model()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"Serveur en attente sur {HOST}:{PORT} ...")

        while True:
            conn, addr = server.accept()
            print(f"Unity connecté depuis {addr}")
            with conn:
                handle_client(conn, model)


if __name__ == "__main__":
    main()