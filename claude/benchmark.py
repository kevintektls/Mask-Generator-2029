import torch
import numpy as np
from PIL import Image
import time
import os

# Charge le modèle scripté (pas besoin d'importer UNet)
model = torch.jit.load("model/unet_scripted.pt", map_location="cpu")
model.eval()

# Crée une image factice 256x256 pour le benchmark
dummy = torch.rand(1, 3, 256, 256)

# Chauffe (les premiers appels sont toujours plus lents)
for _ in range(3):
    with torch.no_grad():
        model(dummy)

# Mesure sur 20 passages
times = []
for _ in range(20):
    t0 = time.time()
    with torch.no_grad():
        model(dummy)
    times.append((time.time() - t0) * 1000)

print(f"Temps moyen  : {np.mean(times):.1f} ms")
print(f"Temps min    : {np.min(times):.1f} ms")
print(f"Temps max    : {np.max(times):.1f} ms")
print(f"FPS estimé   : {1000/np.mean(times):.1f}")
