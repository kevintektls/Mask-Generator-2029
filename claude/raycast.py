import numpy as np
from PIL import Image, ImageDraw
import torch
import time

# ── CONFIG ────────────────────────────────────────────────────────────────────
NUM_RAYS   = 32       # nombre de rayons (modulable)
FOV_DEG    = 180      # champ de vision en degrés
MAX_DIST   = 1.0      # distance max normalisée (1.0 = diagonale de l'image)
THRESHOLD  = 0.5      # seuil de détection sur le masque
# ─────────────────────────────────────────────────────────────────────────────

def raycast(mask_np: np.ndarray, num_rays: int = NUM_RAYS, fov_deg: float = FOV_DEG):
    """
    Lance num_rays rayons depuis le centre bas du masque.
    
    Args:
        mask_np : masque binaire numpy (H, W), valeurs 0.0 ou 1.0
        num_rays : nombre de rayons
        fov_deg  : champ de vision total en degrés
    
    Returns:
        distances : array (num_rays,) — distance normalisée [0, 1] jusqu'à
                    la première ligne détectée, 1.0 si rien détecté
        angles_deg : array (num_rays,) — angle de chaque rayon en degrés
    """
    H, W = mask_np.shape
    diag = np.sqrt(H**2 + W**2)

    # Point d'origine : centre bas
    ox, oy = W / 2, H - 1

    # Angles des rayons : de -90° (gauche) à +90° (droite)
    # 0° = tout droit vers le haut (devant la voiture)
    half_fov = fov_deg / 2
    angles_deg = np.linspace(-half_fov, half_fov, num_rays)

    # Convertit en radians, ajuste : 0° = vers le haut = -π/2 en coords image
    angles_rad = np.deg2rad(angles_deg - 90)

    distances = np.ones(num_rays)  # par défaut : rien détecté = 1.0

    for i, angle in enumerate(angles_rad):
        dx = np.cos(angle)
        dy = np.sin(angle)

        # Marche pixel par pixel le long du rayon
        # Nombre max de steps = diagonale de l'image
        max_steps = int(diag)
        for step in range(1, max_steps):
            x = int(ox + dx * step)
            y = int(oy + dy * step)

            # Sort de l'image → rayon bloqué
            if x < 0 or x >= W or y < 0 or y >= H:
                distances[i] = step / diag
                break

            # Ligne détectée
            if mask_np[y, x] >= THRESHOLD:
                distances[i] = step / diag
                break

    return distances, angles_deg


def visualize_raycast(image_path, mask_np, distances, angles_deg, save_path="raycast_viz.png"):
    """Superpose les rayons sur l'image originale pour vérification visuelle."""
    img = Image.open(image_path).convert("RGB").resize((mask_np.shape[1], mask_np.shape[0]))
    draw = ImageDraw.Draw(img)

    H, W = mask_np.shape
    diag = np.sqrt(H**2 + W**2)
    ox, oy = W / 2, H - 1

    angles_rad = np.deg2rad(angles_deg - 90)

    for i, (angle, dist) in enumerate(zip(angles_rad, distances)):
        dx = np.cos(angle)
        dy = np.sin(angle)
        length = dist * diag

        ex = ox + dx * length
        ey = oy + dy * length

        # Rouge si ligne détectée, vert si bord image
        color = (255, 0, 0) if dist < 1.0 else (0, 255, 0)
        draw.line([(ox, oy), (ex, ey)], fill=color, width=1)

    img.save(save_path)
    print(f"Visualisation sauvegardée : {save_path}")


if __name__ == "__main__":
    # ── Test sur un masque réel ───────────────────────────────────────────────
    import os

    # Prend le premier masque du dataset
    mask_dir = "dataset_final/masks"
    img_dir  = "dataset_final/images"
    fname    = sorted(os.listdir(mask_dir))[0]

    mask = Image.open(os.path.join(mask_dir, fname)).convert("L").resize((256, 256))
    mask_np = np.array(mask, dtype=np.float32) / 255.0

    # Benchmark
    t0 = time.time()
    for _ in range(100):
        distances, angles = raycast(mask_np, num_rays=32)
    elapsed = (time.time() - t0) / 100 * 1000
    print(f"Raycast 32 rayons : {elapsed:.2f} ms en moyenne")

    # Visualisation
    visualize_raycast(os.path.join(img_dir, fname), mask_np, distances, angles)

    print(f"\nDistances (32 rayons) :")
    for a, d in zip(angles, distances):
        bar = "█" * int(d * 20)
        print(f"  {a:+6.1f}° | {d:.3f} | {bar}")
