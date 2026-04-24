import numpy as np
import cv2
import math


def raycast(mask: np.ndarray, n_rays: int = 9, fov: int = 120) -> list[float]:
    """
    Lance n_rays rayons depuis le bas-centre du masque binaire.

    Chaque rayon part dans une direction différente dans le champ de vision (fov).
    Il avance pixel par pixel jusqu'à toucher un pixel blanc (une ligne de piste).
    Il retourne la distance normalisée entre 0.0 et 1.0
    (1.0 = rien trouvé / rayon sorti de l'image).

    Args:
        mask    : image binaire numpy (uint8), blanc = ligne, noir = fond
        n_rays  : nombre de rayons à lancer
        fov     : champ de vision total en degrés (ex: 120° = 60° à gauche, 60° à droite)

    Returns:
        Liste de n_rays distances normalisées (float entre 0.0 et 1.0)
    """
    height, width = mask.shape

    # Point de départ : bas-centre de l'image
    origin_x = width // 2
    origin_y = height - 1

    # Distance max possible dans l'image (diagonale), sert à normaliser
    max_dist = math.sqrt(width**2 + height**2)

    # On calcule les angles de chaque rayon
    # Les angles sont répartis uniformément dans le fov
    # 90° = vers le haut (devant la voiture), 0° = droite, 180° = gauche
    # On part de (90 - fov/2) jusqu'à (90 + fov/2)
    if n_rays == 1:
        angles = [90.0]
    else:
        angle_start = 90 - fov / 2
        angle_end = 90 + fov / 2
        angles = [
            angle_start + i * (fov / (n_rays - 1))
            for i in range(n_rays)
        ]

    distances = []

    for angle_deg in angles:
        # Conversion angle → direction (dx, dy)
        # En trigonométrie standard : x = cos(angle), y = sin(angle)
        # Mais en image, l'axe Y est inversé (0 en haut, max en bas)
        # Donc dy est négatif pour aller vers le haut
        angle_rad = math.radians(angle_deg)
        dx = math.cos(angle_rad)
        dy = -math.sin(angle_rad)   # négatif car Y image = vers le bas

        # On avance pixel par pixel le long du rayon
        dist = 0.0
        found = False

        # Longueur max du rayon = diagonale de l'image (on ne peut pas aller plus loin)
        max_steps = int(max_dist) + 1

        for step in range(1, max_steps):
            # Position actuelle du rayon
            px = int(round(origin_x + dx * step))
            py = int(round(origin_y + dy * step))

            # Si on sort de l'image → on arrête
            if px < 0 or px >= width or py < 0 or py >= height:
                dist = max_dist
                break

            # Si on touche un pixel blanc (ligne de piste) → on enregistre la distance
            if mask[py, px] > 0:
                dist = math.sqrt((px - origin_x)**2 + (py - origin_y)**2)
                found = True
                break
        else:
            # On a parcouru tous les steps sans rien trouver
            dist = max_dist

        # Normalisation : distance entre 0.0 et 1.0
        distances.append(dist / max_dist)

    return distances


def visualize(image: np.ndarray, mask: np.ndarray, n_rays: int = 9, fov: int = 120) -> np.ndarray:
    """
    Dessine les rayons sur l'image originale pour visualiser le résultat.

    - Rayon VERT  → a touché une ligne (distance < 1.0)
    - Rayon ROUGE → n'a rien trouvé (sorti de l'image)
    - Point JAUNE → point de contact avec la ligne

    Args:
        image   : image BGR originale (pour l'affichage)
        mask    : masque binaire correspondant
        n_rays  : nombre de rayons
        fov     : champ de vision en degrés

    Returns:
        Image BGR avec les rayons dessinés dessus
    """
    height, width = mask.shape
    max_dist = math.sqrt(width**2 + height**2)

    origin_x = width // 2
    origin_y = height - 1

    if n_rays == 1:
        angles = [90.0]
    else:
        angle_start = 90 - fov / 2
        angle_end = 90 + fov / 2
        angles = [
            angle_start + i * (fov / (n_rays - 1))
            for i in range(n_rays)
        ]

    # Copie de l'image pour ne pas modifier l'original
    output = image.copy()

    # Si l'image est en niveaux de gris, on la convertit en BGR pour afficher des couleurs
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    for angle_deg in angles:
        angle_rad = math.radians(angle_deg)
        dx = math.cos(angle_rad)
        dy = -math.sin(angle_rad)

        max_steps = int(max_dist) + 1
        hit_x, hit_y = None, None

        for step in range(1, max_steps):
            px = int(round(origin_x + dx * step))
            py = int(round(origin_y + dy * step))

            if px < 0 or px >= width or py < 0 or py >= height:
                # Rayon sorti → on dessine jusqu'au bord en rouge
                end_x = int(round(origin_x + dx * (step - 1)))
                end_y = int(round(origin_y + dy * (step - 1)))
                cv2.line(output, (origin_x, origin_y), (end_x, end_y), (0, 0, 255), 1)
                break

            if mask[py, px] > 0:
                hit_x, hit_y = px, py
                # Rayon qui touche → on dessine en vert jusqu'au point de contact
                cv2.line(output, (origin_x, origin_y), (hit_x, hit_y), (0, 255, 0), 1)
                # Point de contact en jaune
                cv2.circle(output, (hit_x, hit_y), 3, (0, 255, 255), -1)
                break

    # Point d'origine en blanc
    cv2.circle(output, (origin_x, origin_y), 5, (255, 255, 255), -1)

    return output


# --- TEST RAPIDE ---
# Lance ce fichier directement pour tester avec un masque synthétique
if __name__ == "__main__":

    # Crée un masque de test : image noire avec deux lignes diagonales blanches
    mask = np.zeros((240, 320), dtype=np.uint8)

    # Ligne gauche (diagonale)
    cv2.line(mask, (80, 240), (120, 100), 255, 3)
    # Ligne droite (diagonale)
    cv2.line(mask, (240, 240), (200, 100), 255, 3)

    # Lance le raycast
    distances = raycast(mask, n_rays=9, fov=120)

    print("Distances normalisées par rayon :")
    for i, d in enumerate(distances):
        status = "TOUCHÉ" if d < 1.0 else "rien"
        print(f"  Rayon {i+1:2d} : {d:.3f}  ({status})")

    # Visualisation
    result = visualize(mask, mask, n_rays=9, fov=120)
    cv2.imwrite("raycast_test.png", result)
    print("\nImage sauvegardée : raycast_test.png")