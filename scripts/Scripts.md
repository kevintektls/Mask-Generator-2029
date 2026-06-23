# 🏎️ Pipeline Autopilote IA : Du Dataset à l'Utilisation en Python

Ce guide détaille l'ensemble du cycle de développement de l'autopilote hybride (Vision Stéréo & VESC) entièrement basé sur Python. Il couvre la préparation des données, l'entraînement du réseau de neurones avec PyTorch et son déploiement en mode autonome sur la Jetson Nano.

---

## 📌 Architecture Globale du Projet

```text
├── Gamepad/               # Bibliothèque de gestion de la manette Xbox
├── model/
│   ├── train_model.py     # Script d'entraînement PyTorch
│   ├── model_def.py       # Architecture du réseau (BehavioralCloningCNN)
│   └── pilot_model.pth    # Poids du modèle entraîné (.pth)
├── scripts/
│   ├── vision_preprocess.py # Fonctions de traitement (make_mask_stereo, etc.)
│   └── autopilot_car.py   # Script principal de pilotage autonome
```

---

## 🛠️ Étape 1 : Création et Préparation du Dataset

L'autopilote n'utilise pas des images brutes (RGB/Gris), mais des **masques binaires** (noir et blanc) générés par la vision stéréo de la caméra OAK-D Lite. Cela permet d'isoler la piste et de s'affranchir des variations de luminosité.

### 1. Collecte des images (Mode Manuel)

Pilote la voiture manuellement à l'aide de la manette Xbox. Pendant la conduite, enregistre :

- L'image **gauche** et l'image **droite** de l'OAK-D.
- La valeur de braquage du servo (`servo_pos` entre `0.0` et `1.0`) associée à chaque paire d'images.

### 2. Génération des masques de traitement

Le script `vision_preprocess.py` applique la logique suivante sur ton dossier d'images brutes :

1. Calcul de la disparité/profondeur stéréo via `make_mask_stereo(left, right)`.
2. Application du rognage supérieur (`CROP_TOP_RATIO`) pour supprimer l'horizon inutile.
3. Redimensionnement via `resize_for_model(mask)` à la taille d'entrée du réseau (ex: `64x128`).
4. Sauvegarde des images traitées dans `dataset/train/` et `dataset/val/`.

---

## 🏋️ Étape 2 : Entraînement du Modèle

Le modèle est un **réseau de neurones à convolution (CNN)** conçu pour le **Behavioral Cloning** (clonage de comportement). Il prend le masque en entrée et prédit une valeur continue pour le servo.

### 1. Activer l'environnement virtuel et lancer l'entraînement

Assure-toi que ton GPU NVIDIA est disponible. Si CUDA est verrouillé par un processus zombie, réinitialise-le ou redémarre ta session (`sudo reboot`).

```bash
# Activation du venv
source venv/bin/activate

# Export du chemin des scripts pour que train_model.py trouve vision_preprocess.py
export PYTHONPATH="${PYTHONPATH}:../scripts"

# Lancement de l'entraînement
cd model
python3 train_model.py
```

Le script va sauvegarder les meilleurs poids entraînés dans le fichier `../model/pilot_model.pth`.

---

## 🚀 Étape 3 : Déploiement et Mode Autonome

Une fois le fichier `pilot_model.pth` généré, tu peux exécuter ton script d'autopilote directement sur la Jetson Nano.

### 1. Lancement de l'Autopilote

> 🚨 **SÉCURITÉ** : Place impérativement le véhicule sur un support surélevé (les roues dans le vide) lors du tout premier test pour éviter que la voiture ne fonce dans un mur en cas de bug de prédiction ou d'inversion des commandes.

```bash
# Configuration du chemin des scripts
export PYTHONPATH="${PYTHONPATH}:../scripts"

# Lancement de l'autopilote
python3 autopilot_car.py
```

### 2. Fonctionnalités en cours d'exécution

#### 📡 Retour Vidéo Live (MJPEG)

Ouvre un navigateur web sur ton PC connecté au même réseau local que la voiture et accède à :

```
http://<IP_DE_LA_JETSON>:8080
```

Tu y verras :
- Le masque binaire calculé en temps réel par l'OAK-D.
- La ligne rouge de coupe (`CROP_TOP_RATIO`).
- Les commandes envoyées au VESC.

#### ⚡ Gestion de la Vitesse Dynamique

| Situation | Duty Cycle |
|-----------|-----------|
| Ligne droite (braquage ≈ 0.5) | `DUTY_MAX` (0.1) |
| Virage détecté | `DUTY_MIN` (0.05) |

En ligne droite, la voiture accélère jusqu'à `DUTY_MAX`. Dès que l'IA commence à braquer, le script réduit automatiquement le Duty Cycle vers `DUTY_MIN` pour stabiliser le châssis et éviter le sous-virage.

#### 🛑 Arrêt d'Urgence Instantané

- **Via la Manette** : Presse le bouton `LB` à tout moment. Le script injectera immédiatement un freinage électrique puissant (`vesc.set_brake(15.0)`) et arrêtera la boucle.
- **Via le Terminal** : Un simple `CTRL+C` coupe proprement la commande de gaz, applique un frein de sécurité et remet les roues droites.