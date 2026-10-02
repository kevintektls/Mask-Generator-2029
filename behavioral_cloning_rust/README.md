# Behavioral cloning LiDAR en Rust

Ce projet reprend la collecte manuelle du script de clonage comportemental et entraîne un MLP directement en Rust à partir des scans avant du LDROBOT D500. Les entrées du modèle sont les trois scans successifs normalisés sur 12 m ; la sortie est la position du servo dans `[0, 1]`. Le duty est enregistré et sert à écarter les lignes à l'arrêt, mais le réseau prédit uniquement le servo.

Le CSV est compatible avec `scripts/Behavioral_Cloning_Lidar.py` : `timestamp,servo,duty,lidar`, où `lidar` est un tableau JSON. Le projet réutilise le décodeur D500 Rust de `lidarcontrol`.

Le collecteur caméra Python demandé est aussi disponible ici : [`Behavioral_Cloning.py`](Behavioral_Cloning.py). Il sauvegarde les masques stéréo OAK-D dans `behavioral_cloning_rust/dataset/images/` et écrit pour chaque image `image_path,servo,duty,lidar_timestamp,lidar` dans le CSV. Les mesures LiDAR sont un tableau JSON de 180 distances en mètres. Le collecteur Rust décrit plus bas reste une option distincte, sans caméra.

Pour lancer la collecte Python, installer `depthai`, `opencv-python`, `numpy`, `pyserial`, `pyvesc` et la bibliothèque `Gamepad`, puis lancer depuis la racine du dépôt :

```sh
python3 behavioral_cloning_rust/Behavioral_Cloning.py
```

Les ports série et le dossier de sortie se règlent en haut du script. L'entraîneur Rust lit la colonne `lidar` de ce CSV ; les lignes avec `image_path` et `lidar_timestamp` supplémentaires sont acceptées.

## Préparer

Installer Rust/Cargo et les dépendances système de `gilrs` (sur Ubuntu : `libudev-dev`), brancher la manette, le VESC et le D500, puis vérifier les ports série. L'acquisition pilote réellement le VESC : roues motrices levées ou véhicule immobilisé lors des premiers essais.

## Collecter

Depuis la racine du dépôt :

```sh
cargo run --manifest-path behavioral_cloning_rust/Cargo.toml -- collect \
  --lidar-port /dev/ttyTHS1 --vesc-port /dev/ttyACM0 \
  --csv dataset_lidar/driving_log.csv
```

RT/LT accélèrent et freinent, le joystick gauche dirige, A démarre/met en pause l'enregistrement, LB et Ctrl+C coupent les commandes. Le fichier est ajouté au fil des sessions. Les axes triggers exposés par `gilrs` sont normalisés depuis `[-1, 1]`.

## Entraîner

```sh
cargo run --release --manifest-path behavioral_cloning_rust/Cargo.toml -- train \
  --csv dataset_lidar/driving_log.csv \
  --model-out behavioral_cloning_rust/model_lidar.json \
  --epochs 150 --learning-rate 0.001
```

Le programme garde les 20 % les plus récents pour la validation, affiche la MSE par époque, arrête après 15 époques sans amélioration et sauvegarde le meilleur modèle JSON. Une compilation Release est conseillée, car l'entraînement utilise un MLP de 540 entrées pour un LiDAR à 180 rayons.

## Charger le modèle et obtenir une prédiction

Créer `scan.json` comme un tableau de 180 distances en mètres (ordre de -90° à +90°), puis lancer :

```sh
cargo run --release --manifest-path behavioral_cloning_rust/Cargo.toml -- predict \
  --model behavioral_cloning_rust/model_lidar.json --scan scan.json
```

La sortie `servo=...` est une prédiction hors ligne. Cette commande ne connecte pas le modèle au VESC ; il faut d'abord valider les prédictions sur des scans enregistrés avant d'ajouter une conduite autonome.
