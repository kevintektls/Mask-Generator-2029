# lidarcontrol — étape 1 : acquisition et replay LiDAR

Programme Rust autonome pour le LDROBOT D500 / STL-19P. Le framing UART, le
CRC-8, l’interpolation des 12 points et la convention angulaire reprennent le
lecteur fonctionnel de `scripts/Behavioral_Cloning_Lidar.py`.

## Compiler et valider sans matériel

Depuis la racine du dépôt :

```bash
cargo test --manifest-path lidarcontrol/Cargo.toml
cargo run --manifest-path lidarcontrol/Cargo.toml -- replay --input chemin/vers/scan.jsonl --speed 0
```

Le replay valide le JSONL, le nombre de bins, les distances et l’ordre des
timestamps. `--speed 0` lit les scans aussi vite que possible.

## Enregistrer sur la voiture

Les valeurs par défaut sont dans `lidarcontrol/config.toml` : UART
`/dev/ttyTHS1`, 230400 bauds, sortie `lidarcontrol/data/scan.jsonl`. Elles
peuvent être remplacées par `--port`, `--baud` et `--output` :

```bash
cargo run --release --manifest-path lidarcontrol/Cargo.toml -- record
cargo run --release --manifest-path lidarcontrol/Cargo.toml -- record --port /dev/ttyTHS1 --output lidarcontrol/data/essai.jsonl
```

`Ctrl-C` termine l’enregistrement après le scan courant. Chaque ligne JSON
contient un scan frontal de 180 bins à 1°, les distances en mètres, le temps
UTC de réception, le temps monotone depuis le démarrage du processus et la
durée estimée du scan. Les temps actuels sont à la granularité d’un paquet
UART ; ils ne constituent pas encore une synchronisation avec la caméra.

## Essai matériel

1. Mettre la voiture sur cales, moteur désactivé, et vérifier que le D500 est
   visible sur le port série configuré.
2. Lancer `record`, faire tourner le LiDAR sans déplacer la voiture, puis
   arrêter avec `Ctrl-C`.
3. Vérifier que le fichier contient plusieurs centaines de lignes et lancer
   le replay. Faire pivoter lentement la voiture à la main et recommencer pour
   vérifier que les bins gauche/droite suivent la convention du script Python.

Cette étape n’envoie aucune commande au Flipsky/VESC. La cartographie SLAM,
l’odométrie, la fusion caméra-LiDAR et le contrôle autonome ne sont pas encore
implémentés.
