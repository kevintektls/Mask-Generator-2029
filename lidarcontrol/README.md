# lidarcontrol — acquisition, mapping LiDAR et capture caméra

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

## Étape 2 : construire la carte 2D

Depuis un enregistrement :

```bash
cargo run --release --manifest-path lidarcontrol/Cargo.toml -- map --input lidarcontrol/data/scan.jsonl
# ou directement depuis le CSV du collecteur existant :
cargo run --release --manifest-path lidarcontrol/Cargo.toml -- map --input dataset_lidar/driving_log.csv
```

La commande produit `lidarcontrol/maps/floor.pgm` (occupation), `floor.yaml`
(résolution et origine) et `floor.poses.jsonl` (poses estimées et confiance par
scan). La résolution par défaut est 5 cm par cellule et se règle avec
`map_resolution_m` dans `config.toml` ou `--resolution`.
Le CSV existant doit garder l’en-tête `timestamp,servo,duty,lidar` ; le champ
timestamp ISO local est conservé tel quel dans le journal de poses, car il ne
contient pas de fuseau horaire.

La pose est estimée par ICP entre scans successifs. Les correspondances trop
faibles sont rejetées et la pose précédente est conservée ; elles sont
signalées dans la sortie. C’est un premier mapping séquentiel sans fermeture de
boucle, pas un SLAM complet. La dérive peut s’accumuler, notamment dans les
couloirs ou avec peu de structure, et le mouvement pendant un tour du LiDAR
n’est pas compensé.

Le dépôt contient aussi un viewer polaire dans
`scripts/Behavioral_Cloning_Lidar.py` (`--preview`, port 5001). Il affiche le
scan courant du collecteur Python ; il n’affiche pas encore la carte Rust ni la
caméra. Le collecteur attend également la manette et se connecte au contrôleur
moteur, donc ne le lance pas comme simple viewer sur une voiture prête à rouler.

## Étape 3 : capture OAK-D Lite et synchronisation temporelle

Le pont utilise les mêmes API DepthAI v2.29 que les scripts du dépôt : caméra
mono `CAM_B`, `THE_480_P`, et `getCvFrame()`. Il envoie des JPEG avec leur
timestamp Unix en nanosecondes. Le receiver Rust enregistre les images dans
`lidarcontrol/data/camera/frames/` et les indexe dans `frames.jsonl`.

Terminal 1 sur la Jetson :

```bash
python3 lidarcontrol/tools/oak_bridge.py --fps 15
```

Terminal 2 :

```bash
cargo run --release --manifest-path lidarcontrol/Cargo.toml -- camera-record
```

Après connexion du programme Rust, la preview mono est disponible à
`http://<jetson-ip>:9011/`. Elle affiche seulement la caméra. Pour enregistrer
en parallèle les scans LiDAR, lancer aussi `lidarcontrol record` dans un autre
terminal ; le pilotage manuel peut rester dans le contrôleur existant.

Après avoir produit la carte et le journal de poses, associer chaque pose au
frame caméra le plus proche :

```bash
cargo run --release --manifest-path lidarcontrol/Cargo.toml -- sync \
  --poses lidarcontrol/maps/floor.poses.jsonl \
  --frames lidarcontrol/data/camera/frames.jsonl
```

La tolérance par défaut est de 80 ms (`camera_sync_tolerance_ms` dans
`config.toml`) ; chaque paire indique l’écart de temps signé. Les deux temps
utilisent l’horloge Unix de la Jetson. Le timestamp caméra est pris à la
réception de la frame par DepthAI côté hôte, pas à l’exposition du capteur ;
celui du LiDAR est estimé au paquet contenant le bin avant central. C’est une
synchronisation d’arrivée approximative, à mesurer et valider sur les logs.
Les extrinsèques LiDAR-caméra ne sont pas encore appliquées ; l’assemblage
spatial, la fermeture de boucle visuelle et la fusion obstacle restent à faire.

## Étape 4 : localisation sur la carte enregistrée

La commande `localize` recharge le PGM/YAML et fait un appariement scan-vers-
carte autour de la dernière pose (ou d’une pose initiale fournie en mètres et
degrés) :

```bash
cargo run --release --manifest-path lidarcontrol/Cargo.toml -- localize \
  --map lidarcontrol/maps/floor.yaml \
  --input lidarcontrol/data/scan.jsonl \
  --initial-x 0 --initial-y 0 --initial-yaw-deg 0
```

La sortie JSONL contient `x_m`, `y_m`, `yaw_rad`, confiance, erreur moyenne et
un indicateur `stop`. Sous le seuil `localization_min_confidence` (défaut 0.30),
la pose retenue est conservée et le traitement marque `STOP`. Cette commande
traite des enregistrements : elle n’envoie pas encore d’arrêt au Flipsky. La
relocalisation globale si le point de départ est inconnu, la fusion des images
et le contrôleur d’arrêt en temps réel restent à intégrer. Le CSV historique
peut servir à la carte/localisation, mais pas à l’association temporelle avec
la caméra, car son timestamp local ne contient pas de fuseau horaire.

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

Cette étape n’envoie aucune commande au Flipsky/VESC.

## Étape 5 : planification, interface et replay de suivi

Planifier une route A* sur les cellules connues libres. Les cellules inconnues
et occupées sont bloquées, et la marge d’obstacle est configurable :

```bash
cargo run --release --manifest-path lidarcontrol/Cargo.toml -- plan \
  --map lidarcontrol/maps/floor.yaml --start 1.0 1.0 --goal 3.0 2.0 --margin 0.25
```

L’interface web locale propose de cliquer le départ et l’arrivée, puis affiche
la route calculée :

```bash
cargo run --release --manifest-path lidarcontrol/Cargo.toml -- ui \
  --map lidarcontrol/maps/floor.yaml --bind 127.0.0.1:8765
```

Ouvrir `http://127.0.0.1:8765`. Elle ne commande aucun moteur. Le départ doit
être cliqué car aucune pose courante temps réel n’est publiée à l’interface.

Le replay de navigation relocalise les scans enregistrés, calcule la poursuite
Pure Pursuit et journalise les commandes simulées. Il arrête le replay si la
localisation tombe sous le seuil, si un scan est absent plus de 500 ms, ou si
un obstacle apparaît dans le secteur avant sous `obstacle_stop_distance_m` :

```bash
cargo run --release --manifest-path lidarcontrol/Cargo.toml -- simulate \
  --map lidarcontrol/maps/floor.yaml --input lidarcontrol/data/scan.jsonl \
  --start 1.0 1.0 --goal 3.0 2.0 --initial-yaw-deg 0
```

Le suivi simulé exige de renseigner `simulation_wheelbase_m`,
`simulation_max_steering_rad` et les valeurs de servo gauche/centre/droite dans
`config.toml`. Elles sont volontairement absentes tant que les mesures de la
voiture ne sont pas fournies. Cette commande est un replay des commandes, pas
une simulation dynamique complète du véhicule, et ne pilote pas le VESC.

## Limites et étape matérielle restante

La carte est construite par ICP scan-à-scan sans IMU, encodeurs ni fermeture de
boucle : la dérive s’accumule, surtout dans les couloirs longs ou les pièces
symétriques. Le LiDAR plan peut manquer les obstacles au-dessus/en dessous de
son plan et peut mal voir le verre. La caméra est capturée et appariée par
horodatage hôte, mais elle n’est pas encore fusionnée dans la carte ou la
localisation ; ses extrinsèques ne sont pas définies. Il n’y a pas de
localisation temps réel, de détection caméra d’obstacle, d’évitement local
actif, d’interface de progression en direct, de bouton d’arrêt moteur ni de
sortie VESC dans `lidarcontrol`.

Ces fonctions ne peuvent pas être activées sans les mesures de géométrie et de
servo demandées, la transformation extrinsèque LiDAR-caméra, ainsi que la
confirmation du comportement électrique d’arrêt attendu pour le Flipsky. Les
protections de cette version s’appliquent uniquement au replay et au
planificateur logiciel.
