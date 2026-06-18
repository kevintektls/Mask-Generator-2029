# Reinforcement Learning — conduite entre les bandes blanches (caméra seule)

Pipeline RL pour améliorer `model/pilot_model.pth` via le simulateur Unity.
**Pas de lidar** : la perception repose uniquement sur la caméra (masque binaire des lignes blanches), comme sur la Jetson.

## Architecture

```
Unity (CameraLaneRLBridge.cs)
    │  image RGB caméra + métriques piste
    ▼ TCP :7777
Python (rl/unity_env.py + rl/train_rl.py)
    │  masque 160×120 → CNN (pilot_model.pth)
    ▼ PPO (Stable-Baselines3)
model/pilot_model_rl.pth  →  scripts/Autopilot_IA++.py
```

## Nombre d'agents (important)

**Unity spawn une voiture par entrée dans `agents-config.json`.**  
Si le fichier contient 100 entrées, Unity charge 100 voitures même si Python n'en utilise que 50 → lag.

**Source unique de vérité** : `Assets/agents-config.json` (50 agents par défaut).

Changer le nombre d'agents :
```bash
python rl/sync_agents_config.py 50
```
Puis **Stop Play → Play** dans Unity.

`train_rl.py` lit automatiquement le nombre d'agents dans ce fichier.

## Entraînement multi-agents (plus rapide)

Par défaut **50 voitures** tournent en parallèle dans Unity.

1. **`CameraLaneRLBridge`** doit être sur le prefab `AgentCar Continuous` (pas besoin de le dupliquer à la main).
2. **`Assets/agents-config.json`** : une entrée `agents` par voiture (50 par défaut).
3. Au **Play**, la Console affiche les ports **7777–7826**.
4. Python (lit le nombre d'agents depuis `agents-config.json`) :
   ```bash
   python rl/train_rl.py
   ```

Une seule voiture :
```bash
python rl/train_rl.py --n-envs 1
```
(et remets 1 seule entrée dans `agents-config.json`)

## 1. Configuration Unity

1. Ouvrir `MaskGenerator-main/unitySimulator` dans Unity 6 / ML-Agents 3.0.
2. Sur le prefab **`Assets/Prefabs/AgentCar Continuous`** :
   - **Ajouter** `CameraLaneRLBridge` (une fois sur le prefab — `ConfigLoader` duplique les voitures).
   - Assigner : `carController`, `carVisionCamera`.
   - Cocher `rlMode = true` (le port est assigné automatiquement : 7777, 7778, …).
3. Vérifier **`Assets/agents-config.json`** (50 agents par défaut pour le RL).
4. Appuyer sur **Play**.

## 2. Installation Python

```bash
pip install -r requirements-rl.txt
```

## 3. Entraînement

Terminal 1 — Unity en Play.

Terminal 2 :

```bash
python rl/train_rl.py
```

Options utiles :

```bash
python rl/train_rl.py --timesteps 500000 --fixed-throttle 0.35 --bc-model model/pilot_model.pth
```

Le modèle final est exporté dans `model/pilot_model_rl.pth` (format identique à `pilot_model.pth`).

## 4. Reprendre l'entraînement (autre circuit / généralisation)

Une fois le circuit simple maîtrisé, **repartir du checkpoint PPO** (pas de `pilot_model.pth`) :

```bash
py rl/train_rl.py --resume model/rl_checkpoints/ppo_lane_final.zip --timesteps 500000 --lr 1e-4
```

- `--resume` : charge le réseau déjà entraîné
- `--lr 1e-4` : learning rate plus bas pour affiner (recommandé)
- `--timesteps 500000` : steps **supplémentaires** (pas un total cumulé affiché)

Export Jetson après la session :
```bash
py rl/train_rl.py --distill-only model/rl_checkpoints/ppo_lane_final.zip
```

### Multi-circuits automatique (Unity)

1. Sur **GameManager**, ajouter le script **`RlTrackRotator`**
2. Assigner le **TrackDropDown** de la scène
3. Cocher `rlMode = true`
4. `switchIntervalSeconds = 180` → change de circuit toutes les 3 minutes
5. `randomOrder = true` → circuits aléatoires

Les circuits disponibles sont ceux du dropdown Unity (Track1, Track2, Track3…).

### Multi-circuits manuel (session par session)

1. Unity **Play** → dropdown **Track2**
2. `py rl/train_rl.py --resume model/rl_checkpoints/ppo_lane_final.zip --timesteps 300000 --lr 1e-4`
3. Puis **Track3**, relancer `--resume` avec le nouveau `ppo_lane_final.zip`

### Stratégie recommandée

| Phase | Circuit | Commande |
|-------|---------|----------|
| 1 | Circuit simple | `py rl/train_rl.py` (déjà fait) |
| 2 | Circuit simple + autres (rotator) | `--resume ppo_lane_final.zip --timesteps 500000 --lr 1e-4` |
| 3 | Affinage | `--resume ... --timesteps 300000 --fixed-throttle 0.25` |
| 4 | Export Jetson | `--distill-only ppo_lane_final.zip` |

## 5. Déploiement Jetson

Copier `model/pilot_model_rl.pth` sur la Jetson et modifier `MODEL_PATH` dans `scripts/Autopilot_IA++.py` :

```python
MODEL_PATH = "../model/pilot_model_rl.pth"
```

## Fonction de récompense

| Signal | Effet |
|--------|-------|
| Centrage entre les bandes | + jusqu'à +2.5 |
| Avancer (vitesse) | +0.05 × speed |
| Changement brusque de direction | −0.3 |
| Sortie de piste (tag `Lines`) | −50, épisode terminé |

## Notes

- Les raycasts Unity existants (`Raycast.cs`) ne sont **pas** utilisés en mode RL : seule la caméra compte.
- Chaque caméra RL **ignore le layer `Player`** → les voitures ne se voient pas entre elles (comme sur la Jetson).
- L'initialisation depuis `pilot_model.pth` (Behavioral Cloning) accélère fortement la convergence.
- Checkpoints intermédiaires : `model/rl_checkpoints/`.
