#!/usr/bin/env python3
"""
Entraînement Reinforcement Learning — conduite entre les bandes blanches (caméra seule).

Prérequis :
  1. Unity : ouvrir MaskGenerator-main/unitySimulator, activer rlMode sur CameraLaneRLBridge, Play.
  2. Python : pip install -r requirements-rl.txt
  3. Lancer : python rl/train_rl.py

Initialise depuis model/pilot_model.pth (Behavioral Cloning).
Exporte vers model/pilot_model_rl.pth (compatible Autopilot_IA++.py).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rl.models import BehavioralCloningCNN
from rl.policy import MaskCNNExtractor, init_policy_from_bc
from rl.sync_agents_config import DEFAULT_CONFIG, read_agent_count
from rl.unity_env import UnityLaneEnv


def resolve_device(request: str = "auto") -> torch.device:
    """Choisit cuda/cpu et affiche un diagnostic si le GPU est absent."""
    if request == "cpu":
        return torch.device("cpu")

    if request == "cuda" and not torch.cuda.is_available():
        print("[RL] ERREUR: --device cuda demandé mais CUDA indisponible.")
        print_cuda_help()
        sys.exit(1)

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[RL] GPU détecté : {name} (CUDA {torch.version.cuda})")
        return torch.device("cuda")

    print(f"[RL] PyTorch {torch.__version__} — pas de GPU, fallback CPU")
    if "+cpu" in torch.__version__:
        print_cuda_help()
    return torch.device("cpu")


def print_cuda_help() -> None:
    print(
        "[RL] Installe PyTorch CUDA (RTX 3070) :\n"
        "     py -m pip install --upgrade --force-reinstall torch "
        "--index-url https://download.pytorch.org/whl/cu126"
    )


def can_use_progress_bar() -> bool:
    try:
        import rich  # noqa: F401
        import tqdm  # noqa: F401
        return True
    except ImportError:
        return False


@torch.no_grad()
def ppo_steering_to_servo(policy, obs: torch.Tensor) -> torch.Tensor:
    """Sortie direction PPO complète (CNN → MLP → action) mappée en servo [0, 1]."""
    actions, _, _ = policy(obs)
    return ((actions[:, 0] + 1.0) * 0.5).clamp(0.0, 1.0)


def distill_to_bc(
    sb3_model: PPO,
    out_path: Path,
    device: torch.device,
    steps: int = 2000,
    batch_size: int = 64,
) -> None:
    """
    Distille la policy PPO vers un BehavioralCloningCNN (sortie servo 0..1).
    Compatible scripts/Autopilot_IA++.py sur la Jetson.
    """
    policy = sb3_model.policy
    policy.eval()
    fe: MaskCNNExtractor = policy.features_extractor

    bc = BehavioralCloningCNN().to(device)
    bc.load_state_dict(fe.cnn.state_dict())

    opt = torch.optim.Adam(bc.parameters(), lr=1e-3)
    bc.train()

    for step in range(steps):
        obs = torch.rand(batch_size, 1, 120, 160, device=device)
        with torch.no_grad():
            target = ppo_steering_to_servo(policy, obs)

        pred = bc(obs)
        loss = F.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 400 == 0:
            print(f"[RL] Distillation {step + 1}/{steps} — loss={loss.item():.5f}")

    bc.eval()
    torch.save(bc.state_dict(), out_path)
    print(f"[RL] Modele Jetson exporte -> {out_path}")


def make_env(host: str, port: int, fixed_throttle: float | None):
    def _init():
        return UnityLaneEnv(host=host, port=port, fixed_throttle=fixed_throttle)

    return _init


def build_vec_env(
    n_envs: int,
    host: str,
    base_port: int,
    fixed_throttle: float | None,
):
    env_fns = [make_env(host, base_port + i, fixed_throttle) for i in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    print(f"[RL] {n_envs} environnements parallèles (ports {base_port}–{base_port + n_envs - 1})")
    return SubprocVecEnv(env_fns, start_method="spawn")


def main():
    parser = argparse.ArgumentParser(description="RL lane keeping — caméra seule, Unity simulator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--agents-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-port", type=int, default=7777)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Doit correspondre au nombre d'entrées dans agents-config.json (Unity spawn tout le fichier)",
    )
    parser.add_argument("--bc-model", default=str(ROOT / "model" / "pilot_model.pth"))
    parser.add_argument("--out-model", default=str(ROOT / "model" / "pilot_model_rl.pth"))
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fixed-throttle", type=float, default=0.22,
                        help="Vitesse fixe en sim (plus bas = meilleur pour apprendre les virages)")
    parser.add_argument("--distill-only", type=str, default=None,
                        help="Exporte seulement : chemin vers ppo_lane_final.zip (sans ré-entraîner)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Reprend un checkpoint PPO (ex: model/rl_checkpoints/ppo_lane_final.zip)")
    parser.add_argument("--no-bc-init", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device PyTorch pour le réseau PPO (auto = GPU si disponible)",
    )
    parser.add_argument("--save-dir", default=str(ROOT / "model" / "rl_checkpoints"))
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Barre de progression (nécessite tqdm + rich). Défaut : auto.",
    )
    args = parser.parse_args()

    config_count = read_agent_count(args.agents_config)
    n_envs = config_count if args.n_envs is None else args.n_envs
    if n_envs != config_count:
        print(
            f"[RL] ERREUR: --n-envs={n_envs} mais agents-config.json contient {config_count} agents.\n"
            f"     Unity charge TOUTES les entrées du JSON → lag si le fichier est trop gros.\n"
            f"     Corrige avec: python rl/sync_agents_config.py {n_envs}"
        )
        sys.exit(1)

    use_progress_bar = args.progress_bar if args.progress_bar is not None else can_use_progress_bar()
    if use_progress_bar and not can_use_progress_bar():
        print("[RL] tqdm/rich manquants — barre de progression désactivée.")
        print("     Installe : pip install tqdm rich")
        use_progress_bar = False

    os.makedirs(args.save_dir, exist_ok=True)
    device = resolve_device(args.device)
    print(f"[RL] Device: {device}")
    print(f"[RL] {n_envs} agent(s) — lu depuis {args.agents_config.name}")

    if args.distill_only:
        device = resolve_device(args.device)
        print(f"[RL] Export seul depuis {args.distill_only}")
        model = PPO.load(args.distill_only, device=str(device))
        distill_to_bc(model, Path(args.out_model), device)
        print("[RL] Export termine.")
        return

    env = build_vec_env(n_envs, args.host, args.base_port, args.fixed_throttle)

    policy_kwargs = dict(
        features_extractor_class=MaskCNNExtractor,
        features_extractor_kwargs=dict(features_dim=50),
        net_arch=dict(pi=[64, 32], vf=[64, 32]),
        activation_fn=nn.ReLU,
    )

    if args.resume:
        print(f"[RL] Reprise depuis {args.resume}")
        model = PPO.load(args.resume, env=env, device=str(device))
    else:
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=args.lr,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            policy_kwargs=policy_kwargs,
            device=str(device),
        )

        if not args.no_bc_init and os.path.exists(args.bc_model):
            init_policy_from_bc(model, args.bc_model, device)
        else:
            print("[RL] Entraînement sans initialisation BC")

    checkpoint_cb = CheckpointCallback(
        save_freq=max(5_000, 10_000 // n_envs),
        save_path=args.save_dir,
        name_prefix="ppo_lane",
    )

    print(f"[RL] Debut entrainement ({args.timesteps} steps)...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_cb,
            progress_bar=use_progress_bar,
            reset_num_timesteps=not bool(args.resume),
        )
    except KeyboardInterrupt:
        print("\n[RL] Interruption — sauvegarde du modèle courant...")

    model.save(os.path.join(args.save_dir, "ppo_lane_final"))
    try:
        distill_to_bc(model, Path(args.out_model), device)
    except Exception as exc:
        print(f"[RL] ERREUR export Jetson : {exc}")
        print(f"[RL] Le checkpoint PPO est sauvegardé : {args.save_dir}/ppo_lane_final.zip")
        print(f"[RL] Réessaie : py rl/train_rl.py --distill-only {args.save_dir}/ppo_lane_final.zip")
    env.close()
    print("[RL] Terminé.")


if __name__ == "__main__":
    main()
