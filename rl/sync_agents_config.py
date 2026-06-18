#!/usr/bin/env python3
"""Génère agents-config.json avec exactement N agents (Unity ne spawn que ce fichier)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "MaskGenerator-main" / "unitySimulator" / "Assets" / "agents-config.json"


def write_config(path: Path, n_agents: int, base_port: int = 7777, spacing: float = 1.0) -> None:
    cfg = {
        "rlBasePort": base_port,
        "rlSpawnSpacing": spacing,
        "agents": [{"fov": 180, "nbRay": 10} for _ in range(n_agents)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(f"[sync] {n_agents} agents → {path}")


def read_agent_count(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    agents = data.get("agents")
    if not agents:
        raise ValueError(f"Aucun agent dans {path}")
    return len(agents)


def main():
    parser = argparse.ArgumentParser(description="Sync agents-config.json agent count")
    parser.add_argument("count", type=int, nargs="?", default=50, help="Nombre d'agents (défaut: 50)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-port", type=int, default=7777)
    parser.add_argument("--spacing", type=float, default=1.0)
    args = parser.parse_args()
    write_config(args.config, args.count, args.base_port, args.spacing)


if __name__ == "__main__":
    main()
