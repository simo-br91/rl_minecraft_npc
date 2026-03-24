"""
python_rl/utils/eval_utils.py
-----------------------------
Shared evaluation utilities used by evaluate_navigation.py,
evaluate_farming.py, evaluate_combat.py, evaluate_multitask.py.

Full type annotations throughout. (Issue 8.1)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from python_rl.env.minecraft_env import MinecraftEnv

CHECKPOINTS_DIR = Path("python_rl/checkpoints")

# Type alias: eval scripts may pass either a raw MinecraftEnv or a VecEnv
# (e.g. DummyVecEnv + VecNormalize for evaluate_navigation).
AnyEnv = Union[MinecraftEnv, VecEnv]


def load_model(model_name: str) -> PPO:
    """
    Load a PPO model from the checkpoints directory.

    Parameters
    ----------
    model_name : str
        Checkpoint name (with or without .zip extension).
    """
    path = CHECKPOINTS_DIR / model_name
    if not path.with_suffix(".zip").exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}.zip\n"
            f"Run the training script first, e.g.:\n"
            f"  python -m python_rl.train.train_navigation"
        )
    print(f"[eval_utils] Loading model: {path}")
    return PPO.load(str(path))


def run_episode(
    model:         PPO,
    env:           AnyEnv,
    task:          str,
    reset_options: Optional[Dict[str, Any]] = None,
    verbose:       bool                     = True,
) -> Dict[str, Any]:
    """
    Run a single deterministic episode and return per-episode statistics.

    Returns
    -------
    dict with keys: success, total_reward, steps, crops_harvested, mobs_killed,
                    health, info
    """
    opts: Dict[str, Any] = reset_options or {}
    opts.setdefault("task", task)
    obs, info = env.reset(options=opts)

    done: bool       = False
    truncated: bool  = False
    total_reward:    float = 0.0
    steps:           int   = 0

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        total_reward += reward
        steps        += 1
        if verbose:
            print(
                f"  step={steps:3d}  action={action}  reward={reward:+.3f}"
                f"  dist={info.get('distance_to_target', '?'):.2f}"
                f"  success={info.get('success', False)}"
            )

    return {
        "success":         bool(info.get("success", False)),
        "total_reward":    total_reward,
        "steps":           steps,
        "crops_harvested": int(info.get("crops_harvested", 0)),
        "mobs_killed":     int(info.get("mobs_killed", 0)),
        "health":          float(info.get("health", 20.0)),
        "info":            info,
    }


def run_episodes(
    model:         PPO,
    env:           AnyEnv,
    task:          str,
    n_episodes:    int                      = 5,
    reset_options: Optional[Dict[str, Any]] = None,
    verbose:       bool                     = True,
) -> Dict[str, Any]:
    """
    Run n_episodes and return aggregate statistics.

    Returns
    -------
    dict with keys: task, episodes, success_rate, avg_reward, avg_steps,
                    avg_crops, avg_mobs, avg_health
    """
    successes:    int   = 0
    total_reward: float = 0.0
    total_steps:  int   = 0
    total_crops:  int   = 0
    total_mobs:   int   = 0
    total_health: float = 0.0

    for ep in range(1, n_episodes + 1):
        if verbose:
            print(f"\n--- Episode {ep}/{n_episodes} ---")
        result = run_episode(model, env, task, reset_options, verbose)
        successes    += int(result["success"])
        total_reward += result["total_reward"]
        total_steps  += result["steps"]
        total_crops  += result["crops_harvested"]
        total_mobs   += result["mobs_killed"]
        total_health += result["health"]

        if verbose:
            print(
                f"  → reward={result['total_reward']:+.2f}  "
                f"steps={result['steps']}  "
                f"success={result['success']}"
            )

    return {
        "task":         task,
        "episodes":     n_episodes,
        "success_rate": successes / n_episodes,
        "avg_reward":   total_reward / n_episodes,
        "avg_steps":    total_steps  / n_episodes,
        "avg_crops":    total_crops  / n_episodes,
        "avg_mobs":     total_mobs   / n_episodes,
        "avg_health":   total_health / n_episodes,
    }
