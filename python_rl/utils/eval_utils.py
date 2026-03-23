"""
eval_utils.py
-------------
Shared helpers used by all evaluate_*.py scripts.

Eliminates the copy-pasted run_episode() that previously appeared in
evaluate.py, evaluate_farming.py, and evaluate_multitask.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO

from python_rl.env.minecraft_env import MinecraftEnv

CHECKPOINTS_DIR = Path("python_rl/checkpoints")


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_model(name: str) -> PPO:
    """Load a PPO checkpoint by name (no extension needed)."""
    path = CHECKPOINTS_DIR / name
    zip_path = path.with_suffix(".zip")
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {zip_path}\n"
            "Train first with the corresponding train_*.py script."
        )
    return PPO.load(str(path))


# ------------------------------------------------------------------
# Episode runner
# ------------------------------------------------------------------

def run_episode(
    model: PPO,
    env: MinecraftEnv,
    task_name: str,
    *,
    reset_options: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """
    Run one deterministic episode and return aggregate stats.

    Parameters
    ----------
    model        : trained PPO model
    env          : MinecraftEnv instance (already connected)
    task_name    : passed as ``options["task"]`` to env.reset()
    reset_options: additional reset options merged over the task default
    verbose      : print per-step info

    Returns
    -------
    dict with keys: success, steps, total_reward, crops_harvested,
                    mobs_killed, health, food_level, task_progress, info
    """
    opts = {"task": task_name}
    if reset_options:
        opts.update(reset_options)

    obs, _ = env.reset(options=opts)
    done = truncated = False
    total_reward = 0.0
    steps = 0

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if verbose:
            dist   = info.get("distance_to_target", float("nan"))
            prog   = info.get("task_progress", 0.0)
            health = info.get("health", 20.0)
            food   = info.get("food_level", 20)
            item   = info.get("active_item", "?")
            crops  = info.get("crops_harvested", 0)
            mobs   = info.get("mobs_killed", 0)
            print(
                f"  step={steps:3d}  action={int(action):2d}  "
                f"reward={reward:+.3f}  dist={dist:.2f}  prog={prog:.2f}  "
                f"hp={health:.0f}  food={food}  item={item}  "
                f"crops={crops}  mobs_k={mobs}  "
                f"done={done}  trunc={truncated}"
            )

    return {
        "success":        info.get("success", False),
        "steps":          steps,
        "total_reward":   round(total_reward, 3),
        "crops_harvested": info.get("crops_harvested", 0),
        "mobs_killed":    info.get("mobs_killed", 0),
        "health":         info.get("health", 20.0),
        "food_level":     info.get("food_level", 20),
        "task_progress":  info.get("task_progress", 0.0),
        "info":           info,
    }


# ------------------------------------------------------------------
# Multi-episode summary
# ------------------------------------------------------------------

def run_episodes(
    model: PPO,
    env: MinecraftEnv,
    task_name: str,
    n_episodes: int,
    *,
    reset_options: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """
    Run n_episodes and return aggregate statistics.
    """
    all_stats = []
    for ep in range(1, n_episodes + 1):
        if verbose:
            print(f"\n--- Episode {ep}/{n_episodes} ---")
        stats = run_episode(model, env, task_name,
                            reset_options=reset_options, verbose=verbose)
        all_stats.append(stats)
        if verbose:
            print(
                f"  → success={stats['success']}  steps={stats['steps']}  "
                f"reward={stats['total_reward']}  "
                f"crops={stats['crops_harvested']}  mobs={stats['mobs_killed']}"
            )

    n = len(all_stats)
    return {
        "task":           task_name,
        "episodes":       n,
        "success_rate":   round(sum(s["success"] for s in all_stats) / n, 3),
        "avg_reward":     round(np.mean([s["total_reward"]   for s in all_stats]), 3),
        "avg_steps":      round(np.mean([s["steps"]          for s in all_stats]), 1),
        "avg_crops":      round(np.mean([s["crops_harvested"] for s in all_stats]), 2),
        "avg_mobs":       round(np.mean([s["mobs_killed"]    for s in all_stats]), 2),
        "avg_health":     round(np.mean([s["health"]         for s in all_stats]), 1),
    }