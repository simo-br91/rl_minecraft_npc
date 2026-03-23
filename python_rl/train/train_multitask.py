"""
train_multitask.py
------------------
Trains a shared PPO policy on the **combined multitask episode**.

Every episode in "multitask" mode sets up ALL three tasks simultaneously:
  • Navigation  — a gold-block target the agent must reach
  • Farming     — N wheat crops to find, navigate to, and harvest
  • Combat      — 2 hostile mobs (zombie + skeleton) that attack the agent

The agent must use its full 13-action set and inventory to:
  - Kill mobs with the iron sword when they are nearby
  - Navigate to and harvest each crop with the interact action
  - Reach the navigation target marker to complete the episode
  - Eat cooked beef when food level is low
  - Sprint when no mobs are close to cover distance faster

Success = all crops harvested AND agent within 1.5 blocks of the nav target.
Truncation = maxSteps (400) exceeded, OR agent dies.

This design produces a genuinely multi-skill agent that must judge the
situation and adapt its behaviour — rather than simply alternating between
single-task episodes.

Warm-start: attempts to load from multitask_run1 (resume), then farming,
then navigation — whichever exists first.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor

from python_rl.env.minecraft_env import MinecraftEnv
from python_rl.train.train_utils import (
    SuccessLogger,
    EarlyStoppingCallback,
    make_periodic_checkpoint,
    load_config,
    load_model_with_warmstart,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="multitask")
    parser.add_argument("--crops",   type=int, default=5,
                        help="Wheat plots per multitask episode.")
    parser.add_argument("--resume",  action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    logs_dir        = Path("python_rl/logs")
    checkpoints_dir = Path("python_rl/checkpoints")
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Single combined episode type — agent must handle all tasks at once
    env = MinecraftEnv(task="multitask", num_crops=args.crops)
    env = Monitor(env, filename=str(logs_dir / "multitask_monitor.csv"))

    success_log   = str(logs_dir / "multitask_success.csv")
    success_cb    = SuccessLogger(success_log)
    checkpoint_cb = make_periodic_checkpoint(
        str(checkpoints_dir / "multitask_checkpoints"),
        prefix="multitask",
    )
    early_stop_cb = EarlyStoppingCallback(
        success_log_path=success_log,
        target_success_rate=0.80,
        window=50,
        patience=3,
    )

    # Warm-start priority: resumed run > best farming > best navigation
    warm_candidates = []
    if args.resume:
        warm_candidates.append(checkpoints_dir / "multitask_run1")
    warm_candidates += [
        checkpoints_dir / "farm_run1",
        checkpoints_dir / "nav_curriculum_run1",
        checkpoints_dir / "nav_shaped_run1",
    ]

    model = load_model_with_warmstart(warm_candidates, env, cfg, str(logs_dir))

    model.learn(
        total_timesteps=cfg.get("total_timesteps", 500_000),
        callback=CallbackList([success_cb, checkpoint_cb, early_stop_cb]),
        reset_num_timesteps=not args.resume,
    )
    model.save(str(checkpoints_dir / "multitask_run1"))
    env.close()

    print("Multi-task training complete.")
    print("Checkpoint : python_rl/checkpoints/multitask_run1")
    print("Monitor CSV: python_rl/logs/multitask_monitor.csv")
    print("Success CSV: python_rl/logs/multitask_success.csv")


if __name__ == "__main__":
    main()