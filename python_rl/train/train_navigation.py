"""
train_navigation.py
-------------------
Trains navigation with SHAPED rewards (primary baseline).

Reads hyperparameters from python_rl/configs/nav_shaped.yaml.
Saves periodic checkpoints every 25k steps.
Uses VecNormalize for observation normalization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3.common.callbacks import CallbackList

from python_rl.env.minecraft_env import MinecraftEnv
from python_rl.train.train_utils import (
    SuccessLogger, EarlyStoppingCallback, make_periodic_checkpoint,
    load_config, wrap_env, load_model_with_warmstart
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="nav_shaped")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    logs_dir        = Path("python_rl/logs")
    checkpoints_dir = Path("python_rl/checkpoints")
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    env = MinecraftEnv(task="navigation")
    vec_env, vec_norm = wrap_env(env, str(logs_dir / "nav_shaped_monitor.csv"), normalize=True)

    success_log   = str(logs_dir / "nav_shaped_success.csv")
    success_cb    = SuccessLogger(success_log)
    checkpoint_cb = make_periodic_checkpoint(
        str(checkpoints_dir / "nav_shaped_checkpoints"), prefix="nav_shaped")
    early_stop_cb = EarlyStoppingCallback(
        success_log_path=success_log,
        target_success_rate=0.95,
        window=50,
        patience=3,
    )

    warmstart = [checkpoints_dir / "nav_shaped_run1"] if args.resume else []
    model = load_model_with_warmstart(warmstart, vec_env, cfg, str(logs_dir))

    model.learn(
        total_timesteps=cfg.get("total_timesteps", 200_000),
        callback=CallbackList([success_cb, checkpoint_cb, early_stop_cb]),
        reset_num_timesteps=not args.resume,
    )
    model.save(str(checkpoints_dir / "nav_shaped_run1"))
    if vec_norm is not None:
        vec_norm.save(str(checkpoints_dir / "nav_shaped_vecnorm.pkl"))
        
    vec_env.close()
    print("Navigation (shaped) training complete.")
    print("Checkpoint : python_rl/checkpoints/nav_shaped_run1")


if __name__ == "__main__":
    main()