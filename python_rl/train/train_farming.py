"""
train_farming.py
----------------
Trains single-task farming: navigate to wheat, optionally bonemeal it,
harvest, repeat for all crops in the episode.

Key changes vs Day 1:
 - Multi-crop episodes (default 5 wheat plots)
 - Optional full farming cycle (seeds → bonemeal → harvest)
 - Episode ends immediately when ALL crops harvested (no wasted steps)
 - Periodic checkpoints every 25k steps
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3.common.callbacks import CallbackList

from python_rl.env.minecraft_env import MinecraftEnv
from python_rl.train.train_utils import (
    SuccessLogger, make_periodic_checkpoint, load_config, wrap_env, load_model_with_warmstart
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="farming")
    parser.add_argument("--num-crops", type=int, default=5, help="Wheat plots per episode")
    parser.add_argument("--full-cycle", action="store_true",
                        help="Full farming: seeds → bonemeal → harvest")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    logs_dir        = Path("python_rl/logs")
    checkpoints_dir = Path("python_rl/checkpoints")
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    env = MinecraftEnv(
        task="farming",
        num_crops=args.num_crops,
        full_farm_cycle=args.full_cycle,
    )
    vec_env, _ = wrap_env(env, str(logs_dir / "farm_monitor.csv"), normalize=False)

    success_cb    = SuccessLogger(str(logs_dir / "farm_success.csv"))
    checkpoint_cb = make_periodic_checkpoint(
        str(checkpoints_dir / "farm_checkpoints"), prefix="farm")

    warmstart = [checkpoints_dir / "farm_run1"] if args.resume else []
    model = load_model_with_warmstart(warmstart, vec_env, cfg, str(logs_dir))

    model.learn(
        total_timesteps=cfg.get("total_timesteps", 400_000),
        callback=CallbackList([success_cb, checkpoint_cb]),
        reset_num_timesteps=not args.resume,
    )
    model.save(str(checkpoints_dir / "farm_run1"))
    vec_env.close()
    print("Farming training complete.")
    print(f"  num_crops={args.num_crops}, full_cycle={args.full_cycle}")
    print("Checkpoint : python_rl/checkpoints/farm_run1")


if __name__ == "__main__":
    main()