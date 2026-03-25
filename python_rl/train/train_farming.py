"""
train_farming.py
----------------
Trains the single-task farming agent.

Key features:
 - Multi-crop episodes (default 5 wheat plots)
 - Optional full farming cycle (seeds → bonemeal → harvest)
 - Optional 4-level farming curriculum
 - Episode ends immediately when all crops are harvested
 - Periodic checkpoints every 25 k steps
 - Early stopping at 90% rolling success rate

Usage
-----
    python -m python_rl.train.train_farming
    python -m python_rl.train.train_farming --num-crops 10 --full-cycle
    python -m python_rl.train.train_farming --curriculum
    python -m python_rl.train.train_farming --resume
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor

from python_rl.env.minecraft_env import MinecraftEnv
from python_rl.train.curriculum_scheduler import FarmingCurriculumScheduler
from python_rl.train.train_utils import (
    SuccessLogger,
    EarlyStoppingCallback,
    make_periodic_checkpoint,
    load_config,
    load_model_with_warmstart,
)


# ------------------------------------------------------------------
# Curriculum wrapper for farming
# ------------------------------------------------------------------

class FarmingCurriculumEnv(gym.Wrapper):
    """Injects current farming curriculum level into every reset()."""

    def __init__(self, env: MinecraftEnv,
                 scheduler: FarmingCurriculumScheduler) -> None:
        super().__init__(env)
        self.scheduler = scheduler
        self._timestep = 0

    def reset(self, seed=None, options=None):
        opts = self.scheduler.get_reset_options(task="farming")
        if options:
            opts.update(options)
        return self.env.reset(seed=seed, options=opts)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._timestep += 1
        if done or truncated:
            self.scheduler.record_episode(
                success=bool(info.get("success", False)),
                timestep=self._timestep,
            )
        return obs, reward, done, truncated, info


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="farming")
    parser.add_argument("--num-crops",  type=int, default=5,
                        help="Wheat plots per episode (ignored with --curriculum).")
    parser.add_argument("--full-cycle", action="store_true",
                        help="Full farming: seeds → bonemeal → harvest.")
    parser.add_argument("--curriculum", action="store_true",
                        help="Use 4-level farming curriculum.")
    parser.add_argument("--resume",     action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    logs_dir        = Path("python_rl/logs")
    checkpoints_dir = Path("python_rl/checkpoints")
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    success_log = str(logs_dir / "farm_success.csv")

    base_env = MinecraftEnv(
        task="farming",
        num_crops=args.num_crops,
        full_farm_cycle=args.full_cycle,
    )

    if args.curriculum:
        scheduler = FarmingCurriculumScheduler(
            advance_threshold=0.70,
            regress_threshold=0.35,
            advance_window=20,
            log_path=str(logs_dir / "farm_curriculum_levels.csv"),
        )
        env = FarmingCurriculumEnv(base_env, scheduler)
        print(f"[train_farming] Farming curriculum enabled. "
              f"Starting at level {scheduler.level_number}: "
              f"{scheduler.current_level['description']}")
    else:
        env = base_env

    env = Monitor(env, filename=str(logs_dir / "farm_monitor.csv"))

    success_cb    = SuccessLogger(success_log)
    checkpoint_cb = make_periodic_checkpoint(
        str(checkpoints_dir / "farm_checkpoints"), prefix="farm")
    early_stop_cb = EarlyStoppingCallback(
        success_log_path=success_log,
        target_success_rate=0.90,
        window=30,
        patience=1,
        task_name="Farming",
    )

    warmstart = [checkpoints_dir / "farm_run1"] if args.resume else []
    model = load_model_with_warmstart(warmstart, env, cfg, str(logs_dir))

    model.learn(
        total_timesteps=cfg.get("total_timesteps", 400_000),
        callback=CallbackList([success_cb, checkpoint_cb, early_stop_cb]),
        reset_num_timesteps=not args.resume,
    )
    model.save(str(checkpoints_dir / "farm_run1"))
    env.close()

    print("Farming training complete.")
    if args.curriculum:
        print(f"Final scheduler state: {scheduler}")
    print("Checkpoint :", checkpoints_dir / "farm_run1")


if __name__ == "__main__":
    main()