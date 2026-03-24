"""
train_combat.py
---------------
Trains the agent to fight zombies and skeletons.

The agent:
 - Starts with an iron sword (slot 0), food (slot 1), iron armor
 - Must kill all spawned mobs to succeed
 - Can die and episode resets (agent respawns next episode)
 - Uses switch_item, attack, eat actions
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
    parser.add_argument("--config", default="combat")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    logs_dir        = Path("python_rl/logs")
    checkpoints_dir = Path("python_rl/checkpoints")
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    env = MinecraftEnv(task="combat")
    vec_env, _ = wrap_env(env, str(logs_dir / "combat_monitor.csv"), normalize=False)

    success_log   = str(logs_dir / "combat_success.csv")
    success_cb    = SuccessLogger(success_log)
    checkpoint_cb = make_periodic_checkpoint(
        str(checkpoints_dir / "combat_checkpoints"), prefix="combat")
    early_stop_cb = EarlyStoppingCallback(
        success_log_path=success_log,
        target_success_rate=0.90,
        window=30,
        patience=1,
    )

    warmstart = [checkpoints_dir / "combat_run1"] if args.resume else []
    model = load_model_with_warmstart(warmstart, vec_env, cfg, str(logs_dir))

    model.learn(
        total_timesteps=cfg.get("total_timesteps", 300_000),
        callback=CallbackList([success_cb, checkpoint_cb, early_stop_cb]),
        reset_num_timesteps=not args.resume,
    )
    model.save(str(checkpoints_dir / "combat_run1"))
    vec_env.close()
    print("Combat training complete.")
    print("Checkpoint : python_rl/checkpoints/combat_run1")


if __name__ == "__main__":
    main()