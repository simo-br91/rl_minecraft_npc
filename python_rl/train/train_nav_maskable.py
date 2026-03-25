"""
train_nav_maskable.py
---------------------
Navigation training using sb3-contrib MaskablePPO with action masking.

Action masking prevents the policy from sampling actions that are known to
be invalid at each step (e.g., interact when no crop is in front, attack
when no mob is nearby, jump when no wall is ahead).  This typically speeds
up convergence and reduces wasted samples.

The Java bridge exposes /masks (GET) which returns a 13-element validity
array computed from the same preconditions as ActionExecutor.  The mask is
fetched by MinecraftEnv.action_masks() after every step.

Usage
-----
    # Install sb3-contrib first:
    pip install sb3-contrib

    python -m python_rl.train.train_nav_maskable
    python -m python_rl.train.train_nav_maskable --config nav_shaped
    python -m python_rl.train.train_nav_maskable --resume

Implements Issue 6.3 — MaskablePPO / action masking.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3.common.callbacks import CallbackList

from python_rl.env.minecraft_env import MaskableMinecraftEnv
from python_rl.train.train_utils import (
    SuccessLogger,
    EarlyStoppingCallback,
    make_periodic_checkpoint,
    load_config,
    wrap_env,
    make_maskable_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Navigation training with MaskablePPO action masking."
    )
    parser.add_argument("--config", default="nav_shaped",
                        help="YAML config name (without .yaml).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    logs_dir        = Path("python_rl/logs")
    checkpoints_dir = Path("python_rl/checkpoints")
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # MaskableMinecraftEnv exposes action_masks() for MaskablePPO
    base_env = MaskableMinecraftEnv(task="navigation")
    vec_env, vec_norm = wrap_env(
        base_env,
        monitor_path=str(logs_dir / "nav_maskable_monitor.csv"),
        normalize=True,
        norm_reward=True,
    )

    success_log   = str(logs_dir / "nav_maskable_success.csv")
    success_cb    = SuccessLogger(success_log)
    checkpoint_cb = make_periodic_checkpoint(
        str(checkpoints_dir / "nav_maskable_checkpoints"),
        prefix="nav_maskable",
    )
    early_stop_cb = EarlyStoppingCallback(
        success_log_path=success_log,
        target_success_rate=0.95,
        window=30,
        patience=1,
    )

    checkpoint = checkpoints_dir / "nav_maskable_run1"
    if args.resume and checkpoint.with_suffix(".zip").exists():
        try:
            from sb3_contrib import MaskablePPO
            print(f"[train_nav_maskable] Resuming from {checkpoint}")
            model = MaskablePPO.load(str(checkpoint), env=vec_env)
            model.learning_rate   = cfg.get("learning_rate", model.learning_rate)
            model.ent_coef        = cfg.get("ent_coef",      model.ent_coef)
            model.tensorboard_log = str(logs_dir / "tb")
            model.verbose = 1
        except Exception as e:
            print(f"[train_nav_maskable] Could not resume: {e} — training from scratch.")
            model = make_maskable_model(vec_env, cfg, str(logs_dir))
    else:
        model = make_maskable_model(vec_env, cfg, str(logs_dir))

    model.learn(
        total_timesteps=cfg.get("total_timesteps", 200_000),
        callback=CallbackList([success_cb, checkpoint_cb, early_stop_cb]),
        reset_num_timesteps=not args.resume,
    )
    model.save(str(checkpoint))
    if vec_norm is not None:
        vec_norm.save(str(checkpoints_dir / "nav_maskable_vecnorm.pkl"))

    vec_env.close()
    print("Navigation (masked) training complete.")
    print(f"Checkpoint  : {checkpoint}.zip")
    print(f"Monitor CSV : {logs_dir}/nav_maskable_monitor.csv")
    print(f"Success CSV : {logs_dir}/nav_maskable_success.csv")


if __name__ == "__main__":
    main()
