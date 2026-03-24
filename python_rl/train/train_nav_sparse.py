"""
train_nav_sparse.py
-------------------
Trains navigation with SPARSE rewards (no distance shaping).
Only terminal signals: +10 on success, tiny step penalty.

Use this to compare against the shaped-reward baseline in
compare_experiments.py.

Reads all hyperparameters from python_rl/configs/nav_sparse.yaml.
Key difference from shaped: ent_coef=0.10 (higher entropy to
encourage exploration when reward signal is near-zero most episodes).
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


# ------------------------------------------------------------------
# Thin env wrapper — injects sparse_reward=True on every reset
# ------------------------------------------------------------------

class SparseNavEnv(MinecraftEnv):
    """Forces sparse_reward=True on every reset, regardless of options."""

    def reset(self, seed=None, options=None):
        opts = dict(options or {})
        opts.setdefault("task", "navigation")
        opts["sparse_reward"] = True
        return super().reset(seed=seed, options=opts)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train navigation with sparse rewards.")
    parser.add_argument("--config", default="nav_sparse",
                        help="YAML config name under python_rl/configs/")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing nav_sparse_run1 checkpoint.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    logs_dir        = Path("python_rl/logs")
    checkpoints_dir = Path("python_rl/checkpoints")
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    success_log = str(logs_dir / cfg.get("success_log", "nav_sparse_success.csv"))

    base_env = SparseNavEnv()
    env      = Monitor(base_env,
                       filename=str(logs_dir / cfg.get("monitor_log",
                                                        "nav_sparse_monitor.csv")))

    success_cb    = SuccessLogger(success_log)
    checkpoint_cb = make_periodic_checkpoint(
        str(checkpoints_dir / "nav_sparse_checkpoints"),
        prefix="nav_sparse",
    )
    # Sparse training needs a slightly larger window for a stable signal.
    early_stop_cb = EarlyStoppingCallback(
        success_log_path=success_log,
        target_success_rate=0.85,
        window=30,
        patience=2,
    )

    warmstart = [checkpoints_dir / cfg.get("checkpoint_name", "nav_sparse_run1")] \
        if args.resume else []
    model = load_model_with_warmstart(warmstart, env, cfg, str(logs_dir))

    model.learn(
        total_timesteps=cfg.get("total_timesteps", 200_000),
        callback=CallbackList([success_cb, checkpoint_cb, early_stop_cb]),
        reset_num_timesteps=not args.resume,
    )

    ckpt = cfg.get("checkpoint_name", "nav_sparse_run1")
    model.save(str(checkpoints_dir / ckpt))
    env.close()

    print("Sparse-reward navigation training complete.")
    print(f"Checkpoint : python_rl/checkpoints/{ckpt}")
    print(f"Monitor CSV: python_rl/logs/{cfg.get('monitor_log', 'nav_sparse_monitor.csv')}")
    print(f"Success CSV: python_rl/logs/{cfg.get('success_log', 'nav_sparse_success.csv')}")


if __name__ == "__main__":
    main()
