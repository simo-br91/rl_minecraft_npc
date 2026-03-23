"""
train_utils.py
--------------
Shared utilities for training scripts:
 - SuccessLogger callback
 - CheckpointCallback with periodic saves
 - TensorBoard custom metrics
 - Config loading from YAML
 - VecEnv / VecNormalize setup
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------

class SuccessLogger(BaseCallback):
    """Appends (timestep, task, success, health, crops_harvested) to a CSV."""

    def __init__(self, log_path: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("w", newline="") as f:
            csv.writer(f).writerow(
                ["timestep", "task", "success", "health", "food_level",
                 "crops_harvested", "mobs_killed", "episode_steps"])

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                with self._log_path.open("a", newline="") as f:
                    csv.writer(f).writerow([
                        self.num_timesteps,
                        info.get("task_name", "?"),
                        int(info.get("success", False)),
                        info.get("health", 20),
                        info.get("food_level", 20),
                        info.get("crops_harvested", 0),
                        info.get("mobs_killed", 0),
                        info.get("episode_step", 0),
                    ])
        return True

    def _on_rollout_end(self) -> None:
        # Push custom metrics to TensorBoard
        if hasattr(self.model, "logger"):
            try:
                # Compute rolling success rate over last 50 episodes
                df = _load_last_n(self._log_path, 50)
                if df:
                    sr = sum(row["success"] for row in df) / len(df)
                    self.model.logger.record("custom/rolling_success_rate", sr)
                    avg_health = sum(row["health"] for row in df) / len(df)
                    self.model.logger.record("custom/avg_health", avg_health)
                    avg_crops = sum(row["crops_harvested"] for row in df) / len(df)
                    self.model.logger.record("custom/avg_crops_harvested", avg_crops)
            except Exception:
                pass


def _load_last_n(path: Path, n: int) -> list[dict]:
    """Load last n rows from a CSV as list of dicts."""
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "success": int(row.get("success", 0)),
                "health": float(row.get("health", 20)),
                "crops_harvested": int(row.get("crops_harvested", 0)),
            })
    return rows[-n:]


def make_periodic_checkpoint(checkpoint_dir: str, save_freq: int = 25_000,
                              prefix: str = "checkpoint") -> CheckpointCallback:
    return CheckpointCallback(
        save_freq=save_freq,
        save_path=str(checkpoint_dir),
        name_prefix=prefix,
        verbose=1,
    )


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------

def load_config(config_name: str) -> Dict[str, Any]:
    """Load a YAML config from python_rl/configs/ by name (with or without .yaml)."""
    path = Path("python_rl/configs")
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"
    full = path / config_name
    if not full.exists():
        raise FileNotFoundError(f"Config not found: {full}")
    with full.open() as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# VecEnv helpers
# ------------------------------------------------------------------

def wrap_env(env, monitor_path: str, normalize: bool = True):
    """
    Wrap env in DummyVecEnv → Monitor → optional VecNormalize.
    Returns (vec_env, vec_normalize_or_None).
    """
    monitored = Monitor(env, filename=monitor_path)
    vec = DummyVecEnv([lambda: monitored])
    if normalize:
        vnorm = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)
        return vnorm, vnorm
    return vec, None


def load_model_with_warmstart(
    candidates: list[str | Path],
    env,
    config: Dict[str, Any],
    logs_dir: str,
) -> PPO:
    """
    Try to warm-start from the first existing checkpoint.
    If none found, create a new PPO from config.
    """
    for path in candidates:
        p = Path(path)
        if p.with_suffix(".zip").exists():
            print(f"[train_utils] Warm-starting from {p}")
            model = PPO.load(str(p), env=env)
            # Override hyperparameters from config
            model.learning_rate  = config.get("learning_rate", 3e-4)
            model.ent_coef       = config.get("ent_coef", 0.05)
            model.clip_range     = config.get("clip_range", 0.2)
            model.tensorboard_log = str(Path(logs_dir) / "tb")
            model.verbose        = 1
            return model

    print("[train_utils] No warmstart checkpoint found — training from scratch.")
    return PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=config.get("n_steps", 2048),
        batch_size=config.get("batch_size", 256),
        n_epochs=config.get("n_epochs", 10),
        learning_rate=config.get("learning_rate", 3e-4),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        ent_coef=config.get("ent_coef", 0.05),
        clip_range=config.get("clip_range", 0.2),
        policy_kwargs=dict(net_arch=config.get("net_arch", [256, 256])),
        tensorboard_log=str(Path(logs_dir) / "tb"),
    )