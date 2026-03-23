"""
train_utils.py
--------------
Shared utilities for all training scripts:
  - SuccessLogger callback (extended fields)
  - EarlyStoppingCallback
  - CheckpointCallback factory
  - Config loading from YAML
  - Env wrapping (Monitor + optional VecNormalize)
  - Warm-start model loading with hyperparameter override
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------

class SuccessLogger(BaseCallback):
    """
    Appends one row per completed episode to a CSV:
      timestep, task, success, health, food_level,
      crops_harvested, mobs_killed, episode_steps
    """

    def __init__(self, log_path: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("w", newline="") as f:
            csv.writer(f).writerow([
                "timestep", "task", "success", "health", "food_level",
                "crops_harvested", "mobs_killed", "episode_steps",
            ])

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
        """Push custom metrics to TensorBoard at the end of each rollout."""
        if not hasattr(self.model, "logger"):
            return
        try:
            rows = _load_last_n(self._log_path, 50)
            if rows:
                sr        = sum(r["success"]          for r in rows) / len(rows)
                avg_health= sum(r["health"]            for r in rows) / len(rows)
                avg_crops = sum(r["crops_harvested"]   for r in rows) / len(rows)
                avg_mobs  = sum(r["mobs_killed"]       for r in rows) / len(rows)
                self.model.logger.record("custom/rolling_success_rate", sr)
                self.model.logger.record("custom/avg_health",           avg_health)
                self.model.logger.record("custom/avg_crops_harvested",  avg_crops)
                self.model.logger.record("custom/avg_mobs_killed",      avg_mobs)
        except Exception:
            pass


def _load_last_n(path: Path, n: int) -> List[dict]:
    """Load the last n data rows from a SuccessLogger CSV."""
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "success":         int(row.get("success", 0)),
                "health":          float(row.get("health", 20)),
                "crops_harvested": int(row.get("crops_harvested", 0)),
                "mobs_killed":     int(row.get("mobs_killed", 0)),
            })
    return rows[-n:]


class EarlyStoppingCallback(BaseCallback):
    """
    Stops training when the rolling success rate has exceeded
    ``target_success_rate`` for ``patience`` consecutive rollout ends.

    Parameters
    ----------
    success_log_path : str
        Path to the SuccessLogger CSV (must share the same log file).
    target_success_rate : float
        Success rate threshold (default 0.90).
    window : int
        Rolling window size in episodes (default 50).
    patience : int
        How many consecutive rollout-end checks must all be above target
        before training is stopped (default 3).
    """

    def __init__(
        self,
        success_log_path:    str,
        target_success_rate: float = 0.90,
        window:              int   = 50,
        patience:            int   = 3,
        verbose:             int   = 0,
    ) -> None:
        super().__init__(verbose)
        self._log_path   = Path(success_log_path)
        self._target     = target_success_rate
        self._window     = window
        self._patience   = patience
        self._above_count = 0

    def _on_step(self) -> bool:
        return True   # checked in _on_rollout_end

    def _on_rollout_end(self) -> bool:
        rows = _load_last_n(self._log_path, self._window)
        if len(rows) < self._window:
            return True   # not enough data yet

        sr = sum(r["success"] for r in rows) / len(rows)
        if sr >= self._target:
            self._above_count += 1
            if self._above_count >= self._patience:
                print(
                    f"[EarlyStopping] Rolling SR={sr:.2f} ≥ {self._target:.2f} "
                    f"for {self._patience} checks in a row — stopping."
                )
                return False   # signal SB3 to stop training
        else:
            self._above_count = 0   # reset streak if rate drops

        return True


def make_periodic_checkpoint(
    checkpoint_dir: str,
    save_freq:      int = 25_000,
    prefix:         str = "checkpoint",
) -> CheckpointCallback:
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
# Env wrapping
# ------------------------------------------------------------------

def wrap_env(env, monitor_path: str, normalize: bool = False):
    """
    Wrap env in Monitor → DummyVecEnv → optional VecNormalize.

    Returns (wrapped_env, vec_normalize_or_None).
    normalize=True is recommended for observations with heterogeneous scale.
    """
    monitored = Monitor(env, filename=monitor_path)
    vec = DummyVecEnv([lambda: monitored])   # type: ignore[list-item]
    if normalize:
        vnorm = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)
        return vnorm, vnorm
    return vec, None


# ------------------------------------------------------------------
# Warm-start / model creation
# ------------------------------------------------------------------

def load_model_with_warmstart(
    candidates:  List[str | Path],
    env:         Any,
    config:      Dict[str, Any],
    logs_dir:    str,
) -> PPO:
    """
    Try to warm-start from the first existing checkpoint.
    If none found, create a fresh PPO from config.

    When warm-starting, key hyperparameters from config are applied
    to the loaded model so they are not silently ignored.
    """
    for path in candidates:
        p = Path(path)
        if p.with_suffix(".zip").exists():
            print(f"[train_utils] Warm-starting from {p}")
            model = PPO.load(str(p), env=env)
            # Apply config hyperparameters — PPO.load preserves old values otherwise
            model.learning_rate  = config.get("learning_rate", model.learning_rate)
            model.ent_coef       = config.get("ent_coef",      model.ent_coef)
            model.clip_range     = config.get("clip_range",    model.clip_range)
            model.gamma          = config.get("gamma",         model.gamma)
            model.tensorboard_log = str(Path(logs_dir) / "tb")
            model.verbose = 1
            return model

    print("[train_utils] No warm-start checkpoint found — training from scratch.")
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