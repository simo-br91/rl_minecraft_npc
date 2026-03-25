"""
train_utils.py
--------------
Shared utilities for all training scripts.

New in this version
-------------------
* SuccessLogger._on_rollout_end now pushes rich custom metrics to TensorBoard:
  rolling_success_rate, avg_health, avg_crops_harvested, avg_mobs_killed,
  curriculum_level (if available). (Issue 6.6)
* make_multitask_env() provides dynamic task rebalancing: after every
  REBALANCE_WINDOW episodes, if one task's rolling SR significantly exceeds
  the others, its sampling probability is reduced. (Issue 6.8)
* run_multi_seed() helper trains the same config with N seeds and collects
  aggregate results for variance analysis. (Issue 6.9)
* wrap_env() defaults to VecNormalize (obs+reward normalisation). (Issues 5.1, 4.2)
* EarlyStoppingCallback window is now configurable (was hardcoded 50). (Issue 8.4)
* Full type annotations throughout. (Issue 8.1)
* make_maskable_model() creates a MaskablePPO from config. (Issue 6.3)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
# Task rebalancing constants (Issue 6.8)
# ------------------------------------------------------------------
REBALANCE_WINDOW = 50   # episodes per task before rebalancing


# ------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------

class SuccessLogger(BaseCallback):
    """
    Appends one row per completed episode to a CSV and pushes custom
    metrics to TensorBoard at the end of each rollout. (Issues 6.6, 8.1)

    CSV columns:
      timestep, task, success, health, food_level,
      crops_harvested, mobs_killed, episode_steps
    """

    def __init__(
        self,
        log_path:       str,
        curriculum_env: Any = None,   # optional: CurriculumEnv to log level
        verbose:        int = 0,
    ) -> None:
        super().__init__(verbose)
        self._log_path:       Path = Path(log_path)
        self._curriculum_env: Any  = curriculum_env
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
        """Push custom metrics to TensorBoard. (Issue 6.6)"""
        if not hasattr(self.model, "logger"):
            return
        try:
            rows = _load_last_n(self._log_path, 50)
            if not rows:
                return
            sr         = sum(r["success"]          for r in rows) / len(rows)
            avg_health = sum(r["health"]            for r in rows) / len(rows)
            avg_crops  = sum(r["crops_harvested"]   for r in rows) / len(rows)
            avg_mobs   = sum(r["mobs_killed"]       for r in rows) / len(rows)
            avg_steps  = sum(r["episode_steps"]     for r in rows) / len(rows)
            self.model.logger.record("custom/rolling_success_rate", sr)
            self.model.logger.record("custom/avg_health",           avg_health)
            self.model.logger.record("custom/avg_crops_harvested",  avg_crops)
            self.model.logger.record("custom/avg_mobs_killed",      avg_mobs)
            self.model.logger.record("custom/avg_episode_steps",    avg_steps)
            # Curriculum level (if available)
            if self._curriculum_env is not None:
                lv = getattr(
                    getattr(self._curriculum_env, "scheduler", None),
                    "level_number", None,
                )
                if lv is not None:
                    self.model.logger.record("custom/curriculum_level", lv)
        except Exception:
            pass


def _load_last_n(path: Path, n: int) -> List[Dict[str, Any]]:
    """Load the last n data rows from a SuccessLogger CSV."""
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "success":         int(row.get("success", 0)),
                "health":          float(row.get("health", 20)),
                "crops_harvested": int(row.get("crops_harvested", 0)),
                "mobs_killed":     int(row.get("mobs_killed", 0)),
                "episode_steps":   int(row.get("episode_steps", 0)),
            })
    return rows[-n:]


class EarlyStoppingCallback(BaseCallback):
    """
    Stops training when the rolling success rate has exceeded
    ``target_success_rate`` for ``patience`` consecutive rollout ends.

    Parameters
    ----------
    success_log_path : str
        Path to the SuccessLogger CSV.
    target_success_rate : float
        Success rate threshold (default 0.90).
    window : int
        Rolling window size in episodes (default 50).
        Previously hardcoded at 50; now configurable. (Issue 8.4)
    patience : int
        Consecutive checks above target before stopping (default 3).
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
        self._log_path:    Path  = Path(success_log_path)
        self._target:      float = target_success_rate
        self._window:      int   = window
        self._patience:    int   = patience
        self._above_count: int   = 0
        self._should_stop: bool  = False  # FIX: flag read by _on_step to actually halt SB3

    def _on_step(self) -> bool:
        # SB3 checks _on_step() return value to stop training.
        # _on_rollout_end() return value is ignored, so we use this flag instead.
        return not self._should_stop

    def _on_rollout_end(self) -> None:
        rows = _load_last_n(self._log_path, self._window)
        if len(rows) < self._window:
            return

        sr = sum(r["success"] for r in rows) / len(rows)
        if sr >= self._target:
            self._above_count += 1
            if self._above_count >= self._patience:
                print(
                    f"[EarlyStopping] Rolling SR={sr:.2f} ≥ {self._target:.2f} "
                    f"for {self._above_count} consecutive checks — stopping."
                )
                self._should_stop = True
        else:
            self._above_count = 0


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
    """Load a YAML config from python_rl/configs/ by name."""
    path = Path("python_rl/configs")
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"
    full = path / config_name
    if not full.exists():
        raise FileNotFoundError(f"Config not found: {full}")
    with full.open() as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Env wrapping — Issues 4.2, 5.1
# ------------------------------------------------------------------

def wrap_env(
    env:            Any,
    monitor_path:   str,
    normalize:      bool  = True,
    norm_reward:    bool  = True,
    clip_obs:       float = 10.0,
) -> Tuple[Any, Optional[VecNormalize]]:
    """
    Wrap env: Monitor → DummyVecEnv → VecNormalize (default on).

    VecNormalize handles observation normalisation (Issues 4.2) and reward
    normalisation (Issue 5.1).  Save the VecNormalize stats alongside the
    model checkpoint so they can be restored at evaluation time.

    Returns
    -------
    (wrapped_env, vec_normalize_or_None)
    """
    monitored = Monitor(env, filename=monitor_path)
    vec = DummyVecEnv([lambda: monitored])   # type: ignore[list-item]
    if normalize:
        vnorm = VecNormalize(
            vec,
            norm_obs=True,
            norm_reward=norm_reward,
            clip_obs=clip_obs,
        )
        return vnorm, vnorm
    return vec, None


# ------------------------------------------------------------------
# Dynamic task rebalancing (Issue 6.8)
# ------------------------------------------------------------------

class TaskRebalancer:
    """
    Tracks per-task rolling success rates and adjusts sampling probabilities
    so that harder tasks are sampled more often.

    Intended for use with MinecraftEnv(sample_tasks=[...]).

    Usage
    -----
    rebalancer = TaskRebalancer(["navigation", "farming", "combat"])
    # ... after each episode:
    rebalancer.record(task_name, success)
    # ... at reset time:
    next_task = rebalancer.sample()
    """

    def __init__(
        self,
        tasks:  List[str],
        window: int   = REBALANCE_WINDOW,
        min_p:  float = 0.10,
    ) -> None:
        self.tasks:   List[str]           = tasks
        self.window:  int                 = window
        self.min_p:   float               = min_p
        self._recent: Dict[str, List[int]] = {t: [] for t in tasks}
        self._probs:  np.ndarray           = np.ones(len(tasks)) / len(tasks)

    def record(self, task: str, success: bool) -> None:
        if task not in self._recent:
            return
        buf = self._recent[task]
        buf.append(int(success))
        if len(buf) > self.window:
            buf.pop(0)
        self._rebalance()

    def _rebalance(self) -> None:
        """Inverse-SR weighting: harder tasks get more sampling weight."""
        rates = np.array([
            (1.0 - (sum(self._recent[t]) / len(self._recent[t]))
             if self._recent[t] else 0.5)
            for t in self.tasks
        ])
        rates = np.clip(rates, self.min_p, 1.0 - self.min_p)
        self._probs = rates / rates.sum()

    def sample(self, rng: Optional[np.random.Generator] = None) -> str:
        """Sample a task according to current difficulty-weighted probabilities."""
        if rng is None:
            rng = np.random.default_rng()
        idx = int(rng.choice(len(self.tasks), p=self._probs))
        return self.tasks[idx]

    @property
    def probabilities(self) -> Dict[str, float]:
        return {t: float(self._probs[i]) for i, t in enumerate(self.tasks)}


# ------------------------------------------------------------------
# Multi-seed training (Issue 6.9)
# ------------------------------------------------------------------

def run_multi_seed(
    train_fn:    Callable[[int, str], Dict[str, Any]],
    seeds:       List[int],
    experiment:  str,
    logs_dir:    str = "python_rl/logs",
) -> Dict[str, Any]:
    """
    Run the same training function with multiple seeds and aggregate results.

    Parameters
    ----------
    train_fn : Callable[[seed, run_name], dict]
        Function that trains one run and returns a stats dict with at least
        {"final_success_rate": float, "total_timesteps": int}.
    seeds : list[int]
        Random seeds to use.
    experiment : str
        Base name for logging (e.g. "nav_shaped").
    logs_dir : str
        Directory to write the aggregated results CSV.

    Returns
    -------
    dict with keys: mean_sr, std_sr, runs (list of per-seed results)
    """
    results: List[Dict[str, Any]] = []
    for seed in seeds:
        run_name = f"{experiment}_seed{seed}"
        print(f"\n{'='*60}")
        print(f"[multi_seed] Seed {seed} — {run_name}")
        print(f"{'='*60}")
        stats = train_fn(seed, run_name)
        stats["seed"]     = seed
        stats["run_name"] = run_name
        results.append(stats)

    srs    = [r.get("final_success_rate", float("nan")) for r in results]
    valid  = [s for s in srs if not np.isnan(s)]
    mean   = float(np.mean(valid))  if valid else float("nan")
    std    = float(np.std(valid))   if valid else float("nan")

    # Write aggregate CSV
    out_dir = Path(logs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{experiment}_multiseed.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "run_name", "final_success_rate"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "seed":                r["seed"],
                "run_name":            r["run_name"],
                "final_success_rate":  r.get("final_success_rate", float("nan")),
            })

    print(f"\n[multi_seed] {experiment}: mean SR={mean:.3f} ± {std:.3f} over {len(valid)} seeds")
    print(f"[multi_seed] Results written to {csv_path}")

    return {"mean_sr": mean, "std_sr": std, "runs": results}


# ------------------------------------------------------------------
# Warm-start / model creation
# ------------------------------------------------------------------

def load_model_with_warmstart(
    candidates: List[str | Path],
    env:        Any,
    config:     Dict[str, Any],
    logs_dir:   str,
) -> PPO:
    """
    Try to warm-start from the first existing checkpoint.
    If none found, create a fresh PPO from config.
    Key hyperparameters from config are always applied to the loaded model.
    """
    for path in candidates:
        p = Path(path)
        if p.with_suffix(".zip").exists():
            print(f"[train_utils] Warm-starting from {p}")
            model = PPO.load(str(p), env=env)
            model.learning_rate   = config.get("learning_rate", model.learning_rate)
            model.ent_coef        = config.get("ent_coef",      model.ent_coef)
            model.clip_range      = config.get("clip_range",    model.clip_range)
            model.gamma           = config.get("gamma",         model.gamma)
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


# ------------------------------------------------------------------
# MaskablePPO factory (Issue 6.3)
# ------------------------------------------------------------------

def make_maskable_model(
    env:      Any,
    config:   Dict[str, Any],
    logs_dir: str,
) -> Any:
    """
    Create a MaskablePPO model from config.

    Requires sb3-contrib (``pip install sb3-contrib``).
    Falls back to plain PPO with a warning if sb3-contrib is not installed.

    Parameters
    ----------
    env : gymnasium Env (should be a MaskableMinecraftEnv or subclass)
    config : dict loaded from a YAML experiment config
    logs_dir : str path to TensorBoard log directory

    Returns
    -------
    MaskablePPO or PPO instance.
    """
    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        print(
            "[train_utils] WARNING: sb3-contrib not installed. "
            "Falling back to plain PPO without action masking.\n"
            "  Install with: pip install sb3-contrib"
        )
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

    return MaskablePPO(
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
