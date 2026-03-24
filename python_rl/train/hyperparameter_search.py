"""
hyperparameter_search.py
------------------------
Optuna-based hyperparameter search for navigation and farming experiments.

Usage
-----
    pip install optuna
    python -m python_rl.train.hyperparameter_search --task navigation --trials 30
    python -m python_rl.train.hyperparameter_search --task farming   --trials 20

Each trial trains for a short budget (default 50k steps) and reports the
rolling success rate from the last 50 episodes as the objective.

Results are saved to python_rl/logs/optuna/<task>_study.db (SQLite) and
a summary CSV to python_rl/logs/optuna/<task>_best_params.yaml.

Issue 6.5 — hyperparameter search was listed as missing.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import yaml

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from python_rl.env.minecraft_env import MinecraftEnv
from python_rl.train.train_utils import _load_last_n

LOGS_DIR        = Path("python_rl/logs")
OPTUNA_DIR      = LOGS_DIR / "optuna"
CHECKPOINTS_DIR = Path("python_rl/checkpoints")
EVAL_EPISODES   = 50    # rolling window for objective


def _compute_sr(success_log: Path) -> float:
    rows = _load_last_n(success_log, EVAL_EPISODES)
    if not rows:
        return 0.0
    return sum(r["success"] for r in rows) / len(rows)


def _make_env(task: str, monitor_path: str) -> Any:
    base = MinecraftEnv(task=task)
    mon  = Monitor(base, filename=monitor_path)
    vec  = DummyVecEnv([lambda: mon])       # type: ignore[list-item]
    return VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)


def _objective(trial: "optuna.Trial", task: str, timesteps: int) -> float:
    """Single Optuna trial: sample hyperparameters, train, return SR."""
    lr         = trial.suggest_float("learning_rate",  1e-5,  5e-4, log=True)
    ent_coef   = trial.suggest_float("ent_coef",       0.01,  0.15, log=True)
    n_steps    = trial.suggest_categorical("n_steps",  [1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size",[64, 128, 256])
    gamma      = trial.suggest_float("gamma",          0.95,  0.999)
    clip_range = trial.suggest_float("clip_range",     0.1,   0.3)
    net_width  = trial.suggest_categorical("net_width", [128, 256, 512])

    trial_id     = trial.number
    monitor_path = str(OPTUNA_DIR / f"trial_{trial_id}_monitor.csv")
    success_log  = OPTUNA_DIR / f"trial_{trial_id}_success.csv"
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

    env = _make_env(task, monitor_path)

    from python_rl.train.train_utils import SuccessLogger
    success_cb = SuccessLogger(str(success_log))

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        ent_coef=ent_coef,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        clip_range=clip_range,
        policy_kwargs=dict(net_arch=[net_width, net_width]),
        verbose=0,
    )

    try:
        model.learn(total_timesteps=timesteps, callback=success_cb)
    except Exception as e:
        print(f"  Trial {trial_id} failed: {e}")
        env.close()
        return 0.0

    sr = _compute_sr(success_log)
    env.close()
    print(f"  Trial {trial_id}: SR={sr:.3f}  lr={lr:.2e}  ent={ent_coef:.3f}  "
          f"n_steps={n_steps}  batch={batch_size}  γ={gamma:.4f}")
    return sr


def run_search(
    task:      str,
    n_trials:  int,
    timesteps: int,
) -> Dict[str, Any]:
    if not OPTUNA_AVAILABLE:
        print("[hyperparameter_search] optuna not installed. Run: pip install optuna")
        return {}

    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    db_path = OPTUNA_DIR / f"{task}_study.db"
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{task}_search",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    print(f"[hyperparameter_search] Starting {n_trials} trials for '{task}' "
          f"({timesteps} steps each)")
    study.optimize(
        lambda trial: _objective(trial, task, timesteps),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best = study.best_params
    best["final_success_rate"] = study.best_value

    out_yaml = OPTUNA_DIR / f"{task}_best_params.yaml"
    with out_yaml.open("w") as f:
        yaml.dump(best, f, default_flow_style=False)

    print(f"\n[hyperparameter_search] Best params for '{task}':")
    for k, v in best.items():
        print(f"  {k}: {v}")
    print(f"[hyperparameter_search] Results saved to {out_yaml}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search.")
    parser.add_argument("--task",      default="navigation",
                        choices=["navigation", "farming", "combat", "multitask"])
    parser.add_argument("--trials",    type=int, default=30)
    parser.add_argument("--timesteps", type=int, default=50_000,
                        help="Steps per trial (short budget for search).")
    args = parser.parse_args()

    run_search(args.task, args.trials, args.timesteps)


if __name__ == "__main__":
    main()
