"""
train_nav_curriculum.py
-----------------------
Trains navigation with a 4-level curriculum:

  Level 1:  3–6 blocks,  0 obstacles   (flat, short)
  Level 2:  5–9 blocks,  1 obstacle
  Level 3:  7–14 blocks, 2 obstacles   ← default training difficulty
  Level 4: 10–18 blocks, 3 obstacles

Advancement: rolling success rate ≥ 0.70 over the last 20 episodes.
Regression : rolling success rate < 0.40 (agent can go back a level).

Bug fixed vs previous version
------------------------------
* CurriculumEnv._current_timestep no longer relies on a fragile external
  callback setting a ``_ts`` attribute.  The env now maintains its own
  internal step counter, so ``record_episode`` always gets a correct
  timestep even when used outside of training (e.g. evaluation, testing).
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor

from python_rl.env.minecraft_env import MinecraftEnv
from python_rl.train.curriculum_scheduler import NavCurriculumScheduler
from python_rl.train.train_utils import (
    SuccessLogger,
    EarlyStoppingCallback,
    make_periodic_checkpoint,
    load_config,
    load_model_with_warmstart,
)


# ------------------------------------------------------------------
# Curriculum wrapper
# ------------------------------------------------------------------

class CurriculumEnv(gym.Wrapper):
    """
    Wraps MinecraftEnv so that every reset() automatically injects the
    current curriculum level's difficulty parameters.

    Episode outcomes are recorded in the scheduler so the level advances
    when the rolling success-rate threshold is met.

    The internal ``_timestep`` counter is maintained by this class itself,
    so it works correctly with or without an external callback.
    """

    def __init__(self, env: MinecraftEnv, scheduler: NavCurriculumScheduler) -> None:
        super().__init__(env)
        self.scheduler  = scheduler
        self._timestep  = 0          # maintained internally — no callback needed

    def reset(self, seed=None, options=None):
        opts = self.scheduler.get_reset_options(task="navigation")
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
# Logging callback for level tracking
# ------------------------------------------------------------------

class CurriculumLevelLogger:
    """
    Thin wrapper: logs (timestep, level, success) to a CSV after each
    episode.  Implemented as a SB3 BaseCallback.
    """

    from stable_baselines3.common.callbacks import BaseCallback as _BC

    class _Impl(_BC):
        def __init__(self, curriculum_env: "CurriculumEnv",
                     log_path: str, verbose: int = 0) -> None:
            super().__init__(verbose)
            self._curriculum_env = curriculum_env
            self._log_path = Path(log_path)
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("w") as f:
                f.write("timestep,level,success\n")

        def _on_step(self) -> bool:
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done:
                    success = int(info.get("success", False))
                    level   = self._curriculum_env.scheduler.level_number
                    with self._log_path.open("a") as f:
                        f.write(f"{self.num_timesteps},{level},{success}\n")
            return True


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    logs_dir        = Path("python_rl/logs")
    checkpoints_dir = Path("python_rl/checkpoints")
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config("nav_curriculum")

    scheduler = NavCurriculumScheduler(
        advance_threshold=cfg.get("advance_threshold", 0.70),
        regress_threshold=cfg.get("regress_threshold", 0.40),
        advance_window=cfg.get("advance_window", 20),
        start_level=cfg.get("curriculum_start_level", 1),
        log_path=str(logs_dir / "nav_curriculum_levels.csv"),
    )

    base_env = MinecraftEnv(task="navigation")
    curr_env = CurriculumEnv(base_env, scheduler)
    env      = Monitor(curr_env, filename=str(logs_dir / "nav_curriculum_monitor.csv"))

    success_log  = str(logs_dir / "nav_curriculum_success.csv")
    success_cb   = SuccessLogger(success_log)
    level_cb     = CurriculumLevelLogger._Impl(
        curr_env, str(logs_dir / "nav_curriculum_success.csv"))
    checkpoint_cb = make_periodic_checkpoint(
        str(checkpoints_dir / "nav_curriculum_checkpoints"),
        prefix="nav_curriculum",
    )
    early_stop_cb = EarlyStoppingCallback(
        success_log_path=success_log,
        target_success_rate=0.95,
        window=50,
        patience=3,
    )

    model = load_model_with_warmstart([], env, cfg, str(logs_dir))

    model.learn(
        total_timesteps=cfg.get("total_timesteps", 250_000),
        callback=CallbackList([success_cb, level_cb, checkpoint_cb, early_stop_cb]),
    )
    model.save(str(checkpoints_dir / "nav_curriculum_run1"))
    env.close()

    print("Curriculum training complete.")
    print(f"Final scheduler state: {scheduler}")
    print("Checkpoint   :", checkpoints_dir / "nav_curriculum_run1")
    print("Monitor CSV  :", logs_dir / "nav_curriculum_monitor.csv")
    print("Success CSV  :", logs_dir / "nav_curriculum_success.csv")
    print("Level log CSV:", logs_dir / "nav_curriculum_levels.csv")


if __name__ == "__main__":
    main()