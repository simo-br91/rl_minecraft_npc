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

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import VecNormalize

from python_rl.env.minecraft_env import MinecraftEnv
from python_rl.train.curriculum_scheduler import NavCurriculumScheduler
from python_rl.train.train_utils import (
    SuccessLogger,
    EarlyStoppingCallback,
    make_periodic_checkpoint,
    load_config,
    load_model_with_warmstart,
    wrap_env,
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
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="nav_curriculum")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logs_dir        = Path("python_rl/logs")
    checkpoints_dir = Path("python_rl/checkpoints")
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)

    scheduler = NavCurriculumScheduler(
        advance_threshold=cfg.get("advance_threshold", 0.70),
        regress_threshold=cfg.get("regress_threshold", 0.40),
        advance_window=cfg.get("advance_window", 20),
        start_level=cfg.get("curriculum_start_level", 1),
        log_path=str(logs_dir / "nav_curriculum_levels.csv"),
    )

    base_env = MinecraftEnv(task="navigation")
    curr_env = CurriculumEnv(base_env, scheduler)

    # wrap_env applies Monitor → DummyVecEnv → VecNormalize (obs + reward),
    # consistent with all other training scripts.
    env, vec_normalize = wrap_env(
        curr_env,
        monitor_path=str(logs_dir / "nav_curriculum_monitor.csv"),
        normalize=True,
        norm_reward=True,
    )

    # If resuming, restore saved normalisation stats so running mean/std
    # is not reset from scratch.
    vnorm_path = checkpoints_dir / "nav_curriculum_run1_vecnorm.pkl"
    if args.resume and vnorm_path.exists() and vec_normalize is not None:
        env = VecNormalize.load(str(vnorm_path), env)
        print(f"[train_nav_curriculum] Restored VecNormalize stats from {vnorm_path}")

    success_log  = str(logs_dir / "nav_curriculum_success.csv")
    success_cb   = SuccessLogger(success_log)
    # Level logging is handled by NavCurriculumScheduler directly
    # (writes episode,timestep,level,success,rolling_success_rate to nav_curriculum_levels.csv).
    checkpoint_cb = make_periodic_checkpoint(
        str(checkpoints_dir / "nav_curriculum_checkpoints"),
        prefix="nav_curriculum",
    )
    early_stop_cb = EarlyStoppingCallback(
        success_log_path=success_log,
        target_success_rate=0.95,
        window=30,
        patience=1,
        task_name="Nav Curriculum",
    )

    model = load_model_with_warmstart([], env, cfg, str(logs_dir))

    model.learn(
        total_timesteps=cfg.get("total_timesteps", 250_000),
        callback=CallbackList([success_cb, checkpoint_cb, early_stop_cb]),
        reset_num_timesteps=not args.resume,
    )
    model.save(str(checkpoints_dir / "nav_curriculum_run1"))

    # Save VecNormalize running stats alongside the model checkpoint.
    if vec_normalize is not None:
        vec_normalize.save(str(vnorm_path))
        print(f"[train_nav_curriculum] VecNormalize stats saved to {vnorm_path}")

    env.close()

    print("Curriculum training complete.")
    print(f"Final scheduler state: {scheduler}")
    print("Checkpoint   :", checkpoints_dir / "nav_curriculum_run1")
    print("VecNorm      :", vnorm_path)
    print("Monitor CSV  :", logs_dir / "nav_curriculum_monitor.csv")
    print("Success CSV  :", logs_dir / "nav_curriculum_success.csv")
    print("Level log CSV:", logs_dir / "nav_curriculum_levels.csv")


if __name__ == "__main__":
    main()
