"""
train_nav_curriculum.py
-----------------------
Trains navigation with a 4-level curriculum:

  Level 1:  3–6 blocks,    0 obstacles   (flat, short)
  Level 2:  5–9 blocks,    1 obstacle
  Level 3:  7–14 blocks,   2 obstacles
  Level 4:  10–18 blocks,  3 obstacles

Advancement: rolling success-rate >= 0.70 over the last 20 episodes.

Compare the resulting reward/success curves against nav_shaped (no
curriculum) using compare_experiments.py.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from python_rl.env.minecraft_env import MinecraftEnv
from python_rl.train.curriculum_scheduler import CurriculumScheduler


# ------------------------------------------------------------------
# CurriculumEnv wrapper
# ------------------------------------------------------------------

class CurriculumEnv(gym.Wrapper):
    """
    Wraps MinecraftEnv so that every reset() automatically injects
    the current curriculum level's difficulty parameters.

    It also records episode outcomes in the scheduler so the level
    can advance when the rolling success-rate threshold is met.
    """

    def __init__(self, env: MinecraftEnv, scheduler: CurriculumScheduler) -> None:
        super().__init__(env)
        self.scheduler = scheduler

    def reset(self, seed=None, options=None):
        opts = self.scheduler.get_reset_options(task="navigation")
        if options:
            opts.update(options)   # allow explicit overrides
        return self.env.reset(seed=seed, options=opts)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if done or truncated:
            self.scheduler.record_episode(
                success=bool(info.get("success", False)),
                timestep=self._current_timestep,
            )
        return obs, reward, done, truncated, info

    # SB3 calls this internally; we shadow it so we can read the timestep
    # from the callback instead of maintaining our own counter.
    @property
    def _current_timestep(self) -> int:
        return getattr(self, "_ts", 0)


# ------------------------------------------------------------------
# Callback: expose SB3 timestep to CurriculumEnv + log level to CSV
# ------------------------------------------------------------------

class CurriculumCallback(BaseCallback):
    """
    Feeds the current SB3 timestep back into CurriculumEnv and logs
    per-episode curriculum level + success to a CSV.
    """

    def __init__(
        self,
        curriculum_env: CurriculumEnv,
        success_log_path: str,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._curriculum_env = curriculum_env
        self._log_path = Path(success_log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("w") as f:
            f.write("timestep,level,success\n")

    def _on_step(self) -> bool:
        # Keep CurriculumEnv's timestep in sync
        self._curriculum_env._ts = self.num_timesteps

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

    scheduler    = CurriculumScheduler(
        advance_threshold=0.70,
        advance_window=20,
        start_level=1,
        log_path=str(logs_dir / "nav_curriculum_levels.csv"),
    )
    base_env     = MinecraftEnv(task="navigation")
    curr_env     = CurriculumEnv(base_env, scheduler)
    env          = Monitor(curr_env, filename=str(logs_dir / "nav_curriculum_monitor.csv"))

    callback = CurriculumCallback(
        curriculum_env=curr_env,
        success_log_path=str(logs_dir / "nav_curriculum_success.csv"),
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.05,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=str(logs_dir / "tb"),
    )

    # 250 k steps to give room for the curriculum to progress all 4 levels
    model.learn(total_timesteps=250_000, callback=callback)
    model.save(str(checkpoints_dir / "nav_curriculum_run1"))
    env.close()

    print("Curriculum training complete.")
    print(f"Final scheduler state: {scheduler}")
    print("Checkpoint   : python_rl/checkpoints/nav_curriculum_run1")
    print("Monitor CSV  : python_rl/logs/nav_curriculum_monitor.csv")
    print("Success CSV  : python_rl/logs/nav_curriculum_success.csv")
    print("Level log CSV: python_rl/logs/nav_curriculum_levels.csv")


if __name__ == "__main__":
    main()
