"""
train_combat.py
---------------
Trains the agent to fight zombies and skeletons.

The agent:
 - Starts with an iron sword (slot 0), food (slot 1), iron armor
 - Must kill all spawned mobs to succeed
 - Can die and episode resets (agent respawns next episode)
 - Uses switch_item, attack, eat actions

Optional curriculum (--curriculum flag):
  Level 1: 1 mob,  close range  (2–4 b)   — learn to fight at all
  Level 2: 2 mobs, medium range (4–7 b)   — learn to navigate and fight
  Level 3: 3 mobs, far range    (6–10 b)  — full episode difficulty
  Advance when rolling success rate ≥ 0.70 over 20 episodes.
  Regress  when rolling success rate ≤ 0.40 over 20 episodes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3.common.callbacks import CallbackList, BaseCallback

from python_rl.env.minecraft_env import MinecraftEnv
from python_rl.train.train_utils import (
    SuccessLogger, EarlyStoppingCallback, make_periodic_checkpoint,
    load_config, wrap_env, load_model_with_warmstart
)
from python_rl.train.curriculum_scheduler import CombatCurriculumScheduler


class CombatCurriculumCallback(BaseCallback):
    """
    Per-episode callback that records success into the combat curriculum
    scheduler and updates the next reset options accordingly.
    """

    def __init__(self, env: MinecraftEnv, scheduler: CombatCurriculumScheduler,
                 verbose: int = 0) -> None:
        super().__init__(verbose)
        self._env       = env
        self._scheduler = scheduler

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            # SB3 wraps episode end info under "episode" key
            if "episode" in info or info.get("done", False):
                success = bool(info.get("success", False))
                self._scheduler.record_episode(
                    success, timestep=self.num_timesteps)
                # Inject updated options for next episode
                opts = self._scheduler.get_reset_options(task="combat")
                self._env._curriculum_options = opts
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="combat")
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable 3-level combat difficulty curriculum.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    logs_dir        = Path("python_rl/logs")
    checkpoints_dir = Path("python_rl/checkpoints")
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    base_env = MinecraftEnv(task="combat")
    vec_env, _ = wrap_env(base_env, str(logs_dir / "combat_monitor.csv"),
                          normalize=False)

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

    callbacks = [success_cb, checkpoint_cb, early_stop_cb]

    if args.curriculum:
        scheduler = CombatCurriculumScheduler(
            advance_threshold=0.70,
            regress_threshold=0.40,
            advance_window=20,
            log_path=logs_dir / "combat_curriculum.csv",
        )
        curriculum_cb = CombatCurriculumCallback(base_env, scheduler)
        callbacks.append(curriculum_cb)
        print(f"[train_combat] Curriculum enabled — starting at "
              f"level {scheduler.level_number}: "
              f"{scheduler.current_level['description']}")

    warmstart = [checkpoints_dir / "combat_run1"] if args.resume else []
    model = load_model_with_warmstart(warmstart, vec_env, cfg, str(logs_dir))

    model.learn(
        total_timesteps=cfg.get("total_timesteps", 300_000),
        callback=CallbackList(callbacks),
        reset_num_timesteps=not args.resume,
    )
    model.save(str(checkpoints_dir / "combat_run1"))
    vec_env.close()
    print("Combat training complete.")
    print("Checkpoint : python_rl/checkpoints/combat_run1")


if __name__ == "__main__":
    main()
