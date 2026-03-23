"""
train_multitask.py
------------------
Trains a shared PPO policy across navigation + farming.
Farming is sampled 2× more often because it is harder.
Produces:
  python_rl/logs/multitask_monitor.csv
  python_rl/logs/multitask_success.csv
  python_rl/checkpoints/multitask_run1
"""

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from python_rl.env.minecraft_env import MinecraftEnv


class SuccessLogger(BaseCallback):
    def __init__(self, log_path: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("w") as f:
            f.write("timestep,task,success\n")

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                success  = int(info.get("success", False))
                task     = info.get("task_name", "unknown")
                with self._log_path.open("a") as f:
                    f.write(f"{self.num_timesteps},{task},{success}\n")
        return True


def main() -> None:
    logs_dir        = Path("python_rl/logs")
    checkpoints_dir = Path("python_rl/checkpoints")
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Bias farming 2:1 — it requires more practice
    env = MinecraftEnv(sample_tasks=["farming", "farming", "navigation"])
    env = Monitor(env, filename=str(logs_dir / "multitask_monitor.csv"))

    success_cb = SuccessLogger(str(logs_dir / "multitask_success.csv"))

    # Warm-start from the best available single-task checkpoint
    warm_start_candidates = [
        checkpoints_dir / "farm_run1",
        checkpoints_dir / "nav_shaped_run1",
        checkpoints_dir / "nav_curriculum_run1",
    ]
    model = None
    for path in warm_start_candidates:
        if path.with_suffix(".zip").exists():
            model = PPO.load(str(path), env=env)
            model.tensorboard_log = str(logs_dir / "tb")
            model.verbose = 1
            print(f"Warm-starting from {path}")
            break

    if model is None:
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
            ent_coef=0.03,
            clip_range=0.2,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=str(logs_dir / "tb"),
        )

    model.learn(total_timesteps=200_000, reset_num_timesteps=False,
                callback=success_cb)
    model.save(str(checkpoints_dir / "multitask_run1"))
    env.close()

    print("Multi-task training complete.")
    print("Checkpoint : python_rl/checkpoints/multitask_run1")
    print("Monitor CSV: python_rl/logs/multitask_monitor.csv")
    print("Success CSV: python_rl/logs/multitask_success.csv")


if __name__ == "__main__":
    main()
