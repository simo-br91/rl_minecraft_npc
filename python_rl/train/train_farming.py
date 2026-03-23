"""
train_farming.py
----------------
Trains single-task farming (harvest a mature wheat crop).
Produces:
  python_rl/logs/farm_monitor.csv
  python_rl/logs/farm_success.csv
  python_rl/checkpoints/farm_run1
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
            f.write("timestep,success\n")

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                success = int(info.get("success", False))
                with self._log_path.open("a") as f:
                    f.write(f"{self.num_timesteps},{success}\n")
        return True


def main() -> None:
    logs_dir        = Path("python_rl/logs")
    checkpoints_dir = Path("python_rl/checkpoints")
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    env = MinecraftEnv(task="farming")
    env = Monitor(env, filename=str(logs_dir / "farm_monitor.csv"))

    success_cb = SuccessLogger(str(logs_dir / "farm_success.csv"))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        learning_rate=2e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.04,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=str(logs_dir / "tb"),
    )

    model.learn(total_timesteps=300_000, callback=success_cb)
    model.save(str(checkpoints_dir / "farm_run1"))
    env.close()

    print("Farming training complete.")
    print("Checkpoint : python_rl/checkpoints/farm_run1")
    print("Monitor CSV: python_rl/logs/farm_monitor.csv")
    print("Success CSV: python_rl/logs/farm_success.csv")


if __name__ == "__main__":
    main()
