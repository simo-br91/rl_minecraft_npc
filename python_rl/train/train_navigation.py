from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from python_rl.env.minecraft_env import MinecraftEnv


def main():
    logs_dir = Path("python_rl/logs")
    checkpoints_dir = Path("python_rl/checkpoints")
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    env = MinecraftEnv(task="navigation")
    env = Monitor(env, filename=str(logs_dir / "nav_day1_monitor.csv"))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        tensorboard_log=str(logs_dir / "tb"),
    )

    model.learn(total_timesteps=5000)

    model.save(str(checkpoints_dir / "nav_day1_run1"))
    env.close()

    print("Training complete. Model saved to python_rl/checkpoints/nav_day1_run1")


if __name__ == "__main__":
    main()