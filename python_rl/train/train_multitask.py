from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from python_rl.env.minecraft_env import MinecraftEnv


def main():
    logs_dir = Path("python_rl/logs")
    checkpoints_dir = Path("python_rl/checkpoints")
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Bias farming a bit more because navigation is much easier.
    env = MinecraftEnv(sample_tasks=["farming", "farming", "navigation"])
    env = Monitor(env, filename=str(logs_dir / "multitask_day2_playernpc_monitor.csv"))

    warm_start_candidates = [
        checkpoints_dir / "farm_day2_playernpc_run1",
        checkpoints_dir / "nav_day1_run1",
    ]

    model = None
    for model_path in warm_start_candidates:
        if model_path.exists():
            model = PPO.load(str(model_path), env=env)
            model.tensorboard_log = str(logs_dir / "tb")
            model.verbose = 1
            print(f"Warm-starting from {model_path}")
            break

    if model is None:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            n_steps=512,
            batch_size=128,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.02,
            clip_range=0.2,
            tensorboard_log=str(logs_dir / "tb"),
        )

    model.learn(total_timesteps=40000, reset_num_timesteps=False)
    model.save(str(checkpoints_dir / "multitask_day2_playernpc_run1"))
    env.close()

    print("Training complete. Model saved to python_rl/checkpoints/multitask_day2_playernpc_run1")


if __name__ == "__main__":
    main()