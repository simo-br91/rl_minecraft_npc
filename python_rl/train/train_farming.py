from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from python_rl.env.minecraft_env import MinecraftEnv


def main():
    logs_dir        = Path("python_rl/logs")
    checkpoints_dir = Path("python_rl/checkpoints")
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    env = MinecraftEnv(task="farming")
    env = Monitor(env, filename=str(logs_dir / "farm_monitor.csv"))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        # ------ rollout ------
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        # ------ learning ------
        learning_rate=2e-4,    # slightly smaller LR: farming reward is spiky
        gamma=0.99,
        gae_lambda=0.95,
        # ------ exploration ------
        ent_coef=0.04,
        clip_range=0.2,
        # ------ network ------
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=str(logs_dir / "tb"),
    )

    # 300k steps: farming is harder (reach + position + interact sequence)
    model.learn(total_timesteps=300_000)
    model.save(str(checkpoints_dir / "farm_run1"))

    env.close()
    print("Farming training complete. Checkpoint: python_rl/checkpoints/farm_run1")


if __name__ == "__main__":
    main()
