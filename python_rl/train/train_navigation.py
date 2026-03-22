from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from python_rl.env.minecraft_env import MinecraftEnv


def main():
    logs_dir        = Path("python_rl/logs")
    checkpoints_dir = Path("python_rl/checkpoints")
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    env = MinecraftEnv(task="navigation")
    env = Monitor(env, filename=str(logs_dir / "nav_monitor.csv"))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        # ------ rollout ------
        n_steps=2048,          # longer rollouts → better credit assignment
        batch_size=256,
        n_epochs=10,
        # ------ learning ------
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        # ------ exploration ------
        ent_coef=0.05,         # higher entropy → agent explores jump more freely
        clip_range=0.2,
        # ------ network ------
        policy_kwargs=dict(net_arch=[256, 256]),   # bigger net for richer obs
        tensorboard_log=str(logs_dir / "tb"),
    )

    # 200k steps is a reasonable budget for navigation + obstacles
    model.learn(total_timesteps=200_000)
    model.save(str(checkpoints_dir / "nav_run1"))

    env.close()
    print("Navigation training complete. Checkpoint: python_rl/checkpoints/nav_run1")


if __name__ == "__main__":
    main()
