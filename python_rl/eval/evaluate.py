from stable_baselines3 import PPO
from python_rl.env.minecraft_env import MinecraftEnv


def main():
    env = MinecraftEnv()
    model = PPO.load("python_rl/checkpoints/nav_day2_run1")

    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0.0
    step_count = 0

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        print(
            f"step={step_count} action={action} reward={reward:.3f} "
            f"done={done} truncated={truncated} info={info}"
        )

    print("FINAL")
    print("steps:", step_count)
    print("total_reward:", total_reward)
    print("info:", info)


if __name__ == "__main__":
    main()