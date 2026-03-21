from collections import defaultdict

from stable_baselines3 import PPO

from python_rl.env.minecraft_env import MinecraftEnv


def evaluate_task(model, task_name, episodes=10):
    env = MinecraftEnv(task=task_name)
    stats = defaultdict(float)

    for _ in range(episodes):
        obs, info = env.reset(options={"task": task_name})
        done = False
        truncated = False

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            stats["reward_sum"] += reward

        stats["successes"] += float(info.get("success", False))
        stats["episodes"] += 1
        stats["steps_sum"] += info.get("episode_step", 0)

    env.close()
    return {
        "task": task_name,
        "avg_reward": stats["reward_sum"] / max(stats["episodes"], 1),
        "success_rate": stats["successes"] / max(stats["episodes"], 1),
        "avg_steps": stats["steps_sum"] / max(stats["episodes"], 1),
    }


def main():
    model = PPO.load("python_rl/checkpoints/multitask_day2_playernpc_run1")
    for task in ["navigation", "farming"]:
        print(evaluate_task(model, task))


if __name__ == "__main__":
    main()