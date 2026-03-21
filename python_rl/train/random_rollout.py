import random
from python_rl.env.minecraft_env import MinecraftEnv

env = MinecraftEnv()

num_episodes = 20

for ep in range(num_episodes):
    obs, info = env.reset()
    total_reward = 0.0
    done = False
    truncated = False
    steps = 0

    while not done and not truncated:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    print(
        f"Episode {ep + 1} | reward={total_reward:.3f} | "
        f"steps={steps} | success={info['success']} | "
        f"final_dist={info['distance_to_target']}"
    )