from python_rl.env.minecraft_env import MinecraftEnv

def run_task(task_name, actions):
    env = MinecraftEnv(task=task_name)
    obs, info = env.reset(options={"task": task_name})
    print(f"RESET task={task_name}")
    print("obs:", obs)
    print("info:", info)

    for action in actions:
        obs, reward, done, truncated, info = env.step(action)
        print(f"action={action} reward={reward:.3f} done={done} truncated={truncated} info={info}")
        if done or truncated:
            break


if __name__ == "__main__":
    run_task("navigation", [0, 1, 2, 4])
    print("-" * 60)
    run_task("farming", [0, 0, 3])
