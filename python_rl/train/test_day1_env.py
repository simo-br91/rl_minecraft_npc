from python_rl.env.minecraft_env import MinecraftEnv

env = MinecraftEnv()

obs, info = env.reset()
print("RESET")
print("obs:", obs)
print("info:", info)

for action in [0, 1, 2, 4]:
    print(f"\nSTEP action={action}")
    obs, reward, done, truncated, info = env.step(action)
    print("obs:", obs)
    print("reward:", reward)
    print("done:", done)
    print("truncated:", truncated)
    print("info:", info)