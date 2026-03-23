"""
evaluate.py
-----------
Runs a trained navigation model for N deterministic episodes and
prints per-step info plus a final summary.

Usage
-----
    python -m python_rl.eval.evaluate
    python -m python_rl.eval.evaluate --model nav_curriculum_run1 --episodes 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from python_rl.env.minecraft_env import MinecraftEnv

CHECKPOINTS_DIR = Path("python_rl/checkpoints")


def run_episode(model: PPO, env: MinecraftEnv, *, verbose: bool = True) -> dict:
    obs, _  = env.reset(options={"task": "navigation"})
    done = truncated = False
    total_reward = 0.0
    steps = 0

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if verbose:
            print(
                f"  step={steps:3d}  action={int(action)}  reward={reward:+.3f}  "
                f"dist={info.get('distance_to_target', '?'):.2f}  "
                f"done={done}  trunc={truncated}"
            )

    return {
        "success":      info.get("success", False),
        "steps":        steps,
        "total_reward": round(total_reward, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a navigation model.")
    parser.add_argument("--model",    default="nav_shaped_run1")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    model_path = CHECKPOINTS_DIR / args.model
    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}.zip")

    model = PPO.load(str(model_path))
    env   = MinecraftEnv(task="navigation")

    successes = 0
    for ep in range(1, args.episodes + 1):
        print(f"\n--- Episode {ep} ---")
        stats = run_episode(model, env, verbose=True)
        successes += int(stats["success"])
        print(
            f"  → success={stats['success']}  steps={stats['steps']}  "
            f"total_reward={stats['total_reward']}"
        )

    env.close()
    print(f"\nSummary: {successes}/{args.episodes} successes "
          f"({successes/args.episodes:.0%})")


if __name__ == "__main__":
    main()
