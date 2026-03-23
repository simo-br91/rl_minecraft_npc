"""
evaluate_multitask.py
---------------------
Evaluates a multi-task model separately on navigation and farming,
then prints a per-task breakdown.

Usage
-----
    python -m python_rl.eval.evaluate_multitask
    python -m python_rl.eval.evaluate_multitask --model multitask_run1 --episodes 10
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from stable_baselines3 import PPO

from python_rl.env.minecraft_env import MinecraftEnv

CHECKPOINTS_DIR = Path("python_rl/checkpoints")


def evaluate_task(
    model: PPO,
    task_name: str,
    n_episodes: int,
) -> dict:
    env = MinecraftEnv(task=task_name)
    stats: dict[str, float] = defaultdict(float)

    for _ in range(n_episodes):
        obs, _ = env.reset(options={"task": task_name})
        done = truncated = False

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            stats["reward_sum"] += reward

        stats["successes"]  += float(info.get("success", False))
        stats["episodes"]   += 1
        stats["steps_sum"]  += info.get("episode_step", 0)

    env.close()
    n = max(stats["episodes"], 1)
    return {
        "task":         task_name,
        "episodes":     int(stats["episodes"]),
        "success_rate": round(stats["successes"] / n, 3),
        "avg_reward":   round(stats["reward_sum"] / n, 3),
        "avg_steps":    round(stats["steps_sum"] / n, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a multi-task model.")
    parser.add_argument("--model",    default="multitask_run1")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Episodes per task (default 10).")
    args = parser.parse_args()

    model_path = CHECKPOINTS_DIR / args.model
    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}.zip")

    print(f"Loading model: {model_path}")
    model = PPO.load(str(model_path))

    print(f"\nEvaluating {args.episodes} episodes per task …\n")
    results = []
    for task in ("navigation", "farming"):
        r = evaluate_task(model, task, args.episodes)
        results.append(r)
        print(
            f"  {task:<12}  success={r['success_rate']:.2f}  "
            f"avg_reward={r['avg_reward']:+.1f}  avg_steps={r['avg_steps']:.0f}"
        )

    print("\n" + "="*60)
    print(f"{'Task':<14} {'SR':>6} {'Avg R':>8} {'Avg steps':>10}")
    print("-"*60)
    for r in results:
        print(f"{r['task']:<14} {r['success_rate']:>6.2f} "
              f"{r['avg_reward']:>8.1f} {r['avg_steps']:>10.0f}")
    print("="*60)


if __name__ == "__main__":
    main()
