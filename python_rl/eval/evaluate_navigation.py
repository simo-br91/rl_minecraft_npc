"""
evaluate.py
-----------
Deterministic rollout for the navigation task.

Usage
-----
    python -m python_rl.eval.evaluate
    python -m python_rl.eval.evaluate --model nav_curriculum_run1 --episodes 10
"""

from __future__ import annotations

import argparse

from python_rl.env.minecraft_env import MinecraftEnv
from python_rl.eval.eval_utils import load_model, run_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a navigation model.")
    parser.add_argument("--model",    default="nav_shaped_run1")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--quiet",    action="store_true", help="Suppress per-step output.")
    args = parser.parse_args()

    model = load_model(args.model)
    env   = MinecraftEnv(task="navigation")

    summary = run_episodes(model, env, "navigation",
                           args.episodes, verbose=not args.quiet)
    env.close()

    print(f"\n{'='*60}")
    print(f"Model     : {args.model}")
    print(f"Episodes  : {summary['episodes']}")
    print(f"Success   : {summary['success_rate']:.0%}")
    print(f"Avg reward: {summary['avg_reward']:+.2f}")
    print(f"Avg steps : {summary['avg_steps']:.0f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()