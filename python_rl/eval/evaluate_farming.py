"""
evaluate_farming.py
-------------------
Deterministic rollout for the farming task.

Usage
-----
    python -m python_rl.eval.evaluate_farming
    python -m python_rl.eval.evaluate_farming --model farm_run1 --episodes 10
    python -m python_rl.eval.evaluate_farming --crops 5 --full-cycle
"""

from __future__ import annotations

import argparse

from python_rl.env.minecraft_env import MinecraftEnv
from python_rl.eval.eval_utils import load_model, run_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a farming model.")
    parser.add_argument("--model",      default="farm_run1")
    parser.add_argument("--episodes",   type=int, default=5)
    parser.add_argument("--crops",      type=int, default=5,
                        help="Number of wheat plots per episode.")
    parser.add_argument("--full-cycle", action="store_true",
                        help="Use seeds→bonemeal→harvest cycle instead of pre-grown.")
    parser.add_argument("--quiet",      action="store_true")
    args = parser.parse_args()

    model = load_model(args.model)
    env   = MinecraftEnv(task="farming",
                         num_crops=args.crops,
                         full_farm_cycle=args.full_cycle)

    reset_opts = {
        "num_crops":      args.crops,
        "full_farm_cycle": args.full_cycle,
    }

    summary = run_episodes(model, env, "farming",
                           args.episodes,
                           reset_options=reset_opts,
                           verbose=not args.quiet)
    env.close()

    print(f"\n{'='*60}")
    print(f"Model      : {args.model}")
    print(f"Crops/ep   : {args.crops}  full_cycle={args.full_cycle}")
    print(f"Episodes   : {summary['episodes']}")
    print(f"Success    : {summary['success_rate']:.0%}")
    print(f"Avg reward : {summary['avg_reward']:+.2f}")
    print(f"Avg steps  : {summary['avg_steps']:.0f}")
    print(f"Avg crops  : {summary['avg_crops']:.1f}/{args.crops}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()