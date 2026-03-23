"""
evaluate_multitask.py
---------------------
Evaluates a multitask model on the combined "multitask" episode type
(navigation + farming + combat in a single episode) and optionally also
evaluates single-task generalisation.

Usage
-----
    python -m python_rl.eval.evaluate_multitask
    python -m python_rl.eval.evaluate_multitask --model multitask_run1 --episodes 10
    python -m python_rl.eval.evaluate_multitask --also-single-task
"""

from __future__ import annotations

import argparse

from python_rl.env.minecraft_env import MinecraftEnv
from python_rl.eval.eval_utils import load_model, run_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a multitask model.")
    parser.add_argument("--model",            default="multitask_run1")
    parser.add_argument("--episodes",         type=int, default=10)
    parser.add_argument("--crops",            type=int, default=5)
    parser.add_argument("--also-single-task", action="store_true",
                        help="Also benchmark on isolated navigation/farming/combat.")
    parser.add_argument("--quiet",            action="store_true")
    args = parser.parse_args()

    model = load_model(args.model)
    verbose = not args.quiet

    # ------------------------------------------------------------------
    # Primary: combined multitask episode
    # ------------------------------------------------------------------
    print("\n=== Multitask (combined) ===")
    env = MinecraftEnv(task="multitask", num_crops=args.crops)
    summary_mt = run_episodes(model, env, "multitask", args.episodes,
                               reset_options={"num_crops": args.crops},
                               verbose=verbose)
    env.close()

    results = [summary_mt]

    # ------------------------------------------------------------------
    # Optional: single-task generalisation
    # ------------------------------------------------------------------
    if args.also_single_task:
        for task in ("navigation", "farming", "combat"):
            print(f"\n=== Single-task: {task} ===")
            task_env = MinecraftEnv(task=task,
                                    num_crops=args.crops if task == "farming" else 1)
            s = run_episodes(model, task_env, task, args.episodes, verbose=verbose)
            task_env.close()
            results.append(s)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"{'Task':<18} {'SR':>6} {'AvgR':>8} {'Steps':>7} "
          f"{'Crops':>7} {'Mobs':>6} {'HP':>6}")
    print(f"{'-'*72}")
    for r in results:
        print(
            f"{r['task']:<18} {r['success_rate']:>6.2f} "
            f"{r['avg_reward']:>8.1f} {r['avg_steps']:>7.0f} "
            f"{r['avg_crops']:>7.1f} {r['avg_mobs']:>6.1f} "
            f"{r['avg_health']:>6.1f}"
        )
    print(f"{'='*72}")


if __name__ == "__main__":
    main()