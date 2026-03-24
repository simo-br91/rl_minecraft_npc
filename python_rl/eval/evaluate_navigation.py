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
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from pathlib import Path

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
    vec_env = DummyVecEnv([lambda: env])
    vnorm_path = Path("python_rl/checkpoints/nav_shaped_vecnorm.pkl")
    if vnorm_path.exists():
        vec_env = VecNormalize.load(str(vnorm_path), vec_env)
        vec_env.training = False   # freeze stats during eval
        vec_env.norm_reward = False
    env = MinecraftEnv(task="navigation")

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