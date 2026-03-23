"""
generalization_test.py
-----------------------
Tests a trained navigation model on held-out difficulty configurations
that were NOT necessarily part of training.

Configurations tested
---------------------
  A  Short, flat          (3–6 blocks, 0 obstacles)   — easy
  B  Medium, 1 obstacle   (5–9 blocks, 1 obstacle)    — medium
  C  Long, 2 obstacles    (7–14 blocks, 2 obstacles)  — same as default train
  D  Very long, flat      (12–18 blocks, 0 obstacles) — distance stress test
  E  Short, 2 obstacles   (3–6 blocks, 2 obstacles)   — obstacle stress test
  F  Very long, 3 walls   (14–20 blocks, 3 obstacles) — hardest

Usage
-----
    # defaults to nav_shaped_run1
    python -m python_rl.eval.generalization_test

    # specify a different checkpoint
    python -m python_rl.eval.generalization_test --model nav_curriculum_run1
    python -m python_rl.eval.generalization_test --model nav_sparse_run1

Results are printed to stdout AND saved as a CSV + a bar chart.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from python_rl.env.minecraft_env import MinecraftEnv

CHECKPOINTS_DIR = Path("python_rl/checkpoints")
LOGS_DIR        = Path("python_rl/logs")
PLOTS_DIR       = LOGS_DIR / "plots"

# ------------------------------------------------------------------
# Hold-out configurations
# ------------------------------------------------------------------

GENERALIZATION_CONFIGS = [
    {
        "id":          "A",
        "label":       "Short, flat\n(3–6 b, 0 obs)",
        "min_dist":    3.0,
        "max_dist":    6.0,
        "num_obstacles": 0,
    },
    {
        "id":          "B",
        "label":       "Medium, 1 obs\n(5–9 b, 1 obs)",
        "min_dist":    5.0,
        "max_dist":    9.0,
        "num_obstacles": 1,
    },
    {
        "id":          "C",
        "label":       "Default train\n(7–14 b, 2 obs)",
        "min_dist":    7.0,
        "max_dist":   14.0,
        "num_obstacles": 2,
    },
    {
        "id":          "D",
        "label":       "Long, flat\n(12–18 b, 0 obs)",
        "min_dist":   12.0,
        "max_dist":   18.0,
        "num_obstacles": 0,
    },
    {
        "id":          "E",
        "label":       "Short, dense\n(3–6 b, 2 obs)",
        "min_dist":    3.0,
        "max_dist":    6.0,
        "num_obstacles": 2,
    },
    {
        "id":          "F",
        "label":       "Very long, 3 obs\n(14–20 b, 3 obs)",
        "min_dist":   14.0,
        "max_dist":   20.0,
        "num_obstacles": 3,
    },
]


# ------------------------------------------------------------------
# Evaluation helper
# ------------------------------------------------------------------

def evaluate_config(
    model: PPO,
    env: MinecraftEnv,
    config: dict,
    n_episodes: int = 20,
) -> dict:
    """Run *n_episodes* with the given config and return aggregate stats."""
    successes   = 0
    total_reward = 0.0
    total_steps  = 0

    reset_opts = {
        "task":          "navigation",
        "min_dist":      config["min_dist"],
        "max_dist":      config["max_dist"],
        "num_obstacles": config["num_obstacles"],
    }

    for _ in range(n_episodes):
        obs, _ = env.reset(options=reset_opts)
        done = truncated = False
        ep_reward = 0.0
        ep_steps  = 0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps  += 1

        successes    += int(info.get("success", False))
        total_reward += ep_reward
        total_steps  += ep_steps

    return {
        "id":           config["id"],
        "label":        config["label"],
        "success_rate": round(successes / n_episodes, 3),
        "avg_reward":   round(total_reward / n_episodes, 3),
        "avg_steps":    round(total_steps / n_episodes, 1),
        "n_episodes":   n_episodes,
    }


# ------------------------------------------------------------------
# Report / plot
# ------------------------------------------------------------------

def save_csv(results: list[dict], model_name: str) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out = LOGS_DIR / f"generalization_{model_name}.csv"
    fieldnames = ["id", "label", "success_rate", "avg_reward", "avg_steps", "n_episodes"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fieldnames})
    return out


def plot_results(results: list[dict], model_name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    ids    = [r["id"]          for r in results]
    labels = [r["label"]       for r in results]
    srs    = [r["success_rate"] for r in results]

    colors = ["steelblue" if r["id"] == "C" else "darkorange" for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(ids)), srs, color=colors, edgecolor="white")
    for bar, sr in zip(bars, srs):
        ax.text(bar.get_x() + bar.get_width() / 2, sr + 0.015,
                f"{sr:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.7, linestyle="--", color="gray", linewidth=1, label="0.70 target")
    ax.set_ylabel("Success Rate")
    ax.set_title(f"Generalization Test — {model_name}\n(blue = training distribution C)")
    ax.legend()
    plt.tight_layout()

    out = PLOTS_DIR / f"generalization_{model_name}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Plot saved: {out}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generalization test for a navigation model.")
    parser.add_argument("--model",      default="nav_shaped_run1",
                        help="Checkpoint name (no extension) in python_rl/checkpoints/.")
    parser.add_argument("--episodes",   type=int, default=20,
                        help="Episodes per configuration (default 20).")
    args = parser.parse_args()

    model_path = CHECKPOINTS_DIR / args.model
    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}.zip\n"
            "Train first with train_navigation.py or train_nav_curriculum.py."
        )

    print(f"Loading model: {model_path}")
    model = PPO.load(str(model_path))
    env   = MinecraftEnv(task="navigation")

    print(f"\nEvaluating {len(GENERALIZATION_CONFIGS)} configurations "
          f"× {args.episodes} episodes each …\n")

    results = []
    for cfg in GENERALIZATION_CONFIGS:
        print(f"  Config {cfg['id']}: {cfg['label'].replace(chr(10), ' ')} … ", end="", flush=True)
        stats = evaluate_config(model, env, cfg, n_episodes=args.episodes)
        results.append(stats)
        print(f"success={stats['success_rate']:.2f}  avg_reward={stats['avg_reward']:.1f}  "
              f"avg_steps={stats['avg_steps']:.0f}")

    env.close()

    # Print table
    print("\n" + "="*72)
    print(f"{'ID':<4} {'Config':<30} {'SR':>6} {'Avg R':>8} {'Avg steps':>10}")
    print("-"*72)
    for r in results:
        label_flat = r["label"].replace("\n", " ")
        print(f"{r['id']:<4} {label_flat:<30} {r['success_rate']:>6.2f} "
              f"{r['avg_reward']:>8.1f} {r['avg_steps']:>10.0f}")
    print("="*72)

    # Save
    csv_out = save_csv(results, args.model)
    print(f"\nCSV saved: {csv_out}")
    plot_results(results, args.model)


if __name__ == "__main__":
    main()
