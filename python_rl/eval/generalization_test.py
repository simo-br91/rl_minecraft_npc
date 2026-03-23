"""
generalization_test.py
-----------------------
Tests trained models on held-out difficulty configurations.

Navigation configs (A–F): vary distance and obstacle count.
Farming configs (G–J): vary crop count and farming cycle complexity.

Usage
-----
    python -m python_rl.eval.generalization_test
    python -m python_rl.eval.generalization_test --model nav_curriculum_run1
    python -m python_rl.eval.generalization_test --model farm_run1 --task farming
    python -m python_rl.eval.generalization_test --model multitask_run1 --task both
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from python_rl.env.minecraft_env import MinecraftEnv
from python_rl.eval.eval_utils import load_model

LOGS_DIR  = Path("python_rl/logs")
PLOTS_DIR = LOGS_DIR / "plots"

# ------------------------------------------------------------------
# Navigation hold-out configs
# ------------------------------------------------------------------

NAV_CONFIGS = [
    {"id": "A", "label": "Short flat\n(3–6b, 0obs)",
     "min_dist": 3.0,  "max_dist": 6.0,  "num_obstacles": 0, "task": "navigation"},
    {"id": "B", "label": "Medium 1obs\n(5–9b, 1obs)",
     "min_dist": 5.0,  "max_dist": 9.0,  "num_obstacles": 1, "task": "navigation"},
    {"id": "C", "label": "Default train\n(7–14b, 2obs)",
     "min_dist": 7.0,  "max_dist": 14.0, "num_obstacles": 2, "task": "navigation"},
    {"id": "D", "label": "Long flat\n(12–18b, 0obs)",
     "min_dist": 12.0, "max_dist": 18.0, "num_obstacles": 0, "task": "navigation"},
    {"id": "E", "label": "Short dense\n(3–6b, 2obs)",
     "min_dist": 3.0,  "max_dist": 6.0,  "num_obstacles": 2, "task": "navigation"},
    {"id": "F", "label": "Very long 3obs\n(14–20b, 3obs)",
     "min_dist": 14.0, "max_dist": 20.0, "num_obstacles": 3, "task": "navigation"},
]

# ------------------------------------------------------------------
# Farming hold-out configs
# ------------------------------------------------------------------

FARMING_CONFIGS = [
    {"id": "G", "label": "1 crop\nharvest-only",
     "num_crops": 1,  "full_farm_cycle": False, "task": "farming"},
    {"id": "H", "label": "3 crops\nharvest-only",
     "num_crops": 3,  "full_farm_cycle": False, "task": "farming"},
    {"id": "I", "label": "5 crops\nharvest-only (train)",
     "num_crops": 5,  "full_farm_cycle": False, "task": "farming"},
    {"id": "J", "label": "5 crops\nfull cycle",
     "num_crops": 5,  "full_farm_cycle": True,  "task": "farming"},
    {"id": "K", "label": "10 crops\nharvest-only",
     "num_crops": 10, "full_farm_cycle": False, "task": "farming"},
]


# ------------------------------------------------------------------
# Evaluation helper
# ------------------------------------------------------------------

def evaluate_config(model: PPO, config: dict, n_episodes: int = 20) -> dict:
    """Run n_episodes with the given config and return aggregate stats."""
    task = config["task"]
    num_crops       = config.get("num_crops", 5)
    full_farm_cycle = config.get("full_farm_cycle", False)

    env = MinecraftEnv(task=task, num_crops=num_crops, full_farm_cycle=full_farm_cycle)
    reset_opts = {k: v for k, v in config.items()
                  if k not in ("id", "label", "task")}
    reset_opts["task"] = task

    successes = 0; total_reward = 0.0; total_steps = 0; total_crops = 0

    for _ in range(n_episodes):
        obs, _ = env.reset(options=reset_opts)
        done = truncated = False
        ep_reward = 0.0; ep_steps = 0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward; ep_steps += 1

        successes    += int(info.get("success", False))
        total_reward += ep_reward
        total_steps  += ep_steps
        total_crops  += info.get("crops_harvested", 0)

    env.close()
    return {
        "id":           config["id"],
        "label":        config["label"],
        "task":         task,
        "success_rate": round(successes / n_episodes, 3),
        "avg_reward":   round(total_reward / n_episodes, 3),
        "avg_steps":    round(total_steps / n_episodes, 1),
        "avg_crops":    round(total_crops / n_episodes, 2),
        "n_episodes":   n_episodes,
        "train_dist":   config.get("id") in ("C", "I"),   # highlight training distributions
    }


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------

def save_csv(results: list[dict], model_name: str, suffix: str = "") -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out = LOGS_DIR / f"generalization_{model_name}{suffix}.csv"
    fields = ["id", "label", "task", "success_rate",
              "avg_reward", "avg_steps", "avg_crops", "n_episodes"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})
    return out


def plot_results(results: list[dict], model_name: str, suffix: str = "") -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = [r["label"]        for r in results]
    srs    = [r["success_rate"] for r in results]
    colors = ["steelblue" if r["train_dist"] else "darkorange" for r in results]

    fig, ax = plt.subplots(figsize=(max(10, len(results) * 1.6), 5))
    bars = ax.bar(range(len(labels)), srs, color=colors, edgecolor="white")
    for bar, sr in zip(bars, srs):
        ax.text(bar.get_x() + bar.get_width() / 2, sr + 0.015,
                f"{sr:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.7, linestyle="--", color="gray", linewidth=1, label="0.70 target")
    ax.set_ylabel("Success Rate")
    ax.set_title(f"Generalization Test — {model_name}\n(blue = training distribution)")
    ax.legend(); plt.tight_layout()

    out = PLOTS_DIR / f"generalization_{model_name}{suffix}.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Plot saved: {out}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="nav_shaped_run1")
    parser.add_argument("--task",     default="navigation",
                        choices=["navigation", "farming", "both"],
                        help="Which task family to test.")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    if args.task in ("navigation", "both"):
        print(f"\nNavigation generalization ({len(NAV_CONFIGS)} configs × {args.episodes} eps)\n")
        nav_results = []
        for cfg in NAV_CONFIGS:
            print(f"  Config {cfg['id']}: {cfg['label'].replace(chr(10), ' ')} … ",
                  end="", flush=True)
            stats = evaluate_config(model, cfg, n_episodes=args.episodes)
            nav_results.append(stats)
            print(f"SR={stats['success_rate']:.2f}  "
                  f"avgR={stats['avg_reward']:.1f}  "
                  f"steps={stats['avg_steps']:.0f}")

        csv_out = save_csv(nav_results, args.model, "_nav")
        print(f"\nCSV: {csv_out}")
        plot_results(nav_results, args.model, "_nav")

    if args.task in ("farming", "both"):
        print(f"\nFarming generalization ({len(FARMING_CONFIGS)} configs × {args.episodes} eps)\n")
        farm_results = []
        for cfg in FARMING_CONFIGS:
            print(f"  Config {cfg['id']}: {cfg['label'].replace(chr(10), ' ')} … ",
                  end="", flush=True)
            stats = evaluate_config(model, cfg, n_episodes=args.episodes)
            farm_results.append(stats)
            print(f"SR={stats['success_rate']:.2f}  "
                  f"crops={stats['avg_crops']:.1f}  "
                  f"steps={stats['avg_steps']:.0f}")

        csv_out = save_csv(farm_results, args.model, "_farm")
        print(f"\nCSV: {csv_out}")
        plot_results(farm_results, args.model, "_farm")


if __name__ == "__main__":
    main()