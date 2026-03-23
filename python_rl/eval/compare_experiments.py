"""
compare_experiments.py
----------------------
Generates comparison plots for the two key experiments required by
the project plan:

  1. Shaped vs Sparse rewards  (navigation)
  2. Curriculum vs No-curriculum  (navigation)

Also produces a summary bar chart of final success rates.

Usage
-----
    python -m python_rl.eval.compare_experiments

All monitor / success CSVs are loaded from python_rl/logs/.
Missing files are gracefully skipped with a warning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGS_DIR  = Path("python_rl/logs")
PLOTS_DIR = LOGS_DIR / "plots"


# ------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------

def load_monitor(filename: str) -> Optional[pd.DataFrame]:
    p = LOGS_DIR / filename
    if not p.exists():
        print(f"  [SKIP] {p} not found")
        return None
    return pd.read_csv(p, comment="#")


def load_success(filename: str) -> Optional[pd.DataFrame]:
    p = LOGS_DIR / filename
    if not p.exists():
        print(f"  [SKIP] {p} not found")
        return None
    return pd.read_csv(p)


def smooth(values: list[float], window: int) -> np.ndarray:
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def rolling_sr(successes: list[int], window: int) -> np.ndarray:
    arr = np.array(successes, dtype=float)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def final_success_rate(success_file: str, last_n: int = 50) -> float:
    df = load_success(success_file)
    if df is None or len(df) == 0:
        return float("nan")
    tail = df["success"].values[-last_n:]
    return float(tail.mean())


# ------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------

def plot_comparison(
    ax_reward: plt.Axes,
    ax_success: plt.Axes,
    runs: list[dict],
    window: int = 20,
) -> None:
    """
    *runs* is a list of dicts with keys:
        label        : str
        monitor_csv  : str   (filename in LOGS_DIR)
        success_csv  : str   (filename in LOGS_DIR)
        color        : str
    """
    for run in runs:
        label   = run["label"]
        color   = run.get("color", None)

        # --- reward ---
        df_mon = load_monitor(run["monitor_csv"])
        if df_mon is not None:
            rewards  = df_mon["r"].tolist()
            sm       = smooth(rewards, window)
            ax_reward.plot(range(len(rewards)), rewards, alpha=0.18,
                           linewidth=0.7, color=color)
            ax_reward.plot(range(window - 1, len(rewards)), sm,
                           linewidth=2, label=label, color=color)

        # --- success rate ---
        df_suc = load_success(run["success_csv"])
        if df_suc is not None:
            successes = df_suc["success"].tolist()
            sr        = rolling_sr(successes, window)
            ax_success.plot(range(len(successes)), successes, alpha=0.12,
                            linewidth=0.6, color=color)
            ax_success.plot(range(window - 1, len(successes)), sr,
                            linewidth=2, label=label, color=color)


# ------------------------------------------------------------------
# Experiment 1: Shaped vs Sparse
# ------------------------------------------------------------------

def plot_shaped_vs_sparse(window: int = 20) -> None:
    print("Plotting: shaped vs sparse rewards …")
    runs = [
        {
            "label":       "Shaped reward",
            "monitor_csv": "nav_shaped_monitor.csv",
            "success_csv": "nav_shaped_success.csv",
            "color":       "steelblue",
        },
        {
            "label":       "Sparse reward",
            "monitor_csv": "nav_sparse_monitor.csv",
            "success_csv": "nav_sparse_success.csv",
            "color":       "darkorange",
        },
    ]

    fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize=(14, 5))
    plot_comparison(ax_r, ax_s, runs, window)

    ax_r.set_xlabel("Episode"); ax_r.set_ylabel("Episode Reward")
    ax_r.set_title("Navigation: Shaped vs Sparse — Reward")
    ax_r.legend()

    ax_s.set_xlabel("Episode"); ax_s.set_ylabel("Rolling Success Rate")
    ax_s.set_title("Navigation: Shaped vs Sparse — Success Rate")
    ax_s.set_ylim(-0.05, 1.05)
    ax_s.axhline(0.7, linestyle="--", color="gray", linewidth=1)
    ax_s.legend()

    plt.tight_layout()
    out = PLOTS_DIR / "shaped_vs_sparse.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ------------------------------------------------------------------
# Experiment 2: Curriculum vs No-curriculum
# ------------------------------------------------------------------

def plot_curriculum_vs_nocurriculum(window: int = 20) -> None:
    print("Plotting: curriculum vs no-curriculum …")
    runs = [
        {
            "label":       "With curriculum",
            "monitor_csv": "nav_curriculum_monitor.csv",
            "success_csv": "nav_curriculum_success.csv",
            "color":       "seagreen",
        },
        {
            "label":       "No curriculum (shaped)",
            "monitor_csv": "nav_shaped_monitor.csv",
            "success_csv": "nav_shaped_success.csv",
            "color":       "steelblue",
        },
    ]

    fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize=(14, 5))
    plot_comparison(ax_r, ax_s, runs, window)

    # Overlay level-advancement markers on the success plot
    level_csv = LOGS_DIR / "nav_curriculum_levels.csv"
    if level_csv.exists():
        df_lv = pd.read_csv(level_csv)
        for _, row in df_lv.iterrows():
            if row.get("success", 0) == 0:
                continue
            # Mark level transitions in the reward plot
        # Find episodes where level changed
        if "level" in df_lv.columns:
            prev = None
            for idx, row in df_lv.iterrows():
                lv = int(row["level"])
                if prev is not None and lv != prev:
                    ax_r.axvline(idx, linestyle=":", color="seagreen",
                                 alpha=0.6, linewidth=1)
                    ax_s.axvline(idx, linestyle=":", color="seagreen",
                                 alpha=0.6, linewidth=1,
                                 label=f"→ level {lv}" if idx < 5 else None)
                prev = lv

    ax_r.set_xlabel("Episode"); ax_r.set_ylabel("Episode Reward")
    ax_r.set_title("Navigation: Curriculum vs No Curriculum — Reward")
    ax_r.legend()

    ax_s.set_xlabel("Episode"); ax_s.set_ylabel("Rolling Success Rate")
    ax_s.set_title("Navigation: Curriculum vs No Curriculum — Success Rate")
    ax_s.set_ylim(-0.05, 1.05)
    ax_s.axhline(0.7, linestyle="--", color="gray", linewidth=1)
    ax_s.legend()

    plt.tight_layout()
    out = PLOTS_DIR / "curriculum_vs_nocurriculum.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ------------------------------------------------------------------
# Summary bar chart: final success rates across all experiments
# ------------------------------------------------------------------

def plot_final_success_bar() -> None:
    print("Plotting: final success-rate bar chart …")
    experiments = [
        ("Nav shaped",      "nav_shaped_success.csv"),
        ("Nav sparse",      "nav_sparse_success.csv"),
        ("Nav curriculum",  "nav_curriculum_success.csv"),
        ("Farming",         "farm_success.csv"),
        ("Multi-task",      "multitask_success.csv"),
    ]

    labels, rates = [], []
    for label, csv_name in experiments:
        sr = final_success_rate(csv_name)
        labels.append(label)
        rates.append(sr)

    valid = [(l, r) for l, r in zip(labels, rates) if not np.isnan(r)]
    if not valid:
        print("  No success CSV files found — skipping bar chart.")
        return

    labels_v, rates_v = zip(*valid)
    colors = ["steelblue", "darkorange", "seagreen", "mediumpurple", "crimson"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels_v, rates_v,
                  color=colors[:len(labels_v)], edgecolor="white", linewidth=0.8)
    for bar, rate in zip(bars, rates_v):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.01,
                f"{rate:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.axhline(0.7, linestyle="--", color="gray", linewidth=1, label="0.70 target")
    ax.set_ylabel("Final Success Rate (last 50 episodes)")
    ax.set_title("Final Success Rate — All Experiments")
    ax.legend()
    plt.tight_layout()
    out = PLOTS_DIR / "final_success_rates.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ------------------------------------------------------------------
# Multi-task reward overview
# ------------------------------------------------------------------

def plot_multitask_overview(window: int = 20) -> None:
    print("Plotting: multi-task overview …")
    runs = [
        {
            "label":       "Multi-task (nav+farm)",
            "monitor_csv": "multitask_day2_playernpc_monitor.csv",
            "success_csv": "multitask_success.csv",
            "color":       "crimson",
        },
        {
            "label":       "Navigation only",
            "monitor_csv": "nav_shaped_monitor.csv",
            "success_csv": "nav_shaped_success.csv",
            "color":       "steelblue",
        },
        {
            "label":       "Farming only",
            "monitor_csv": "farm_monitor.csv",
            "success_csv": "farm_success.csv",
            "color":       "mediumpurple",
        },
    ]

    fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize=(14, 5))
    plot_comparison(ax_r, ax_s, runs, window)

    ax_r.set_xlabel("Episode"); ax_r.set_ylabel("Episode Reward")
    ax_r.set_title("Multi-task vs Single-task — Reward")
    ax_r.legend()

    ax_s.set_xlabel("Episode"); ax_s.set_ylabel("Rolling Success Rate")
    ax_s.set_title("Multi-task vs Single-task — Success Rate")
    ax_s.set_ylim(-0.05, 1.05)
    ax_s.legend()

    plt.tight_layout()
    out = PLOTS_DIR / "multitask_overview.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Generating comparison plots ===")
    plot_shaped_vs_sparse()
    plot_curriculum_vs_nocurriculum()
    plot_multitask_overview()
    plot_final_success_bar()
    print("=== Done ===")
    print(f"All plots saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
