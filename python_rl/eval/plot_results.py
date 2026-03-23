"""
plot_results.py
---------------
Plots episode reward and rolling success rate for a single training run.

Usage
-----
    python -m python_rl.eval.plot_results --monitor nav_shaped_monitor.csv \\
                                          --success nav_shaped_success.csv \\
                                          --title "Navigation (shaped)"

Both --monitor and --success are optional: the script will plot whatever
files are provided.

All files are looked up inside python_rl/logs/ unless an absolute path
is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGS_DIR = Path("python_rl/logs")
PLOTS_DIR = LOGS_DIR / "plots"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def smooth(values: list[float], window: int = 20) -> np.ndarray:
    """Simple moving average."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def load_monitor(path: Path) -> pd.DataFrame:
    """Load a Stable-Baselines3 Monitor CSV (skips the 2-line header)."""
    return pd.read_csv(path, comment="#")


def load_success(path: Path) -> pd.DataFrame:
    """Load a success CSV written by SuccessLogger / CurriculumCallback."""
    return pd.read_csv(path)


def rolling_success_rate(successes: list[int], window: int = 20) -> np.ndarray:
    """Rolling success rate over *window* episodes."""
    arr = np.array(successes, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a single training run.")
    parser.add_argument("--monitor", default=None,
                        help="Monitor CSV filename (inside python_rl/logs/).")
    parser.add_argument("--success", default=None,
                        help="Success CSV filename (inside python_rl/logs/).")
    parser.add_argument("--title",   default="Training run",
                        help="Plot title.")
    parser.add_argument("--smooth",  type=int, default=20,
                        help="Smoothing window (episodes).")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_title = args.title.replace(" ", "_").replace("/", "-")

    fig, axes = plt.subplots(
        1 + int(args.success is not None), 1,
        figsize=(10, 4 * (1 + int(args.success is not None))),
        squeeze=False,
    )
    row = 0

    # ---- Episode reward ----
    if args.monitor:
        monitor_path = LOGS_DIR / args.monitor if not Path(args.monitor).is_absolute() else Path(args.monitor)
        if not monitor_path.exists():
            print(f"[WARN] Monitor file not found: {monitor_path}")
        else:
            df = load_monitor(monitor_path)
            rewards = df["r"].tolist()
            smoothed = smooth(rewards, args.smooth)
            ax = axes[row][0]
            ax.plot(range(len(rewards)), rewards, alpha=0.25, color="steelblue", linewidth=0.8)
            ax.plot(range(args.smooth - 1, len(rewards)), smoothed, color="steelblue", linewidth=2,
                    label=f"Smoothed (w={args.smooth})")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Episode Reward")
            ax.set_title(f"{args.title} — Episode Reward")
            ax.legend()
            row += 1

    # ---- Success rate ----
    if args.success:
        success_path = LOGS_DIR / args.success if not Path(args.success).is_absolute() else Path(args.success)
        if not success_path.exists():
            print(f"[WARN] Success file not found: {success_path}")
        else:
            df = load_success(success_path)
            successes = df["success"].tolist()
            rate = rolling_success_rate(successes, args.smooth)
            ax = axes[row][0]
            ax.plot(range(len(successes)), successes, alpha=0.15, color="darkorange",
                    linewidth=0.6, label="Raw (0/1)")
            ax.plot(range(args.smooth - 1, len(successes)), rate, color="darkorange",
                    linewidth=2, label=f"Rolling rate (w={args.smooth})")
            ax.axhline(0.7, linestyle="--", color="gray", linewidth=1, label="0.70 threshold")
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Success Rate")
            ax.set_title(f"{args.title} — Success Rate")
            ax.legend()

    plt.tight_layout()
    out = PLOTS_DIR / f"{safe_title}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
