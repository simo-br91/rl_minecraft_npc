from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    log_file = Path("python_rl/logs/nav_day1_monitor.csv")
    if not log_file.exists():
        raise FileNotFoundError(f"Missing log file: {log_file}")

    df = pd.read_csv(log_file, comment="#")

    if "r" not in df.columns:
        raise ValueError("Monitor file does not contain episodic reward column 'r'.")

    plots_dir = Path("python_rl/logs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["r"])
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Day 1 Navigation Training Reward")
    plt.tight_layout()
    out_path = plots_dir / "nav_day1_reward.png"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()