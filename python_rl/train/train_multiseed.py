"""
train_multiseed.py
------------------
Runs the same experiment with multiple seeds and reports variance.

Usage
-----
    python -m python_rl.train.train_multiseed --experiment nav_shaped --seeds 3
    python -m python_rl.train.train_multiseed --experiment farming    --seeds 3
    python -m python_rl.train.train_multiseed --experiment nav_shaped --seed-list 42 123 999

Issue 6.9 — all experiments were single-seed; this script enables multi-seed
training for any experiment defined in python_rl/configs/.

The script:
  1. Runs train_fn for each seed independently.
  2. Saves each checkpoint as <name>_seed<N>.
  3. Writes an aggregate summary CSV and prints mean ± std SR.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

import yaml
import numpy as np

LOGS_DIR        = Path("python_rl/logs")
CHECKPOINTS_DIR = Path("python_rl/checkpoints")

# Map experiment name → training module
EXPERIMENT_MAP: Dict[str, str] = {
    "nav_shaped":     "python_rl.train.train_navigation",
    "nav_sparse":     "python_rl.train.train_nav_sparse",
    "nav_curriculum": "python_rl.train.train_nav_curriculum",
    "farming":        "python_rl.train.train_farming",
    "combat":         "python_rl.train.train_combat",
    "multitask":      "python_rl.train.train_multitask",
}

SUCCESS_LOG_MAP: Dict[str, str] = {
    "nav_shaped":     "nav_shaped_success.csv",
    "nav_sparse":     "nav_sparse_success.csv",
    "nav_curriculum": "nav_curriculum_success.csv",
    "farming":        "farm_success.csv",
    "combat":         "combat_success.csv",
    "multitask":      "multitask_success.csv",
}


def _final_sr(success_log: Path, last_n: int = 100) -> float:
    if not success_log.exists():
        return float("nan")
    import csv
    rows = []
    with success_log.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(int(row.get("success", 0)))
    tail = rows[-last_n:]
    return float(sum(tail) / len(tail)) if tail else float("nan")


def run_single_seed(
    experiment: str,
    seed: int,
    extra_args: List[str],
) -> Dict[str, Any]:
    """Invoke the training script for one seed as a subprocess."""
    module = EXPERIMENT_MAP.get(experiment)
    if module is None:
        raise ValueError(
            f"Unknown experiment '{experiment}'. "
            f"Valid options: {list(EXPERIMENT_MAP.keys())}"
        )

    print(f"\n{'='*60}")
    print(f"[multiseed] {experiment} — seed {seed}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-m", module,
        "--seed", str(seed),
    ] + extra_args

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[multiseed] WARNING: seed {seed} exited with code {result.returncode}")

    # Read final success rate from the log written by the training script
    success_log = LOGS_DIR / SUCCESS_LOG_MAP.get(experiment, "nav_shaped_success.csv")
    sr = _final_sr(success_log)
    return {"seed": seed, "final_success_rate": sr}


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed training runner.")
    parser.add_argument("--experiment", default="nav_shaped",
                        choices=list(EXPERIMENT_MAP.keys()))
    parser.add_argument("--seeds",      type=int, default=3,
                        help="Number of seeds (uses 0, 1, ..., N-1 by default).")
    parser.add_argument("--seed-list",  type=int, nargs="+",
                        help="Explicit list of seeds (overrides --seeds).")
    parser.add_argument("--last-n",     type=int, default=100,
                        help="Episodes to average for final SR.")
    args, extra = parser.parse_known_args()

    seeds: List[int] = args.seed_list if args.seed_list else list(range(args.seeds))

    print(f"[multiseed] Experiment: {args.experiment}")
    print(f"[multiseed] Seeds: {seeds}")

    results = []
    for seed in seeds:
        r = run_single_seed(args.experiment, seed, extra)
        results.append(r)
        print(f"[multiseed] Seed {seed}: SR = {r['final_success_rate']:.3f}")

    srs   = [r["final_success_rate"] for r in results if not np.isnan(r["final_success_rate"])]
    mean  = float(np.mean(srs))  if srs else float("nan")
    std   = float(np.std(srs))   if srs else float("nan")

    print(f"\n{'='*60}")
    print(f"[multiseed] {args.experiment}")
    print(f"  Mean SR : {mean:.3f}")
    print(f"  Std SR  : {std:.3f}")
    print(f"  Seeds   : {seeds}")
    print(f"{'='*60}")

    # Write summary CSV
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = LOGS_DIR / f"{args.experiment}_multiseed.csv"
    import csv
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "final_success_rate"])
        writer.writeheader()
        writer.writerows(results)

    # Write summary YAML
    out_yaml = LOGS_DIR / f"{args.experiment}_multiseed_summary.yaml"
    with out_yaml.open("w") as f:
        yaml.dump({"mean_sr": mean, "std_sr": std, "seeds": seeds,
                   "results": results}, f, default_flow_style=False)

    print(f"[multiseed] Summary CSV  : {out_csv}")
    print(f"[multiseed] Summary YAML : {out_yaml}")


if __name__ == "__main__":
    main()
