"""
tests/test_train_utils.py
--------------------------
Unit tests for python_rl/train/train_utils.py

Tests that do NOT require a running Minecraft server:
  - Config loading from YAML
  - EarlyStoppingCallback logic
  - SuccessLogger CSV writing
  - make_periodic_checkpoint factory

Run with:
    python -m pytest tests/test_train_utils.py -v
"""

from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from python_rl.train.train_utils import (
    SuccessLogger,
    EarlyStoppingCallback,
    make_periodic_checkpoint,
    load_config,
    _load_last_n,
)


# ------------------------------------------------------------------ #
# load_config
# ------------------------------------------------------------------ #

class TestLoadConfig:
    def test_loads_nav_shaped(self):
        cfg = load_config("nav_shaped")
        assert isinstance(cfg, dict)
        assert "total_timesteps" in cfg
        assert "learning_rate" in cfg
        assert "ent_coef" in cfg

    def test_loads_with_yaml_extension(self):
        cfg = load_config("nav_shaped.yaml")
        assert isinstance(cfg, dict)

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config_xyz")

    def test_nav_shaped_has_correct_task(self):
        cfg = load_config("nav_shaped")
        assert cfg.get("task") == "navigation"

    def test_nav_sparse_has_higher_ent_coef(self):
        shaped = load_config("nav_shaped")
        sparse = load_config("nav_sparse")
        assert sparse["ent_coef"] >= shaped["ent_coef"]

    def test_farming_config_has_num_crops(self):
        cfg = load_config("farming")
        assert "num_crops" in cfg

    def test_multitask_config_loads(self):
        cfg = load_config("multitask")
        assert cfg.get("task") == "multitask"


# ------------------------------------------------------------------ #
# _load_last_n helper
# ------------------------------------------------------------------ #

class TestLoadLastN:
    def _write_csv(self, path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timestep", "task", "success",
                                               "health", "food_level",
                                               "crops_harvested", "mobs_killed",
                                               "episode_steps"])
            w.writeheader()
            w.writerows(rows)

    def test_returns_empty_for_missing_file(self, tmp_path):
        result = _load_last_n(tmp_path / "no_such.csv", 10)
        assert result == []

    def test_returns_all_rows_when_fewer_than_n(self, tmp_path):
        p = tmp_path / "log.csv"
        self._write_csv(p, [
            {"timestep": 1, "task": "navigation", "success": 1,
             "health": 20, "food_level": 20, "crops_harvested": 0,
             "mobs_killed": 0, "episode_steps": 50},
        ])
        rows = _load_last_n(p, 10)
        assert len(rows) == 1

    def test_returns_last_n_rows(self, tmp_path):
        p = tmp_path / "log.csv"
        data = [{"timestep": i, "task": "navigation",
                  "success": int(i % 2 == 0),
                  "health": 20, "food_level": 20,
                  "crops_harvested": 0, "mobs_killed": 0,
                  "episode_steps": 10}
                for i in range(20)]
        self._write_csv(p, data)
        rows = _load_last_n(p, 5)
        assert len(rows) == 5
        # Last 5 rows: indices 15-19 → success = [0,1,0,1,0] alternating
        assert rows[0]["success"] == int(15 % 2 == 0)


# ------------------------------------------------------------------ #
# SuccessLogger
# ------------------------------------------------------------------ #

class TestSuccessLogger:
    def test_creates_csv_with_header(self, tmp_path):
        log = str(tmp_path / "success.csv")
        callback = SuccessLogger(log)
        assert Path(log).exists()
        with open(log) as f:
            header = f.readline()
        assert "success" in header
        assert "timestep" in header

    def test_creates_parent_dirs(self, tmp_path):
        log = str(tmp_path / "deep" / "nested" / "success.csv")
        callback = SuccessLogger(log)
        assert Path(log).exists()

    def test_writes_row_on_done(self, tmp_path):
        log = str(tmp_path / "success.csv")
        callback = SuccessLogger(log)
        # Simulate SB3 callback internals
        callback.num_timesteps = 100
        callback.locals = {
            "dones": [True],
            "infos": [{"success": True, "task_name": "navigation",
                       "health": 20.0, "food_level": 20,
                       "crops_harvested": 0, "mobs_killed": 0,
                       "episode_step": 50}],
        }
        callback._on_step()
        with open(log) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["success"] == "1"
        assert rows[0]["task"] == "navigation"

    def test_skips_non_done_steps(self, tmp_path):
        log = str(tmp_path / "success.csv")
        callback = SuccessLogger(log)
        callback.num_timesteps = 100
        callback.locals = {
            "dones": [False],
            "infos": [{"success": False}],
        }
        callback._on_step()
        with open(log) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 0

    def test_handles_multiple_envs_in_batch(self, tmp_path):
        log = str(tmp_path / "success.csv")
        callback = SuccessLogger(log)
        callback.num_timesteps = 200
        callback.locals = {
            "dones": [True, False, True],
            "infos": [
                {"success": True,  "task_name": "navigation", "health": 20,
                 "food_level": 20, "crops_harvested": 0, "mobs_killed": 0,
                 "episode_step": 10},
                {"success": False, "task_name": "farming"},
                {"success": False, "task_name": "combat", "health": 15,
                 "food_level": 18, "crops_harvested": 2, "mobs_killed": 1,
                 "episode_step": 80},
            ],
        }
        callback._on_step()
        with open(log) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2   # only the two done=True steps


# ------------------------------------------------------------------ #
# EarlyStoppingCallback
# ------------------------------------------------------------------ #

class TestEarlyStoppingCallback:
    def _make_csv(self, path: Path, successes: list[int]) -> None:
        """Write a minimal success CSV."""
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timestep", "task", "success",
                                               "health", "food_level",
                                               "crops_harvested", "mobs_killed",
                                               "episode_steps"])
            w.writeheader()
            for i, s in enumerate(successes):
                w.writerow({"timestep": i * 100, "task": "navigation",
                             "success": s, "health": 20, "food_level": 20,
                             "crops_harvested": 0, "mobs_killed": 0,
                             "episode_steps": 50})

    def test_continues_when_below_target(self, tmp_path):
        log = str(tmp_path / "s.csv")
        self._make_csv(Path(log), [1, 0, 1, 0, 1] * 10)  # 50% success
        cb = EarlyStoppingCallback(log, target_success_rate=0.80, window=50, patience=1)
        cb.num_timesteps = 0
        result = cb._on_rollout_end()
        assert result is True   # don't stop

    def test_continues_when_insufficient_data(self, tmp_path):
        log = str(tmp_path / "s.csv")
        self._make_csv(Path(log), [1] * 10)  # only 10 episodes, window=50
        cb = EarlyStoppingCallback(log, target_success_rate=0.80, window=50, patience=1)
        cb.num_timesteps = 0
        result = cb._on_rollout_end()
        assert result is True   # window not full yet

    def test_stops_after_patience(self, tmp_path):
        log = str(tmp_path / "s.csv")
        self._make_csv(Path(log), [1] * 50)  # 100% success, 50 episodes
        cb = EarlyStoppingCallback(log, target_success_rate=0.80, window=50, patience=2)
        cb.num_timesteps = 0
        cb._on_rollout_end()   # count = 1
        result = cb._on_rollout_end()  # count = 2 → stop
        assert result is False

    def test_resets_streak_on_drop(self, tmp_path):
        log = str(tmp_path / "s.csv")
        self._make_csv(Path(log), [1] * 50)
        cb = EarlyStoppingCallback(log, target_success_rate=0.80, window=50, patience=3)
        cb.num_timesteps = 0
        cb._on_rollout_end()   # above: count=1
        # Rewrite CSV with low success rate
        self._make_csv(Path(log), [0] * 50)
        cb._on_rollout_end()   # below: count resets to 0
        # Rewrite with high rate again
        self._make_csv(Path(log), [1] * 50)
        cb._on_rollout_end()   # above: count=1 (not 2!)
        result = cb._on_rollout_end()  # count=2, not yet 3
        assert result is True   # patience=3, only at 2


# ------------------------------------------------------------------ #
# make_periodic_checkpoint
# ------------------------------------------------------------------ #

class TestMakePeriodicCheckpoint:
    def test_returns_checkpoint_callback(self):
        from stable_baselines3.common.callbacks import CheckpointCallback
        cb = make_periodic_checkpoint("/tmp/ckpts", save_freq=1000, prefix="test")
        assert isinstance(cb, CheckpointCallback)

    def test_default_save_freq(self):
        from stable_baselines3.common.callbacks import CheckpointCallback
        cb = make_periodic_checkpoint("/tmp/ckpts")
        assert cb.save_freq == 25_000
