"""
tests/test_curriculum_scheduler.py
------------------------------------
Unit tests for CurriculumScheduler, NavCurriculumScheduler,
and FarmingCurriculumScheduler.

Run with:
    python -m pytest tests/test_curriculum_scheduler.py -v
"""

import csv
import tempfile
from pathlib import Path

import pytest

import python_rl.train.curriculum_scheduler as _sched_mod
from python_rl.train.curriculum_scheduler import (
    CurriculumScheduler,
    NavCurriculumScheduler,
    FarmingCurriculumScheduler,
    NAV_CURRICULUM_LEVELS,
    FARMING_CURRICULUM_LEVELS,
)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def make_scheduler(**kwargs) -> CurriculumScheduler:
    """Create a NavCurriculumScheduler with sensible test defaults."""
    kwargs.setdefault("advance_threshold", 0.70)
    kwargs.setdefault("regress_threshold", 0.40)
    kwargs.setdefault("advance_window", 5)
    return NavCurriculumScheduler(**kwargs)


def feed(scheduler: CurriculumScheduler, pattern: list[bool]) -> None:
    """Feed a sequence of success/failure outcomes to the scheduler."""
    for i, success in enumerate(pattern):
        scheduler.record_episode(success=success, timestep=i)


# ------------------------------------------------------------------ #
# Level definitions
# ------------------------------------------------------------------ #

class TestLevelDefinitions:
    def test_nav_has_four_levels(self):
        assert len(NAV_CURRICULUM_LEVELS) == 4

    def test_nav_levels_have_required_keys(self):
        required = {"level", "min_dist", "max_dist", "num_obstacles", "description"}
        for lvl in NAV_CURRICULUM_LEVELS:
            assert required.issubset(lvl.keys()), f"Missing keys in {lvl}"

    def test_nav_levels_increasing_distance(self):
        for a, b in zip(NAV_CURRICULUM_LEVELS, NAV_CURRICULUM_LEVELS[1:]):
            assert b["min_dist"] >= a["min_dist"]
            assert b["max_dist"] >= a["max_dist"]

    def test_nav_levels_non_decreasing_obstacles(self):
        obs = [lvl["num_obstacles"] for lvl in NAV_CURRICULUM_LEVELS]
        assert obs == sorted(obs)

    def test_farming_has_four_levels(self):
        assert len(FARMING_CURRICULUM_LEVELS) == 4

    def test_farming_levels_have_required_keys(self):
        required = {"level", "num_crops", "full_farm_cycle", "description"}
        for lvl in FARMING_CURRICULUM_LEVELS:
            assert required.issubset(lvl.keys())

    def test_backward_compat_alias(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            alias = _sched_mod.CURRICULUM_LEVELS
        assert alias is NAV_CURRICULUM_LEVELS
        assert any("CURRICULUM_LEVELS is deprecated" in str(x.message) for x in w)


# ------------------------------------------------------------------ #
# CurriculumScheduler — basic properties
# ------------------------------------------------------------------ #

class TestSchedulerProperties:
    def test_starts_at_level_1(self):
        s = make_scheduler()
        assert s.level_number == 1

    def test_starts_at_custom_level(self):
        s = make_scheduler(start_level=3)
        assert s.level_number == 3

    def test_invalid_start_level_raises(self):
        with pytest.raises(AssertionError):
            make_scheduler(start_level=0)
        with pytest.raises(AssertionError):
            make_scheduler(start_level=99)

    def test_at_min_level_initially(self):
        s = make_scheduler()
        assert s.at_min_level

    def test_not_at_max_level_initially(self):
        s = make_scheduler()
        assert not s.at_max_level

    def test_at_max_level_when_at_last(self):
        s = make_scheduler(start_level=4)
        assert s.at_max_level

    def test_current_level_matches_level_number(self):
        s = make_scheduler()
        assert s.current_level["level"] == s.level_number

    def test_success_rate_zero_initially(self):
        s = make_scheduler()
        assert s.success_rate() == 0.0

    def test_repr_contains_level(self):
        s = make_scheduler()
        r = repr(s)
        assert "level=1" in r


# ------------------------------------------------------------------ #
# Advancement logic
# ------------------------------------------------------------------ #

class TestAdvancement:
    def test_no_advance_below_threshold(self):
        s = make_scheduler(advance_threshold=0.70, advance_window=5)
        # 3/5 = 0.60 — below threshold
        feed(s, [True, True, True, False, False])
        assert s.level_number == 1

    def test_advance_at_threshold(self):
        s = make_scheduler(advance_threshold=0.60, advance_window=5)
        # 3/5 = 0.60 — exactly at threshold
        feed(s, [True, True, True, False, False])
        assert s.level_number == 2

    def test_advance_above_threshold(self):
        s = make_scheduler(advance_threshold=0.70, advance_window=5)
        feed(s, [True, True, True, True, False])  # 4/5 = 0.80
        assert s.level_number == 2

    def test_no_advance_before_window_full(self):
        s = make_scheduler(advance_threshold=0.70, advance_window=5)
        # Only 4 episodes fed — window not yet full
        feed(s, [True, True, True, True])
        assert s.level_number == 1

    def test_window_resets_after_advance(self):
        s = make_scheduler(advance_threshold=0.70, advance_window=5)
        feed(s, [True] * 5)  # advance to level 2
        assert s.level_number == 2
        # Window should be cleared; 4 more wins shouldn't advance yet
        feed(s, [True] * 4)
        assert s.level_number == 2

    def test_no_advance_past_max_level(self):
        s = make_scheduler(advance_threshold=0.70, advance_window=5, start_level=4)
        feed(s, [True] * 10)
        assert s.level_number == 4

    def test_advance_through_all_levels(self):
        s = make_scheduler(advance_threshold=0.70, advance_window=5)
        for expected_level in range(1, 4):
            assert s.level_number == expected_level
            feed(s, [True] * 5)
        assert s.level_number == 4


# ------------------------------------------------------------------ #
# Regression logic
# ------------------------------------------------------------------ #

class TestRegression:
    def test_regression_below_threshold(self):
        s = make_scheduler(advance_threshold=0.70, regress_threshold=0.40, advance_window=5)
        feed(s, [True] * 5)   # advance to level 2
        assert s.level_number == 2
        feed(s, [False] * 5)  # 0/5 = 0.0 < 0.40 → regress
        assert s.level_number == 1

    def test_no_regression_at_threshold(self):
        s = make_scheduler(advance_threshold=0.70, regress_threshold=0.40, advance_window=5)
        feed(s, [True] * 5)
        # 2/5 = 0.40 exactly — should NOT regress (strictly less than)
        feed(s, [True, True, False, False, False])
        assert s.level_number == 2

    def test_no_regression_at_min_level(self):
        s = make_scheduler(regress_threshold=0.40, advance_window=5)
        feed(s, [False] * 5)  # already at level 1 — can't go lower
        assert s.level_number == 1

    def test_regression_disabled_when_none(self):
        s = make_scheduler(regress_threshold=None, advance_window=5)
        feed(s, [True] * 5)   # advance
        feed(s, [False] * 5)  # should NOT regress
        assert s.level_number == 2


# ------------------------------------------------------------------ #
# CSV logging
# ------------------------------------------------------------------ #

class TestCSVLogging:
    def test_creates_csv_with_header(self, tmp_path):
        log = str(tmp_path / "log.csv")
        s = make_scheduler(log_path=log)
        assert Path(log).exists()
        with open(log) as f:
            header = f.readline().strip()
        assert "episode" in header
        assert "level" in header
        assert "success" in header

    def test_logs_each_episode(self, tmp_path):
        log = str(tmp_path / "log.csv")
        s = make_scheduler(log_path=log)
        feed(s, [True, False, True])
        with open(log) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert rows[0]["success"] == "1"
        assert rows[1]["success"] == "0"
        assert rows[2]["success"] == "1"

    def test_logs_correct_level(self, tmp_path):
        log = str(tmp_path / "log.csv")
        s = make_scheduler(advance_threshold=0.70, advance_window=5, log_path=log)
        feed(s, [True] * 5)   # advance at episode 5
        feed(s, [True])        # episode 6 — should be level 2
        with open(log) as f:
            rows = list(csv.DictReader(f))
        assert rows[4]["level"] == "1"   # still level 1 at episode 5
        assert rows[5]["level"] == "2"   # advanced to level 2

    def test_no_log_file_when_path_none(self):
        s = make_scheduler(log_path=None)
        feed(s, [True, False])
        # No file created — just verify no exception


# ------------------------------------------------------------------ #
# get_reset_options
# ------------------------------------------------------------------ #

class TestGetResetOptions:
    def test_nav_reset_options_level_1(self):
        s = NavCurriculumScheduler()
        opts = s.get_reset_options(task="navigation")
        assert opts["task"] == "navigation"
        assert opts["min_dist"] == 3.0
        assert opts["max_dist"] == 6.0
        assert opts["num_obstacles"] == 0

    def test_nav_reset_options_level_4(self):
        s = NavCurriculumScheduler(start_level=4)
        opts = s.get_reset_options()
        assert opts["min_dist"] == 10.0
        assert opts["max_dist"] == 18.0
        assert opts["num_obstacles"] == 3

    def test_nav_options_no_meta_keys(self):
        s = NavCurriculumScheduler()
        opts = s.get_reset_options()
        assert "level" not in opts
        assert "description" not in opts

    def test_farming_reset_options_level_1(self):
        s = FarmingCurriculumScheduler()
        opts = s.get_reset_options(task="farming")
        assert opts["task"] == "farming"
        assert opts["num_crops"] == 1
        assert opts["full_farm_cycle"] is False

    def test_farming_reset_options_level_4(self):
        s = FarmingCurriculumScheduler(start_level=4)
        opts = s.get_reset_options()
        assert opts["num_crops"] == 5
        assert opts["full_farm_cycle"] is True


# ------------------------------------------------------------------ #
# deque / O(1) window
# ------------------------------------------------------------------ #

class TestDequeWindow:
    def test_window_size_respected(self):
        s = make_scheduler(advance_threshold=0.70, advance_window=3)
        # Feed 10 failures then 3 successes — window should only see last 3
        feed(s, [False] * 10)
        assert s.success_rate() == 0.0
        feed(s, [True, True, True])
        assert s.success_rate() == 1.0

    def test_rolling_rate_updates_correctly(self):
        s = make_scheduler(advance_threshold=0.70, advance_window=4)
        feed(s, [True, True, False, False])  # 2/4 = 0.50
        assert abs(s.success_rate() - 0.50) < 1e-9
        feed(s, [True])   # window now: [True, False, False, True] = 0.50
        assert abs(s.success_rate() - 0.50) < 1e-9
