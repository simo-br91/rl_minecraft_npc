"""
curriculum_scheduler.py
-----------------------
Generic, levels-agnostic curriculum scheduler.

Navigation curriculum (4 levels):
  1  short flat          (3–6 b, 0 obs)
  2  medium 1 obstacle   (5–9 b, 1 obs)
  3  long 2 obstacles    (7–14 b, 2 obs)  ← default training
  4  very long 3 obs    (10–18 b, 3 obs)

Farming curriculum (4 levels):
  1  1 crop,  harvest-only (pre-grown)
  2  3 crops, harvest-only
  3  5 crops, harvest-only
  4  5 crops, full farming cycle (seeds → bonemeal → harvest)

Advancement: rolling success rate ≥ advance_threshold over advance_window episodes.
Regression : rolling success rate ≤ regress_threshold (optional).

Changes vs previous version
----------------------------
* Uses collections.deque for O(1) window update (was O(n) list.pop(0)).
* Scheduler is now generic: accepts a ``levels`` list so the same class
  serves navigation, farming, or any future task.
* ``FarmingCurriculumScheduler`` is a thin subclass that overrides
  ``get_reset_options`` for farming-specific parameters.
"""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from typing import Optional

# ------------------------------------------------------------------
# Level definitions
# ------------------------------------------------------------------

NAV_CURRICULUM_LEVELS: list[dict] = [
    {"level": 1, "min_dist":  3.0, "max_dist":  6.0, "num_obstacles": 0,
     "description": "short, flat"},
    {"level": 2, "min_dist":  5.0, "max_dist":  9.0, "num_obstacles": 1,
     "description": "medium, 1 obstacle"},
    {"level": 3, "min_dist":  7.0, "max_dist": 14.0, "num_obstacles": 2,
     "description": "long, 2 obstacles"},
    {"level": 4, "min_dist": 10.0, "max_dist": 18.0, "num_obstacles": 3,
     "description": "very long, 3 obstacles"},
]

# Backward-compat alias (old scripts used CURRICULUM_LEVELS)
CURRICULUM_LEVELS = NAV_CURRICULUM_LEVELS

FARMING_CURRICULUM_LEVELS: list[dict] = [
    {"level": 1, "num_crops": 1,  "full_farm_cycle": False,
     "description": "1 crop, harvest-only"},
    {"level": 2, "num_crops": 3,  "full_farm_cycle": False,
     "description": "3 crops, harvest-only"},
    {"level": 3, "num_crops": 5,  "full_farm_cycle": False,
     "description": "5 crops, harvest-only"},
    {"level": 4, "num_crops": 5,  "full_farm_cycle": True,
     "description": "5 crops, full farming cycle"},
]


# ------------------------------------------------------------------
# Base scheduler
# ------------------------------------------------------------------

class CurriculumScheduler:
    """
    Generic threshold-based curriculum scheduler.

    Parameters
    ----------
    levels : list[dict]
        Curriculum levels.  Each dict must have a "level" key and a
        "description" key; all other keys are forwarded to ``get_reset_options``.
    advance_threshold : float
        Minimum rolling success rate to advance (default 0.70).
    regress_threshold : float | None
        Regress one level when rolling SR drops below this (default 0.40).
        Set to None to disable regression.
    advance_window : int
        Rolling window size in episodes (default 20).
    start_level : int
        1-based starting level (default 1).
    log_path : str | Path | None
        Path for the per-episode CSV log.
    """

    def __init__(
        self,
        levels:             list[dict],
        advance_threshold:  float               = 0.70,
        regress_threshold:  Optional[float]     = 0.40,
        advance_window:     int                 = 20,
        start_level:        int                 = 1,
        log_path:           Optional[str | Path] = None,
    ) -> None:
        assert 1 <= start_level <= len(levels), \
            f"start_level {start_level} out of range [1, {len(levels)}]"
        self._levels            = levels
        self._level_idx         = start_level - 1
        self.advance_threshold  = advance_threshold
        self.regress_threshold  = regress_threshold
        self.advance_window     = advance_window
        self._recent: deque[float] = deque(maxlen=advance_window)
        self._total_episodes    = 0
        self._log_path          = Path(log_path) if log_path else None

        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("w", newline="") as f:
                csv.writer(f).writerow(
                    ["episode", "timestep", "level",
                     "success", "rolling_success_rate"])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_level(self) -> dict:
        return self._levels[self._level_idx]

    @property
    def level_number(self) -> int:
        return self._level_idx + 1

    @property
    def at_max_level(self) -> bool:
        return self._level_idx >= len(self._levels) - 1

    @property
    def at_min_level(self) -> bool:
        return self._level_idx == 0

    def success_rate(self) -> float:
        if not self._recent:
            return 0.0
        return sum(self._recent) / len(self._recent)

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def get_reset_options(self, task: str = "navigation") -> dict:
        """
        Build the options dict to pass to env.reset().
        Override in subclasses for task-specific keys.
        """
        lvl = self.current_level
        opts = {"task": task}
        # Forward all keys except meta-keys
        for k, v in lvl.items():
            if k not in ("level", "description"):
                opts[k] = v
        return opts

    def record_episode(self, success: bool, timestep: int = 0) -> None:
        """Call once per completed episode."""
        self._total_episodes += 1
        self._recent.append(float(success))
        rate = self.success_rate()

        if self._log_path:
            with self._log_path.open("a", newline="") as f:
                csv.writer(f).writerow([
                    self._total_episodes, timestep,
                    self.level_number, int(success), round(rate, 4)])

        full_window = len(self._recent) >= self.advance_window

        if not self.at_max_level and full_window and rate >= self.advance_threshold:
            self._level_idx += 1
            self._recent.clear()
            print(f"[Curriculum] ▶ Level {self.level_number}: "
                  f"{self.current_level['description']}"
                  f"  (ep {self._total_episodes})")

        elif (self.regress_threshold is not None
              and not self.at_min_level
              and full_window
              and rate < self.regress_threshold):
            self._level_idx -= 1
            self._recent.clear()
            print(f"[Curriculum] ◀ Regressed to level {self.level_number}: "
                  f"{self.current_level['description']}"
                  f"  (ep {self._total_episodes})")

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(level={self.level_number}/{len(self._levels)}, "
                f"sr={self.success_rate():.2f}, ep={self._total_episodes})")


# ------------------------------------------------------------------
# Navigation convenience class (default levels)
# ------------------------------------------------------------------

class NavCurriculumScheduler(CurriculumScheduler):
    """Curriculum scheduler pre-configured for navigation."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("levels", NAV_CURRICULUM_LEVELS)
        super().__init__(**kwargs)

    def get_reset_options(self, task: str = "navigation") -> dict:
        return super().get_reset_options(task)


# ------------------------------------------------------------------
# Farming curriculum scheduler
# ------------------------------------------------------------------

class FarmingCurriculumScheduler(CurriculumScheduler):
    """Curriculum scheduler pre-configured for farming."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("levels", FARMING_CURRICULUM_LEVELS)
        super().__init__(**kwargs)

    def get_reset_options(self, task: str = "farming") -> dict:
        lvl = self.current_level
        return {
            "task":           task,
            "num_crops":      lvl["num_crops"],
            "full_farm_cycle": lvl["full_farm_cycle"],
        }