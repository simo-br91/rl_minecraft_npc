"""
curriculum_scheduler.py
-----------------------
Manages a 4-level navigation curriculum with:
 - O(1) window update via collections.deque
 - Optional difficulty regression when rolling SR drops below regress_threshold
 - Per-level logging to CSV

Levels:
  Level  min_dist  max_dist  obstacles
  -----  --------  --------  ---------
    1     3–6 b       0
    2     5–9 b       1
    3     7–14 b      2       (default training)
    4    10–18 b      3
"""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from typing import Optional

CURRICULUM_LEVELS: list[dict] = [
    {"level": 1, "min_dist":  3.0, "max_dist":  6.0, "num_obstacles": 0, "description": "short, flat"},
    {"level": 2, "min_dist":  5.0, "max_dist":  9.0, "num_obstacles": 1, "description": "medium, 1 obstacle"},
    {"level": 3, "min_dist":  7.0, "max_dist": 14.0, "num_obstacles": 2, "description": "long, 2 obstacles"},
    {"level": 4, "min_dist": 10.0, "max_dist": 18.0, "num_obstacles": 3, "description": "very long, 3 obstacles"},
]


class CurriculumScheduler:
    """
    Parameters
    ----------
    advance_threshold : float
        Minimum rolling success rate to advance (default 0.70).
    regress_threshold : float | None
        If rolling SR drops below this after advancing, regress one level.
        None = no regression (default).
    advance_window : int
        Rolling window size in episodes (default 20).
    start_level : int
        1-based starting level (default 1).
    log_path : str | Path | None
        CSV log path.
    """

    def __init__(
        self,
        advance_threshold:  float               = 0.70,
        regress_threshold:  Optional[float]     = 0.40,
        advance_window:     int                 = 20,
        start_level:        int                 = 1,
        log_path:           Optional[str | Path] = None,
    ) -> None:
        assert 1 <= start_level <= len(CURRICULUM_LEVELS)
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
                    ["episode", "timestep", "level", "success", "rolling_success_rate"])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_level(self) -> dict:
        return CURRICULUM_LEVELS[self._level_idx]

    @property
    def level_number(self) -> int:
        return self._level_idx + 1

    @property
    def at_max_level(self) -> bool:
        return self._level_idx >= len(CURRICULUM_LEVELS) - 1

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
        lvl = self.current_level
        return {
            "task":          task,
            "min_dist":      lvl["min_dist"],
            "max_dist":      lvl["max_dist"],
            "num_obstacles": lvl["num_obstacles"],
        }

    def record_episode(self, success: bool, timestep: int = 0) -> None:
        """Call once per completed episode."""
        self._total_episodes += 1
        self._recent.append(float(success))
        rate = self.success_rate()

        if self._log_path:
            with self._log_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [self._total_episodes, timestep, self.level_number, int(success), round(rate, 4)])

        full_window = len(self._recent) >= self.advance_window

        # Advance
        if (not self.at_max_level and full_window and rate >= self.advance_threshold):
            self._level_idx += 1
            self._recent.clear()
            print(f"[Curriculum] ▶ Level {self.level_number}: "
                  f"{self.current_level['description']}  (ep {self._total_episodes})")

        # Regress
        elif (self.regress_threshold is not None
              and not self.at_min_level
              and full_window
              and rate < self.regress_threshold):
            self._level_idx -= 1
            self._recent.clear()
            print(f"[Curriculum] ◀ Regressed to level {self.level_number}: "
                  f"{self.current_level['description']}  (ep {self._total_episodes})")

    def __repr__(self) -> str:
        return (f"CurriculumScheduler(level={self.level_number}/4, "
                f"sr={self.success_rate():.2f}, ep={self._total_episodes})")