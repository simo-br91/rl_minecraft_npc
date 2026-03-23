"""
curriculum_scheduler.py
-----------------------
Manages a 4-level navigation curriculum.  The difficulty level controls the
target distance range and number of 1-block wall obstacles.

Level  min_dist  max_dist  obstacles  description
-----  --------  --------  ---------  -----------
  1      3.0       6.0         0      short, flat
  2      5.0       9.0         1      medium, 1 obstacle
  3      7.0      14.0         2      long,   2 obstacles  (default difficulty)
  4     10.0      18.0         3      very long, 3 obstacles

Advancement rule: success_rate >= advance_threshold over the last
advance_window episodes triggers a move to the next level.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

# ------------------------------------------------------------------
# Level definitions
# ------------------------------------------------------------------

CURRICULUM_LEVELS: list[dict] = [
    {
        "level":        1,
        "min_dist":     3.0,
        "max_dist":     6.0,
        "num_obstacles": 0,
        "description": "short, flat",
    },
    {
        "level":        2,
        "min_dist":     5.0,
        "max_dist":     9.0,
        "num_obstacles": 1,
        "description": "medium, 1 obstacle",
    },
    {
        "level":        3,
        "min_dist":     7.0,
        "max_dist":    14.0,
        "num_obstacles": 2,
        "description": "long, 2 obstacles",
    },
    {
        "level":        4,
        "min_dist":    10.0,
        "max_dist":    18.0,
        "num_obstacles": 3,
        "description": "very long, 3 obstacles",
    },
]


# ------------------------------------------------------------------
# Scheduler
# ------------------------------------------------------------------

class CurriculumScheduler:
    """
    Track episode outcomes and advance the difficulty level when the
    rolling success rate exceeds *advance_threshold* for *advance_window*
    consecutive logged episodes.

    Parameters
    ----------
    advance_threshold : float
        Minimum rolling success rate required to advance (default 0.70).
    advance_window : int
        Number of most-recent episodes to compute the rolling rate over
        (default 20).
    start_level : int
        1-based level index to start training at (default 1).
    log_path : str | Path | None
        If given, a CSV file is written with columns
        [episode, timestep, level, success, rolling_success_rate].
    """

    def __init__(
        self,
        advance_threshold: float = 0.70,
        advance_window:    int   = 20,
        start_level:       int   = 1,
        log_path: Optional[str | Path] = None,
    ) -> None:
        assert 1 <= start_level <= len(CURRICULUM_LEVELS)
        self._level_idx         = start_level - 1
        self.advance_threshold  = advance_threshold
        self.advance_window     = advance_window
        self._recent_successes: list[float] = []
        self._total_episodes    = 0
        self._total_timesteps   = 0

        self._log_path = Path(log_path) if log_path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("w", newline="") as f:
                csv.writer(f).writerow(
                    ["episode", "timestep", "level", "success", "rolling_success_rate"]
                )

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

    def success_rate(self) -> float:
        if not self._recent_successes:
            return 0.0
        return sum(self._recent_successes) / len(self._recent_successes)

    # ------------------------------------------------------------------
    # Interface used by CurriculumEnv
    # ------------------------------------------------------------------

    def get_reset_options(self, task: str = "navigation") -> dict:
        """Return the reset *options* dict for the current level."""
        lvl = self.current_level
        return {
            "task":          task,
            "min_dist":      lvl["min_dist"],
            "max_dist":      lvl["max_dist"],
            "num_obstacles": lvl["num_obstacles"],
        }

    def record_episode(self, success: bool, timestep: int = 0) -> None:
        """
        Call once per completed episode.  Advances the level if the
        rolling success rate has crossed the threshold.
        """
        self._total_episodes  += 1
        self._total_timesteps  = timestep

        s = float(success)
        self._recent_successes.append(s)
        if len(self._recent_successes) > self.advance_window:
            self._recent_successes.pop(0)

        rate = self.success_rate()

        if self._log_path:
            with self._log_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [self._total_episodes, timestep,
                     self.level_number, int(success), round(rate, 4)]
                )

        if (
            not self.at_max_level
            and len(self._recent_successes) >= self.advance_window
            and rate >= self.advance_threshold
        ):
            self._level_idx += 1
            self._recent_successes = []
            print(
                f"[Curriculum] ▶ Advanced to level {self.level_number}: "
                f"{self.current_level['description']}  "
                f"(after ep {self._total_episodes})"
            )

    def __repr__(self) -> str:
        return (
            f"CurriculumScheduler(level={self.level_number}/4, "
            f"success_rate={self.success_rate():.2f}, "
            f"episodes={self._total_episodes})"
        )
