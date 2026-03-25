"""
minecraft_env.py
----------------
Gymnasium wrapper over the Minecraft Forge HTTP bridge.

Action Space (13 discrete actions):
    0  forward (walk)
    1  turn_left   (-15° yaw)
    2  turn_right  (+15° yaw)
    3  interact    (harvest crop / use bonemeal / till soil)
    4  no_op
    5  jump        (1-block teleport up over obstacle)
    6  sprint_forward
    7  move_backward
    8  strafe_left
    9  strafe_right
   10  attack      (melee swing at nearest hostile mob)
   11  eat         (consume cooked beef from inventory)
   12  switch_item (cycle active hotbar slot)

Observation Space (104 dims) — see ObservationBuilder.java for full layout.
  Indices  0-28: hand-crafted features (nav, task, farming, combat, obstacles)
  Indices 29-103: local 5×3×5 voxel grid (solid=1, passable=0)

Action Masking (Issue 6.3):
  MaskableMinecraftEnv subclass exposes action_masks() compatible with
  sb3-contrib MaskablePPO.  The Java bridge provides /masks (GET) returning
  a 13-element validity array.  This prevents the policy from sampling
  known-invalid actions, speeding up convergence.

Fixes vs previous version
--------------------------
* sample_tasks now uses the gymnasium-seeded internal RNG (_np_random) so
  task sampling is reproducible when a seed is passed to reset().
  (Bug 6.7 — was using numpy global random state)
* Full type annotations added throughout. (Issue 8.1)
* OBS_DIM updated to 104 to include the voxel grid. (Req 1.1)
* Seed is forwarded to the Java bridge for reproducible Java RNG. (Req 3.7)
* action_masks() added for MaskablePPO support. (Issue 6.3)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import gymnasium as gym
from gymnasium import spaces


# Named action indices for readability
ACTION_FORWARD      = 0
ACTION_TURN_LEFT    = 1
ACTION_TURN_RIGHT   = 2
ACTION_INTERACT     = 3
ACTION_NOOP         = 4
ACTION_JUMP         = 5
ACTION_SPRINT       = 6
ACTION_BACKWARD     = 7
ACTION_STRAFE_LEFT  = 8
ACTION_STRAFE_RIGHT = 9
ACTION_ATTACK       = 10
ACTION_EAT          = 11
ACTION_SWITCH_ITEM  = 12

OBS_DIM = 104   # 29 hand-crafted + 75 voxel grid (5×3×5)


class MinecraftEnv(gym.Env):
    """
    Gymnasium environment wrapping the Minecraft Forge HTTP bridge.

    Parameters
    ----------
    base_url : str
        URL of the bridge server (default: http://127.0.0.1:8765).
    task : str
        Default task for this env instance.
        One of: "navigation", "farming", "combat", "multitask".
    sample_tasks : list[str] | None
        If set, task is sampled from this list each reset.
        Uses the seeded internal RNG — reproducible when seed is passed.
    num_crops : int
        Default number of wheat plots for farming episodes (1–10).
    full_farm_cycle : bool
        If True crops start as seeds and need bonemeal/time to grow.
    retry_attempts : int
        HTTP retry attempts on timeout before raising (default 3).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        base_url:        str                  = "http://127.0.0.1:8765",
        task:            str                  = "navigation",
        sample_tasks:    Optional[List[str]]  = None,
        num_crops:       int                  = 5,
        full_farm_cycle: bool                 = False,
        retry_attempts:  int                  = 3,
    ) -> None:
        super().__init__()
        self.base_url:        str                 = base_url
        self.fixed_task:      str                 = task
        self.sample_tasks:    Optional[List[str]] = sample_tasks
        self.num_crops:       int                 = num_crops
        self.full_farm_cycle: bool                = full_farm_cycle
        self.retry_attempts:  int                 = retry_attempts
        # Internal RNG — seeded via reset(seed=...) for reproducibility
        self._np_random: np.random.Generator = np.random.default_rng()

        # 13 discrete actions
        self.action_space: spaces.Discrete = spaces.Discrete(13)

        # 104-dim observation vector — indices 0-28 hand-crafted, 29-103 voxel grid
        # All elements normalised to [-1, 1] or [0, 1].
        low  = np.array(
            # 0    1    2    3    4    5    6  7  8  9   10  11  12  13  14
            [-1., -1.,  0., -1., -1., -1.,  0.,0.,0.,0., 0., 0., 0., 0., 0.,
            # 15   16   17   18   19   20   21  22  23  24  25  26  27  28
              0.,  0.,  0.,  0.,  0.,  0., -1., 0., 0., 0., 0., 0., 0., 0.]
            + [0.] * 75,   # voxel grid
            dtype=np.float32,
        )
        high = np.ones(OBS_DIM, dtype=np.float32)
        self.observation_space: spaces.Box = spaces.Box(
            low=low, high=high, dtype=np.float32
        )

        self._wait_for_server()

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        seed:    Optional[int]            = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            # FIX 6.7: seed the internal RNG used by _choose_task
            self._np_random = np.random.default_rng(seed)

        opts: Dict[str, Any] = options or {}
        task:            str   = opts.get("task",           self._choose_task())
        sparse_reward:   bool  = bool(opts.get("sparse_reward",   False))
        min_dist:        float = float(opts.get("min_dist",       -1.0))
        max_dist:        float = float(opts.get("max_dist",       -1.0))
        num_obstacles:   int   = int(opts.get("num_obstacles",    -1))
        num_crops:       int   = int(opts.get("num_crops",        self.num_crops))
        full_farm_cycle: bool  = bool(opts.get("full_farm_cycle", self.full_farm_cycle))
        # Fix 4.5: combat curriculum parameters (-1 / -1.0 = server defaults)
        num_mobs:        int   = int(opts.get("num_mobs",         -1))
        mob_dist_min:    float = float(opts.get("mob_dist_min",   -1.0))
        mob_dist_max:    float = float(opts.get("mob_dist_max",   -1.0))

        payload: Dict[str, Any] = {
            "task":            task,
            "sparse_reward":   sparse_reward,
            "num_crops":       num_crops,
            "full_farm_cycle": full_farm_cycle,
        }
        # FIX 3.7: forward seed to Java bridge for reproducible RNG
        if seed is not None:
            payload["seed"] = seed
        if min_dist >= 0:
            payload["min_dist"]      = min_dist
        if max_dist >= 0:
            payload["max_dist"]      = max_dist
        if num_obstacles >= 0:
            payload["num_obstacles"] = num_obstacles
        # Fix 4.5: only send combat curriculum params when explicitly set
        if num_mobs >= 0:
            payload["num_mobs"]      = num_mobs
        if mob_dist_min >= 0:
            payload["mob_dist_min"]  = mob_dist_min
        if mob_dist_max >= 0:
            payload["mob_dist_max"]  = mob_dist_max

        data = self._post("/reset", payload, timeout=15)
        if "error" in data:
            raise RuntimeError(data["error"])

        obs  = np.clip(
            np.array(data["obs"], dtype=np.float32),
            self.observation_space.low,
            self.observation_space.high,
        )
        info: Dict[str, Any] = data.get("info", {})
        return obs, info

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        data = self._post("/step", {"action": int(action)}, timeout=10)
        if "error" in data:
            raise RuntimeError(data["error"])

        obs       = np.clip(
            np.array(data["obs"], dtype=np.float32),
            self.observation_space.low,
            self.observation_space.high,
        )
        reward:    float             = float(data["reward"])
        done:      bool              = bool(data["done"])
        truncated: bool              = bool(data["truncated"])
        info:      Dict[str, Any]    = data.get("info", {})
        return obs, reward, done, truncated, info

    # ------------------------------------------------------------------
    # render() — no-op (Minecraft IS the renderer)
    # ------------------------------------------------------------------

    def render(self) -> None:
        """No-op. The Minecraft client window is the renderer."""
        pass

    # ------------------------------------------------------------------
    # close()
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Send a cleanup reset so Minecraft removes NPC markers."""
        try:
            self._post(
                "/reset",
                {"task": "navigation", "min_dist": 1.0, "max_dist": 2.0,
                 "num_obstacles": 0, "num_crops": 0},
                timeout=5,
            )
        except Exception:
            pass
        super().close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _choose_task(self) -> str:
        """
        Return the task name for the next episode.
        FIX 6.7: uses the seeded internal RNG (_np_random) instead of
        numpy's global random state, so task sampling is reproducible.
        """
        if self.sample_tasks:
            idx = int(self._np_random.integers(0, len(self.sample_tasks)))
            return str(self.sample_tasks[idx])
        return self.fixed_task

    def _post(
        self, path: str, payload: Dict[str, Any], timeout: float = 10
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None
        for attempt in range(self.retry_attempts):
            try:
                r = requests.post(url, json=payload, timeout=timeout)
                r.raise_for_status()
                return r.json()
            except requests.Timeout as e:
                last_exc = e
                time.sleep(0.5 * (2 ** attempt))
            except requests.RequestException as e:
                last_exc = e
                time.sleep(0.2)
        raise RuntimeError(
            f"Bridge request to {path} failed after "
            f"{self.retry_attempts} attempts: {last_exc}"
        )

    def action_masks(self) -> np.ndarray:
        """
        Return a boolean mask of shape (13,) indicating valid actions.

        Compatible with sb3-contrib MaskablePPO — call this from the
        MaskableMinecraftEnv subclass or pass it to the policy directly.

        Queries the Java bridge /masks endpoint (GET).  Falls back to
        all-valid mask on any error so training is never blocked.
        """
        try:
            r = requests.get(f"{self.base_url}/masks", timeout=3)
            r.raise_for_status()
            masks: List[int] = r.json().get("masks", [1] * 13)
            return np.array(masks, dtype=bool)
        except Exception:
            return np.ones(13, dtype=bool)

    def _wait_for_server(
        self, max_wait: float = 60.0, interval: float = 2.0
    ) -> None:
        """Block until the bridge server responds to /health."""
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.base_url}/health", timeout=3)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(interval)
        raise RuntimeError(
            f"Minecraft bridge at {self.base_url} did not respond within "
            f"{max_wait}s.  Make sure Minecraft is running with the rlnpc mod."
        )


# ---------------------------------------------------------------------------
# MaskableMinecraftEnv — drop-in for sb3-contrib MaskablePPO (Issue 6.3)
# ---------------------------------------------------------------------------

class MaskableMinecraftEnv(MinecraftEnv):
    """
    Subclass of MinecraftEnv that exposes action_masks() in the format
    expected by sb3-contrib MaskablePPO.

    Usage
    -----
        from sb3_contrib import MaskablePPO
        from python_rl.env.minecraft_env import MaskableMinecraftEnv

        env = MaskableMinecraftEnv(task="navigation")
        model = MaskablePPO("MlpPolicy", env, ...)
        model.learn(...)

    The mask is fetched from the Java bridge on every step via /masks.
    It mirrors ActionExecutor preconditions so the policy never wastes
    samples on actions like interact (when no crop is in front) or attack
    (when no mobs are nearby).
    """

    def action_masks(self) -> np.ndarray:
        """Return boolean validity mask, shape (13,)."""
        return super().action_masks()
