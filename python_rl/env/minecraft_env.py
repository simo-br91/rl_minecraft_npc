"""
minecraft_env.py
----------------
Gymnasium wrapper over the Minecraft Forge HTTP bridge.

Action Space (13 discrete actions):
    0  forward (walk)
    1  turn_left   (-15° yaw)
    2  turn_right  (+15° yaw)
    3  interact    (harvest / bonemeal / till)
    4  no_op
    5  jump        (1-block teleport up)
    6  sprint_forward
    7  move_backward
    8  strafe_left
    9  strafe_right
   10  attack      (melee swing)
   11  eat         (consume food)
   12  switch_item (cycle active slot)

Observation Space (28 dims) — see ObservationBuilder.java for full docs.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import gymnasium as gym
from gymnasium import spaces


# Name→index for readability
ACTION_FORWARD       = 0
ACTION_TURN_LEFT     = 1
ACTION_TURN_RIGHT    = 2
ACTION_INTERACT      = 3
ACTION_NOOP          = 4
ACTION_JUMP          = 5
ACTION_SPRINT        = 6
ACTION_BACKWARD      = 7
ACTION_STRAFE_LEFT   = 8
ACTION_STRAFE_RIGHT  = 9
ACTION_ATTACK        = 10
ACTION_EAT           = 11
ACTION_SWITCH_ITEM   = 12

OBS_DIM = 28


class MinecraftEnv(gym.Env):
    """
    Gymnasium environment wrapping the Minecraft Forge HTTP bridge.

    Parameters
    ----------
    base_url : str
        URL of the bridge server (default: http://127.0.0.1:8765).
    task : str
        Default task: "navigation", "farming", "combat", "multitask".
    sample_tasks : list[str] | None
        If set, task is sampled from this list each episode.
    num_crops : int
        Number of wheat plots per farming episode (1–10, default 5).
    full_farm_cycle : bool
        If True, crops start as seeds and must be bonemeal'd / waited on.
    retry_attempts : int
        HTTP retry attempts on timeout/error before raising.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_url:        str             = "http://127.0.0.1:8765",
        task:            str             = "navigation",
        sample_tasks:    Optional[List[str]] = None,
        num_crops:       int             = 5,
        full_farm_cycle: bool            = False,
        retry_attempts:  int             = 3,
    ) -> None:
        super().__init__()
        self.base_url        = base_url
        self.fixed_task      = task
        self.sample_tasks    = sample_tasks
        self.num_crops       = num_crops
        self.full_farm_cycle = full_farm_cycle
        self.retry_attempts  = retry_attempts
        self._np_random      = np.random.default_rng()

        # 13 discrete actions
        self.action_space = spaces.Discrete(13)

        # 28-dim observation — bounds are generous to avoid clipping
        low  = np.array([-1,-1, 0,-1,-1,-1, 0,0,0,0, 0,0,0, 0,0, 0,0,-1, 0,0,0, 0, 0,0,0,0, 0,0], dtype=np.float32)
        high = np.array([ 1, 1, 1, 1, 1, 1, 1,1,1,1, 1,1,1, 1,1, 1,1, 1, 1,1,1, 1, 1,1,1,1, 1,1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Validate connection
        self._wait_for_server()

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        seed:    Optional[int]       = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        # seed numpy rng but don't forward to Java (not deterministic)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        opts = options or {}
        task            = opts.get("task", self._choose_task())
        sparse_reward   = bool(opts.get("sparse_reward", False))
        min_dist        = float(opts.get("min_dist", -1.0))
        max_dist        = float(opts.get("max_dist", -1.0))
        num_obstacles   = int(opts.get("num_obstacles", -1))
        num_crops       = int(opts.get("num_crops", self.num_crops))
        full_farm_cycle = bool(opts.get("full_farm_cycle", self.full_farm_cycle))

        payload: Dict = {
            "task":           task,
            "sparse_reward":  sparse_reward,
            "num_crops":      num_crops,
            "full_farm_cycle": full_farm_cycle,
        }
        if min_dist >= 0:      payload["min_dist"]      = min_dist
        if max_dist >= 0:      payload["max_dist"]       = max_dist
        if num_obstacles >= 0: payload["num_obstacles"]  = num_obstacles

        data = self._post("/reset", payload, timeout=15)
        if "error" in data:
            raise RuntimeError(data["error"])

        obs  = np.clip(np.array(data["obs"], dtype=np.float32),
                       self.observation_space.low, self.observation_space.high)
        info = data.get("info", {})
        return obs, info

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        data = self._post("/step", {"action": int(action)}, timeout=10)
        if "error" in data:
            raise RuntimeError(data["error"])

        obs       = np.clip(np.array(data["obs"], dtype=np.float32),
                            self.observation_space.low, self.observation_space.high)
        reward    = float(data["reward"])
        done      = bool(data["done"])
        truncated = bool(data["truncated"])
        info      = data.get("info", {})
        return obs, reward, done, truncated, info

    # ------------------------------------------------------------------
    # render / close
    # ------------------------------------------------------------------

    def render(self) -> None:
        # No-op: Minecraft is the renderer
        pass

    def close(self) -> None:
        # Send a clean no-op step to leave the server in a stable state
        try:
            self._post("/step", {"action": ACTION_NOOP}, timeout=3)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _choose_task(self) -> str:
        if self.sample_tasks:
            idx = int(self._np_random.integers(0, len(self.sample_tasks)))
            return str(self.sample_tasks[idx])
        return self.fixed_task

    def _post(self, path: str, payload: Dict, timeout: float = 10) -> Dict:
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None
        for attempt in range(self.retry_attempts):
            try:
                r = requests.post(url, json=payload, timeout=timeout)
                r.raise_for_status()
                return r.json()
            except requests.Timeout as e:
                last_exc = e
                delay = 0.5 * (2 ** attempt)
                time.sleep(delay)
            except requests.RequestException as e:
                last_exc = e
                time.sleep(0.2)
        raise RuntimeError(f"Bridge request to {path} failed after "
                           f"{self.retry_attempts} attempts: {last_exc}")

    def _wait_for_server(self, max_wait: float = 60.0, interval: float = 2.0) -> None:
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
            f"{max_wait}s. Make sure Minecraft is running with the rlnpc mod."
        )