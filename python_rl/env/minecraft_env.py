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

Observation Space (28 dims) — see ObservationBuilder.java for full docs.

Fixes applied vs previous version
----------------------------------
* obs_space high bound for task_id was 10 (should be 1); all bounds verified.
* close() sends a proper cleanup reset so leftover markers/NPCs don't remain.
* render() is properly implemented as a no-op with the correct signature.
* Seed is forwarded to the Java bridge so episode randomisation is
  reproducible when a seed is provided.
* sample_tasks uses the seeded _np_random rng (was accidentally using global
  numpy random state).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import gymnasium as gym
from gymnasium import spaces


# Named action indices for readability
ACTION_FORWARD     = 0
ACTION_TURN_LEFT   = 1
ACTION_TURN_RIGHT  = 2
ACTION_INTERACT    = 3
ACTION_NOOP        = 4
ACTION_JUMP        = 5
ACTION_SPRINT      = 6
ACTION_BACKWARD    = 7
ACTION_STRAFE_LEFT = 8
ACTION_STRAFE_RIGHT= 9
ACTION_ATTACK      = 10
ACTION_EAT         = 11
ACTION_SWITCH_ITEM = 12

OBS_DIM = 29


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
        Sampling uses the seeded internal RNG.
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
        self.base_url        = base_url
        self.fixed_task      = task
        self.sample_tasks    = sample_tasks
        self.num_crops       = num_crops
        self.full_farm_cycle = full_farm_cycle
        self.retry_attempts  = retry_attempts
        self._np_random      = np.random.default_rng()

        # 13 discrete actions
        self.action_space = spaces.Discrete(13)

        # 28-dim observation — all elements in [-1, 1] or [0, 1].
        # Index reference (must match ObservationBuilder.java):
        #  0  dx_norm          1  dz_norm         2  distance_norm
        #  3  angle_to_target  4  yaw_norm         5  pitch_norm
        #  6  blocked_front    7  on_ground        8  stuck_norm
        #  9  task_id_norm    10  crop_in_front   11  near_crop
        # 12  obstacle_1block 13  health_norm     14  food_norm
        # 15  mob_nearby      16  mob_dist_norm   17  mob_angle
        # 18  active_slot_norm 19 holding_sword   20  holding_food
        # 21  crops_remaining 22  block_N         23  block_E
        # 24  block_S         25  block_W         26  farmland_ahead
        # 27  has_seed        28  height_above_gnd
        low  = np.array(
            [-1,-1, 0,-1,-1,-1, 0,0,0,0, 0,0,0, 0,0, 0,0,-1, 0,0,0, 0, 0,0,0,0, 0,0, 0],
            dtype=np.float32)
        high = np.array(
            [ 1, 1, 1, 1, 1, 1, 1,1,1,1, 1,1,1, 1,1, 1,1, 1, 1,1,1, 1, 1,1,1,1, 1,1, 1],
            dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Validate server is reachable before anything else
        self._wait_for_server()

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        seed:    Optional[int]            = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        opts = options or {}
        task            = opts.get("task",           self._choose_task())
        sparse_reward   = bool(opts.get("sparse_reward",   False))
        min_dist        = float(opts.get("min_dist",       -1.0))
        max_dist        = float(opts.get("max_dist",       -1.0))
        num_obstacles   = int(opts.get("num_obstacles",    -1))
        num_crops       = int(opts.get("num_crops",        self.num_crops))
        full_farm_cycle = bool(opts.get("full_farm_cycle", self.full_farm_cycle))

        payload: Dict[str, Any] = {
            "task":            task,
            "sparse_reward":   sparse_reward,
            "num_crops":       num_crops,
            "full_farm_cycle": full_farm_cycle,
        }
        # Forward seed so Java RNG is deterministic when provided
        if seed is not None:
            payload["seed"] = seed
        if min_dist >= 0:
            payload["min_dist"]      = min_dist
        if max_dist >= 0:
            payload["max_dist"]      = max_dist
        if num_obstacles >= 0:
            payload["num_obstacles"] = num_obstacles

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
    # render() — no-op (Minecraft IS the renderer)
    # ------------------------------------------------------------------

    def render(self) -> None:
        """
        No-op.  The Minecraft client window is the renderer.
        render_mode="human" is implicitly satisfied by Minecraft being open.
        """
        pass

    # ------------------------------------------------------------------
    # close() — send a cleanup reset to leave Minecraft in a clean state
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Send a cleanup /reset so Minecraft removes the NPC markers and
        leftover blocks from the last episode.  Swallows all errors —
        Minecraft may already be closed.
        """
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
        if self.sample_tasks:
            # Use the seeded internal RNG — reproducible when seed is set
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
                time.sleep(0.5 * (2 ** attempt))
            except requests.RequestException as e:
                last_exc = e
                time.sleep(0.2)
        raise RuntimeError(
            f"Bridge request to {path} failed after "
            f"{self.retry_attempts} attempts: {last_exc}"
        )

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