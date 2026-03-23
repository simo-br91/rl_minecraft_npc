import numpy as np
import requests
import gymnasium as gym
from gymnasium import spaces


class MinecraftEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, base_url="http://127.0.0.1:8765", task="navigation", sample_tasks=None):
        super().__init__()
        self.base_url     = base_url
        self.fixed_task   = task
        self.sample_tasks = sample_tasks

        # Actions: 0 forward, 1 turn_left, 2 turn_right, 3 interact, 4 no_op, 5 jump
        self.action_space = spaces.Discrete(6)

        # 11 features — must match ObservationBuilder.build() on the Java side:
        #   0  dx
        #   1  dz
        #   2  distance
        #   3  yaw_norm
        #   4  blocked_front
        #   5  on_ground
        #   6  stuck_norm
        #   7  task_id
        #   8  crop_in_front
        #   9  near_target
        #  10  obstacle_1block_ahead
        self.observation_space = spaces.Box(
            low=np.array([-1000, -1000, 0, -1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1000, 1000, 1000, 1, 1, 1, 1, 10, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

    def _choose_task(self):
        if self.sample_tasks:
            return str(np.random.choice(self.sample_tasks))
        return self.fixed_task

    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        Supported keys in ``options`` dict
        -----------------------------------
        task           : str   – "navigation" or "farming"
        sparse_reward  : bool  – True for terminal-only rewards (default False)
        min_dist       : float – minimum XZ target distance  (-1 = Java default)
        max_dist       : float – maximum XZ target distance  (-1 = Java default)
        num_obstacles  : int   – exact obstacle count        (-1 = Java random 1-2)
        """
        super().reset(seed=seed)
        opts = options or {}

        task          = opts.get("task", self._choose_task())
        sparse_reward = bool(opts.get("sparse_reward", False))
        min_dist      = float(opts.get("min_dist", -1.0))
        max_dist      = float(opts.get("max_dist", -1.0))
        num_obstacles = int(opts.get("num_obstacles", -1))

        payload: dict = {"task": task, "sparse_reward": sparse_reward}
        if min_dist >= 0:    payload["min_dist"]      = min_dist
        if max_dist >= 0:    payload["max_dist"]       = max_dist
        if num_obstacles >= 0: payload["num_obstacles"] = num_obstacles

        r = requests.post(f"{self.base_url}/reset", json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(data["error"])

        obs  = np.array(data["obs"], dtype=np.float32)
        info = data["info"]
        return obs, info

    def step(self, action):
        r = requests.post(f"{self.base_url}/step", json={"action": int(action)}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(data["error"])

        obs       = np.array(data["obs"],     dtype=np.float32)
        reward    = float(data["reward"])
        done      = bool(data["done"])
        truncated = bool(data["truncated"])
        info      = data["info"]
        return obs, reward, done, truncated, info

    def close(self):
        pass
