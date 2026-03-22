import numpy as np
import requests
import gymnasium as gym
from gymnasium import spaces


class MinecraftEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, base_url="http://127.0.0.1:8765", task="navigation", sample_tasks=None):
        super().__init__()
        self.base_url    = base_url
        self.fixed_task  = task
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
        #  10  obstacle_1block_ahead   ← NEW: 1 = jumpable 1-block wall ahead
        self.observation_space = spaces.Box(
            low=np.array(
                [-1000, -1000, 0, -1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32
            ),
            high=np.array(
                [1000, 1000, 1000, 1, 1, 1, 1, 10, 1, 1, 1], dtype=np.float32
            ),
            dtype=np.float32,
        )

    def _choose_task(self):
        if self.sample_tasks:
            return str(np.random.choice(self.sample_tasks))
        return self.fixed_task

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        task = options.get("task") if options else self._choose_task()
        r = requests.post(f"{self.base_url}/reset", json={"task": task}, timeout=15)
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
        obs      = np.array(data["obs"], dtype=np.float32)
        reward   = float(data["reward"])
        done     = bool(data["done"])
        truncated = bool(data["truncated"])
        info     = data["info"]
        return obs, reward, done, truncated, info

    def close(self):
        pass
