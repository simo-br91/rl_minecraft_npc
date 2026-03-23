"""
train_nav_sparse.py
-------------------
Trains navigation with SPARSE rewards (no distance shaping).
Only terminal signals: +10 on success, tiny step penalty.

Use this to compare against the shaped-reward baseline in
compare_experiments.py.
"""

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from python_rl.env.minecraft_env import MinecraftEnv


# ------------------------------------------------------------------
# Callback: log per-episode success to a separate CSV
# ------------------------------------------------------------------

class SuccessLogger(BaseCallback):
    """Appends (timestep, success) to a CSV after each episode ends."""

    def __init__(self, log_path: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("w") as f:
            f.write("timestep,success\n")

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                success = int(info.get("success", False))
                with self._log_path.open("a") as f:
                    f.write(f"{self.num_timesteps},{success}\n")
        return True


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    logs_dir        = Path("python_rl/logs")
    checkpoints_dir = Path("python_rl/checkpoints")
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Sparse reward is requested via reset options injected by a wrapper
    class SparseNavEnv(MinecraftEnv):
        """Thin override: always adds sparse_reward=True to reset options."""
        def reset(self, seed=None, options=None):
            opts = dict(options or {})
            opts.setdefault("task",         "navigation")
            opts["sparse_reward"] = True
            return super().reset(seed=seed, options=opts)

    base_env = SparseNavEnv()
    env      = Monitor(base_env, filename=str(logs_dir / "nav_sparse_monitor.csv"))

    success_callback = SuccessLogger(str(logs_dir / "nav_sparse_success.csv"))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        # Higher entropy is critical for sparse rewards — the agent must
        # explore extensively before stumbling on its first success.
        ent_coef=0.10,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=str(logs_dir / "tb"),
    )

    model.learn(total_timesteps=200_000, callback=success_callback)
    model.save(str(checkpoints_dir / "nav_sparse_run1"))
    env.close()

    print("Sparse-reward navigation training complete.")
    print("Checkpoint : python_rl/checkpoints/nav_sparse_run1")
    print("Monitor CSV: python_rl/logs/nav_sparse_monitor.csv")
    print("Success CSV: python_rl/logs/nav_sparse_success.csv")


if __name__ == "__main__":
    main()
