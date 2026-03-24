"""
tests/test_minecraft_env_unit.py
---------------------------------
Unit tests for MinecraftEnv that do NOT require a running Minecraft server.
All HTTP calls are intercepted by unittest.mock.

Run with:
    python -m pytest tests/test_minecraft_env_unit.py -v
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# We patch requests before importing the env so _wait_for_server doesn't block.


def _make_mock_response(data: dict) -> MagicMock:
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = data
    r.raise_for_status = MagicMock()
    return r


def _fake_obs(n: int = 28) -> list[float]:
    return [0.0] * n


def _reset_response(**overrides) -> dict:
    base = {
        "obs": _fake_obs(),
        "reward": 0.0,
        "done": False,
        "truncated": False,
        "info": {"task_name": "navigation", "success": False,
                 "episode_step": 0, "distance_to_target": 10.0,
                 "stuck_steps": 0, "task_progress": 0.0,
                 "invalid_action_count": 0, "sparse_reward": False,
                 "health": 20.0, "food_level": 20,
                 "crops_harvested": 0, "total_crops": 0,
                 "mobs_killed": 0, "active_slot": 0, "active_item": "iron_sword"},
    }
    base.update(overrides)
    return base


def _step_response(**overrides) -> dict:
    return _reset_response(**overrides)


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def mock_requests():
    """Patch requests.get (health) and requests.post (reset/step)."""
    health_resp = MagicMock()
    health_resp.status_code = 200

    with patch("requests.get", return_value=health_resp) as mock_get, \
         patch("requests.post") as mock_post:
        mock_post.return_value = _make_mock_response(_reset_response())
        yield mock_get, mock_post


@pytest.fixture
def env(mock_requests):
    from python_rl.env.minecraft_env import MinecraftEnv
    e = MinecraftEnv()
    yield e
    # Don't call close() as it sends a reset that complicates the mock


# ------------------------------------------------------------------ #
# Initialization
# ------------------------------------------------------------------ #

class TestInit:
    def test_action_space_is_discrete_13(self, env):
        from gymnasium.spaces import Discrete
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 13

    def test_observation_space_dim(self, env):
        assert env.observation_space.shape == (28,)

    def test_obs_space_low_values(self, env):
        # dx, dz, angle_to_target, yaw_norm, pitch_norm, mob_angle can be negative
        assert env.observation_space.low[0] == -1.0   # dx_norm
        assert env.observation_space.low[1] == -1.0   # dz_norm
        assert env.observation_space.low[3] == -1.0   # angle_to_target

    def test_obs_space_high_values(self, env):
        # task_id_norm high should be 1.0, NOT 10.0
        assert env.observation_space.high[9] == 1.0   # task_id_norm

    def test_obs_space_all_high_are_one(self, env):
        assert np.all(env.observation_space.high == 1.0), \
            "All high bounds should be 1.0"

    def test_health_check_called_on_init(self, mock_requests):
        mock_get, _ = mock_requests
        from python_rl.env.minecraft_env import MinecraftEnv
        MinecraftEnv()
        mock_get.assert_called()


# ------------------------------------------------------------------ #
# reset()
# ------------------------------------------------------------------ #

class TestReset:
    def test_returns_obs_and_info(self, env, mock_requests):
        _, mock_post = mock_requests
        mock_post.return_value = _make_mock_response(_reset_response())
        obs, info = env.reset()
        assert obs.shape == (28,)
        assert isinstance(info, dict)

    def test_obs_clipped_to_bounds(self, env, mock_requests):
        _, mock_post = mock_requests
        # Return obs values outside [-1, 1] to test clipping
        raw_obs = [999.0] * 28
        mock_post.return_value = _make_mock_response(
            _reset_response(obs=raw_obs))
        obs, _ = env.reset()
        assert np.all(obs <= 1.0)
        assert np.all(obs >= -1.0)

    def test_forwards_task_to_payload(self, env, mock_requests):
        _, mock_post = mock_requests
        mock_post.return_value = _make_mock_response(_reset_response())
        env.reset(options={"task": "farming"})
        payload = mock_post.call_args[1]["json"]
        assert payload["task"] == "farming"

    def test_forwards_sparse_reward(self, env, mock_requests):
        _, mock_post = mock_requests
        mock_post.return_value = _make_mock_response(_reset_response())
        env.reset(options={"sparse_reward": True})
        payload = mock_post.call_args[1]["json"]
        assert payload["sparse_reward"] is True

    def test_forwards_seed_to_payload(self, env, mock_requests):
        _, mock_post = mock_requests
        mock_post.return_value = _make_mock_response(_reset_response())
        env.reset(seed=42)
        payload = mock_post.call_args[1]["json"]
        assert payload["seed"] == 42

    def test_no_seed_in_payload_when_none(self, env, mock_requests):
        _, mock_post = mock_requests
        mock_post.return_value = _make_mock_response(_reset_response())
        env.reset()
        payload = mock_post.call_args[1]["json"]
        assert "seed" not in payload

    def test_raises_on_error_response(self, env, mock_requests):
        _, mock_post = mock_requests
        mock_post.return_value = _make_mock_response({"error": "No player found"})
        with pytest.raises(RuntimeError, match="No player found"):
            env.reset()

    def test_seed_sets_internal_rng(self, env, mock_requests):
        _, mock_post = mock_requests
        mock_post.return_value = _make_mock_response(_reset_response())
        env.reset(seed=1234)
        assert env._np_random is not None


# ------------------------------------------------------------------ #
# step()
# ------------------------------------------------------------------ #

class TestStep:
    def test_returns_five_tuple(self, env, mock_requests):
        _, mock_post = mock_requests
        mock_post.return_value = _make_mock_response(_step_response())
        result = env.step(0)
        assert len(result) == 5

    def test_obs_is_float32(self, env, mock_requests):
        _, mock_post = mock_requests
        mock_post.return_value = _make_mock_response(_step_response())
        obs, *_ = env.step(0)
        assert obs.dtype == np.float32

    def test_step_sends_action_in_payload(self, env, mock_requests):
        _, mock_post = mock_requests
        mock_post.return_value = _make_mock_response(_step_response())
        env.step(7)
        payload = mock_post.call_args[1]["json"]
        assert payload["action"] == 7

    def test_done_flag_propagated(self, env, mock_requests):
        _, mock_post = mock_requests
        mock_post.return_value = _make_mock_response(
            _step_response(done=True))
        _, _, done, truncated, _ = env.step(4)
        assert done is True

    def test_truncated_flag_propagated(self, env, mock_requests):
        _, mock_post = mock_requests
        mock_post.return_value = _make_mock_response(
            _step_response(truncated=True))
        _, _, done, truncated, _ = env.step(4)
        assert truncated is True

    def test_reward_is_float(self, env, mock_requests):
        _, mock_post = mock_requests
        mock_post.return_value = _make_mock_response(
            _step_response(reward=5.5))
        _, reward, *_ = env.step(0)
        assert isinstance(reward, float)
        assert reward == pytest.approx(5.5)

    def test_raises_on_error_response(self, env, mock_requests):
        _, mock_post = mock_requests
        mock_post.return_value = _make_mock_response({"error": "Step error"})
        with pytest.raises(RuntimeError):
            env.step(0)


# ------------------------------------------------------------------ #
# _choose_task (sampling)
# ------------------------------------------------------------------ #

class TestChooseTask:
    def test_fixed_task_returned(self, mock_requests):
        from python_rl.env.minecraft_env import MinecraftEnv
        e = MinecraftEnv(task="farming")
        assert e._choose_task() == "farming"

    def test_sample_tasks_picks_from_list(self, mock_requests):
        from python_rl.env.minecraft_env import MinecraftEnv
        tasks = ["navigation", "farming", "combat"]
        e = MinecraftEnv(sample_tasks=tasks)
        for _ in range(30):
            chosen = e._choose_task()
            assert chosen in tasks

    def test_sample_tasks_uses_seeded_rng(self, mock_requests):
        """Same seed → same task sequence."""
        from python_rl.env.minecraft_env import MinecraftEnv
        import numpy as np
        tasks = ["navigation", "farming"]
        e1 = MinecraftEnv(sample_tasks=tasks)
        e2 = MinecraftEnv(sample_tasks=tasks)
        e1._np_random = np.random.default_rng(42)
        e2._np_random = np.random.default_rng(42)
        seq1 = [e1._choose_task() for _ in range(10)]
        seq2 = [e2._choose_task() for _ in range(10)]
        assert seq1 == seq2


# ------------------------------------------------------------------ #
# Retry logic
# ------------------------------------------------------------------ #

class TestRetryLogic:
    def test_retries_on_timeout(self, mock_requests):
        _, mock_post = mock_requests
        import requests as req_lib
        # First call times out, second succeeds
        mock_post.side_effect = [
            req_lib.Timeout(),
            _make_mock_response(_reset_response()),
        ]
        from python_rl.env.minecraft_env import MinecraftEnv
        e = MinecraftEnv(retry_attempts=3)
        obs, info = e.reset()
        assert obs.shape == (28,)
        assert mock_post.call_count == 2

    def test_raises_after_max_attempts(self, mock_requests):
        _, mock_post = mock_requests
        import requests as req_lib
        mock_post.side_effect = req_lib.Timeout()
        from python_rl.env.minecraft_env import MinecraftEnv
        e = MinecraftEnv(retry_attempts=2)
        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            e.reset()


# ------------------------------------------------------------------ #
# render() — should not raise
# ------------------------------------------------------------------ #

class TestRender:
    def test_render_does_not_raise(self, env):
        env.render()   # should be a no-op
