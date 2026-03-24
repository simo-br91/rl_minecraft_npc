"""
conftest.py
-----------
Shared pytest fixtures available to all tests.
"""

import pytest


@pytest.fixture(autouse=False)
def tmp_logs(tmp_path):
    """Provides a temporary python_rl/logs directory structure."""
    logs = tmp_path / "python_rl" / "logs"
    logs.mkdir(parents=True)
    return logs


@pytest.fixture(autouse=False)
def tmp_checkpoints(tmp_path):
    """Provides a temporary python_rl/checkpoints directory."""
    ckpts = tmp_path / "python_rl" / "checkpoints"
    ckpts.mkdir(parents=True)
    return ckpts
