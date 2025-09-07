"""Minimal Gymnasium environment for plume navigation tasks."""
from __future__ import annotations
from loguru import logger
from typing import Tuple, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .spaces import default_action_space, default_observation_space
class PlumeNavEnv(gym.Env):
    """A lightweight environment exposing a 2D position state."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.observation_space = default_observation_space()
        self.action_space = default_action_space()
        self.state = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._max_steps = 100
        logger.debug("PlumeNavEnv initialized")

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.state[:] = 0.0
        self._step_count = 0
        logger.debug("Environment reset", extra={"seed": seed})
        return self.state.copy(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.state += action
        self._step_count += 1

        reward = -float(np.linalg.norm(self.state))
        terminated = bool(np.linalg.norm(self.state) > 10)
        truncated = bool(self._step_count >= self._max_steps)

        logger.debug(
            "Step",
            extra={
                "action": action.tolist(),
                "state": self.state.tolist(),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            },
        )
        return self.state.copy(), reward, terminated, truncated, {}

    def render(self):
        # Minimal environment has no rendering
        return None
