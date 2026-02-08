"""Starter template for ObservationModel implementations."""

from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
import numpy as np

from plume_nav_sim.core.types import EnvState
from plume_nav_sim.interfaces.observation import ObservationModel, ObservationType


class ExampleObservationModel(ObservationModel):
    """Minimal observation model stub you can copy and customize."""

    def __init__(self) -> None:
        self._observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    def get_observation(self, env_state: EnvState) -> ObservationType:
        # TODO: build your observation from EnvState keys in core/types.py.
        agent_state = env_state["agent_state"]
        if agent_state is None:
            return np.zeros(2, dtype=np.float32)
        grid = env_state["grid_size"]
        return np.array(
            [
                agent_state.position.x / max(1, grid.width - 1),
                agent_state.position.y / max(1, grid.height - 1),
            ],
            dtype=np.float32,
        )

    def get_metadata(self) -> Dict[str, Any]:
        return {"name": "example_observation_model"}
