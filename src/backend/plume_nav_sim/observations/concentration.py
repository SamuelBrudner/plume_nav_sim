from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
import numpy as np

from ..core.types import EnvState

try:  # pragma: no cover - numpy<1.20 compatibility
    from numpy.typing import NDArray
except ImportError:  # pragma: no cover
    NDArray = np.ndarray  # type: ignore[assignment]


class ConcentrationSensor:
    def __init__(self):
        """Initialize ConcentrationSensor."""
        self._observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    def get_observation(self, env_state: EnvState) -> NDArray[np.floating]:
        agent_state = env_state["agent_state"]
        plume_field = env_state["plume_field"]

        # Extract agent position
        pos = agent_state.position

        # Sample plume at agent position (y, x indexing for numpy arrays)
        concentration = float(plume_field[pos.y, pos.x])

        # Clamp to [0, 1] range
        concentration = np.clip(concentration, 0.0, 1.0)

        # Return as 1D array
        return np.array([concentration], dtype=np.float32)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "type": "concentration_sensor",
            "modality": "olfactory",
            "parameters": {
                "n_sensors": 1,
                "sensor_type": "point",
                "range": [0.0, 1.0],
            },
            "required_state_keys": ["agent_state", "plume_field"],
            "observation_shape": (1,),
            "observation_dtype": "float32",
        }
