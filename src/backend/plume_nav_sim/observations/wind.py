from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
import numpy as np

try:  # pragma: no cover - numpy<1.20 compatibility
    from numpy.typing import NDArray
except ImportError:  # pragma: no cover
    NDArray = np.ndarray  # type: ignore[assignment]


class WindVectorSensor:
    def __init__(self, noise_std: float = 0.0):
        self.noise_std = float(noise_std)
        self._observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32,
        )
        self._local_rng = np.random.default_rng()

    @property
    def observation_space(self) -> gym.Space:
        """Gymnasium observation space."""
        return self._observation_space

    def get_observation(self, env_state: Dict[str, Any]) -> NDArray[np.floating]:
        """Sample wind vector at the agent's position."""
        base_vector = self._sample_base_vector(env_state)
        if self.noise_std <= 0.0:
            return base_vector

        rng = env_state.get("rng") or self._local_rng
        noise = rng.normal(0.0, self.noise_std, size=base_vector.shape)
        return (base_vector + noise.astype(np.float32)).astype(np.float32)

    def get_metadata(self) -> Dict[str, Any]:
        """Return sensor metadata."""
        return {
            "type": "wind_vector_sensor",
            "modality": "mechanosensory",
            "parameters": {"noise_std": self.noise_std},
            "required_state_keys": ["agent_state"],
            "optional_state_keys": ["wind_field", "rng"],
            "observation_shape": (2,),
            "observation_dtype": "float32",
        }

    def _sample_base_vector(self, env_state: Dict[str, Any]) -> NDArray[np.floating]:
        agent_state = env_state["agent_state"]
        wind_field = env_state.get("wind_field")
        if wind_field is None:
            return np.zeros(2, dtype=np.float32)

        vector = wind_field.sample(agent_state.position)
        return np.asarray(vector, dtype=np.float32)
