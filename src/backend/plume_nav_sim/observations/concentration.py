"""
ConcentrationSensor: Single odor sensor at agent position.

Contract: src/backend/contracts/observation_model_interface.md

This sensor samples the plume concentration at the agent's current position,
providing a scalar observation in [0, 1].
"""

from typing import Any, Dict

import numpy as np

import gymnasium as gym

try:  # pragma: no cover - numpy<1.20 compatibility
    from numpy.typing import NDArray
except ImportError:  # pragma: no cover
    NDArray = np.ndarray  # type: ignore[assignment]


class ConcentrationSensor:
    """Single-point concentration sensor at agent position.

    Satisfies ObservationModel protocol via duck typing.

    Observation Space:
        Box(low=0.0, high=1.0, shape=(1,), dtype=float32)

    Required env_state Keys:
        - 'agent_state': AgentState with position
        - 'plume_field': 2D numpy array of concentrations

    Properties:
        - Deterministic: Same position → same observation
        - Pure: No side effects
        - Space Containment: Always returns value in [0, 1]

    Example:
        >>> sensor = ConcentrationSensor()
        >>> env_state = {
        ...     'agent_state': AgentState(position=Coordinates(5, 5)),
        ...     'plume_field': np.random.rand(10, 10),
        ... }
        >>> obs = sensor.get_observation(env_state)
        >>> obs.shape
        (1,)
    """

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
        """Gymnasium observation space.

        Returns:
            Box space with shape (1,) in range [0, 1]

        Contract: observation_model_interface.md - Postcondition C2
        Immutable: Returns same instance every call
        """
        return self._observation_space

    def get_observation(self, env_state: Dict[str, Any]) -> NDArray[np.floating]:
        """Sample concentration at agent's position.

        Args:
            env_state: Dictionary containing:
                - 'agent_state': AgentState with current position
                - 'plume_field': 2D numpy array (height, width) of concentrations

        Returns:
            1D array of shape (1,) with concentration value in [0, 1]

        Contract: observation_model_interface.md - get_observation()

        Preconditions:
            P1: env_state contains 'agent_state' and 'plume_field'
            P2: agent_state.position is valid Coordinates
            P3: plume_field is 2D numpy array

        Postconditions:
            C1: observation ∈ observation_space
            C2: observation.shape == (1,)
            C3: Deterministic (same env_state → same observation)
        """
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
        """Return sensor metadata.

        Returns:
            Dictionary with sensor configuration and requirements

        Contract: observation_model_interface.md - get_metadata()
        """
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
