from __future__ import annotations

from typing import Any, Dict, List

import gymnasium as gym
import numpy as np

try:  # pragma: no cover - numpy<1.20 compatibility
    from numpy.typing import NDArray
except ImportError:  # pragma: no cover
    NDArray = np.ndarray  # type: ignore[assignment]


class AntennaeArraySensor:
    def __init__(
        self,
        n_sensors: int = 2,
        sensor_angles: List[float] | None = None,
        sensor_distance: float = 1.0,
    ):
        self.n_sensors = n_sensors
        self.sensor_distance = sensor_distance

        # Set sensor angles
        if sensor_angles is None:
            # Evenly distribute sensors around agent
            angle_step = 360.0 / n_sensors
            self.sensor_angles = [i * angle_step for i in range(n_sensors)]
        else:
            if len(sensor_angles) != n_sensors:
                raise ValueError(
                    f"sensor_angles length ({len(sensor_angles)}) must match "
                    f"n_sensors ({n_sensors})"
                )
            self.sensor_angles = list(sensor_angles)

        # Create observation space
        self._observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_sensors,),
            dtype=np.float32,
        )

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    def get_observation(self, env_state: Dict[str, Any]) -> NDArray[np.floating]:
        agent_state = env_state["agent_state"]
        plume_field = env_state["plume_field"]
        grid_size = env_state["grid_size"]

        # Extract agent state
        agent_pos = agent_state.position
        agent_orientation = agent_state.orientation

        # Sample each sensor
        concentrations = []
        for sensor_angle in self.sensor_angles:
            # Compute absolute angle in world frame
            # 0° = East, 90° = North (standard mathematical convention)
            absolute_angle_deg = agent_orientation + sensor_angle
            absolute_angle_rad = np.deg2rad(absolute_angle_deg)

            # Compute sensor position offset
            # Note: In grid coordinates, +x is East, +y is South (array indexing)
            # So we need: dx = distance * cos(angle), dy = -distance * sin(angle)
            dx = self.sensor_distance * np.cos(absolute_angle_rad)
            dy = -self.sensor_distance * np.sin(absolute_angle_rad)

            # Compute sensor position
            sensor_x = int(round(agent_pos.x + dx))
            sensor_y = int(round(agent_pos.y + dy))

            # Check bounds and sample
            if 0 <= sensor_x < grid_size.width and 0 <= sensor_y < grid_size.height:
                concentration = float(plume_field[sensor_y, sensor_x])
            else:
                # Out of bounds → 0 concentration
                concentration = 0.0

            # Clamp to [0, 1]
            concentration = np.clip(concentration, 0.0, 1.0)
            concentrations.append(concentration)

        return np.array(concentrations, dtype=np.float32)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "type": "antennae_array_sensor",
            "modality": "olfactory",
            "parameters": {
                "n_sensors": self.n_sensors,
                "sensor_angles": self.sensor_angles,
                "sensor_distance": self.sensor_distance,
                "sensor_type": "array",
                "range": [0.0, 1.0],
            },
            "required_state_keys": ["agent_state", "plume_field", "grid_size"],
            "observation_shape": (self.n_sensors,),
            "observation_dtype": "float32",
        }
