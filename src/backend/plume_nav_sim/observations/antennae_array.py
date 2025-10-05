"""
AntennaeArraySensor: Multiple odor sensors with orientation-relative positioning.

Contract: src/backend/contracts/observation_model_interface.md

This sensor models an array of concentration sensors positioned at specified
angles and distances relative to the agent's heading, similar to insect antennae.
"""

from typing import Any, Dict, List

import numpy as np

import gymnasium as gym

try:  # pragma: no cover - numpy<1.20 compatibility
    from numpy.typing import NDArray
except ImportError:  # pragma: no cover
    NDArray = np.ndarray  # type: ignore[assignment]


class AntennaeArraySensor:
    """Multi-sensor array with orientation-relative positioning.

    Satisfies ObservationModel protocol via duck typing.

    Models an array of concentration sensors positioned relative to the agent's
    heading. Each sensor samples the plume at a position determined by:
    - sensor_angles: Angle relative to agent heading (degrees)
    - sensor_distance: Distance from agent position (grid cells)

    Observation Space:
        Box(low=0.0, high=1.0, shape=(n_sensors,), dtype=float32)

    Required env_state Keys:
        - 'agent_state': AgentState with position and orientation
        - 'plume_field': 2D numpy array of concentrations
        - 'grid_size': GridSize for boundary checking

    Properties:
        - Deterministic: Same state → same observation
        - Pure: No side effects
        - Space Containment: Always returns values in [0, 1]
        - Orientation-Aware: Sensors rotate with agent heading

    Example:
        >>> # Two-sensor array (left/right antennae)
        >>> sensor = AntennaeArraySensor(
        ...     n_sensors=2,
        ...     sensor_angles=[45.0, -45.0],  # ±45° from heading
        ...     sensor_distance=1.0,
        ... )
        >>> env_state = {
        ...     'agent_state': AgentState(
        ...         position=Coordinates(10, 10),
        ...         orientation=0.0,  # Facing East
        ...     ),
        ...     'plume_field': np.random.rand(20, 20),
        ...     'grid_size': GridSize(20, 20),
        ... }
        >>> obs = sensor.get_observation(env_state)
        >>> obs.shape
        (2,)
    """

    def __init__(
        self,
        n_sensors: int = 2,
        sensor_angles: List[float] | None = None,
        sensor_distance: float = 1.0,
    ):
        """Initialize AntennaeArraySensor.

        Args:
            n_sensors: Number of sensors in array
            sensor_angles: Angle of each sensor relative to agent heading (degrees)
                          If None, sensors are evenly distributed around agent
            sensor_distance: Distance from agent to each sensor (grid cells)

        Raises:
            ValueError: If n_sensors != len(sensor_angles) when angles provided
        """
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
        """Gymnasium observation space.

        Returns:
            Box space with shape (n_sensors,) in range [0, 1]

        Contract: observation_model_interface.md - Postcondition C2
        Immutable: Returns same instance every call
        """
        return self._observation_space

    def get_observation(self, env_state: Dict[str, Any]) -> NDArray[np.floating]:
        """Sample concentrations at sensor positions.

        Args:
            env_state: Dictionary containing:
                - 'agent_state': AgentState with position and orientation
                - 'plume_field': 2D numpy array (height, width) of concentrations
                - 'grid_size': GridSize for boundary checking

        Returns:
            1D array of shape (n_sensors,) with concentration values in [0, 1]

        Contract: observation_model_interface.md - get_observation()

        Sensor Positioning:
            Each sensor's world position is computed as:
            1. absolute_angle = agent_orientation + sensor_angle
            2. dx = sensor_distance * cos(absolute_angle)
            3. dy = sensor_distance * sin(absolute_angle)
            4. sensor_pos = agent_pos + (dx, dy)

        Boundary Handling:
            Sensors outside grid bounds return 0.0 concentration.

        Postconditions:
            C1: observation ∈ observation_space
            C2: observation.shape == (n_sensors,)
            C3: Deterministic (same env_state → same observation)
        """
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
        """Return sensor metadata.

        Returns:
            Dictionary with sensor configuration and requirements

        Contract: observation_model_interface.md - get_metadata()
        """
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
