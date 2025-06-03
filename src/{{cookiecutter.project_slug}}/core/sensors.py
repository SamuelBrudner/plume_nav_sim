"""
Sensor strategies, position calculations, and odor sampling utilities.

This module centralizes all sensor-related functionality for both single-agent 
and multi-agent navigation scenarios. It provides flexible sensor placement systems,
predefined layouts, custom sensor configurations, and comprehensive odor sampling
capabilities integrated with Hydra configuration management.

Key Features:
- Predefined sensor layouts (SINGLE, LEFT_RIGHT, FRONT_SIDES, etc.)
- Custom sensor geometry with configurable distance and angular parameters
- Position calculation utilities for both single and multi-agent scenarios
- Odor sampling at sensor positions with bounds checking and normalization
- Integration with Hydra configuration system for parameter management
- Support for biological sensing strategy modeling and optimization
"""

from typing import Optional, Dict, Any, Union, Protocol, runtime_checkable
import numpy as np


# Predefined sensor layouts in agent's local coordinate frame
# Each layout defines base offsets where the agent faces +x direction
# and +y is "left" of the agent (standard mathematical orientation)
PREDEFINED_SENSOR_LAYOUTS: Dict[str, np.ndarray] = {
    # Single sensor at agent position
    "SINGLE": np.array([[0.0, 0.0]]),
    
    # Left-Right configuration: one sensor at +y, another at -y
    "LEFT_RIGHT": np.array([
        [0.0,  1.0],  # Left sensor
        [0.0, -1.0],  # Right sensor
    ]),
    
    # Front and sides configuration: forward sensor plus left and right
    "FRONT_SIDES": np.array([
        [1.0,  0.0],  # Front sensor
        [0.0,  1.0],  # Left sensor
        [0.0, -1.0],  # Right sensor
    ]),
    
    # Triangular configuration for enhanced spatial coverage
    "TRIANGLE": np.array([
        [1.0,  0.0],   # Front
        [-0.5, 0.866], # Back-left (120 degrees)
        [-0.5, -0.866] # Back-right (240 degrees)
    ]),
    
    # Cross configuration for four-directional sensing
    "CROSS": np.array([
        [1.0,  0.0],  # Front
        [0.0,  1.0],  # Left
        [-1.0, 0.0],  # Back
        [0.0, -1.0],  # Right
    ]),
    
    # Linear array for directional gradient sensing
    "LINEAR": np.array([
        [1.5,  0.0],  # Far front
        [0.5,  0.0],  # Near front
        [0.0,  0.0],  # Center
        [-0.5, 0.0],  # Near back
        [-1.5, 0.0],  # Far back
    ])
}


@runtime_checkable
class NavigatorProtocol(Protocol):
    """Protocol defining the navigator interface for sensor calculations.
    
    This protocol ensures compatibility with navigator implementations
    by defining the minimum required properties for sensor positioning.
    """
    
    @property
    def positions(self) -> np.ndarray:
        """Agent position(s) as numpy array.
        
        Returns
        -------
        np.ndarray
            Shape (1, 2) for single agent or (num_agents, 2) for multiple agents
        """
        ...
    
    @property
    def orientations(self) -> np.ndarray:
        """Agent orientation(s) in degrees.
        
        Returns
        -------
        np.ndarray
            Shape (1,) for single agent or (num_agents,) for multiple agents
        """
        ...
    
    @property
    def num_agents(self) -> int:
        """Number of agents.
        
        Returns
        -------
        int
            Number of agents (1 for single agent, >1 for multi-agent)
        """
        ...


class SensorLayout:
    """Represents a sensor layout configuration.
    
    This class encapsulates sensor layout information including predefined
    layouts and custom configurations, providing a unified interface for
    sensor position calculations.
    
    Parameters
    ----------
    layout_name : str, optional
        Name of predefined layout from PREDEFINED_SENSOR_LAYOUTS
    distance : float, default=5.0
        Distance from agent center to sensors
    num_sensors : int, optional
        Number of sensors (used for custom layouts)
    angle : float, default=45.0
        Angular separation between sensors in degrees (for custom layouts)
    custom_offsets : np.ndarray, optional
        Custom sensor offsets in agent's local coordinate frame
    
    Examples
    --------
    >>> # Using predefined layout
    >>> layout = SensorLayout(layout_name="LEFT_RIGHT", distance=10.0)
    >>> 
    >>> # Custom layout with specific parameters
    >>> layout = SensorLayout(distance=5.0, num_sensors=3, angle=30.0)
    >>> 
    >>> # Fully custom layout
    >>> offsets = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    >>> layout = SensorLayout(custom_offsets=offsets, distance=1.0)
    """
    
    def __init__(
        self,
        layout_name: Optional[str] = None,
        distance: float = 5.0,
        num_sensors: Optional[int] = None,
        angle: float = 45.0,
        custom_offsets: Optional[np.ndarray] = None
    ):
        self.layout_name = layout_name
        self.distance = distance
        self.num_sensors = num_sensors
        self.angle = angle
        self.custom_offsets = custom_offsets
        
        # Validate and compute local offsets
        self._local_offsets = self._compute_local_offsets()
    
    def _compute_local_offsets(self) -> np.ndarray:
        """Compute local sensor offsets based on configuration.
        
        Returns
        -------
        np.ndarray
            Shape (num_sensors, 2) array of local offsets
            
        Raises
        ------
        ValueError
            If configuration is invalid or incomplete
        """
        if self.custom_offsets is not None:
            # Use custom offsets, scaled by distance
            offsets = np.array(self.custom_offsets, dtype=float)
            if offsets.ndim != 2 or offsets.shape[1] != 2:
                raise ValueError(
                    f"custom_offsets must be shape (N, 2), got {offsets.shape}"
                )
            return offsets * self.distance
        
        elif self.layout_name is not None:
            # Use predefined layout
            return get_predefined_sensor_layout(self.layout_name, self.distance)
        
        elif self.num_sensors is not None:
            # Generate custom layout based on parameters
            return define_sensor_offsets(self.num_sensors, self.distance, self.angle)
        
        else:
            raise ValueError(
                "Must specify either layout_name, num_sensors, or custom_offsets"
            )
    
    @property
    def local_offsets(self) -> np.ndarray:
        """Get local sensor offsets.
        
        Returns
        -------
        np.ndarray
            Shape (num_sensors, 2) array of (x, y) offsets in agent's local frame
        """
        return self._local_offsets.copy()
    
    @property
    def sensor_count(self) -> int:
        """Get number of sensors in this layout.
        
        Returns
        -------
        int
            Number of sensors
        """
        return self._local_offsets.shape[0]
    
    def __repr__(self) -> str:
        """String representation of the sensor layout."""
        if self.layout_name:
            return f"SensorLayout('{self.layout_name}', distance={self.distance})"
        elif self.custom_offsets is not None:
            return f"SensorLayout(custom, {self.sensor_count} sensors, distance={self.distance})"
        else:
            return f"SensorLayout({self.sensor_count} sensors, angle={self.angle}°, distance={self.distance})"


def get_predefined_sensor_layout(
    layout_name: str,
    distance: float = 5.0
) -> np.ndarray:
    """Get a predefined sensor layout scaled by distance.
    
    Retrieves base sensor offsets from PREDEFINED_SENSOR_LAYOUTS and scales
    them by the specified distance parameter.
    
    Parameters
    ----------
    layout_name : str
        Name of the layout; must be a key in PREDEFINED_SENSOR_LAYOUTS
    distance : float, default=5.0
        Scaling distance from agent center to sensors
        
    Returns
    -------
    np.ndarray
        Shape (num_sensors, 2) array of local (x, y) offsets
        
    Raises
    ------
    ValueError
        If layout_name is not found in PREDEFINED_SENSOR_LAYOUTS
        
    Examples
    --------
    >>> offsets = get_predefined_sensor_layout("LEFT_RIGHT", distance=10.0)
    >>> print(offsets)
    [[ 0. 10.]
     [ 0. -10.]]
    """
    try:
        base_layout = PREDEFINED_SENSOR_LAYOUTS[layout_name]
    except KeyError as e:
        available_layouts = list(PREDEFINED_SENSOR_LAYOUTS.keys())
        raise ValueError(
            f"Unknown sensor layout: '{layout_name}'. "
            f"Available layouts: {available_layouts}"
        ) from e
    
    # Scale by the requested distance
    return base_layout.astype(float) * distance


def define_sensor_offsets(
    num_sensors: int,
    distance: float,
    angle: float
) -> np.ndarray:
    """Define custom sensor offsets in agent's local coordinate frame.
    
    Creates sensor positions distributed symmetrically around the agent's
    forward direction. The sensors are positioned at equal angular intervals
    with the specified angular separation.
    
    Parameters
    ----------
    num_sensors : int
        Number of sensors to create (must be >= 1)
    distance : float
        Distance from agent center to each sensor (must be > 0)
    angle : float
        Angular increment between adjacent sensors in degrees
        
    Returns
    -------
    np.ndarray
        Shape (num_sensors, 2) array of (x, y) offsets in local coordinates
        
    Raises
    ------
    ValueError
        If num_sensors < 1 or distance <= 0
        
    Notes
    -----
    - For odd number of sensors, one sensor is placed directly ahead
    - For even numbers, sensors are placed symmetrically around forward direction
    - The total angular span is (num_sensors - 1) * angle
    - Agent's forward direction is +x, left is +y
    
    Examples
    --------
    >>> # Three sensors with 30° separation
    >>> offsets = define_sensor_offsets(3, distance=5.0, angle=30.0)
    >>> # Results in sensors at -30°, 0°, +30° relative to forward direction
    """
    if num_sensors < 1:
        raise ValueError(f"num_sensors must be >= 1, got {num_sensors}")
    if distance <= 0:
        raise ValueError(f"distance must be > 0, got {distance}")
    
    # Single sensor case - place at agent center
    if num_sensors == 1:
        return np.array([[0.0, 0.0]])
    
    # Calculate angular range and starting angle
    total_angular_range = (num_sensors - 1) * angle
    start_angle = -total_angular_range / 2.0
    
    # Generate offsets for each sensor
    offsets = np.zeros((num_sensors, 2), dtype=float)
    
    for i in range(num_sensors):
        # Calculate current angle in degrees
        current_angle_deg = start_angle + i * angle
        
        # Convert to radians for trigonometric functions
        current_angle_rad = np.deg2rad(current_angle_deg)
        
        # Calculate position using polar coordinates
        offsets[i, 0] = distance * np.cos(current_angle_rad)  # x-component
        offsets[i, 1] = distance * np.sin(current_angle_rad)  # y-component
    
    return offsets


def rotate_offset(local_offset: np.ndarray, orientation_deg: float) -> np.ndarray:
    """Rotate a local offset by agent's orientation.
    
    Transforms sensor positions from agent's local coordinate frame to
    global coordinate frame using 2D rotation matrix.
    
    Parameters
    ----------
    local_offset : np.ndarray
        Shape (2,) array representing (x, y) in local coordinates
    orientation_deg : float
        Agent's orientation in degrees (0° = facing +x direction)
        
    Returns
    -------
    np.ndarray
        Shape (2,) array representing the offset in global coordinates
        
    Notes
    -----
    Uses standard 2D rotation matrix:
    [cos(θ) -sin(θ)]
    [sin(θ)  cos(θ)]
    
    Examples
    --------
    >>> local_pos = np.array([1.0, 0.0])  # 1 unit forward
    >>> global_pos = rotate_offset(local_pos, 90.0)  # Rotate 90°
    >>> print(global_pos)  # Should be [0.0, 1.0] (pointing up)
    """
    # Convert orientation to radians
    orientation_rad = np.deg2rad(orientation_deg)
    
    # Create 2D rotation matrix
    cos_theta = np.cos(orientation_rad)
    sin_theta = np.sin(orientation_rad)
    
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta]
    ])
    
    # Apply rotation transformation
    return rotation_matrix @ local_offset


def compute_sensor_positions(
    agent_positions: np.ndarray,
    agent_orientations: np.ndarray,
    sensor_layout: Union[SensorLayout, str, np.ndarray],
    distance: float = 5.0,
    angle: float = 45.0,
    num_sensors: int = 2
) -> np.ndarray:
    """Compute global sensor positions for all agents.
    
    Transforms sensor offsets from local agent coordinates to global
    coordinates based on each agent's position and orientation.
    
    Parameters
    ----------
    agent_positions : np.ndarray
        Shape (num_agents, 2) array of agent (x, y) positions
    agent_orientations : np.ndarray
        Shape (num_agents,) array of agent orientations in degrees
    sensor_layout : Union[SensorLayout, str, np.ndarray]
        Sensor configuration - can be:
        - SensorLayout object with predefined configuration
        - String name of predefined layout
        - np.ndarray of custom local offsets
    distance : float, default=5.0
        Distance from agent center (used if sensor_layout is string)
    angle : float, default=45.0
        Angular separation (used for custom layouts)
    num_sensors : int, default=2
        Number of sensors (used for custom layouts)
        
    Returns
    -------
    np.ndarray
        Shape (num_agents, num_sensors, 2) array of global sensor positions
        
    Raises
    ------
    ValueError
        If agent_positions and agent_orientations have incompatible shapes
        
    Examples
    --------
    >>> positions = np.array([[0.0, 0.0], [10.0, 10.0]])
    >>> orientations = np.array([0.0, 90.0])  # First agent faces east, second north
    >>> sensor_pos = compute_sensor_positions(positions, orientations, "LEFT_RIGHT")
    """
    # Validate input shapes
    if agent_positions.shape[0] != agent_orientations.shape[0]:
        raise ValueError(
            f"Incompatible shapes: positions {agent_positions.shape} "
            f"vs orientations {agent_orientations.shape}"
        )
    
    num_agents = agent_positions.shape[0]
    
    # Handle different sensor_layout types
    if isinstance(sensor_layout, SensorLayout):
        local_offsets = sensor_layout.local_offsets
    elif isinstance(sensor_layout, str):
        local_offsets = get_predefined_sensor_layout(sensor_layout, distance)
    elif isinstance(sensor_layout, np.ndarray):
        local_offsets = sensor_layout.astype(float)
        if local_offsets.ndim != 2 or local_offsets.shape[1] != 2:
            raise ValueError(
                f"Custom offsets must be shape (N, 2), got {local_offsets.shape}"
            )
    else:
        # Fallback to custom layout generation
        local_offsets = define_sensor_offsets(num_sensors, distance, angle)
    
    num_sensors = local_offsets.shape[0]
    sensor_positions = np.zeros((num_agents, num_sensors, 2), dtype=float)
    
    # Transform each agent's sensors to global coordinates
    for agent_idx in range(num_agents):
        agent_pos = agent_positions[agent_idx]
        agent_orientation = agent_orientations[agent_idx]
        
        # Transform each sensor for this agent
        for sensor_idx in range(num_sensors):
            local_offset = local_offsets[sensor_idx]
            
            # Rotate offset to global frame and add agent position
            global_offset = rotate_offset(local_offset, agent_orientation)
            sensor_positions[agent_idx, sensor_idx] = agent_pos + global_offset
    
    return sensor_positions


def calculate_sensor_positions(
    navigator: NavigatorProtocol,
    sensor_distance: float = 5.0,
    sensor_angle: float = 45.0,
    num_sensors: int = 2,
    layout_name: Optional[str] = None
) -> np.ndarray:
    """Calculate sensor positions using navigator state.
    
    Convenience function that extracts position and orientation from
    a navigator and computes sensor positions.
    
    Parameters
    ----------
    navigator : NavigatorProtocol
        Navigator with position and orientation information
    sensor_distance : float, default=5.0
        Distance from agent center to each sensor
    sensor_angle : float, default=45.0
        Angular separation between sensors in degrees
    num_sensors : int, default=2
        Number of sensors per agent
    layout_name : str, optional
        Name of predefined sensor layout. If provided, other parameters ignored.
        
    Returns
    -------
    np.ndarray
        Shape (num_agents, num_sensors, 2) array of global sensor positions
        
    Examples
    --------
    >>> # Assuming navigator is already configured
    >>> sensor_pos = calculate_sensor_positions(
    ...     navigator, 
    ...     layout_name="LEFT_RIGHT",
    ...     sensor_distance=8.0
    ... )
    """
    return compute_sensor_positions(
        navigator.positions,
        navigator.orientations,
        layout_name if layout_name else "custom",
        distance=sensor_distance,
        angle=sensor_angle,
        num_sensors=num_sensors
    )


def read_odor_values(
    env_array: np.ndarray,
    positions: np.ndarray
) -> np.ndarray:
    """Read odor concentration values at specified positions.
    
    Samples the environment array at given positions with proper bounds
    checking, coordinate conversion, and data type handling.
    
    Parameters
    ----------
    env_array : np.ndarray
        Environment array representing odor concentrations
        Can be 2D array or object with 'current_frame' attribute
    positions : np.ndarray
        Shape (N, 2) array of (x, y) positions to sample
        
    Returns
    -------
    np.ndarray
        Shape (N,) array of odor concentration values
        
    Notes
    -----
    - Positions outside array bounds return 0.0
    - uint8 arrays are automatically normalized to [0, 1] range
    - Coordinates are converted to integer indices using floor operation
    - Supports mock plume objects for testing
    
    Examples
    --------
    >>> env = np.random.rand(100, 100) * 255
    >>> env = env.astype(np.uint8)
    >>> positions = np.array([[25.7, 30.2], [50.0, 75.5]])
    >>> odor_values = read_odor_values(env, positions)
    >>> print(odor_values.shape)  # (2,)
    """
    # Handle mock plume objects (for testing compatibility)
    if hasattr(env_array, 'current_frame'):
        env_array = env_array.current_frame
    
    # Validate environment array
    if not hasattr(env_array, 'shape') or len(env_array.shape) < 2:
        # Return zeros for invalid or mock arrays
        return np.zeros(len(positions), dtype=float)
    
    height, width = env_array.shape[:2]
    num_positions = positions.shape[0]
    odor_values = np.zeros(num_positions, dtype=float)
    
    # Convert positions to integer pixel coordinates
    x_indices = np.floor(positions[:, 0]).astype(int)
    y_indices = np.floor(positions[:, 1]).astype(int)
    
    # Create bounds mask for valid positions
    valid_mask = (
        (x_indices >= 0) & (x_indices < width) &
        (y_indices >= 0) & (y_indices < height)
    )
    
    # Sample values for positions within bounds
    for i in range(num_positions):
        if valid_mask[i]:
            # Note: array indexing is [row, col] = [y, x]
            raw_value = env_array[y_indices[i], x_indices[i]]
            
            # Normalize uint8 values to [0, 1] range
            if hasattr(env_array, 'dtype') and env_array.dtype == np.uint8:
                odor_values[i] = float(raw_value) / 255.0
            else:
                odor_values[i] = float(raw_value)
    
    return odor_values


def sample_odor_at_sensors(
    navigator: NavigatorProtocol,
    env_array: np.ndarray,
    sensor_distance: float = 5.0,
    sensor_angle: float = 45.0,
    num_sensors: int = 2,
    layout_name: Optional[str] = None
) -> np.ndarray:
    """Sample odor concentrations at all sensor positions.
    
    Combines sensor position calculation with odor sampling to provide
    a complete sensor reading system for navigation algorithms.
    
    Parameters
    ----------
    navigator : NavigatorProtocol
        Navigator instance with current agent state
    env_array : np.ndarray
        Environment array representing odor concentrations
    sensor_distance : float, default=5.0
        Distance from agent center to sensors
    sensor_angle : float, default=45.0
        Angular separation between sensors in degrees
    num_sensors : int, default=2
        Number of sensors per agent
    layout_name : str, optional
        Name of predefined sensor layout
        
    Returns
    -------
    np.ndarray
        Shape (num_agents, num_sensors) array of odor readings
        
    Examples
    --------
    >>> # Sample with left-right sensor configuration
    >>> readings = sample_odor_at_sensors(
    ...     navigator, 
    ...     env_array,
    ...     layout_name="LEFT_RIGHT", 
    ...     sensor_distance=10.0
    ... )
    >>> print(f"Left sensor: {readings[0, 0]}, Right sensor: {readings[0, 1]}")
    """
    # Calculate global sensor positions
    sensor_positions = calculate_sensor_positions(
        navigator, sensor_distance, sensor_angle, num_sensors, layout_name
    )
    
    # Reshape for batch sampling: (num_agents * num_sensors, 2)
    flat_positions = sensor_positions.reshape(-1, 2)
    
    # Sample odor at all sensor positions
    flat_readings = read_odor_values(env_array, flat_positions)
    
    # Reshape back to (num_agents, num_sensors)
    num_agents = navigator.num_agents
    actual_num_sensors = sensor_positions.shape[1]
    odor_readings = flat_readings.reshape(num_agents, actual_num_sensors)
    
    return odor_readings


def validate_sensor_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize sensor configuration parameters.
    
    Ensures sensor configuration dictionaries contain valid parameters
    and provides sensible defaults for missing values.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Sensor configuration dictionary
        
    Returns
    -------
    Dict[str, Any]
        Validated and normalized configuration
        
    Raises
    ------
    ValueError
        If configuration contains invalid parameters
        
    Examples
    --------
    >>> config = {"layout_name": "LEFT_RIGHT", "distance": 8.0}
    >>> validated = validate_sensor_configuration(config)
    >>> print(validated["distance"])  # 8.0
    """
    # Define valid configuration keys and their types
    valid_keys = {
        'layout_name': (str, type(None)),
        'distance': (int, float),
        'num_sensors': int,
        'angle': (int, float),
        'custom_offsets': (np.ndarray, type(None))
    }
    
    # Default values
    defaults = {
        'distance': 5.0,
        'angle': 45.0,
        'num_sensors': 2
    }
    
    # Create validated configuration
    validated_config = {}
    
    # Check for unknown keys
    unknown_keys = set(config.keys()) - set(valid_keys.keys())
    if unknown_keys:
        raise ValueError(f"Unknown sensor configuration keys: {unknown_keys}")
    
    # Validate and set known keys
    for key, expected_types in valid_keys.items():
        if key in config:
            value = config[key]
            if not isinstance(value, expected_types):
                raise ValueError(
                    f"Invalid type for '{key}': expected {expected_types}, "
                    f"got {type(value)}"
                )
            validated_config[key] = value
        elif key in defaults:
            validated_config[key] = defaults[key]
    
    # Validate specific constraints
    if 'distance' in validated_config and validated_config['distance'] <= 0:
        raise ValueError("Sensor distance must be positive")
    
    if 'num_sensors' in validated_config and validated_config['num_sensors'] < 1:
        raise ValueError("Number of sensors must be at least 1")
    
    return validated_config


def create_sensor_layout_from_config(config: Dict[str, Any]) -> SensorLayout:
    """Create a SensorLayout from configuration dictionary.
    
    Factory function to create SensorLayout instances from configuration
    data, typically loaded from Hydra configuration files.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary with sensor parameters
        
    Returns
    -------
    SensorLayout
        Configured sensor layout instance
        
    Examples
    --------
    >>> config = {
    ...     "layout_name": "LEFT_RIGHT",
    ...     "distance": 10.0
    ... }
    >>> layout = create_sensor_layout_from_config(config)
    >>> print(layout.sensor_count)  # 2
    """
    # Validate configuration
    validated_config = validate_sensor_configuration(config)
    
    # Create SensorLayout with validated parameters
    return SensorLayout(**validated_config)


# Convenience functions for common sensor operations
def get_available_layouts() -> list[str]:
    """Get list of available predefined sensor layouts.
    
    Returns
    -------
    list[str]
        Names of all predefined sensor layouts
    """
    return list(PREDEFINED_SENSOR_LAYOUTS.keys())


def describe_layout(layout_name: str) -> Dict[str, Any]:
    """Get description and metadata for a sensor layout.
    
    Parameters
    ----------
    layout_name : str
        Name of the sensor layout
        
    Returns
    -------
    Dict[str, Any]
        Layout metadata including sensor count and offsets
        
    Raises
    ------
    ValueError
        If layout_name is not found
    """
    if layout_name not in PREDEFINED_SENSOR_LAYOUTS:
        raise ValueError(f"Unknown layout: {layout_name}")
    
    layout = PREDEFINED_SENSOR_LAYOUTS[layout_name]
    
    return {
        'name': layout_name,
        'sensor_count': layout.shape[0],
        'base_offsets': layout.tolist(),
        'description': _get_layout_description(layout_name)
    }


def _get_layout_description(layout_name: str) -> str:
    """Get human-readable description of sensor layout."""
    descriptions = {
        "SINGLE": "Single sensor at agent position",
        "LEFT_RIGHT": "Two sensors positioned left and right of agent",
        "FRONT_SIDES": "Three sensors: front, left, and right",
        "TRIANGLE": "Three sensors in triangular configuration",
        "CROSS": "Four sensors in cross pattern (front, left, back, right)",
        "LINEAR": "Five sensors in linear array along agent's forward axis"
    }
    return descriptions.get(layout_name, "Custom sensor layout")