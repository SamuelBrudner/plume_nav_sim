"""
Sensor strategies, position calculations, and odor sampling utilities.

This module centralizes sensor-related functionality for both single-agent and 
multi-agent navigation scenarios, providing flexible sensor placement systems, 
predefined layouts, and comprehensive odor sampling capabilities with Hydra 
configuration integration.

Key Features:
- Predefined sensor layouts (SINGLE, LEFT_RIGHT, FRONT_SIDES, etc.)
- Custom sensor geometry configuration
- Position calculations with rotation matrices
- Multi-sensor odor sampling with bilinear interpolation
- Integration with Hydra configuration system
- Support for both single and multi-agent scenarios

Examples
--------
Basic sensor usage:

>>> from {{cookiecutter.project_slug}}.core.sensors import calculate_sensor_positions
>>> from {{cookiecutter.project_slug}}.api.navigation import create_navigator
>>> navigator = create_navigator({"type": "single", "position": [10, 10]})
>>> positions = calculate_sensor_positions(navigator, layout_name="LEFT_RIGHT")

Custom sensor configuration:

>>> from {{cookiecutter.project_slug}}.core.sensors import sample_odor_at_sensors
>>> odor_values = sample_odor_at_sensors(
...     navigator, env_array, 
...     sensor_distance=8.0, 
...     sensor_angle=30.0, 
...     num_sensors=4
... )

Hydra integration:

>>> # In conf/config.yaml:
>>> # sensors:
>>> #   layout_name: "BILATERAL" 
>>> #   distance: 5.0
>>> #   angle: 45.0
>>> from hydra import compose, initialize
>>> with initialize(config_path="../conf"):
...     cfg = compose(config_name="config")
...     positions = calculate_sensor_positions(navigator, **cfg.sensors)
"""

from typing import Dict, Optional, Tuple, Union, Any
import numpy as np

# Import types and protocols from dependencies
try:
    from .navigator import NavigatorProtocol
except ImportError:
    # Fallback for development environments
    from typing import Protocol
    
    class NavigatorProtocol(Protocol):
        """Fallback NavigatorProtocol for development."""
        positions: np.ndarray
        orientations: np.ndarray
        num_agents: int


# Predefined sensor layouts in local agent coordinates
# Each layout defines sensor offsets in the agent's local frame where:
# - Agent faces positive x-direction
# - Positive y is to the left of the agent (standard mathematical orientation)
# - Values are base offsets that get scaled by sensor_distance parameter
PREDEFINED_SENSOR_LAYOUTS: Dict[str, np.ndarray] = {
    # Single sensor at agent center
    "SINGLE": np.array([[0.0, 0.0]]),
    
    # Two sensors: left and right of agent
    "LEFT_RIGHT": np.array([
        [0.0,  1.0],  # Left sensor
        [0.0, -1.0],  # Right sensor
    ]),
    
    # Three sensors: forward, left, and right
    "FRONT_SIDES": np.array([
        [1.0,  0.0],  # Forward sensor
        [0.0,  1.0],  # Left sensor
        [0.0, -1.0],  # Right sensor
    ]),
    
    # Bilateral arrangement at 45-degree angles (common in biological systems)
    "BILATERAL": np.array([
        [0.707,  0.707],  # Forward-left at 45°
        [0.707, -0.707],  # Forward-right at -45°
    ]),
    
    # Four sensors in cardinal directions
    "CARDINAL": np.array([
        [1.0,  0.0],  # Forward
        [0.0,  1.0],  # Left
        [-1.0, 0.0],  # Backward
        [0.0, -1.0],  # Right
    ]),
    
    # Six sensors in hexagonal arrangement
    "HEXAGONAL": np.array([
        [1.0,    0.0],       # 0°
        [0.5,    0.866],     # 60°
        [-0.5,   0.866],     # 120°
        [-1.0,   0.0],       # 180°
        [-0.5,  -0.866],     # 240°
        [0.5,   -0.866],     # 300°
    ]),
    
    # Linear array extending forward from agent
    "FORWARD": np.array([
        [0.5, 0.0],  # Close forward sensor
        [1.0, 0.0],  # Mid forward sensor
        [1.5, 0.0],  # Far forward sensor
    ]),
    
    # Radial arrangement with 8 sensors
    "RADIAL": np.array([
        [1.0,    0.0],       # 0°
        [0.707,  0.707],     # 45°
        [0.0,    1.0],       # 90°
        [-0.707, 0.707],     # 135°
        [-1.0,   0.0],       # 180°
        [-0.707, -0.707],    # 225°
        [0.0,    -1.0],      # 270°
        [0.707,  -0.707],    # 315°
    ]),
}


def get_predefined_sensor_layout(
    layout_name: str,
    distance: float = 5.0
) -> np.ndarray:
    """
    Retrieve a predefined sensor layout scaled by the specified distance.

    Parameters
    ----------
    layout_name : str
        Name of the predefined layout. Must be a key in PREDEFINED_SENSOR_LAYOUTS.
        Available layouts: SINGLE, LEFT_RIGHT, FRONT_SIDES, BILATERAL, CARDINAL,
        HEXAGONAL, FORWARD, RADIAL.
    distance : float, default 5.0
        Scaling distance from agent center to sensors in environment units.

    Returns
    -------
    np.ndarray
        Sensor offsets with shape (num_sensors, 2). Each row contains the
        local (x, y) offset for a sensor in the agent's coordinate frame.

    Raises
    ------
    ValueError
        If layout_name is not found in predefined layouts.

    Examples
    --------
    >>> offsets = get_predefined_sensor_layout("LEFT_RIGHT", distance=3.0)
    >>> print(offsets)
    [[ 0.  3.]
     [ 0. -3.]]
    
    >>> offsets = get_predefined_sensor_layout("BILATERAL", distance=5.0)
    >>> print(offsets.shape)
    (2, 2)
    """
    if layout_name not in PREDEFINED_SENSOR_LAYOUTS:
        available_layouts = list(PREDEFINED_SENSOR_LAYOUTS.keys())
        raise ValueError(
            f"Unknown sensor layout: '{layout_name}'. "
            f"Available layouts: {available_layouts}"
        )

    # Get base layout and scale by distance
    base_layout = PREDEFINED_SENSOR_LAYOUTS[layout_name]
    return base_layout * distance


def define_custom_sensor_offsets(
    num_sensors: int,
    distance: float,
    angle: float,
    start_angle: float = 0.0
) -> np.ndarray:
    """
    Create custom sensor offsets in the agent's local coordinate frame.
    
    Sensors are distributed evenly with the specified angular separation,
    starting from the start_angle and extending symmetrically around the agent.

    Parameters
    ----------
    num_sensors : int
        Number of sensors to create. Must be positive.
    distance : float
        Distance from agent center to each sensor in environment units.
    angle : float
        Angular separation between adjacent sensors in degrees.
    start_angle : float, default 0.0
        Starting angle for sensor placement in degrees. 0° points forward.

    Returns
    -------
    np.ndarray
        Sensor offsets with shape (num_sensors, 2). Each row contains
        the (x, y) offset in the agent's local coordinate frame.

    Examples
    --------
    >>> # Create 3 sensors with 45° separation
    >>> offsets = define_custom_sensor_offsets(3, distance=5.0, angle=45.0)
    >>> print(offsets.shape)
    (3, 2)
    
    >>> # Create forward-facing sensors starting at -30°
    >>> offsets = define_custom_sensor_offsets(
    ...     4, distance=3.0, angle=20.0, start_angle=-30.0
    ... )
    """
    if num_sensors <= 0:
        raise ValueError("num_sensors must be positive")
    
    if num_sensors == 1:
        # Single sensor at start_angle
        angle_rad = np.deg2rad(start_angle)
        return np.array([[
            distance * np.cos(angle_rad),
            distance * np.sin(angle_rad)
        ]])
    
    # Calculate symmetric distribution around start_angle
    total_angle_range = (num_sensors - 1) * angle
    half_range = total_angle_range / 2
    first_angle = start_angle - half_range
    
    # Generate sensor offsets
    offsets = np.zeros((num_sensors, 2))
    for i in range(num_sensors):
        current_angle = first_angle + i * angle
        angle_rad = np.deg2rad(current_angle)
        
        offsets[i, 0] = distance * np.cos(angle_rad)  # x component
        offsets[i, 1] = distance * np.sin(angle_rad)  # y component
    
    return offsets


def rotate_sensor_offset(
    local_offset: np.ndarray, 
    orientation_deg: float
) -> np.ndarray:
    """
    Rotate a local sensor offset by an agent's orientation.
    
    Transforms sensor positions from the agent's local coordinate frame
    to the global coordinate frame using a 2D rotation matrix.

    Parameters
    ----------
    local_offset : np.ndarray
        Local offset with shape (2,) representing (x, y) in agent coordinates.
    orientation_deg : float
        Agent's orientation in degrees (0° = positive x-axis).

    Returns
    -------
    np.ndarray
        Global offset with shape (2,) representing the transformed (x, y).

    Notes
    -----
    The rotation follows standard mathematical conventions:
    - Positive angles rotate counter-clockwise
    - 0° orientation points in the positive x-direction

    Examples
    --------
    >>> local_offset = np.array([1.0, 0.0])  # Forward sensor
    >>> global_offset = rotate_sensor_offset(local_offset, 90.0)  # Rotate 90°
    >>> print(np.round(global_offset, 2))
    [0. 1.]
    """
    orientation_rad = np.deg2rad(orientation_deg)
    
    # 2D rotation matrix
    cos_theta = np.cos(orientation_rad)
    sin_theta = np.sin(orientation_rad)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta]
    ])
    
    return rotation_matrix @ local_offset


def calculate_sensor_positions(
    navigator: NavigatorProtocol,
    sensor_distance: float = 5.0,
    sensor_angle: float = 45.0,
    num_sensors: int = 2,
    layout_name: Optional[str] = None,
    start_angle: float = 0.0
) -> np.ndarray:
    """
    Calculate global sensor positions for all agents in a navigator.
    
    Computes sensor positions by transforming local sensor geometry to global
    coordinates based on each agent's position and orientation. Supports both
    predefined layouts and custom sensor configurations.

    Parameters
    ----------
    navigator : NavigatorProtocol
        Navigator instance containing agent positions and orientations.
    sensor_distance : float, default 5.0
        Distance from agent center to sensors in environment units.
    sensor_angle : float, default 45.0
        Angular separation between adjacent sensors in degrees.
        Ignored if layout_name is provided.
    num_sensors : int, default 2
        Number of sensors per agent. Ignored if layout_name is provided.
    layout_name : str, optional
        Name of predefined sensor layout. If provided, overrides sensor_angle
        and num_sensors parameters.
    start_angle : float, default 0.0
        Starting angle for custom sensor arrangements in degrees.

    Returns
    -------
    np.ndarray
        Global sensor positions with shape (num_agents, num_sensors, 2).
        Each element contains the (x, y) position of a sensor in global coordinates.

    Examples
    --------
    >>> # Using predefined layout
    >>> positions = calculate_sensor_positions(navigator, layout_name="LEFT_RIGHT")
    >>> print(positions.shape)  # (num_agents, 2, 2)
    
    >>> # Using custom configuration
    >>> positions = calculate_sensor_positions(
    ...     navigator,
    ...     sensor_distance=8.0,
    ...     sensor_angle=30.0,
    ...     num_sensors=4
    ... )
    
    Notes
    -----
    This function is the primary interface for sensor position calculation and
    integrates seamlessly with Hydra configuration systems for parameter management.
    """
    # Get agent positions and orientations
    agent_positions = navigator.positions
    agent_orientations = navigator.orientations
    
    # Ensure positions are 2D array (num_agents, 2)
    if agent_positions.ndim == 1:
        agent_positions = agent_positions.reshape(1, -1)
    if agent_orientations.ndim == 0:
        agent_orientations = np.array([agent_orientations])
    
    # Determine sensor layout
    if layout_name is not None:
        local_offsets = get_predefined_sensor_layout(layout_name, sensor_distance)
        actual_num_sensors = local_offsets.shape[0]
    else:
        local_offsets = define_custom_sensor_offsets(
            num_sensors, sensor_distance, sensor_angle, start_angle
        )
        actual_num_sensors = num_sensors
    
    num_agents = agent_positions.shape[0]
    sensor_positions = np.zeros((num_agents, actual_num_sensors, 2))
    
    # Calculate sensor positions for each agent
    for agent_idx in range(num_agents):
        agent_pos = agent_positions[agent_idx]
        agent_orientation = agent_orientations[agent_idx]
        
        for sensor_idx in range(actual_num_sensors):
            local_offset = local_offsets[sensor_idx]
            # Transform to global coordinates
            global_offset = rotate_sensor_offset(local_offset, agent_orientation)
            sensor_positions[agent_idx, sensor_idx] = agent_pos + global_offset
    
    return sensor_positions


def sample_odor_with_interpolation(
    env_array: np.ndarray,
    positions: np.ndarray,
    interpolation: str = "bilinear"
) -> np.ndarray:
    """
    Sample odor values from environment array with sub-pixel interpolation.
    
    Provides accurate odor sampling at arbitrary positions using interpolation
    methods to handle non-integer coordinates. Includes comprehensive bounds
    checking and data type normalization.

    Parameters
    ----------
    env_array : np.ndarray
        Environment array with shape (height, width) containing odor data.
        Supports uint8 (automatically normalized to [0,1]) and float types.
    positions : np.ndarray
        Positions to sample with shape (N, 2) where each row is (x, y).
    interpolation : str, default "bilinear"
        Interpolation method. Currently supports "bilinear" and "nearest".

    Returns
    -------
    np.ndarray
        Odor values with shape (N,) corresponding to input positions.
        Out-of-bounds positions return 0.0.

    Examples
    --------
    >>> env = np.random.rand(100, 100)
    >>> positions = np.array([[25.5, 30.7], [50.0, 50.0]])
    >>> odor_values = sample_odor_with_interpolation(env, positions)
    >>> print(odor_values.shape)
    (2,)
    
    Notes
    -----
    Bilinear interpolation provides smooth odor gradients essential for
    gradient-following navigation algorithms.
    """
    # Handle mock objects or invalid arrays
    if hasattr(env_array, 'current_frame'):
        env_array = env_array.current_frame
    
    if not hasattr(env_array, 'shape') or len(env_array.shape) < 2:
        return np.zeros(positions.shape[0])
    
    height, width = env_array.shape[:2]
    num_positions = positions.shape[0]
    odor_values = np.zeros(num_positions)
    
    for i in range(num_positions):
        x, y = positions[i]
        
        # Check bounds
        if x < 0 or x >= width or y < 0 or y >= height:
            odor_values[i] = 0.0
            continue
        
        if interpolation == "nearest":
            # Nearest neighbor interpolation
            x_idx = int(round(x))
            y_idx = int(round(y))
            x_idx = np.clip(x_idx, 0, width - 1)
            y_idx = np.clip(y_idx, 0, height - 1)
            odor_values[i] = env_array[y_idx, x_idx]
            
        elif interpolation == "bilinear":
            # Bilinear interpolation
            x0 = int(np.floor(x))
            x1 = min(x0 + 1, width - 1)
            y0 = int(np.floor(y))
            y1 = min(y0 + 1, height - 1)
            
            # Calculate interpolation weights
            wx = x - x0
            wy = y - y0
            
            # Get corner values
            q00 = env_array[y0, x0]
            q01 = env_array[y1, x0]
            q10 = env_array[y0, x1]
            q11 = env_array[y1, x1]
            
            # Bilinear interpolation
            odor_values[i] = (
                q00 * (1 - wx) * (1 - wy) +
                q10 * wx * (1 - wy) +
                q01 * (1 - wx) * wy +
                q11 * wx * wy
            )
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation}")
        
        # Normalize uint8 to [0, 1]
        if hasattr(env_array, 'dtype') and env_array.dtype == np.uint8:
            odor_values[i] /= 255.0
    
    return odor_values


def sample_odor_at_sensors(
    navigator: NavigatorProtocol,
    env_array: np.ndarray,
    sensor_distance: float = 5.0,
    sensor_angle: float = 45.0,
    num_sensors: int = 2,
    layout_name: Optional[str] = None,
    interpolation: str = "bilinear",
    start_angle: float = 0.0
) -> np.ndarray:
    """
    Sample odor values at sensor positions for all agents.
    
    High-level interface that combines sensor position calculation with odor
    sampling, providing comprehensive multi-sensor readings for navigation
    algorithms. Supports both predefined and custom sensor configurations.

    Parameters
    ----------
    navigator : NavigatorProtocol
        Navigator instance containing agent state information.
    env_array : np.ndarray
        Environment array containing odor concentration data.
    sensor_distance : float, default 5.0
        Distance from agent center to sensors.
    sensor_angle : float, default 45.0
        Angular separation between sensors in degrees.
    num_sensors : int, default 2
        Number of sensors per agent.
    layout_name : str, optional
        Predefined sensor layout name. Overrides other parameters if provided.
    interpolation : str, default "bilinear"
        Interpolation method for sub-pixel sampling.
    start_angle : float, default 0.0
        Starting angle for custom sensor arrangements.

    Returns
    -------
    np.ndarray
        Odor readings with shape (num_agents, num_sensors).
        Each element contains the odor concentration at a sensor location.

    Examples
    --------
    >>> # Basic bilateral sensing
    >>> odor_readings = sample_odor_at_sensors(
    ...     navigator, env_array, layout_name="LEFT_RIGHT"
    ... )
    
    >>> # Custom 4-sensor array
    >>> odor_readings = sample_odor_at_sensors(
    ...     navigator, env_array,
    ...     sensor_distance=8.0,
    ...     sensor_angle=30.0,
    ...     num_sensors=4
    ... )
    
    Notes
    -----
    This function is the primary interface for multi-sensor odor sampling and
    integrates seamlessly with navigation algorithms and visualization systems.
    """
    # Calculate sensor positions
    sensor_positions = calculate_sensor_positions(
        navigator=navigator,
        sensor_distance=sensor_distance,
        sensor_angle=sensor_angle,
        num_sensors=num_sensors,
        layout_name=layout_name,
        start_angle=start_angle
    )
    
    # Reshape for batch sampling
    num_agents, actual_num_sensors, _ = sensor_positions.shape
    flat_positions = sensor_positions.reshape(-1, 2)
    
    # Sample odor at all sensor positions
    flat_odor_values = sample_odor_with_interpolation(
        env_array, flat_positions, interpolation
    )
    
    # Reshape back to (num_agents, num_sensors)
    odor_values = flat_odor_values.reshape(num_agents, actual_num_sensors)
    
    return odor_values


def sample_single_antenna_odor(
    navigator: NavigatorProtocol,
    env_array: np.ndarray,
    antenna_distance: float = 5.0,
    interpolation: str = "bilinear"
) -> Union[float, np.ndarray]:
    """
    Sample odor at each agent's primary antenna location.
    
    Provides simplified single-sensor odor reading at a fixed forward position
    relative to each agent. Useful for basic gradient-following algorithms.

    Parameters
    ----------
    navigator : NavigatorProtocol
        Navigator instance with agent state information.
    env_array : np.ndarray
        Environment array containing odor data.
    antenna_distance : float, default 5.0
        Distance from agent center to antenna in forward direction.
    interpolation : str, default "bilinear"
        Interpolation method for sampling.

    Returns
    -------
    Union[float, np.ndarray]
        Antenna odor readings:
        - float: For single agent navigator
        - np.ndarray: For multi-agent navigator with shape (num_agents,)

    Examples
    --------
    >>> # Single agent
    >>> odor_value = sample_single_antenna_odor(single_navigator, env_array)
    >>> print(type(odor_value))
    <class 'float'>
    
    >>> # Multi-agent
    >>> odor_values = sample_single_antenna_odor(multi_navigator, env_array)
    >>> print(odor_values.shape)
    (num_agents,)
    """
    # Use SINGLE layout for antenna positioning
    antenna_readings = sample_odor_at_sensors(
        navigator=navigator,
        env_array=env_array,
        sensor_distance=antenna_distance,
        layout_name="SINGLE",
        interpolation=interpolation
    )
    
    # Return appropriate format based on navigator type
    if navigator.num_agents == 1:
        return antenna_readings[0, 0]  # Single float value
    else:
        return antenna_readings[:, 0]  # Array of values


# Configuration utilities for Hydra integration
def create_sensor_config_from_hydra(
    cfg: Any,
    default_distance: float = 5.0,
    default_angle: float = 45.0,
    default_num_sensors: int = 2
) -> Dict[str, Any]:
    """
    Create sensor configuration dictionary from Hydra config object.
    
    Extracts sensor parameters from Hydra configuration with fallback defaults,
    enabling seamless integration with structured configuration systems.

    Parameters
    ----------
    cfg : Any
        Hydra configuration object or dictionary containing sensor parameters.
    default_distance : float, default 5.0
        Default sensor distance if not specified in config.
    default_angle : float, default 45.0
        Default sensor angle if not specified in config.
    default_num_sensors : int, default 2
        Default number of sensors if not specified in config.

    Returns
    -------
    Dict[str, Any]
        Dictionary of sensor parameters ready for use with sensor functions.

    Examples
    --------
    >>> # Hydra config in conf/config.yaml:
    >>> # sensors:
    >>> #   layout_name: "BILATERAL"
    >>> #   distance: 8.0
    >>> #   interpolation: "bilinear"
    >>> 
    >>> sensor_config = create_sensor_config_from_hydra(cfg.sensors)
    >>> odor_values = sample_odor_at_sensors(navigator, env_array, **sensor_config)
    """
    # Handle both dict and DictConfig objects
    if hasattr(cfg, 'get'):
        # DictConfig or dict-like object
        config = {
            'sensor_distance': cfg.get('distance', default_distance),
            'sensor_angle': cfg.get('angle', default_angle),
            'num_sensors': cfg.get('num_sensors', default_num_sensors),
            'layout_name': cfg.get('layout_name', None),
            'interpolation': cfg.get('interpolation', 'bilinear'),
            'start_angle': cfg.get('start_angle', 0.0),
        }
    else:
        # Fallback for simple dict or missing config
        config = {
            'sensor_distance': default_distance,
            'sensor_angle': default_angle,
            'num_sensors': default_num_sensors,
            'layout_name': None,
            'interpolation': 'bilinear',
            'start_angle': 0.0,
        }
    
    # Remove None values to use function defaults
    return {k: v for k, v in config.items() if v is not None}


# Export public API
__all__ = [
    # Layout and configuration
    "PREDEFINED_SENSOR_LAYOUTS",
    "get_predefined_sensor_layout",
    "define_custom_sensor_offsets",
    
    # Position calculations
    "rotate_sensor_offset",
    "calculate_sensor_positions",
    
    # Odor sampling
    "sample_odor_with_interpolation",
    "sample_odor_at_sensors",
    "sample_single_antenna_odor",
    
    # Hydra integration
    "create_sensor_config_from_hydra",
]