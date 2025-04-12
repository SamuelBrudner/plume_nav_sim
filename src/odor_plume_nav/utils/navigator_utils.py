"""
Utility functions for navigator creation and management.

This module provides helper functions for creating and manipulating
navigator instances.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import suppress
import itertools
import numpy as np

from odor_plume_nav.core.navigator import Navigator
from odor_plume_nav.core.protocols import NavigatorProtocol


# A dictionary of *base* local sensor offsets (in arbitrary units).
# Each row is [x_offset, y_offset] in the agent's local frame:
#   - The agent faces +x
#   - +y is "left" of the agent (using standard math orientation)
PREDEFINED_SENSOR_LAYOUTS: Dict[str, np.ndarray] = {
    "SINGLE": np.array([[0.0, 0.0]]),
    # Left–Right: one sensor at +y, the other at –y
    "LEFT_RIGHT": np.array([
        [0.0,  1.0],
        [0.0, -1.0],
    ]),
    # Example: place one sensor forward, plus left and right.
    "FRONT_SIDES": np.array([
        [1.0,  0.0],
        [0.0,  1.0],
        [0.0, -1.0],
    ]),
}


def normalize_array_parameter(param: Any, num_agents: int) -> Optional[np.ndarray]:
    """
    Normalize a parameter to a numpy array of the appropriate length.
    
    Args:
        param: Parameter value, which can be None, a scalar, a list, or a numpy array
        num_agents: Number of agents to normalize for
        
    Returns:
        Normalized parameter as a numpy array, or None if param is None
    """
    if param is None:
        return None
    
    # Convert to numpy array if not already
    if not isinstance(param, np.ndarray):
        param = np.array(param)
    
    # If it's a scalar, broadcast to the desired length
    if param.ndim == 0:
        param = np.full(num_agents, param)
    
    return param


def create_navigator_from_params(
    positions: Optional[Union[Tuple[float, float], List[Tuple[float, float]], np.ndarray]] = None,
    orientations: Optional[Union[float, List[float], np.ndarray]] = None,
    speeds: Optional[Union[float, List[float], np.ndarray]] = None,
    max_speeds: Optional[Union[float, List[float], np.ndarray]] = None,
    angular_velocities: Optional[Union[float, List[float], np.ndarray]] = None,
) -> Navigator:
    """
    Create a navigator from parameter values.
    
    Args:
        positions: Initial positions of the agents
        orientations: Initial orientations of the agents (in degrees)
        speeds: Initial speeds of the agents
        max_speeds: Maximum speeds of the agents
        angular_velocities: Initial angular velocities of the agents (in degrees per second)
        
    Returns:
        A Navigator instance that automatically handles single or multi-agent scenarios
    """
    # Detect if we're creating a single or multi-agent navigator
    is_multi_agent = False
    num_agents = 1
    
    # If positions is provided and is not a simple (x, y) tuple, it's multi-agent
    if positions is not None:
        if isinstance(positions, np.ndarray) and positions.ndim > 1:
            is_multi_agent = True
            num_agents = positions.shape[0]
        elif isinstance(positions, list) and positions and isinstance(positions[0], (list, tuple)):
            is_multi_agent = True
            num_agents = len(positions)
    
    # For multi-agent mode, normalize parameters to ensure they're arrays of correct length
    if is_multi_agent:
        # Convert orientations, speeds, etc. to arrays if they're scalar values
        orientations = normalize_array_parameter(orientations, num_agents)
        speeds = normalize_array_parameter(speeds, num_agents)
        max_speeds = normalize_array_parameter(max_speeds, num_agents)
        angular_velocities = normalize_array_parameter(angular_velocities, num_agents)
        
        # Create a multi-agent navigator
        return Navigator(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds,
            angular_velocities=angular_velocities
        )
    else:
        # Create a single-agent navigator
        return Navigator(
            position=positions,
            orientation=orientations,
            speed=speeds,
            max_speed=max_speeds,
            angular_velocity=angular_velocities
        )


def get_predefined_sensor_layout(
    layout_name: str,
    distance: float = 5.0
) -> np.ndarray:
    """
    Return a predefined set of sensor offsets (in the agent's local frame),
    scaled by the given distance.

    Parameters
    ----------
    layout_name : str
        Name of the layout; must be a key of PREDEFINED_SENSOR_LAYOUTS.
    distance : float, default=5.0
        Scaling distance.

    Returns
    -------
    offsets : np.ndarray
        Shape (num_sensors, 2). The local (x, y) offsets for each sensor.
    """
    # Get the base layout
    try:
        layout = PREDEFINED_SENSOR_LAYOUTS[layout_name]
    except KeyError as e:
        raise ValueError(
            f"Unknown sensor layout: {layout_name}. "
            f"Available layouts: {list(PREDEFINED_SENSOR_LAYOUTS.keys())}"
        ) from e

    # Scale by the requested distance
    return layout * distance


def define_sensor_offsets(
    num_sensors: int,
    distance: float,
    angle: float
) -> np.ndarray:
    """
    Define sensor offsets in the agent's local coordinate frame,
    where the agent's heading is [1, 0].
    
    Angles are distributed symmetrically around 0 degrees for even counts,
    or from -angle_range/2 to +angle_range/2 for odd counts.
    
    Parameters
    ----------
    num_sensors : int
        Number of sensors to create.
    distance : float
        Distance from agent center to sensors.
    angle : float
        Angular increment between sensors, in degrees.
        If there are n sensors, the total angular range will be:
        (n-1) * angle.
        
    Returns
    -------
    offsets : np.ndarray
        Shape (num_sensors, 2). Each row is the (x, y) offset
        in the agent's local coordinates.
    """
    # Calculate start angle
    total_angle_range = (num_sensors - 1) * angle
    start_angle = -total_angle_range / 2
    
    # Create array to store offsets
    offsets = np.zeros((num_sensors, 2))
    
    # Calculate offset for each sensor
    for i in range(num_sensors):
        # Current angle in degrees, starting from -total_range/2
        current_angle = start_angle + i * angle
        
        # Convert to radians
        current_angle_rad = np.deg2rad(current_angle)
        
        # Calculate offset using polar coordinates
        offsets[i, 0] = distance * np.cos(current_angle_rad)  # x
        offsets[i, 1] = distance * np.sin(current_angle_rad)  # y
        
    return offsets


def rotate_offset(local_offset: np.ndarray, orientation_deg: float) -> np.ndarray:
    """
    Rotate a local offset by an orientation (in degrees).
    
    Parameters
    ----------
    local_offset : np.ndarray
        (2,) array representing (x, y) in local coordinates.
    orientation_deg : float
        Agent's orientation in degrees.

    Returns
    -------
    global_offset : np.ndarray
        (2,) array representing the offset in global coordinates.
    """
    # Convert to radians
    orientation_rad = np.deg2rad(orientation_deg)
    
    # Create rotation matrix
    rotation_matrix = np.array([
        [np.cos(orientation_rad), -np.sin(orientation_rad)],
        [np.sin(orientation_rad),  np.cos(orientation_rad)]
    ])
    
    # Apply rotation
    return rotation_matrix @ local_offset


def calculate_sensor_positions(
    navigator: NavigatorProtocol,
    sensor_distance: float = 5.0,
    sensor_angle: float = 45.0,
    num_sensors: int = 2,
    layout_name: Optional[str] = None
) -> np.ndarray:
    """
    Calculate sensor positions given a local sensor geometry and each agent's
    global position/orientation.

    Parameters
    ----------
    navigator : NavigatorProtocol
        Navigator with position and orientation information
    sensor_distance : float
        Distance from agent center to each sensor
    sensor_angle : float
        Angle between adjacent sensors in degrees
    num_sensors : int
        Number of sensors per agent
    layout_name : str, optional
        Name of a predefined sensor layout. If provided, sensor_angle is ignored.

    Returns
    -------
    sensor_positions : np.ndarray
        Shape (num_agents, num_sensors, 2). Each row is the (x, y)
        position of a sensor in global coordinates.
    """
    return compute_sensor_positions(
        navigator.positions,
        navigator.orientations,
        layout_name=layout_name,
        distance=sensor_distance,
        angle=sensor_angle,
        num_sensors=num_sensors
    )


def compute_sensor_positions(
    agent_positions: np.ndarray,
    agent_orientations: np.ndarray,
    layout_name: str = None,
    distance: float = 5.0,
    angle: float = 45.0,
    num_sensors: int = 2
) -> np.ndarray:
    """
    Compute sensor positions in global coordinates, given each agent's position
    and orientation, and either a named sensor layout or parameters to create one.

    Parameters
    ----------
    agent_positions : np.ndarray
        Shape (num_agents, 2). The (x, y) positions of each agent.
    agent_orientations : np.ndarray
        Shape (num_agents,). The orientation in degrees of each agent.
    layout_name : str, optional
        Name of a predefined sensor layout. If provided, `angle` and 
        `num_sensors` will be ignored.
    distance : float
        Distance from agent center to each sensor.
    angle : float
        Angular increment between sensors in degrees.
    num_sensors : int
        Number of sensors per agent, used only if layout_name is None.

    Returns
    -------
    sensor_positions : np.ndarray
        Shape (num_agents, num_sensors, 2). The global positions of each sensor.
    """
    # 1. Get the local offsets
    if layout_name is not None:
        local_offsets = get_predefined_sensor_layout(layout_name, distance=distance)
        num_sensors = local_offsets.shape[0]  # Update num_sensors based on layout
    else:
        local_offsets = define_sensor_offsets(num_sensors, distance, angle)

    num_agents = agent_positions.shape[0]
    sensor_positions = np.zeros((num_agents, num_sensors, 2), dtype=float)
    
    # 2. For each agent, rotate each offset by orientation & add the agent's position
    for agent_idx in range(num_agents):
        # Current agent's global position and heading
        agent_pos = agent_positions[agent_idx]
        agent_orientation = agent_orientations[agent_idx]
        
        for sensor_idx in range(num_sensors):
            local_offset = local_offsets[sensor_idx]
            rotated = rotate_offset(local_offset, agent_orientation)
            sensor_positions[agent_idx, sensor_idx] = agent_pos + rotated

    return sensor_positions


def sample_odor_at_sensors(
    navigator: NavigatorProtocol,
    env_array: np.ndarray,
    sensor_distance: float = 5.0,
    sensor_angle: float = 45.0,
    num_sensors: int = 2,
    layout_name: Optional[str] = None
) -> np.ndarray:
    """
    Sample odor values at sensor positions for all navigators.
    
    Args:
        navigator: Navigator instance
        env_array: 2D array representing the environment (e.g., video frame)
        sensor_distance: Distance of sensors from navigator position
        sensor_angle: Angle between sensors in degrees
        num_sensors: Number of sensors per navigator
        layout_name: If provided, use this predefined sensor layout instead of 
                    creating one based on num_sensors and sensor_angle
        
    Returns:
        Array of odor readings with shape (num_agents, num_sensors)
    """
    # Calculate sensor positions
    sensor_positions = calculate_sensor_positions(
        navigator, sensor_distance, sensor_angle, num_sensors, layout_name
    )
    
    # Check if this is a mock plume object (for testing)
    if hasattr(env_array, 'current_frame'):
        env_array = env_array.current_frame
    
    # Get dimensions of environment array or handle mock objects
    if hasattr(env_array, 'shape') and len(env_array.shape) >= 2:
        # Regular NumPy array with valid shape
        height, width = env_array.shape[:2]
    else:
        # For mock objects in tests or arrays without shape
        # Return zeros for all sensors and agents
        return np.zeros((navigator.num_agents, sensor_positions.shape[1]))
    
    # Initialize odor values
    num_agents = navigator.num_agents
    num_sensors = sensor_positions.shape[1]
    odor_values = np.zeros((num_agents, num_sensors))
    
    # Sample odor at each sensor position using itertools.product
    for agent_idx, sensor_idx in itertools.product(range(num_agents), range(num_sensors)):
        # Get sensor position
        x_pos = int(sensor_positions[agent_idx, sensor_idx, 0])
        y_pos = int(sensor_positions[agent_idx, sensor_idx, 1])
        
        # Check if position is within bounds
        if 0 <= x_pos < width and 0 <= y_pos < height:
            with suppress(IndexError, TypeError):
                # Note: NumPy indexing is [y, x]
                odor_value = env_array[y_pos, x_pos]
                
                # Normalize if uint8 array (likely from video frame)
                odor_values[agent_idx, sensor_idx] = (
                    odor_value / 255.0 if hasattr(env_array, 'dtype') and env_array.dtype == np.uint8
                    else odor_value
                )
    
    return odor_values
