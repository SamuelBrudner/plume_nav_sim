"""
Utility functions for navigator creation and management.

This module provides helper functions for creating and manipulating
navigator instances.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Type
from contextlib import suppress
import itertools
import numpy as np
from dataclasses import dataclass, field

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
) -> NavigatorProtocol:
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

    # Create a multi-agent navigator
    from odor_plume_nav.core.navigator import Navigator
    # For multi-agent mode, normalize parameters to ensure they're arrays of correct length
    if is_multi_agent:
        # Convert orientations, speeds, etc. to arrays if they're scalar values
        orientations = normalize_array_parameter(orientations, num_agents)
        speeds = normalize_array_parameter(speeds, num_agents)
        max_speeds = normalize_array_parameter(max_speeds, num_agents)
        angular_velocities = normalize_array_parameter(angular_velocities, num_agents)

        return Navigator(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds,
            angular_velocities=angular_velocities
        )
    else:
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


def read_odor_values(
    env_array: np.ndarray, 
    positions: np.ndarray
) -> np.ndarray:
    """
    Read odor values from an environment array at specific positions.
    
    This function handles bounds checking, pixel coordinate conversion,
    and normalization of uint8 arrays.
    
    Parameters
    ----------
    env_array : np.ndarray
        Environment array (e.g., odor concentration grid)
    positions : np.ndarray
        Array of positions with shape (N, 2) where each row is (x, y)
        
    Returns
    -------
    np.ndarray
        Array of odor values with shape (N,)
    """
    # Check if this is a mock plume object (for testing)
    if hasattr(env_array, 'current_frame'):
        env_array = env_array.current_frame

    # Get dimensions of environment array
    if not hasattr(env_array, 'shape') or len(env_array.shape) < 2:
        # For mock objects in tests or arrays without shape
        return np.zeros(len(positions))

    height, width = env_array.shape[:2]
    num_positions = positions.shape[0]
    odor_values = np.zeros(num_positions)

    # Convert positions to integers for indexing
    x_pos = np.floor(positions[:, 0]).astype(int)
    y_pos = np.floor(positions[:, 1]).astype(int)

    # Create a mask for positions that are within bounds
    within_bounds = (
        (x_pos >= 0) & (x_pos < width) & (y_pos >= 0) & (y_pos < height)
    )

    # Read values for positions within bounds
    for i in range(num_positions):
        if within_bounds[i]:
            odor_values[i] = env_array[y_pos[i], x_pos[i]]

            # Normalize if uint8
            if hasattr(env_array, 'dtype') and env_array.dtype == np.uint8:
                odor_values[i] /= 255.0

    return odor_values


def update_positions_and_orientations(
    positions: np.ndarray, 
    orientations: np.ndarray, 
    speeds: np.ndarray, 
    angular_velocities: np.ndarray,
    dt: float = 1.0
) -> None:
    """
    Update positions and orientations based on speeds and angular velocities.
    
    This function handles the vectorized movement calculation for single or multiple agents,
    with proper time step scaling. Position updates are scaled by speed * dt, and 
    orientation updates are scaled by angular_velocity * dt.
    
    It modifies the input arrays in-place.
    
    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 2) with agent positions
    orientations : np.ndarray
        Array of shape (N,) with agent orientations in degrees
    speeds : np.ndarray
        Array of shape (N,) with agent speeds (units/second)
    angular_velocities : np.ndarray
        Array of shape (N,) with agent angular velocities in degrees/second
    dt : float, optional
        Time step size in seconds, by default 1.0
        
    Returns
    -------
    None
        The function modifies the input arrays in-place
        
    Notes
    -----
    The default dt=1.0 maintains backward compatibility with existing code
    that doesn't explicitly handle time steps. To properly incorporate physics
    time steps, pass the actual dt value from your simulation.
    """
    # Convert orientations to radians
    rad_orientations = np.radians(orientations)
    
    # Calculate movement deltas, scaled by dt
    dx = speeds * np.cos(rad_orientations) * dt
    dy = speeds * np.sin(rad_orientations) * dt
    
    # Update positions (vectorized for all agents)
    if positions.ndim == 2:
        # For multiple agents: positions has shape (N, 2)
        positions += np.column_stack((dx, dy))
    else:
        # Handle single agent case with different indexing
        for i in range(len(positions)):
            positions[i] += np.array([dx[i], dy[i]])
    
    # Update orientations with angular velocities, scaled by dt
    orientations += angular_velocities * dt
    
    # Wrap orientations to [0, 360) degrees
    orientations %= 360.0


@dataclass
class SingleAgentParams:
    """Parameters for resetting a single agent navigator."""
    position: Optional[Tuple[float, float]] = None
    orientation: Optional[float] = None
    speed: Optional[float] = None
    max_speed: Optional[float] = None
    angular_velocity: Optional[float] = None


@dataclass
class MultiAgentParams:
    """Parameters for resetting a multi-agent navigator."""
    positions: Optional[np.ndarray] = None
    orientations: Optional[np.ndarray] = None
    speeds: Optional[np.ndarray] = None
    max_speeds: Optional[np.ndarray] = None
    angular_velocities: Optional[np.ndarray] = None


def reset_navigator_state(
    controller_state: Dict[str, np.ndarray],
    is_single_agent: bool,
    **kwargs: Any
) -> None:
    """
    Reset navigator controller state based on provided parameters.
    
    This function handles updating controller state arrays from kwargs,
    ensuring proper array shapes and consistent array sizes.
    
    Parameters
    ----------
    controller_state : Dict[str, np.ndarray]
        Dictionary of current controller state arrays, where keys are:
        - '_position'/'_positions': Array of shape (N, 2) or (1, 2)
        - '_orientation'/'_orientations': Array of shape (N,) or (1,)
        - '_speed'/'_speeds': Array of shape (N,) or (1,)
        - '_max_speed'/'_max_speeds': Array of shape (N,) or (1,)
        - '_angular_velocity'/'_angular_velocities': Array of shape (N,) or (1,)
    is_single_agent : bool
        Whether this is a single agent controller
    **kwargs
        Parameters to update. 
        Valid keys for single-agent controllers:
            - 'position': Tuple[float, float] or array-like
            - 'orientation': float
            - 'speed': float
            - 'max_speed': float
            - 'angular_velocity': float
        Valid keys for multi-agent controllers:
            - 'positions': np.ndarray of shape (N, 2)
            - 'orientations': np.ndarray of shape (N,)
            - 'speeds': np.ndarray of shape (N,)
            - 'max_speeds': np.ndarray of shape (N,)
            - 'angular_velocities': np.ndarray of shape (N,)
    
    Returns
    -------
    None
        The function modifies the input state dictionary in-place
    
    Raises
    ------
    ValueError
        If invalid parameter keys are provided
    
    Notes
    -----
    For stronger type safety, consider using the SingleAgentParams or MultiAgentParams
    dataclasses instead of kwargs. Example:
    
    ```python
    params = SingleAgentParams(position=(10, 20), speed=1.5)
    reset_navigator_state_with_params(controller_state, is_single_agent=True, params=params)
    ```
    """
    # Define valid keys and attribute mappings based on controller type
    if is_single_agent:
        position_key = 'position'
        orientation_key = 'orientation'
        speed_key = 'speed'
        max_speed_key = 'max_speed'
        angular_velocity_key = 'angular_velocity'

        # Map state dictionary keys to internal attribute names
        positions_attr = '_position'
        orientations_attr = '_orientation'
        speeds_attr = '_speed'
        max_speeds_attr = '_max_speed'
        angular_velocities_attr = '_angular_velocity'
    else:
        position_key = 'positions'
        orientation_key = 'orientations'
        speed_key = 'speeds'
        max_speed_key = 'max_speeds'
        angular_velocity_key = 'angular_velocities'

        # Map state dictionary keys to internal attribute names
        positions_attr = '_positions'
        orientations_attr = '_orientations'
        speeds_attr = '_speeds'
        max_speeds_attr = '_max_speeds'
        angular_velocities_attr = '_angular_velocities'

    # Define common valid keys and param mapping for both controller types
    valid_keys = {position_key, orientation_key, speed_key, max_speed_key, angular_velocity_key}
    param_mapping = [
        (position_key, positions_attr),
        (orientation_key, orientations_attr),
        (speed_key, speeds_attr),
        (max_speed_key, max_speeds_attr),
        (angular_velocity_key, angular_velocities_attr)
    ]

    if invalid_keys := set(kwargs.keys()) - valid_keys:
        raise ValueError(f"Invalid parameters: {invalid_keys}. Valid keys are: {valid_keys}")

    # Handle position update (which may require resizing other arrays)
    if (position_value := kwargs.get(position_key)) is not None:
        if is_single_agent:
            # Single agent case: wrap in array
            controller_state[positions_attr] = np.array([position_value])
        else:
            # Multi agent case: convert to array
            controller_state[positions_attr] = np.array(position_value)

            # For multi-agent, we may need to resize other arrays
            num_agents = controller_state[positions_attr].shape[0]

            # Resize other arrays if needed
            arrays_to_check = [
                (orientations_attr, np.zeros, num_agents),
                (speeds_attr, np.zeros, num_agents),
                (max_speeds_attr, np.ones, num_agents),
                (angular_velocities_attr, np.zeros, num_agents)
            ]

            for attr_name, default_fn, size in arrays_to_check:
                if attr_name in controller_state and controller_state[attr_name].shape[0] != num_agents:
                    controller_state[attr_name] = default_fn(size)

    # Update other values if provided
    for kwarg_key, attr_key in param_mapping[1:]:  # Skip position which was handled above
        if kwarg_key in kwargs:
            value = kwargs[kwarg_key]
            if is_single_agent:
                controller_state[attr_key] = np.array([value])
            else:
                controller_state[attr_key] = np.array(value)


def reset_navigator_state_with_params(
    controller_state: Dict[str, np.ndarray],
    is_single_agent: bool,
    params: Union[SingleAgentParams, MultiAgentParams]
) -> None:
    """
    Reset navigator controller state using type-safe parameter objects.
    
    This is a type-safe alternative to reset_navigator_state that uses dataclasses
    instead of kwargs for stronger type safety.
    
    Parameters
    ----------
    controller_state : Dict[str, np.ndarray]
        Dictionary of current controller state arrays
    is_single_agent : bool
        Whether this is a single agent controller
    params : Union[SingleAgentParams, MultiAgentParams]
        Parameters to update, as a dataclass instance
    
    Returns
    -------
    None
        The function modifies the input state dictionary in-place
    
    Raises
    ------
    TypeError
        If params is not the correct type for the controller
    """
    # Validate parameter type
    if is_single_agent and not isinstance(params, SingleAgentParams):
        raise TypeError(
            f"Expected SingleAgentParams for single agent controller, got {type(params)}"
        )
    if not is_single_agent and not isinstance(params, MultiAgentParams):
        raise TypeError(
            f"Expected MultiAgentParams for multi-agent controller, got {type(params)}"
        )
    
    # Convert dataclass to dictionary for the existing function
    kwargs = {k: v for k, v in params.__dict__.items() if v is not None}
    
    # Delegate to the existing function
    reset_navigator_state(controller_state, is_single_agent, **kwargs)


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
    
    # Read odor values at sensor positions
    odor_values = read_odor_values(env_array, sensor_positions.reshape(-1, 2))
    
    # Reshape to (num_agents, num_sensors)
    num_agents = navigator.num_agents
    num_sensors = sensor_positions.shape[1]
    odor_values = odor_values.reshape(num_agents, num_sensors)
    
    return odor_values


def get_property_name(is_single_agent: bool, property_name: str) -> str:
    """
    Get the correct attribute name for a property based on controller type.
    
    Parameters
    ----------
    is_single_agent : bool
        Whether this is a single agent controller
    property_name : str
        Base property name (e.g., 'position', 'orientation')
    
    Returns
    -------
    str
        The correct attribute name for the property ('_position' or '_positions')
    """
    suffix = "" if is_single_agent else "s"
    return f"_{property_name}{suffix}"


def get_property_value(controller: Any, property_name: str) -> Union[float, np.ndarray]:
    """
    Get a property value from a controller, handling single vs multi-agent cases.
    
    For single agent controllers, returns a scalar value instead of an array.
    
    Parameters
    ----------
    controller : Any
        The controller instance (SingleAgentController or MultiAgentController)
    property_name : str
        Property name without underscore prefix (e.g., 'position', 'orientation')
    
    Returns
    -------
    Union[float, np.ndarray]
        Property value, as scalar for single agent and array for multi-agent
    """
    is_single_agent = hasattr(controller, '_position')
    attr_name = get_property_name(is_single_agent, property_name)
    
    value = getattr(controller, attr_name)
    
    # Return scalar for single agent with size 1 array, otherwise return the array
    return value[0] if is_single_agent and value.size == 1 else value


def set_property_value(
    controller: Any, 
    property_name: str, 
    value: Union[float, np.ndarray]
) -> None:
    """
    Set a property value on a controller, handling single vs multi-agent cases.
    
    Parameters
    ----------
    controller : Any
        The controller instance (SingleAgentController or MultiAgentController)
    property_name : str
        Property name without underscore prefix (e.g., 'position', 'orientation')
    value : Union[float, np.ndarray]
        Value to set, can be scalar or array
    
    Returns
    -------
    None
        The function modifies the controller in-place
    """
    is_single_agent = hasattr(controller, '_position')
    attr_name = get_property_name(is_single_agent, property_name)
    
    # For single agent, wrap scalar in array
    if is_single_agent and not isinstance(value, np.ndarray):
        value = np.array([value])
    
    setattr(controller, attr_name, value)


def create_single_agent_navigator(
    navigator_class: Type, 
    params: SingleAgentParams
) -> Any:
    """
    Create a single-agent navigator using type-safe parameter object.
    
    Parameters
    ----------
    navigator_class : Type
        The Navigator class to instantiate
    params : SingleAgentParams
        Parameters for creating the navigator
        
    Returns
    -------
    Any
        A navigator instance with a single agent
    """
    # Convert dataclass to dictionary, preserving only non-None values
    kwargs = {k: v for k, v in params.__dict__.items() if v is not None}
    
    # Create navigator instance
    return navigator_class(**kwargs)


def create_multi_agent_navigator(
    navigator_class: Type, 
    params: MultiAgentParams
) -> Any:
    """
    Create a multi-agent navigator using type-safe parameter object.
    
    Parameters
    ----------
    navigator_class : Type
        The Navigator class to instantiate
    params : MultiAgentParams
        Parameters for creating the navigator
        
    Returns
    -------
    Any
        A navigator instance with multiple agents
    """
    # Convert dataclass to dictionary, preserving only non-None values
    kwargs = {k: v for k, v in params.__dict__.items() if v is not None}
    
    # Create navigator instance
    return navigator_class(**kwargs)
