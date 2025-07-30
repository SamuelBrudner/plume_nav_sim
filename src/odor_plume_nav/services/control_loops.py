"""Controller implementations for the navigator system.

This module contains the concrete implementations of the NavigatorProtocol,
providing specialized controllers for single and multi-agent navigation.
"""

import contextlib
from typing import Optional, Union, Any, Tuple, List
import numpy as np

class SingleAgentController:
    """Controller for single agent navigation.
    
    This implements the NavigatorProtocol for a single agent case, 
    simplifying navigation logic without conditional branches.
    """
    
    def __init__(
        self,
        position: Optional[Tuple[float, float]] = None,
        orientation: float = 0.0,
        speed: float = 0.0,
        max_speed: float = 1.0,
        angular_velocity: float = 0.0
    ) -> None:
        """Initialize a single agent controller.
        
        Parameters
        ----------
        position : Optional[Tuple[float, float]], optional
            Initial (x, y) position, by default None which becomes (0, 0)
        orientation : float, optional
            Initial orientation in degrees, by default 0.0
        speed : float, optional
            Initial speed, by default 0.0
        max_speed : float, optional
            Maximum allowed speed, by default 1.0
        angular_velocity : float, optional
            Initial angular velocity in degrees/second, by default 0.0
        """
        self._position = np.array([position], dtype=np.float64) if position is not None else np.array([[0.0, 0.0]], dtype=np.float64)
        self._orientation = np.array([orientation], dtype=np.float64)
        self._speed = np.array([speed])
        self._max_speed = np.array([max_speed])
        self._angular_velocity = np.array([angular_velocity])
        
        # Initialize sensor attributes for test compatibility
        self._sensors = []  # List of sensor configurations
        self._primary_sensor = None  # Primary sensor for odor sampling
    
    @property
    def positions(self) -> np.ndarray:
        """Get agent position as a numpy array with shape (1, 2)."""
        return self._position
    
    @property
    def orientations(self) -> np.ndarray:
        """Get agent orientation as a numpy array with shape (1,)."""
        return self._orientation
    
    @property
    def speeds(self) -> np.ndarray:
        """Get agent speed as a numpy array with shape (1,)."""
        return self._speed
    
    @property
    def max_speeds(self) -> np.ndarray:
        """Get maximum agent speed as a numpy array with shape (1,)."""
        return self._max_speed
    
    @property
    def angular_velocities(self) -> np.ndarray:
        """Get agent angular velocity as a numpy array with shape (1,)."""
        return self._angular_velocity
    
    @property
    def num_agents(self) -> int:
        """Get the number of agents, always 1 for SingleAgentController."""
        return 1
    
    def reset(self, **kwargs: Any) -> None:
        """Reset the agent to initial state.
        
        Parameters
        ----------
        **kwargs
            Optional parameters to override initial settings.
            Valid keys are:
            - position: Tuple[float, float] or array-like
            - orientation: float
            - speed: float
            - max_speed: float
            - angular_velocity: float
        
        Notes
        -----
        For stricter type checking, you can use the SingleAgentParams dataclass:
        
        ```python
        from odor_plume_nav.utils.navigator_utils import SingleAgentParams
        
        params = SingleAgentParams(position=(10, 20), speed=1.5)
        navigator.reset_with_params(params)
        ```
        """
        # Import at function level to avoid circular import
        from odor_plume_nav.utils.navigator_utils import reset_navigator_state
        
        # Create a dictionary of current state
        controller_state = {
            '_position': self._position,
            '_orientation': self._orientation,
            '_speed': self._speed,
            '_max_speed': self._max_speed,
            '_angular_velocity': self._angular_velocity
        }
        
        # Use the utility function to reset state
        reset_navigator_state(controller_state, is_single_agent=True, **kwargs)
        
        # Update instance attributes
        self._position = controller_state['_position']
        self._orientation = controller_state['_orientation']
        self._speed = controller_state['_speed']
        self._max_speed = controller_state['_max_speed']
        self._angular_velocity = controller_state['_angular_velocity']
    
    def reset_with_params(self, params: 'SingleAgentParams') -> None:
        """Reset the agent to initial state using a type-safe parameter object.
        
        This method provides stronger type checking than the kwargs-based reset method.
        
        Parameters
        ----------
        params : SingleAgentParams
            Parameters to update, as a dataclass instance
        """
        # Import at function level to avoid circular import
        from odor_plume_nav.utils.navigator_utils import reset_navigator_state_with_params, SingleAgentParams
        
        # Create a dictionary of current state
        controller_state = {
            '_position': self._position,
            '_orientation': self._orientation,
            '_speed': self._speed,
            '_max_speed': self._max_speed,
            '_angular_velocity': self._angular_velocity
        }
        
        # Use the utility function to reset state
        reset_navigator_state_with_params(controller_state, is_single_agent=True, params=params)
        
        # Update instance attributes
        self._position = controller_state['_position']
        self._orientation = controller_state['_orientation']
        self._speed = controller_state['_speed']
        self._max_speed = controller_state['_max_speed']
        self._angular_velocity = controller_state['_angular_velocity']
    
    def step(self, env_array: np.ndarray, dt: float = 1.0) -> None:
        """Take a simulation step to update agent position and orientation.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array (e.g., odor concentration grid)
        dt : float, optional
            Time step size in seconds, by default 1.0
        """
        # Import at function level to avoid circular import
        from odor_plume_nav.utils.navigator_utils import update_positions_and_orientations
        
        # Use the utility function to update position and orientation
        update_positions_and_orientations(
            self._position, 
            self._orientation, 
            self._speed, 
            self._angular_velocity,
            dt=dt
        )
    
    def sample_odor(self, env_array: np.ndarray) -> float:
        """Sample odor at the current agent position.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array
            
        Returns
        -------
        float
            Odor value at the agent's position
        """
        return self.read_single_antenna_odor(env_array)
    
    def read_single_antenna_odor(self, env_array: np.ndarray) -> float:
        """Sample odor at the agent's single antenna.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array
            
        Returns
        -------
        float
            Odor value at the agent's position
        """
        # Import at function level to avoid circular import
        from odor_plume_nav.utils.navigator_utils import read_odor_values
        
        # Use the utility function to read odor value
        odor_values = read_odor_values(env_array, self._position)
        return float(odor_values[0])
    
    def sample_multiple_sensors(
        self, 
        env_array: np.ndarray, 
        sensor_distance: float = 5.0,
        sensor_angle: float = 45.0,
        num_sensors: int = 2,
        layout_name: Optional[str] = None
    ) -> np.ndarray:
        """Sample odor at multiple sensor positions.
        
        Parameters
        ----------
        env_array : np.ndarray
            Environment array
        sensor_distance : float, optional
            Distance from agent to each sensor, by default 5.0
        sensor_angle : float, optional
            Angular separation between sensors in degrees, by default 45.0
        num_sensors : int, optional
            Number of sensors per agent, by default 2
        layout_name : Optional[str], optional
            Predefined sensor layout name, by default None
            
        Returns
        -------
        np.ndarray
            Array of shape (num_sensors,) with odor values
        """
        from odor_plume_nav.utils.navigator_utils import sample_odor_at_sensors
        
        # Delegate to utility function and reshape result for a single agent
        odor_values = sample_odor_at_sensors(
            self, 
            env_array, 
            sensor_distance=sensor_distance,
            sensor_angle=sensor_angle, 
            num_sensors=num_sensors,
            layout_name=layout_name
        )
        
        # Return as a 1D array
        return odor_values[0]
    
    def load_memory(self) -> Optional[Any]:
        """Load memory state for the navigator.
        
        Returns:
            Optional[Any]: Memory state or None if not initialized
        """
        return getattr(self, '_memory_state', None)
    
    def save_memory(self, memory_state: Any) -> None:
        """Save memory state for the navigator.
        
        Parameters:
            memory_state: Memory state to save
        """
        self._memory_state = memory_state


class MultiAgentController:
    """Controller for multi-agent navigation.
    
    This implements the NavigatorProtocol for multiple agents,
    with all data represented as arrays without conditional branching.
    """
    
    def __init__(
        self,
        positions: Optional[np.ndarray] = None,
        orientations: Optional[np.ndarray] = None,
        speeds: Optional[np.ndarray] = None,
        max_speeds: Optional[np.ndarray] = None,
        angular_velocities: Optional[np.ndarray] = None
    ) -> None:
        """Initialize a multi-agent controller.
        
        Parameters
        ----------
        positions : Optional[np.ndarray], optional
            Array of (x, y) positions with shape (num_agents, 2), by default None
        orientations : Optional[np.ndarray], optional
            Array of orientations in degrees with shape (num_agents,), by default None
        speeds : Optional[np.ndarray], optional
            Array of speeds with shape (num_agents,), by default None
        max_speeds : Optional[np.ndarray], optional
            Array of maximum speeds with shape (num_agents,), by default None
        angular_velocities : Optional[np.ndarray], optional
            Array of angular velocities with shape (num_agents,), by default None
        """
        # Ensure we have at least one agent position
        if positions is None:
            self._positions = np.array([[0.0, 0.0]], dtype=np.float64)
        else:
            self._positions = np.array(positions, dtype=np.float64)

        num_agents = self._positions.shape[0]

        # Set defaults for other parameters if not provided
        self._orientations = np.zeros(num_agents, dtype=np.float64) if orientations is None else np.array(orientations, dtype=np.float64)
        self._speeds = np.zeros(num_agents) if speeds is None else np.array(speeds)
        self._max_speeds = np.ones(num_agents) if max_speeds is None else np.array(max_speeds)
        self._angular_velocities = np.zeros(num_agents) if angular_velocities is None else np.array(angular_velocities)
        
        # Initialize sensor attributes for test compatibility
        self._sensors = []  # List of sensor configurations
        self._primary_sensor = None  # Primary sensor for odor sampling
    
    @property
    def positions(self) -> np.ndarray:
        """Get agent positions as a numpy array with shape (num_agents, 2)."""
        return self._positions
    
    @property
    def orientations(self) -> np.ndarray:
        """Get agent orientations as a numpy array with shape (num_agents,)."""
        return self._orientations
    
    @property
    def speeds(self) -> np.ndarray:
        """Get agent speeds as a numpy array with shape (num_agents,)."""
        return self._speeds
    
    @property
    def max_speeds(self) -> np.ndarray:
        """Get maximum agent speeds as a numpy array with shape (num_agents,)."""
        return self._max_speeds
    
    @property
    def angular_velocities(self) -> np.ndarray:
        """Get agent angular velocities as a numpy array with shape (num_agents,)."""
        return self._angular_velocities
    
    @property
    def num_agents(self) -> int:
        """Get the number of agents."""
        return self._positions.shape[0]
    
    def reset(self, **kwargs: Any) -> None:
        """Reset all agents to initial state or update with new settings.
        
        Parameters
        ----------
        **kwargs
            Optional parameters to override initial settings.
            Valid keys are:
            - positions: np.ndarray of shape (N, 2)
            - orientations: np.ndarray of shape (N,)
            - speeds: np.ndarray of shape (N,)
            - max_speeds: np.ndarray of shape (N,)
            - angular_velocities: np.ndarray of shape (N,)
            
        Notes
        -----
        For stricter type checking, you can use the MultiAgentParams dataclass:
        
        ```python
        from odor_plume_nav.utils.navigator_utils import MultiAgentParams
        import numpy as np
        
        params = MultiAgentParams(
            positions=np.array([[10, 20], [30, 40]]),
            speeds=np.array([1.5, 2.0])
        )
        navigator.reset_with_params(params)
        ```
        """
        # Import at function level to avoid circular import
        from odor_plume_nav.utils.navigator_utils import reset_navigator_state
        
        # Create a dictionary of current state
        controller_state = {
            '_positions': self._positions,
            '_orientations': self._orientations,
            '_speeds': self._speeds,
            '_max_speeds': self._max_speeds,
            '_angular_velocities': self._angular_velocities
        }
        
        # Use the utility function to reset state
        reset_navigator_state(controller_state, is_single_agent=False, **kwargs)
        
        # Update instance attributes
        self._positions = controller_state['_positions']
        self._orientations = controller_state['_orientations']
        self._speeds = controller_state['_speeds']
        self._max_speeds = controller_state['_max_speeds']
        self._angular_velocities = controller_state['_angular_velocities']
    
    def reset_with_params(self, params: 'MultiAgentParams') -> None:
        """Reset all agents to initial state using a type-safe parameter object.
        
        This method provides stronger type checking than the kwargs-based reset method.
        
        Parameters
        ----------
        params : MultiAgentParams
            Parameters to update, as a dataclass instance
        """
        # Import at function level to avoid circular import
        from odor_plume_nav.utils.navigator_utils import reset_navigator_state_with_params, MultiAgentParams
        
        # Create a dictionary of current state
        controller_state = {
            '_positions': self._positions,
            '_orientations': self._orientations,
            '_speeds': self._speeds,
            '_max_speeds': self._max_speeds,
            '_angular_velocities': self._angular_velocities
        }
        
        # Use the utility function to reset state
        reset_navigator_state_with_params(controller_state, is_single_agent=False, params=params)
        
        # Update instance attributes
        self._positions = controller_state['_positions']
        self._orientations = controller_state['_orientations']
        self._speeds = controller_state['_speeds']
        self._max_speeds = controller_state['_max_speeds']
        self._angular_velocities = controller_state['_angular_velocities']
    
    def step(self, env_array: np.ndarray, dt: float = 1.0) -> None:
        """Take a simulation step to update all agent positions and orientations.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array (e.g., odor concentration grid)
        dt : float, optional
            Time step size in seconds, by default 1.0
        """
        # Import at function level to avoid circular import
        from odor_plume_nav.utils.navigator_utils import update_positions_and_orientations
        
        # Use the utility function to update positions and orientations
        update_positions_and_orientations(
            self._positions, 
            self._orientations, 
            self._speeds, 
            self._angular_velocities,
            dt=dt
        )
    
    def sample_odor(self, env_array: np.ndarray) -> np.ndarray:
        """Sample odor at all agent positions.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array
            
        Returns
        -------
        np.ndarray
            Odor values at each agent's position, shape (num_agents,)
        """
        return self.read_single_antenna_odor(env_array)
    
    def read_single_antenna_odor(self, env_array: np.ndarray) -> np.ndarray:
        """Sample odor at each agent's position.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array
            
        Returns
        -------
        np.ndarray
            Odor values at each agent's position, shape (num_agents,)
        """
        # Import at function level to avoid circular import
        from odor_plume_nav.utils.navigator_utils import read_odor_values
        
        # Use the utility function to read odor values
        return read_odor_values(env_array, self._positions)
    
    def sample_multiple_sensors(
        self, 
        env_array: np.ndarray, 
        sensor_distance: float = 5.0,
        sensor_angle: float = 45.0,
        num_sensors: int = 2,
        layout_name: Optional[str] = None
    ) -> np.ndarray:
        """Sample odor at multiple sensor positions for all agents.
        
        Parameters
        ----------
        env_array : np.ndarray
            Environment array
        sensor_distance : float, optional
            Distance from each agent to each sensor, by default 5.0
        sensor_angle : float, optional
            Angular separation between sensors in degrees, by default 45.0
        num_sensors : int, optional
            Number of sensors per agent, by default 2
        layout_name : Optional[str], optional
            Predefined sensor layout name, by default None
            
        Returns
        -------
        np.ndarray
            Array of shape (num_agents, num_sensors) with odor values
        """
        from odor_plume_nav.utils.navigator_utils import sample_odor_at_sensors
        
        # Delegate to utility function
        return sample_odor_at_sensors(
            self, 
            env_array, 
            sensor_distance=sensor_distance,
            sensor_angle=sensor_angle, 
            num_sensors=num_sensors,
            layout_name=layout_name
        )
    
    def load_memory(self) -> Optional[Any]:
        """Load memory state for the navigator.
        
        Returns:
            Optional[Any]: Memory state or None if not initialized
        """
        return getattr(self, '_memory_state', None)
    
    def save_memory(self, memory_state: Any) -> None:
        """Save memory state for the navigator.
        
        Parameters:
            memory_state: Memory state to save
        """
        self._memory_state = memory_state
