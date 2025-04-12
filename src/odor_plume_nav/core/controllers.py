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
        self._position = np.array([position]) if position is not None else np.array([[0.0, 0.0]])
        self._orientation = np.array([orientation])
        self._speed = np.array([speed])
        self._max_speed = np.array([max_speed])
        self._angular_velocity = np.array([angular_velocity])
    
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
            Optional parameters to override initial settings
        """
        # Reset logic for a single agent
        if 'position' in kwargs:
            self._position = np.array([kwargs['position']])
        if 'orientation' in kwargs:
            self._orientation = np.array([kwargs['orientation']])
        if 'speed' in kwargs:
            self._speed = np.array([kwargs['speed']])
        if 'max_speed' in kwargs:
            self._max_speed = np.array([kwargs['max_speed']])
        if 'angular_velocity' in kwargs:
            self._angular_velocity = np.array([kwargs['angular_velocity']])
    
    def step(self, env_array: np.ndarray) -> None:
        """Take a simulation step to update agent position and orientation.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array (e.g., odor concentration grid)
        """
        # Movement logic for a single agent
        rad_orientation = np.radians(self._orientation[0])
        dx = self._speed[0] * np.cos(rad_orientation)
        dy = self._speed[0] * np.sin(rad_orientation)
        self._position[0] += np.array([dx, dy])
        self._orientation[0] += self._angular_velocity[0]
        
        # Wrap orientation to [0, 360)
        self._orientation[0] = self._orientation[0] % 360.0
    
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
        # Simplified odor sampling at current position
        x, y = int(self._position[0, 0]), int(self._position[0, 1])
        
        # Check bounds
        height, width = env_array.shape[:2]
        if 0 <= x < width and 0 <= y < height:
            odor_value = env_array[y, x]
            # Normalize if uint8
            if hasattr(env_array, 'dtype') and env_array.dtype == np.uint8:
                return float(odor_value) / 255.0
            return float(odor_value)
        return 0.0
    
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
            self._positions = np.array([[0.0, 0.0]])
        else:
            self._positions = np.array(positions)
        
        num_agents = self._positions.shape[0]
        
        # Set defaults for other parameters if not provided
        if orientations is None:
            self._orientations = np.zeros(num_agents)
        else:
            self._orientations = np.array(orientations)
            
        if speeds is None:
            self._speeds = np.zeros(num_agents)
        else:
            self._speeds = np.array(speeds)
            
        if max_speeds is None:
            self._max_speeds = np.ones(num_agents)
        else:
            self._max_speeds = np.array(max_speeds)
            
        if angular_velocities is None:
            self._angular_velocities = np.zeros(num_agents)
        else:
            self._angular_velocities = np.array(angular_velocities)
    
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
        """Reset all agents to initial state.
        
        Parameters
        ----------
        **kwargs
            Optional parameters to override initial settings
        """
        # Reset logic for multiple agents
        if 'positions' in kwargs:
            self._positions = np.array(kwargs['positions'])
            num_agents = self._positions.shape[0]
            
            # Resize other arrays if needed
            if self._orientations.shape[0] != num_agents:
                self._orientations = np.zeros(num_agents)
            if self._speeds.shape[0] != num_agents:
                self._speeds = np.zeros(num_agents)
            if self._max_speeds.shape[0] != num_agents:
                self._max_speeds = np.ones(num_agents)
            if self._angular_velocities.shape[0] != num_agents:
                self._angular_velocities = np.zeros(num_agents)
                
        # Update other arrays if provided
        if 'orientations' in kwargs:
            self._orientations = np.array(kwargs['orientations'])
        if 'speeds' in kwargs:
            self._speeds = np.array(kwargs['speeds'])
        if 'max_speeds' in kwargs:
            self._max_speeds = np.array(kwargs['max_speeds'])
        if 'angular_velocities' in kwargs:
            self._angular_velocities = np.array(kwargs['angular_velocities'])
    
    def step(self, env_array: np.ndarray) -> None:
        """Take a simulation step to update all agent positions and orientations.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array (e.g., odor concentration grid)
        """
        # Movement logic for multiple agents
        # Vectorized computation for efficiency
        rad_orientations = np.radians(self._orientations)
        dx = self._speeds * np.cos(rad_orientations)
        dy = self._speeds * np.sin(rad_orientations)
        
        # Update positions using broadcasting
        self._positions += np.column_stack((dx, dy))
        self._orientations += self._angular_velocities
        
        # Wrap orientations to [0, 360)
        self._orientations = self._orientations % 360.0
    
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
        # Get dimensions
        height, width = env_array.shape[:2]
        num_agents = self.num_agents
        odor_values = np.zeros(num_agents)
        
        # Convert positions to integers for indexing
        x_pos = np.floor(self._positions[:, 0]).astype(int)
        y_pos = np.floor(self._positions[:, 1]).astype(int)
        
        # Check bounds and sample odor for each agent
        within_bounds = (
            (0 <= x_pos) & (x_pos < width) & 
            (0 <= y_pos) & (y_pos < height)
        )
        
        for i in range(num_agents):
            if within_bounds[i]:
                odor_values[i] = env_array[y_pos[i], x_pos[i]]
                
                # Normalize if uint8
                if hasattr(env_array, 'dtype') and env_array.dtype == np.uint8:
                    odor_values[i] /= 255.0
                    
        return odor_values
    
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
