"""Protocol definitions for the navigator system.

This module defines the protocols (interfaces) that navigator controllers
must satisfy, enabling a clear separation of single and multi-agent logic.
"""

from typing import Protocol, Union, Optional, Any, Tuple, List, Dict
import numpy as np


class NavigatorProtocol(Protocol):
    """Protocol defining the navigator interface for agent navigation.
    
    All navigator implementations should satisfy this protocol, which defines
    the common properties and methods required for odor plume navigation.
    """
    
    @property
    def positions(self) -> np.ndarray:
        """Get current agent position(s) as a numpy array.
        
        Returns
        -------
        np.ndarray
            For single agent: shape (1, 2)
            For multiple agents: shape (num_agents, 2)
        """
        ...
    
    @property
    def orientations(self) -> np.ndarray:
        """Get current agent orientation(s) in degrees.
        
        Returns
        -------
        np.ndarray
            For single agent: shape (1,)
            For multiple agents: shape (num_agents,)
        """
        ...
    
    @property
    def speeds(self) -> np.ndarray:
        """Get current agent speed(s).
        
        Returns
        -------
        np.ndarray
            For single agent: shape (1,)
            For multiple agents: shape (num_agents,)
        """
        ...
    
    @property
    def max_speeds(self) -> np.ndarray:
        """Get maximum agent speed(s).
        
        Returns
        -------
        np.ndarray
            For single agent: shape (1,)
            For multiple agents: shape (num_agents,)
        """
        ...
    
    @property
    def angular_velocities(self) -> np.ndarray:
        """Get current agent angular velocity/velocities in degrees per second.
        
        Returns
        -------
        np.ndarray
            For single agent: shape (1,)
            For multiple agents: shape (num_agents,)
        """
        ...
    
    @property
    def num_agents(self) -> int:
        """Get the number of agents.
        
        Returns
        -------
        int
            Number of agents (1 for single agent, >1 for multi-agent)
        """
        ...
    
    def reset(self, **kwargs: Any) -> None:
        """Reset the navigator to initial state.
        
        Parameters
        ----------
        **kwargs
            Optional parameters to override initial settings
        """
        ...
    
    def step(self, env_array: np.ndarray) -> None:
        """Take a simulation step to update agent position(s) and orientation(s).
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array (e.g., odor concentration grid)
        """
        ...
    
    def sample_odor(self, env_array: np.ndarray) -> Union[float, np.ndarray]:
        """Sample odor at the current agent position(s).
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array (e.g., odor concentration grid)
            
        Returns
        -------
        Union[float, np.ndarray]
            For single agent: float odor value
            For multiple agents: np.ndarray of odor values
        """
        ...
    
    def read_single_antenna_odor(self, env_array: np.ndarray) -> Union[float, np.ndarray]:
        """Sample odor at the agent's single antenna.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array
            
        Returns
        -------
        Union[float, np.ndarray]
            For single agent: float odor value
            For multiple agents: np.ndarray of odor values
        """
        ...
    
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
            Distance from navigator to each sensor, by default 5.0
        sensor_angle : float, optional
            Angular separation between sensors in degrees, by default 45.0
        num_sensors : int, optional
            Number of sensors per navigator, by default 2
        layout_name : Optional[str], optional
            Predefined sensor layout name, by default None
            
        Returns
        -------
        np.ndarray
            For single agent: np.ndarray of shape (num_sensors,)
            For multiple agents: np.ndarray of shape (num_agents, num_sensors)
        """
        ...
