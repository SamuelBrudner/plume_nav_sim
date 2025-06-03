"""Navigator protocol and core navigation interfaces.

This module defines the NavigatorProtocol that prescribes the exact properties
and methods that any concrete Navigator implementation must provide, ensuring
uniform API across single-agent and multi-agent navigation logic.

The protocol supports type safety, IDE tooling requirements, and integrates
with the Hydra configuration system for ML framework compatibility.
"""

from typing import Protocol, Union, Optional, Any, Tuple, List, Dict
from typing_extensions import runtime_checkable
import numpy as np


@runtime_checkable
class NavigatorProtocol(Protocol):
    """Protocol defining the navigator interface for agent navigation.
    
    This protocol prescribes the structural interface that all navigator
    implementations must satisfy, enabling uniform API across single-agent
    and multi-agent navigation scenarios. The protocol supports type safety,
    IDE autocompletion, and runtime consistency validation.
    
    The interface is designed to integrate seamlessly with Hydra configuration
    management and supports both direct instantiation and factory-based creation
    patterns for enhanced ML framework compatibility.
    
    Examples
    --------
    Creating a navigator that satisfies this protocol:
    
    >>> from {{cookiecutter.project_slug}}.api.navigation import create_navigator
    >>> from omegaconf import DictConfig
    >>> cfg = DictConfig({"type": "single", "max_speed": 10.0})
    >>> navigator = create_navigator(cfg)
    >>> isinstance(navigator, NavigatorProtocol)
    True
    
    Notes
    -----
    This protocol is marked with @runtime_checkable to enable isinstance()
    checks at runtime, facilitating dynamic type validation and debugging
    in research environments.
    """
    
    @property
    def positions(self) -> np.ndarray:
        """Get current agent position(s) as a numpy array.
        
        Returns the current positions of all agents in the navigator.
        For single-agent scenarios, returns a single position. For multi-agent
        scenarios, returns positions for all agents.
        
        Returns
        -------
        np.ndarray
            Agent positions with shape:
            - Single agent: (1, 2) representing [x, y]
            - Multiple agents: (num_agents, 2) representing [[x1, y1], [x2, y2], ...]
            
        Notes
        -----
        Position coordinates are in the environment's coordinate system,
        typically with origin at the video frame's top-left corner.
        """
        ...
    
    @property
    def orientations(self) -> np.ndarray:
        """Get current agent orientation(s) in degrees.
        
        Returns the current heading angles of all agents, measured in degrees
        from the positive x-axis (east direction) with counter-clockwise
        positive rotation following standard mathematical convention.
        
        Returns
        -------
        np.ndarray
            Agent orientations with shape:
            - Single agent: (1,) representing angle in degrees
            - Multiple agents: (num_agents,) representing [angle1, angle2, ...]
            
        Notes
        -----
        Orientations are normalized to the range [0, 360) degrees.
        """
        ...
    
    @property
    def speeds(self) -> np.ndarray:
        """Get current agent speed(s) in environment units per timestep.
        
        Returns the current movement speeds of all agents. Speed represents
        the distance traveled per simulation timestep.
        
        Returns
        -------
        np.ndarray
            Current agent speeds with shape:
            - Single agent: (1,) representing current speed
            - Multiple agents: (num_agents,) representing [speed1, speed2, ...]
            
        Notes
        -----
        Speed values are non-negative and should not exceed the corresponding
        max_speeds values for stable navigation behavior.
        """
        ...
    
    @property
    def max_speeds(self) -> np.ndarray:
        """Get maximum agent speed(s) in environment units per timestep.
        
        Returns the maximum allowable speeds for all agents. These values
        constrain the agents' movement capabilities and are used for
        navigation algorithm parameterization.
        
        Returns
        -------
        np.ndarray
            Maximum agent speeds with shape:
            - Single agent: (1,) representing max speed limit
            - Multiple agents: (num_agents,) representing [max_speed1, max_speed2, ...]
            
        Notes
        -----
        Maximum speeds are positive values that define the upper bounds
        for agent movement in each timestep.
        """
        ...
    
    @property
    def angular_velocities(self) -> np.ndarray:
        """Get current agent angular velocity/velocities in degrees per timestep.
        
        Returns the current rotational speeds of all agents, representing
        the rate of change of orientation angle per simulation timestep.
        
        Returns
        -------
        np.ndarray
            Angular velocities with shape:
            - Single agent: (1,) representing rotational speed
            - Multiple agents: (num_agents,) representing [ang_vel1, ang_vel2, ...]
            
        Notes
        -----
        Angular velocities can be positive (counter-clockwise) or negative
        (clockwise) following standard mathematical conventions.
        """
        ...
    
    @property
    def num_agents(self) -> int:
        """Get the number of agents in this navigator.
        
        Returns the total count of agents managed by this navigator instance.
        This value is immutable after navigator initialization.
        
        Returns
        -------
        int
            Number of agents:
            - 1 for single-agent navigators
            - >1 for multi-agent navigators
            
        Notes
        -----
        This property enables polymorphic handling of single and multi-agent
        scenarios without explicit type checking.
        """
        ...
    
    def reset(self, **kwargs: Any) -> None:
        """Reset the navigator to initial state.
        
        Reinitializes all agent positions, orientations, speeds, and other
        state variables to their configured starting values. This method
        is typically called at the beginning of each simulation episode.
        
        Parameters
        ----------
        **kwargs : Any
            Optional parameters to override initial settings:
            - positions: np.ndarray, optional
                Override initial positions
            - orientations: np.ndarray, optional
                Override initial orientations
            - speeds: np.ndarray, optional
                Override initial speeds
            
        Notes
        -----
        The reset operation should be deterministic when using the same
        configuration and random seed values, ensuring reproducible
        experiment conditions.
        """
        ...
    
    def step(self, env_array: np.ndarray) -> None:
        """Take a simulation step to update agent position(s) and orientation(s).
        
        Executes one timestep of the navigation algorithm, updating agent
        states based on the current environment conditions and internal
        navigation logic. This is the primary method for advancing the
        simulation forward in time.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array representing odor concentration or other
            environmental data. Shape should be (height, width) for 2D
            environments, with values typically normalized to [0, 1] range.
            
        Notes
        -----
        The step operation modifies the navigator's internal state, updating
        positions, orientations, and potentially speeds based on the
        navigation algorithm and environmental feedback.
        
        For reproducible behavior, ensure consistent random seeding using
        the seed manager utility before calling this method.
        """
        ...
    
    def sample_odor(self, env_array: np.ndarray) -> Union[float, np.ndarray]:
        """Sample odor concentration at the current agent position(s).
        
        Retrieves odor concentration values from the environment array at
        the current agent locations using appropriate interpolation methods.
        This method provides the primary sensory input for navigation algorithms.
        
        Parameters
        ----------
        env_array : np.ndarray
            Environment array with shape (height, width) containing odor
            concentration data. Values should be normalized appropriately
            for the navigation algorithm.
            
        Returns
        -------
        Union[float, np.ndarray]
            Odor concentration values:
            - Single agent: float representing odor concentration
            - Multiple agents: np.ndarray with shape (num_agents,) containing
              odor concentrations for each agent
              
        Notes
        -----
        Sampling uses bilinear interpolation for sub-pixel accuracy when
        agent positions don't align exactly with array indices. Out-of-bounds
        positions return zero concentration values.
        """
        ...
    
    def read_single_antenna_odor(self, env_array: np.ndarray) -> Union[float, np.ndarray]:
        """Sample odor at the agent's primary antenna location.
        
        Provides odor concentration at a single sensor point for each agent,
        typically located at the agent's front position based on current
        orientation and configured antenna offset.
        
        Parameters
        ----------
        env_array : np.ndarray
            Environment array with odor concentration data, shape (height, width)
            
        Returns
        -------
        Union[float, np.ndarray]
            Antenna odor readings:
            - Single agent: float representing odor at antenna position
            - Multiple agents: np.ndarray with shape (num_agents,) containing
              antenna readings for each agent
              
        Notes
        -----
        The antenna position is calculated based on the agent's current
        position, orientation, and a configured antenna distance offset.
        This simulates a forward-facing chemical sensor.
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
        """Sample odor at multiple sensor positions around each agent.
        
        Performs odor sampling at multiple sensor locations arranged around
        each agent according to the specified geometric configuration.
        This enables sophisticated sensing strategies for navigation algorithms.
        
        Parameters
        ----------
        env_array : np.ndarray
            Environment array containing odor concentration data
        sensor_distance : float, optional
            Distance from agent center to each sensor position, by default 5.0
            Units are in environment coordinates (typically pixels)
        sensor_angle : float, optional
            Angular separation between adjacent sensors in degrees, by default 45.0
            Sensors are arranged symmetrically around the agent's orientation
        num_sensors : int, optional
            Number of sensors per agent, by default 2
            Must be positive integer, typically 2-8 for practical navigation
        layout_name : Optional[str], optional
            Name of predefined sensor layout configuration, by default None
            If specified, overrides distance, angle, and num_sensors parameters
            
        Returns
        -------
        np.ndarray
            Sensor readings with shape:
            - Single agent: (num_sensors,) containing odor values for each sensor
            - Multiple agents: (num_agents, num_sensors) containing sensor arrays
              for each agent
              
        Notes
        -----
        Sensor positions are calculated relative to each agent's current
        position and orientation. The sensor array is typically arranged
        symmetrically around the forward direction to provide bilateral
        sensing capabilities for gradient-following algorithms.
        
        Common predefined layouts include:
        - "bilateral": Two sensors at ±45° from forward direction
        - "radial": Evenly spaced sensors in a circle around the agent
        - "forward": Linear array of sensors extending forward from the agent
        """
        ...