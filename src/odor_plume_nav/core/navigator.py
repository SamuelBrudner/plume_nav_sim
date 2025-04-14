"""Navigator class implementing the protocol-based architecture.

This module contains the main Navigator class that delegates to specialized
controllers for single and multi-agent navigation.
"""

from typing import Optional, Union, Any, Tuple, List, Dict, Type, ClassVar
import numpy as np

from odor_plume_nav.core.controllers import SingleAgentController, MultiAgentController


class Navigator:
    """Main navigator class for odor plume navigation.
    
    This class serves as a facade to the underlying controller implementation,
    delegating calls to either a SingleAgentController or MultiAgentController
    based on the initialization parameters.
    """
    
    def __init__(
        self,
        position: Optional[Tuple[float, float]] = None,
        orientation: float = 0.0,
        speed: float = 0.0,
        max_speed: float = 1.0,
        angular_velocity: float = 0.0,
        positions: Optional[np.ndarray] = None,
        orientations: Optional[np.ndarray] = None,
        speeds: Optional[np.ndarray] = None,
        max_speeds: Optional[np.ndarray] = None,
        angular_velocities: Optional[np.ndarray] = None
    ) -> None:
        """Initialize a navigator.
        
        If 'positions' is provided, initializes a multi-agent controller.
        Otherwise, initializes a single-agent controller with the given parameters.
        
        Parameters
        ----------
        position : Optional[Tuple[float, float]], optional
            Initial position for a single agent, by default None
        orientation : float, optional
            Initial orientation for a single agent in degrees, by default 0.0
        speed : float, optional
            Initial speed for a single agent, by default 0.0
        max_speed : float, optional
            Maximum speed for a single agent, by default 1.0
        angular_velocity : float, optional
            Initial angular velocity for a single agent, by default 0.0
        positions : Optional[np.ndarray], optional
            Array of positions for multiple agents, by default None
        orientations : Optional[np.ndarray], optional
            Array of orientations for multiple agents, by default None
        speeds : Optional[np.ndarray], optional
            Array of speeds for multiple agents, by default None
        max_speeds : Optional[np.ndarray], optional
            Array of maximum speeds for multiple agents, by default None
        angular_velocities : Optional[np.ndarray], optional
            Array of angular velocities for multiple agents, by default None
        """
        if positions is not None:
            # Multi-agent mode
            self._controller = MultiAgentController(
                positions=positions,
                orientations=orientations,
                speeds=speeds,
                max_speeds=max_speeds,
                angular_velocities=angular_velocities
            )
            self._is_single_agent = False
        else:
            # Single-agent mode
            self._controller = SingleAgentController(
                position=position,
                orientation=orientation,
                speed=speed,
                max_speed=max_speed,
                angular_velocity=angular_velocity
            )
            self._is_single_agent = True
    
    # Delegate properties to the controller
    
    @property
    def positions(self) -> np.ndarray:
        """Get current agent position(s)."""
        return self._controller.positions
    
    @property
    def orientations(self) -> np.ndarray:
        """Get current agent orientation(s)."""
        return self._controller.orientations
    
    @property
    def speeds(self) -> np.ndarray:
        """Get current agent speed(s)."""
        return self._controller.speeds
    
    @property
    def max_speeds(self) -> np.ndarray:
        """Get maximum agent speed(s)."""
        return self._controller.max_speeds
    
    @property
    def angular_velocities(self) -> np.ndarray:
        """Get current agent angular velocity/velocities."""
        return self._controller.angular_velocities
    
    @property
    def num_agents(self) -> int:
        """Get the number of agents."""
        return self._controller.num_agents
    
    @property
    def is_single_agent(self) -> bool:
        """Check if this is a single-agent navigator."""
        return self._is_single_agent
    
    # Factory methods for clear instantiation
    
    @classmethod
    def single(
        cls,
        position: Optional[Tuple[float, float]] = None,
        orientation: float = 0.0,
        speed: float = 0.0,
        max_speed: float = 1.0,
        angular_velocity: float = 0.0
    ) -> 'Navigator':
        """Create a single-agent navigator.
        
        Parameters
        ----------
        position : Optional[Tuple[float, float]], optional
            Initial position, by default None
        orientation : float, optional
            Initial orientation in degrees, by default 0.0
        speed : float, optional
            Initial speed, by default 0.0
        max_speed : float, optional
            Maximum speed, by default 1.0
        angular_velocity : float, optional
            Initial angular velocity, by default 0.0
            
        Returns
        -------
        Navigator
            A Navigator instance with a single agent
            
        Notes
        -----
        For stricter type checking, you can use the single_from_params method with SingleAgentParams.
        """
        return cls(
            position=position,
            orientation=orientation,
            speed=speed,
            max_speed=max_speed,
            angular_velocity=angular_velocity
        )
    
    @classmethod
    def single_from_params(cls, params: 'SingleAgentParams') -> 'Navigator':
        """Create a single-agent navigator using type-safe parameter object.
        
        This method provides stronger type checking than the positional parameter approach.
        
        Parameters
        ----------
        params : SingleAgentParams
            Parameters for creating the navigator
            
        Returns
        -------
        Navigator
            A Navigator instance with a single agent
            
        Examples
        --------
        >>> from odor_plume_nav.utils.navigator_utils import SingleAgentParams
        >>> params = SingleAgentParams(position=(10, 20), speed=1.5)
        >>> navigator = Navigator.single_from_params(params)
        """
        from odor_plume_nav.utils.navigator_utils import create_single_agent_navigator, SingleAgentParams
        return create_single_agent_navigator(cls, params)
    
    @classmethod
    def multi(
        cls,
        positions: np.ndarray,
        orientations: Optional[np.ndarray] = None,
        speeds: Optional[np.ndarray] = None,
        max_speeds: Optional[np.ndarray] = None,
        angular_velocities: Optional[np.ndarray] = None
    ) -> 'Navigator':
        """Create a multi-agent navigator.
        
        Parameters
        ----------
        positions : np.ndarray
            Array of initial positions, shape (num_agents, 2)
        orientations : Optional[np.ndarray], optional
            Array of initial orientations, by default None
        speeds : Optional[np.ndarray], optional
            Array of initial speeds, by default None
        max_speeds : Optional[np.ndarray], optional
            Array of maximum speeds, by default None
        angular_velocities : Optional[np.ndarray], optional
            Array of initial angular velocities, by default None
            
        Returns
        -------
        Navigator
            A Navigator instance with multiple agents
            
        Notes
        -----
        For stricter type checking, you can use the multi_from_params method with MultiAgentParams.
        """
        return cls(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds,
            angular_velocities=angular_velocities
        )
    
    @classmethod
    def multi_from_params(cls, params: 'MultiAgentParams') -> 'Navigator':
        """Create a multi-agent navigator using type-safe parameter object.
        
        This method provides stronger type checking than the positional parameter approach.
        
        Parameters
        ----------
        params : MultiAgentParams
            Parameters for creating the navigator
            
        Returns
        -------
        Navigator
            A Navigator instance with multiple agents
            
        Examples
        --------
        >>> from odor_plume_nav.utils.navigator_utils import MultiAgentParams
        >>> import numpy as np
        >>> params = MultiAgentParams(
        ...     positions=np.array([[10, 20], [30, 40]]),
        ...     speeds=np.array([1.5, 2.0])
        ... )
        >>> navigator = Navigator.multi_from_params(params)
        """
        from odor_plume_nav.utils.navigator_utils import create_multi_agent_navigator, MultiAgentParams
        return create_multi_agent_navigator(cls, params)
    
    # Factory methods for clear instantiation
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Navigator':
        """Create a navigator from a configuration dictionary.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration parameters for the navigator
            
        Returns
        -------
        Navigator
            Configured navigator instance
        """
        # Determine if it's a single or multi-agent from the config
        if 'positions' in config and isinstance(config['positions'], np.ndarray):
            # Multi-agent mode
            return cls.multi(**config)
        else:
            # Single-agent mode
            return cls.single(**config)
    
    # Delegate methods to the controller
    
    def reset(self, **kwargs: Any) -> None:
        """Reset the navigator to initial state.
        
        Parameters
        ----------
        **kwargs
            Optional parameters to override initial settings
        """
        self._controller.reset(**kwargs)
    
    def step(self, env_array: np.ndarray) -> None:
        """Take a simulation step.
        
        Parameters
        ----------
        env_array : np.ndarray
            Environment array (e.g., odor plume frame)
        """
        self._controller.step(env_array)
    
    def sample_odor(self, env_array: np.ndarray) -> Union[float, np.ndarray]:
        """Sample odor at the current agent position(s).
        
        Parameters
        ----------
        env_array : np.ndarray
            Environment array
            
        Returns
        -------
        Union[float, np.ndarray]
            Odor values
        """
        return self._controller.sample_odor(env_array)
    
    def read_single_antenna_odor(self, env_array: np.ndarray) -> Union[float, np.ndarray]:
        """Sample odor at the agent's single antenna.
        
        Parameters
        ----------
        env_array : np.ndarray
            Environment array
            
        Returns
        -------
        Union[float, np.ndarray]
            Odor values
        """
        return self._controller.read_single_antenna_odor(env_array)
    
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
            Odor values at sensor positions
        """
        return self._controller.sample_multiple_sensors(
            env_array, 
            sensor_distance=sensor_distance,
            sensor_angle=sensor_angle, 
            num_sensors=num_sensors,
            layout_name=layout_name
        )
