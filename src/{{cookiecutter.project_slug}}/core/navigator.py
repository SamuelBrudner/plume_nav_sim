"""
NavigatorProtocol interface defining the structural contract for navigation controllers.

This module implements the core NavigatorProtocol that prescribes the exact properties 
and methods that any concrete Navigator implementation must provide, ensuring uniform 
API across single-agent and multi-agent navigation logic. The protocol supports type 
safety, IDE tooling requirements, and seamless integration with Hydra configuration 
management for enhanced ML framework compatibility.

The protocol-based design enables researchers to implement custom navigation algorithms 
while maintaining compatibility with the existing framework, supporting both research 
extensibility and production-grade type safety.
"""

from __future__ import annotations
from typing import Protocol, Union, Optional, Tuple, List, Any, runtime_checkable
from typing_extensions import Self
import numpy as np

# Hydra imports for configuration integration
try:
    from omegaconf import DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    # Fallback for environments without Hydra
    DictConfig = dict
    HYDRA_AVAILABLE = False

# Import configuration schemas for type hints
if HYDRA_AVAILABLE:
    from ..config.schemas import NavigatorConfig, SingleAgentConfig, MultiAgentConfig


@runtime_checkable
class NavigatorProtocol(Protocol):
    """
    Protocol defining the structural interface for navigator controllers.
    
    This protocol prescribes the exact properties and methods that any concrete 
    Navigator implementation must provide, ensuring uniform API across single-agent 
    and multi-agent navigation logic. The protocol supports type safety, IDE tooling 
    requirements, and Hydra configuration integration for ML framework compatibility.
    
    The protocol-based design enables algorithm extensibility while maintaining 
    compatibility with existing framework components including simulation runners, 
    visualization systems, and data recording infrastructure.
    
    Key Design Principles:
    - Uniform interface for both single and multi-agent scenarios
    - NumPy-based state representation for performance and compatibility
    - Hydra configuration integration for research workflow support  
    - Protocol-based extensibility for custom algorithm implementation
    - Type safety for enhanced IDE tooling and error prevention
    
    Performance Requirements:
    - Step execution: <1ms for single agent, <10ms for 100 agents
    - Memory efficiency: <10MB overhead per 100 agents
    - Vectorized operations for scalable multi-agent performance
    
    Examples:
        Basic protocol compliance check:
        >>> from typing import TYPE_CHECKING
        >>> if TYPE_CHECKING:
        ...     # Type checker validates protocol compliance
        ...     navigator: NavigatorProtocol = create_navigator()
        
        Custom implementation:
        >>> class CustomNavigator:
        ...     def __init__(self):
        ...         self._positions = np.array([[0.0, 0.0]])
        ...         # ... implement all required properties and methods
        ...     
        ...     @property 
        ...     def positions(self) -> np.ndarray:
        ...         return self._positions
        ...     # ... implement remaining protocol methods
        
        Factory method with Hydra integration:
        >>> from hydra import compose, initialize
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        ...     navigator = Navigator.from_config(cfg.navigator)
    """
    
    # Core state properties - must be implemented by all navigators
    
    @property
    def positions(self) -> np.ndarray:
        """
        Get current agent position(s) as numpy array.
        
        Returns:
            np.ndarray: Agent positions with shape:
                - Single agent: (1, 2) for [x, y] coordinates
                - Multi-agent: (n_agents, 2) for [[x1, y1], [x2, y2], ...]
                
        Notes:
            Positions are returned as float64 arrays for numerical precision.
            Coordinates are in the environment's coordinate system (typically
            with origin at top-left for video-based environments).
            
        Performance:
            Property access should be O(1) - no computation during retrieval.
        """
        ...
    
    @property
    def orientations(self) -> np.ndarray:
        """
        Get current agent orientation(s) in degrees.
        
        Returns:
            np.ndarray: Agent orientations with shape:
                - Single agent: (1,) for [orientation]
                - Multi-agent: (n_agents,) for [ori1, ori2, ...]
                
        Notes:
            Orientations are in degrees with 0° = right (positive x-axis),
            90° = up (negative y-axis) following standard navigation conventions.
            Values are normalized to [0, 360) range.
            
        Performance:
            Property access should be O(1) - no computation during retrieval.
        """
        ...
    
    @property  
    def speeds(self) -> np.ndarray:
        """
        Get current agent speed(s) in units per time step.
        
        Returns:
            np.ndarray: Agent speeds with shape:
                - Single agent: (1,) for [speed]
                - Multi-agent: (n_agents,) for [speed1, speed2, ...]
                
        Notes:
            Speeds are non-negative values representing magnitude of velocity.
            Units depend on environment scale (typically pixels per frame).
            Speed values are constrained by max_speeds property.
            
        Performance:
            Property access should be O(1) - no computation during retrieval.
        """
        ...
    
    @property
    def max_speeds(self) -> np.ndarray:
        """
        Get maximum allowed speed(s) for each agent.
        
        Returns:
            np.ndarray: Maximum speeds with shape:
                - Single agent: (1,) for [max_speed]  
                - Multi-agent: (n_agents,) for [max1, max2, ...]
                
        Notes:
            Maximum speeds define upper bounds for agent velocities.
            Current speeds should never exceed corresponding max_speeds.
            Used for constraint validation and control system limits.
            
        Performance:
            Property access should be O(1) - no computation during retrieval.
        """
        ...
    
    @property
    def angular_velocities(self) -> np.ndarray:
        """
        Get current agent angular velocity/velocities in degrees per second.
        
        Returns:
            np.ndarray: Angular velocities with shape:
                - Single agent: (1,) for [angular_velocity]
                - Multi-agent: (n_agents,) for [ang_vel1, ang_vel2, ...]
                
        Notes:
            Positive values indicate counterclockwise rotation.
            Angular velocities are applied during step() method execution.
            Units are degrees per second, scaled by time step dt.
            
        Performance:
            Property access should be O(1) - no computation during retrieval.
        """
        ...
    
    @property
    def num_agents(self) -> int:
        """
        Get the total number of agents managed by this navigator.
        
        Returns:
            int: Number of agents (always >= 1)
            
        Notes:
            This value determines the first dimension of all state arrays.
            Used for validation and vectorized operation sizing.
            Remains constant after navigator initialization.
            
        Performance:
            Property access should be O(1) - typically returns cached value.
        """
        ...
    
    # Configuration and initialization methods
    
    def reset(self, **kwargs: Any) -> None:
        """
        Reset navigator to initial state with optional parameter overrides.
        
        Args:
            **kwargs: Optional parameters to override initial settings.
                Valid keys depend on implementation but typically include:
                - position/positions: New initial position(s)
                - orientation/orientations: New initial orientation(s) 
                - speed/speeds: New initial speed(s)
                - max_speed/max_speeds: New maximum speed(s)
                - angular_velocity/angular_velocities: New angular velocity/velocities
                
        Notes:
            Resets all agent state to initial conditions while preserving
            navigator configuration. Parameter overrides are temporary for
            this reset only and don't permanently modify configuration.
            
            Implementations should validate override parameters match the
            expected agent count and value ranges.
            
        Performance:
            Should complete in <1ms for single agent, <10ms for 100 agents.
            
        Raises:
            ValueError: If override parameters are invalid or incompatible
            TypeError: If override parameter types are incorrect
            
        Examples:
            Reset to initial state:
            >>> navigator.reset()
            
            Reset with new starting position:
            >>> navigator.reset(position=(10.0, 20.0))  # Single agent
            >>> navigator.reset(positions=[[0,0], [10,10]])  # Multi-agent
        """
        ...
    
    # Core simulation methods
    
    def step(self, env_array: np.ndarray, dt: float = 1.0) -> None:
        """
        Execute one simulation time step with environment interaction.
        
        Args:
            env_array: Environment data array (e.g., odor plume frame) with shape
                (height, width) or (height, width, channels). Used for navigation
                decisions and sensor sampling.
            dt: Time step size in seconds (default: 1.0). Scales position updates
                (velocity * dt) and orientation updates (angular_velocity * dt).
                
        Notes:
            Updates agent positions based on current speeds and orientations.
            Updates agent orientations based on angular velocities.
            May sample environment for navigation decisions (implementation-specific).
            
            Position updates follow: new_pos = pos + speed * dt * [cos(θ), sin(θ)]
            Orientation updates follow: new_θ = θ + angular_velocity * dt
            
            Positions are automatically constrained to environment boundaries.
            Speeds are automatically constrained by max_speeds.
            
        Performance:
            Should execute in <1ms for single agent, <10ms for 100 agents.
            Must support 30+ fps simulation for real-time visualization.
            
        Raises:
            ValueError: If env_array shape is incompatible or dt <= 0
            TypeError: If env_array is not a numpy array
            
        Examples:
            Basic simulation step:
            >>> frame = video_plume.get_frame(0)
            >>> navigator.step(frame, dt=1.0)
            
            High-frequency simulation:
            >>> for i in range(1000):
            ...     frame = video_plume.get_frame(i)
            ...     navigator.step(frame, dt=0.1)
        """
        ...
    
    # Environment sensing methods
    
    def sample_odor(self, env_array: np.ndarray) -> Union[float, np.ndarray]:
        """
        Sample odor concentration(s) at current agent position(s).
        
        Args:
            env_array: Environment array containing odor concentration data
                with shape (height, width) or (height, width, channels).
                
        Returns:
            Union[float, np.ndarray]: Odor concentration value(s):
                - Single agent: float value
                - Multi-agent: np.ndarray with shape (n_agents,)
                
        Notes:
            Samples environment data at agent center positions using bilinear
            interpolation for sub-pixel accuracy. Values are normalized to
            [0, 1] range representing odor concentration intensity.
            
            Positions outside environment boundaries return 0.0 concentration.
            Grayscale environments use pixel intensity as concentration.
            Color environments may use specific channels (implementation-specific).
            
        Performance:
            Should execute in <100μs per agent for sub-millisecond total sampling.
            
        Raises:
            ValueError: If env_array shape is incompatible
            TypeError: If env_array is not a numpy array
            IndexError: If agent positions are severely out of bounds
            
        Examples:
            Single agent odor sampling:
            >>> frame = video_plume.get_frame(0)
            >>> concentration = navigator.sample_odor(frame)
            >>> print(f"Odor level: {concentration:.3f}")
            
            Multi-agent batch sampling:
            >>> concentrations = navigator.sample_odor(frame)
            >>> max_concentration = np.max(concentrations)
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
        """
        Sample odor at multiple sensor positions relative to each agent.
        
        Args:
            env_array: Environment array containing odor concentration data.
            sensor_distance: Distance from agent center to each sensor (default: 5.0).
            sensor_angle: Angular separation between sensors in degrees (default: 45.0).
            num_sensors: Number of sensors per agent (default: 2).
            layout_name: Predefined sensor layout name. Options:
                - "LEFT_RIGHT": Two sensors at ±90° from agent orientation
                - "FORWARD_BACK": Sensors at 0° and 180° from agent orientation  
                - "TRIANGLE": Three sensors in triangular arrangement
                - None: Use custom angle-based layout
                
        Returns:
            np.ndarray: Sensor readings with shape:
                - Single agent: (num_sensors,) 
                - Multi-agent: (n_agents, num_sensors)
                
        Notes:
            Sensor positions are calculated relative to agent orientation:
            - sensor_pos = agent_pos + distance * [cos(θ + angle), sin(θ + angle)]
            - Supports biological sensing strategies (antennae, whiskers, etc.)
            - Each sensor uses bilinear interpolation for sub-pixel accuracy
            
            Predefined layouts override sensor_angle and num_sensors parameters.
            Custom layouts distribute sensors evenly around sensor_angle range.
            
        Performance:
            Should execute in <500μs per agent for efficient multi-sensor sampling.
            
        Raises:
            ValueError: If sensor parameters are invalid or layout_name unknown
            TypeError: If env_array is not a numpy array
            
        Examples:
            Bilateral sensing (left-right antennae):
            >>> readings = navigator.sample_multiple_sensors(
            ...     frame, sensor_distance=10.0, layout_name="LEFT_RIGHT"
            ... )
            
            Custom triangular sensor array:
            >>> readings = navigator.sample_multiple_sensors(
            ...     frame, sensor_distance=8.0, sensor_angle=120.0, num_sensors=3
            ... )
            
            Multi-agent sensor array:
            >>> # Returns shape (n_agents, num_sensors) for batch processing
            >>> all_readings = navigator.sample_multiple_sensors(frame)
        """
        ...


# Factory methods and configuration integration

class NavigatorFactory:
    """
    Factory class providing Hydra-integrated navigator creation methods.
    
    This factory enables configuration-driven navigator instantiation with 
    comprehensive parameter validation, type safety, and seamless integration 
    with Hydra configuration management. Supports both programmatic creation 
    and CLI-driven automation workflows.
    
    The factory pattern decouples navigator creation from specific implementation 
    classes, enabling algorithm extensibility while maintaining consistent 
    initialization patterns across the research framework.
    """
    
    @staticmethod
    def from_config(config: Union[DictConfig, NavigatorConfig, dict]) -> NavigatorProtocol:
        """
        Create navigator from Hydra configuration object.
        
        Args:
            config: Configuration object containing navigator parameters.
                Supports DictConfig (Hydra), NavigatorConfig (Pydantic), 
                or plain dict with required fields.
                
        Returns:
            NavigatorProtocol: Configured navigator instance implementing
                the full protocol interface.
                
        Notes:
            Automatically detects single vs multi-agent mode from configuration:
            - Single agent: Uses position, orientation, speed, etc.
            - Multi-agent: Uses positions, orientations, speeds, etc.
            
            Configuration validation ensures parameter compatibility and
            constraint satisfaction before navigator instantiation.
            
        Raises:
            ValueError: If configuration is invalid or incomplete
            TypeError: If configuration type is unsupported
            
        Examples:
            From Hydra configuration:
            >>> @hydra.main(config_path="../conf", config_name="config")
            >>> def main(cfg: DictConfig) -> None:
            ...     navigator = NavigatorFactory.from_config(cfg.navigator)
            
            From Pydantic model:
            >>> from {{cookiecutter.project_slug}}.config.schemas import NavigatorConfig
            >>> config = NavigatorConfig(position=(10.0, 20.0), max_speed=2.0)
            >>> navigator = NavigatorFactory.from_config(config)
        """
        # Import here to avoid circular imports
        from ..core.controllers import SingleAgentController, MultiAgentController
        from ..config.schemas import NavigatorConfig
        
        # Convert to NavigatorConfig if needed for validation
        if isinstance(config, dict):
            config = NavigatorConfig(**config)
        elif hasattr(config, 'to_container') and HYDRA_AVAILABLE:
            # Handle DictConfig from Hydra
            config = NavigatorConfig(**config)
        
        # Determine navigator type and create appropriate implementation
        if _is_multi_agent_config(config):
            return MultiAgentController(
                positions=config.positions,
                orientations=config.orientations,
                speeds=config.speeds,
                max_speeds=config.max_speeds,
                angular_velocities=config.angular_velocities
            )
        else:
            return SingleAgentController(
                position=config.position,
                orientation=config.orientation,
                speed=config.speed,
                max_speed=config.max_speed,
                angular_velocity=config.angular_velocity
            )
    
    @staticmethod
    def single_agent(
        position: Optional[Tuple[float, float]] = None,
        orientation: float = 0.0,
        speed: float = 0.0,
        max_speed: float = 1.0,
        angular_velocity: float = 0.0
    ) -> NavigatorProtocol:
        """
        Create single-agent navigator with explicit parameters.
        
        Args:
            position: Initial (x, y) position (default: (0, 0))
            orientation: Initial orientation in degrees (default: 0.0)
            speed: Initial speed (default: 0.0)
            max_speed: Maximum allowed speed (default: 1.0) 
            angular_velocity: Initial angular velocity in deg/s (default: 0.0)
            
        Returns:
            NavigatorProtocol: Single-agent navigator implementation
            
        Examples:
            Basic single agent:
            >>> navigator = NavigatorFactory.single_agent()
            
            Configured single agent:
            >>> navigator = NavigatorFactory.single_agent(
            ...     position=(50.0, 100.0), max_speed=2.5
            ... )
        """
        from ..core.controllers import SingleAgentController
        
        return SingleAgentController(
            position=position,
            orientation=orientation,
            speed=speed,
            max_speed=max_speed,
            angular_velocity=angular_velocity
        )
    
    @staticmethod
    def multi_agent(
        positions: Union[List[List[float]], np.ndarray],
        orientations: Optional[Union[List[float], np.ndarray]] = None,
        speeds: Optional[Union[List[float], np.ndarray]] = None,
        max_speeds: Optional[Union[List[float], np.ndarray]] = None,
        angular_velocities: Optional[Union[List[float], np.ndarray]] = None
    ) -> NavigatorProtocol:
        """
        Create multi-agent navigator with explicit parameters.
        
        Args:
            positions: Initial positions as list or array with shape (n_agents, 2)
            orientations: Initial orientations for each agent (optional)
            speeds: Initial speeds for each agent (optional)
            max_speeds: Maximum speeds for each agent (optional)
            angular_velocities: Initial angular velocities for each agent (optional)
            
        Returns:
            NavigatorProtocol: Multi-agent navigator implementation
            
        Examples:
            Two-agent navigator:
            >>> positions = [[0.0, 0.0], [10.0, 10.0]]
            >>> navigator = NavigatorFactory.multi_agent(positions)
            
            Fully configured multi-agent:
            >>> navigator = NavigatorFactory.multi_agent(
            ...     positions=[[0, 0], [20, 0], [40, 0]],
            ...     orientations=[0, 45, 90],
            ...     max_speeds=[1.0, 1.5, 2.0]
            ... )
        """
        from ..core.controllers import MultiAgentController
        
        return MultiAgentController(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds,
            angular_velocities=angular_velocities
        )


# Utility functions for configuration processing

def _is_multi_agent_config(config: NavigatorConfig) -> bool:
    """
    Determine if configuration specifies multi-agent navigation.
    
    Args:
        config: NavigatorConfig instance to analyze
        
    Returns:
        bool: True if multi-agent configuration, False for single-agent
        
    Notes:
        Multi-agent is indicated by:
        - positions (list/array) is not None
        - num_agents > 1
        - Any of orientations, speeds, max_speeds, angular_velocities is a list
    """
    if config.positions is not None:
        return True
    if config.num_agents is not None and config.num_agents > 1:
        return True
    
    # Check if any parameters are lists (indicating multi-agent)
    list_params = [
        config.orientations, config.speeds, 
        config.max_speeds, config.angular_velocities
    ]
    return any(isinstance(param, list) for param in list_params if param is not None)


# Type aliases for enhanced IDE support and documentation

PositionType = Union[Tuple[float, float], List[float], np.ndarray]
"""Type alias for agent position - supports tuple, list, or numpy array."""

PositionsType = Union[List[List[float]], List[Tuple[float, float]], np.ndarray]
"""Type alias for multi-agent positions - supports nested lists or numpy array."""

OrientationType = Union[float, int]
"""Type alias for agent orientation in degrees."""

OrientationsType = Union[List[float], np.ndarray]
"""Type alias for multi-agent orientations."""

SpeedType = Union[float, int]
"""Type alias for agent speed value."""

SpeedsType = Union[List[float], np.ndarray]
"""Type alias for multi-agent speeds."""

ConfigType = Union[DictConfig, NavigatorConfig, dict]
"""Type alias for configuration objects - supports Hydra, Pydantic, or dict."""


# Re-export protocol and factory for public API
__all__ = [
    "NavigatorProtocol",
    "NavigatorFactory", 
    "PositionType",
    "PositionsType",
    "OrientationType", 
    "OrientationsType",
    "SpeedType",
    "SpeedsType",
    "ConfigType",
]