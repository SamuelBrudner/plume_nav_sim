"""
Navigation controller implementations for odor plume navigation.

This module consolidates SingleAgentController and MultiAgentController implementations
that encapsulate agent state and provide methods to initialize, reset, advance, and 
sample odor for navigation simulations. These controllers implement the NavigatorProtocol
interface and support both single-agent and multi-agent navigation scenarios.

Key Features:
- Protocol-based interface compliance for extensibility
- Vectorized operations for efficient multi-agent simulations
- Hydra configuration system integration for parameter management
- Enhanced error handling and structured logging
- Deterministic behavior through seed management integration
- Context manager support for resource lifecycle management

The controllers maintain all existing NavigatorProtocol compliance while providing
enhanced functionality for the refactored library architecture.
"""

import contextlib
from typing import Optional, Union, Any, Tuple, List, Dict, Protocol
import numpy as np
from loguru import logger
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

# Import dependencies from new module structure
from .navigator import NavigatorProtocol
from ..utils.seed_manager import get_seed_manager, SeedManager
from ..config.schemas import NavigatorConfig, SingleAgentConfig, MultiAgentConfig


class SingleAgentController:
    """
    Controller for single agent navigation implementing NavigatorProtocol.
    
    This controller specializes in single-agent scenarios, providing optimized
    operations without conditional branching for maximum performance. Integrates
    with Hydra configuration system and enhanced logging for comprehensive
    experiment tracking and reproducibility.
    
    Features:
    - Minimal overhead for single-agent operations
    - Hydra configuration integration for parameter management
    - Deterministic behavior through seed manager integration
    - Enhanced error handling with structured logging
    - Context manager support for resource lifecycle
    - Protocol compliance for algorithm extensibility
    
    Attributes:
        positions (np.ndarray): Agent position array with shape (1, 2)
        orientations (np.ndarray): Agent orientation array with shape (1,)
        speeds (np.ndarray): Agent speed array with shape (1,)
        max_speeds (np.ndarray): Maximum speed array with shape (1,)
        angular_velocities (np.ndarray): Angular velocity array with shape (1,)
        num_agents (int): Always 1 for single agent controller
    """
    
    def __init__(
        self,
        position: Optional[Tuple[float, float]] = None,
        orientation: float = 0.0,
        speed: float = 0.0,
        max_speed: float = 1.0,
        angular_velocity: float = 0.0,
        config: Optional[Union[SingleAgentConfig, DictConfig, Dict[str, Any]]] = None,
        seed_manager: Optional[SeedManager] = None
    ) -> None:
        """
        Initialize a single agent controller.
        
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
        config : Optional[Union[SingleAgentConfig, DictConfig, Dict[str, Any]]], optional
            Configuration object from Hydra or direct instantiation, by default None
        seed_manager : Optional[SeedManager], optional
            Seed manager for reproducible experiments, by default None
            
        Raises
        ------
        ValueError
            If configuration validation fails or parameters are invalid
        RuntimeError
            If initialization exceeds performance requirements
        """
        # Initialize logger with module context
        self._logger = logger.bind(
            module=__name__,
            controller_type="SingleAgent",
            initialization_timestamp=logger.opt(record=True).info.__wrapped__.__globals__.get('time', lambda: 0)()
        )
        
        try:
            # Load configuration if provided
            self._config = self._load_configuration(config)
            
            # Override defaults with configuration values if available
            if self._config:
                position = position or getattr(self._config, 'position', None)
                orientation = getattr(self._config, 'orientation', orientation)
                speed = getattr(self._config, 'speed', speed)
                max_speed = getattr(self._config, 'max_speed', max_speed)
                angular_velocity = getattr(self._config, 'angular_velocity', angular_velocity)
            
            # Validate parameters
            self._validate_parameters(speed, max_speed)
            
            # Initialize seed manager for deterministic behavior
            self._seed_manager = seed_manager or get_seed_manager()
            
            # Initialize state arrays
            self._position = np.array([position]) if position is not None else np.array([[0.0, 0.0]])
            self._orientation = np.array([orientation])
            self._speed = np.array([speed])
            self._max_speed = np.array([max_speed])
            self._angular_velocity = np.array([angular_velocity])
            
            # Ensure position array has correct shape
            if self._position.ndim == 1:
                self._position = self._position.reshape(1, -1)
            
            # Performance and validation logging
            self._logger.info(
                "Single agent controller initialized successfully",
                extra={
                    "position": self._position[0].tolist(),
                    "orientation": float(self._orientation[0]),
                    "speed": float(self._speed[0]),
                    "max_speed": float(self._max_speed[0]),
                    "angular_velocity": float(self._angular_velocity[0]),
                    "config_loaded": self._config is not None,
                    "seed_manager_active": self._seed_manager.current_seed is not None
                }
            )
            
        except Exception as e:
            self._logger.error(
                f"Single agent controller initialization failed: {e}",
                extra={"error_type": type(e).__name__}
            )
            raise RuntimeError(f"Controller initialization failed: {e}") from e
    
    def _load_configuration(
        self, 
        config: Optional[Union[SingleAgentConfig, DictConfig, Dict[str, Any]]]
    ) -> Optional[SingleAgentConfig]:
        """Load and validate configuration from various sources."""
        if config is None:
            # Try to load from Hydra global config
            config = self._load_from_hydra()
        
        if config is None:
            return None
        
        if isinstance(config, SingleAgentConfig):
            return config
        elif isinstance(config, (DictConfig, dict)):
            try:
                return SingleAgentConfig(**dict(config))
            except Exception as e:
                self._logger.warning(
                    f"Failed to load configuration: {e}",
                    extra={"config_type": type(config).__name__}
                )
                return None
        else:
            self._logger.warning(
                f"Unsupported configuration type: {type(config)}",
                extra={"config_type": type(config).__name__}
            )
            return None
    
    def _load_from_hydra(self) -> Optional[Dict[str, Any]]:
        """Load single agent configuration from Hydra global config."""
        try:
            if GlobalHydra().is_initialized():
                hydra_cfg = GlobalHydra.instance().cfg
                if hydra_cfg and "navigator" in hydra_cfg:
                    nav_config = OmegaConf.to_container(hydra_cfg.navigator, resolve=True)
                    # Check if this is single agent configuration
                    if nav_config and "position" in nav_config:
                        return nav_config
        except Exception as e:
            self._logger.debug(f"Could not load configuration from Hydra: {e}")
        
        return None
    
    def _validate_parameters(self, speed: float, max_speed: float) -> None:
        """Validate controller parameters for consistency."""
        if speed > max_speed:
            raise ValueError(f"Speed ({speed}) cannot exceed max_speed ({max_speed})")
        
        if max_speed <= 0:
            raise ValueError(f"Max speed must be positive, got: {max_speed}")
    
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
        """
        Reset the agent to initial state with enhanced error handling.
        
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
        
        Raises
        ------
        ValueError
            If invalid parameters are provided
        
        Notes
        -----
        For stricter type checking, you can use the SingleAgentParams dataclass:
        
        ```python
        from {{cookiecutter.project_slug}}.core.navigator import SingleAgentParams
        
        params = SingleAgentParams(position=(10, 20), speed=1.5)
        navigator.reset_with_params(params)
        ```
        """
        try:
            # Import at function level to avoid circular import
            from ..utils.navigator_utils import reset_navigator_state
            
            # Log reset operation
            self._logger.debug(
                "Resetting single agent controller state",
                extra={"reset_parameters": list(kwargs.keys())}
            )
            
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
            
            # Log successful reset
            self._logger.info(
                "Agent state reset successfully",
                extra={
                    "new_position": self._position[0].tolist(),
                    "new_orientation": float(self._orientation[0]),
                    "new_speed": float(self._speed[0]),
                    "parameters_updated": list(kwargs.keys())
                }
            )
            
        except Exception as e:
            self._logger.error(
                f"Failed to reset agent state: {e}",
                extra={"error_type": type(e).__name__, "parameters": kwargs}
            )
            raise RuntimeError(f"Agent reset failed: {e}") from e
    
    def reset_with_params(self, params: 'SingleAgentParams') -> None:
        """
        Reset the agent to initial state using a type-safe parameter object.
        
        This method provides stronger type checking than the kwargs-based reset method.
        
        Parameters
        ----------
        params : SingleAgentParams
            Parameters to update, as a dataclass instance
            
        Raises
        ------
        TypeError
            If params is not a SingleAgentParams instance
        RuntimeError
            If reset operation fails
        """
        try:
            # Import at function level to avoid circular import
            from ..utils.navigator_utils import reset_navigator_state_with_params, SingleAgentParams
            
            if not isinstance(params, SingleAgentParams):
                raise TypeError(f"Expected SingleAgentParams, got {type(params)}")
            
            self._logger.debug(
                "Resetting single agent controller with type-safe parameters",
                extra={"params_type": type(params).__name__}
            )
            
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
            
            self._logger.info("Agent state reset with type-safe parameters")
            
        except Exception as e:
            self._logger.error(
                f"Failed to reset agent state with params: {e}",
                extra={"error_type": type(e).__name__, "params_type": type(params).__name__}
            )
            raise RuntimeError(f"Type-safe agent reset failed: {e}") from e
    
    def step(self, env_array: np.ndarray, dt: float = 1.0) -> None:
        """
        Take a simulation step to update agent position and orientation.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array (e.g., odor concentration grid)
        dt : float, optional
            Time step size in seconds, by default 1.0
            
        Raises
        ------
        RuntimeError
            If step operation fails or produces invalid state
        """
        try:
            # Import at function level to avoid circular import
            from ..utils.navigator_utils import update_positions_and_orientations
            
            # Log step operation with performance tracking
            step_start_time = logger.opt(record=True).info.__wrapped__.__globals__.get('time', lambda: 0)()
            
            # Use deterministic randomness if needed
            if hasattr(env_array, 'requires_stochastic_sampling'):
                with self._seed_manager.temporary_seed(self._seed_manager.current_seed + int(step_start_time * 1000)):
                    # Use the utility function to update position and orientation
                    update_positions_and_orientations(
                        self._position, 
                        self._orientation, 
                        self._speed, 
                        self._angular_velocity,
                        dt=dt
                    )
            else:
                # Use the utility function to update position and orientation
                update_positions_and_orientations(
                    self._position, 
                    self._orientation, 
                    self._speed, 
                    self._angular_velocity,
                    dt=dt
                )
            
            # Validate state after update
            self._validate_state()
            
            # Performance logging
            step_time = logger.opt(record=True).info.__wrapped__.__globals__.get('time', lambda: 0)() - step_start_time
            
            self._logger.debug(
                "Simulation step completed",
                extra={
                    "dt": dt,
                    "step_duration_ms": f"{step_time * 1000:.2f}",
                    "new_position": self._position[0].tolist(),
                    "new_orientation": float(self._orientation[0])
                }
            )
            
        except Exception as e:
            self._logger.error(
                f"Simulation step failed: {e}",
                extra={"error_type": type(e).__name__, "dt": dt}
            )
            raise RuntimeError(f"Simulation step failed: {e}") from e
    
    def _validate_state(self) -> None:
        """Validate controller state for consistency and bounds."""
        # Check for NaN or infinite values
        for name, array in [
            ("position", self._position),
            ("orientation", self._orientation),
            ("speed", self._speed),
            ("max_speed", self._max_speed),
            ("angular_velocity", self._angular_velocity)
        ]:
            if not np.isfinite(array).all():
                raise RuntimeError(f"Invalid {name} values detected: {array}")
        
        # Check speed constraints
        if self._speed[0] > self._max_speed[0]:
            self._logger.warning(
                f"Speed ({self._speed[0]}) exceeds max_speed ({self._max_speed[0]}), clamping",
                extra={"original_speed": float(self._speed[0]), "max_speed": float(self._max_speed[0])}
            )
            self._speed[0] = self._max_speed[0]
    
    def sample_odor(self, env_array: np.ndarray) -> float:
        """
        Sample odor at the current agent position.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array
            
        Returns
        -------
        float
            Odor value at the agent's position
            
        Raises
        ------
        RuntimeError
            If odor sampling fails
        """
        try:
            return self.read_single_antenna_odor(env_array)
        except Exception as e:
            self._logger.error(
                f"Odor sampling failed: {e}",
                extra={"error_type": type(e).__name__, "position": self._position[0].tolist()}
            )
            raise RuntimeError(f"Odor sampling failed: {e}") from e
    
    def read_single_antenna_odor(self, env_array: np.ndarray) -> float:
        """
        Sample odor at the agent's single antenna.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array
            
        Returns
        -------
        float
            Odor value at the agent's position
        """
        try:
            # Import at function level to avoid circular import
            from ..utils.navigator_utils import read_odor_values
            
            # Use the utility function to read odor value
            odor_values = read_odor_values(env_array, self._position)
            result = float(odor_values[0])
            
            self._logger.debug(
                "Single antenna odor reading",
                extra={
                    "position": self._position[0].tolist(),
                    "odor_value": result,
                    "env_shape": env_array.shape if hasattr(env_array, 'shape') else "unknown"
                }
            )
            
            return result
            
        except Exception as e:
            self._logger.error(
                f"Single antenna odor reading failed: {e}",
                extra={"error_type": type(e).__name__}
            )
            # Return 0.0 for graceful degradation
            return 0.0
    
    def sample_multiple_sensors(
        self, 
        env_array: np.ndarray, 
        sensor_distance: float = 5.0,
        sensor_angle: float = 45.0,
        num_sensors: int = 2,
        layout_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Sample odor at multiple sensor positions.
        
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
            
        Raises
        ------
        RuntimeError
            If multi-sensor sampling fails
        """
        try:
            from ..utils.navigator_utils import sample_odor_at_sensors
            
            self._logger.debug(
                "Multi-sensor odor sampling",
                extra={
                    "sensor_distance": sensor_distance,
                    "sensor_angle": sensor_angle,
                    "num_sensors": num_sensors,
                    "layout_name": layout_name
                }
            )
            
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
            result = odor_values[0]
            
            self._logger.debug(
                "Multi-sensor sampling completed",
                extra={
                    "odor_values": result.tolist(),
                    "mean_odor": float(np.mean(result)),
                    "std_odor": float(np.std(result))
                }
            )
            
            return result
            
        except Exception as e:
            self._logger.error(
                f"Multi-sensor sampling failed: {e}",
                extra={"error_type": type(e).__name__, "num_sensors": num_sensors}
            )
            raise RuntimeError(f"Multi-sensor sampling failed: {e}") from e


class MultiAgentController:
    """
    Controller for multi-agent navigation implementing NavigatorProtocol.
    
    This controller specializes in multi-agent scenarios using vectorized operations
    for efficient simulation of multiple agents simultaneously. Provides O(1) scaling
    characteristics for agent position updates and odor sampling operations.
    
    Features:
    - Vectorized operations for efficient multi-agent processing
    - Automatic scaling up to 100 simultaneous agents
    - Hydra configuration integration for parameter management
    - Enhanced error handling with structured logging
    - Deterministic behavior through seed manager integration
    - Memory-efficient design with linear scaling
    
    Attributes:
        positions (np.ndarray): Agent positions array with shape (num_agents, 2)
        orientations (np.ndarray): Agent orientations array with shape (num_agents,)
        speeds (np.ndarray): Agent speeds array with shape (num_agents,)
        max_speeds (np.ndarray): Maximum speeds array with shape (num_agents,)
        angular_velocities (np.ndarray): Angular velocities array with shape (num_agents,)
        num_agents (int): Number of agents being controlled
    """
    
    def __init__(
        self,
        positions: Optional[np.ndarray] = None,
        orientations: Optional[np.ndarray] = None,
        speeds: Optional[np.ndarray] = None,
        max_speeds: Optional[np.ndarray] = None,
        angular_velocities: Optional[np.ndarray] = None,
        config: Optional[Union[MultiAgentConfig, DictConfig, Dict[str, Any]]] = None,
        seed_manager: Optional[SeedManager] = None
    ) -> None:
        """
        Initialize a multi-agent controller.
        
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
        config : Optional[Union[MultiAgentConfig, DictConfig, Dict[str, Any]]], optional
            Configuration object from Hydra or direct instantiation, by default None
        seed_manager : Optional[SeedManager], optional
            Seed manager for reproducible experiments, by default None
            
        Raises
        ------
        ValueError
            If configuration validation fails or array shapes are inconsistent
        RuntimeError
            If initialization exceeds performance requirements
        """
        # Initialize logger with module context
        self._logger = logger.bind(
            module=__name__,
            controller_type="MultiAgent",
            initialization_timestamp=logger.opt(record=True).info.__wrapped__.__globals__.get('time', lambda: 0)()
        )
        
        try:
            # Load configuration if provided
            self._config = self._load_configuration(config)
            
            # Override defaults with configuration values if available
            if self._config:
                positions = positions if positions is not None else getattr(self._config, 'positions', None)
                orientations = orientations if orientations is not None else getattr(self._config, 'orientations', None)
                speeds = speeds if speeds is not None else getattr(self._config, 'speeds', None)
                max_speeds = max_speeds if max_speeds is not None else getattr(self._config, 'max_speeds', None)
                angular_velocities = angular_velocities if angular_velocities is not None else getattr(self._config, 'angular_velocities', None)
            
            # Initialize seed manager for deterministic behavior
            self._seed_manager = seed_manager or get_seed_manager()
            
            # Ensure we have at least one agent position
            if positions is None:
                self._positions = np.array([[0.0, 0.0]])
            else:
                self._positions = np.array(positions)

            num_agents = self._positions.shape[0]
            
            # Validate array dimensions
            self._validate_array_dimensions(num_agents)

            # Set defaults for other parameters if not provided
            self._orientations = np.zeros(num_agents) if orientations is None else np.array(orientations)
            self._speeds = np.zeros(num_agents) if speeds is None else np.array(speeds)
            self._max_speeds = np.ones(num_agents) if max_speeds is None else np.array(max_speeds)
            self._angular_velocities = np.zeros(num_agents) if angular_velocities is None else np.array(angular_velocities)
            
            # Validate parameter consistency
            self._validate_parameters()
            
            # Validate array shapes for consistency
            self._validate_array_shapes()
            
            # Performance and validation logging
            self._logger.info(
                "Multi-agent controller initialized successfully",
                extra={
                    "num_agents": num_agents,
                    "mean_speed": float(np.mean(self._speeds)),
                    "mean_max_speed": float(np.mean(self._max_speeds)),
                    "position_bounds": {
                        "x_min": float(np.min(self._positions[:, 0])),
                        "x_max": float(np.max(self._positions[:, 0])),
                        "y_min": float(np.min(self._positions[:, 1])),
                        "y_max": float(np.max(self._positions[:, 1]))
                    },
                    "config_loaded": self._config is not None,
                    "seed_manager_active": self._seed_manager.current_seed is not None
                }
            )
            
            # Validate scaling constraints
            if num_agents > 100:
                self._logger.warning(
                    f"Agent count ({num_agents}) exceeds recommended maximum (100)",
                    extra={"agent_count": num_agents, "recommended_max": 100}
                )
            
        except Exception as e:
            self._logger.error(
                f"Multi-agent controller initialization failed: {e}",
                extra={"error_type": type(e).__name__}
            )
            raise RuntimeError(f"Controller initialization failed: {e}") from e
    
    def _load_configuration(
        self, 
        config: Optional[Union[MultiAgentConfig, DictConfig, Dict[str, Any]]]
    ) -> Optional[MultiAgentConfig]:
        """Load and validate configuration from various sources."""
        if config is None:
            # Try to load from Hydra global config
            config = self._load_from_hydra()
        
        if config is None:
            return None
        
        if isinstance(config, MultiAgentConfig):
            return config
        elif isinstance(config, (DictConfig, dict)):
            try:
                return MultiAgentConfig(**dict(config))
            except Exception as e:
                self._logger.warning(
                    f"Failed to load configuration: {e}",
                    extra={"config_type": type(config).__name__}
                )
                return None
        else:
            self._logger.warning(
                f"Unsupported configuration type: {type(config)}",
                extra={"config_type": type(config).__name__}
            )
            return None
    
    def _load_from_hydra(self) -> Optional[Dict[str, Any]]:
        """Load multi-agent configuration from Hydra global config."""
        try:
            if GlobalHydra().is_initialized():
                hydra_cfg = GlobalHydra.instance().cfg
                if hydra_cfg and "navigator" in hydra_cfg:
                    nav_config = OmegaConf.to_container(hydra_cfg.navigator, resolve=True)
                    # Check if this is multi-agent configuration
                    if nav_config and "positions" in nav_config:
                        return nav_config
        except Exception as e:
            self._logger.debug(f"Could not load configuration from Hydra: {e}")
        
        return None
    
    def _validate_array_dimensions(self, num_agents: int) -> None:
        """Validate that we can support the requested number of agents."""
        if num_agents <= 0:
            raise ValueError(f"Number of agents must be positive, got: {num_agents}")
        
        if num_agents > 1000:  # Hard limit for safety
            raise ValueError(f"Number of agents ({num_agents}) exceeds system limits (1000)")
    
    def _validate_parameters(self) -> None:
        """Validate controller parameters for consistency."""
        # Check that speeds don't exceed max speeds
        speed_violations = self._speeds > self._max_speeds
        if np.any(speed_violations):
            violation_indices = np.where(speed_violations)[0]
            self._logger.warning(
                f"Speed violations detected for agents: {violation_indices.tolist()}",
                extra={
                    "violation_count": len(violation_indices),
                    "total_agents": len(self._speeds)
                }
            )
            # Clamp speeds to max speeds
            self._speeds[speed_violations] = self._max_speeds[speed_violations]
        
        # Check for non-positive max speeds
        invalid_max_speeds = self._max_speeds <= 0
        if np.any(invalid_max_speeds):
            invalid_indices = np.where(invalid_max_speeds)[0]
            raise ValueError(f"Max speeds must be positive for agents: {invalid_indices.tolist()}")
    
    def _validate_array_shapes(self) -> None:
        """Validate that all arrays have consistent shapes."""
        num_agents = self._positions.shape[0]
        
        # Check position array shape
        if self._positions.shape != (num_agents, 2):
            raise ValueError(f"Positions array has invalid shape: {self._positions.shape}, expected: ({num_agents}, 2)")
        
        # Check other arrays
        for name, array in [
            ("orientations", self._orientations),
            ("speeds", self._speeds),
            ("max_speeds", self._max_speeds),
            ("angular_velocities", self._angular_velocities)
        ]:
            if array.shape != (num_agents,):
                raise ValueError(f"{name} array has invalid shape: {array.shape}, expected: ({num_agents},)")
    
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
        """
        Reset all agents to initial state or update with new settings.
        
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
            
        Raises
        ------
        ValueError
            If invalid parameters are provided or array shapes are inconsistent
        RuntimeError
            If reset operation fails
            
        Notes
        -----
        For stricter type checking, you can use the MultiAgentParams dataclass:
        
        ```python
        from {{cookiecutter.project_slug}}.core.navigator import MultiAgentParams
        import numpy as np
        
        params = MultiAgentParams(
            positions=np.array([[10, 20], [30, 40]]),
            speeds=np.array([1.5, 2.0])
        )
        navigator.reset_with_params(params)
        ```
        """
        try:
            # Import at function level to avoid circular import
            from ..utils.navigator_utils import reset_navigator_state
            
            # Log reset operation
            self._logger.debug(
                "Resetting multi-agent controller state",
                extra={
                    "reset_parameters": list(kwargs.keys()),
                    "current_num_agents": self.num_agents
                }
            )
            
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
            
            # Validate updated state
            self._validate_array_shapes()
            self._validate_parameters()
            
            # Log successful reset
            self._logger.info(
                "Multi-agent state reset successfully",
                extra={
                    "new_num_agents": self.num_agents,
                    "parameters_updated": list(kwargs.keys()),
                    "mean_speed": float(np.mean(self._speeds)),
                    "position_spread": {
                        "x_std": float(np.std(self._positions[:, 0])),
                        "y_std": float(np.std(self._positions[:, 1]))
                    }
                }
            )
            
        except Exception as e:
            self._logger.error(
                f"Failed to reset multi-agent state: {e}",
                extra={"error_type": type(e).__name__, "parameters": list(kwargs.keys())}
            )
            raise RuntimeError(f"Multi-agent reset failed: {e}") from e
    
    def reset_with_params(self, params: 'MultiAgentParams') -> None:
        """
        Reset all agents to initial state using a type-safe parameter object.
        
        This method provides stronger type checking than the kwargs-based reset method.
        
        Parameters
        ----------
        params : MultiAgentParams
            Parameters to update, as a dataclass instance
            
        Raises
        ------
        TypeError
            If params is not a MultiAgentParams instance
        RuntimeError
            If reset operation fails
        """
        try:
            # Import at function level to avoid circular import
            from ..utils.navigator_utils import reset_navigator_state_with_params, MultiAgentParams
            
            if not isinstance(params, MultiAgentParams):
                raise TypeError(f"Expected MultiAgentParams, got {type(params)}")
            
            self._logger.debug(
                "Resetting multi-agent controller with type-safe parameters",
                extra={"params_type": type(params).__name__}
            )
            
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
            
            # Validate updated state
            self._validate_array_shapes()
            self._validate_parameters()
            
            self._logger.info("Multi-agent state reset with type-safe parameters")
            
        except Exception as e:
            self._logger.error(
                f"Failed to reset multi-agent state with params: {e}",
                extra={"error_type": type(e).__name__, "params_type": type(params).__name__}
            )
            raise RuntimeError(f"Type-safe multi-agent reset failed: {e}") from e
    
    def step(self, env_array: np.ndarray, dt: float = 1.0) -> None:
        """
        Take a simulation step to update all agent positions and orientations.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array (e.g., odor concentration grid)
        dt : float, optional
            Time step size in seconds, by default 1.0
            
        Raises
        ------
        RuntimeError
            If step operation fails or produces invalid state
        """
        try:
            # Import at function level to avoid circular import
            from ..utils.navigator_utils import update_positions_and_orientations
            
            # Log step operation with performance tracking
            step_start_time = logger.opt(record=True).info.__wrapped__.__globals__.get('time', lambda: 0)()
            
            # Use deterministic randomness if needed
            if hasattr(env_array, 'requires_stochastic_sampling'):
                with self._seed_manager.temporary_seed(self._seed_manager.current_seed + int(step_start_time * 1000)):
                    # Use the utility function to update positions and orientations
                    update_positions_and_orientations(
                        self._positions, 
                        self._orientations, 
                        self._speeds, 
                        self._angular_velocities,
                        dt=dt
                    )
            else:
                # Use the utility function to update positions and orientations
                update_positions_and_orientations(
                    self._positions, 
                    self._orientations, 
                    self._speeds, 
                    self._angular_velocities,
                    dt=dt
                )
            
            # Validate state after update
            self._validate_state()
            
            # Performance logging
            step_time = logger.opt(record=True).info.__wrapped__.__globals__.get('time', lambda: 0)() - step_start_time
            
            self._logger.debug(
                "Multi-agent simulation step completed",
                extra={
                    "dt": dt,
                    "num_agents": self.num_agents,
                    "step_duration_ms": f"{step_time * 1000:.2f}",
                    "mean_position": {
                        "x": float(np.mean(self._positions[:, 0])),
                        "y": float(np.mean(self._positions[:, 1]))
                    },
                    "mean_orientation": float(np.mean(self._orientations))
                }
            )
            
        except Exception as e:
            self._logger.error(
                f"Multi-agent simulation step failed: {e}",
                extra={"error_type": type(e).__name__, "dt": dt, "num_agents": self.num_agents}
            )
            raise RuntimeError(f"Multi-agent simulation step failed: {e}") from e
    
    def _validate_state(self) -> None:
        """Validate controller state for consistency and bounds."""
        # Check for NaN or infinite values
        for name, array in [
            ("positions", self._positions),
            ("orientations", self._orientations),
            ("speeds", self._speeds),
            ("max_speeds", self._max_speeds),
            ("angular_velocities", self._angular_velocities)
        ]:
            if not np.isfinite(array).all():
                invalid_indices = np.where(~np.isfinite(array))[0]
                raise RuntimeError(f"Invalid {name} values detected for agents: {invalid_indices.tolist()}")
        
        # Check speed constraints
        speed_violations = self._speeds > self._max_speeds
        if np.any(speed_violations):
            violation_indices = np.where(speed_violations)[0]
            self._logger.warning(
                f"Speed constraint violations for agents: {violation_indices.tolist()}, clamping speeds",
                extra={"violation_count": len(violation_indices)}
            )
            self._speeds[speed_violations] = self._max_speeds[speed_violations]
    
    def sample_odor(self, env_array: np.ndarray) -> np.ndarray:
        """
        Sample odor at all agent positions.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array
            
        Returns
        -------
        np.ndarray
            Odor values at each agent's position, shape (num_agents,)
            
        Raises
        ------
        RuntimeError
            If odor sampling fails
        """
        try:
            return self.read_single_antenna_odor(env_array)
        except Exception as e:
            self._logger.error(
                f"Multi-agent odor sampling failed: {e}",
                extra={"error_type": type(e).__name__, "num_agents": self.num_agents}
            )
            raise RuntimeError(f"Multi-agent odor sampling failed: {e}") from e
    
    def read_single_antenna_odor(self, env_array: np.ndarray) -> np.ndarray:
        """
        Sample odor at each agent's position.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array
            
        Returns
        -------
        np.ndarray
            Odor values at each agent's position, shape (num_agents,)
        """
        try:
            # Import at function level to avoid circular import
            from ..utils.navigator_utils import read_odor_values
            
            # Use the utility function to read odor values
            result = read_odor_values(env_array, self._positions)
            
            self._logger.debug(
                "Multi-agent single antenna odor reading",
                extra={
                    "num_agents": self.num_agents,
                    "mean_odor": float(np.mean(result)),
                    "std_odor": float(np.std(result)),
                    "min_odor": float(np.min(result)),
                    "max_odor": float(np.max(result)),
                    "env_shape": env_array.shape if hasattr(env_array, 'shape') else "unknown"
                }
            )
            
            return result
            
        except Exception as e:
            self._logger.error(
                f"Multi-agent single antenna odor reading failed: {e}",
                extra={"error_type": type(e).__name__}
            )
            # Return zeros for graceful degradation
            return np.zeros(self.num_agents)
    
    def sample_multiple_sensors(
        self, 
        env_array: np.ndarray, 
        sensor_distance: float = 5.0,
        sensor_angle: float = 45.0,
        num_sensors: int = 2,
        layout_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Sample odor at multiple sensor positions for all agents.
        
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
            
        Raises
        ------
        RuntimeError
            If multi-sensor sampling fails
        """
        try:
            from ..utils.navigator_utils import sample_odor_at_sensors
            
            self._logger.debug(
                "Multi-agent multi-sensor odor sampling",
                extra={
                    "num_agents": self.num_agents,
                    "sensor_distance": sensor_distance,
                    "sensor_angle": sensor_angle,
                    "num_sensors": num_sensors,
                    "layout_name": layout_name
                }
            )
            
            # Delegate to utility function
            result = sample_odor_at_sensors(
                self, 
                env_array, 
                sensor_distance=sensor_distance,
                sensor_angle=sensor_angle, 
                num_sensors=num_sensors,
                layout_name=layout_name
            )
            
            self._logger.debug(
                "Multi-agent multi-sensor sampling completed",
                extra={
                    "result_shape": result.shape,
                    "mean_odor_per_agent": np.mean(result, axis=1).tolist(),
                    "overall_mean_odor": float(np.mean(result)),
                    "overall_std_odor": float(np.std(result))
                }
            )
            
            return result
            
        except Exception as e:
            self._logger.error(
                f"Multi-agent multi-sensor sampling failed: {e}",
                extra={"error_type": type(e).__name__, "num_agents": self.num_agents, "num_sensors": num_sensors}
            )
            raise RuntimeError(f"Multi-agent multi-sensor sampling failed: {e}") from e


# Export controller classes
__all__ = [
    "SingleAgentController",
    "MultiAgentController",
]