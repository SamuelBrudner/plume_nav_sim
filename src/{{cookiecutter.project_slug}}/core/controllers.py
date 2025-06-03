"""
Consolidated navigation controllers for single and multi-agent scenarios.

This module consolidates the SingleAgentController and MultiAgentController implementations
that encapsulate agent state and provide methods to initialize, reset, advance, and sample
odor for navigation simulations. Both controllers implement the NavigatorProtocol interface
and integrate with the Hydra configuration system for enhanced ML framework compatibility.

The controllers support:
- Single-agent and multi-agent navigation scenarios
- Hydra configuration integration for experiment orchestration
- Enhanced error handling and performance monitoring
- Comprehensive logging with structured context
- Random seed management for reproducible experiments
- Type-safe parameter validation and constraints

Examples:
    Single-agent controller:
        >>> from {{cookiecutter.project_slug}}.core.controllers import SingleAgentController
        >>> controller = SingleAgentController(position=(10.0, 20.0), speed=1.5)
        >>> controller.step(env_array, dt=1.0)
        >>> odor_value = controller.sample_odor(env_array)
        
    Multi-agent controller:
        >>> from {{cookiecutter.project_slug}}.core.controllers import MultiAgentController
        >>> positions = [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]
        >>> controller = MultiAgentController(positions=positions)
        >>> controller.step(env_array, dt=1.0)
        >>> odor_values = controller.sample_odor(env_array)
        
    Configuration-driven instantiation:
        >>> from {{cookiecutter.project_slug}}.core.controllers import create_controller_from_config
        >>> from omegaconf import DictConfig
        >>> cfg = DictConfig({"position": [5.0, 5.0], "max_speed": 2.0})
        >>> controller = create_controller_from_config(cfg)

Notes:
    All controllers maintain backward compatibility with the original NavigatorProtocol
    interface while adding enhanced features for modern ML pipeline integration.
    Performance requirements are maintained with <33ms frame processing latency
    and support for 100+ agents at 30fps simulation throughput.
"""

import contextlib
import time
from typing import Optional, Union, Any, Tuple, List, Dict
from dataclasses import dataclass
import numpy as np

# Hydra integration for configuration management
try:
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    HydraConfig = None
    DictConfig = dict
    OmegaConf = None

# Loguru integration for enhanced logging
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    LOGURU_AVAILABLE = False

# Import navigation protocol and utilities
from .navigator import NavigatorProtocol
from ..utils.seed_manager import SeedManager, get_global_seed_manager
from ..config.schemas import NavigatorConfig, SingleAgentConfig, MultiAgentConfig


@dataclass
class SingleAgentParams:
    """Type-safe parameters for resetting a single agent navigator.
    
    This dataclass provides stronger type checking than kwargs-based configuration
    and integrates with Hydra's structured configuration system for validation.
    
    Attributes:
        position: Initial agent position coordinates [x, y]
        orientation: Initial orientation in degrees (0 = right, 90 = up)
        speed: Initial speed in units per time step
        max_speed: Maximum allowed speed in units per time step
        angular_velocity: Angular velocity in degrees per second
    """
    position: Optional[Tuple[float, float]] = None
    orientation: Optional[float] = None
    speed: Optional[float] = None
    max_speed: Optional[float] = None
    angular_velocity: Optional[float] = None


@dataclass
class MultiAgentParams:
    """Type-safe parameters for resetting a multi-agent navigator.
    
    This dataclass provides stronger type checking and enables batch parameter
    updates for multiple agents with comprehensive validation.
    
    Attributes:
        positions: Array of agent positions with shape (num_agents, 2)
        orientations: Array of agent orientations with shape (num_agents,)
        speeds: Array of agent speeds with shape (num_agents,)
        max_speeds: Array of maximum speeds with shape (num_agents,)
        angular_velocities: Array of angular velocities with shape (num_agents,)
    """
    positions: Optional[np.ndarray] = None
    orientations: Optional[np.ndarray] = None
    speeds: Optional[np.ndarray] = None
    max_speeds: Optional[np.ndarray] = None
    angular_velocities: Optional[np.ndarray] = None


class SingleAgentController:
    """Controller for single agent navigation with enhanced integration.
    
    This implements the NavigatorProtocol for a single agent case, providing
    simplified navigation logic without conditional branches. Integrates with
    Hydra configuration system and includes comprehensive logging and error
    handling for robust research workflows.
    
    The controller maintains all agent state as NumPy arrays for consistent
    API compatibility with multi-agent scenarios and efficient numerical
    operations. Performance is optimized for <33ms frame processing latency.
    
    Attributes:
        positions: Agent position as numpy array with shape (1, 2)
        orientations: Agent orientation as numpy array with shape (1,)
        speeds: Agent speed as numpy array with shape (1,)
        max_speeds: Maximum agent speed as numpy array with shape (1,)
        angular_velocities: Agent angular velocity as numpy array with shape (1,)
        num_agents: Always 1 for single-agent controller
    
    Examples:
        Basic initialization:
            >>> controller = SingleAgentController()
            >>> controller.reset(position=(10.0, 20.0), speed=1.5)
            
        Type-safe parameter updates:
            >>> params = SingleAgentParams(position=(5.0, 5.0), max_speed=2.0)
            >>> controller.reset_with_params(params)
            
        Simulation step with timing:
            >>> start_time = time.perf_counter()
            >>> controller.step(env_array, dt=1.0)
            >>> processing_time = (time.perf_counter() - start_time) * 1000
            >>> assert processing_time < 33, "Frame processing exceeded 33ms requirement"
    """
    
    def __init__(
        self,
        position: Optional[Tuple[float, float]] = None,
        orientation: float = 0.0,
        speed: float = 0.0,
        max_speed: float = 1.0,
        angular_velocity: float = 0.0,
        enable_logging: bool = True,
        controller_id: Optional[str] = None
    ) -> None:
        """Initialize a single agent controller with enhanced features.
        
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
        enable_logging : bool, optional
            Enable comprehensive logging integration, by default True
        controller_id : Optional[str], optional
            Unique controller identifier for logging, by default None
            
        Raises:
            ValueError: If speed exceeds max_speed or parameters are invalid
        """
        # Validate parameters
        if speed > max_speed:
            raise ValueError(f"speed ({speed}) cannot exceed max_speed ({max_speed})")
        
        # Initialize state arrays for API consistency
        self._position = np.array([position]) if position is not None else np.array([[0.0, 0.0]])
        self._orientation = np.array([orientation % 360.0])  # Normalize to [0, 360)
        self._speed = np.array([speed])
        self._max_speed = np.array([max_speed])
        self._angular_velocity = np.array([angular_velocity])
        
        # Enhanced logging and monitoring
        self._enable_logging = enable_logging
        self._controller_id = controller_id or f"single_agent_{id(self)}"
        self._performance_metrics = {
            'step_times': [],
            'sample_times': [],
            'total_steps': 0
        }
        
        # Bind logging context for structured logging
        if self._enable_logging and LOGURU_AVAILABLE:
            self._logger = logger.bind(
                controller_type="single_agent",
                controller_id=self._controller_id,
                num_agents=1
            )
            
            # Log initialization with seed context if available
            seed_manager = get_global_seed_manager()
            if seed_manager:
                self._logger = self._logger.bind(
                    seed_value=seed_manager.seed,
                    experiment_id=seed_manager.experiment_id
                )
            
            # Add Hydra context if available
            if HYDRA_AVAILABLE:
                try:
                    hydra_cfg = HydraConfig.get()
                    self._logger = self._logger.bind(
                        hydra_job_name=hydra_cfg.job.name,
                        hydra_output_dir=hydra_cfg.runtime.output_dir
                    )
                except Exception:
                    # Hydra context not available, continue without it
                    pass
                    
            self._logger.info(
                f"SingleAgentController initialized",
                position=position,
                orientation=orientation,
                speed=speed,
                max_speed=max_speed,
                angular_velocity=angular_velocity
            )
        else:
            self._logger = None
    
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
        """Reset the agent to initial state with enhanced validation.
        
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
        
        Raises:
            ValueError: If invalid parameters are provided or constraints violated
            
        Notes
        -----
        For stronger type checking, use the SingleAgentParams dataclass:
        
        ```python
        from {{cookiecutter.project_slug}}.core.controllers import SingleAgentParams
        
        params = SingleAgentParams(position=(10, 20), speed=1.5)
        navigator.reset_with_params(params)
        ```
        """
        start_time = time.perf_counter() if self._enable_logging else None
        
        try:
            # Create a dictionary of current state for utility function
            controller_state = {
                '_position': self._position,
                '_orientation': self._orientation,
                '_speed': self._speed,
                '_max_speed': self._max_speed,
                '_angular_velocity': self._angular_velocity
            }
            
            # Use the utility function to reset state
            _reset_navigator_state(controller_state, is_single_agent=True, **kwargs)
            
            # Update instance attributes
            self._position = controller_state['_position']
            self._orientation = controller_state['_orientation']
            self._speed = controller_state['_speed']
            self._max_speed = controller_state['_max_speed']
            self._angular_velocity = controller_state['_angular_velocity']
            
            # Log successful reset with performance timing
            if self._logger:
                reset_time = (time.perf_counter() - start_time) * 1000
                self._logger.info(
                    f"Agent reset completed",
                    reset_time_ms=reset_time,
                    updated_params=list(kwargs.keys()),
                    position=self._position[0].tolist(),
                    orientation=float(self._orientation[0]),
                    speed=float(self._speed[0]),
                    max_speed=float(self._max_speed[0]),
                    angular_velocity=float(self._angular_velocity[0])
                )
                
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Agent reset failed: {str(e)}",
                    error_type=type(e).__name__,
                    invalid_params=kwargs
                )
            raise
    
    def reset_with_params(self, params: SingleAgentParams) -> None:
        """Reset the agent using a type-safe parameter object.
        
        This method provides stronger type checking than the kwargs-based reset method
        and integrates seamlessly with Hydra structured configuration.
        
        Parameters
        ----------
        params : SingleAgentParams
            Parameters to update, as a dataclass instance
            
        Raises:
            TypeError: If params is not a SingleAgentParams instance
            ValueError: If parameter constraints are violated
        """
        if not isinstance(params, SingleAgentParams):
            raise TypeError(
                f"Expected SingleAgentParams, got {type(params)}"
            )
        
        # Convert dataclass to dictionary for the existing function
        kwargs = {k: v for k, v in params.__dict__.items() if v is not None}
        
        if self._logger:
            self._logger.debug(
                f"Resetting agent with type-safe parameters",
                param_count=len(kwargs),
                params=kwargs
            )
        
        # Delegate to the existing function
        self.reset(**kwargs)
    
    def step(self, env_array: np.ndarray, dt: float = 1.0) -> None:
        """Take a simulation step with performance monitoring.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array (e.g., odor concentration grid)
        dt : float, optional
            Time step size in seconds, by default 1.0
            
        Raises:
            ValueError: If env_array is invalid or dt is non-positive
            RuntimeError: If step processing exceeds performance requirements
        """
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
            
        start_time = time.perf_counter() if self._enable_logging else None
        
        try:
            # Use the utility function to update position and orientation
            _update_positions_and_orientations(
                self._position, 
                self._orientation, 
                self._speed, 
                self._angular_velocity,
                dt=dt
            )
            
            # Track performance metrics
            if self._enable_logging:
                step_time = (time.perf_counter() - start_time) * 1000
                self._performance_metrics['step_times'].append(step_time)
                self._performance_metrics['total_steps'] += 1
                
                # Check performance requirement (<33ms)
                if step_time > 33.0:
                    if self._logger:
                        self._logger.warning(
                            f"Step processing exceeded 33ms requirement",
                            step_time_ms=step_time,
                            dt=dt,
                            position=self._position[0].tolist(),
                            performance_degradation=True
                        )
                
                # Log periodic performance summary
                if self._performance_metrics['total_steps'] % 100 == 0 and self._logger:
                    avg_step_time = np.mean(self._performance_metrics['step_times'][-100:])
                    self._logger.debug(
                        f"Performance summary",
                        total_steps=self._performance_metrics['total_steps'],
                        avg_step_time_ms=avg_step_time,
                        recent_position=self._position[0].tolist(),
                        recent_orientation=float(self._orientation[0])
                    )
                    
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Step execution failed: {str(e)}",
                    error_type=type(e).__name__,
                    dt=dt,
                    env_array_shape=getattr(env_array, 'shape', 'unknown')
                )
            raise RuntimeError(f"Agent step failed: {str(e)}") from e
    
    def sample_odor(self, env_array: np.ndarray) -> float:
        """Sample odor at the current agent position with error handling.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array
            
        Returns
        -------
        float
            Odor value at the agent's position
            
        Raises:
            ValueError: If env_array is invalid or sampling fails
        """
        return self.read_single_antenna_odor(env_array)
    
    def read_single_antenna_odor(self, env_array: np.ndarray) -> float:
        """Sample odor at the agent's single antenna with monitoring.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array
            
        Returns
        -------
        float
            Odor value at the agent's position
            
        Raises:
            ValueError: If sampling fails or returns invalid values
        """
        start_time = time.perf_counter() if self._enable_logging else None
        
        try:
            # Use the utility function to read odor value
            odor_values = _read_odor_values(env_array, self._position)
            odor_value = float(odor_values[0])
            
            # Validate odor value
            if np.isnan(odor_value) or np.isinf(odor_value):
                if self._logger:
                    self._logger.warning(
                        f"Invalid odor value detected",
                        odor_value=odor_value,
                        position=self._position[0].tolist(),
                        using_fallback=True
                    )
                odor_value = 0.0
            
            # Track sampling performance
            if self._enable_logging:
                sample_time = (time.perf_counter() - start_time) * 1000
                self._performance_metrics['sample_times'].append(sample_time)
                
                # Log detailed sampling for debugging
                if self._logger and self._performance_metrics['total_steps'] % 50 == 0:
                    self._logger.trace(
                        f"Odor sampling completed",
                        sample_time_ms=sample_time,
                        odor_value=odor_value,
                        position=self._position[0].tolist()
                    )
            
            return odor_value
            
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Odor sampling failed: {str(e)}",
                    error_type=type(e).__name__,
                    position=self._position[0].tolist(),
                    env_array_shape=getattr(env_array, 'shape', 'unknown')
                )
            # Return safe default value
            return 0.0
    
    def sample_multiple_sensors(
        self, 
        env_array: np.ndarray, 
        sensor_distance: float = 5.0,
        sensor_angle: float = 45.0,
        num_sensors: int = 2,
        layout_name: Optional[str] = None
    ) -> np.ndarray:
        """Sample odor at multiple sensor positions with validation.
        
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
            
        Raises:
            ValueError: If sensor parameters are invalid
        """
        # Validate sensor parameters
        if sensor_distance <= 0:
            raise ValueError(f"sensor_distance must be positive, got {sensor_distance}")
        if num_sensors <= 0:
            raise ValueError(f"num_sensors must be positive, got {num_sensors}")
            
        start_time = time.perf_counter() if self._enable_logging else None
        
        try:
            # Delegate to utility function and reshape result for a single agent
            odor_values = _sample_odor_at_sensors(
                self, 
                env_array, 
                sensor_distance=sensor_distance,
                sensor_angle=sensor_angle, 
                num_sensors=num_sensors,
                layout_name=layout_name
            )
            
            # Return as a 1D array for single agent
            result = odor_values[0] if odor_values.ndim > 1 else odor_values
            
            # Validate sensor readings
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                if self._logger:
                    self._logger.warning(
                        f"Invalid sensor readings detected",
                        invalid_count=np.sum(np.isnan(result) | np.isinf(result)),
                        sensor_layout=layout_name or "custom",
                        applying_cleanup=True
                    )
                result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Log multi-sensor sampling for debugging
            if self._enable_logging:
                sample_time = (time.perf_counter() - start_time) * 1000
                if self._logger and self._performance_metrics['total_steps'] % 25 == 0:
                    self._logger.trace(
                        f"Multi-sensor sampling completed",
                        sample_time_ms=sample_time,
                        num_sensors=num_sensors,
                        sensor_distance=sensor_distance,
                        mean_odor=float(np.mean(result)),
                        max_odor=float(np.max(result))
                    )
            
            return result
            
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Multi-sensor sampling failed: {str(e)}",
                    error_type=type(e).__name__,
                    num_sensors=num_sensors,
                    sensor_distance=sensor_distance,
                    layout_name=layout_name
                )
            # Return safe default values
            return np.zeros(num_sensors)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for monitoring.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing performance statistics and metrics
        """
        if not self._enable_logging:
            return {}
            
        metrics = {
            'controller_type': 'single_agent',
            'controller_id': self._controller_id,
            'total_steps': self._performance_metrics['total_steps'],
            'num_agents': 1
        }
        
        if self._performance_metrics['step_times']:
            step_times = np.array(self._performance_metrics['step_times'])
            metrics.update({
                'step_time_mean_ms': float(np.mean(step_times)),
                'step_time_std_ms': float(np.std(step_times)),
                'step_time_max_ms': float(np.max(step_times)),
                'step_time_p95_ms': float(np.percentile(step_times, 95)),
                'performance_violations': int(np.sum(step_times > 33.0))
            })
        
        if self._performance_metrics['sample_times']:
            sample_times = np.array(self._performance_metrics['sample_times'])
            metrics.update({
                'sample_time_mean_ms': float(np.mean(sample_times)),
                'sample_time_max_ms': float(np.max(sample_times))
            })
        
        return metrics


class MultiAgentController:
    """Controller for multi-agent navigation with enhanced features.
    
    This implements the NavigatorProtocol for multiple agents, with all data
    represented as arrays without conditional branching. Optimized for performance
    with vectorized operations and comprehensive monitoring for research workflows.
    
    The controller maintains consistent API compatibility with single-agent scenarios
    while providing efficient batch operations for multiple agents. Designed to
    support 100+ agents at 30fps simulation throughput with <33ms frame processing.
    
    Attributes:
        positions: Agent positions as numpy array with shape (num_agents, 2)
        orientations: Agent orientations as numpy array with shape (num_agents,)
        speeds: Agent speeds as numpy array with shape (num_agents,)
        max_speeds: Maximum agent speeds as numpy array with shape (num_agents,)
        angular_velocities: Agent angular velocities as numpy array with shape (num_agents,)
        num_agents: Number of agents in the controller
    
    Examples:
        Multi-agent swarm initialization:
            >>> positions = [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]
            >>> controller = MultiAgentController(positions=positions)
            >>> controller.step(env_array, dt=1.0)
            
        Batch parameter updates:
            >>> new_speeds = np.array([1.0, 1.5, 2.0])
            >>> controller.reset(speeds=new_speeds)
            
        Performance monitoring:
            >>> metrics = controller.get_performance_metrics()
            >>> print(f"Processing {metrics['num_agents']} agents at {metrics['step_time_mean_ms']:.2f}ms/step")
    """
    
    def __init__(
        self,
        positions: Optional[np.ndarray] = None,
        orientations: Optional[np.ndarray] = None,
        speeds: Optional[np.ndarray] = None,
        max_speeds: Optional[np.ndarray] = None,
        angular_velocities: Optional[np.ndarray] = None,
        enable_logging: bool = True,
        controller_id: Optional[str] = None
    ) -> None:
        """Initialize a multi-agent controller with enhanced monitoring.
        
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
        enable_logging : bool, optional
            Enable comprehensive logging integration, by default True
        controller_id : Optional[str], optional
            Unique controller identifier for logging, by default None
            
        Raises:
            ValueError: If array dimensions are inconsistent or constraints violated
        """
        # Ensure we have at least one agent position
        if positions is None:
            self._positions = np.array([[0.0, 0.0]])
        else:
            self._positions = np.array(positions)
            if self._positions.ndim != 2 or self._positions.shape[1] != 2:
                raise ValueError(
                    f"positions must have shape (num_agents, 2), got {self._positions.shape}"
                )

        num_agents = self._positions.shape[0]
        
        # Set defaults for other parameters if not provided
        self._orientations = (
            np.zeros(num_agents) if orientations is None 
            else np.array(orientations) % 360.0  # Normalize to [0, 360)
        )
        self._speeds = np.zeros(num_agents) if speeds is None else np.array(speeds)
        self._max_speeds = np.ones(num_agents) if max_speeds is None else np.array(max_speeds)
        self._angular_velocities = (
            np.zeros(num_agents) if angular_velocities is None 
            else np.array(angular_velocities)
        )
        
        # Validate array shapes and constraints
        self._validate_array_shapes()
        self._validate_speed_constraints()
        
        # Enhanced logging and monitoring
        self._enable_logging = enable_logging
        self._controller_id = controller_id or f"multi_agent_{id(self)}"
        self._performance_metrics = {
            'step_times': [],
            'sample_times': [],
            'total_steps': 0,
            'agents_per_step': []
        }
        
        # Bind logging context for structured logging
        if self._enable_logging and LOGURU_AVAILABLE:
            self._logger = logger.bind(
                controller_type="multi_agent",
                controller_id=self._controller_id,
                num_agents=num_agents
            )
            
            # Log initialization with seed context if available
            seed_manager = get_global_seed_manager()
            if seed_manager:
                self._logger = self._logger.bind(
                    seed_value=seed_manager.seed,
                    experiment_id=seed_manager.experiment_id
                )
            
            # Add Hydra context if available
            if HYDRA_AVAILABLE:
                try:
                    hydra_cfg = HydraConfig.get()
                    self._logger = self._logger.bind(
                        hydra_job_name=hydra_cfg.job.name,
                        hydra_output_dir=hydra_cfg.runtime.output_dir
                    )
                except Exception:
                    # Hydra context not available, continue without it
                    pass
                    
            self._logger.info(
                f"MultiAgentController initialized",
                num_agents=num_agents,
                position_bounds={
                    'x_min': float(np.min(self._positions[:, 0])),
                    'x_max': float(np.max(self._positions[:, 0])),
                    'y_min': float(np.min(self._positions[:, 1])),
                    'y_max': float(np.max(self._positions[:, 1]))
                },
                speed_stats={
                    'mean': float(np.mean(self._speeds)),
                    'max': float(np.max(self._speeds))
                }
            )
        else:
            self._logger = None
    
    def _validate_array_shapes(self) -> None:
        """Validate that all parameter arrays have consistent shapes."""
        num_agents = self._positions.shape[0]
        
        array_checks = [
            ('orientations', self._orientations, (num_agents,)),
            ('speeds', self._speeds, (num_agents,)),
            ('max_speeds', self._max_speeds, (num_agents,)),
            ('angular_velocities', self._angular_velocities, (num_agents,))
        ]
        
        for name, array, expected_shape in array_checks:
            if array.shape != expected_shape:
                raise ValueError(
                    f"{name} shape {array.shape} does not match expected {expected_shape}"
                )
    
    def _validate_speed_constraints(self) -> None:
        """Validate that speeds do not exceed max_speeds for any agent."""
        violations = self._speeds > self._max_speeds
        if np.any(violations):
            violating_agents = np.where(violations)[0]
            raise ValueError(
                f"Speed exceeds max_speed for agents {violating_agents.tolist()}: "
                f"speeds={self._speeds[violations].tolist()}, "
                f"max_speeds={self._max_speeds[violations].tolist()}"
            )
    
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
        """Reset all agents to initial state with comprehensive validation.
        
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
            
        Raises:
            ValueError: If parameter arrays have inconsistent shapes or violate constraints
            
        Notes
        -----
        For stronger type checking, use the MultiAgentParams dataclass:
        
        ```python
        from {{cookiecutter.project_slug}}.core.controllers import MultiAgentParams
        import numpy as np
        
        params = MultiAgentParams(
            positions=np.array([[10, 20], [30, 40]]),
            speeds=np.array([1.5, 2.0])
        )
        navigator.reset_with_params(params)
        ```
        """
        start_time = time.perf_counter() if self._enable_logging else None
        
        try:
            # Create a dictionary of current state for utility function
            controller_state = {
                '_positions': self._positions,
                '_orientations': self._orientations,
                '_speeds': self._speeds,
                '_max_speeds': self._max_speeds,
                '_angular_velocities': self._angular_velocities
            }
            
            # Use the utility function to reset state
            _reset_navigator_state(controller_state, is_single_agent=False, **kwargs)
            
            # Update instance attributes
            self._positions = controller_state['_positions']
            self._orientations = controller_state['_orientations']
            self._speeds = controller_state['_speeds']
            self._max_speeds = controller_state['_max_speeds']
            self._angular_velocities = controller_state['_angular_velocities']
            
            # Validate updated state
            self._validate_array_shapes()
            self._validate_speed_constraints()
            
            # Log successful reset with performance timing
            if self._logger:
                reset_time = (time.perf_counter() - start_time) * 1000
                self._logger.info(
                    f"Multi-agent reset completed",
                    reset_time_ms=reset_time,
                    updated_params=list(kwargs.keys()),
                    num_agents=self.num_agents,
                    position_bounds={
                        'x_min': float(np.min(self._positions[:, 0])),
                        'x_max': float(np.max(self._positions[:, 0])),
                        'y_min': float(np.min(self._positions[:, 1])),
                        'y_max': float(np.max(self._positions[:, 1]))
                    },
                    speed_stats={
                        'mean': float(np.mean(self._speeds)),
                        'max': float(np.max(self._speeds)),
                        'min': float(np.min(self._speeds))
                    }
                )
                
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Multi-agent reset failed: {str(e)}",
                    error_type=type(e).__name__,
                    num_agents=self.num_agents,
                    invalid_params=list(kwargs.keys())
                )
            raise
    
    def reset_with_params(self, params: MultiAgentParams) -> None:
        """Reset all agents using a type-safe parameter object.
        
        This method provides stronger type checking than the kwargs-based reset method
        and integrates seamlessly with Hydra structured configuration for batch experiments.
        
        Parameters
        ----------
        params : MultiAgentParams
            Parameters to update, as a dataclass instance
            
        Raises:
            TypeError: If params is not a MultiAgentParams instance
            ValueError: If parameter arrays have invalid shapes or violate constraints
        """
        if not isinstance(params, MultiAgentParams):
            raise TypeError(
                f"Expected MultiAgentParams, got {type(params)}"
            )
        
        # Convert dataclass to dictionary for the existing function
        kwargs = {k: v for k, v in params.__dict__.items() if v is not None}
        
        if self._logger:
            self._logger.debug(
                f"Resetting multi-agent controller with type-safe parameters",
                param_count=len(kwargs),
                num_agents=self.num_agents
            )
        
        # Delegate to the existing function
        self.reset(**kwargs)
    
    def step(self, env_array: np.ndarray, dt: float = 1.0) -> None:
        """Take a simulation step for all agents with performance optimization.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array (e.g., odor concentration grid)
        dt : float, optional
            Time step size in seconds, by default 1.0
            
        Raises:
            ValueError: If env_array is invalid or dt is non-positive
            RuntimeError: If step processing exceeds performance requirements
        """
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
            
        start_time = time.perf_counter() if self._enable_logging else None
        
        try:
            # Use the utility function to update positions and orientations
            # This performs vectorized operations for optimal performance
            _update_positions_and_orientations(
                self._positions, 
                self._orientations, 
                self._speeds, 
                self._angular_velocities,
                dt=dt
            )
            
            # Track performance metrics
            if self._enable_logging:
                step_time = (time.perf_counter() - start_time) * 1000
                self._performance_metrics['step_times'].append(step_time)
                self._performance_metrics['total_steps'] += 1
                self._performance_metrics['agents_per_step'].append(self.num_agents)
                
                # Calculate throughput (agents × frames / second)
                throughput = self.num_agents * (1000.0 / step_time) if step_time > 0 else 0
                
                # Check performance requirements
                performance_issues = []
                if step_time > 33.0:
                    performance_issues.append(f"step_time:{step_time:.1f}ms")
                if throughput < 3000:  # 100 agents × 30fps
                    performance_issues.append(f"throughput:{throughput:.0f}")
                
                if performance_issues and self._logger:
                    self._logger.warning(
                        f"Performance degradation detected",
                        step_time_ms=step_time,
                        throughput_agents_fps=throughput,
                        num_agents=self.num_agents,
                        dt=dt,
                        issues=performance_issues
                    )
                
                # Log periodic performance summary
                if self._performance_metrics['total_steps'] % 100 == 0 and self._logger:
                    recent_steps = self._performance_metrics['step_times'][-100:]
                    avg_step_time = np.mean(recent_steps)
                    avg_throughput = np.mean([
                        na * (1000.0 / st) for na, st in 
                        zip(self._performance_metrics['agents_per_step'][-100:], recent_steps)
                        if st > 0
                    ])
                    
                    self._logger.debug(
                        f"Multi-agent performance summary",
                        total_steps=self._performance_metrics['total_steps'],
                        avg_step_time_ms=avg_step_time,
                        avg_throughput_agents_fps=avg_throughput,
                        num_agents=self.num_agents,
                        agent_positions_sample=self._positions[:min(3, self.num_agents)].tolist()
                    )
                    
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Multi-agent step execution failed: {str(e)}",
                    error_type=type(e).__name__,
                    num_agents=self.num_agents,
                    dt=dt,
                    env_array_shape=getattr(env_array, 'shape', 'unknown')
                )
            raise RuntimeError(f"Multi-agent step failed: {str(e)}") from e
    
    def sample_odor(self, env_array: np.ndarray) -> np.ndarray:
        """Sample odor at all agent positions with error handling.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array
            
        Returns
        -------
        np.ndarray
            Odor values at each agent's position, shape (num_agents,)
            
        Raises:
            ValueError: If env_array is invalid or sampling fails
        """
        return self.read_single_antenna_odor(env_array)
    
    def read_single_antenna_odor(self, env_array: np.ndarray) -> np.ndarray:
        """Sample odor at each agent's position with batch optimization.
        
        Parameters
        ----------
        env_array : np.ndarray
            The environment array
            
        Returns
        -------
        np.ndarray
            Odor values at each agent's position, shape (num_agents,)
            
        Raises:
            ValueError: If sampling fails or returns invalid values
        """
        start_time = time.perf_counter() if self._enable_logging else None
        
        try:
            # Use the utility function to read odor values
            odor_values = _read_odor_values(env_array, self._positions)
            
            # Validate odor values
            invalid_mask = np.isnan(odor_values) | np.isinf(odor_values)
            if np.any(invalid_mask):
                if self._logger:
                    invalid_count = np.sum(invalid_mask)
                    self._logger.warning(
                        f"Invalid odor values detected for multi-agent sampling",
                        invalid_count=invalid_count,
                        total_agents=self.num_agents,
                        invalid_fraction=invalid_count / self.num_agents,
                        applying_cleanup=True
                    )
                odor_values = np.nan_to_num(odor_values, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Track sampling performance
            if self._enable_logging:
                sample_time = (time.perf_counter() - start_time) * 1000
                self._performance_metrics['sample_times'].append(sample_time)
                
                # Log detailed sampling for debugging
                if self._logger and self._performance_metrics['total_steps'] % 50 == 0:
                    self._logger.trace(
                        f"Multi-agent odor sampling completed",
                        sample_time_ms=sample_time,
                        num_agents=self.num_agents,
                        odor_stats={
                            'mean': float(np.mean(odor_values)),
                            'max': float(np.max(odor_values)),
                            'min': float(np.min(odor_values)),
                            'std': float(np.std(odor_values))
                        }
                    )
            
            return odor_values
            
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Multi-agent odor sampling failed: {str(e)}",
                    error_type=type(e).__name__,
                    num_agents=self.num_agents,
                    env_array_shape=getattr(env_array, 'shape', 'unknown')
                )
            # Return safe default values
            return np.zeros(self.num_agents)
    
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
            
        Raises:
            ValueError: If sensor parameters are invalid
        """
        # Validate sensor parameters
        if sensor_distance <= 0:
            raise ValueError(f"sensor_distance must be positive, got {sensor_distance}")
        if num_sensors <= 0:
            raise ValueError(f"num_sensors must be positive, got {num_sensors}")
            
        start_time = time.perf_counter() if self._enable_logging else None
        
        try:
            # Delegate to utility function
            odor_values = _sample_odor_at_sensors(
                self, 
                env_array, 
                sensor_distance=sensor_distance,
                sensor_angle=sensor_angle, 
                num_sensors=num_sensors,
                layout_name=layout_name
            )
            
            # Validate sensor readings
            invalid_mask = np.isnan(odor_values) | np.isinf(odor_values)
            if np.any(invalid_mask):
                if self._logger:
                    invalid_count = np.sum(invalid_mask)
                    total_readings = odor_values.size
                    self._logger.warning(
                        f"Invalid multi-sensor readings detected",
                        invalid_count=invalid_count,
                        total_readings=total_readings,
                        invalid_fraction=invalid_count / total_readings,
                        sensor_layout=layout_name or "custom",
                        applying_cleanup=True
                    )
                odor_values = np.nan_to_num(odor_values, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Log multi-sensor sampling for debugging
            if self._enable_logging:
                sample_time = (time.perf_counter() - start_time) * 1000
                if self._logger and self._performance_metrics['total_steps'] % 25 == 0:
                    self._logger.trace(
                        f"Multi-agent multi-sensor sampling completed",
                        sample_time_ms=sample_time,
                        num_agents=self.num_agents,
                        num_sensors=num_sensors,
                        sensor_distance=sensor_distance,
                        total_readings=odor_values.size,
                        odor_stats={
                            'mean': float(np.mean(odor_values)),
                            'max': float(np.max(odor_values)),
                            'agents_with_max_reading': int(np.sum(np.any(odor_values > 0.8, axis=1)))
                        }
                    )
            
            return odor_values
            
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Multi-agent multi-sensor sampling failed: {str(e)}",
                    error_type=type(e).__name__,
                    num_agents=self.num_agents,
                    num_sensors=num_sensors,
                    sensor_distance=sensor_distance,
                    layout_name=layout_name
                )
            # Return safe default values
            return np.zeros((self.num_agents, num_sensors))
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for monitoring and optimization.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing detailed performance statistics and metrics
        """
        if not self._enable_logging:
            return {}
            
        metrics = {
            'controller_type': 'multi_agent',
            'controller_id': self._controller_id,
            'total_steps': self._performance_metrics['total_steps'],
            'num_agents': self.num_agents
        }
        
        if self._performance_metrics['step_times']:
            step_times = np.array(self._performance_metrics['step_times'])
            agents_per_step = np.array(self._performance_metrics['agents_per_step'])
            
            # Calculate throughput metrics
            throughputs = [
                na * (1000.0 / st) for na, st in zip(agents_per_step, step_times) if st > 0
            ]
            
            metrics.update({
                'step_time_mean_ms': float(np.mean(step_times)),
                'step_time_std_ms': float(np.std(step_times)),
                'step_time_max_ms': float(np.max(step_times)),
                'step_time_p95_ms': float(np.percentile(step_times, 95)),
                'performance_violations': int(np.sum(step_times > 33.0)),
                'throughput_mean_agents_fps': float(np.mean(throughputs)) if throughputs else 0,
                'throughput_max_agents_fps': float(np.max(throughputs)) if throughputs else 0,
                'throughput_violations': int(np.sum(np.array(throughputs) < 3000))
            })
        
        if self._performance_metrics['sample_times']:
            sample_times = np.array(self._performance_metrics['sample_times'])
            metrics.update({
                'sample_time_mean_ms': float(np.mean(sample_times)),
                'sample_time_max_ms': float(np.max(sample_times))
            })
        
        return metrics


# Utility functions for state management and operations
# These functions are adapted from the original navigator_utils but updated for the new structure

def _read_odor_values(env_array: np.ndarray, positions: np.ndarray) -> np.ndarray:
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


def _update_positions_and_orientations(
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


def _reset_navigator_state(
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
        Dictionary of current controller state arrays
    is_single_agent : bool
        Whether this is a single agent controller
    **kwargs
        Parameters to update
    
    Returns
    -------
    None
        The function modifies the input state dictionary in-place
    
    Raises
    ------
    ValueError
        If invalid parameter keys are provided
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
                if kwarg_key == orientation_key:
                    value = value % 360.0  # Normalize orientation
                controller_state[attr_key] = np.array([value])
            else:
                if kwarg_key == orientation_key:
                    value = np.array(value) % 360.0  # Normalize orientations
                controller_state[attr_key] = np.array(value)


def _sample_odor_at_sensors(
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
    # Calculate sensor positions using a simplified approach
    # This is a basic implementation - in practice you might want to import
    # the full sensor layout utilities from a dedicated module
    
    num_agents = navigator.num_agents
    sensor_positions = np.zeros((num_agents, num_sensors, 2))
    
    # Simple sensor layout: place sensors at fixed offsets
    for agent_idx in range(num_agents):
        agent_pos = navigator.positions[agent_idx]
        agent_orientation = navigator.orientations[agent_idx]
        
        for sensor_idx in range(num_sensors):
            # Calculate sensor angle relative to agent orientation
            if num_sensors == 1:
                relative_angle = 0
            elif num_sensors == 2:
                relative_angle = (sensor_idx - 0.5) * sensor_angle
            else:
                relative_angle = (sensor_idx - (num_sensors - 1) / 2) * sensor_angle
            
            # Convert to global angle
            global_angle = agent_orientation + relative_angle
            
            # Calculate sensor position
            sensor_x = agent_pos[0] + sensor_distance * np.cos(np.deg2rad(global_angle))
            sensor_y = agent_pos[1] + sensor_distance * np.sin(np.deg2rad(global_angle))
            
            sensor_positions[agent_idx, sensor_idx] = [sensor_x, sensor_y]
    
    # Read odor values at sensor positions
    odor_values = _read_odor_values(env_array, sensor_positions.reshape(-1, 2))
    
    # Reshape to (num_agents, num_sensors)
    odor_values = odor_values.reshape(num_agents, num_sensors)
    
    return odor_values


def create_controller_from_config(
    config: Union[DictConfig, Dict[str, Any], NavigatorConfig],
    controller_id: Optional[str] = None,
    enable_logging: bool = True
) -> Union[SingleAgentController, MultiAgentController]:
    """
    Create a navigation controller from configuration with automatic type detection.
    
    This factory function automatically detects whether to create a single-agent or
    multi-agent controller based on the configuration parameters and creates the
    appropriate controller type with full Hydra integration.
    
    Parameters
    ----------
    config : Union[DictConfig, Dict[str, Any], NavigatorConfig]
        Configuration object containing navigation parameters
    controller_id : Optional[str], optional
        Unique identifier for the controller, by default None
    enable_logging : bool, optional
        Enable comprehensive logging integration, by default True
        
    Returns
    -------
    Union[SingleAgentController, MultiAgentController]
        Configured navigation controller instance
        
    Raises:
        ValueError: If configuration is invalid or inconsistent
        TypeError: If configuration type is not supported
        
    Examples:
        Single-agent from dict:
            >>> config = {"position": [10.0, 20.0], "speed": 1.5}
            >>> controller = create_controller_from_config(config)
            >>> isinstance(controller, SingleAgentController)
            True
            
        Multi-agent from Hydra config:
            >>> from omegaconf import DictConfig
            >>> config = DictConfig({
            ...     "positions": [[0, 0], [10, 10]], 
            ...     "speeds": [1.0, 1.5]
            ... })
            >>> controller = create_controller_from_config(config)
            >>> isinstance(controller, MultiAgentController)
            True
    """
    start_time = time.perf_counter() if enable_logging else None
    
    try:
        # Handle different configuration types
        if isinstance(config, NavigatorConfig):
            # Pydantic model - extract relevant parameters
            config_dict = config.model_dump(exclude_none=True)
        elif isinstance(config, DictConfig) and HYDRA_AVAILABLE:
            # Hydra OmegaConf configuration
            config_dict = OmegaConf.to_container(config, resolve=True)
        elif isinstance(config, dict):
            # Regular dictionary
            config_dict = config.copy()
        else:
            raise TypeError(
                f"Unsupported configuration type: {type(config)}. "
                f"Expected DictConfig, dict, or NavigatorConfig"
            )
        
        # Detect controller type based on configuration parameters
        has_multi_params = any([
            'positions' in config_dict,
            'num_agents' in config_dict and config_dict.get('num_agents', 1) > 1,
            isinstance(config_dict.get('orientations'), (list, np.ndarray)),
            isinstance(config_dict.get('speeds'), (list, np.ndarray)),
            isinstance(config_dict.get('max_speeds'), (list, np.ndarray)),
            isinstance(config_dict.get('angular_velocities'), (list, np.ndarray))
        ])
        
        has_single_params = 'position' in config_dict
        
        # Validate mode exclusivity
        if has_multi_params and has_single_params:
            raise ValueError(
                "Configuration contains both single-agent (position) and "
                "multi-agent (positions, arrays) parameters. Use only one mode."
            )
        
        # Create appropriate controller
        if has_multi_params:
            # Multi-agent controller
            controller = MultiAgentController(
                positions=config_dict.get('positions'),
                orientations=config_dict.get('orientations'),
                speeds=config_dict.get('speeds'),
                max_speeds=config_dict.get('max_speeds'),
                angular_velocities=config_dict.get('angular_velocities'),
                enable_logging=enable_logging,
                controller_id=controller_id
            )
            
        else:
            # Single-agent controller (default)
            controller = SingleAgentController(
                position=config_dict.get('position'),
                orientation=config_dict.get('orientation', 0.0),
                speed=config_dict.get('speed', 0.0),
                max_speed=config_dict.get('max_speed', 1.0),
                angular_velocity=config_dict.get('angular_velocity', 0.0),
                enable_logging=enable_logging,
                controller_id=controller_id
            )
        
        # Log successful creation with performance timing
        if enable_logging and LOGURU_AVAILABLE:
            creation_time = (time.perf_counter() - start_time) * 1000
            logger.bind(
                controller_type=type(controller).__name__,
                controller_id=controller_id,
                creation_time_ms=creation_time,
                num_agents=controller.num_agents
            ).info(
                f"Controller created from configuration",
                config_keys=list(config_dict.keys())
            )
        
        return controller
        
    except Exception as e:
        if enable_logging and LOGURU_AVAILABLE:
            logger.error(
                f"Controller creation failed: {str(e)}",
                error_type=type(e).__name__,
                config_type=type(config).__name__,
                controller_id=controller_id
            )
        raise


# Export public API
__all__ = [
    'SingleAgentController',
    'MultiAgentController', 
    'SingleAgentParams',
    'MultiAgentParams',
    'create_controller_from_config'
]