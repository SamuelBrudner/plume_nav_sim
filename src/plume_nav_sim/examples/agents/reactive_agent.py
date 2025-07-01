"""
Memoryless reactive agent implementation demonstrating pure reactive navigation.

This module implements a ReactiveAgent that follows odor gradients using instantaneous 
sensor readings without maintaining internal state, serving as an example of memory-less 
cognitive modeling approach that can be toggled via configuration without code changes.

The ReactiveAgent demonstrates the sensor abstraction layer by using GradientSensor for 
directional navigation cues instead of direct odor sampling, supporting the modular 
architecture requirements from Section 0.2.1 of the technical specification.

Key Design Principles:
- Agent-agnostic design: Implements NavigatorProtocol without memory assumptions
- Sensor-based perception: Uses configurable sensors rather than direct field sampling  
- Instant response: Reacts immediately to local environmental gradients
- Zero internal state: No memory dependencies or historical information storage
- Configuration-driven: Runtime parameter adjustment via Hydra integration
- Protocol compliance: Full NavigatorProtocol implementation for seamless integration

The agent uses a simple gradient ascent algorithm with configurable step size and turning 
rate parameters, demonstrating how navigation strategies can be implemented through the 
sensor abstraction layer while maintaining compatibility with both memory-based and 
memory-less simulation scenarios.

Performance Requirements:
- Step execution: <1ms per agent for real-time response
- Sensor integration: <0.1ms additional latency per gradient computation
- Memory footprint: <1KB per agent (excluding sensor overhead)
- Configuration overhead: <10ms for parameter updates

Examples:
    Basic reactive agent usage:
    >>> from plume_nav_sim.examples.agents.reactive_agent import ReactiveAgent
    >>> from plume_nav_sim.core.sensors import create_sensor_from_config
    >>> 
    >>> # Create gradient sensor for navigation
    >>> sensor_config = {
    ...     'type': 'GradientSensor',
    ...     'spatial_resolution': (0.5, 0.5),
    ...     'method': 'central'
    ... }
    >>> gradient_sensor = create_sensor_from_config(sensor_config)
    >>> 
    >>> # Create reactive agent with sensor
    >>> agent = ReactiveAgent(
    ...     position=(10.0, 20.0),
    ...     speed=1.5,
    ...     max_speed=3.0,
    ...     gradient_sensor=gradient_sensor,
    ...     step_size=0.8,
    ...     turning_rate=45.0
    ... )
    >>> 
    >>> # Agent reacts instantly to environment
    >>> agent.step(env_array, dt=1.0)
    >>> current_position = agent.positions[0]
    
    Configuration-driven instantiation:
    >>> from plume_nav_sim.examples.agents.reactive_agent import create_reactive_agent_from_config
    >>> config = {
    ...     'position': (0.0, 0.0),
    ...     'max_speed': 2.0,
    ...     'step_size': 1.0,
    ...     'turning_rate': 30.0,
    ...     'sensor': {
    ...         'type': 'GradientSensor',
    ...         'spatial_resolution': (0.2, 0.2)
    ...     }
    ... }
    >>> agent = create_reactive_agent_from_config(config)
    
    CLI usage example:
    >>> # From command line:
    >>> # python -m plume_nav_sim.examples.agents.reactive_agent --config-name=reactive_agent
    >>> # 
    >>> # Or programmatically:
    >>> if __name__ == "__main__":
    ...     main()  # Runs CLI demo with Hydra configuration

Notes:
    The ReactiveAgent serves as a reference implementation for memory-less navigation 
    strategies and demonstrates the sensor abstraction layer integration. It can be 
    extended to support different gradient-following algorithms while maintaining 
    the same protocol-compliant interface.
    
    The agent's memoryless design ensures it can be used in simulation scenarios 
    where memory features are disabled or where instant response to environmental 
    changes is required without historical bias.
"""

from __future__ import annotations
import time
import warnings
from typing import Optional, Union, Dict, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np

# Core protocol and controller imports for NavigatorProtocol compliance
from ...core.protocols import NavigatorProtocol
from ...core.controllers import SingleAgentController

# Sensor abstraction layer for gradient-based navigation
from ...core.sensors import (
    create_sensor_from_config, 
    GradientSensor,
    SensorProtocol,
    validate_sensor_config
)

# Configuration management and CLI support
try:
    import hydra
    from hydra import compose, initialize
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    hydra = None
    DictConfig = dict
    OmegaConf = None
    HYDRA_AVAILABLE = False

# Enhanced logging support
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    LOGURU_AVAILABLE = False

# Type checking imports
if TYPE_CHECKING:
    from ...core.sensors import SensorProtocol


@dataclass
class ReactiveAgentConfig:
    """
    Configuration schema for ReactiveAgent with Hydra integration support.
    
    This dataclass defines the configuration parameters for reactive agents,
    supporting both programmatic instantiation and Hydra-based configuration
    management for research workflow integration.
    
    Attributes:
        position: Initial agent position coordinates [x, y]
        orientation: Initial orientation in degrees (0 = right, 90 = up)
        speed: Initial speed in units per time step
        max_speed: Maximum allowed speed in units per time step
        angular_velocity: Initial angular velocity in degrees per second
        step_size: Gradient step size multiplier for movement adjustments
        turning_rate: Maximum turning rate in degrees per second
        gradient_threshold: Minimum gradient magnitude to trigger movement
        sensor: Sensor configuration dictionary for gradient computation
        enable_logging: Enable comprehensive logging and performance monitoring
        performance_monitoring: Enable detailed performance metrics collection
        
    Examples:
        Basic configuration:
        >>> config = ReactiveAgentConfig(
        ...     position=(10.0, 20.0),
        ...     max_speed=2.0,
        ...     step_size=0.8,
        ...     turning_rate=45.0
        ... )
        
        Advanced configuration with custom sensor:
        >>> config = ReactiveAgentConfig(
        ...     position=(0.0, 0.0),
        ...     max_speed=3.0,
        ...     step_size=1.2,
        ...     turning_rate=60.0,
        ...     gradient_threshold=0.01,
        ...     sensor={
        ...         'type': 'GradientSensor',
        ...         'spatial_resolution': (0.1, 0.1),
        ...         'method': 'central',
        ...         'order': 2
        ...     },
        ...     enable_logging=True,
        ...     performance_monitoring=True
        ... )
    """
    position: Tuple[float, float] = (0.0, 0.0)
    orientation: float = 0.0
    speed: float = 0.0
    max_speed: float = 1.0
    angular_velocity: float = 0.0
    step_size: float = 1.0
    turning_rate: float = 30.0
    gradient_threshold: float = 1e-6
    sensor: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'GradientSensor',
        'spatial_resolution': (0.5, 0.5),
        'method': 'central'
    })
    enable_logging: bool = True
    performance_monitoring: bool = False
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.speed > self.max_speed:
            raise ValueError(f"speed ({self.speed}) cannot exceed max_speed ({self.max_speed})")
        
        if self.step_size <= 0:
            raise ValueError(f"step_size must be positive, got {self.step_size}")
        
        if self.turning_rate <= 0:
            raise ValueError(f"turning_rate must be positive, got {self.turning_rate}")
        
        if self.gradient_threshold < 0:
            raise ValueError(f"gradient_threshold must be non-negative, got {self.gradient_threshold}")
        
        # Validate sensor configuration
        if self.sensor:
            try:
                validate_sensor_config(self.sensor)
            except Exception as e:
                raise ValueError(f"Invalid sensor configuration: {e}")


class ReactiveAgent:
    """
    Memoryless reactive agent implementing gradient-following navigation strategy.
    
    This agent demonstrates pure reactive navigation by immediately responding to local 
    concentration gradients through GradientSensor without maintaining internal state 
    or memory. Serves as an example of memory-less cognitive modeling approach that 
    can be toggled via configuration without code changes.
    
    The agent implements NavigatorProtocol for seamless integration with existing 
    simulation infrastructure while using the sensor abstraction layer for perception,
    supporting the goal of agent-agnostic design where the simulator core makes no 
    assumptions about agent internal logic.
    
    Key Features:
    - Pure reactive behavior: No internal state or memory dependencies
    - Sensor-based perception: Uses GradientSensor for directional navigation cues
    - Configurable parameters: Runtime adjustment of step size and turning rate
    - Protocol compliance: Full NavigatorProtocol implementation
    - Performance monitoring: Optional detailed metrics collection
    - Gradient ascent algorithm: Simple but effective navigation strategy
    
    Navigation Algorithm:
    1. Compute concentration gradient at current position using GradientSensor
    2. If gradient magnitude exceeds threshold, adjust orientation toward gradient
    3. Set speed proportional to gradient magnitude (clamped to max_speed)
    4. Update position and orientation using standard NavigatorProtocol mechanics
    
    Performance Characteristics:
    - Step execution: <1ms per agent for real-time response
    - Memory footprint: <1KB per agent (excluding sensor overhead)
    - Gradient computation: <0.1ms additional latency per step
    - Configuration updates: <10ms for parameter changes
    
    Examples:
        Basic agent creation:
        >>> gradient_sensor = create_sensor_from_config({
        ...     'type': 'GradientSensor',
        ...     'spatial_resolution': (0.5, 0.5)
        ... })
        >>> agent = ReactiveAgent(
        ...     position=(10.0, 20.0),
        ...     gradient_sensor=gradient_sensor,
        ...     step_size=0.8,
        ...     turning_rate=45.0
        ... )
        
        Configuration-driven creation:
        >>> config = ReactiveAgentConfig(
        ...     position=(0.0, 0.0),
        ...     max_speed=2.0,
        ...     step_size=1.0
        ... )
        >>> agent = ReactiveAgent.from_config(config)
        
        Performance monitoring:
        >>> agent = ReactiveAgent(
        ...     position=(5.0, 5.0),
        ...     performance_monitoring=True
        ... )
        >>> agent.step(env_array, dt=1.0)
        >>> metrics = agent.get_performance_metrics()
        >>> print(f"Step time: {metrics['last_step_time_ms']:.3f} ms")
    """
    
    def __init__(
        self,
        position: Optional[Tuple[float, float]] = None,
        orientation: float = 0.0,
        speed: float = 0.0,
        max_speed: float = 1.0,
        angular_velocity: float = 0.0,
        gradient_sensor: Optional[SensorProtocol] = None,
        step_size: float = 1.0,
        turning_rate: float = 30.0,
        gradient_threshold: float = 1e-6,
        enable_logging: bool = True,
        performance_monitoring: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize reactive agent with gradient-following behavior.
        
        Args:
            position: Initial (x, y) position, defaults to (0, 0)
            orientation: Initial orientation in degrees, defaults to 0.0
            speed: Initial speed, defaults to 0.0
            max_speed: Maximum allowed speed, defaults to 1.0
            angular_velocity: Initial angular velocity in degrees/second, defaults to 0.0
            gradient_sensor: Sensor for gradient computation, creates default if None
            step_size: Gradient step size multiplier, defaults to 1.0
            turning_rate: Maximum turning rate in degrees/second, defaults to 30.0
            gradient_threshold: Minimum gradient magnitude to trigger movement
            enable_logging: Enable comprehensive logging, defaults to True
            performance_monitoring: Enable detailed performance metrics, defaults to False
            **kwargs: Additional parameters passed to underlying controller
            
        Raises:
            ValueError: If parameters violate constraints or sensor is invalid
            ImportError: If required sensor implementations are not available
        """
        # Initialize underlying SingleAgentController for NavigatorProtocol compliance
        self._controller = SingleAgentController(
            position=position,
            orientation=orientation,
            speed=speed,
            max_speed=max_speed,
            angular_velocity=angular_velocity,
            enable_logging=enable_logging,
            **kwargs
        )
        
        # Reactive agent specific parameters
        self._step_size = step_size
        self._turning_rate = turning_rate
        self._gradient_threshold = gradient_threshold
        self._enable_logging = enable_logging
        self._performance_monitoring = performance_monitoring
        
        # Validate reactive agent parameters
        if self._step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        if self._turning_rate <= 0:
            raise ValueError(f"turning_rate must be positive, got {turning_rate}")
        if self._gradient_threshold < 0:
            raise ValueError(f"gradient_threshold must be non-negative, got {gradient_threshold}")
        
        # Initialize gradient sensor for reactive navigation
        if gradient_sensor is None:
            # Create default GradientSensor with reasonable parameters
            sensor_config = {
                'type': 'GradientSensor',
                'spatial_resolution': (0.5, 0.5),
                'method': 'central'
            }
            try:
                self._gradient_sensor = create_sensor_from_config(sensor_config)
            except ImportError as e:
                if enable_logging and LOGURU_AVAILABLE:
                    logger.warning(
                        f"Failed to create default GradientSensor: {e}. "
                        f"Using fallback gradient computation."
                    )
                # Create minimal fallback sensor
                self._gradient_sensor = GradientSensor()
        else:
            self._gradient_sensor = gradient_sensor
        
        # Performance metrics tracking
        self._performance_metrics = {
            'total_steps': 0,
            'gradient_computations': 0,
            'step_times': [],
            'gradient_computation_times': [],
            'movement_decisions': 0,
            'stationary_decisions': 0,
            'total_distance_traveled': 0.0,
            'average_gradient_magnitude': 0.0
        }
        
        # Enhanced logging setup
        if self._enable_logging and LOGURU_AVAILABLE:
            self._logger = logger.bind(
                agent_type="ReactiveAgent",
                agent_id=id(self),
                step_size=self._step_size,
                turning_rate=self._turning_rate,
                gradient_threshold=self._gradient_threshold,
                sensor_type=type(self._gradient_sensor).__name__
            )
            
            self._logger.info(
                "ReactiveAgent initialized with gradient-following behavior",
                position=position or (0.0, 0.0),
                max_speed=max_speed,
                step_size=self._step_size,
                turning_rate=self._turning_rate,
                gradient_threshold=self._gradient_threshold,
                sensor_type=type(self._gradient_sensor).__name__,
                performance_monitoring=self._performance_monitoring
            )
        else:
            self._logger = None
    
    # NavigatorProtocol property implementations (delegate to controller)
    
    @property
    def positions(self) -> np.ndarray:
        """Get agent position as numpy array with shape (1, 2)."""
        return self._controller.positions
    
    @property
    def orientations(self) -> np.ndarray:
        """Get agent orientation as numpy array with shape (1,)."""
        return self._controller.orientations
    
    @property
    def speeds(self) -> np.ndarray:
        """Get agent speed as numpy array with shape (1,)."""
        return self._controller.speeds
    
    @property
    def max_speeds(self) -> np.ndarray:
        """Get maximum agent speed as numpy array with shape (1,)."""
        return self._controller.max_speeds
    
    @property
    def angular_velocities(self) -> np.ndarray:
        """Get agent angular velocity as numpy array with shape (1,)."""
        return self._controller.angular_velocities
    
    @property
    def num_agents(self) -> int:
        """Get the number of agents, always 1 for ReactiveAgent."""
        return 1
    
    # Reactive agent specific properties
    
    @property
    def step_size(self) -> float:
        """Get current gradient step size multiplier."""
        return self._step_size
    
    @step_size.setter
    def step_size(self, value: float) -> None:
        """Set gradient step size multiplier with validation."""
        if value <= 0:
            raise ValueError(f"step_size must be positive, got {value}")
        
        old_value = self._step_size
        self._step_size = value
        
        if self._logger:
            self._logger.debug(
                "Updated step_size parameter",
                old_value=old_value,
                new_value=value
            )
    
    @property
    def turning_rate(self) -> float:
        """Get current maximum turning rate in degrees per second."""
        return self._turning_rate
    
    @turning_rate.setter
    def turning_rate(self, value: float) -> None:
        """Set maximum turning rate with validation."""
        if value <= 0:
            raise ValueError(f"turning_rate must be positive, got {value}")
        
        old_value = self._turning_rate
        self._turning_rate = value
        
        if self._logger:
            self._logger.debug(
                "Updated turning_rate parameter",
                old_value=old_value,
                new_value=value
            )
    
    @property
    def gradient_threshold(self) -> float:
        """Get current gradient magnitude threshold for movement."""
        return self._gradient_threshold
    
    @gradient_threshold.setter
    def gradient_threshold(self, value: float) -> None:
        """Set gradient threshold with validation."""
        if value < 0:
            raise ValueError(f"gradient_threshold must be non-negative, got {value}")
        
        old_value = self._gradient_threshold
        self._gradient_threshold = value
        
        if self._logger:
            self._logger.debug(
                "Updated gradient_threshold parameter",
                old_value=old_value,
                new_value=value
            )
    
    # NavigatorProtocol method implementations
    
    def reset(self, **kwargs: Any) -> None:
        """
        Reset agent to initial state with optional parameter overrides.
        
        Args:
            **kwargs: Optional parameters to override initial settings.
                Valid keys include all NavigatorProtocol parameters plus:
                - step_size: New gradient step size multiplier
                - turning_rate: New maximum turning rate
                - gradient_threshold: New gradient magnitude threshold
        """
        # Extract reactive agent specific parameters
        if 'step_size' in kwargs:
            self.step_size = kwargs.pop('step_size')
        if 'turning_rate' in kwargs:
            self.turning_rate = kwargs.pop('turning_rate')
        if 'gradient_threshold' in kwargs:
            self.gradient_threshold = kwargs.pop('gradient_threshold')
        
        # Reset underlying controller
        self._controller.reset(**kwargs)
        
        # Reset performance metrics
        if self._performance_monitoring:
            self._performance_metrics = {
                'total_steps': 0,
                'gradient_computations': 0,
                'step_times': [],
                'gradient_computation_times': [],
                'movement_decisions': 0,
                'stationary_decisions': 0,
                'total_distance_traveled': 0.0,
                'average_gradient_magnitude': 0.0
            }
        
        if self._logger:
            self._logger.info(
                "ReactiveAgent reset completed",
                updated_params=list(kwargs.keys()),
                position=self.positions[0].tolist(),
                orientation=float(self.orientations[0]),
                step_size=self._step_size,
                turning_rate=self._turning_rate,
                gradient_threshold=self._gradient_threshold
            )
    
    def step(self, env_array: np.ndarray, dt: float = 1.0) -> None:
        """
        Execute reactive navigation step using gradient-following algorithm.
        
        This method implements the core reactive behavior:
        1. Compute concentration gradient at current position
        2. Determine new orientation based on gradient direction
        3. Set speed proportional to gradient magnitude
        4. Update position and orientation via underlying controller
        
        Args:
            env_array: Environment array (e.g., odor concentration grid)
            dt: Time step size in seconds, defaults to 1.0
            
        Raises:
            ValueError: If env_array is invalid or dt is non-positive
            RuntimeError: If gradient computation or navigation fails
        """
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
        
        start_time = time.perf_counter() if self._performance_monitoring else None
        
        try:
            # Get current position for gradient computation
            current_position = self.positions[0]  # Shape: (2,)
            
            # Compute concentration gradient using sensor abstraction
            gradient_start_time = time.perf_counter() if self._performance_monitoring else None
            
            try:
                # Ensure position is in correct format for gradient sensor
                position_for_sensor = current_position.reshape(1, -1)  # Shape: (1, 2)
                gradient = self._gradient_sensor.compute_gradient(env_array, position_for_sensor)
                
                # Handle different gradient return formats
                if gradient.ndim > 1:
                    gradient = gradient[0]  # Extract single agent gradient
                
                # Ensure gradient is a 2D vector
                if gradient.shape != (2,):
                    if self._logger:
                        self._logger.warning(
                            f"Unexpected gradient shape: {gradient.shape}, expected (2,). Using zero gradient."
                        )
                    gradient = np.zeros(2)
                
            except Exception as e:
                if self._logger:
                    self._logger.warning(
                        f"Gradient computation failed: {e}. Using zero gradient for safety."
                    )
                gradient = np.zeros(2)
            
            if self._performance_monitoring and gradient_start_time:
                gradient_computation_time = (time.perf_counter() - gradient_start_time) * 1000
                self._performance_metrics['gradient_computation_times'].append(gradient_computation_time)
                self._performance_metrics['gradient_computations'] += 1
            
            # Compute gradient magnitude for decision making
            gradient_magnitude = np.linalg.norm(gradient)
            
            # Update average gradient magnitude for performance metrics
            if self._performance_monitoring:
                if self._performance_metrics['total_steps'] > 0:
                    current_avg = self._performance_metrics['average_gradient_magnitude']
                    n = self._performance_metrics['total_steps']
                    self._performance_metrics['average_gradient_magnitude'] = (
                        (current_avg * n + gradient_magnitude) / (n + 1)
                    )
                else:
                    self._performance_metrics['average_gradient_magnitude'] = gradient_magnitude
            
            # Store previous position for distance tracking
            previous_position = current_position.copy() if self._performance_monitoring else None
            
            # Reactive decision making: respond only if gradient exceeds threshold
            if gradient_magnitude > self._gradient_threshold:
                # Compute desired orientation from gradient direction
                desired_orientation = np.degrees(np.arctan2(gradient[1], gradient[0]))
                
                # Normalize to [0, 360) range
                desired_orientation = desired_orientation % 360.0
                
                # Compute angular difference and apply turning rate limits
                current_orientation = float(self.orientations[0])
                angle_diff = desired_orientation - current_orientation
                
                # Handle angle wrapping for shortest path
                if angle_diff > 180:
                    angle_diff -= 360
                elif angle_diff < -180:
                    angle_diff += 360
                
                # Limit turning rate based on time step
                max_turn = self._turning_rate * dt
                if abs(angle_diff) > max_turn:
                    angle_diff = np.sign(angle_diff) * max_turn
                
                # Set new angular velocity for smooth turning
                new_angular_velocity = angle_diff / dt if dt > 0 else 0.0
                
                # Set speed proportional to gradient magnitude with step size scaling
                new_speed = min(
                    gradient_magnitude * self._step_size,
                    float(self.max_speeds[0])
                )
                
                # Update controller state for movement
                self._controller.reset(
                    speed=new_speed,
                    angular_velocity=new_angular_velocity
                )
                
                if self._performance_monitoring:
                    self._performance_metrics['movement_decisions'] += 1
                
                if self._logger and self._performance_metrics['total_steps'] % 10 == 0:
                    self._logger.debug(
                        "Reactive movement decision",
                        gradient_magnitude=gradient_magnitude,
                        gradient_direction=gradient.tolist(),
                        desired_orientation=desired_orientation,
                        angle_diff=angle_diff,
                        new_speed=new_speed,
                        position=current_position.tolist()
                    )
            
            else:
                # Gradient below threshold: remain stationary or reduce speed
                self._controller.reset(speed=0.0, angular_velocity=0.0)
                
                if self._performance_monitoring:
                    self._performance_metrics['stationary_decisions'] += 1
                
                if self._logger and self._performance_metrics['total_steps'] % 50 == 0:
                    self._logger.trace(
                        "Reactive stationary decision",
                        gradient_magnitude=gradient_magnitude,
                        threshold=self._gradient_threshold,
                        position=current_position.tolist()
                    )
            
            # Execute movement using underlying controller
            self._controller.step(env_array, dt)
            
            # Update performance metrics
            if self._performance_monitoring:
                self._performance_metrics['total_steps'] += 1
                
                if previous_position is not None:
                    distance_moved = np.linalg.norm(self.positions[0] - previous_position)
                    self._performance_metrics['total_distance_traveled'] += distance_moved
                
                if start_time:
                    step_time = (time.perf_counter() - start_time) * 1000
                    self._performance_metrics['step_times'].append(step_time)
                    
                    # Check performance requirements
                    if step_time > 1.0 and self._logger:
                        self._logger.warning(
                            "Reactive agent step exceeded 1ms requirement",
                            step_time_ms=step_time,
                            gradient_magnitude=gradient_magnitude,
                            performance_degradation=True
                        )
            
            # Periodic performance logging
            if (self._logger and self._performance_monitoring and 
                self._performance_metrics['total_steps'] % 100 == 0):
                
                metrics = self.get_performance_metrics()
                self._logger.debug(
                    "Reactive agent performance summary",
                    total_steps=metrics['total_steps'],
                    avg_step_time_ms=metrics.get('avg_step_time_ms', 0),
                    avg_gradient_magnitude=metrics['average_gradient_magnitude'],
                    movement_ratio=metrics.get('movement_ratio', 0),
                    total_distance=metrics['total_distance_traveled']
                )
        
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Reactive agent step failed: {str(e)}",
                    error_type=type(e).__name__,
                    position=self.positions[0].tolist(),
                    dt=dt
                )
            raise RuntimeError(f"ReactiveAgent step failed: {str(e)}") from e
    
    def sample_odor(self, env_array: np.ndarray) -> float:
        """
        Sample odor concentration at current position (NavigatorProtocol compliance).
        
        Delegates to underlying controller for consistency with protocol.
        
        Args:
            env_array: Environment array containing odor data
            
        Returns:
            float: Odor concentration at agent's current position
        """
        return self._controller.sample_odor(env_array)
    
    def sample_multiple_sensors(
        self,
        env_array: np.ndarray,
        sensor_distance: float = 5.0,
        sensor_angle: float = 45.0,
        num_sensors: int = 2,
        layout_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Sample odor at multiple sensor positions (NavigatorProtocol compliance).
        
        Delegates to underlying controller for consistency with protocol.
        
        Args:
            env_array: Environment array
            sensor_distance: Distance from agent to each sensor
            sensor_angle: Angular separation between sensors in degrees
            num_sensors: Number of sensors per agent
            layout_name: Predefined sensor layout name
            
        Returns:
            np.ndarray: Array of sensor readings with shape (num_sensors,)
        """
        return self._controller.sample_multiple_sensors(
            env_array, sensor_distance, sensor_angle, num_sensors, layout_name
        )
    
    # Extensibility hooks (NavigatorProtocol compliance)
    
    def compute_additional_obs(self, base_obs: dict) -> dict:
        """
        Compute additional observations for custom environment extensions.
        
        For reactive agents, this can include gradient-specific information
        while maintaining memoryless behavior.
        
        Args:
            base_obs: Base observation dict containing standard navigation data
            
        Returns:
            dict: Additional observation components for reactive navigation
        """
        additional_obs = {}
        
        # Add reactive agent specific observations
        additional_obs.update({
            'agent_type': 'reactive',
            'step_size': self._step_size,
            'turning_rate': self._turning_rate,
            'gradient_threshold': self._gradient_threshold,
            'is_memory_based': False,  # Explicitly indicate memory-less design
            'sensor_type': type(self._gradient_sensor).__name__
        })
        
        # Include performance metrics if monitoring is enabled
        if self._performance_monitoring and self._performance_metrics['total_steps'] > 0:
            additional_obs.update({
                'total_steps': self._performance_metrics['total_steps'],
                'average_gradient_magnitude': self._performance_metrics['average_gradient_magnitude'],
                'total_distance_traveled': self._performance_metrics['total_distance_traveled']
            })
        
        return additional_obs
    
    def compute_extra_reward(self, base_reward: float, info: dict) -> float:
        """
        Compute additional reward components for reactive navigation strategies.
        
        Args:
            base_reward: Base reward computed by environment
            info: Environment info dict containing episode state
            
        Returns:
            float: Additional reward component for reactive behavior
        """
        extra_reward = 0.0
        
        # Provide small bonus for movement decisions to encourage exploration
        if (self._performance_monitoring and 
            self._performance_metrics['total_steps'] > 0 and
            self._performance_metrics['movement_decisions'] > 0):
            
            movement_ratio = (
                self._performance_metrics['movement_decisions'] / 
                self._performance_metrics['total_steps']
            )
            
            # Small exploration bonus for active movement
            if movement_ratio > 0.1:  # At least 10% movement decisions
                extra_reward += 0.01 * movement_ratio
        
        return extra_reward
    
    def on_episode_end(self, final_info: dict) -> None:
        """
        Handle episode completion events for logging and cleanup.
        
        Args:
            final_info: Final environment info dict containing episode summary
        """
        if not self._logger:
            return
        
        episode_length = final_info.get('episode_length', 0)
        success = final_info.get('success', False)
        
        # Calculate reactive agent specific metrics
        agent_metrics = {}
        if self._performance_monitoring and self._performance_metrics['total_steps'] > 0:
            total_steps = self._performance_metrics['total_steps']
            agent_metrics.update({
                'movement_ratio': self._performance_metrics['movement_decisions'] / total_steps,
                'avg_gradient_magnitude': self._performance_metrics['average_gradient_magnitude'],
                'total_distance_traveled': self._performance_metrics['total_distance_traveled'],
                'avg_distance_per_step': self._performance_metrics['total_distance_traveled'] / total_steps
            })
            
            if self._performance_metrics['step_times']:
                step_times = np.array(self._performance_metrics['step_times'])
                agent_metrics.update({
                    'avg_step_time_ms': float(np.mean(step_times)),
                    'max_step_time_ms': float(np.max(step_times)),
                    'performance_violations': int(np.sum(step_times > 1.0))
                })
            
            if self._performance_metrics['gradient_computation_times']:
                grad_times = np.array(self._performance_metrics['gradient_computation_times'])
                agent_metrics.update({
                    'avg_gradient_computation_ms': float(np.mean(grad_times)),
                    'max_gradient_computation_ms': float(np.max(grad_times))
                })
        
        self._logger.info(
            "ReactiveAgent episode completed",
            episode_length=episode_length,
            success=success,
            agent_type="reactive",
            memoryless_design=True,
            **agent_metrics
        )
    
    # Performance monitoring and utility methods
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for reactive agent monitoring.
        
        Returns:
            Dict[str, Any]: Performance metrics including timing, navigation, and behavior data
        """
        if not self._performance_monitoring:
            return {
                'performance_monitoring_enabled': False,
                'agent_type': 'reactive',
                'total_steps': self._performance_metrics['total_steps']
            }
        
        metrics = {
            'agent_type': 'reactive',
            'performance_monitoring_enabled': True,
            'total_steps': self._performance_metrics['total_steps'],
            'gradient_computations': self._performance_metrics['gradient_computations'],
            'movement_decisions': self._performance_metrics['movement_decisions'],
            'stationary_decisions': self._performance_metrics['stationary_decisions'],
            'total_distance_traveled': self._performance_metrics['total_distance_traveled'],
            'average_gradient_magnitude': self._performance_metrics['average_gradient_magnitude'],
            
            # Configuration parameters
            'step_size': self._step_size,
            'turning_rate': self._turning_rate,
            'gradient_threshold': self._gradient_threshold,
            'sensor_type': type(self._gradient_sensor).__name__
        }
        
        # Add timing statistics
        if self._performance_metrics['step_times']:
            step_times = np.array(self._performance_metrics['step_times'])
            metrics.update({
                'avg_step_time_ms': float(np.mean(step_times)),
                'min_step_time_ms': float(np.min(step_times)),
                'max_step_time_ms': float(np.max(step_times)),
                'std_step_time_ms': float(np.std(step_times)),
                'performance_violations': int(np.sum(step_times > 1.0))
            })
        
        if self._performance_metrics['gradient_computation_times']:
            grad_times = np.array(self._performance_metrics['gradient_computation_times'])
            metrics.update({
                'avg_gradient_computation_ms': float(np.mean(grad_times)),
                'min_gradient_computation_ms': float(np.min(grad_times)),
                'max_gradient_computation_ms': float(np.max(grad_times))
            })
        
        # Add behavioral statistics
        if self._performance_metrics['total_steps'] > 0:
            metrics.update({
                'movement_ratio': self._performance_metrics['movement_decisions'] / self._performance_metrics['total_steps'],
                'avg_distance_per_step': self._performance_metrics['total_distance_traveled'] / self._performance_metrics['total_steps']
            })
        
        return metrics
    
    def configure(self, **kwargs: Any) -> None:
        """
        Update reactive agent configuration parameters at runtime.
        
        Args:
            **kwargs: Configuration parameters to update including:
                - step_size: Gradient step size multiplier
                - turning_rate: Maximum turning rate in degrees/second
                - gradient_threshold: Minimum gradient magnitude for movement
                - Any NavigatorProtocol configuration parameters
        """
        # Handle reactive agent specific parameters
        if 'step_size' in kwargs:
            self.step_size = kwargs.pop('step_size')
        if 'turning_rate' in kwargs:
            self.turning_rate = kwargs.pop('turning_rate')
        if 'gradient_threshold' in kwargs:
            self.gradient_threshold = kwargs.pop('gradient_threshold')
        
        # Handle sensor configuration
        if 'sensor_config' in kwargs:
            sensor_config = kwargs.pop('sensor_config')
            try:
                validate_sensor_config(sensor_config)
                self._gradient_sensor = create_sensor_from_config(sensor_config)
                
                if self._logger:
                    self._logger.info(
                        "Updated gradient sensor configuration",
                        new_sensor_type=type(self._gradient_sensor).__name__,
                        config=sensor_config
                    )
            except Exception as e:
                if self._logger:
                    self._logger.error(f"Failed to update sensor configuration: {e}")
                raise ValueError(f"Invalid sensor configuration: {e}")
        
        # Delegate remaining parameters to underlying controller
        if kwargs:
            self._controller.reset(**kwargs)
        
        if self._logger:
            self._logger.debug(
                "ReactiveAgent configuration updated",
                step_size=self._step_size,
                turning_rate=self._turning_rate,
                gradient_threshold=self._gradient_threshold
            )
    
    @classmethod
    def from_config(cls, config: Union[ReactiveAgentConfig, Dict[str, Any], DictConfig]) -> 'ReactiveAgent':
        """
        Create ReactiveAgent instance from configuration object.
        
        Args:
            config: Configuration object (ReactiveAgentConfig, dict, or DictConfig)
            
        Returns:
            ReactiveAgent: Configured reactive agent instance
            
        Examples:
            From ReactiveAgentConfig:
            >>> config = ReactiveAgentConfig(position=(10, 20), step_size=0.8)
            >>> agent = ReactiveAgent.from_config(config)
            
            From dictionary:
            >>> config = {'position': (0, 0), 'max_speed': 2.0, 'step_size': 1.0}
            >>> agent = ReactiveAgent.from_config(config)
        """
        # Handle different configuration types
        if isinstance(config, ReactiveAgentConfig):
            config_dict = {
                'position': config.position,
                'orientation': config.orientation,
                'speed': config.speed,
                'max_speed': config.max_speed,
                'angular_velocity': config.angular_velocity,
                'step_size': config.step_size,
                'turning_rate': config.turning_rate,
                'gradient_threshold': config.gradient_threshold,
                'enable_logging': config.enable_logging,
                'performance_monitoring': config.performance_monitoring
            }
            
            # Create gradient sensor from sensor configuration
            if config.sensor:
                sensor = create_sensor_from_config(config.sensor)
                config_dict['gradient_sensor'] = sensor
                
        elif isinstance(config, DictConfig) and HYDRA_AVAILABLE:
            config_dict = OmegaConf.to_container(config, resolve=True)
            
            # Handle sensor configuration
            if 'sensor' in config_dict:
                sensor = create_sensor_from_config(config_dict.pop('sensor'))
                config_dict['gradient_sensor'] = sensor
        
        elif isinstance(config, dict):
            config_dict = config.copy()
            
            # Handle sensor configuration
            if 'sensor' in config_dict:
                sensor = create_sensor_from_config(config_dict.pop('sensor'))
                config_dict['gradient_sensor'] = sensor
        
        else:
            raise TypeError(f"Unsupported configuration type: {type(config)}")
        
        return cls(**config_dict)


# Factory functions for configuration-driven instantiation

def create_reactive_agent_from_config(config: Union[Dict[str, Any], DictConfig]) -> ReactiveAgent:
    """
    Factory function to create ReactiveAgent from configuration dictionary.
    
    Args:
        config: Configuration dictionary containing reactive agent parameters
        
    Returns:
        ReactiveAgent: Configured reactive agent instance
        
    Examples:
        Basic configuration:
        >>> config = {
        ...     'position': (10.0, 20.0),
        ...     'max_speed': 2.0,
        ...     'step_size': 0.8,
        ...     'turning_rate': 45.0,
        ...     'sensor': {
        ...         'type': 'GradientSensor',
        ...         'spatial_resolution': (0.5, 0.5)
        ...     }
        ... }
        >>> agent = create_reactive_agent_from_config(config)
    """
    return ReactiveAgent.from_config(config)


def create_reactive_agent_with_sensor(
    sensor_config: Dict[str, Any],
    agent_config: Optional[Dict[str, Any]] = None
) -> ReactiveAgent:
    """
    Factory function to create ReactiveAgent with specific sensor configuration.
    
    Args:
        sensor_config: Configuration for the gradient sensor
        agent_config: Optional agent-specific configuration parameters
        
    Returns:
        ReactiveAgent: Reactive agent with configured sensor
        
    Examples:
        High-resolution gradient sensor:
        >>> sensor_config = {
        ...     'type': 'GradientSensor',
        ...     'spatial_resolution': (0.1, 0.1),
        ...     'method': 'central',
        ...     'order': 2
        ... }
        >>> agent_config = {'step_size': 1.2, 'turning_rate': 60.0}
        >>> agent = create_reactive_agent_with_sensor(sensor_config, agent_config)
    """
    gradient_sensor = create_sensor_from_config(sensor_config)
    
    # Merge agent configuration with sensor
    final_config = agent_config.copy() if agent_config else {}
    final_config['gradient_sensor'] = gradient_sensor
    
    return ReactiveAgent(**final_config)


# CLI and demonstration functionality

def demonstrate_reactive_navigation(
    environment_size: Tuple[int, int] = (100, 100),
    num_steps: int = 500,
    visualization: bool = False
) -> Dict[str, Any]:
    """
    Demonstrate reactive agent navigation in a simple test environment.
    
    Args:
        environment_size: Environment dimensions (width, height)
        num_steps: Number of simulation steps to run
        visualization: Enable basic visualization (requires matplotlib)
        
    Returns:
        Dict[str, Any]: Demonstration results including trajectory and metrics
    """
    width, height = environment_size
    
    # Create simple test environment with gradient
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    X, Y = np.meshgrid(x, y)
    
    # Simple Gaussian concentration field
    source_x, source_y = width * 0.8, height * 0.5
    env_array = np.exp(-((X - source_x)**2 + (Y - source_y)**2) / (2 * 20**2))
    
    # Create reactive agent
    agent_config = ReactiveAgentConfig(
        position=(width * 0.2, height * 0.5),
        max_speed=2.0,
        step_size=1.0,
        turning_rate=45.0,
        gradient_threshold=1e-6,
        performance_monitoring=True,
        sensor={
            'type': 'GradientSensor',
            'spatial_resolution': (1.0, 1.0),
            'method': 'central'
        }
    )
    
    agent = ReactiveAgent.from_config(agent_config)
    
    # Record trajectory
    trajectory = []
    concentrations = []
    
    if LOGURU_AVAILABLE:
        logger.info(
            "Starting reactive navigation demonstration",
            environment_size=environment_size,
            num_steps=num_steps,
            agent_config=agent_config.__dict__
        )
    
    # Simulation loop
    for step in range(num_steps):
        # Record current state
        position = agent.positions[0]
        concentration = agent.sample_odor(env_array)
        
        trajectory.append(position.copy())
        concentrations.append(concentration)
        
        # Execute agent step
        agent.step(env_array, dt=1.0)
        
        # Check if reached high concentration area (success condition)
        if concentration > 0.8:
            if LOGURU_AVAILABLE:
                logger.success(
                    f"Agent reached high concentration area at step {step}",
                    position=position.tolist(),
                    concentration=concentration
                )
            break
    
    # Collect results
    trajectory = np.array(trajectory)
    concentrations = np.array(concentrations)
    
    results = {
        'trajectory': trajectory,
        'concentrations': concentrations,
        'final_position': trajectory[-1] if len(trajectory) > 0 else np.array([0.0, 0.0]),
        'final_concentration': concentrations[-1] if len(concentrations) > 0 else 0.0,
        'steps_taken': len(trajectory),
        'success': concentrations[-1] > 0.8 if len(concentrations) > 0 else False,
        'agent_metrics': agent.get_performance_metrics(),
        'environment_size': environment_size
    }
    
    # Add trajectory analysis
    if len(trajectory) > 1:
        distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        results.update({
            'total_distance': float(np.sum(distances)),
            'average_speed': float(np.mean(distances)),
            'max_concentration_reached': float(np.max(concentrations)),
            'concentration_improvement': float(concentrations[-1] - concentrations[0])
        })
    
    # Visualization if requested
    if visualization:
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 5))
            
            # Plot environment and trajectory
            plt.subplot(1, 2, 1)
            plt.contourf(X, Y, env_array, levels=20, alpha=0.7, cmap='viridis')
            plt.colorbar(label='Concentration')
            if len(trajectory) > 0:
                plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Trajectory')
                plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, label='Start')
                plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=8, label='End')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title('Reactive Agent Navigation')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot concentration over time
            plt.subplot(1, 2, 2)
            if len(concentrations) > 0:
                plt.plot(concentrations, 'b-', linewidth=2)
                plt.axhline(y=0.8, color='r', linestyle='--', label='Success threshold')
            plt.xlabel('Time Step')
            plt.ylabel('Concentration')
            plt.title('Concentration Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            warnings.warn("Matplotlib not available for visualization", UserWarning)
    
    if LOGURU_AVAILABLE:
        logger.info(
            "Reactive navigation demonstration completed",
            steps_taken=results['steps_taken'],
            success=results['success'],
            final_concentration=results['final_concentration'],
            total_distance=results.get('total_distance', 0)
        )
    
    return results


@hydra.main(version_base=None, config_path="../../conf", config_name="reactive_agent")
def main(cfg: DictConfig) -> None:
    """
    Main CLI entry point for reactive agent demonstration.
    
    Args:
        cfg: Hydra configuration object
    """
    if not HYDRA_AVAILABLE:
        print("Hydra not available. Using default configuration.")
        cfg = DictConfig({
            'agent': {
                'position': (20.0, 50.0),
                'max_speed': 2.0,
                'step_size': 1.0,
                'turning_rate': 45.0,
                'gradient_threshold': 1e-6,
                'performance_monitoring': True,
                'sensor': {
                    'type': 'GradientSensor',
                    'spatial_resolution': [0.5, 0.5],
                    'method': 'central'
                }
            },
            'simulation': {
                'environment_size': [100, 100],
                'num_steps': 500,
                'visualization': True
            }
        })
    
    print("ReactiveAgent CLI Demonstration")
    print("=" * 50)
    
    # Create agent from configuration
    agent_config = cfg.get('agent', {})
    print(f"Agent configuration: {OmegaConf.to_yaml(agent_config) if HYDRA_AVAILABLE else agent_config}")
    
    try:
        agent = create_reactive_agent_from_config(agent_config)
        print(f"✓ Created ReactiveAgent with {type(agent._gradient_sensor).__name__}")
    except Exception as e:
        print(f"✗ Failed to create ReactiveAgent: {e}")
        return
    
    # Run demonstration
    sim_config = cfg.get('simulation', {})
    environment_size = sim_config.get('environment_size', [100, 100])
    num_steps = sim_config.get('num_steps', 500)
    visualization = sim_config.get('visualization', True)
    
    print(f"\nRunning demonstration:")
    print(f"  Environment size: {environment_size}")
    print(f"  Number of steps: {num_steps}")
    print(f"  Visualization: {visualization}")
    
    try:
        results = demonstrate_reactive_navigation(
            environment_size=tuple(environment_size),
            num_steps=num_steps,
            visualization=visualization
        )
        
        print(f"\nResults:")
        print(f"  Steps taken: {results['steps_taken']}")
        print(f"  Success: {results['success']}")
        print(f"  Final concentration: {results['final_concentration']:.6f}")
        print(f"  Total distance: {results.get('total_distance', 0):.2f}")
        
        # Display agent metrics
        metrics = results['agent_metrics']
        if metrics.get('performance_monitoring_enabled', False):
            print(f"\nAgent Performance Metrics:")
            print(f"  Average step time: {metrics.get('avg_step_time_ms', 0):.3f} ms")
            print(f"  Movement ratio: {metrics.get('movement_ratio', 0):.3f}")
            print(f"  Average gradient magnitude: {metrics.get('average_gradient_magnitude', 0):.6f}")
            print(f"  Performance violations: {metrics.get('performance_violations', 0)}")
        
    except Exception as e:
        print(f"✗ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if HYDRA_AVAILABLE:
        main()
    else:
        # Fallback for environments without Hydra
        print("Running ReactiveAgent demonstration without Hydra...")
        results = demonstrate_reactive_navigation(visualization=True)
        print(f"Demonstration completed. Success: {results['success']}")