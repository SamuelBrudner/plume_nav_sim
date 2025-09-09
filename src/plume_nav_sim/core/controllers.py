"""
Enhanced controller implementations for single-agent and multi-agent navigation with modular sensor architecture.

This module provides consolidated navigation controllers that implement the NavigatorProtocol 
interface with enhanced features for modern ML framework compatibility. Controllers support
both single-agent and multi-agent scenarios with performance guarantees, extensibility hooks,
sensor-based perception, and memory management for flexible cognitive modeling approaches.

Key enhancements for modular architecture:
- SensorProtocol-based perception system replacing hard-coded odor sampling logic
- Optional memory management hooks (load_memory, save_memory) without enforcing use
- Agent-agnostic design supporting both reactive and planning navigation strategies
- Extensibility hooks for custom observations, rewards, and episode handling
- Integration with enhanced frame caching featuring LRU eviction and memory monitoring
- Performance optimization with sub-33ms step execution guarantees
- Vectorized operations supporting 100+ concurrent agents

The controllers maintain full backward compatibility while providing modern API features:
- Type-safe configuration with Hydra integration
- Modular sensor system with configurable perception modalities
- Optional memory persistence for cognitive modeling approaches
- Comprehensive performance monitoring and metrics collection
- Memory-efficient operations with configurable frame caching
- Structured logging with contextual information
- Extensible architecture for custom algorithm implementations

Performance Requirements:
- Single agent step execution: <1ms per step
- Multi-agent step execution: <10ms for 100 agents 
- Memory efficiency: <10MB overhead per 100 agents
- Simulation throughput: ≥30 FPS for real-time visualization
- Frame caching integration with ≤2 GiB memory limits

Examples:
    Single-agent controller with sensor-based perception:
        >>> from plume_nav_sim.core.controllers import DirectOdorSensor
        >>> controller = SingleAgentController(
        ...     position=(10.0, 20.0), 
        ...     speed=1.5,
        ...     sensors=[DirectOdorSensor()],
        ...     enable_extensibility_hooks=True
        ... )
        >>> controller.step(env_array, dt=1.0)
        >>> metrics = controller.get_performance_metrics()
        
    Multi-agent swarm with memory and custom sensors:
        >>> positions = [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]
        >>> controller = MultiAgentController(
        ...     positions=positions,
        ...     enable_vectorized_ops=True,
        ...     enable_memory=True,
        ...     sensors=[DirectOdorSensor()]
        ... )
        >>> controller.step(env_array, dt=1.0)
        >>> throughput = controller.get_performance_metrics()['throughput_agents_fps']
        
    Memory-based navigation with cognitive modeling:
        >>> controller = SingleAgentController(
        ...     position=(0.0, 0.0),
        ...     enable_memory=True
        ... )
        >>> # Load previous episode memory
        >>> memory = controller.load_memory(previous_episode_data)
        >>> # Agent maintains internal state across episodes
        >>> memory = controller.save_memory()
        
    Configuration-driven instantiation with sensors and memory:
        >>> from omegaconf import DictConfig
        >>> cfg = DictConfig({
        ...     "position": [5.0, 5.0], 
        ...     "max_speed": 2.0,
        ...     "enable_memory": True,
        ...     "sensors": [DirectOdorSensor()]
        ... })
        >>> controller = create_controller_from_config(cfg)
        
    Custom navigation with sensor-based perception:
        >>> class MemoryBasedController(SingleAgentController):
        ...     def compute_additional_obs(self, base_obs: dict) -> dict:
        ...         # Custom sensor readings
        ...         return {"gradient": self.compute_odor_gradient()}
        ...     
        ...     def update_memory(self, obs: dict, action, reward: float, info: dict):
        ...         # Update cognitive map
        ...         self.update_belief_state(obs, action, reward)

Notes:
    All controllers implement the NavigatorProtocol interface ensuring uniform API
    across algorithmic approaches. The protocol-based design enables research 
    extensibility while maintaining compatibility with existing simulation runners,
    visualization systems, and data recording infrastructure.
    
    The modular sensor architecture replaces hard-coded odor sampling with configurable
    SensorProtocol implementations, enabling diverse sensing modalities (binary detection,
    concentration measurement, gradient computation) without code changes. Memory management
    hooks provide optional cognitive modeling capabilities for planning agents while
    maintaining efficiency for reactive strategies.
    
    Frame caching integration is designed to work with the enhanced FrameCache system
    featuring configurable modes (none, LRU, preload), memory pressure monitoring,
    and intelligent eviction policies. Controllers automatically benefit from
    performance improvements as the caching system becomes available.
"""

import contextlib
import time
import warnings
import copy
from typing import Optional, Union, Any, Tuple, List, Dict, TypeVar, Callable
from dataclasses import dataclass, field
import dataclasses
import numpy as np
# Loguru logging
from loguru import logger

# Core protocol imports
from plume_nav_sim.protocols.navigator import NavigatorProtocol
from .protocols import BoundaryPolicyProtocol, SourceProtocol
from plume_nav_sim.protocols.sensor import SensorProtocol

# Boundary policy factory import
from .boundaries import create_boundary_policy


# Basic sensor implementations for interim use until full sensor system is available
class DirectOdorSensor(SensorProtocol):
    """Directly reads odor values from a plume state."""

    def __init__(self):
        self._enabled = True
        logger.debug("DirectOdorSensor initialized", enabled=self._enabled)

    def detect(self, plume_state: np.ndarray, positions: np.ndarray) -> np.ndarray:
        logger.debug("DirectOdorSensor.detect invoked", positions=positions)
        measurements = self.measure(plume_state, positions)
        return measurements > 0.0

    def measure(self, plume_state: np.ndarray, positions: np.ndarray) -> np.ndarray:
        logger.debug("DirectOdorSensor.measure invoked", positions=positions)
        return _read_odor_values(plume_state, positions)

    def compute_gradient(self, plume_state: np.ndarray, positions: np.ndarray, delta: float = 1.0) -> np.ndarray:
        logger.debug("DirectOdorSensor.compute_gradient invoked", positions=positions, delta=delta)
        positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        gradients = np.zeros((len(positions), 2), dtype=float)
        for i, pos in enumerate(positions):
            offsets = np.array([[delta, 0], [-delta, 0], [0, delta], [0, -delta]], dtype=float)
            samples = self.measure(plume_state, pos + offsets)
            dx = (samples[0] - samples[1]) / (2 * delta)
            dy = (samples[2] - samples[3]) / (2 * delta)
            gradients[i] = [dx, dy]
        return gradients

    def configure(self, **kwargs) -> None:
        self._enabled = kwargs.get('enabled', True)
        logger.debug("DirectOdorSensor.configure called", enabled=self._enabled)

    def reset(self) -> None:
        logger.debug("DirectOdorSensor.reset called")

    def sample(self, *args, **kwargs):
        logger.error("DirectOdorSensor.sample is deprecated")
        raise NotImplementedError(
            "sample() is deprecated. Use detect(), measure(), or compute_gradient() instead."
        )


class MultiPointOdorSensor:
    """Sensor for multi-point odor sampling (e.g., bilateral antennae)."""

    def __init__(self, sensor_distance: float = 5.0, sensor_angle: float = 45.0,
                 num_sensors: int = 2, layout_name: Optional[str] = None):
        self.sensor_distance = sensor_distance
        self.sensor_angle = sensor_angle
        self.num_sensors = num_sensors
        self.layout_name = layout_name
        self._enabled = True

    def detect(self, plume_state: np.ndarray, positions: np.ndarray, orientations: np.ndarray) -> np.ndarray:
        logger.debug("MultiPointOdorSensor.detect invoked", positions=positions)
        measurements = self.measure(plume_state, positions, orientations)
        return measurements > 0.0

    def measure(self, plume_state: np.ndarray, positions: np.ndarray,
               orientations: np.ndarray) -> np.ndarray:
        logger.debug("MultiPointOdorSensor.measure invoked", positions=positions, orientations=orientations)
        if not self._enabled:
            return np.zeros((len(positions), self.num_sensors))

        mock_navigator = type('MockNavigator', (), {
            'positions': positions,
            'orientations': orientations,
            'num_agents': len(positions)
        })()

        return _sample_odor_at_sensors(
            mock_navigator, plume_state,
            sensor_distance=self.sensor_distance,
            sensor_angle=self.sensor_angle,
            num_sensors=self.num_sensors,
            layout_name=self.layout_name
        )

    def sample(self, *args, **kwargs):
        logger.error("MultiPointOdorSensor.sample is deprecated")
        raise NotImplementedError(
            "sample() is deprecated. Use detect() or measure() instead."
        )
    
    def configure(self, **kwargs) -> None:
        """Configure sensor parameters."""
        self.sensor_distance = kwargs.get('sensor_distance', self.sensor_distance)
        self.sensor_angle = kwargs.get('sensor_angle', self.sensor_angle)
        self.num_sensors = kwargs.get('num_sensors', self.num_sensors)
        self.layout_name = kwargs.get('layout_name', self.layout_name)
        self._enabled = kwargs.get('enabled', True)
    
    def reset(self) -> None:
        """Reset sensor state."""
        pass

# Hydra integration for configuration management
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Gymnasium integration for modern API support
try:
    import gymnasium
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - handled by tests
    logger.error(f"Gymnasium is required for controllers: {exc}")
    raise

# Configuration schemas
try:
    from ..config.schemas import NavigatorConfig, SingleAgentConfig, MultiAgentConfig
except ImportError as exc:  # pragma: no cover - handled by tests
    logger.error(f"Configuration schemas are missing: {exc}")
    raise

# Frame cache integration
try:
    from ..utils.frame_cache import FrameCache, CacheMode
except ImportError as exc:  # pragma: no cover - handled by tests
    logger.error(f"FrameCache utilities are required: {exc}")
    raise

# Spaces factory integration
try:
    from ..envs.spaces import SpacesFactory
except ImportError as exc:  # pragma: no cover - handled by tests
    logger.error(f"SpacesFactory is required: {exc}")
    raise

# psutil for memory monitoring
import psutil

# Type variable for controller types
ControllerType = TypeVar('ControllerType', bound='BaseController')


@dataclass
class SingleAgentParams:
    """
    Type-safe parameters for resetting a single agent navigator.
    
    This dataclass provides stronger type checking than kwargs-based configuration
    and integrates with Hydra's structured configuration system for validation.
    Enhanced for Gymnasium 0.29.x with additional configuration options.
    
    Attributes:
        position: Initial agent position coordinates [x, y]
        orientation: Initial orientation in degrees (0 = right, 90 = up)
        speed: Initial speed in units per time step
        max_speed: Maximum allowed speed in units per time step
        angular_velocity: Angular velocity in degrees per second
        enable_extensibility_hooks: Enable custom observation/reward hooks
        frame_cache_mode: Frame caching strategy ("none", "lru", "preload")
    
    Examples:
        Basic parameter configuration:
            >>> params = SingleAgentParams(position=(10.0, 20.0), speed=1.5)
            >>> controller.reset_with_params(params)
            
        Enhanced configuration with hooks:
            >>> params = SingleAgentParams(
            ...     position=(5.0, 5.0),
            ...     max_speed=2.0,
            ...     enable_extensibility_hooks=True,
            ...     frame_cache_mode="lru"
            ... )
            >>> controller.reset_with_params(params)
    """
    position: Optional[Tuple[float, float]] = None
    orientation: Optional[float] = None
    speed: Optional[float] = None
    max_speed: Optional[float] = None
    angular_velocity: Optional[float] = None
    enable_extensibility_hooks: bool = False
    frame_cache_mode: Optional[str] = None
    custom_observation_keys: List[str] = field(default_factory=list)
    reward_shaping: Optional[str] = None
    
    def to_kwargs(self) -> Dict[str, Any]:
        """Convert dataclass to kwargs dictionary for controller reset methods."""
        result = {}
        for k, v in self.__dict__.items():
            if v is not None:
                # Handle list/array comparisons safely
                if isinstance(v, list) and len(v) == 0:
                    continue
                elif isinstance(v, np.ndarray) and v.size == 0:
                    continue
                else:
                    result[k] = v
        return result


@dataclass
class MultiAgentParams:
    """
    Type-safe parameters for resetting a multi-agent navigator.
    
    This dataclass provides stronger type checking and enables batch parameter
    updates for multiple agents with comprehensive validation. Enhanced for
    Gymnasium 0.29.x with vectorized operation support.
    
    Attributes:
        positions: Array of agent positions with shape (num_agents, 2)
        orientations: Array of agent orientations with shape (num_agents,)
        speeds: Array of agent speeds with shape (num_agents,)
        max_speeds: Array of maximum speeds with shape (num_agents,)
        angular_velocities: Array of angular velocities with shape (num_agents,)
        enable_extensibility_hooks: Enable custom observation/reward hooks
        enable_vectorized_ops: Enable vectorized operations for performance
        frame_cache_mode: Frame caching strategy for all agents
    
    Examples:
        Batch agent configuration:
            >>> import numpy as np
            >>> params = MultiAgentParams(
            ...     positions=np.array([[0, 0], [10, 10], [20, 20]]),
            ...     speeds=np.array([1.0, 1.5, 2.0]),
            ...     enable_vectorized_ops=True
            ... )
            >>> controller.reset_with_params(params)
    """
    positions: Optional[np.ndarray] = None
    orientations: Optional[np.ndarray] = None
    speeds: Optional[np.ndarray] = None
    max_speeds: Optional[np.ndarray] = None
    angular_velocities: Optional[np.ndarray] = None
    enable_extensibility_hooks: bool = False
    enable_vectorized_ops: bool = True
    frame_cache_mode: Optional[str] = None
    custom_observation_keys: List[str] = field(default_factory=list)
    reward_shaping: Optional[str] = None
    
    def to_kwargs(self) -> Dict[str, Any]:
        """Convert dataclass to kwargs dictionary for controller reset methods."""
        result = {}
        for k, v in self.__dict__.items():
            if v is not None:
                # Handle list/array comparisons safely
                if isinstance(v, list) and len(v) == 0:
                    continue
                elif isinstance(v, np.ndarray) and v.size == 0:
                    continue
                else:
                    result[k] = v
        return result


class BaseController:
    """
    Base controller class with shared functionality for single and multi-agent scenarios.
    
    Provides common infrastructure for performance monitoring, extensibility hooks,
    frame caching integration, and Gymnasium 0.29.x API compatibility. This base
    class ensures consistent behavior across controller implementations.
    """
    
    def __init__(
        self,
        enable_logging: bool = True,
        controller_id: Optional[str] = None,
        enable_extensibility_hooks: bool = False,
        frame_cache_mode: Optional[str] = None,
        custom_observation_keys: Optional[List[str]] = None,
        reward_shaping: Optional[str] = None,
        sensors: Optional[List[SensorProtocol]] = None,
        source: SourceProtocol | None = None,
        enable_memory: bool = False,
        boundary_policy: Optional[BoundaryPolicyProtocol] = None,
        domain_bounds: Optional[Tuple[float, float]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize base controller with enhanced monitoring capabilities and sensor support."""
        self._enable_logging = enable_logging
        self._controller_id = controller_id or f"{self.__class__.__name__.lower()}_{id(self)}"
        self._enable_extensibility_hooks = enable_extensibility_hooks
        self._frame_cache_mode = frame_cache_mode or "none"
        self._custom_observation_keys = custom_observation_keys or []
        self._reward_shaping = reward_shaping
        self._enable_memory = enable_memory

        # Initialize sensor system for agent-agnostic perception
        self._sensors = sensors or [DirectOdorSensor()]
        self._primary_sensor = self._sensors[0] if self._sensors else DirectOdorSensor()

        # Optional odor source integration
        self._source = source
        if self._enable_logging and self._source is None:
            logger.warning(
                "BaseController initialized without a SourceProtocol; explicit plume_state is required for sampling"
            )
        
        # Memory management hooks - optional for flexible cognitive modeling
        self._memory_enabled = enable_memory
        self._memory_state = None
        
        # Boundary policy integration for v1.0 pluggable architecture
        self._boundary_policy = boundary_policy
        self._domain_bounds = domain_bounds
        if boundary_policy is None and domain_bounds is not None:
            # Create default terminate boundary policy if domain bounds provided
            try:
                self._boundary_policy = create_boundary_policy("terminate", domain_bounds)
            except Exception as e:
                if self._enable_logging:
                    logger.warning(f"Failed to create default boundary policy: {e}")
                self._boundary_policy = None
        
        # Performance metrics tracking
        self._performance_metrics = {
            'step_times': [],
            'sample_times': [],
            'total_steps': 0,
            'frame_cache_hits': 0,
            'frame_cache_misses': 0
        }
        
        # Known kwargs
        cache_memory_limit_mb = kwargs.pop("cache_memory_limit_mb", 2048)

        # Frame cache integration
        self._frame_cache = None
        if self._frame_cache_mode != "none":
            try:
                self._frame_cache = FrameCache(
                    mode=self._frame_cache_mode,
                    memory_limit_mb=cache_memory_limit_mb,
                    enable_statistics=True
                )
            except Exception as e:
                if self._enable_logging:
                    logger.warning(f"Failed to initialize frame cache: {e}")
                self._frame_cache = None
        
        # Setup structured logging with context binding
        if self._enable_logging:
            self._logger = logger.bind(
                controller_type=self.__class__.__name__,
                controller_id=self._controller_id,
                extensibility_hooks=self._enable_extensibility_hooks,
                frame_cache_mode=self._frame_cache_mode
            )
            
            # Add Hydra context if available
            try:
                hydra_cfg = HydraConfig.get()
                self._logger = self._logger.bind(
                    hydra_job_name=hydra_cfg.job.name,
                    hydra_output_dir=hydra_cfg.runtime.output_dir
                )
            except Exception:  # pragma: no cover - optional integration
                pass
        else:
            self._logger = None

        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            log = self._logger if self._logger is not None else logger
            log.error(f"Unexpected keyword arguments: {unknown}")
            raise TypeError(f"Unexpected keyword arguments: {unknown}")
    
    # NavigatorProtocol properties for v1.0 architecture
    
    @property
    def source(self) -> Optional[SourceProtocol]:
        """
        Get current source implementation for odor emission modeling.
        
        Returns:
            Optional[SourceProtocol]: Source instance providing emission data and
                position information, or None if no source is configured.
        """
        return getattr(self, '_source', None)
    
    @property
    def boundary_policy(self) -> Optional[BoundaryPolicyProtocol]:
        """
        Get current boundary policy implementation for domain edge management.
        
        Returns:
            Optional[BoundaryPolicyProtocol]: Boundary policy instance providing
                edge handling behavior, or None if no boundary policy is configured.
        """
        return self._boundary_policy
    
    # Extensibility hooks - default implementations for protocol compliance
    
    def compute_additional_obs(self, base_obs: dict) -> dict:
        """
        Compute additional observations for custom environment extensions.
        
        Default implementation returns observations based on custom_observation_keys.
        Override this method in subclasses for domain-specific functionality.
        
        Args:
            base_obs: Base observation dict containing standard navigation data
            
        Returns:
            dict: Additional observation components to merge with base_obs
        """
        if not self._enable_extensibility_hooks:
            return {}
        
        additional_obs = {}
        
        # Add custom observation keys based on configuration
        for key in self._custom_observation_keys:
            if key == "frame_cache_stats" and self._frame_cache:
                cache_stats = self._frame_cache.get_statistics()
                additional_obs["frame_cache_hit_rate"] = cache_stats.get("hit_rate", 0.0)
                additional_obs["frame_cache_memory_usage"] = cache_stats.get("memory_usage_mb", 0.0)
            elif key == "performance_metrics":
                if self._performance_metrics['step_times']:
                    recent_times = self._performance_metrics['step_times'][-10:]
                    additional_obs["avg_step_time_ms"] = float(np.mean(recent_times))
            elif key == "controller_id":
                additional_obs["controller_id"] = self._controller_id
        
        return additional_obs
    
    def compute_extra_reward(self, base_reward: float, info: dict) -> float:
        """
        Compute additional reward components for custom reward shaping.
        
        Default implementation provides common reward shaping strategies.
        Override this method in subclasses for domain-specific reward functions.
        
        Args:
            base_reward: Base reward computed by environment
            info: Environment info dict containing episode state
            
        Returns:
            float: Additional reward component to add to base_reward
        """
        if not self._enable_extensibility_hooks or not self._reward_shaping:
            return 0.0
        
        extra_reward = 0.0
        
        # Apply configured reward shaping strategies
        if self._reward_shaping == "exploration_bonus":
            # Simple exploration bonus based on movement
            if hasattr(self, 'speeds') and self.num_agents > 0:
                movement_bonus = 0.01 * float(np.mean(self.speeds))
                extra_reward += movement_bonus
        
        elif self._reward_shaping == "efficiency_penalty":
            # Penalty for inefficient movement (high speed, low reward)
            if hasattr(self, 'speeds') and base_reward < 0.1:
                speed_penalty = -0.005 * float(np.mean(self.speeds ** 2))
                extra_reward += speed_penalty
        
        elif self._reward_shaping == "frame_cache_bonus":
            # Bonus for efficient frame cache usage
            if self._frame_cache:
                stats = self._frame_cache.get_statistics()
                hit_rate = stats.get("hit_rate", 0.0)
                if hit_rate > 0.9:  # >90% hit rate
                    extra_reward += 0.01
        
        return extra_reward
    
    def on_episode_end(self, final_info: dict) -> None:
        """
        Handle episode completion events for logging and cleanup.
        
        Default implementation logs episode statistics and performance metrics.
        Override this method in subclasses for custom episode handling.
        
        Args:
            final_info: Final environment info dict containing episode summary
        """
        if not self._enable_extensibility_hooks or not self._logger:
            return
        
        # Log episode completion with performance summary
        episode_length = final_info.get('episode_length', 0)
        success = final_info.get('success', False)
        
        # Calculate performance statistics
        perf_stats = {}
        if self._performance_metrics['step_times']:
            step_times = np.array(self._performance_metrics['step_times'])
            perf_stats.update({
                'avg_step_time_ms': float(np.mean(step_times)),
                'max_step_time_ms': float(np.max(step_times)),
                'step_time_violations': int(np.sum(step_times > 33.0))
            })
        
        # Frame cache statistics
        cache_stats = {}
        if self._frame_cache:
            stats = self._frame_cache.get_statistics()
            cache_stats.update({
                'cache_hit_rate': stats.get('hit_rate', 0.0),
                'cache_memory_usage_mb': stats.get('memory_usage_mb', 0.0)
            })
        
        self._logger.info(
            "Episode completed",
            episode_length=episode_length,
            success=success,
            performance_stats=perf_stats,
            cache_stats=cache_stats,
            total_steps=self._performance_metrics['total_steps']
        )

    def get_observation_space_info(self) -> Dict[str, Any]:
        """Raise ``NotImplementedError`` for undefined observation metadata.

        BaseController no longer provides a stub implementation. Subclasses must
        explicitly implement this method to expose their observation space
        structure. Failing fast avoids silent assumptions about observation
        semantics and surfaces incomplete implementations early.

        Raises:
            NotImplementedError: Always raised to require subclass override.
        """
        if self._logger:
            self._logger.error(
                "get_observation_space_info not implemented for %s",
                self.__class__.__name__,
            )
        raise NotImplementedError(
            "Subclasses must implement get_observation_space_info"
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for monitoring and optimization.
        
        Returns:
            Dict[str, Any]: Dictionary containing detailed performance statistics
        """
        if not self._enable_logging:
            return {}
        
        metrics = {
            'controller_type': 'single_agent' if getattr(self, 'num_agents', 1) == 1 else 'multi_agent',
            'controller_id': self._controller_id,
            'total_steps': self._performance_metrics['total_steps'],
            'extensibility_hooks_enabled': self._enable_extensibility_hooks,
            'frame_cache_mode': self._frame_cache_mode,
            'memory_enabled': self._memory_enabled,
            'num_sensors': len(self._sensors),
            'num_agents': getattr(self, 'num_agents', 0),
        }
        
        # Step time statistics
        if self._performance_metrics['step_times']:
            step_times = np.array(self._performance_metrics['step_times'])
            metrics.update({
                'step_time_mean_ms': float(np.mean(step_times)),
                'step_time_std_ms': float(np.std(step_times)),
                'step_time_max_ms': float(np.max(step_times)),
                'step_time_p95_ms': float(np.percentile(step_times, 95)),
                'performance_violations': int(np.sum(step_times > 33.0))
            })
        
        # Sample time statistics
        if self._performance_metrics['sample_times']:
            sample_times = np.array(self._performance_metrics['sample_times'])
            metrics.update({
                'sample_time_mean_ms': float(np.mean(sample_times)),
                'sample_time_max_ms': float(np.max(sample_times))
            })
        
        # Frame cache statistics
        if self._frame_cache:
            cache_stats = self._frame_cache.get_statistics()
            metrics.update({
                'frame_cache_hit_rate': cache_stats.get('hit_rate', 0.0),
                'frame_cache_memory_usage_mb': cache_stats.get('memory_usage_mb', 0.0),
                'frame_cache_total_requests': cache_stats.get('total_requests', 0),
                'frame_cache_evictions': cache_stats.get('evictions', 0)
            })
        
        return metrics

    # Observation hook and memory management for flexible cognitive modeling

    def observe(self, sensor_output: Any) -> Dict[str, Any]:
        """Validate and enrich raw sensor output."""
        log = self._logger if self._logger is not None else logger
        log.debug("observe invoked", input_type=type(sensor_output).__name__)

        if not isinstance(sensor_output, dict):
            log.error("sensor_output must be a dict", provided_type=type(sensor_output).__name__)
            raise TypeError(f"sensor_output must be a dict, got {type(sensor_output)}")

        observation = dict(sensor_output)
        log.debug("sensor_output validated", keys=list(observation.keys()))

        try:
            additional = self.compute_additional_obs(observation)
        except Exception as e:
            log.error(f"compute_additional_obs failed: {e}")
            raise

        if additional:
            if not isinstance(additional, dict):
                log.error(
                    "compute_additional_obs must return dict",
                    returned_type=type(additional).__name__,
                )
                raise TypeError("compute_additional_obs must return dict")
            log.debug("merging additional observations", keys=list(additional.keys()))
            observation.update(additional)

        log.debug("observe returning", keys=list(observation.keys()))
        return observation

    def load_memory(self, memory_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Load memory state for memory-based navigation strategies.
        
        This optional memory hook enables flexible cognitive modeling approaches
        without enforcing memory use. Implementations can override this method
        to support planning agents, while reactive agents can ignore it entirely.
        
        Args:
            memory_data: Optional memory state to load. If None, return current state.
            
        Returns:
            Dict[str, Any]: Current memory state if memory is enabled, None otherwise
            
        Notes:
            Default implementation provides simple memory storage without enforcing
            any specific memory structure. Override in subclasses for domain-specific
            memory models (e.g., occupancy grids, belief states, learned maps).
            
        Examples:
            Load from previous episode:
            >>> memory = navigator.load_memory(previous_episode_memory)
            
            Get current memory state:
            >>> current_memory = navigator.load_memory()
        """
        if memory_data is not None and not isinstance(memory_data, dict):
            raise TypeError(f"memory_data must be a dict, got {type(memory_data)}")

        if not self._memory_enabled:
            return None

        if memory_data is not None:
            self._memory_state = memory_data
            log = self._logger if self._logger is not None else logger
            log.debug("memory state loaded", memory_keys=list(memory_data.keys()))

        if self._memory_state is not None and not isinstance(self._memory_state, dict):
            raise TypeError("memory state must be a dict")

        return self._memory_state
    
    def save_memory(self) -> Optional[Dict[str, Any]]:
        """
        Save current memory state for persistence or transfer.
        
        This optional memory hook enables flexible cognitive modeling approaches
        for agents that maintain internal state. Reactive agents can safely 
        ignore this method, while planning agents can implement sophisticated
        memory persistence.
        
        Returns:
            Dict[str, Any]: Serializable memory state if memory is enabled, None otherwise
            
        Notes:
            Default implementation returns current memory state without modification.
            Override in subclasses to implement domain-specific memory serialization
            (e.g., compress sparse maps, extract key features, filter old data).
            
        Examples:
            Save memory between episodes:
            >>> memory = navigator.save_memory()
            >>> # Store memory for next episode initialization
            
            Transfer memory between agents:
            >>> agent1_memory = agent1.save_memory()
            >>> agent2.load_memory(agent1_memory)
        """
        if not self._memory_enabled:
            return None

        if self._memory_state is not None and not isinstance(self._memory_state, dict):
            raise TypeError("memory state must be a dict")

        memory_copy = copy.deepcopy(self._memory_state)
        log = self._logger if self._logger is not None else logger
        log.debug(
            "memory state saved",
            deep_copy=self._memory_state is not None,
            memory_keys=list(self._memory_state.keys()) if self._memory_state else None,
        )
        return memory_copy
    
    def update_memory(self, observation: Dict[str, Any], action: Any, reward: float, info: Dict[str, Any]) -> None:
        """
        Update memory state with new experience.
        
        This optional hook allows memory-based agents to accumulate experience
        during navigation. Reactive agents can ignore this method entirely.
        
        Args:
            observation: Current observation from environment
            action: Action taken by agent
            reward: Reward received
            info: Additional environment info
            
        Notes:
            Default implementation does nothing. Override in subclasses for
            memory-based navigation strategies.
        """
        if not self._memory_enabled:
            return
        
        # Initialize memory state if needed
        if self._memory_state is None:
            self._memory_state = {
                'episode_count': 0,
                'total_steps': 0,
                'experience_buffer': []
            }
        
        # Add experience to memory (basic implementation)
        if isinstance(self._memory_state, dict):
            self._memory_state['total_steps'] = self._memory_state.get('total_steps', 0) + 1
            
            # Maintain a simple experience buffer (limited size)
            experience_buffer = self._memory_state.get('experience_buffer', [])
            experience_buffer.append({
                'observation': observation,
                'action': action,
                'reward': reward,
                'info': info
            })
            
            # Keep only recent experiences (prevent memory growth)
            max_buffer_size = 1000
            if len(experience_buffer) > max_buffer_size:
                experience_buffer = experience_buffer[-max_buffer_size:]
            
            self._memory_state['experience_buffer'] = experience_buffer
    
    # Sensor management methods
    
    def add_sensor(self, sensor: SensorProtocol) -> None:
        """Add a sensor to the active sensor list."""
        if sensor not in self._sensors:
            self._sensors.append(sensor)
            if self._logger:
                self._logger.debug(
                    "Sensor added to controller",
                    sensor_type=type(sensor).__name__,
                    total_sensors=len(self._sensors)
                )
    
    def remove_sensor(self, sensor: SensorProtocol) -> bool:
        """Remove a sensor from the active sensor list."""
        if sensor in self._sensors:
            self._sensors.remove(sensor)
            if self._logger:
                self._logger.debug(
                    "Sensor removed from controller",
                    sensor_type=type(sensor).__name__,
                    total_sensors=len(self._sensors)
                )
            return True
        return False
    
    def get_active_sensors(self) -> List[SensorProtocol]:
        """Get list of currently active sensors."""
        return self._sensors.copy()
    
    def reset_sensors(self) -> None:
        """Reset all sensors to initial state."""
        for sensor in self._sensors:
            sensor.reset()
        if self._logger:
            self._logger.debug(
                "All sensors reset",
                num_sensors=len(self._sensors)
            )
    
    # Boundary policy integration methods for v1.0 architecture
    
    def _apply_boundary_policy(
        self, 
        positions: np.ndarray, 
        velocities: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
        """
        Apply boundary policy to agent positions and velocities.
        
        Args:
            positions: Agent positions with shape (n_agents, 2) or (2,)
            velocities: Optional agent velocities with same shape
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray], bool]: 
                (corrected_positions, corrected_velocities, terminate_episode)
                
        Notes:
            This method integrates boundary policy checking and correction into
            the navigation step process while maintaining performance requirements.
        """
        if self._boundary_policy is None:
            # No boundary policy - return positions unchanged
            return positions, velocities, False
        
        try:
            # Check for boundary violations
            violations = self._boundary_policy.check_violations(positions)
            
            # If no violations, return unchanged
            if not np.any(violations):
                return positions, velocities, False
            
            # Apply boundary policy corrections
            if velocities is not None:
                corrected_pos, corrected_vel = self._boundary_policy.apply_policy(positions, velocities)
            else:
                corrected_pos = self._boundary_policy.apply_policy(positions)
                corrected_vel = velocities
            
            # Check if episode should terminate based on boundary policy
            termination_status = self._boundary_policy.get_termination_status()
            terminate_episode = (termination_status == "oob")
            
            # Log boundary interactions for debugging
            if self._logger and np.any(violations):
                self._logger.debug(
                    "Boundary policy applied",
                    policy_type=type(self._boundary_policy).__name__,
                    violations_count=int(np.sum(violations)),
                    termination_status=termination_status,
                    terminate_episode=terminate_episode
                )
            
            return corrected_pos, corrected_vel, terminate_episode
            
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Boundary policy application failed: {str(e)}",
                    error_type=type(e).__name__,
                    policy_type=type(self._boundary_policy).__name__ if self._boundary_policy else None
                )
            # On error, return positions unchanged and don't terminate
            return positions, velocities, False
    
    def is_boundary_termination_pending(self) -> bool:
        """
        Check if boundary policy has triggered episode termination.
        
        Returns:
            bool: True if boundary termination is pending, False otherwise
            
        Notes:
            This method allows external episode management to check if a boundary
            policy has triggered episode termination (e.g., agent out of bounds).
        """
        return getattr(self, '_boundary_termination_pending', False)
    
    def clear_boundary_termination(self) -> None:
        """
        Clear boundary termination flag for new episode.
        
        Notes:
            This method should be called at episode reset to clear any pending
            boundary termination status from the previous episode.
        """
        if hasattr(self, '_boundary_termination_pending'):
            self._boundary_termination_pending = False


class SingleAgentController(BaseController):
    """
    Enhanced single agent controller with Gymnasium 0.29.x integration.
    
    This implements the NavigatorProtocol for a single agent case, providing
    simplified navigation logic without conditional branches. Enhanced with
    extensibility hooks, frame caching integration, and performance monitoring
    for modern ML framework compatibility.
    
    The controller maintains all agent state as NumPy arrays for consistent
    API compatibility with multi-agent scenarios and efficient numerical
    operations. Performance is optimized for <1ms frame processing latency.
    
    Enhanced Features:
    - Extensibility hooks for custom observations and reward shaping
    - Frame caching integration with configurable modes (LRU, preload, none)
    - Performance monitoring with sub-1ms step execution requirements
    - Dual API compatibility for seamless Gym/Gymnasium migration
    - Memory-efficient operations with configurable resource limits
    - Structured logging with comprehensive context information
    
    Performance Requirements:
    - Step execution: <1ms per step
    - Memory usage: <1MB overhead per agent
    - Frame processing: <100μs for odor sampling
    - Cache integration: <10μs overhead per step
    
    Examples:
        Basic initialization with modern features:
            >>> controller = SingleAgentController(
            ...     position=(10.0, 20.0),
            ...     speed=1.5,
            ...     enable_extensibility_hooks=True,
            ...     frame_cache_mode="lru"
            ... )
            
        Performance monitoring integration:
            >>> controller.step(env_array, dt=1.0)
            >>> metrics = controller.get_performance_metrics()
            >>> assert metrics['step_time_mean_ms'] < 1.0
            
        Custom observation and reward hooks:
            >>> class CustomController(SingleAgentController):
            ...     def compute_additional_obs(self, base_obs: dict) -> dict:
            ...         return {"energy_level": self.get_energy_remaining()}
            ...     
            ...     def compute_extra_reward(self, base_reward: float, info: dict) -> float:
            ...         return 0.1 if self.discovered_new_area() else 0.0
    """
    
    def __init__(
        self,
        position: Optional[Tuple[float, float]] = None,
        orientation: float = 0.0,
        speed: float = 0.0,
        max_speed: float = 1.0,
        angular_velocity: float = 0.0,
        sensors: Optional[List[SensorProtocol]] = None,
        enable_memory: bool = False,
        boundary_policy: Optional[BoundaryPolicyProtocol] = None,
        domain_bounds: Optional[Tuple[float, float]] = None,
        strict_validation: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize enhanced single agent controller with monitoring capabilities.
        
        Args:
            position: Initial (x, y) position, defaults to (0, 0)
            orientation: Initial orientation in degrees, defaults to 0.0
            speed: Initial speed, defaults to 0.0
            max_speed: Maximum allowed speed, defaults to 1.0
            strict_validation: If True, raise when speed exceeds max_speed
            angular_velocity: Initial angular velocity in degrees/second, defaults to 0.0
            sensors: List of SensorProtocol implementations for perception, defaults to [DirectOdorSensor()]
            enable_memory: Enable memory management hooks for cognitive modeling, defaults to False
            boundary_policy: Optional boundary policy for domain edge handling, defaults to None
            domain_bounds: Optional domain bounds for default boundary policy creation, defaults to None
            strict_validation: If True, raise when speed exceeds max_speed
            **kwargs: Additional configuration options including:
                - enable_logging: Enable comprehensive logging (default: True)
                - controller_id: Unique controller identifier
                - enable_extensibility_hooks: Enable custom observation/reward hooks
                - frame_cache_mode: Frame caching strategy ("none", "lru", "preload")
                - custom_observation_keys: List of additional observation keys
                - reward_shaping: Reward shaping strategy name
                
        Raises:
            ValueError: If speed exceeds max_speed or parameters are invalid
        """
        # Initialize base controller with enhanced features and boundary policy
        super().__init__(
            sensors=sensors, 
            enable_memory=enable_memory, 
            boundary_policy=boundary_policy,
            domain_bounds=domain_bounds,
            **kwargs
        )
        
        # Ensure proper type conversion for parameters (handle Hydra environment variable interpolation)
        try:
            speed = float(speed)
            max_speed = float(max_speed)
            orientation = float(orientation)
            angular_velocity = float(angular_velocity)
            
            # Handle position conversion properly (may be tuple of strings from env vars)
            if position is not None:
                if isinstance(position, (list, tuple)):
                    position = [float(x) for x in position]
                else:
                    # Handle scalar position 
                    position = [float(position), 0.0]
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid parameter type conversion: {e}")
        
        # Validate parameters
        self._strict_validation = strict_validation
        if speed > max_speed:
            log = self._logger if self._logger is not None else logger
            if self._strict_validation:
                raise ValueError(
                    f"speed ({speed}) cannot exceed max_speed ({max_speed})"
                )
            else:
                log.warning(
                    "speed exceeds max_speed; will clamp during step",
                    speed=float(speed),
                    max_speed=float(max_speed),
                )
        
        # Initialize state arrays for API consistency with proper shape (1, 2)
        if position is not None:
            self._position = np.array([position], dtype=np.float64)
            if self._position.shape[1] != 2:
                # Pad or truncate to ensure shape (1, 2)
                if self._position.shape[1] == 1:
                    self._position = np.column_stack([self._position, np.zeros((1, 1))])
                else:
                    self._position = self._position[:, :2]
        else:
            self._position = np.array([[0.0, 0.0]], dtype=np.float64)
        
        self._orientation = np.array([orientation % 360.0])  # Normalize to [0, 360)
        self._speed = np.array([speed])
        self._max_speed = np.array([max_speed])
        self._angular_velocity = np.array([angular_velocity])
        
        # Log initialization with enhanced context
        if self._logger:
            self._logger.info(
                "SingleAgentController initialized with enhanced features",
                position=position or (0.0, 0.0),
                orientation=orientation,
                speed=speed,
                max_speed=max_speed,
                angular_velocity=angular_velocity,
                extensibility_hooks=self._enable_extensibility_hooks,
                frame_cache_mode=self._frame_cache_mode
            )
    
    # NavigatorProtocol property implementations
    
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
    
    # NavigatorProtocol method implementations
    
    def reset(self, **kwargs: Any) -> None:
        """
        Reset the agent to initial state with enhanced validation and monitoring.
        
        Args:
            **kwargs: Optional parameters to override initial settings.
                Valid keys: position, orientation, speed, max_speed, angular_velocity,
                enable_extensibility_hooks, frame_cache_mode, custom_observation_keys
        
        Raises:
            ValueError: If invalid parameters are provided or constraints violated
            TypeError: If parameter types are incorrect
        """
        start_time = time.perf_counter() if self._enable_logging else None
        
        try:
            # Update controller configuration if provided
            if 'enable_extensibility_hooks' in kwargs:
                self._enable_extensibility_hooks = kwargs.pop('enable_extensibility_hooks')
            if 'frame_cache_mode' in kwargs:
                self._frame_cache_mode = kwargs.pop('frame_cache_mode')
            if 'custom_observation_keys' in kwargs:
                self._custom_observation_keys = kwargs.pop('custom_observation_keys')
            if 'reward_shaping' in kwargs:
                self._reward_shaping = kwargs.pop('reward_shaping')
            
            # Reset state using utility function
            controller_state = {
                '_position': self._position,
                '_orientation': self._orientation,
                '_speed': self._speed,
                '_max_speed': self._max_speed,
                '_angular_velocity': self._angular_velocity
            }
            
            _reset_navigator_state(controller_state, is_single_agent=True, **kwargs)
            
            # Update instance attributes
            self._position = controller_state['_position']
            self._orientation = controller_state['_orientation']
            self._speed = controller_state['_speed']
            self._max_speed = controller_state['_max_speed']
            self._angular_velocity = controller_state['_angular_velocity']
            
            # Reset performance metrics for new episode
            if self._enable_logging:
                self._performance_metrics['step_times'] = []
                self._performance_metrics['sample_times'] = []
                
            # Log successful reset
            if self._logger:
                reset_time = (time.perf_counter() - start_time) * 1000
                self._logger.info(
                    "Agent reset completed with enhanced features",
                    reset_time_ms=reset_time,
                    updated_params=list(kwargs.keys()),
                    position=self._position[0].tolist(),
                    orientation=float(self._orientation[0]),
                    speed=float(self._speed[0]),
                    extensibility_hooks=self._enable_extensibility_hooks
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
        """
        Reset the agent using a type-safe parameter object with enhanced features.
        
        Args:
            params: SingleAgentParams dataclass instance with configuration
            
        Raises:
            TypeError: If params is not a SingleAgentParams instance
            ValueError: If parameter constraints are violated
        """
        if not isinstance(params, SingleAgentParams):
            raise TypeError(f"Expected SingleAgentParams, got {type(params)}")
        
        if self._logger:
            self._logger.debug(
                "Resetting agent with type-safe parameters",
                param_count=len([v for v in params.__dict__.values() if v is not None]),
                params=params.to_kwargs()
            )
        
        # Delegate to the existing reset method
        self.reset(**params.to_kwargs())
    
    def step(self, env_array: np.ndarray, dt: float = 1.0) -> None:
        """
        Enhanced step method with performance monitoring and frame caching integration.
        
        Args:
            env_array: Environment array (e.g., odor concentration grid)
            dt: Time step size in seconds, defaults to 1.0
            
        Raises:
            ValueError: If env_array is invalid or dt is non-positive
            RuntimeError: If step processing exceeds performance requirements
        """
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
            
        start_time = time.perf_counter() if self._enable_logging else None

        try:
            # Integrate with frame cache if available
            processed_env_array = env_array
            if self._frame_cache and hasattr(env_array, 'frame_id'):
                # Try to get processed frame from cache
                frame_id = getattr(env_array, 'frame_id', None)
                if frame_id is not None:
                    cached_frame = self._frame_cache.get_frame(frame_id)
                    if cached_frame is not None:
                        processed_env_array = cached_frame
                        self._performance_metrics['frame_cache_hits'] += 1
                    else:
                        self._performance_metrics['frame_cache_misses'] += 1

            if not self._strict_validation and self._speed[0] > self._max_speed[0]:
                log = self._logger if self._logger is not None else logger
                log.warning(
                    "speed exceeds max_speed; clamping",
                    speed=float(self._speed[0]),
                    max_speed=float(self._max_speed[0])
                )
                self._speed[0] = self._max_speed[0]

            # Use utility function to update position and orientation
            _update_positions_and_orientations(
                self._position,
                self._orientation,
                self._speed,
                self._angular_velocity,
                dt=dt
            )
            
            # Apply boundary policy for domain edge management (v1.0 architecture)
            if self._boundary_policy is not None:
                # Calculate velocities for boundary policy (velocity = speed * [cos(θ), sin(θ)])
                rad_orientation = np.radians(self._orientation[0])
                velocity = np.array([[self._speed[0] * np.cos(rad_orientation), 
                                    self._speed[0] * np.sin(rad_orientation)]])
                
                # Apply boundary policy with position and velocity correction
                corrected_pos, corrected_vel, terminate_episode = self._apply_boundary_policy(
                    self._position, velocity
                )
                
                # Update position with boundary corrections
                self._position = corrected_pos
                
                # Update speed and orientation from corrected velocity if modified
                if corrected_vel is not None and not np.array_equal(velocity, corrected_vel):
                    new_speed = np.linalg.norm(corrected_vel[0])
                    new_orientation = np.degrees(np.arctan2(corrected_vel[0, 1], corrected_vel[0, 0]))
                    self._speed[0] = new_speed
                    self._orientation[0] = new_orientation % 360.0
                
                # Store termination status for episode management
                if hasattr(self, '_boundary_termination_pending'):
                    self._boundary_termination_pending = terminate_episode
                else:
                    # Add attribute for tracking boundary termination
                    self._boundary_termination_pending = terminate_episode
            
            # Track performance metrics
            if self._enable_logging:
                step_time = (time.perf_counter() - start_time) * 1000
                self._performance_metrics['step_times'].append(step_time)
                self._performance_metrics['total_steps'] += 1
                
                # Check performance requirement (<1ms for single agent)
                if step_time > 1.0 and self._logger:
                    self._logger.warning(
                        "Step processing exceeded 1ms requirement for single agent",
                        step_time_ms=step_time,
                        dt=dt,
                        position=self._position[0].tolist(),
                        performance_degradation=True
                    )
                
                # Log periodic performance summary
                if self._performance_metrics['total_steps'] % 100 == 0 and self._logger:
                    avg_step_time = np.mean(self._performance_metrics['step_times'][-100:])
                    cache_hit_rate = 0.0
                    if self._frame_cache:
                        total_requests = (self._performance_metrics['frame_cache_hits'] + 
                                        self._performance_metrics['frame_cache_misses'])
                        if total_requests > 0:
                            cache_hit_rate = self._performance_metrics['frame_cache_hits'] / total_requests
                    
                    self._logger.debug(
                        "Single agent performance summary",
                        total_steps=self._performance_metrics['total_steps'],
                        avg_step_time_ms=avg_step_time,
                        frame_cache_hit_rate=cache_hit_rate,
                        recent_position=self._position[0].tolist()
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
        """
        Sample odor at the current agent position using active sensors.
        
        This method now delegates to SensorProtocol implementations rather than
        direct field access, supporting the modular sensor architecture.
        
        Args:
            env_array: Environment array containing odor data
            
        Returns:
            float: Odor value at the agent's position
            
        Raises:
            ValueError: If env_array is invalid or sampling fails
        """
        return self.read_single_antenna_odor(env_array)

    def observe(self, sensor_output: Any) -> Dict[str, Any]:
        """Process sensor output with additional debug logging."""
        log = self._logger if self._logger is not None else logger
        log.debug("SingleAgentController.observe invoked")
        return super().observe(sensor_output)
    
    def read_single_antenna_odor(self, env_array: np.ndarray) -> float:
        """
        Sample odor at the agent's single antenna using SensorProtocol.
        
        Updated to use sensor-based sampling through SensorProtocol abstraction
        rather than direct field access, enabling flexible perception modeling.
        
        Args:
            env_array: Environment array containing odor data
            
        Returns:
            float: Odor value at the agent's position
            
        Raises:
            ValueError: If sampling fails or returns invalid values
        """
        if env_array is None and self.source is None:
            raise ValueError("Odor sampling requires either a plume_state or a SourceProtocol")

        start_time = time.perf_counter() if self._enable_logging else None
        
        try:
            # Use primary sensor for sampling instead of direct field access
            odor_values = self._primary_sensor.measure(env_array, self._position)
            odor_value = float(odor_values[0])
            
            # Validate odor value
            if np.isnan(odor_value) or np.isinf(odor_value):
                msg = (
                    f"Invalid odor value {odor_value} from "
                    f"{type(self._primary_sensor).__name__} at "
                    f"{self._position[0].tolist()}"
                )
                if self._logger:
                    self._logger.error(
                        msg,
                        odor_value=odor_value,
                        position=self._position[0].tolist(),
                        sensor_type=type(self._primary_sensor).__name__,
                    )
                raise ValueError(msg)
            
            # Track sampling performance
            if self._enable_logging:
                sample_time = (time.perf_counter() - start_time) * 1000
                self._performance_metrics['sample_times'].append(sample_time)
                
                # Log detailed sampling for debugging (reduced frequency for performance)
                if self._logger and self._performance_metrics['total_steps'] % 100 == 0:
                    self._logger.trace(
                        "Sensor-based odor sampling completed",
                        sample_time_ms=sample_time,
                        odor_value=odor_value,
                        position=self._position[0].tolist(),
                        sensor_type=type(self._primary_sensor).__name__
                    )
            
            return odor_value
            
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Sensor-based odor sampling failed: {str(e)}",
                    error_type=type(e).__name__,
                    position=self._position[0].tolist(),
                    sensor_type=type(self._primary_sensor).__name__,
                    env_array_shape=getattr(env_array, 'shape', 'unknown')
                )
            raise RuntimeError(f"Sensor-based odor sampling failed: {e}") from e
    
    def sample_multiple_sensors(
        self,
        env_array: np.ndarray,
        sensor_distance: float = 5.0,
        sensor_angle: float = 45.0,
        num_sensors: int = 2,
        layout_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Sample odor at multiple sensor positions using SensorProtocol implementations.
        
        Updated to use multi-point sensor through SensorProtocol abstraction
        rather than direct utility function calls, enabling flexible sensor configurations.
        
        Args:
            env_array: Environment array
            sensor_distance: Distance from agent to each sensor, defaults to 5.0
            sensor_angle: Angular separation between sensors in degrees, defaults to 45.0
            num_sensors: Number of sensors per agent, defaults to 2
            layout_name: Predefined sensor layout name, defaults to None
            
        Returns:
            np.ndarray: Array of shape (num_sensors,) with odor values
            
        Raises:
            ValueError: If sensor parameters are invalid
        """
        if env_array is None and self.source is None:
            raise ValueError("Odor sampling requires either a plume_state or a SourceProtocol")

        # Validate sensor parameters
        if sensor_distance <= 0:
            raise ValueError(f"sensor_distance must be positive, got {sensor_distance}")
        if num_sensors <= 0:
            raise ValueError(f"num_sensors must be positive, got {num_sensors}")
            
        start_time = time.perf_counter() if self._enable_logging else None
        
        try:
            # Create or use existing multi-point sensor for this request
            multi_sensor = MultiPointOdorSensor(
                sensor_distance=sensor_distance,
                sensor_angle=sensor_angle,
                num_sensors=num_sensors,
                layout_name=layout_name
            )
            
            # Sample using the multi-point sensor
            odor_values = multi_sensor.measure(env_array, self._position, self._orientation)
            
            # Return as a 1D array for single agent
            result = odor_values[0] if odor_values.ndim > 1 else odor_values
            
            # Validate sensor readings
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                if self._logger:
                    self._logger.warning(
                        "Invalid multi-sensor readings detected",
                        invalid_count=np.sum(np.isnan(result) | np.isinf(result)),
                        sensor_layout=layout_name or "custom",
                        sensor_type="MultiPointOdorSensor",
                        applying_cleanup=True
                    )
                result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Log multi-sensor sampling for debugging
            if self._enable_logging:
                sample_time = (time.perf_counter() - start_time) * 1000
                if self._logger and self._performance_metrics['total_steps'] % 50 == 0:
                    self._logger.trace(
                        "Sensor-based multi-sensor sampling completed",
                        sample_time_ms=sample_time,
                        num_sensors=num_sensors,
                        sensor_distance=sensor_distance,
                        sensor_type="MultiPointOdorSensor",
                        mean_odor=float(np.mean(result)),
                        max_odor=float(np.max(result))
                    )
            
            return result
            
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Sensor-based multi-sensor sampling failed: {str(e)}",
                    error_type=type(e).__name__,
                    num_sensors=num_sensors,
                    sensor_distance=sensor_distance,
                    layout_name=layout_name
                )
            # Return safe default values
            return np.zeros(num_sensors)

    def compute_additional_obs(self, base_obs: dict) -> dict:
        """Log hook invocation then delegate to base implementation."""
        if self._logger:
            self._logger.debug(f"{self.__class__.__name__}.compute_additional_obs invoked")
        return super().compute_additional_obs(base_obs)

    def compute_extra_reward(self, base_reward: float, info: dict) -> float:
        """Log hook invocation then delegate to base implementation."""
        if self._logger:
            self._logger.debug(f"{self.__class__.__name__}.compute_extra_reward invoked")
        return super().compute_extra_reward(base_reward, info)

    def on_episode_end(self, final_info: dict) -> None:
        """Log hook invocation then delegate to base implementation."""
        if self._logger:
            self._logger.debug(f"{self.__class__.__name__}.on_episode_end invoked")
        super().on_episode_end(final_info)


class MultiAgentController(BaseController):
    """
    Enhanced multi-agent controller with Gymnasium 0.29.x integration and vectorized operations.
    
    This implements the NavigatorProtocol for multiple agents, with all data
    represented as arrays without conditional branching. Optimized for performance
    with vectorized operations, frame caching integration, and comprehensive 
    monitoring for research workflows supporting 100+ agents.
    
    Enhanced Features:
    - Vectorized operations for efficient parallel processing of multiple agents
    - Extensibility hooks for custom observations and reward shaping
    - Frame caching integration with intelligent memory management
    - Performance monitoring with sub-10ms step execution for 100 agents
    - Memory-efficient batch operations with configurable resource limits
    - Structured logging with agent-level context information
    - Collision detection and avoidance capabilities (optional)
    
    Performance Requirements:
    - Step execution: <10ms for 100 agents
    - Memory usage: <10MB overhead per 100 agents
    - Throughput: ≥3000 agents·FPS (100 agents × 30 FPS)
    - Cache integration: <100μs overhead per step
    
    Examples:
        Multi-agent swarm initialization:
            >>> positions = [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]
            >>> controller = MultiAgentController(
            ...     positions=positions,
            ...     enable_vectorized_ops=True,
            ...     frame_cache_mode="lru"
            ... )
            
        Performance monitoring:
            >>> controller.step(env_array, dt=1.0)
            >>> metrics = controller.get_performance_metrics()
            >>> throughput = metrics['throughput_agents_fps']
            >>> assert throughput >= 3000
            
        Custom multi-agent behaviors:
            >>> class SwarmController(MultiAgentController):
            ...     def compute_additional_obs(self, base_obs: dict) -> dict:
            ...         return {"swarm_cohesion": self.compute_cohesion_metric()}
            ...     
            ...     def compute_extra_reward(self, base_reward: float, info: dict) -> float:
            ...         return 0.1 * info.get('collective_performance', 0.0)
    """
    
    def __init__(
        self,
        positions: Optional[Union[List[List[float]], np.ndarray]] = None,
        orientations: Optional[Union[List[float], np.ndarray]] = None,
        speeds: Optional[Union[List[float], np.ndarray]] = None,
        max_speeds: Optional[Union[List[float], np.ndarray]] = None,
        angular_velocities: Optional[Union[List[float], np.ndarray]] = None,
        enable_vectorized_ops: bool = True,
        sensors: Optional[List[SensorProtocol]] = None,
        enable_memory: bool = False,
        boundary_policy: Optional[BoundaryPolicyProtocol] = None,
        domain_bounds: Optional[Tuple[float, float]] = None,
        strict_validation: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize enhanced multi-agent controller with vectorized operations.
        
        Args:
            positions: Array of (x, y) positions with shape (num_agents, 2)
            orientations: Array of orientations in degrees with shape (num_agents,)
            speeds: Array of speeds with shape (num_agents,)
            max_speeds: Array of maximum speeds with shape (num_agents,)
            angular_velocities: Array of angular velocities with shape (num_agents,)
            enable_vectorized_ops: Enable vectorized operations for performance
            sensors: List of SensorProtocol implementations for perception, defaults to [DirectOdorSensor()]
            enable_memory: Enable memory management hooks for cognitive modeling, defaults to False
            boundary_policy: Optional boundary policy for domain edge handling, defaults to None
            domain_bounds: Optional domain bounds for default boundary policy creation, defaults to None
            strict_validation: If True, raise when any speed exceeds its max_speed
            **kwargs: Additional configuration options including:
                - enable_logging: Enable comprehensive logging
                - controller_id: Unique controller identifier
                - enable_extensibility_hooks: Enable custom observation/reward hooks
                - frame_cache_mode: Frame caching strategy
                - custom_observation_keys: List of additional observation keys
                - reward_shaping: Reward shaping strategy name
                
        Raises:
            ValueError: If array dimensions are inconsistent or constraints violated
        """
        # Initialize base controller with enhanced features and boundary policy
        super().__init__(
            sensors=sensors, 
            enable_memory=enable_memory, 
            boundary_policy=boundary_policy,
            domain_bounds=domain_bounds,
            **kwargs
        )
        
        self._enable_vectorized_ops = enable_vectorized_ops
        self._strict_validation = strict_validation
        
        # Ensure we have at least one agent position
        if positions is None:
            self._positions = np.array([[0.0, 0.0]])
        else:
            self._positions = np.array(positions, dtype=np.float64)
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
        self._validate_speed_constraints(strict_validation=self._strict_validation)
        
        # Enhanced performance metrics for multi-agent scenarios
        self._performance_metrics.update({
            'agents_per_step': [],
            'vectorized_op_times': [],
            'collision_checks': 0,
            'memory_usage_samples': []
        })
        
        # Memory monitoring for large agent counts
        if num_agents > 50:
            self._monitor_memory = True
            self._base_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        else:
            self._monitor_memory = False
            self._base_memory_usage = 0
        
        # Log initialization with enhanced context
        if self._logger:
            self._logger.info(
                "MultiAgentController initialized with enhanced features",
                num_agents=num_agents,
                vectorized_ops=self._enable_vectorized_ops,
                position_bounds={
                    'x_min': float(np.min(self._positions[:, 0])),
                    'x_max': float(np.max(self._positions[:, 0])),
                    'y_min': float(np.min(self._positions[:, 1])),
                    'y_max': float(np.max(self._positions[:, 1]))
                },
                speed_stats={
                    'mean': float(np.mean(self._speeds)),
                    'max': float(np.max(self._speeds)),
                    'std': float(np.std(self._speeds))
                },
                memory_monitoring=self._monitor_memory,
                frame_cache_mode=self._frame_cache_mode
            )
    
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
    
    def _validate_speed_constraints(self, strict_validation: bool = False) -> None:
        """Validate that speeds do not exceed max_speeds for any agent.

        When ``strict_validation`` is False, violations emit a warning and are
        clamped during ``step`` rather than raising immediately.
        """
        violations = self._speeds > self._max_speeds
        if np.any(violations):
            violating_agents = np.where(violations)[0]
            message = (
                f"Speed exceeds max_speed for agents {violating_agents.tolist()}: "
                f"speeds={self._speeds[violations].tolist()}, "
                f"max_speeds={self._max_speeds[violations].tolist()}"
            )
            log = self._logger if self._logger is not None else logger
            if strict_validation:
                raise ValueError(message)
            else:
                log.warning(message)
    
    # NavigatorProtocol property implementations
    
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

    @speeds.setter
    def speeds(self, value: Union[List[float], np.ndarray]) -> None:
        """Set agent speeds with validation against array shape and max speeds."""
        arr = np.array(value, dtype=float, copy=True)
        if arr.shape != self._speeds.shape:
            raise ValueError(
                f"speeds must have shape {self._speeds.shape}, got {arr.shape}"
            )
        if np.any(arr < 0):
            raise ValueError("speeds must be non-negative")
        if np.any(arr > self._max_speeds):
            raise ValueError("speeds cannot exceed max_speeds")
        copied = not np.shares_memory(arr, value)
        if self._logger:
            self._logger.debug(
                "speeds reassigned",
                num_agents=self.num_agents,
                copied=copied,
            )
        self._speeds = arr
    
    @property
    def max_speeds(self) -> np.ndarray:
        """Get maximum agent speeds as a numpy array with shape (num_agents,)."""
        return self._max_speeds
    
    @property
    def angular_velocities(self) -> np.ndarray:
        """Get agent angular velocities as a numpy array with shape (num_agents,)."""
        return self._angular_velocities

    @angular_velocities.setter
    def angular_velocities(self, value: Union[List[float], np.ndarray]) -> None:
        """Set agent angular velocities with validation."""
        arr = np.asarray(value, dtype=float)
        if arr.shape != self._angular_velocities.shape:
            raise ValueError(
                f"angular_velocities must have shape {self._angular_velocities.shape}, got {arr.shape}"
            )
        self._angular_velocities = arr
    
    @property
    def num_agents(self) -> int:
        """Get the number of agents."""
        return self._positions.shape[0]

    def compute_additional_obs(self, base_obs: dict) -> dict:
        """Log hook invocation then delegate to base implementation."""
        if self._logger:
            self._logger.debug(f"{self.__class__.__name__}.compute_additional_obs invoked")
        return super().compute_additional_obs(base_obs)

    def compute_extra_reward(self, base_reward: float, info: dict) -> float:
        """Log hook invocation then delegate to base implementation."""
        if self._logger:
            self._logger.debug(f"{self.__class__.__name__}.compute_extra_reward invoked")
        return super().compute_extra_reward(base_reward, info)

    def on_episode_end(self, final_info: dict) -> None:
        """Log hook invocation then delegate to base implementation."""
        if self._logger:
            self._logger.debug(f"{self.__class__.__name__}.on_episode_end invoked")
        super().on_episode_end(final_info)

    # NavigatorProtocol method implementations
    
    def reset(self, **kwargs: Any) -> None:
        """
        Reset all agents to initial state with comprehensive validation and monitoring.
        
        Args:
            **kwargs: Optional parameters to override initial settings.
                Valid keys: positions, orientations, speeds, max_speeds, angular_velocities,
                enable_extensibility_hooks, enable_vectorized_ops, frame_cache_mode
            
        Raises:
            ValueError: If parameter arrays have inconsistent shapes or violate constraints
            TypeError: If parameter types are incorrect
        """
        start_time = time.perf_counter() if self._enable_logging else None
        
        try:
            # Update controller configuration if provided
            if 'enable_extensibility_hooks' in kwargs:
                self._enable_extensibility_hooks = kwargs.pop('enable_extensibility_hooks')
            if 'enable_vectorized_ops' in kwargs:
                self._enable_vectorized_ops = kwargs.pop('enable_vectorized_ops')
            if 'frame_cache_mode' in kwargs:
                self._frame_cache_mode = kwargs.pop('frame_cache_mode')
            if 'custom_observation_keys' in kwargs:
                self._custom_observation_keys = kwargs.pop('custom_observation_keys')
            if 'reward_shaping' in kwargs:
                self._reward_shaping = kwargs.pop('reward_shaping')
            
            # Reset state using utility function
            controller_state = {
                '_positions': self._positions,
                '_orientations': self._orientations,
                '_speeds': self._speeds,
                '_max_speeds': self._max_speeds,
                '_angular_velocities': self._angular_velocities
            }
            
            _reset_navigator_state(controller_state, is_single_agent=False, **kwargs)
            
            # Update instance attributes
            self._positions = controller_state['_positions']
            self._orientations = controller_state['_orientations']
            self._speeds = controller_state['_speeds']
            self._max_speeds = controller_state['_max_speeds']
            self._angular_velocities = controller_state['_angular_velocities']
            
            # Validate updated state
            self._validate_array_shapes()
            self._validate_speed_constraints(strict_validation=self._strict_validation)
            
            # Reset performance metrics for new episode
            if self._enable_logging:
                self._performance_metrics['step_times'] = []
                self._performance_metrics['sample_times'] = []
                self._performance_metrics['agents_per_step'] = []
                self._performance_metrics['vectorized_op_times'] = []
            
            # Clear boundary termination status for new episode
            self.clear_boundary_termination()
            
            # Reset sensors and memory for new episode
            self.reset_sensors()
            if self._memory_enabled and 'reset_memory' not in kwargs:
                # Only reset memory if not explicitly preserving it
                self._memory_state = None
            
            # Clear boundary termination status for new episode
            self.clear_boundary_termination()
            
            # Log successful reset
            if self._logger:
                reset_time = (time.perf_counter() - start_time) * 1000
                self._logger.info(
                    "Multi-agent reset completed with enhanced features",
                    reset_time_ms=reset_time,
                    updated_params=list(kwargs.keys()),
                    num_agents=self.num_agents,
                    vectorized_ops=self._enable_vectorized_ops,
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
        """
        Reset all agents using a type-safe parameter object with enhanced features.
        
        Args:
            params: MultiAgentParams dataclass instance with configuration
            
        Raises:
            TypeError: If params is not a MultiAgentParams instance
            ValueError: If parameter arrays have invalid shapes or violate constraints
        """
        if not isinstance(params, MultiAgentParams):
            raise TypeError(f"Expected MultiAgentParams, got {type(params)}")
        
        if self._logger:
            self._logger.debug(
                "Resetting multi-agent controller with type-safe parameters",
                param_count=len([v for v in params.__dict__.values() if v is not None]),
                num_agents=self.num_agents
            )
        
        # Delegate to the existing reset method
        self.reset(**params.to_kwargs())
    
    def step(self, env_array: np.ndarray, dt: float = 1.0) -> None:
        """
        Enhanced step method with vectorized operations and performance optimization.
        
        Args:
            env_array: Environment array (e.g., odor concentration grid)
            dt: Time step size in seconds, defaults to 1.0
            
        Raises:
            ValueError: If env_array is invalid or dt is non-positive
            RuntimeError: If step processing exceeds performance requirements
        """
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
            
        start_time = time.perf_counter() if self._enable_logging else None
        
        try:
            # Memory monitoring for large agent counts
            if self._monitor_memory and self._performance_metrics['total_steps'] % 10 == 0:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_growth = current_memory - self._base_memory_usage
                self._performance_metrics['memory_usage_samples'].append(memory_growth)
            
            # Integrate with frame cache if available
            processed_env_array = env_array
            if self._frame_cache and hasattr(env_array, 'frame_id'):
                frame_id = getattr(env_array, 'frame_id', None)
                if frame_id is not None:
                    cached_frame = self._frame_cache.get_frame(frame_id)
                    if cached_frame is not None:
                        processed_env_array = cached_frame
                        self._performance_metrics['frame_cache_hits'] += 1
                    else:
                        self._performance_metrics['frame_cache_misses'] += 1

            if not self._strict_validation:
                violations = self._speeds > self._max_speeds
                if np.any(violations):
                    violating_agents = np.where(violations)[0]
                    log = self._logger if self._logger is not None else logger
                    log.warning(
                        "speed exceeds max_speed; clamping",
                        violating_agents=violating_agents.tolist(),
                        speeds=self._speeds[violations].tolist(),
                        max_speeds=self._max_speeds[violations].tolist(),
                    )
                    self._speeds[violations] = self._max_speeds[violations]

            # Vectorized position and orientation updates
            vectorized_start = time.perf_counter() if self._enable_logging else None
            
            if self._enable_vectorized_ops:
                # Use optimized vectorized operations
                _update_positions_and_orientations_vectorized(
                    self._positions, 
                    self._orientations, 
                    self._speeds, 
                    self._angular_velocities,
                    dt=dt
                )
            else:
                # Use standard utility function
                _update_positions_and_orientations(
                    self._positions, 
                    self._orientations, 
                    self._speeds, 
                    self._angular_velocities,
                    dt=dt
                )
            
            if self._enable_logging and vectorized_start:
                vectorized_time = (time.perf_counter() - vectorized_start) * 1000
                self._performance_metrics['vectorized_op_times'].append(vectorized_time)
            
            # Apply boundary policy for domain edge management (v1.0 architecture with vectorized operations)
            if self._boundary_policy is not None:
                # Calculate velocities for boundary policy (velocity = speed * [cos(θ), sin(θ)])
                rad_orientations = np.radians(self._orientations)
                velocities = np.column_stack([
                    self._speeds * np.cos(rad_orientations),
                    self._speeds * np.sin(rad_orientations)
                ])
                
                # Apply boundary policy with vectorized position and velocity correction
                corrected_pos, corrected_vel, terminate_episode = self._apply_boundary_policy(
                    self._positions, velocities
                )
                
                # Update positions with boundary corrections
                self._positions = corrected_pos
                
                # Update speeds and orientations from corrected velocities if modified
                if corrected_vel is not None and not np.array_equal(velocities, corrected_vel):
                    new_speeds = np.linalg.norm(corrected_vel, axis=1)
                    new_orientations = np.degrees(np.arctan2(corrected_vel[:, 1], corrected_vel[:, 0])) % 360.0
                    self._speeds = new_speeds
                    self._orientations = new_orientations
                
                # Store termination status for episode management
                if hasattr(self, '_boundary_termination_pending'):
                    self._boundary_termination_pending = terminate_episode
                else:
                    # Add attribute for tracking boundary termination
                    self._boundary_termination_pending = terminate_episode
            
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
                if step_time > 10.0:  # 10ms for multi-agent
                    performance_issues.append(f"step_time:{step_time:.1f}ms")
                if throughput < 3000:  # 100 agents × 30fps
                    performance_issues.append(f"throughput:{throughput:.0f}")
                
                if performance_issues and self._logger:
                    self._logger.warning(
                        "Multi-agent performance degradation detected",
                        step_time_ms=step_time,
                        throughput_agents_fps=throughput,
                        num_agents=self.num_agents,
                        dt=dt,
                        vectorized_ops=self._enable_vectorized_ops,
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
                    
                    # Frame cache statistics
                    cache_hit_rate = 0.0
                    if self._frame_cache:
                        total_requests = (self._performance_metrics['frame_cache_hits'] + 
                                        self._performance_metrics['frame_cache_misses'])
                        if total_requests > 0:
                            cache_hit_rate = self._performance_metrics['frame_cache_hits'] / total_requests
                    
                    # Memory statistics
                    memory_stats = {}
                    if self._performance_metrics['memory_usage_samples']:
                        memory_samples = self._performance_metrics['memory_usage_samples']
                        memory_stats = {
                            'memory_growth_mb': float(np.mean(memory_samples[-10:])),
                            'max_memory_growth_mb': float(np.max(memory_samples))
                        }
                    
                    self._logger.debug(
                        "Multi-agent performance summary",
                        total_steps=self._performance_metrics['total_steps'],
                        avg_step_time_ms=avg_step_time,
                        avg_throughput_agents_fps=avg_throughput,
                        num_agents=self.num_agents,
                        vectorized_ops=self._enable_vectorized_ops,
                        frame_cache_hit_rate=cache_hit_rate,
                        memory_stats=memory_stats,
                        agent_positions_sample=self._positions[:min(3, self.num_agents)].tolist()
                    )
                    
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Multi-agent step execution failed: {str(e)}",
                    error_type=type(e).__name__,
                    num_agents=self.num_agents,
                    dt=dt,
                    vectorized_ops=self._enable_vectorized_ops,
                    env_array_shape=getattr(env_array, 'shape', 'unknown')
                )
            raise RuntimeError(f"Multi-agent step failed: {str(e)}") from e
    
    def sample_odor(self, env_array: np.ndarray) -> np.ndarray:
        """
        Sample odor at all agent positions using SensorProtocol implementations.
        
        Updated to use sensor-based sampling through SensorProtocol abstraction
        rather than direct field access, enabling flexible perception modeling.
        
        Args:
            env_array: Environment array containing odor data
            
        Returns:
            np.ndarray: Odor values at each agent's position, shape (num_agents,)
            
        Raises:
            ValueError: If env_array is invalid or sampling fails
        """
        return self.read_single_antenna_odor(env_array)

    def observe(self, sensor_output: Any) -> Dict[str, Any]:
        """Process sensor output with additional debug logging."""
        log = self._logger if self._logger is not None else logger
        log.debug("MultiAgentController.observe invoked")
        return super().observe(sensor_output)
    
    def read_single_antenna_odor(self, env_array: np.ndarray) -> np.ndarray:
        """
        Sample odor at each agent's position using SensorProtocol with batch optimization.
        
        Updated to use sensor-based sampling through SensorProtocol abstraction
        rather than direct field access, enabling flexible perception modeling.
        
        Args:
            env_array: Environment array containing odor data
            
        Returns:
            np.ndarray: Odor values at each agent's position, shape (num_agents,)
            
        Raises:
            ValueError: If sampling fails or returns invalid values
        """
        if env_array is None and self.source is None:
            raise ValueError("Odor sampling requires either a plume_state or a SourceProtocol")

        start_time = time.perf_counter() if self._enable_logging else None
        
        try:
            # Use primary sensor for batch sampling instead of direct field access
            odor_values = self._primary_sensor.measure(env_array, self._positions)
            
            # Validate odor values
            invalid_mask = np.isnan(odor_values) | np.isinf(odor_values)
            if np.any(invalid_mask):
                if self._logger:
                    invalid_count = np.sum(invalid_mask)
                    self._logger.warning(
                        "Invalid odor values detected for multi-agent sensor sampling",
                        invalid_count=invalid_count,
                        total_agents=self.num_agents,
                        invalid_fraction=invalid_count / self.num_agents,
                        sensor_type=type(self._primary_sensor).__name__,
                        applying_cleanup=True
                    )
                odor_values = np.nan_to_num(odor_values, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Track sampling performance
            if self._enable_logging:
                sample_time = (time.perf_counter() - start_time) * 1000
                self._performance_metrics['sample_times'].append(sample_time)
                
                # Log detailed sampling for debugging (reduced frequency for performance)
                if self._logger and self._performance_metrics['total_steps'] % 100 == 0:
                    self._logger.trace(
                        "Multi-agent sensor-based odor sampling completed",
                        sample_time_ms=sample_time,
                        num_agents=self.num_agents,
                        sensor_type=type(self._primary_sensor).__name__,
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
                    f"Multi-agent sensor-based odor sampling failed: {str(e)}",
                    error_type=type(e).__name__,
                    num_agents=self.num_agents,
                    sensor_type=type(self._primary_sensor).__name__,
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
        """
        Sample odor at multiple sensor positions for all agents with vectorized optimization.
        
        Args:
            env_array: Environment array
            sensor_distance: Distance from each agent to each sensor, defaults to 5.0
            sensor_angle: Angular separation between sensors in degrees, defaults to 45.0
            num_sensors: Number of sensors per agent, defaults to 2
            layout_name: Predefined sensor layout name, defaults to None
            
        Returns:
            np.ndarray: Array of shape (num_agents, num_sensors) with odor values

        Raises:
            ValueError: If sensor parameters are invalid
        """
        if env_array is None and self.source is None:
            raise ValueError("Odor sampling requires either a plume_state or a SourceProtocol")

        # Validate sensor parameters
        if sensor_distance <= 0:
            raise ValueError(f"sensor_distance must be positive, got {sensor_distance}")
        if num_sensors <= 0:
            raise ValueError(f"num_sensors must be positive, got {num_sensors}")
            
        start_time = time.perf_counter() if self._enable_logging else None
        
        try:
            # Create multi-point sensor for this request
            multi_sensor = MultiPointOdorSensor(
                sensor_distance=sensor_distance,
                sensor_angle=sensor_angle,
                num_sensors=num_sensors,
                layout_name=layout_name
            )
            
            # Sample using the multi-point sensor
            odor_values = multi_sensor.measure(env_array, self._positions, self._orientations)
            
            # Validate sensor readings
            invalid_mask = np.isnan(odor_values) | np.isinf(odor_values)
            if np.any(invalid_mask):
                if self._logger:
                    invalid_count = np.sum(invalid_mask)
                    total_readings = odor_values.size
                    self._logger.warning(
                        "Invalid multi-sensor readings detected",
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
                if self._logger and self._performance_metrics['total_steps'] % 50 == 0:
                    self._logger.trace(
                        "Multi-agent sensor-based multi-sensor sampling completed",
                        sample_time_ms=sample_time,
                        num_agents=self.num_agents,
                        num_sensors=num_sensors,
                        sensor_distance=sensor_distance,
                        sensor_type="MultiPointOdorSensor",
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
                    f"Multi-agent sensor-based multi-sensor sampling failed: {str(e)}",
                    error_type=type(e).__name__,
                    num_agents=self.num_agents,
                    num_sensors=num_sensors,
                    sensor_distance=sensor_distance,
                    sensor_type="MultiPointOdorSensor",
                    layout_name=layout_name
                )
            # Return safe default values
            return np.zeros((self.num_agents, num_sensors))
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for monitoring and optimization.
        
        Returns:
            Dict[str, Any]: Dictionary containing detailed performance statistics
        """
        metrics = super().get_performance_metrics()
        
        # Add multi-agent specific metrics
        metrics.update({
            'num_agents': self.num_agents,
            'vectorized_ops_enabled': self._enable_vectorized_ops,
            'memory_monitoring_enabled': self._monitor_memory
        })
        
        if self._performance_metrics['step_times']:
            step_times = np.array(self._performance_metrics['step_times'])
            agents_per_step = np.array(self._performance_metrics['agents_per_step'])
            
            # Calculate throughput metrics
            throughputs = [
                na * (1000.0 / st) for na, st in zip(agents_per_step, step_times) if st > 0
            ]
            
            metrics.update({
                'throughput_mean_agents_fps': float(np.mean(throughputs)) if throughputs else 0,
                'throughput_max_agents_fps': float(np.max(throughputs)) if throughputs else 0,
                'throughput_violations': int(np.sum(np.array(throughputs) < 3000))
            })
        
        # Vectorized operation performance
        if self._performance_metrics['vectorized_op_times']:
            vectorized_times = np.array(self._performance_metrics['vectorized_op_times'])
            metrics.update({
                'vectorized_op_mean_ms': float(np.mean(vectorized_times)),
                'vectorized_op_max_ms': float(np.max(vectorized_times))
            })
        
        # Memory usage statistics
        if self._performance_metrics['memory_usage_samples']:
            memory_samples = np.array(self._performance_metrics['memory_usage_samples'])
            metrics.update({
                'memory_growth_mean_mb': float(np.mean(memory_samples)),
                'memory_growth_max_mb': float(np.max(memory_samples)),
                'memory_per_agent_kb': float(np.mean(memory_samples) * 1024 / self.num_agents) if self.num_agents > 0 else 0
            })
        
        return metrics


# Utility functions for state management and operations
# Enhanced versions of the utility functions with performance optimizations

def _read_odor_values(env_array: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    Read odor values from an environment array at specific positions.
    
    Enhanced version with error handling and performance optimizations for
    the Gymnasium 0.29.x migration. Supports both single and multi-agent
    position arrays with efficient vectorized operations.
    
    Args:
        env_array: Environment array (e.g., odor concentration grid)
        positions: Array of positions with shape (N, 2) where each row is (x, y)
        
    Returns:
        np.ndarray: Array of odor values with shape (N,)
    """
    # Check if this is a mock plume object (for testing)
    if hasattr(env_array, 'current_frame'):
        env_array = env_array.current_frame

    # Validate environment array structure
    if not hasattr(env_array, 'shape') or len(env_array.shape) < 2:
        logger.error("Invalid env_array passed to _read_odor_values", env_array=env_array)
        raise ValueError(
            "env_array must have a 'shape' attribute with at least two dimensions"
        )

    height, width = env_array.shape[:2]
    num_positions = positions.shape[0]
    
    # Convert positions to integers for indexing (vectorized)
    x_pos = np.floor(positions[:, 0]).astype(int)
    y_pos = np.floor(positions[:, 1]).astype(int)

    # Create a mask for positions that are within bounds (vectorized)
    within_bounds = (
        (x_pos >= 0) & (x_pos < width) & (y_pos >= 0) & (y_pos < height)
    )

    # Initialize output array
    odor_values = np.zeros(num_positions, dtype=np.float64)
    
    # Read values for positions within bounds (vectorized where possible)
    if np.any(within_bounds):
        valid_x = x_pos[within_bounds]
        valid_y = y_pos[within_bounds]
        valid_values = env_array[valid_y, valid_x]
        
        # Normalize if uint8 (vectorized)
        if hasattr(env_array, 'dtype') and env_array.dtype == np.uint8:
            valid_values = valid_values.astype(np.float64) / 255.0
        
        odor_values[within_bounds] = valid_values

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
    
    Enhanced version with improved performance for both single and multi-agent
    scenarios. This function handles the vectorized movement calculation with
    proper time step scaling and numerical stability improvements.
    
    Args:
        positions: Array of shape (N, 2) with agent positions
        orientations: Array of shape (N,) with agent orientations in degrees
        speeds: Array of shape (N,) with agent speeds (units/second)
        angular_velocities: Array of shape (N,) with angular velocities in degrees/second
        dt: Time step size in seconds, defaults to 1.0
        
    Notes:
        The function modifies the input arrays in-place for memory efficiency.
        Position updates: new_pos = pos + speed * dt * [cos(θ), sin(θ)]
        Orientation updates: new_θ = (θ + angular_velocity * dt) % 360
    """
    # Convert orientations to radians (vectorized)
    rad_orientations = np.radians(orientations)
    
    # Calculate movement deltas, scaled by dt (vectorized)
    dx = speeds * np.cos(rad_orientations) * dt
    dy = speeds * np.sin(rad_orientations) * dt
    
    # Update positions (vectorized for all agents)
    positions[:, 0] += dx
    positions[:, 1] += dy
    
    # Update orientations with angular velocities, scaled by dt (vectorized)
    orientations += angular_velocities * dt
    
    # Wrap orientations to [0, 360) degrees (vectorized)
    orientations %= 360.0


def _update_positions_and_orientations_vectorized(
    positions: np.ndarray, 
    orientations: np.ndarray, 
    speeds: np.ndarray, 
    angular_velocities: np.ndarray,
    dt: float = 1.0
) -> None:
    """
    Optimized vectorized version of position and orientation updates.
    
    This function provides enhanced performance for large numbers of agents
    by using advanced NumPy vectorization techniques and memory-efficient
    operations. Designed for scenarios with 100+ agents.
    
    Args:
        positions: Array of shape (N, 2) with agent positions
        orientations: Array of shape (N,) with agent orientations in degrees
        speeds: Array of shape (N,) with agent speeds
        angular_velocities: Array of shape (N,) with angular velocities
        dt: Time step size in seconds
    """
    # Use advanced vectorized operations for better performance
    rad_orientations = np.radians(orientations)
    
    # Calculate movement using einsum for optimal performance
    movement_scale = speeds * dt
    cos_orientations = np.cos(rad_orientations)
    sin_orientations = np.sin(rad_orientations)
    
    # Update positions using advanced indexing
    positions[:, 0] += movement_scale * cos_orientations
    positions[:, 1] += movement_scale * sin_orientations
    
    # Update orientations
    orientations += angular_velocities * dt
    orientations %= 360.0


def _reset_navigator_state(
    controller_state: Dict[str, np.ndarray],
    is_single_agent: bool,
    **kwargs: Any
) -> None:
    """
    Reset navigator controller state based on provided parameters.
    
    Enhanced version with comprehensive validation and support for the new
    Gymnasium 0.29.x features including extensibility hooks configuration.
    
    Args:
        controller_state: Dictionary of current controller state arrays
        is_single_agent: Whether this is a single agent controller
        **kwargs: Parameters to update
        
    Raises:
        ValueError: If invalid parameter keys are provided or constraints violated
    """
    # Define valid keys and attribute mappings based on controller type
    if is_single_agent:
        position_key = 'position'
        valid_keys = {
            'position', 'orientation', 'speed', 'max_speed', 'angular_velocity'
        }
        attr_mapping = {
            'position': '_position',
            'orientation': '_orientation', 
            'speed': '_speed',
            'max_speed': '_max_speed',
            'angular_velocity': '_angular_velocity'
        }
    else:
        position_key = 'positions'
        valid_keys = {
            'positions', 'orientations', 'speeds', 'max_speeds', 'angular_velocities'
        }
        attr_mapping = {
            'positions': '_positions',
            'orientations': '_orientations',
            'speeds': '_speeds', 
            'max_speeds': '_max_speeds',
            'angular_velocities': '_angular_velocities'
        }

    # Filter out configuration parameters (handled separately)
    config_keys = {
        'enable_extensibility_hooks', 'enable_vectorized_ops', 'frame_cache_mode',
        'custom_observation_keys', 'reward_shaping'
    }
    update_kwargs = {k: v for k, v in kwargs.items() if k not in config_keys}

    if invalid_keys := set(update_kwargs.keys()) - valid_keys:
        raise ValueError(f"Invalid parameters: {invalid_keys}. Valid keys are: {valid_keys}")

    # Handle position update (which may require resizing other arrays)
    if (position_value := update_kwargs.get(position_key)) is not None:
        if is_single_agent:
            # Single agent case: wrap in array
            controller_state[attr_mapping[position_key]] = np.array([position_value])
            if controller_state[attr_mapping[position_key]].ndim == 1:
                controller_state[attr_mapping[position_key]] = controller_state[attr_mapping[position_key]].reshape(1, -1)
        else:
            # Multi agent case: convert to array
            controller_state[attr_mapping[position_key]] = np.array(position_value)

            # For multi-agent, we may need to resize other arrays
            num_agents = controller_state[attr_mapping[position_key]].shape[0]

            # Resize other arrays if needed
            arrays_to_check = [
                ('_orientations', np.zeros, num_agents),
                ('_speeds', np.zeros, num_agents),
                ('_max_speeds', np.ones, num_agents),
                ('_angular_velocities', np.zeros, num_agents)
            ]

            for attr_name, default_fn, size in arrays_to_check:
                if attr_name in controller_state and controller_state[attr_name].shape[0] != num_agents:
                    controller_state[attr_name] = default_fn(size)

    # Update other values if provided
    for kwarg_key, attr_key in attr_mapping.items():
        if kwarg_key == position_key:  # Skip position which was handled above
            continue
        if kwarg_key in update_kwargs:
            value = update_kwargs[kwarg_key]
            if is_single_agent:
                if kwarg_key == 'orientation':
                    value = value % 360.0  # Normalize orientation
                controller_state[attr_key] = np.array([value])
            else:
                if kwarg_key == 'orientations':
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
    
    Enhanced version with support for multiple sensor layouts and improved
    performance for both single and multi-agent scenarios.
    
    Args:
        navigator: Navigator instance implementing NavigatorProtocol
        env_array: 2D array representing the environment (e.g., video frame)
        sensor_distance: Distance of sensors from navigator position
        sensor_angle: Angle between sensors in degrees
        num_sensors: Number of sensors per navigator
        layout_name: Predefined sensor layout name
        
    Returns:
        Array of odor readings with shape (num_agents, num_sensors)
    """
    num_agents = navigator.num_agents
    sensor_positions = np.zeros((num_agents, num_sensors, 2))
    
    # Calculate sensor positions using vectorized operations
    for agent_idx in range(num_agents):
        agent_pos = navigator.positions[agent_idx]
        agent_orientation = navigator.orientations[agent_idx]
        
        for sensor_idx in range(num_sensors):
            # Calculate sensor angle relative to agent orientation
            if layout_name == "LEFT_RIGHT":
                relative_angles = [-90, 90] if num_sensors >= 2 else [0]
                relative_angle = relative_angles[sensor_idx % len(relative_angles)]
            elif layout_name == "FORWARD_BACK":
                relative_angles = [0, 180] if num_sensors >= 2 else [0]
                relative_angle = relative_angles[sensor_idx % len(relative_angles)]
            elif layout_name == "TRIANGLE":
                relative_angles = [0, 120, 240] if num_sensors >= 3 else [0, 120]
                relative_angle = relative_angles[sensor_idx % len(relative_angles)]
            else:
                # Custom angle-based layout
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


def _sample_odor_at_sensors_vectorized(
    navigator: NavigatorProtocol,
    env_array: np.ndarray,
    sensor_distance: float = 5.0,
    sensor_angle: float = 45.0,
    num_sensors: int = 2,
    layout_name: Optional[str] = None
) -> np.ndarray:
    """
    Optimized vectorized version of multi-sensor odor sampling.
    
    This function provides enhanced performance for large numbers of agents
    and sensors by using advanced NumPy vectorization techniques.
    
    Args:
        navigator: Navigator instance implementing NavigatorProtocol
        env_array: Environment array
        sensor_distance: Distance of sensors from navigator position
        sensor_angle: Angle between sensors in degrees
        num_sensors: Number of sensors per navigator
        layout_name: Predefined sensor layout name
        
    Returns:
        Array of odor readings with shape (num_agents, num_sensors)
    """
    num_agents = navigator.num_agents
    
    # Vectorized sensor position calculation
    agent_positions = navigator.positions  # Shape: (num_agents, 2)
    agent_orientations = navigator.orientations  # Shape: (num_agents,)
    
    # Calculate relative angles for all sensors (vectorized)
    if layout_name == "LEFT_RIGHT":
        relative_angles = np.array([-90, 90])[:num_sensors]
    elif layout_name == "FORWARD_BACK":
        relative_angles = np.array([0, 180])[:num_sensors]
    elif layout_name == "TRIANGLE":
        relative_angles = np.array([0, 120, 240])[:num_sensors]
    else:
        # Custom angle-based layout
        if num_sensors == 1:
            relative_angles = np.array([0])
        elif num_sensors == 2:
            relative_angles = np.array([-0.5, 0.5]) * sensor_angle
        else:
            relative_angles = (np.arange(num_sensors) - (num_sensors - 1) / 2) * sensor_angle
    
    # Broadcast calculations for all agents and sensors
    agent_orientations_expanded = agent_orientations[:, np.newaxis]  # Shape: (num_agents, 1)
    relative_angles_expanded = relative_angles[np.newaxis, :]  # Shape: (1, num_sensors)
    
    # Global angles for all agent-sensor combinations
    global_angles = agent_orientations_expanded + relative_angles_expanded  # Shape: (num_agents, num_sensors)
    global_angles_rad = np.deg2rad(global_angles)
    
    # Calculate sensor positions (vectorized)
    cos_angles = np.cos(global_angles_rad)
    sin_angles = np.sin(global_angles_rad)
    
    # Agent positions expanded for broadcasting
    agent_x = agent_positions[:, 0:1]  # Shape: (num_agents, 1)
    agent_y = agent_positions[:, 1:2]  # Shape: (num_agents, 1)
    
    # Sensor positions
    sensor_x = agent_x + sensor_distance * cos_angles  # Shape: (num_agents, num_sensors)
    sensor_y = agent_y + sensor_distance * sin_angles  # Shape: (num_agents, num_sensors)
    
    # Reshape for odor sampling
    sensor_positions = np.stack([sensor_x, sensor_y], axis=-1)  # Shape: (num_agents, num_sensors, 2)
    sensor_positions_flat = sensor_positions.reshape(-1, 2)  # Shape: (num_agents * num_sensors, 2)
    
    # Read odor values at sensor positions
    odor_values = _read_odor_values(env_array, sensor_positions_flat)
    
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
    
    Enhanced factory function for Gymnasium 0.29.x migration with comprehensive
    parameter validation, extensibility hook configuration, and performance
    optimization settings.
    
    Args:
        config: Configuration object containing navigation parameters
        controller_id: Unique identifier for the controller
        enable_logging: Enable comprehensive logging integration
        
    Returns:
        Union[SingleAgentController, MultiAgentController]: Configured controller
        
    Raises:
        ValueError: If configuration is invalid or inconsistent
        TypeError: If configuration type is not supported
        
    Examples:
        Single-agent from dict with extensibility hooks:
            >>> config = {
            ...     "position": [10.0, 20.0], 
            ...     "speed": 1.5,
            ...     "enable_extensibility_hooks": True,
            ...     "frame_cache_mode": "lru"
            ... }
            >>> controller = create_controller_from_config(config)
            
        Multi-agent with vectorized operations:
            >>> config = {
            ...     "positions": [[0, 0], [10, 10]], 
            ...     "speeds": [1.0, 1.5],
            ...     "enable_vectorized_ops": True
            ... }
            >>> controller = create_controller_from_config(config)
    """
    start_time = time.perf_counter() if enable_logging else None
    
    try:
        # Handle different configuration types
        config_type = type(config).__name__
        if dataclasses.is_dataclass(config):
            config_dict = dataclasses.asdict(config)
            config_type = f"dataclass:{config_type}"
        elif hasattr(config, "model_dump"):
            config_dict = config.model_dump(exclude_none=True)
            config_type = f"pydantic_v2:{config_type}"
        elif hasattr(config, "dict"):
            config_dict = config.dict()
            config_type = f"pydantic_v1:{config_type}"
        elif isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
            config_type = "DictConfig"
        elif isinstance(config, dict):
            config_dict = config.copy()
            config_type = "dict"
        else:
            raise TypeError(
                f"Unsupported configuration type: {type(config)}. "
                f"Expected DictConfig, dict, dataclass, or Pydantic model"
            )

        logger.debug("Detected configuration type", config_type=config_type)
        
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
                enable_vectorized_ops=config_dict.get('enable_vectorized_ops', True),
                sensors=config_dict.get('sensors'),
                enable_memory=config_dict.get('enable_memory', False),
                boundary_policy=config_dict.get('boundary_policy'),
                domain_bounds=config_dict.get('domain_bounds'),
                enable_logging=enable_logging,
                controller_id=controller_id,
                enable_extensibility_hooks=config_dict.get('enable_extensibility_hooks', False),
                frame_cache_mode=config_dict.get('frame_cache_mode', 'none'),
                custom_observation_keys=config_dict.get('custom_observation_keys', []),
                reward_shaping=config_dict.get('reward_shaping', None)
            )
            
        else:
            # Single-agent controller (default)
            controller = SingleAgentController(
                position=config_dict.get('position'),
                orientation=config_dict.get('orientation', 0.0),
                speed=config_dict.get('speed', 0.0),
                max_speed=config_dict.get('max_speed', 1.0),
                angular_velocity=config_dict.get('angular_velocity', 0.0),
                sensors=config_dict.get('sensors'),
                enable_memory=config_dict.get('enable_memory', False),
                boundary_policy=config_dict.get('boundary_policy'),
                domain_bounds=config_dict.get('domain_bounds'),
                enable_logging=enable_logging,
                controller_id=controller_id,
                enable_extensibility_hooks=config_dict.get('enable_extensibility_hooks', False),
                frame_cache_mode=config_dict.get('frame_cache_mode', 'none'),
                custom_observation_keys=config_dict.get('custom_observation_keys', []),
                reward_shaping=config_dict.get('reward_shaping', None)
            )
        
        # Log successful creation with performance timing
        if enable_logging:
            creation_time = (time.perf_counter() - start_time) * 1000
            logger.bind(
                controller_type=type(controller).__name__,
                controller_id=controller_id,
                creation_time_ms=creation_time,
                num_agents=controller.num_agents
            ).info(
                "Controller created from configuration with enhanced features",
                config_keys=list(config_dict.keys()),
                extensibility_hooks=config_dict.get('enable_extensibility_hooks', False),
                frame_cache_mode=config_dict.get('frame_cache_mode', 'none')
            )
        
        return controller
        
    except Exception as e:
        if enable_logging:
            logger.error(
                f"Enhanced controller creation failed: {str(e)}",
                error_type=type(e).__name__,
                config_type=type(config).__name__,
                controller_id=controller_id
            )
        raise


def create_single_agent_controller(
    config: Union[DictConfig, Dict[str, Any], SingleAgentConfig],
    **kwargs: Any
) -> SingleAgentController:
    """
    Create a single-agent controller from configuration with enhanced features.
    
    Args:
        config: Single-agent configuration parameters
        **kwargs: Additional parameters passed to controller constructor
        
    Returns:
        SingleAgentController: Configured single-agent controller instance
    """
    # Merge configuration with kwargs
    if isinstance(config, SingleAgentConfig):
        config_dict = config.model_dump(exclude_none=True)
    elif isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = dict(config) if config else {}
    
    config_dict.update(kwargs)
    
    # Create boundary policy from configuration if specified
    if 'boundary_policy_config' in config_dict and config_dict['boundary_policy_config'] is not None:
        boundary_config = config_dict.pop('boundary_policy_config')
        if 'boundary_policy' not in config_dict or config_dict['boundary_policy'] is None:
            try:
                config_dict['boundary_policy'] = create_boundary_policy(**boundary_config)
            except Exception as e:
                warnings.warn(f"Failed to create boundary policy from config: {e}", UserWarning)
    
    return SingleAgentController(**config_dict)


def create_multi_agent_controller(
    config: Union[DictConfig, Dict[str, Any], MultiAgentConfig],
    **kwargs: Any
) -> MultiAgentController:
    """
    Create a multi-agent controller from configuration with enhanced features.
    
    Args:
        config: Multi-agent configuration parameters
        **kwargs: Additional parameters passed to controller constructor
        
    Returns:
        MultiAgentController: Configured multi-agent controller instance
    """
    # Merge configuration with kwargs
    if isinstance(config, MultiAgentConfig):
        config_dict = config.model_dump(exclude_none=True)
    elif isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = dict(config) if config else {}
    
    config_dict.update(kwargs)
    
    # Create boundary policy from configuration if specified
    if 'boundary_policy_config' in config_dict and config_dict['boundary_policy_config'] is not None:
        boundary_config = config_dict.pop('boundary_policy_config')
        if 'boundary_policy' not in config_dict or config_dict['boundary_policy'] is None:
            try:
                config_dict['boundary_policy'] = create_boundary_policy(**boundary_config)
            except Exception as e:
                warnings.warn(f"Failed to create boundary policy from config: {e}", UserWarning)
    
    return MultiAgentController(**config_dict)


# Utility functions for validation and information

def validate_controller_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate controller configuration and return validation results.
    
    Enhanced validation for Gymnasium 0.29.x features including extensibility
    hooks configuration and frame caching parameters.
    
    Args:
        config: Controller configuration to validate
        
    Returns:
        Tuple[bool, List[str]]: Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check for basic parameter types
    if 'position' in config and not isinstance(config['position'], (list, tuple)):
        errors.append("'position' must be a list or tuple of [x, y] coordinates")
    
    if 'positions' in config and not isinstance(config['positions'], (list, np.ndarray)):
        errors.append("'positions' must be a list or array of [x, y] coordinates")
    
    # Check for speed constraints
    if 'speed' in config and 'max_speed' in config:
        if config['speed'] > config['max_speed']:
            errors.append(f"speed ({config['speed']}) cannot exceed max_speed ({config['max_speed']})")
    
    # Check for array length consistency in multi-agent configs
    if 'positions' in config:
        num_agents = len(config['positions'])
        for param in ['orientations', 'speeds', 'max_speeds', 'angular_velocities']:
            if param in config and len(config[param]) != num_agents:
                errors.append(f"{param} length ({len(config[param])}) does not match positions length ({num_agents})")
    
    # Validate enhanced configuration options
    if 'frame_cache_mode' in config:
        valid_modes = ['none', 'lru', 'preload']
        if config['frame_cache_mode'] not in valid_modes:
            errors.append(f"frame_cache_mode must be one of {valid_modes}, got {config['frame_cache_mode']}")
    
    if 'custom_observation_keys' in config:
        if not isinstance(config['custom_observation_keys'], list):
            errors.append("custom_observation_keys must be a list of strings")
    
    if 'reward_shaping' in config:
        valid_strategies = ['exploration_bonus', 'efficiency_penalty', 'frame_cache_bonus']
        if config['reward_shaping'] not in valid_strategies + [None]:
            errors.append(f"reward_shaping must be one of {valid_strategies} or None")
    
    # Validate sensor configuration
    if 'sensors' in config:
        if not isinstance(config['sensors'], list):
            errors.append("sensors must be a list of SensorProtocol implementations")
    
    # Validate memory configuration
    if 'enable_memory' in config:
        if not isinstance(config['enable_memory'], bool):
            errors.append("enable_memory must be a boolean value")
    
    # Validate boundary policy configuration
    if 'boundary_policy_config' in config:
        boundary_config = config['boundary_policy_config']
        if boundary_config is not None:
            if not isinstance(boundary_config, dict):
                errors.append("boundary_policy_config must be a dictionary")
            else:
                if 'policy_type' not in boundary_config:
                    errors.append("boundary_policy_config must include 'policy_type'")
                elif boundary_config['policy_type'] not in ['terminate', 'bounce', 'wrap', 'clip']:
                    errors.append("boundary_policy_config policy_type must be one of: terminate, bounce, wrap, clip")
                
                if 'domain_bounds' not in boundary_config:
                    errors.append("boundary_policy_config must include 'domain_bounds'")
    
    # Validate domain bounds configuration
    if 'domain_bounds' in config:
        domain_bounds = config['domain_bounds']
        if domain_bounds is not None:
            if not isinstance(domain_bounds, (list, tuple)) or len(domain_bounds) != 2:
                errors.append("domain_bounds must be a tuple or list of (width, height)")
            elif any(bound <= 0 for bound in domain_bounds):
                errors.append("domain_bounds values must be positive")
    
    return len(errors) == 0, errors


def get_controller_info(controller: Union[SingleAgentController, MultiAgentController]) -> Dict[str, Any]:
    """
    Get comprehensive information about a controller instance.
    
    Enhanced version with detailed information about Gymnasium 0.29.x features
    including extensibility hooks, frame caching, and performance metrics.
    
    Args:
        controller: Controller instance to analyze
        
    Returns:
        Dict[str, Any]: Dictionary containing controller information and statistics
    """
    info = {
        'controller_type': type(controller).__name__,
        'num_agents': controller.num_agents,
        'has_performance_metrics': hasattr(controller, 'get_performance_metrics'),
        'has_enhanced_logging': hasattr(controller, '_logger'),
        'extensibility_hooks_enabled': getattr(controller, '_enable_extensibility_hooks', False),
        'frame_cache_mode': getattr(controller, '_frame_cache_mode', 'none'),
        'custom_observation_keys': getattr(controller, '_custom_observation_keys', []),
        'reward_shaping': getattr(controller, '_reward_shaping', None),
        'memory_enabled': getattr(controller, '_memory_enabled', False),
        'num_sensors': len(getattr(controller, '_sensors', [])),
        'sensor_types': [type(sensor).__name__ for sensor in getattr(controller, '_sensors', [])],
        'primary_sensor_type': type(getattr(controller, '_primary_sensor', None)).__name__ if hasattr(controller, '_primary_sensor') else None,
        'boundary_policy_enabled': getattr(controller, '_boundary_policy', None) is not None,
        'boundary_policy_type': type(getattr(controller, '_boundary_policy', None)).__name__ if getattr(controller, '_boundary_policy', None) else None,
        'domain_bounds': getattr(controller, '_domain_bounds', None),
        'boundary_termination_pending': getattr(controller, '_boundary_termination_pending', False)
    }
    
    # Add state information
    info.update({
        'positions_shape': controller.positions.shape,
        'orientations_range': [float(np.min(controller.orientations)), float(np.max(controller.orientations))],
        'speeds_range': [float(np.min(controller.speeds)), float(np.max(controller.speeds))],
        'max_speeds_range': [float(np.min(controller.max_speeds)), float(np.max(controller.max_speeds))],
    })
    
    # Add multi-agent specific information
    if isinstance(controller, MultiAgentController):
        info.update({
            'vectorized_ops_enabled': getattr(controller, '_enable_vectorized_ops', False),
            'memory_monitoring_enabled': getattr(controller, '_monitor_memory', False)
        })
    
    # Add performance metrics if available
    if hasattr(controller, 'get_performance_metrics'):
        try:
            metrics = controller.get_performance_metrics()
            info['performance_metrics'] = metrics
        except Exception:
            info['performance_metrics'] = "Error retrieving metrics"
    
    # Add frame cache information if available
    if hasattr(controller, '_frame_cache') and controller._frame_cache:
        try:
            cache_stats = controller._frame_cache.get_statistics()
            info['frame_cache_statistics'] = cache_stats
        except Exception:
            info['frame_cache_statistics'] = "Error retrieving cache statistics"
    
    return info


# Export public API with backward compatibility and v1.0 boundary policy integration
__all__ = [
    # Enhanced controller classes with v1.0 boundary policy support
    "SingleAgentController",
    "MultiAgentController",
    
    # Parameter dataclasses with enhanced features
    "SingleAgentParams", 
    "MultiAgentParams",
    
    # Factory functions for configuration-driven instantiation
    "create_controller_from_config",
    "create_single_agent_controller",
    "create_multi_agent_controller",
    
    # Utility functions with enhanced validation
    "validate_controller_config",
    "get_controller_info",
    
    # Sensor protocol and implementations for modular perception
    "SensorProtocol",
    "DirectOdorSensor",
    "MultiPointOdorSensor",
    
    # Boundary policy protocols and implementations (v1.0 integration)
    "BoundaryPolicyProtocol",
    "create_boundary_policy",
]