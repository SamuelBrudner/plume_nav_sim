"""
NavigatorProtocol interface defining the structural contract for navigation controllers.

This module implements the core NavigatorProtocol that prescribes the exact properties
and methods that any concrete Navigator implementation must provide, ensuring uniform
API across single-agent and multi-agent navigation logic. The protocol supports type
safety, IDE tooling requirements, and seamless integration with Hydra configuration
management for enhanced ML framework compatibility with Gymnasium 0.29.x.

The protocol-based design enables researchers to implement custom navigation algorithms
while maintaining compatibility with the existing framework, supporting both research
extensibility and production-grade type safety.

Key enhancements for Gymnasium 0.29.x migration:
- Extensibility hooks for custom observations, rewards, and episode handling
- Dual API compatibility detection and automatic format conversion
- Integration with SpaceFactory utilities for type-safe space construction
- Enhanced performance monitoring and memory management integration

Modular Architecture Extensions:
- PlumeModelProtocol: Pluggable plume modeling (Gaussian, Turbulent, Video-based)
- WindFieldProtocol: Environmental wind dynamics with configurable complexity
- SensorProtocol: Flexible sensing modalities (Binary, Concentration, Gradient)
- AgentObservationProtocol: Structured observation construction with sensor integration
- AgentActionProtocol: Standardized action processing for different control modalities
- Optional memory interface: Support for both memory-based and non-memory-based agents
"""

from __future__ import annotations
from typing import Protocol, Union, Optional, Tuple, List, Any, Dict, Callable, runtime_checkable
from typing_extensions import Self
import numpy as np
import warnings
import inspect
from loguru import logger
from plume_nav_sim.protocols.wind_field import WindFieldProtocol
from plume_nav_sim.protocols import PlumeModelProtocol, SensorProtocol

# Use shared NavigatorProtocol definition
from plume_nav_sim.protocols.navigator import NavigatorProtocol
# Hydra imports for configuration integration
try:
    from omegaconf import DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    DictConfig = dict
    HYDRA_AVAILABLE = False

# Gymnasium imports for modern API support
try:
    import gymnasium
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    try:
        import gym as gymnasium
        from gym import spaces
        GYMNASIUM_AVAILABLE = True
    except ImportError:
        gymnasium = None
        spaces = None
        GYMNASIUM_AVAILABLE = False

try:
    from ..config.schemas import NavigatorConfig, SingleAgentConfig, MultiAgentConfig
    logger.info("NavigatorConfig, SingleAgentConfig, MultiAgentConfig successfully imported")
except ImportError as e:
    logger.error("Required configuration schemas are missing", exc_info=e)
    raise ImportError(
        "Required configuration schemas are missing. Ensure `plume_nav_sim.config.schemas` defines `NavigatorConfig`, "
        "`SingleAgentConfig`, and `MultiAgentConfig`."
    ) from e

try:
    from ..envs.spaces import SpaceFactory  # type: ignore
    SPACE_FACTORY_AVAILABLE = True
    logger.info("SpaceFactory successfully imported")
except ImportError as e:  # pragma: no cover - import error path
    SPACE_FACTORY_AVAILABLE = False
    SpaceFactory = None  # type: ignore
    logger.error("SpaceFactory import failed", exc_info=e)


@runtime_checkable
class _LegacyNavigatorProtocol(Protocol):
    """
    Protocol defining the structural interface for navigator controllers.
    
    This protocol prescribes the exact properties and methods that any concrete 
    Navigator implementation must provide, ensuring uniform API across single-agent 
    and multi-agent navigation logic. The protocol supports type safety, IDE tooling 
    requirements, and Hydra configuration integration for ML framework compatibility.
    
    Enhanced for Gymnasium 0.29.x migration with new extensibility hooks:
    - compute_additional_obs(): Custom observation augmentation
    - compute_extra_reward(): Custom reward shaping logic
    - on_episode_end(): Episode completion handling
    
    The protocol-based design enables algorithm extensibility while maintaining 
    compatibility with existing framework components including simulation runners, 
    visualization systems, and data recording infrastructure.
    
    Key Design Principles:
    - Uniform interface for both single and multi-agent scenarios
    - NumPy-based state representation for performance and compatibility
    - Hydra configuration integration for research workflow support  
    - Protocol-based extensibility for custom algorithm implementation
    - Type safety for enhanced IDE tooling and error prevention
    - Gymnasium 0.29.x API compliance with dual compatibility support
    
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
        
        Custom implementation with extensibility hooks:
        >>> class CustomNavigator:
        ...     def __init__(self):
        ...         self._positions = np.array([[0.0, 0.0]])
        ...         # ... implement all required properties and methods
        ...     
        ...     @property 
        ...     def positions(self) -> np.ndarray:
        ...         return self._positions
        ...         
        ...     def compute_additional_obs(self, base_obs: dict) -> dict:
        ...         return {"custom_sensor": self.sample_custom_sensor()}
        ...     # ... implement remaining protocol methods
        
        Factory method with Hydra integration:
        >>> from hydra import compose, initialize
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        ...     navigator = NavigatorFactory.from_config(cfg.navigator)
    """
    
    # Protocol-based dependency injection properties for v1.0 architecture
    
    @property 
    def source(self) -> Optional['SourceProtocol']:
        """
        Get current source implementation for odor emission modeling.
        
        Returns:
            Optional[SourceProtocol]: Source instance providing emission data and
                position information, or None if no source is configured.
                
        Notes:
            Source dependency enables flexible odor source modeling through
            protocol-based composition. Different source implementations can be
            injected at runtime based on experimental requirements.
            
            The source provides emission rate and position data that integrates
            with plume models for realistic concentration field generation.
            
        Examples:
            Source integration in navigation:
            >>> if navigator.source:
            ...     emission_rate = navigator.source.get_emission_rate()
            ...     source_pos = navigator.source.get_position()
            ...     # Use source data for navigation decisions
        """
        ...
    
    @property
    def boundary_policy(self) -> Optional['BoundaryPolicyProtocol']:
        """
        Get current boundary policy implementation for domain edge management.
        
        Returns:
            Optional[BoundaryPolicyProtocol]: Boundary policy instance providing
                edge handling behavior, or None if default behavior is used.
                
        Notes:
            Boundary policy dependency enables configurable domain edge behavior
            through protocol-based composition. Different policies can be injected
            to support various experimental scenarios and navigation strategies.
            
            The policy handles agent position corrections and episode termination
            decisions when agents interact with domain boundaries.
            
        Examples:
            Boundary policy integration:
            >>> if navigator.boundary_policy:
            ...     violations = navigator.boundary_policy.check_violations(positions)
            ...     if violations.any():
            ...         corrected_pos = navigator.boundary_policy.apply_policy(positions)
            ...         termination_status = navigator.boundary_policy.get_termination_status()
        """
        ...
    
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

    # New extensibility hooks for Gymnasium 0.29.x migration
    
    def compute_additional_obs(self, base_obs: dict) -> dict:
        """
        Compute additional observations for custom environment extensions.
        
        This extensibility hook allows downstream implementations to augment 
        the base observation with domain-specific data without modifying the 
        core navigation logic. Override this method to add custom sensors,
        derived metrics, or specialized data processing.
        
        Args:
            base_obs: Base observation dict containing standard navigation data.
                Typically includes 'position', 'orientation', 'speed', 'odor_concentration'.
                
        Returns:
            dict: Additional observation components to merge with base_obs.
                Keys should not conflict with base observation keys unless
                intentionally overriding default behavior.
                
        Notes:
            Default implementation returns empty dict (no additional observations).
            Implementations should maintain consistent key names and value types
            across episodes for stable RL training.
            
            Consider observation space implications when adding new keys -
            ensure the environment's observation space accounts for all data.
            
        Performance:
            Should execute in <1ms to maintain environment step time requirements.
            
        Examples:
            Custom sensor integration:
            >>> def compute_additional_obs(self, base_obs: dict) -> dict:
            ...     return {
            ...         "wind_direction": self.sample_wind_direction(),
            ...         "distance_to_wall": self.compute_wall_distance(),
            ...         "energy_level": self.get_energy_remaining()
            ...     }
            
            Derived metrics:
            >>> def compute_additional_obs(self, base_obs: dict) -> dict:
            ...     return {
            ...         "concentration_gradient": self.compute_gradient(base_obs),
            ...         "exploration_metric": self.compute_exploration_score()
            ...     }
        """
        return {}
    
    def compute_extra_reward(self, base_reward: float, info: dict) -> float:
        """
        Compute additional reward components for custom reward shaping.
        
        This extensibility hook enables reward shaping and custom reward
        function implementations without modifying core environment logic.
        Override this method to implement domain-specific incentives,
        exploration bonuses, or multi-objective reward functions.
        
        Args:
            base_reward: Base reward computed by the environment's standard
                reward function (typically odor-based navigation reward).
            info: Environment info dict containing episode state and metrics.
                May include trajectory history, performance stats, etc.
                
        Returns:
            float: Additional reward component to add to base_reward.
                Can be positive (bonus) or negative (penalty).
                
        Notes:
            Default implementation returns 0.0 (no additional reward).
            The final environment reward will be: base_reward + extra_reward.
            
            Be cautious with reward shaping to avoid unintended behavioral
            changes or reward hacking. Consider potential optimization impacts.
            
        Performance:
            Should execute in <0.5ms to maintain environment step time requirements.
            
        Examples:
            Exploration bonus:
            >>> def compute_extra_reward(self, base_reward: float, info: dict) -> float:
            ...     # Bonus for visiting unexplored areas
            ...     if self.is_novel_position(self.positions[-1]):
            ...         return 0.1
            ...     return 0.0
            
            Energy conservation penalty:
            >>> def compute_extra_reward(self, base_reward: float, info: dict) -> float:
            ...     # Penalty for high speed to encourage efficient movement
            ...     speed_penalty = -0.01 * np.mean(self.speeds ** 2)
            ...     return speed_penalty
            
            Multi-objective reward:
            >>> def compute_extra_reward(self, base_reward: float, info: dict) -> float:
            ...     exploration_bonus = 0.1 * info.get('exploration_score', 0)
            ...     efficiency_bonus = 0.05 * info.get('path_efficiency', 0)
            ...     return exploration_bonus + efficiency_bonus
        """
        return 0.0
    
    def on_episode_end(self, final_info: dict) -> None:
        """
        Handle episode completion events for logging and cleanup.
        
        This extensibility hook is called when an episode terminates or
        truncates, providing an opportunity for custom logging, metric
        computation, state persistence, or cleanup operations.
        
        Args:
            final_info: Final environment info dict containing episode
                summary data. May include trajectory statistics, performance
                metrics, termination reason, etc.
                
        Notes:
            Default implementation is a no-op (no special handling).
            This method should not modify navigator state as it may be
            called after reset() for the next episode.
            
            Use this hook for:
            - Custom logging and metric collection
            - Trajectory analysis and storage
            - Performance monitoring and alerting
            - Cleanup of episode-specific resources
            
            Avoid expensive operations that could impact environment
            throughput in training scenarios.
            
        Performance:
            Should execute in <5ms to avoid blocking episode transitions.
            
        Examples:
            Custom metric logging:
            >>> def on_episode_end(self, final_info: dict) -> None:
            ...     episode_length = final_info.get('episode_length', 0)
            ...     success_rate = final_info.get('success', False)
            ...     self.logger.info(f"Episode ended: length={episode_length}, success={success_rate}")
            
            Trajectory analysis:
            >>> def on_episode_end(self, final_info: dict) -> None:
            ...     trajectory = final_info.get('trajectory', [])
            ...     path_efficiency = self.analyze_path_efficiency(trajectory)
            ...     self.metrics_tracker.record('path_efficiency', path_efficiency)
            
            Performance monitoring:
            >>> def on_episode_end(self, final_info: dict) -> None:
            ...     avg_step_time = final_info.get('avg_step_time', 0)
            ...     if avg_step_time > 0.01:  # 10ms threshold
            ...         self.logger.warning(f"Slow episode: {avg_step_time:.3f}s per step")
        """
        pass

    # Optional memory interface for supporting both memory-based and non-memory-based navigation strategies
    
    def load_memory(self, memory_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Load agent memory state from external storage (optional interface).
        
        This optional method enables memory-based navigation strategies without
        enforcing memory usage on simpler agents. Implementations can choose to:
        - Implement full memory persistence and restoration
        - Provide partial memory support for specific data types
        - No-op for memory-less agents (default behavior)
        
        Args:
            memory_data: Optional dictionary containing serialized memory state.
                Structure depends on implementation but typically includes:
                - 'trajectory_history': List of past positions and actions
                - 'odor_concentration_history': Time series of sensor readings
                - 'spatial_map': Learned environment representation
                - 'internal_state': Algorithm-specific data structures
                
        Notes:
            Default implementation is a no-op (no memory functionality).
            This design enables the same simulator core to support both
            reactive (memory-less) and cognitive (memory-based) agents.
            
            Memory-based implementations should validate data consistency
            and handle missing or corrupted memory gracefully.
            
        Performance:
            Should complete in <10ms to avoid blocking episode initialization.
            
        Examples:
            Memory-less agent (default):
            >>> def load_memory(self, memory_data: Optional[Dict[str, Any]] = None) -> None:
            ...     pass  # No memory functionality needed
            
            Trajectory-based memory:
            >>> def load_memory(self, memory_data: Optional[Dict[str, Any]] = None) -> None:
            ...     if memory_data and 'trajectory_history' in memory_data:
            ...         self.trajectory_buffer = memory_data['trajectory_history']
            ...         self.visited_positions = set(memory_data.get('visited_positions', []))
            
            Full cognitive model:
            >>> def load_memory(self, memory_data: Optional[Dict[str, Any]] = None) -> None:
            ...     if memory_data:
            ...         self.spatial_map.load_state(memory_data.get('spatial_map', {}))
            ...         self.belief_state.restore(memory_data.get('belief_state', {}))
            ...         self.planning_horizon = memory_data.get('planning_horizon', 10)
        """
        pass
    
    def save_memory(self) -> Optional[Dict[str, Any]]:
        """
        Save current agent memory state for external storage (optional interface).
        
        This optional method enables memory-based navigation strategies to persist
        internal state between episodes or simulation runs. Implementations can:
        - Return comprehensive memory snapshots for full cognitive models
        - Export minimal state for lightweight memory systems
        - Return None for memory-less agents (default behavior)
        
        Returns:
            Optional[Dict[str, Any]]: Serializable memory state dictionary or None.
                If implemented, typical structure includes:
                - 'trajectory_history': List of past positions and actions
                - 'odor_concentration_history': Time series of sensor readings
                - 'spatial_map': Learned environment representation
                - 'internal_state': Algorithm-specific data structures
                - 'metadata': Timestamps, episode info, version tags
                
        Notes:
            Default implementation returns None (no memory functionality).
            This design enables the same simulator core to support both
            reactive (memory-less) and cognitive (memory-based) agents.
            
            Memory-based implementations should ensure all returned data
            is JSON-serializable for storage compatibility.
            
        Performance:
            Should complete in <10ms to avoid blocking episode transitions.
            
        Examples:
            Memory-less agent (default):
            >>> def save_memory(self) -> Optional[Dict[str, Any]]:
            ...     return None  # No memory to save
            
            Trajectory-based memory:
            >>> def save_memory(self) -> Optional[Dict[str, Any]]:
            ...     return {
            ...         'trajectory_history': self.trajectory_buffer[-1000:],  # Last 1000 steps
            ...         'visited_positions': list(self.visited_positions),
            ...         'episode_count': self.episode_counter
            ...     }
            
            Full cognitive model:
            >>> def save_memory(self) -> Optional[Dict[str, Any]]:
            ...     return {
            ...         'spatial_map': self.spatial_map.get_state(),
            ...         'belief_state': self.belief_state.serialize(),
            ...         'planning_horizon': self.planning_horizon,
            ...         'learned_parameters': self.policy_parameters.to_dict(),
            ...         'metadata': {'timestamp': time.time(), 'version': '1.0'}
            ...     }
        """
        return None


# New modular component protocols for pluggable architecture

@runtime_checkable
class SourceProtocol(Protocol):
    """
    Protocol defining the interface for pluggable odor source implementations.
    
    This protocol enables seamless switching between different odor source types:
    - PointSource: Single stationary emission point with configurable strength
    - MultiSource: Multiple concurrent sources with independent control  
    - DynamicSource: Time-varying sources with configurable emission patterns
    
    All implementations must provide real-time emission queries, position access,
    and temporal evolution while maintaining performance requirements for interactive simulation.
    
    Key Design Principles:
    - Emission queries via get_emission_rate() for plume model integration
    - Spatial access via get_position() for source-agent distance calculations
    - Temporal dynamics via update_state() for realistic source behavior
    - Configuration-driven instantiation for zero-code source selection
    
    Performance Requirements:
    - get_emission_rate(): <0.1ms for single query supporting real-time simulation
    - get_position(): <0.1ms for spatial queries with minimal overhead
    - update_state(): <1ms per time step for temporal evolution
    - Memory efficiency: <10MB for typical source configurations
    
    Examples:
        Basic point source:
        >>> source = PointSource(position=(50, 50), emission_rate=1000.0)
        >>> rate = source.get_emission_rate()
        >>> position = source.get_position()
        
        Dynamic source with temporal variation:
        >>> source = DynamicSource(
        ...     initial_position=(25, 75), 
        ...     emission_pattern="sinusoidal",
        ...     period=60.0
        ... )
        >>> for t in range(100):
        ...     source.update_state(dt=1.0)
        ...     current_rate = source.get_emission_rate()
        
        Multi-source configuration:
        >>> sources = MultiSource([
        ...     {'position': (20, 20), 'rate': 500.0},
        ...     {'position': (80, 80), 'rate': 750.0}
        ... ])
        >>> total_rate = sources.get_emission_rate()
    """
    
    def get_emission_rate(self) -> float:
        """
        Get current odor emission rate from this source.
        
        Returns:
            float: Emission rate in source-specific units (typically molecules/second
                or concentration units/second). Non-negative value representing
                current source strength.
                
        Notes:
            Emission rate may vary over time for dynamic sources based on internal
            temporal patterns or external control signals. Rate should remain
            physically realistic and consistent with source configuration.
            
            For multi-source implementations, returns the total aggregate emission
            rate across all active sub-sources.
            
        Performance:
            Must execute in <0.1ms for real-time simulation compatibility.
            
        Examples:
            Static source query:
            >>> rate = source.get_emission_rate()
            >>> assert rate >= 0.0, "Emission rate must be non-negative"
            
            Dynamic source monitoring:
            >>> rates = [source.get_emission_rate() for _ in range(100)]
            >>> assert all(r >= 0 for r in rates), "All rates must be valid"
        """
        ...
    
    def get_position(self) -> Tuple[float, float]:
        """
        Get current source position coordinates.
        
        Returns:
            Tuple[float, float]: Source position as (x, y) coordinates in
                environment coordinate system. Values should be within
                environment domain bounds.
                
        Notes:
            Position may be static for fixed sources or dynamic for moving sources.
            Coordinates follow environment convention (typically with origin at
            top-left for video-based environments).
            
            For multi-source implementations, returns the centroid or primary
            source position as appropriate for the source configuration.
            
        Performance:
            Must execute in <0.1ms for minimal spatial query overhead.
            
        Examples:
            Static source position:
            >>> x, y = source.get_position()
            >>> assert 0 <= x <= domain_width and 0 <= y <= domain_height
            
            Dynamic source tracking:
            >>> positions = [source.get_position() for _ in range(100)]
            >>> # Track source movement over time
        """
        ...
    
    def update_state(self, dt: float = 1.0) -> None:
        """
        Advance source state by specified time delta.
        
        Args:
            dt: Time step size in seconds. Controls temporal resolution of
                source dynamics including emission variations, position changes,
                and internal state evolution.
                
        Notes:
            Updates source internal state including:
            - Emission rate variations based on temporal patterns
            - Position changes for mobile sources
            - Internal parameters for complex source dynamics
            - Environmental interactions and external influences
            
            Static sources may have minimal state updates while dynamic
            sources implement complex temporal behaviors.
            
        Performance:
            Must complete in <1ms per step for real-time simulation compatibility.
            
        Examples:
            Standard time evolution:
            >>> source.update_state(dt=1.0)
            
            High-frequency dynamics:
            >>> for _ in range(10):
            ...     source.update_state(dt=0.1)  # 10x higher temporal resolution
            
            Variable time stepping:
            >>> source.update_state(dt=variable_dt)  # Adaptive time stepping
        """
        ...


@runtime_checkable
class BoundaryPolicyProtocol(Protocol):
    """
    Protocol defining configurable boundary handling strategies for domain edge management.
    
    This protocol enables flexible boundary behavior implementations:
    - TerminatePolicy: End episode when agent reaches boundary (status = "oob")
    - BouncePolicy: Reflect agent trajectory off boundary walls with energy conservation
    - WrapPolicy: Periodic boundary conditions wrapping to opposite domain edge
    - ClipPolicy: Constrain agent position to remain within valid domain
    
    All implementations must provide vectorized operations for multi-agent scenarios
    while maintaining performance requirements for real-time simulation.
    
    Key Design Principles:
    - Policy application via apply_policy() for position and velocity corrections
    - Violation detection via check_violations() for efficient boundary checking
    - Termination logic via get_termination_status() for episode management
    - Vectorized operations for scalable multi-agent boundary processing
    
    Performance Requirements:
    - apply_policy(): <1ms for 100 agents with vectorized operations
    - check_violations(): <0.5ms for boundary detection across all agents
    - get_termination_status(): <0.1ms for episode termination decisions
    - Memory efficiency: <1MB for boundary state management
    
    Examples:
        Episode termination boundary:
        >>> policy = TerminatePolicy(domain_bounds=(100, 100))
        >>> violations = policy.check_violations(agent_positions)
        >>> if violations.any():
        ...     status = policy.get_termination_status()
        
        Reflective boundary physics:
        >>> policy = BouncePolicy(domain_bounds=(100, 100), energy_loss=0.1)
        >>> corrected_pos, corrected_vel = policy.apply_policy(positions, velocities)
        
        Periodic domain wrapping:
        >>> policy = WrapPolicy(domain_bounds=(100, 100))
        >>> wrapped_positions = policy.apply_policy(positions, velocities)
    """
    
    def apply_policy(
        self, 
        positions: np.ndarray, 
        velocities: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply boundary policy to agent positions and optionally velocities.
        
        Args:
            positions: Agent positions as array with shape (n_agents, 2) for multiple
                agents or (2,) for single agent. Coordinates in environment units.
            velocities: Optional agent velocities with same shape as positions.
                Required for physics-based policies like bounce behavior.
                
        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
                - If velocities not provided: corrected positions array
                - If velocities provided: tuple of (corrected_positions, corrected_velocities)
                
        Notes:
            Policy application depends on boundary type:
            - Terminate: positions returned unchanged (termination handled separately)
            - Bounce: positions and velocities corrected for elastic/inelastic collisions
            - Wrap: positions wrapped to opposite boundary with velocity preservation
            - Clip: positions constrained to domain bounds with velocity zeroing
            
            Vectorized implementation processes all agents simultaneously for
            performance optimization in multi-agent scenarios.
            
        Performance:
            Must execute in <1ms for 100 agents with vectorized operations.
            
        Examples:
            Position-only correction (clip policy):
            >>> corrected_pos = policy.apply_policy(agent_positions)
            
            Physics-based correction (bounce policy):
            >>> corrected_pos, corrected_vel = policy.apply_policy(
            ...     agent_positions, agent_velocities
            ... )
            
            Multi-agent batch processing:
            >>> positions = np.array([[10, -5], [105, 50], [50, 110]])  # Some out of bounds
            >>> corrected = policy.apply_policy(positions)
        """
        ...
    
    def check_violations(self, positions: np.ndarray) -> np.ndarray:
        """
        Detect boundary violations for given agent positions.
        
        Args:
            positions: Agent positions as array with shape (n_agents, 2) for multiple
                agents or (2,) for single agent. Coordinates in environment units.
                
        Returns:
            np.ndarray: Boolean array with shape (n_agents,) or scalar bool for single
                agent. True indicates boundary violation requiring policy application.
                
        Notes:
            Violation detection is consistent across policy types but may trigger
            different responses based on policy implementation. Efficient vectorized
            implementation enables sub-millisecond checking for large agent populations.
            
            Violation criteria may include:
            - Position outside domain bounds (all policies)
            - Velocity pointing outward at boundary (bounce policy)
            - Distance from boundary below threshold (predictive policies)
            
        Performance:
            Must execute in <0.5ms for boundary detection across 100 agents.
            
        Examples:
            Single agent violation check:
            >>> position = np.array([105, 50])  # Outside domain
            >>> violated = policy.check_violations(position)
            >>> assert violated == True
            
            Multi-agent batch checking:
            >>> positions = np.array([[50, 50], [105, 25], [25, 105]])
            >>> violations = policy.check_violations(positions)
            >>> # Returns [False, True, True] for domain bounds (100, 100)
            
            Performance monitoring:
            >>> start_time = time.time()
            >>> violations = policy.check_violations(large_position_array)
            >>> assert (time.time() - start_time) < 0.0005  # <0.5ms requirement
        """
        ...
    
    def get_termination_status(self) -> str:
        """
        Get episode termination status for boundary policy.
        
        Returns:
            str: Termination status string indicating boundary policy behavior.
                Common values:
                - "oob": Out of bounds termination (TerminatePolicy)
                - "continue": Episode continues with correction (BouncePolicy, WrapPolicy, ClipPolicy)
                - "boundary_contact": Boundary interaction without termination
                - Policy-specific status codes for specialized behaviors
                
        Notes:
            Termination status provides semantic information about boundary
            interactions for episode management and data analysis. Status codes
            are consistent within policy types but may vary between implementations.
            
            Non-terminating policies return "continue" to indicate normal episode
            progression with boundary corrections applied.
            
        Performance:
            Must execute in <0.1ms for immediate episode management decisions.
            
        Examples:
            Termination boundary policy:
            >>> policy = TerminatePolicy(domain_bounds=(100, 100))
            >>> status = policy.get_termination_status()
            >>> assert status == "oob"
            
            Reflection boundary policy:
            >>> policy = BouncePolicy(domain_bounds=(100, 100))
            >>> status = policy.get_termination_status()
            >>> assert status == "continue"
            
            Status-based episode management:
            >>> status = policy.get_termination_status()
            >>> episode_done = (status == "oob")
        """
        ...


@runtime_checkable
class ActionInterfaceProtocol(Protocol):
    """
    Protocol defining standardized action space translation for RL framework integration.
    
    This protocol enables unified action handling across different control paradigms:
    - Continuous2D: Continuous velocity/acceleration control with bounded action spaces
    - CardinalDiscrete: Discrete directional commands (N, S, E, W, NE, NW, SE, SW, Stop)
    - Custom implementations supporting specialized control schemes
    
    All implementations must provide action validation, space translation, and efficient
    processing while maintaining compatibility with RL frameworks and navigation controllers.
    
    Key Design Principles:
    - Action translation via translate_action() for RL-to-navigation command conversion
    - Validation via validate_action() for constraint enforcement and safety
    - Space definition via get_action_space() for Gymnasium compatibility
    - Framework agnostic design supporting multiple RL libraries
    
    Performance Requirements:
    - translate_action(): <0.1ms per agent for minimal control overhead
    - validate_action(): <0.05ms per action for constraint checking
    - get_action_space(): <1ms for space construction (called infrequently)
    - Memory efficiency: <100 bytes per action for structured representations
    
    Examples:
        Continuous velocity control:
        >>> action_interface = Continuous2DAction(
        ...     max_linear_velocity=2.0, 
        ...     max_angular_velocity=45.0
        ... )
        >>> rl_action = np.array([0.8, -0.3])  # Normalized action
        >>> nav_command = action_interface.translate_action(rl_action)
        
        Discrete directional control:
        >>> action_interface = CardinalDiscreteAction()
        >>> rl_action = 3  # Discrete action index
        >>> nav_command = action_interface.translate_action(rl_action)
        
        Action space for RL training:
        >>> action_space = action_interface.get_action_space()
        >>> assert isinstance(action_space, gymnasium.spaces.Space)
    """
    
    def translate_action(
        self, 
        action: Union[np.ndarray, int, float, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Translate RL framework action to navigation controller commands.
        
        Args:
            action: RL action in framework-specific format. Supported types:
                - np.ndarray: Continuous control vectors (velocity, acceleration)
                - int: Discrete action indices for directional commands
                - float: Scalar actions for 1D control
                - Dict[str, Any]: Structured actions for complex control schemes
                
        Returns:
            Dict[str, Any]: Navigation command dictionary with standardized keys:
                - 'linear_velocity': Target linear velocity (float)
                - 'angular_velocity': Target angular velocity (float)
                - 'action_type': Control scheme identifier (str)
                - Additional keys for specialized control modes
                
        Notes:
            Translation handles scaling, bounds checking, and coordinate transformations
            between RL action spaces and navigation controller interfaces. Output
            format is consistent across action interface implementations.
            
            Continuous actions are typically normalized to [-1, 1] range in RL
            frameworks and scaled to physical units during translation.
            
        Performance:
            Must execute in <0.1ms per agent for minimal control overhead.
            
        Examples:
            Continuous action translation:
            >>> rl_action = np.array([0.5, -0.2])  # [linear_vel_norm, angular_vel_norm]
            >>> command = action_interface.translate_action(rl_action)
            >>> assert 'linear_velocity' in command
            >>> assert 'angular_velocity' in command
            
            Discrete action translation:
            >>> rl_action = 2  # East direction
            >>> command = action_interface.translate_action(rl_action)
            >>> assert command['linear_velocity'] > 0
            >>> assert abs(command['angular_velocity']) < 1e-6  # Straight movement
            
            Structured action translation:
            >>> rl_action = {'target_velocity': 1.5, 'turn_rate': 10.0}
            >>> command = action_interface.translate_action(rl_action)
        """
        ...
    
    def validate_action(
        self, 
        action: Union[np.ndarray, int, float, Dict[str, Any]]
    ) -> bool:
        """
        Validate action compliance with interface constraints and safety limits.
        
        Args:
            action: RL action to validate in same format as translate_action().
                
        Returns:
            bool: True if action is valid and safe to execute, False otherwise.
                
        Notes:
            Validation includes:
            - Bounds checking for continuous actions (within normalized ranges)
            - Index validation for discrete actions (valid action indices)
            - Safety constraints (maximum velocities, acceleration limits)
            - Type checking and format validation
            
            Invalid actions should be rejected before translation to prevent
            unsafe navigation commands or system errors.
            
        Performance:
            Must execute in <0.05ms per action for constraint checking.
            
        Examples:
            Continuous action validation:
            >>> rl_action = np.array([0.5, -0.2])  # Valid normalized action
            >>> assert action_interface.validate_action(rl_action) == True
            >>> 
            >>> invalid_action = np.array([2.0, -3.0])  # Out of bounds
            >>> assert action_interface.validate_action(invalid_action) == False
            
            Discrete action validation:
            >>> valid_action = 5  # Valid direction index
            >>> assert action_interface.validate_action(valid_action) == True
            >>> 
            >>> invalid_action = 15  # Invalid index
            >>> assert action_interface.validate_action(invalid_action) == False
            
            Batch validation:
            >>> actions = [np.array([0.1, 0.2]), np.array([1.5, 0.0])]
            >>> validity = [action_interface.validate_action(a) for a in actions]
            >>> # Returns [True, False] for valid and invalid actions
        """
        ...
    
    def get_action_space(self) -> Optional['spaces.Space']:
        """
        Construct Gymnasium action space definition for RL framework integration.
        
        Returns:
            Optional[spaces.Space]: Gymnasium action space defining valid action
                structure and value ranges. Returns None if Gymnasium is not available.
                Common space types:
                - spaces.Box: Continuous control with bounded ranges
                - spaces.Discrete: Discrete action indices  
                - spaces.Dict: Structured action dictionaries
                - spaces.MultiDiscrete: Multiple discrete action dimensions
                
        Notes:
            Action space automatically reflects interface configuration including
            bounds, discrete action counts, and control modality constraints.
            
            Space definition must be consistent with translate_action() and
            validate_action() methods for proper RL framework integration.
            
        Performance:
            Must execute in <1ms for space construction (called infrequently).
            
        Examples:
            Continuous control space:
            >>> action_space = action_interface.get_action_space()
            >>> assert isinstance(action_space, gymnasium.spaces.Box)
            >>> assert action_space.shape == (2,)  # [linear_vel, angular_vel]
            >>> assert np.all(action_space.low == -1.0)
            >>> assert np.all(action_space.high == 1.0)
            
            Discrete control space:
            >>> action_space = action_interface.get_action_space()
            >>> assert isinstance(action_space, gymnasium.spaces.Discrete)
            >>> assert action_space.n == 9  # 8 directions + stop
            
            Space-action consistency:
            >>> action_space = action_interface.get_action_space()
            >>> sample_action = action_space.sample()
            >>> assert action_interface.validate_action(sample_action) == True
        """
        ...


@runtime_checkable
class RecorderProtocol(Protocol):
    """
    Protocol defining comprehensive data recording interfaces for experiment persistence.
    
    This protocol enables configurable data collection with multiple storage backends:
    - ParquetBackend: Columnar storage with compression for large trajectory datasets
    - HDF5Backend: Hierarchical scientific data format with metadata support
    - SQLiteBackend: Embedded database for structured queries and analysis
    - NoneBackend: Disabled recording for performance-critical scenarios
    
    All implementations must provide buffered I/O, compression options, and structured
    output organization while maintaining minimal performance impact on simulation.
    
    Key Design Principles:
    - Step recording via record_step() for detailed trajectory capture
    - Episode recording via record_episode() for summary data persistence
    - Data export via export_data() for analysis and visualization
    - Performance-aware buffering with configurable granularity
    
    Performance Requirements:
    - record_step(): <0.1ms overhead when recording disabled, <1ms when enabled
    - record_episode(): <10ms for episode finalization and metadata storage
    - export_data(): <100ms for typical dataset export with compression
    - Memory efficiency: Configurable buffering with backpressure handling
    
    Examples:
        High-frequency step recording:
        >>> recorder = ParquetRecorder(
        ...     output_dir="./data", 
        ...     buffer_size=1000,
        ...     compression="snappy"
        ... )
        >>> for step in range(episode_length):
        ...     state_data = {'position': agent_pos, 'concentration': odor_level}
        ...     recorder.record_step(state_data, step_number=step)
        
        Episode-level data collection:
        >>> episode_data = {
        ...     'total_steps': 250,
        ...     'success': True,
        ...     'final_position': (85, 92),
        ...     'path_efficiency': 0.75
        ... }
        >>> recorder.record_episode(episode_data, episode_id=42)
        
        Data export for analysis:
        >>> recorder.export_data(
        ...     format="parquet", 
        ...     compression="gzip",
        ...     output_file="experiment_results.parquet"
        ... )
    """
    
    def record_step(
        self, 
        step_data: Dict[str, Any], 
        step_number: int,
        episode_id: Optional[int] = None,
        **metadata: Any
    ) -> None:
        """
        Record simulation state data for a single time step.
        
        Args:
            step_data: Dictionary containing step-level measurements and state.
                Common keys include:
                - 'position': Agent position coordinates
                - 'velocity': Agent velocity vector
                - 'concentration': Sampled odor concentration
                - 'action': Applied navigation command
                - 'reward': Step reward value
                - Additional domain-specific measurements
            step_number: Sequential step index within current episode (0-based).
            episode_id: Optional episode identifier for data organization.
            **metadata: Additional metadata for step context and debugging.
                
        Notes:
            Step recording provides detailed trajectory capture for analysis and
            visualization. Data is buffered for performance and flushed based on
            buffer size and timing configurations.
            
            Recording granularity is configurable - high-frequency recording
            captures every simulation step while reduced frequency captures
            periodic snapshots for memory efficiency.
            
        Performance:
            Must execute in <0.1ms when recording disabled, <1ms when enabled.
            
        Examples:
            Basic step recording:
            >>> step_data = {
            ...     'position': np.array([45.2, 78.1]),
            ...     'concentration': 0.23,
            ...     'action': {'linear_velocity': 1.5, 'angular_velocity': 0.0}
            ... }
            >>> recorder.record_step(step_data, step_number=125)
            
            Multi-agent step recording:
            >>> step_data = {
            ...     'positions': np.array([[45, 78], [52, 81], [49, 75]]),
            ...     'concentrations': np.array([0.23, 0.31, 0.18]),
            ...     'rewards': np.array([0.1, 0.2, 0.05])
            ... }
            >>> recorder.record_step(step_data, step_number=125, episode_id=42)
            
            Performance monitoring:
            >>> import time
            >>> start = time.time()
            >>> recorder.record_step(step_data, step_number=125)
            >>> duration = time.time() - start
            >>> assert duration < 0.001  # <1ms requirement
        """
        ...
    
    def record_episode(
        self, 
        episode_data: Dict[str, Any], 
        episode_id: int,
        **metadata: Any
    ) -> None:
        """
        Record episode-level summary data and metrics.
        
        Args:
            episode_data: Dictionary containing episode summary information.
                Common keys include:
                - 'total_steps': Episode length in simulation steps
                - 'success': Boolean success indicator
                - 'final_position': Agent position at episode end
                - 'total_reward': Cumulative episode reward
                - 'path_efficiency': Navigation performance metric
                - 'exploration_coverage': Spatial coverage measure
                - Additional domain-specific episode metrics
            episode_id: Unique episode identifier for data organization.
            **metadata: Additional metadata for episode context and configuration.
                
        Notes:
            Episode recording provides summary statistics and metadata for
            experimental analysis and comparison. Data includes both computed
            metrics and configuration snapshots for reproducibility.
            
            Episode finalization may trigger buffer flushing and file system
            organization based on recorder configuration.
            
        Performance:
            Must execute in <10ms for episode finalization and metadata storage.
            
        Examples:
            Successful episode recording:
            >>> episode_data = {
            ...     'total_steps': 245,
            ...     'success': True,
            ...     'final_position': (87.3, 91.8),
            ...     'total_reward': 12.4,
            ...     'path_efficiency': 0.78
            ... }
            >>> recorder.record_episode(episode_data, episode_id=42)
            
            Failed episode with diagnostics:
            >>> episode_data = {
            ...     'total_steps': 500,  # Max steps reached
            ...     'success': False,
            ...     'final_position': (23.1, 45.7),
            ...     'total_reward': -2.1,
            ...     'termination_reason': 'timeout'
            ... }
            >>> recorder.record_episode(episode_data, episode_id=43)
            
            Episode with configuration snapshot:
            >>> recorder.record_episode(
            ...     episode_data, episode_id=44,
            ...     config_snapshot=current_config,
            ...     random_seed=12345
            ... )
        """
        ...
    
    def export_data(
        self, 
        output_path: str,
        format: str = "parquet",
        compression: Optional[str] = None,
        filter_episodes: Optional[List[int]] = None,
        **export_options: Any
    ) -> bool:
        """
        Export recorded data to specified file format and location.
        
        Args:
            output_path: File system path for exported data output.
            format: Export format specification. Supported formats:
                - "parquet": Columnar format with excellent compression
                - "hdf5": Hierarchical scientific data format
                - "csv": Human-readable comma-separated values
                - "json": JSON format for structured data exchange
            compression: Optional compression method (format-specific).
                - Parquet: "snappy", "gzip", "brotli", "lz4"
                - HDF5: "gzip", "lzf", "szip"
            filter_episodes: Optional list of episode IDs to export (default: all).
            **export_options: Additional format-specific export parameters.
                
        Returns:
            bool: True if export completed successfully, False otherwise.
                
        Notes:
            Export operation consolidates buffered data and generates output files
            with appropriate compression and metadata. Large datasets may require
            streaming export to manage memory usage.
            
            Export includes both step-level trajectory data and episode-level
            summary information organized for analysis workflows.
            
        Performance:
            Must execute in <100ms for typical dataset export with compression.
            
        Examples:
            Parquet export with compression:
            >>> success = recorder.export_data(
            ...     output_path="./results/experiment_001.parquet",
            ...     format="parquet",
            ...     compression="snappy"
            ... )
            >>> assert success == True
            
            Filtered episode export:
            >>> success = recorder.export_data(
            ...     output_path="./results/successful_episodes.csv",
            ...     format="csv",
            ...     filter_episodes=[42, 47, 51, 63]
            ... )
            
            HDF5 export with metadata:
            >>> success = recorder.export_data(
            ...     output_path="./results/full_dataset.h5",
            ...     format="hdf5",
            ...     compression="gzip",
            ...     include_metadata=True,
            ...     chunking_strategy="auto"
            ... )
        """
        ...


@runtime_checkable
class StatsAggregatorProtocol(Protocol):
    """
    Protocol defining automated statistics collection for research-focused metrics.
    
    This protocol enables standardized analysis and summary generation:
    - Episode-level metrics calculation with statistical measures
    - Run-level aggregation across multiple episodes for comparative analysis  
    - Automated summary export in standardized formats for publication
    - Custom metric definitions and calculation frameworks
    
    All implementations must provide efficient statistical computation and structured
    output generation while maintaining research reproducibility and comparison standards.
    
    Key Design Principles:
    - Episode analysis via calculate_episode_stats() for detailed performance metrics
    - Run aggregation via calculate_run_stats() for comparative study support
    - Summary export via export_summary() for standardized research reporting
    - Extensible metric definitions for domain-specific analysis requirements
    
    Performance Requirements:
    - calculate_episode_stats(): <10ms for episode-level metric computation
    - calculate_run_stats(): <100ms for multi-episode aggregation analysis
    - export_summary(): <50ms for summary generation and file output
    - Memory efficiency: <50MB for typical experimental dataset analysis
    
    Examples:
        Episode performance analysis:
        >>> aggregator = StandardStatsAggregator()
        >>> episode_metrics = aggregator.calculate_episode_stats(
        ...     trajectory_data=episode_trajectories,
        ...     episode_id=42
        ... )
        >>> assert 'path_efficiency' in episode_metrics
        >>> assert 'exploration_coverage' in episode_metrics
        
        Multi-episode comparative analysis:
        >>> run_metrics = aggregator.calculate_run_stats(
        ...     episode_data_list=all_episodes,
        ...     run_id="experiment_001"
        ... )
        >>> assert 'success_rate' in run_metrics
        >>> assert 'mean_path_efficiency' in run_metrics
        
        Research summary generation:
        >>> aggregator.export_summary(
        ...     output_path="./results/summary.json",
        ...     include_distributions=True,
        ...     statistical_tests=["t_test", "anova"]
        ... )
    """
    
    def calculate_episode_stats(
        self, 
        trajectory_data: Dict[str, Any],
        episode_id: int,
        custom_metrics: Optional[Dict[str, callable]] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive statistics for a single episode.
        
        Args:
            trajectory_data: Dictionary containing episode trajectory information.
                Expected keys include:
                - 'positions': Agent position time series
                - 'concentrations': Odor concentration measurements
                - 'actions': Applied navigation commands
                - 'rewards': Step-wise reward values
                - 'timestamps': Temporal information
                - Additional domain-specific trajectory data
            episode_id: Unique episode identifier for metric correlation.
            custom_metrics: Optional dictionary of custom metric calculation functions.
                
        Returns:
            Dict[str, float]: Dictionary of calculated episode-level metrics including:
                - 'path_efficiency': Ratio of direct distance to actual path length
                - 'exploration_coverage': Fraction of domain area explored
                - 'mean_concentration': Average odor concentration encountered
                - 'success_indicator': Binary success metric (1.0 if successful)
                - 'total_reward': Cumulative episode reward
                - 'episode_length': Number of simulation steps
                - Additional computed and custom metrics
                
        Notes:
            Metric calculation uses standard statistical methods and domain-specific
            algorithms appropriate for navigation analysis. Custom metrics enable
            specialized analysis for research-specific requirements.
            
            All metrics are computed as floating-point values for consistent
            analysis and comparison across episodes and experimental conditions.
            
        Performance:
            Must execute in <10ms for episode-level metric computation.
            
        Examples:
            Standard episode analysis:
            >>> trajectory_data = {
            ...     'positions': position_time_series,
            ...     'concentrations': concentration_measurements,
            ...     'actions': action_sequence,
            ...     'rewards': reward_time_series
            ... }
            >>> metrics = aggregator.calculate_episode_stats(trajectory_data, episode_id=42)
            >>> print(f"Path efficiency: {metrics['path_efficiency']:.3f}")
            >>> print(f"Success: {bool(metrics['success_indicator'])}")
            
            Custom metric integration:
            >>> def custom_tortuosity(trajectory_data):
            ...     positions = trajectory_data['positions']
            ...     # Calculate path tortuosity metric
            ...     return computed_tortuosity
            >>> 
            >>> custom_metrics = {'tortuosity': custom_tortuosity}
            >>> metrics = aggregator.calculate_episode_stats(
            ...     trajectory_data, episode_id=42, custom_metrics=custom_metrics
            ... )
            >>> assert 'tortuosity' in metrics
            
            Performance validation:
            >>> import time
            >>> start = time.time()
            >>> metrics = aggregator.calculate_episode_stats(trajectory_data, episode_id=42)
            >>> duration = time.time() - start
            >>> assert duration < 0.01  # <10ms requirement
        """
        ...
    
    def calculate_run_stats(
        self, 
        episode_data_list: List[Dict[str, Any]],
        run_id: str,
        statistical_tests: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate aggregate statistics across multiple episodes for run-level analysis.
        
        Args:
            episode_data_list: List of episode data dictionaries from calculate_episode_stats().
                Each dictionary contains episode-level metrics and metadata.
            run_id: Unique run identifier for experimental tracking and comparison.
            statistical_tests: Optional list of statistical tests to perform.
                Supported tests: ["t_test", "anova", "ks_test", "wilcoxon"]
                
        Returns:
            Dict[str, float]: Dictionary of run-level aggregate metrics including:
                - 'success_rate': Fraction of successful episodes
                - 'mean_path_efficiency': Average path efficiency across episodes
                - 'std_path_efficiency': Standard deviation of path efficiency
                - 'mean_episode_length': Average episode duration
                - 'total_episodes': Number of episodes in run
                - 'confidence_intervals': Statistical confidence bounds (if requested)
                - Additional aggregated metrics and statistical test results
                
        Notes:
            Run-level analysis provides statistical summary across episode populations
            for experimental comparison and hypothesis testing. Statistical tests
            enable rigorous analysis of experimental differences and significance.
            
            Aggregate metrics include central tendency, variability, and distribution
            characteristics appropriate for research publication and comparison.
            
        Performance:
            Must execute in <100ms for multi-episode aggregation analysis.
            
        Examples:
            Standard run analysis:
            >>> episode_data_list = [episode_metrics_1, episode_metrics_2, ...]
            >>> run_metrics = aggregator.calculate_run_stats(
            ...     episode_data_list, run_id="experiment_001"
            ... )
            >>> print(f"Success rate: {run_metrics['success_rate']:.2%}")
            >>> print(f"Mean efficiency: {run_metrics['mean_path_efficiency']:.3f}")
            
            Statistical hypothesis testing:
            >>> run_metrics = aggregator.calculate_run_stats(
            ...     episode_data_list, 
            ...     run_id="experiment_001",
            ...     statistical_tests=["t_test", "anova"]
            ... )
            >>> assert 't_test_p_value' in run_metrics
            >>> assert 'anova_f_statistic' in run_metrics
            
            Comparative analysis:
            >>> control_metrics = aggregator.calculate_run_stats(control_episodes, "control")
            >>> treatment_metrics = aggregator.calculate_run_stats(treatment_episodes, "treatment")
            >>> improvement = (treatment_metrics['success_rate'] - 
            ...               control_metrics['success_rate']) / control_metrics['success_rate']
            >>> print(f"Performance improvement: {improvement:.1%}")
        """
        ...
    
    def export_summary(
        self, 
        output_path: str,
        run_data: Optional[Dict[str, Any]] = None,
        include_distributions: bool = False,
        format: str = "json"
    ) -> bool:
        """
        Generate and export standardized summary report for research publication.
        
        Args:
            output_path: File system path for summary report output.
            run_data: Optional run-level data from calculate_run_stats() for inclusion.
            include_distributions: Include distribution plots and histograms in summary.
            format: Output format specification ("json", "yaml", "markdown", "latex").
                
        Returns:
            bool: True if summary export completed successfully, False otherwise.
                
        Notes:
            Summary export generates publication-ready reports with standardized
            metrics, statistical analysis, and optional visualizations. Output
            format supports research workflows and publication requirements.
            
            Summary includes experiment configuration, statistical results, and
            performance metrics organized for clear presentation and comparison.
            
        Performance:
            Must execute in <50ms for summary generation and file output.
            
        Examples:
            JSON summary export:
            >>> success = aggregator.export_summary(
            ...     output_path="./results/experiment_summary.json",
            ...     run_data=run_metrics,
            ...     include_distributions=False
            ... )
            >>> assert success == True
            
            Markdown report with visualizations:
            >>> success = aggregator.export_summary(
            ...     output_path="./results/experiment_report.md",
            ...     run_data=run_metrics,
            ...     include_distributions=True,
            ...     format="markdown"
            ... )
            
            LaTeX summary for publication:
            >>> success = aggregator.export_summary(
            ...     output_path="./results/paper_summary.tex",
            ...     run_data=run_metrics,
            ...     format="latex",
            ...     citation_style="ieee"
            ... )
        """
        ...


@runtime_checkable
class AgentInitializerProtocol(Protocol):
    """
    Protocol defining configurable agent initialization strategies for diverse experimental setups.
    
    This protocol enables flexible starting position generation patterns:
    - UniformRandomInitializer: Random positions with uniform spatial distribution
    - GridInitializer: Regular grid patterns for systematic spatial coverage
    - FixedListInitializer: Predetermined position lists for reproducible experiments
    - DatasetInitializer: Position loading from experimental datasets or files
    
    All implementations must provide deterministic seeding, domain validation, and
    efficient position generation while supporting both single and multi-agent scenarios.
    
    Key Design Principles:
    - Position generation via initialize_positions() for flexible starting configurations
    - Domain validation via validate_domain() for spatial constraint enforcement
    - State management via reset() for deterministic experiment reproducibility
    - Strategy identification via get_strategy_name() for experimental tracking
    
    Performance Requirements:
    - initialize_positions(): <5ms for 100 agents with spatial distribution algorithms
    - validate_domain(): <1ms for position constraint checking and validation
    - reset(): <1ms for strategy state reset with deterministic seeding
    - Memory efficiency: <10MB for position generation and validation state
    
    Examples:
        Random spatial distribution:
        >>> initializer = UniformRandomInitializer(
        ...     domain_bounds=(100, 100),
        ...     seed=42
        ... )
        >>> positions = initializer.initialize_positions(num_agents=50)
        >>> assert positions.shape == (50, 2)
        
        Regular grid arrangement:
        >>> initializer = GridInitializer(
        ...     domain_bounds=(100, 100),
        ...     grid_spacing=10.0
        ... )
        >>> positions = initializer.initialize_positions(num_agents=25)
        
        Reproducible experiment setup:
        >>> initializer.reset(seed=12345)
        >>> positions_1 = initializer.initialize_positions(num_agents=10)
        >>> initializer.reset(seed=12345)
        >>> positions_2 = initializer.initialize_positions(num_agents=10)
        >>> assert np.allclose(positions_1, positions_2)  # Deterministic reproduction
    """
    
    def initialize_positions(
        self, 
        num_agents: int,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Generate initial agent positions based on configured strategy.
        
        Args:
            num_agents: Number of agent positions to generate (must be positive).
            **kwargs: Additional strategy-specific parameters for position generation.
                Common parameters include:
                - exclusion_zones: List of spatial regions to avoid
                - clustering_factor: Spatial clustering strength for grouped initialization
                - minimum_distance: Minimum separation between agent positions
                - boundary_margin: Margin from domain edges for position placement
                
        Returns:
            np.ndarray: Agent positions as array with shape (num_agents, 2) containing
                [x, y] coordinates in environment coordinate system. All positions
                guaranteed to be within domain bounds and satisfy strategy constraints.
                
        Notes:
            Position generation follows strategy-specific algorithms with deterministic
            behavior based on internal random state. Validation ensures all generated
            positions comply with domain constraints and strategy requirements.
            
            Multi-agent scenarios may include collision avoidance and spatial
            distribution optimization for realistic initialization patterns.
            
        Performance:
            Must execute in <5ms for 100 agents with spatial distribution algorithms.
            
        Examples:
            Basic position generation:
            >>> positions = initializer.initialize_positions(num_agents=25)
            >>> assert positions.shape == (25, 2)
            >>> assert np.all(positions >= 0)  # Within domain bounds
            >>> assert np.all(positions[:, 0] <= domain_width)
            >>> assert np.all(positions[:, 1] <= domain_height)
            
            Constrained initialization:
            >>> exclusion_zones = [{'center': (50, 50), 'radius': 15}]
            >>> positions = initializer.initialize_positions(
            ...     num_agents=20,
            ...     exclusion_zones=exclusion_zones,
            ...     minimum_distance=5.0
            ... )
            
            Performance validation:
            >>> import time
            >>> start = time.time()
            >>> positions = initializer.initialize_positions(num_agents=100)
            >>> duration = time.time() - start
            >>> assert duration < 0.005  # <5ms requirement
        """
        ...
    
    def validate_domain(
        self, 
        positions: np.ndarray,
        domain_bounds: Tuple[float, float]
    ) -> bool:
        """
        Validate that positions comply with domain constraints and strategy requirements.
        
        Args:
            positions: Agent positions to validate as array with shape (n_agents, 2).
            domain_bounds: Spatial domain limits as (width, height) tuple.
                
        Returns:
            bool: True if all positions are valid and compliant, False otherwise.
                
        Notes:
            Validation includes:
            - Boundary checking for domain compliance
            - Strategy-specific constraint verification
            - Collision detection and minimum distance requirements
            - Exclusion zone compliance and spatial restrictions
            
            Validation provides diagnostic feedback for position generation
            debugging and constraint satisfaction verification.
            
        Performance:
            Must execute in <1ms for position constraint checking and validation.
            
        Examples:
            Domain boundary validation:
            >>> positions = np.array([[25, 30], [75, 80], [10, 90]])
            >>> domain_bounds = (100, 100)
            >>> is_valid = initializer.validate_domain(positions, domain_bounds)
            >>> assert is_valid == True
            
            Invalid position detection:
            >>> invalid_positions = np.array([[150, 50], [50, 150]])  # Out of bounds
            >>> is_valid = initializer.validate_domain(invalid_positions, domain_bounds)
            >>> assert is_valid == False
            
            Strategy constraint validation:
            >>> grid_initializer = GridInitializer(grid_spacing=20.0)
            >>> positions = grid_initializer.initialize_positions(num_agents=9)
            >>> is_valid = grid_initializer.validate_domain(positions, domain_bounds)
            >>> assert is_valid == True  # Grid spacing satisfied
        """
        ...
    
    def reset(self, seed: Optional[int] = None, **kwargs: Any) -> None:
        """
        Reset initializer state for deterministic position generation.
        
        Args:
            seed: Optional random seed for deterministic behavior reproduction.
            **kwargs: Additional reset parameters for strategy-specific state.
                
        Notes:
            Reset operation reinitializes internal random number generators and
            strategy-specific state for reproducible experiment conditions.
            
            Deterministic seeding enables exact reproduction of initialization
            patterns for scientific reproducibility and debugging workflows.
            
        Performance:
            Must execute in <1ms for strategy state reset with deterministic seeding.
            
        Examples:
            Deterministic reset:
            >>> initializer.reset(seed=42)
            >>> positions_1 = initializer.initialize_positions(num_agents=10)
            >>> initializer.reset(seed=42)
            >>> positions_2 = initializer.initialize_positions(num_agents=10)
            >>> assert np.array_equal(positions_1, positions_2)
            
            Strategy state reset:
            >>> grid_initializer.reset(grid_origin=(10, 10), grid_spacing=15.0)
            >>> positions = grid_initializer.initialize_positions(num_agents=16)
            
            Random state validation:
            >>> import time
            >>> start = time.time()
            >>> initializer.reset(seed=123)
            >>> duration = time.time() - start
            >>> assert duration < 0.001  # <1ms requirement
        """
        ...
    
    def get_strategy_name(self) -> str:
        """
        Get human-readable strategy name for experimental tracking and logger.
        
        Returns:
            str: Strategy identifier string for documentation and analysis.
                Common names include:
                - "uniform_random": Uniform spatial distribution
                - "grid": Regular grid pattern
                - "fixed_list": Predetermined position list
                - "dataset": External dataset loading
                - Strategy-specific identifiers for custom implementations
                
        Notes:
            Strategy names provide consistent identification for experimental
            tracking, configuration documentation, and analysis workflows.
            
            Names should be descriptive and unique within the initialization
            framework for clear experimental record keeping.
            
        Examples:
            Strategy identification:
            >>> strategy_name = initializer.get_strategy_name()
            >>> assert isinstance(strategy_name, str)
            >>> assert len(strategy_name) > 0
            
            Experimental logging:
            >>> experiment_config = {
            ...     'initialization_strategy': initializer.get_strategy_name(),
            ...     'num_agents': 25,
            ...     'domain_bounds': (100, 100)
            ... }
            >>> log_experiment_setup(experiment_config)
            
            Strategy comparison:
            >>> strategies = [init1.get_strategy_name(), init2.get_strategy_name()]
            >>> assert strategies[0] != strategies[1]  # Different strategies
        """
        ...


@runtime_checkable
class AgentObservationProtocol(Protocol):
    """
    Protocol defining standardized observation structures for agent-environment interaction.
    
    This protocol enables flexible observation space construction that automatically adapts
    to active sensor configurations while maintaining type safety and Gymnasium compatibility.
    
    AgentObservationProtocol implementations integrate with SensorProtocol components to
    provide structured observation dictionaries with consistent naming conventions and
    data formats across different sensing modalities.
    
    Key Design Principles:
    - Structured observations via observation dictionaries with standardized keys
    - Dynamic observation space adaptation based on active sensor configuration
    - Type safety for RL framework integration and debugging support
    - Extensible design for custom observation components and derived metrics
    
    Performance Requirements:
    - Observation construction: <0.5ms per agent for real-time step execution
    - Memory efficiency: <1KB per agent for structured observation data
    - Gymnasium space compatibility with automatic space inference
    
    Examples:
        Basic observation construction:
        >>> obs_protocol = AgentObservationProtocol(sensors=[binary_sensor, conc_sensor])
        >>> observations = obs_protocol.construct_observation(agent_state, plume_state)
        
        Dynamic space inference:
        >>> obs_space = obs_protocol.get_observation_space()
        >>> assert isinstance(obs_space, gymnasium.spaces.Dict)
        
        Custom observation extension:
        >>> class CustomObservation(AgentObservationProtocol):
        ...     def add_custom_data(self, obs_dict, agent_state):
        ...         obs_dict['energy_level'] = agent_state.energy
        ...         return obs_dict
    """
    
    def construct_observation(
        self, 
        agent_state: Dict[str, Any], 
        plume_state: Any, 
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Construct structured observation dictionary from agent and environment state.
        
        Args:
            agent_state: Current agent state dictionary containing position, orientation,
                speed, and other navigator-managed properties.
            plume_state: Current plume model state for sensor sampling.
            **kwargs: Additional observation components (wind data, derived metrics, etc.).
                
        Returns:
            Dict[str, Any]: Structured observation dictionary with standardized keys:
                - 'position': Agent position as (x, y) array
                - 'orientation': Agent orientation in degrees
                - 'speed': Current agent speed
                - 'sensor_readings': Dictionary of sensor-specific observations
                - Additional keys from kwargs and custom observation components
                
        Notes:
            Observation structure adapts automatically to active sensor configuration.
            Sensor readings are organized by sensor type and identifier for clarity.
            
            Custom observation components can extend base observations through
            inheritance or composition patterns.
            
        Performance:
            Must execute in <0.5ms per agent for real-time step performance.
            
        Examples:
            Standard observation construction:
            >>> obs = obs_protocol.construct_observation(agent_state, plume_state)
            >>> assert 'position' in obs and 'sensor_readings' in obs
            
            With additional wind data:
            >>> obs = obs_protocol.construct_observation(
            ...     agent_state, plume_state, wind_velocity=(2.0, 1.0)
            ... )
        """
        ...
    
    def get_observation_space(self) -> Optional['spaces.Space']:
        """
        Construct Gymnasium observation space matching constructed observations.
        
        Returns:
            Optional[spaces.Space]: Gymnasium observation space (typically spaces.Dict)
                defining the structure and bounds of observation dictionaries. Returns
                None if Gymnasium is not available.
                
        Notes:
            Observation space structure automatically adapts to active sensor
            configuration and custom observation components.
            
            Space bounds are inferred from sensor specifications and agent
            state constraints for proper RL framework integration.
            
        Examples:
            Automatic space inference:
            >>> obs_space = obs_protocol.get_observation_space()
            >>> assert isinstance(obs_space, gymnasium.spaces.Dict)
            >>> assert 'position' in obs_space.spaces
        """
        ...


@runtime_checkable
class AgentActionProtocol(Protocol):
    """
    Protocol defining standardized action structures for agent control interfaces.
    
    This protocol enables flexible action space construction supporting various control
    modalities while maintaining consistency with NavigatorProtocol implementations.
    
    AgentActionProtocol implementations bridge between RL framework action outputs
    and NavigatorProtocol input requirements, handling action scaling, validation,
    and transformation as needed for different control schemes.
    
    Key Design Principles:
    - Structured actions via action dictionaries or arrays with clear semantics
    - Flexible control modalities (velocity, acceleration, waypoint-based)
    - Action validation and constraint enforcement for physical realism
    - Gymnasium action space integration with automatic space inference
    
    Performance Requirements:
    - Action processing: <0.1ms per agent for minimal control overhead
    - Validation: <0.05ms per agent for constraint checking
    - Memory efficiency: <100 bytes per action for structured representations
    
    Examples:
        Velocity-based control:
        >>> action_protocol = VelocityActionProtocol(max_speed=2.0, max_angular_velocity=45.0)
        >>> action = {'linear_velocity': 1.5, 'angular_velocity': 15.0}
        >>> validated_action = action_protocol.validate_action(action)
        
        Acceleration-based control:
        >>> action_protocol = AccelerationActionProtocol(max_acceleration=1.0)
        >>> action = np.array([0.5, -0.2])  # [linear_accel, angular_accel]
        >>> processed_action = action_protocol.process_action(action)
        
        Waypoint navigation:
        >>> action_protocol = WaypointActionProtocol(environment_bounds=(100, 100))
        >>> action = {'target_position': (25, 75), 'approach_speed': 1.2}
        >>> waypoint_action = action_protocol.process_action(action)
    """
    
    def validate_action(self, action: Union[np.ndarray, Dict[str, Any]]) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Validate and constrain action values within physical and safety limits.
        
        Args:
            action: Raw action from RL framework as array or dictionary. Structure
                depends on action protocol implementation but typically includes
                control values for linear and angular motion.
                
        Returns:
            Union[np.ndarray, Dict[str, Any]]: Validated action with constraints
                applied. Format matches input action structure.
                
        Notes:
            Constraint enforcement includes:
            - Physical limits (maximum speeds, accelerations)
            - Safety boundaries (collision avoidance, environment bounds)
            - Numerical stability (avoiding division by zero, NaN values)
            
            Invalid actions are clipped or modified to nearest valid values
            with optional warning generation for debugging.
            
        Performance:
            Must execute in <0.05ms per agent for minimal control overhead.
            
        Examples:
            Array action validation:
            >>> action = np.array([2.5, 60.0])  # [linear_vel, angular_vel]
            >>> validated = action_protocol.validate_action(action)
            >>> # Clips to [2.0, 45.0] based on max_speed and max_angular_velocity
            
            Dictionary action validation:
            >>> action = {'linear_velocity': -1.0, 'angular_velocity': 30.0}
            >>> validated = action_protocol.validate_action(action)
            >>> assert validated['linear_velocity'] >= 0  # Non-negative speed constraint
        """
        ...
    
    def process_action(self, action: Union[np.ndarray, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process raw action into navigator-compatible control parameters.
        
        Args:
            action: Validated action from RL framework as array or dictionary.
                
        Returns:
            Dict[str, Any]: Processed action dictionary with keys matching
                NavigatorProtocol control interface:
                - 'target_speed': Desired linear velocity
                - 'target_angular_velocity': Desired angular velocity  
                - 'control_mode': Control method specification
                - Additional control parameters as needed
                
        Notes:
            Action processing handles conversion between different control
            modalities and navigator input formats. May include:
            - Coordinate transformations (global to local coordinates)
            - Control law evaluation (PID, optimal control)
            - Temporal integration (acceleration to velocity)
            
            Processed actions are immediately applicable to navigator step() methods.
            
        Performance:
            Must execute in <0.1ms per agent for minimal control overhead.
            
        Examples:
            Velocity control processing:
            >>> action = np.array([1.5, 20.0])
            >>> processed = action_protocol.process_action(action)
            >>> assert 'target_speed' in processed
            
            Waypoint control processing:
            >>> action = {'target_position': (30, 40)}
            >>> processed = action_protocol.process_action(action)
            >>> # Computes target_speed and target_angular_velocity for waypoint approach
        """
        ...
    
    def get_action_space(self) -> Optional['spaces.Space']:
        """
        Construct Gymnasium action space matching expected action format.
        
        Returns:
            Optional[spaces.Space]: Gymnasium action space defining valid action
                structure and value ranges. Returns None if Gymnasium is not available.
                
        Notes:
            Action space automatically reflects control modality constraints:
            - Box spaces for continuous control (velocity, acceleration)
            - Discrete spaces for discrete control (direction commands)
            - Dict spaces for structured control (waypoint navigation)
            
            Space bounds match validation constraints for consistency.
            
        Examples:
            Continuous velocity control:
            >>> action_space = action_protocol.get_action_space()
            >>> assert isinstance(action_space, gymnasium.spaces.Box)
            >>> assert action_space.shape == (2,)  # [linear_vel, angular_vel]
            
            Structured waypoint control:
            >>> action_space = action_protocol.get_action_space()
            >>> assert isinstance(action_space, gymnasium.spaces.Dict)
            >>> assert 'target_position' in action_space.spaces
        """
        ...


@runtime_checkable
class ObservationSpaceProtocol(Protocol):
    """
    Protocol for type-safe observation space construction integrated with Gymnasium requirements.
    
    This protocol provides standardized interfaces for creating observation spaces that
    automatically adapt to sensor configurations while maintaining compatibility with
    RL frameworks and type checking systems.
    
    ObservationSpaceProtocol implementations work with AgentObservationProtocol to ensure
    consistency between observation dictionaries and their corresponding Gymnasium spaces.
    
    Key Design Principles:
    - Automatic space inference from sensor configuration and observation structure
    - Type safety through protocol-based design and runtime validation
    - Extensible space construction for custom observation components
    - Performance optimization for space creation and validation operations
    
    Examples:
        Automatic space construction:
        >>> space_builder = ObservationSpaceProtocol()
        >>> obs_space = space_builder.construct_space(sensor_configs, agent_config)
        
        Space validation:
        >>> observation = {'position': np.array([10, 20]), 'concentration': 0.5}
        >>> assert space_builder.validate_observation(observation, obs_space)
    """
    
    def construct_space(
        self, 
        sensor_configs: List[Dict[str, Any]], 
        agent_config: Dict[str, Any],
        **kwargs: Any
    ) -> Optional['spaces.Space']:
        """
        Construct observation space from sensor and agent configuration.
        
        Args:
            sensor_configs: List of sensor configuration dictionaries specifying
                active sensors and their parameters.
            agent_config: Agent configuration dictionary with state space bounds.
            **kwargs: Additional space construction parameters (wind integration, etc.).
                
        Returns:
            Optional[spaces.Space]: Gymnasium observation space (typically spaces.Dict)
                or None if Gymnasium is not available.
                
        Notes:
            Space construction automatically infers observation structure from
            sensor types and configurations. Agent state bounds are derived from
            navigation parameter constraints.
            
        Examples:
            Multi-sensor space construction:
            >>> sensor_configs = [
            ...     {'type': 'BinarySensor', 'threshold': 0.1},
            ...     {'type': 'ConcentrationSensor', 'range': (0, 1)}
            ... ]
            >>> space = construct_space(sensor_configs, agent_config)
        """
        ...


@runtime_checkable
class ActionSpaceProtocol(Protocol):
    """
    Protocol for type-safe action space construction integrated with Gymnasium requirements.
    
    This protocol provides standardized interfaces for creating action spaces that match
    agent control requirements while maintaining compatibility with RL frameworks.
    
    ActionSpaceProtocol implementations work with AgentActionProtocol to ensure consistency
    between action processing and action space definitions.
    
    Examples:
        Control-specific space construction:
        >>> space_builder = ActionSpaceProtocol()
        >>> action_space = space_builder.construct_space(control_config)
        
        Action validation:
        >>> action = np.array([1.5, 30.0])  # [linear_vel, angular_vel]
        >>> assert space_builder.validate_action(action, action_space)
    """
    
    def construct_space(
        self, 
        control_config: Dict[str, Any], 
        **kwargs: Any
    ) -> Optional['spaces.Space']:
        """
        Construct action space from control configuration.
        
        Args:
            control_config: Control configuration dictionary specifying control
                modality and parameter constraints.
            **kwargs: Additional space construction parameters.
                
        Returns:
            Optional[spaces.Space]: Gymnasium action space or None if Gymnasium
                is not available.
                
        Examples:
            Velocity control space:
            >>> control_config = {'type': 'velocity', 'max_speed': 2.0, 'max_angular_velocity': 45.0}
            >>> space = construct_space(control_config)
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
    
    Enhanced for Gymnasium 0.29.x migration:
    - Gymnasium-specific parameter validation and space creation integration
    - Dual API compatibility detection and automatic format conversion
    - Integration with SpaceFactory utilities for type-safe space construction
    - Support for new extensibility hooks in navigator creation
    
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
                the full protocol interface with Gymnasium 0.29.x support.
                
        Notes:
            Automatically detects single vs multi-agent mode from configuration:
            - Single agent: Uses position, orientation, speed, etc.
            - Multi-agent: Uses positions, orientations, speeds, etc.
            
            Configuration validation ensures parameter compatibility and
            constraint satisfaction before navigator instantiation.
            
            Enhanced for Gymnasium 0.29.x:
            - Validates observation and action space compatibility
            - Configures extensibility hooks if specified
            - Sets up dual API compatibility mode detection
            
        Raises:
            ValueError: If configuration is invalid or incomplete
            TypeError: If configuration type is unsupported
            ImportError: If required Gymnasium dependencies are missing
            
        Examples:
            From Hydra configuration:
            >>> @hydra.main(config_path="../conf", config_name="config")
            >>> def main(cfg: DictConfig) -> None:
            ...     navigator = NavigatorFactory.from_config(cfg.navigator)
            
            From Pydantic model (when schemas are available):
            >>> config = NavigatorConfig(position=(10.0, 20.0), max_speed=2.0)
            >>> navigator = NavigatorFactory.from_config(config)
            
            With extensibility hooks:
            >>> config = {
            ...     "position": (0, 0),
            ...     "max_speed": 2.0,
            ...     "enable_extensibility_hooks": True,
            ...     "custom_observation_keys": ["wind_direction", "energy_level"]
            ... }
            >>> navigator = NavigatorFactory.from_config(config)
        """
        # Import here to avoid circular imports - these will be created by other agents
        try:
            from ..core.controllers import SingleAgentController, MultiAgentController
        except ImportError:
            # Fallback imports from source packages during migration
            try:
                from plume_nav_sim.core.controllers import SingleAgentController, MultiAgentController
            except ImportError:
                # If controllers don't exist yet, raise informative error
                raise ImportError(
                    "Navigator controllers not yet available. Ensure "
                    "plume_nav_sim.core.controllers module has been created."
                )
        
        # Convert to NavigatorConfig if needed for validation
        if isinstance(config, dict):
            config = NavigatorConfig(**config)
        elif hasattr(config, 'to_container') and HYDRA_AVAILABLE:
            config_dict = config.to_container(resolve=True)
            config = NavigatorConfig(**config_dict)
        
        # Validate Gymnasium availability for enhanced features
        if not GYMNASIUM_AVAILABLE:
            warnings.warn(
                "Gymnasium not available. Some advanced features may be limited. "
                "Install gymnasium>=0.29.0 for full functionality.",
                UserWarning,
                stacklevel=2
            )
        
        # Determine navigator type and create appropriate implementation
        if _is_multi_agent_config(config):
            return MultiAgentController(
                positions=_get_config_value(config, 'positions'),
                orientations=_get_config_value(config, 'orientations'),
                speeds=_get_config_value(config, 'speeds'),
                max_speeds=_get_config_value(config, 'max_speeds'),
                angular_velocities=_get_config_value(config, 'angular_velocities'),
                **_extract_extensibility_config(config)
            )
        else:
            return SingleAgentController(
                position=_get_config_value(config, 'position'),
                orientation=_get_config_value(config, 'orientation', 0.0),
                speed=_get_config_value(config, 'speed', 0.0),
                max_speed=_get_config_value(config, 'max_speed', 1.0),
                angular_velocity=_get_config_value(config, 'angular_velocity', 0.0),
                **_extract_extensibility_config(config)
            )
    
    @staticmethod
    def single_agent(
        position: Optional[Tuple[float, float]] = None,
        orientation: float = 0.0,
        speed: float = 0.0,
        max_speed: float = 1.0,
        angular_velocity: float = 0.0,
        enable_extensibility_hooks: bool = False,
        **kwargs: Any
    ) -> NavigatorProtocol:
        """
        Create single-agent navigator with explicit parameters.
        
        Args:
            position: Initial (x, y) position (default: (0, 0))
            orientation: Initial orientation in degrees (default: 0.0)
            speed: Initial speed (default: 0.0)
            max_speed: Maximum allowed speed (default: 1.0) 
            angular_velocity: Initial angular velocity in deg/s (default: 0.0)
            enable_extensibility_hooks: Enable custom observation/reward hooks
            **kwargs: Additional configuration options for extensibility
            
        Returns:
            NavigatorProtocol: Single-agent navigator implementation with
                Gymnasium 0.29.x compatibility and optional extensibility hooks.
            
        Examples:
            Basic single agent:
            >>> navigator = NavigatorFactory.single_agent()
            
            Configured single agent:
            >>> navigator = NavigatorFactory.single_agent(
            ...     position=(50.0, 100.0), max_speed=2.5
            ... )
            
            With extensibility hooks:
            >>> navigator = NavigatorFactory.single_agent(
            ...     position=(10.0, 20.0),
            ...     enable_extensibility_hooks=True,
            ...     custom_sensors=["wind", "temperature"]
            ... )
        """
        try:
            from ..core.controllers import SingleAgentController
        except ImportError:
            raise ImportError(
                "SingleAgentController not yet available. Ensure "
                "plume_nav_sim.core.controllers module has been created."
            )
        
        return SingleAgentController(
            position=position,
            orientation=orientation,
            speed=speed,
            max_speed=max_speed,
            angular_velocity=angular_velocity,
            enable_extensibility_hooks=enable_extensibility_hooks,
            **kwargs
        )
    
    @staticmethod
    def multi_agent(
        positions: Union[List[List[float]], np.ndarray],
        orientations: Optional[Union[List[float], np.ndarray]] = None,
        speeds: Optional[Union[List[float], np.ndarray]] = None,
        max_speeds: Optional[Union[List[float], np.ndarray]] = None,
        angular_velocities: Optional[Union[List[float], np.ndarray]] = None,
        enable_extensibility_hooks: bool = False,
        **kwargs: Any
    ) -> NavigatorProtocol:
        """
        Create multi-agent navigator with explicit parameters.
        
        Args:
            positions: Initial positions as list or array with shape (n_agents, 2)
            orientations: Initial orientations for each agent (optional)
            speeds: Initial speeds for each agent (optional)
            max_speeds: Maximum speeds for each agent (optional)
            angular_velocities: Initial angular velocities for each agent (optional)
            enable_extensibility_hooks: Enable custom observation/reward hooks
            **kwargs: Additional configuration options for extensibility
            
        Returns:
            NavigatorProtocol: Multi-agent navigator implementation with
                Gymnasium 0.29.x compatibility and optional extensibility hooks.
            
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
            
            With extensibility hooks:
            >>> navigator = NavigatorFactory.multi_agent(
            ...     positions=[[0, 0], [10, 10]],
            ...     enable_extensibility_hooks=True,
            ...     reward_shaping="exploration_bonus"
            ... )
        """
        try:
            from ..core.controllers import MultiAgentController
        except ImportError:
            raise ImportError(
                "MultiAgentController not yet available. Ensure "
                "plume_nav_sim.core.controllers module has been created."
            )
        
        return MultiAgentController(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds,
            angular_velocities=angular_velocities,
            enable_extensibility_hooks=enable_extensibility_hooks,
            **kwargs
        )

    @staticmethod
    def create_observation_space(
        navigator: NavigatorProtocol,
        include_additional_obs: bool = True,
        **space_kwargs: Any
    ) -> Optional[spaces.Space]:
        """
        Create Gymnasium observation space for a navigator instance.
        
        Integrates with SpaceFactory utilities to create type-safe observation
        spaces that account for both base navigation observations and any
        additional observations from extensibility hooks.
        
        Args:
            navigator: Navigator instance to create space for
            include_additional_obs: Include space for additional observations
            **space_kwargs: Additional arguments for space construction
            
        Returns:
            Optional[spaces.Space]: Gymnasium observation space or None if
                Gymnasium is not available.
                
        Examples:
            Basic observation space:
            >>> navigator = NavigatorFactory.single_agent()
            >>> obs_space = NavigatorFactory.create_observation_space(navigator)
            
            With additional observations:
            >>> obs_space = NavigatorFactory.create_observation_space(
            ...     navigator, include_additional_obs=True
            ... )
        """
        if not GYMNASIUM_AVAILABLE:
            return None

        if not SPACE_FACTORY_AVAILABLE or SpaceFactory is None:
            logger.error("SpaceFactory is required for observation space creation")
            raise ImportError(
                "SpaceFactory is required for NavigatorFactory.create_observation_space"
            )

        return SpaceFactory.create_observation_space(
            num_agents=navigator.num_agents,
            include_additional_obs=include_additional_obs,
            **space_kwargs
        )
    
    @staticmethod
    def create_action_space(
        navigator: NavigatorProtocol,
        **space_kwargs: Any
    ) -> Optional[spaces.Space]:
        """
        Create Gymnasium action space for a navigator instance.
        
        Integrates with SpaceFactory utilities to create type-safe action
        spaces appropriate for the navigator's configuration.
        
        Args:
            navigator: Navigator instance to create space for
            **space_kwargs: Additional arguments for space construction
            
        Returns:
            Optional[spaces.Space]: Gymnasium action space or None if
                Gymnasium is not available.
                
        Examples:
            Basic action space:
            >>> navigator = NavigatorFactory.single_agent()
            >>> action_space = NavigatorFactory.create_action_space(navigator)
        """
        if not GYMNASIUM_AVAILABLE:
            return None

        if not SPACE_FACTORY_AVAILABLE or SpaceFactory is None:
            logger.error("SpaceFactory is required for action space creation")
            raise ImportError(
                "SpaceFactory is required for NavigatorFactory.create_action_space"
            )

        return SpaceFactory.create_action_space(
            num_agents=navigator.num_agents,
            **space_kwargs
        )

    @staticmethod
    def create_plume_model(config: Union[DictConfig, dict]) -> 'PlumeModelProtocol':
        """
        Create plume model from configuration using the modular architecture.
        
        Args:
            config: Configuration specifying plume model type and parameters.
                Should include '_target_' field for Hydra instantiation or 'type' field
                for factory-based creation.
                
        Returns:
            PlumeModelProtocol: Configured plume model implementation.
            
        Examples:
            Gaussian plume model:
            >>> config = {
            ...     '_target_': 'plume_nav_sim.models.plume.GaussianPlumeModel',
            ...     'source_position': (50, 50),
            ...     'source_strength': 1000.0
            ... }
            >>> plume_model = NavigatorFactory.create_plume_model(config)
            
            Turbulent plume model:
            >>> config = {
            ...     'type': 'TurbulentPlumeModel',
            ...     'filament_count': 500,
            ...     'turbulence_intensity': 0.3
            ... }
            >>> plume_model = NavigatorFactory.create_plume_model(config)
        """
        if HYDRA_AVAILABLE and '_target_' in config:
            # Use Hydra instantiation for dependency injection
            from hydra import utils as hydra_utils
            return hydra_utils.instantiate(config)
        else:
            # Fallback factory method for simple configurations
            model_type = _get_config_value(config, 'type', 'GaussianPlumeModel')
            
            # Import here to avoid circular dependencies
            try:
                if model_type == 'GaussianPlumeModel':
                    from ..models.plume.gaussian_plume import GaussianPlumeModel
                    return GaussianPlumeModel(**{k: v for k, v in config.items() if k != 'type'})
                elif model_type == 'TurbulentPlumeModel':
                    from ..models.plume.turbulent_plume import TurbulentPlumeModel
                    return TurbulentPlumeModel(**{k: v for k, v in config.items() if k != 'type'})
                elif model_type == 'VideoPlumeAdapter':
                    from ..models.plume.video_plume_adapter import VideoPlumeAdapter
                    return VideoPlumeAdapter(**{k: v for k, v in config.items() if k != 'type'})
                else:
                    raise ValueError(f"Unknown plume model type: {model_type}")
            except ImportError as e:
                raise ImportError(
                    f"Plume model implementation not available: {model_type}. "
                    f"Ensure the model module has been created. Error: {e}"
                )

    @staticmethod
    def create_wind_field(config: Union[DictConfig, dict]) -> 'WindFieldProtocol':
        """
        Create wind field from configuration using the modular architecture.
        
        Args:
            config: Configuration specifying wind field type and parameters.
                
        Returns:
            WindFieldProtocol: Configured wind field implementation.
            
        Examples:
            Constant wind field:
            >>> config = {'type': 'ConstantWindField', 'velocity': (2.0, 0.5)}
            >>> wind_field = NavigatorFactory.create_wind_field(config)
            
            Turbulent wind field:
            >>> config = {
            ...     'type': 'TurbulentWindField',
            ...     'mean_velocity': (3.0, 1.0),
            ...     'turbulence_intensity': 0.2
            ... }
            >>> wind_field = NavigatorFactory.create_wind_field(config)
        """
        if HYDRA_AVAILABLE and '_target_' in config:
            from hydra import utils as hydra_utils
            return hydra_utils.instantiate(config)
        else:
            wind_type = _get_config_value(config, 'type', 'ConstantWindField')
            
            try:
                if wind_type == 'ConstantWindField':
                    from ..models.wind.constant_wind import ConstantWindField
                    return ConstantWindField(**{k: v for k, v in config.items() if k != 'type'})
                elif wind_type == 'TurbulentWindField':
                    from ..models.wind.turbulent_wind import TurbulentWindField
                    return TurbulentWindField(**{k: v for k, v in config.items() if k != 'type'})
                elif wind_type == 'TimeVaryingWindField':
                    from ..models.wind.time_varying_wind import TimeVaryingWindField
                    return TimeVaryingWindField(**{k: v for k, v in config.items() if k != 'type'})
                else:
                    raise ValueError(f"Unknown wind field type: {wind_type}")
            except ImportError as e:
                raise ImportError(
                    f"Wind field implementation not available: {wind_type}. "
                    f"Ensure the wind field module has been created. Error: {e}"
                )

    @staticmethod
    def create_sensors(sensor_configs: List[Union[DictConfig, dict]]) -> List['SensorProtocol']:
        """
        Create list of sensors from configuration using the modular architecture.
        
        Args:
            sensor_configs: List of sensor configurations.
                
        Returns:
            List[SensorProtocol]: List of configured sensor implementations.
            
        Examples:
            Multi-sensor setup:
            >>> sensor_configs = [
            ...     {'type': 'BinarySensor', 'threshold': 0.1},
            ...     {'type': 'ConcentrationSensor', 'dynamic_range': (0, 1)},
            ...     {'type': 'GradientSensor', 'spatial_resolution': (0.5, 0.5)}
            ... ]
            >>> sensors = NavigatorFactory.create_sensors(sensor_configs)
        """
        sensors = []
        for config in sensor_configs:
            if HYDRA_AVAILABLE and '_target_' in config:
                from hydra import utils as hydra_utils
                sensors.append(hydra_utils.instantiate(config))
            else:
                sensor_type = _get_config_value(config, 'type', 'ConcentrationSensor')
                
                try:
                    if sensor_type == 'BinarySensor':
                        from ..core.sensors.binary_sensor import BinarySensor
                        sensors.append(BinarySensor(**{k: v for k, v in config.items() if k != 'type'}))
                    elif sensor_type == 'ConcentrationSensor':
                        from ..core.sensors.concentration_sensor import ConcentrationSensor
                        sensors.append(ConcentrationSensor(**{k: v for k, v in config.items() if k != 'type'}))
                    elif sensor_type == 'GradientSensor':
                        from ..core.sensors.gradient_sensor import GradientSensor
                        sensors.append(GradientSensor(**{k: v for k, v in config.items() if k != 'type'}))
                    else:
                        raise ValueError(f"Unknown sensor type: {sensor_type}")
                except ImportError as e:
                    raise ImportError(
                        f"Sensor implementation not available: {sensor_type}. "
                        f"Ensure the sensor module has been created. Error: {e}"
                    )
        return sensors

    @staticmethod
    def create_source(config: Union[DictConfig, dict]) -> 'SourceProtocol':
        """
        Create source from configuration using the modular architecture.
        
        Args:
            config: Configuration specifying source type and parameters.
                
        Returns:
            SourceProtocol: Configured source implementation.
            
        Examples:
            Point source:
            >>> config = {'type': 'PointSource', 'position': (50, 50), 'emission_rate': 1000.0}
            >>> source = NavigatorFactory.create_source(config)
            
            Dynamic source:
            >>> config = {
            ...     'type': 'DynamicSource',
            ...     'initial_position': (25, 75),
            ...     'emission_pattern': 'sinusoidal'
            ... }
            >>> source = NavigatorFactory.create_source(config)
        """
        if HYDRA_AVAILABLE and '_target_' in config:
            from hydra import utils as hydra_utils
            return hydra_utils.instantiate(config)
        else:
            source_type = _get_config_value(config, 'type', 'PointSource')
            
            try:
                if source_type == 'PointSource':
                    from ..core.sources import PointSource
                    return PointSource(**{k: v for k, v in config.items() if k != 'type'})
                elif source_type == 'MultiSource':
                    from ..core.sources import MultiSource
                    return MultiSource(**{k: v for k, v in config.items() if k != 'type'})
                elif source_type == 'DynamicSource':
                    from ..core.sources import DynamicSource
                    return DynamicSource(**{k: v for k, v in config.items() if k != 'type'})
                else:
                    raise ValueError(f"Unknown source type: {source_type}")
            except ImportError as e:
                raise ImportError(
                    f"Source implementation not available: {source_type}. "
                    f"Ensure the source module has been created. Error: {e}"
                )

    @staticmethod
    def create_boundary_policy(config: Union[DictConfig, dict]) -> 'BoundaryPolicyProtocol':
        """
        Create boundary policy from configuration using the modular architecture.
        
        Args:
            config: Configuration specifying boundary policy type and parameters.
                
        Returns:
            BoundaryPolicyProtocol: Configured boundary policy implementation.
            
        Examples:
            Terminate boundary:
            >>> config = {'type': 'TerminatePolicy', 'domain_bounds': (100, 100)}
            >>> policy = NavigatorFactory.create_boundary_policy(config)
            
            Bounce boundary:
            >>> config = {
            ...     'type': 'BouncePolicy',
            ...     'domain_bounds': (100, 100),
            ...     'energy_loss': 0.1
            ... }
            >>> policy = NavigatorFactory.create_boundary_policy(config)
        """
        if HYDRA_AVAILABLE and '_target_' in config:
            from hydra import utils as hydra_utils
            return hydra_utils.instantiate(config)
        else:
            policy_type = _get_config_value(config, 'type', 'TerminatePolicy')
            
            try:
                if policy_type == 'TerminatePolicy':
                    from ..core.boundaries import TerminatePolicy
                    return TerminatePolicy(**{k: v for k, v in config.items() if k != 'type'})
                elif policy_type == 'BouncePolicy':
                    from ..core.boundaries import BouncePolicy
                    return BouncePolicy(**{k: v for k, v in config.items() if k != 'type'})
                elif policy_type == 'WrapPolicy':
                    from ..core.boundaries import WrapPolicy
                    return WrapPolicy(**{k: v for k, v in config.items() if k != 'type'})
                elif policy_type == 'ClipPolicy':
                    from ..core.boundaries import ClipPolicy
                    return ClipPolicy(**{k: v for k, v in config.items() if k != 'type'})
                else:
                    raise ValueError(f"Unknown boundary policy type: {policy_type}")
            except ImportError as e:
                raise ImportError(
                    f"Boundary policy implementation not available: {policy_type}. "
                    f"Ensure the boundary policy module has been created. Error: {e}"
                )

    @staticmethod
    def create_action_interface(config: Union[DictConfig, dict]) -> 'ActionInterfaceProtocol':
        """
        Create action interface from configuration using the modular architecture.
        
        Args:
            config: Configuration specifying action interface type and parameters.
                
        Returns:
            ActionInterfaceProtocol: Configured action interface implementation.
            
        Examples:
            Continuous 2D control:
            >>> config = {
            ...     'type': 'Continuous2D',
            ...     'max_linear_velocity': 2.0,
            ...     'max_angular_velocity': 45.0
            ... }
            >>> action_interface = NavigatorFactory.create_action_interface(config)
            
            Cardinal discrete control:
            >>> config = {'type': 'CardinalDiscrete', 'action_count': 9}
            >>> action_interface = NavigatorFactory.create_action_interface(config)
        """
        if HYDRA_AVAILABLE and '_target_' in config:
            from hydra import utils as hydra_utils
            return hydra_utils.instantiate(config)
        else:
            interface_type = _get_config_value(config, 'type', 'Continuous2D')
            
            try:
                if interface_type == 'Continuous2D':
                    from ..core.actions import Continuous2DAction
                    return Continuous2DAction(**{k: v for k, v in config.items() if k != 'type'})
                elif interface_type == 'CardinalDiscrete':
                    from ..core.actions import CardinalDiscreteAction
                    return CardinalDiscreteAction(**{k: v for k, v in config.items() if k != 'type'})
                else:
                    raise ValueError(f"Unknown action interface type: {interface_type}")
            except ImportError as e:
                raise ImportError(
                    f"Action interface implementation not available: {interface_type}. "
                    f"Ensure the action interface module has been created. Error: {e}"
                )

    @staticmethod
    def create_recorder(config: Union[DictConfig, dict]) -> 'RecorderProtocol':
        """
        Create recorder from configuration using the modular architecture.
        
        Args:
            config: Configuration specifying recorder backend and parameters.
                
        Returns:
            RecorderProtocol: Configured recorder implementation.
            
        Examples:
            Parquet recorder:
            >>> config = {
            ...     'type': 'ParquetRecorder',
            ...     'output_dir': './data',
            ...     'compression': 'snappy'
            ... }
            >>> recorder = NavigatorFactory.create_recorder(config)
            
            HDF5 recorder:
            >>> config = {
            ...     'type': 'HDF5Recorder',
            ...     'output_dir': './data',
            ...     'compression': 'gzip'
            ... }
            >>> recorder = NavigatorFactory.create_recorder(config)
        """
        if HYDRA_AVAILABLE and '_target_' in config:
            from hydra import utils as hydra_utils
            return hydra_utils.instantiate(config)
        else:
            recorder_type = _get_config_value(config, 'type', 'ParquetRecorder')
            
            try:
                if recorder_type == 'ParquetRecorder':
                    from ..recording.backends.parquet_backend import ParquetRecorder
                    return ParquetRecorder(**{k: v for k, v in config.items() if k != 'type'})
                elif recorder_type == 'HDF5Recorder':
                    from ..recording.backends.hdf5_backend import HDF5Recorder
                    return HDF5Recorder(**{k: v for k, v in config.items() if k != 'type'})
                elif recorder_type == 'SQLiteRecorder':
                    from ..recording.backends.sqlite_backend import SQLiteRecorder
                    return SQLiteRecorder(**{k: v for k, v in config.items() if k != 'type'})
                elif recorder_type == 'NoneRecorder':
                    from ..recording.backends.none_backend import NoneRecorder
                    return NoneRecorder(**{k: v for k, v in config.items() if k != 'type'})
                else:
                    raise ValueError(f"Unknown recorder type: {recorder_type}")
            except ImportError as e:
                raise ImportError(
                    f"Recorder implementation not available: {recorder_type}. "
                    f"Ensure the recorder module has been created. Error: {e}"
                )

    @staticmethod
    def create_stats_aggregator(config: Union[DictConfig, dict]) -> 'StatsAggregatorProtocol':
        """
        Create statistics aggregator from configuration using the modular architecture.
        
        Args:
            config: Configuration specifying stats aggregator type and parameters.
                
        Returns:
            StatsAggregatorProtocol: Configured statistics aggregator implementation.
            
        Examples:
            Standard stats aggregator:
            >>> config = {'type': 'StandardStatsAggregator', 'include_distributions': True}
            >>> aggregator = NavigatorFactory.create_stats_aggregator(config)
            
            Custom metrics aggregator:
            >>> config = {
            ...     'type': 'CustomStatsAggregator',
            ...     'custom_metrics': ['tortuosity', 'efficiency']
            ... }
            >>> aggregator = NavigatorFactory.create_stats_aggregator(config)
        """
        if HYDRA_AVAILABLE and '_target_' in config:
            from hydra import utils as hydra_utils
            return hydra_utils.instantiate(config)
        else:
            aggregator_type = _get_config_value(config, 'type', 'StandardStatsAggregator')
            
            try:
                if aggregator_type == 'StandardStatsAggregator':
                    from ..analysis.stats import StandardStatsAggregator
                    return StandardStatsAggregator(**{k: v for k, v in config.items() if k != 'type'})
                elif aggregator_type == 'CustomStatsAggregator':
                    from ..analysis.stats import CustomStatsAggregator
                    return CustomStatsAggregator(**{k: v for k, v in config.items() if k != 'type'})
                else:
                    raise ValueError(f"Unknown stats aggregator type: {aggregator_type}")
            except ImportError as e:
                raise ImportError(
                    f"Stats aggregator implementation not available: {aggregator_type}. "
                    f"Ensure the stats aggregator module has been created. Error: {e}"
                )

    @staticmethod
    def create_agent_initializer(config: Union[DictConfig, dict]) -> 'AgentInitializerProtocol':
        """
        Create agent initializer from configuration using the modular architecture.
        
        Args:
            config: Configuration specifying initializer strategy and parameters.
                
        Returns:
            AgentInitializerProtocol: Configured agent initializer implementation.
            
        Examples:
            Uniform random initializer:
            >>> config = {
            ...     'type': 'UniformRandomInitializer',
            ...     'domain_bounds': (100, 100),
            ...     'seed': 42
            ... }
            >>> initializer = NavigatorFactory.create_agent_initializer(config)
            
            Grid initializer:
            >>> config = {
            ...     'type': 'GridInitializer',
            ...     'domain_bounds': (100, 100),
            ...     'grid_spacing': 10.0
            ... }
            >>> initializer = NavigatorFactory.create_agent_initializer(config)
        """
        if HYDRA_AVAILABLE and '_target_' in config:
            from hydra import utils as hydra_utils
            return hydra_utils.instantiate(config)
        else:
            initializer_type = _get_config_value(config, 'type', 'UniformRandomInitializer')
            
            try:
                # Extract parameters and handle common parameter mappings
                params = {k: v for k, v in config.items() if k != 'type'}
                
                if initializer_type == 'UniformRandomInitializer':
                    from ..core.initialization import UniformRandomInitializer
                    # Map domain_bounds to bounds for compatibility
                    if 'domain_bounds' in params and 'bounds' not in params:
                        params['bounds'] = params.pop('domain_bounds')
                    return UniformRandomInitializer(**params)
                elif initializer_type == 'GridInitializer':
                    from ..core.initialization import GridInitializer
                    return GridInitializer(**params)
                elif initializer_type == 'FixedListInitializer':
                    from ..core.initialization import FixedListInitializer
                    return FixedListInitializer(**params)
                elif initializer_type == 'DatasetInitializer':
                    from ..core.initialization import DatasetInitializer
                    return DatasetInitializer(**params)
                else:
                    raise ValueError(f"Unknown agent initializer type: {initializer_type}")
            except ImportError as e:
                raise ImportError(
                    f"Agent initializer implementation not available: {initializer_type}. "
                    f"Ensure the agent initializer module has been created. Error: {e}"
                )

    @staticmethod
    def create_modular_environment(
        navigator_config: Union[DictConfig, dict],
        plume_model_config: Union[DictConfig, dict],
        wind_field_config: Optional[Union[DictConfig, dict]] = None,
        sensor_configs: Optional[List[Union[DictConfig, dict]]] = None,
        source_config: Optional[Union[DictConfig, dict]] = None,
        boundary_policy_config: Optional[Union[DictConfig, dict]] = None,
        action_interface_config: Optional[Union[DictConfig, dict]] = None,
        recorder_config: Optional[Union[DictConfig, dict]] = None,
        stats_aggregator_config: Optional[Union[DictConfig, dict]] = None,
        agent_initializer_config: Optional[Union[DictConfig, dict]] = None,
        **env_kwargs: Any
    ) -> 'NavigatorProtocol':
        """
        Create complete modular navigation environment with all v1.0 components.
        
        Args:
            navigator_config: Navigator configuration.
            plume_model_config: Plume model configuration.
            wind_field_config: Optional wind field configuration.
            sensor_configs: Optional list of sensor configurations.
            source_config: Optional source configuration for odor emission modeling.
            boundary_policy_config: Optional boundary policy configuration.
            action_interface_config: Optional action interface configuration.
            recorder_config: Optional recorder configuration for data persistence.
            stats_aggregator_config: Optional statistics aggregator configuration.
            agent_initializer_config: Optional agent initializer configuration.
            **env_kwargs: Additional environment parameters.
            
        Returns:
            NavigatorProtocol: Complete navigation environment with all v1.0 components.
            
        Examples:
            Complete v1.0 modular environment:
            >>> env = NavigatorFactory.create_modular_environment(
            ...     navigator_config={'position': (0, 0), 'max_speed': 2.0},
            ...     plume_model_config={'type': 'GaussianPlumeModel'},
            ...     source_config={'type': 'PointSource', 'position': (50, 50), 'emission_rate': 1000.0},
            ...     boundary_policy_config={'type': 'TerminatePolicy', 'domain_bounds': (100, 100)},
            ...     action_interface_config={'type': 'Continuous2D', 'max_linear_velocity': 2.0},
            ...     recorder_config={'type': 'ParquetRecorder', 'output_dir': './data'},
            ...     stats_aggregator_config={'type': 'StandardStatsAggregator'},
            ...     agent_initializer_config={'type': 'UniformRandomInitializer', 'domain_bounds': (100, 100)},
            ...     wind_field_config={'type': 'ConstantWindField', 'velocity': (1.0, 0.0)},
            ...     sensor_configs=[{'type': 'ConcentrationSensor', 'dynamic_range': (0, 1)}]
            ... )
        """
        # Create individual components
        navigator = NavigatorFactory.from_config(navigator_config)
        plume_model = NavigatorFactory.create_plume_model(plume_model_config)
        
        # Create v1.0 components
        source = None
        if source_config:
            source = NavigatorFactory.create_source(source_config)
        
        boundary_policy = None
        if boundary_policy_config:
            boundary_policy = NavigatorFactory.create_boundary_policy(boundary_policy_config)
        
        action_interface = None
        if action_interface_config:
            action_interface = NavigatorFactory.create_action_interface(action_interface_config)
        
        recorder = None
        if recorder_config:
            recorder = NavigatorFactory.create_recorder(recorder_config)
        
        stats_aggregator = None
        if stats_aggregator_config:
            stats_aggregator = NavigatorFactory.create_stats_aggregator(stats_aggregator_config)
        
        agent_initializer = None
        if agent_initializer_config:
            agent_initializer = NavigatorFactory.create_agent_initializer(agent_initializer_config)
        
        # Create existing components
        wind_field = None
        if wind_field_config:
            wind_field = NavigatorFactory.create_wind_field(wind_field_config)
        
        sensors = []
        if sensor_configs:
            sensors = NavigatorFactory.create_sensors(sensor_configs)
        
        # Integrate components (this would be handled by the environment wrapper)
        # For now, we return the navigator with component information attached
        if hasattr(navigator, '_modular_components'):
            navigator._modular_components = {
                'plume_model': plume_model,
                'wind_field': wind_field,
                'sensors': sensors,
                'source': source,
                'boundary_policy': boundary_policy,
                'action_interface': action_interface,
                'recorder': recorder,
                'stats_aggregator': stats_aggregator,
                'agent_initializer': agent_initializer,
                'env_kwargs': env_kwargs
            }
        
        return navigator

    @staticmethod
    def validate_protocol_compliance(
        component: Any, 
        protocol_type: type
    ) -> bool:
        """
        Validate that a component implements the specified protocol.
        
        Args:
            component: Component instance to validate.
            protocol_type: Protocol class to check against.
            
        Returns:
            bool: True if component implements the protocol.
            
        Examples:
            Validate plume model compliance:
            >>> is_valid = NavigatorFactory.validate_protocol_compliance(
            ...     plume_model, PlumeModelProtocol
            ... )
            >>> assert is_valid, "Component must implement PlumeModelProtocol"
            
            Validate v1.0 component compliance:
            >>> is_valid = NavigatorFactory.validate_protocol_compliance(
            ...     source, SourceProtocol
            ... )
            >>> assert is_valid, "Component must implement SourceProtocol"
        """
        return isinstance(component, protocol_type)

    @staticmethod
    def validate_v1_component_suite(
        source: Optional[Any] = None,
        boundary_policy: Optional[Any] = None,
        action_interface: Optional[Any] = None,
        recorder: Optional[Any] = None,
        stats_aggregator: Optional[Any] = None,
        agent_initializer: Optional[Any] = None
    ) -> Dict[str, bool]:
        """
        Validate complete v1.0 component suite for protocol compliance.
        
        Args:
            source: Optional source component to validate.
            boundary_policy: Optional boundary policy component to validate.
            action_interface: Optional action interface component to validate.
            recorder: Optional recorder component to validate.
            stats_aggregator: Optional stats aggregator component to validate.
            agent_initializer: Optional agent initializer component to validate.
            
        Returns:
            Dict[str, bool]: Validation results for each provided component.
            
        Examples:
            Validate complete component suite:
            >>> validation_results = NavigatorFactory.validate_v1_component_suite(
            ...     source=my_source,
            ...     boundary_policy=my_boundary_policy,
            ...     action_interface=my_action_interface,
            ...     recorder=my_recorder
            ... )
            >>> assert all(validation_results.values()), "All components must be valid"
        """
        results = {}
        
        if source is not None:
            results['source'] = NavigatorFactory.validate_protocol_compliance(source, SourceProtocol)
        
        if boundary_policy is not None:
            results['boundary_policy'] = NavigatorFactory.validate_protocol_compliance(
                boundary_policy, BoundaryPolicyProtocol
            )
        
        if action_interface is not None:
            results['action_interface'] = NavigatorFactory.validate_protocol_compliance(
                action_interface, ActionInterfaceProtocol
            )
        
        if recorder is not None:
            results['recorder'] = NavigatorFactory.validate_protocol_compliance(recorder, RecorderProtocol)
        
        if stats_aggregator is not None:
            results['stats_aggregator'] = NavigatorFactory.validate_protocol_compliance(
                stats_aggregator, StatsAggregatorProtocol
            )
        
        if agent_initializer is not None:
            results['agent_initializer'] = NavigatorFactory.validate_protocol_compliance(
                agent_initializer, AgentInitializerProtocol
            )
        
        return results


# Utility functions for configuration processing and API compatibility

def _is_multi_agent_config(config: Union[NavigatorConfig, dict, DictConfig]) -> bool:
    """
    Determine if configuration specifies multi-agent navigation.
    
    Args:
        config: Configuration object to analyze
        
    Returns:
        bool: True if multi-agent configuration, False for single-agent
        
    Notes:
        Multi-agent is indicated by:
        - positions (list/array) is not None
        - num_agents > 1
        - Any of orientations, speeds, max_speeds, angular_velocities is a list
    """
    positions = _get_config_value(config, 'positions')
    if positions is not None:
        return True
    
    num_agents = _get_config_value(config, 'num_agents')
    if num_agents is not None and num_agents > 1:
        return True
    
    # Check if any parameters are lists (indicating multi-agent)
    list_params = [
        _get_config_value(config, 'orientations'),
        _get_config_value(config, 'speeds'),
        _get_config_value(config, 'max_speeds'),
        _get_config_value(config, 'angular_velocities')
    ]
    return any(isinstance(param, (list, np.ndarray)) for param in list_params if param is not None)


def _get_config_value(config: Union[NavigatorConfig, dict, DictConfig], key: str, default: Any = None) -> Any:
    """
    Safely extract value from configuration object.
    
    Handles different configuration types (Pydantic, dict, DictConfig)
    gracefully during the migration period.
    """
    if hasattr(config, key):
        return getattr(config, key, default)
    elif isinstance(config, dict):
        return config.get(key, default)
    elif hasattr(config, 'get'):
        return config.get(key, default)
    else:
        return default


def _extract_extensibility_config(config: Union[NavigatorConfig, dict, DictConfig]) -> dict:
    """
    Extract extensibility-related configuration options.
    
    Returns a dict of configuration options related to the new
    extensibility hooks and features.
    """
    extensibility_keys = [
        'enable_extensibility_hooks',
        'custom_observation_keys',
        'reward_shaping',
        'custom_sensors',
        'episode_callbacks'
    ]
    
    extensibility_config = {}
    for key in extensibility_keys:
        value = _get_config_value(config, key)
        if value is not None:
            extensibility_config[key] = value
    
    return extensibility_config


def detect_api_compatibility_mode() -> str:
    """
    Detect whether caller expects legacy 4-tuple or modern 5-tuple API returns.
    
    This function inspects the call stack to determine if the calling code
    expects legacy Gym 4-tuple returns (obs, reward, done, info) or modern
    Gymnasium 5-tuple returns (obs, reward, terminated, truncated, info).
    
    Returns:
        str: "legacy" for 4-tuple expectation, "modern" for 5-tuple expectation
        
    Notes:
        Uses stack introspection to detect calling patterns. This is used by
        the compatibility shim to automatically convert between API formats.
        
        Detection heuristics:
        - Legacy: calls from gym.make() or modules importing gym
        - Modern: calls from gymnasium.make() or modules importing gymnasium
        - Default: "modern" if detection is ambiguous
        
    Examples:
        Automatic detection in environment step:
        >>> mode = detect_api_compatibility_mode()
        >>> if mode == "legacy":
        ...     return obs, reward, done, info
        ... else:
        ...     return obs, reward, terminated, truncated, info
    """
    # Inspect the call stack to detect calling patterns
    frame = inspect.currentframe()
    try:
        # Look up the stack for indicators of legacy vs modern usage
        while frame is not None:
            frame = frame.f_back
            if frame is None:
                break
                
            # Check module globals for gym/gymnasium imports
            frame_globals = frame.f_globals
            module_name = frame_globals.get('__name__', '')
            
            # Legacy indicators
            if 'gym' in frame_globals and 'gymnasium' not in frame_globals:
                return "legacy"
            if 'gym.make' in str(frame.f_code.co_names):
                return "legacy"
            if module_name.startswith('gym.'):
                return "legacy"
                
            # Modern indicators
            if 'gymnasium' in frame_globals:
                return "modern"
            if 'gymnasium.make' in str(frame.f_code.co_names):
                return "modern"
            if module_name.startswith('gymnasium.'):
                return "modern"
    finally:
        del frame
    
    # Default to modern API if detection is ambiguous
    return "modern"


def convert_step_return_format(
    step_return: Tuple[Any, ...], 
    target_format: str
) -> Tuple[Any, ...]:
    """
    Convert between 4-tuple and 5-tuple step return formats.
    
    Args:
        step_return: Step return tuple in either format
        target_format: "legacy" for 4-tuple, "modern" for 5-tuple
        
    Returns:
        Tuple: Converted step return in requested format
        
    Notes:
        Legacy format: (obs, reward, done, info)
        Modern format: (obs, reward, terminated, truncated, info)
        
        When converting modern->legacy: done = terminated or truncated
        When converting legacy->modern: terminated=done, truncated=False
        
    Examples:
        Convert to legacy format:
        >>> modern_return = (obs, reward, True, False, info)
        >>> legacy_return = convert_step_return_format(modern_return, "legacy")
        >>> # Returns: (obs, reward, True, info)
        
        Convert to modern format:
        >>> legacy_return = (obs, reward, True, info)
        >>> modern_return = convert_step_return_format(legacy_return, "modern")
        >>> # Returns: (obs, reward, True, False, info)
    """
    if target_format == "legacy":
        if len(step_return) == 5:
            # Convert from modern (5-tuple) to legacy (4-tuple)
            obs, reward, terminated, truncated, info = step_return
            done = terminated or truncated
            return obs, reward, done, info
        elif len(step_return) == 4:
            # Already legacy format
            return step_return
        else:
            raise ValueError(f"Invalid step return length: {len(step_return)}")
    
    elif target_format == "modern":
        if len(step_return) == 4:
            # Convert from legacy (4-tuple) to modern (5-tuple)
            obs, reward, done, info = step_return
            # In legacy format, done=True could be either termination or truncation
            # We assume termination by default, but check info for hints
            if isinstance(info, dict) and info.get('TimeLimit.truncated', False):
                terminated, truncated = False, True
            else:
                terminated, truncated = done, False
            return obs, reward, terminated, truncated, info
        elif len(step_return) == 5:
            # Already modern format
            return step_return
        else:
            raise ValueError(f"Invalid step return length: {len(step_return)}")
    
    else:
        raise ValueError(f"Invalid target format: {target_format}")


def convert_reset_return_format(
    reset_return: Tuple[Any, ...],
    target_format: str
) -> Tuple[Any, ...]:
    """
    Convert between legacy and modern reset return formats.
    
    Args:
        reset_return: Reset return tuple in either format
        target_format: "legacy" for obs-only, "modern" for (obs, info)
        
    Returns:
        Tuple: Converted reset return in requested format
        
    Notes:
        Legacy format: obs (single return value)
        Modern format: (obs, info)
        
    Examples:
        Convert to legacy format:
        >>> modern_return = (obs, info)
        >>> legacy_return = convert_reset_return_format(modern_return, "legacy")
        >>> # Returns: obs
        
        Convert to modern format:
        >>> legacy_return = obs
        >>> modern_return = convert_reset_return_format((legacy_return,), "modern")
        >>> # Returns: (obs, {})
    """
    if target_format == "legacy":
        if len(reset_return) == 2:
            # Convert from modern (obs, info) to legacy (obs)
            obs, info = reset_return
            return obs
        elif len(reset_return) == 1:
            # Already legacy format (obs only)
            return reset_return[0]
        else:
            raise ValueError(f"Invalid reset return length: {len(reset_return)}")
    
    elif target_format == "modern":
        if len(reset_return) == 1:
            # Convert from legacy (obs) to modern (obs, info)
            obs = reset_return[0]
            return obs, {}
        elif len(reset_return) == 2:
            # Already modern format
            return reset_return
        else:
            raise ValueError(f"Invalid reset return length: {len(reset_return)}")
    
    else:
        raise ValueError(f"Invalid target format: {target_format}")


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

ObservationHookType = Callable[[dict], dict]
"""Type alias for compute_additional_obs hook functions.

Hook functions should accept base_obs dict and return additional observations dict.
Example: def custom_obs_hook(base_obs: dict) -> dict: ...
"""

RewardHookType = Callable[[float, dict], float]
"""Type alias for compute_extra_reward hook functions.

Hook functions should accept base_reward float and info dict, return extra reward float.
Example: def custom_reward_hook(base_reward: float, info: dict) -> float: ...
"""

EpisodeEndHookType = Callable[[dict], None]
"""Type alias for on_episode_end hook functions.

Hook functions should accept final_info dict and return None.
Example: def custom_episode_end_hook(final_info: dict) -> None: ...
"""

# Type aliases for new modular component protocols

PlumeStateType = Any
"""Type alias for plume model state - implementation-specific structure."""

WindVelocityType = Union[Tuple[float, float], List[float], np.ndarray]
"""Type alias for wind velocity vectors - supports tuples, lists, or numpy arrays."""

SensorReadingType = Union[float, bool, np.ndarray, Dict[str, Any]]
"""Type alias for sensor reading outputs - varies by sensor type."""

ObservationDictType = Dict[str, Union[np.ndarray, float, bool, Dict[str, Any]]]
"""Type alias for structured observation dictionaries."""

ActionDictType = Dict[str, Union[np.ndarray, float, Dict[str, Any]]]
"""Type alias for structured action dictionaries."""

ComponentConfigType = Dict[str, Any]
"""Type alias for component configuration dictionaries."""

# Type aliases for new v1.0 component protocols

SourceConfigType = Dict[str, Any]
"""Type alias for source configuration dictionaries."""

BoundaryPolicyConfigType = Dict[str, Any] 
"""Type alias for boundary policy configuration dictionaries."""

ActionInterfaceConfigType = Dict[str, Any]
"""Type alias for action interface configuration dictionaries."""

RecorderConfigType = Dict[str, Any]
"""Type alias for recorder configuration dictionaries."""

StatsConfigType = Dict[str, Any]
"""Type alias for statistics aggregator configuration dictionaries."""

AgentInitConfigType = Dict[str, Any]
"""Type alias for agent initializer configuration dictionaries."""

EmissionRateType = float
"""Type alias for odor emission rate values."""

PositionTupleType = Tuple[float, float]
"""Type alias for 2D position coordinate tuples."""

BoundaryViolationType = np.ndarray
"""Type alias for boundary violation boolean arrays."""

NavigationCommandType = Dict[str, Union[float, str]]
"""Type alias for navigation command dictionaries."""

StepDataType = Dict[str, Any]
"""Type alias for step-level recording data dictionaries."""

EpisodeDataType = Dict[str, Any]
"""Type alias for episode-level recording data dictionaries."""

MetricsType = Dict[str, float]
"""Type alias for calculated statistics and metrics dictionaries."""


# Re-export protocol and factory for public API
__all__ = [
    # Core protocol and factory
    "NavigatorProtocol",
    "NavigatorFactory",
    
    # New v1.0 component protocols for pluggable architecture
    "SourceProtocol",
    "BoundaryPolicyProtocol", 
    "ActionInterfaceProtocol",
    "RecorderProtocol",
    "StatsAggregatorProtocol",
    "AgentInitializerProtocol",
    
    # Existing modular component protocols for pluggable architecture
    "PlumeModelProtocol",
    "WindFieldProtocol", 
    "SensorProtocol",
    "AgentObservationProtocol",
    "AgentActionProtocol",
    "ObservationSpaceProtocol",
    "ActionSpaceProtocol",
    
    # Type aliases
    "PositionType",
    "PositionsType", 
    "OrientationType",
    "OrientationsType",
    "SpeedType",
    "SpeedsType",
    "ConfigType",
    "ObservationHookType",
    "RewardHookType", 
    "EpisodeEndHookType",
    
    # Type aliases for modular components
    "PlumeStateType",
    "WindVelocityType",
    "SensorReadingType", 
    "ObservationDictType",
    "ActionDictType",
    "ComponentConfigType",
    
    # Type aliases for new v1.0 components
    "SourceConfigType",
    "BoundaryPolicyConfigType",
    "ActionInterfaceConfigType", 
    "RecorderConfigType",
    "StatsConfigType",
    "AgentInitConfigType",
    "EmissionRateType",
    "PositionTupleType",
    "BoundaryViolationType",
    "NavigationCommandType",
    "StepDataType",
    "EpisodeDataType",
    "MetricsType",
    
    # API compatibility utilities
    "detect_api_compatibility_mode",
    "convert_step_return_format",
    "convert_reset_return_format",
]