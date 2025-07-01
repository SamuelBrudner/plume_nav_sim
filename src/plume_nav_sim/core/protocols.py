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
- Integration with SpacesFactory utilities for type-safe space construction
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
from typing import Protocol, Union, Optional, Tuple, List, Any, Dict, runtime_checkable
from typing_extensions import Self
import numpy as np
import warnings
import inspect

# Hydra imports for configuration integration
try:
    from omegaconf import DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    # Fallback for environments without Hydra
    DictConfig = dict
    HYDRA_AVAILABLE = False

# Gymnasium imports for modern API support
try:
    import gymnasium
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    # Fallback compatibility
    try:
        import gym as gymnasium
        from gym import spaces
        GYMNASIUM_AVAILABLE = True
    except ImportError:
        gymnasium = None
        spaces = None
        GYMNASIUM_AVAILABLE = False

# Import configuration schemas for type hints - handle case where they don't exist yet
try:
    from ..config.schemas import NavigatorConfig, SingleAgentConfig, MultiAgentConfig
    SCHEMAS_AVAILABLE = True
except ImportError:
    # These will be created by other agents - define minimal fallback types
    NavigatorConfig = Dict[str, Any]
    SingleAgentConfig = Dict[str, Any] 
    MultiAgentConfig = Dict[str, Any]
    SCHEMAS_AVAILABLE = False

# Import spaces factory - handle case where it doesn't exist yet
try:
    from ..envs.spaces import SpacesFactory
    SPACES_FACTORY_AVAILABLE = True
except ImportError:
    # This will be created by other agents - define minimal fallback
    class SpacesFactory:
        @staticmethod
        def create_observation_space(**kwargs):
            """Fallback until real SpacesFactory is available."""
            if GYMNASIUM_AVAILABLE and spaces:
                return spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
            return None
            
        @staticmethod
        def create_action_space(**kwargs):
            """Fallback until real SpacesFactory is available."""
            if GYMNASIUM_AVAILABLE and spaces:
                return spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            return None
    SPACES_FACTORY_AVAILABLE = False


@runtime_checkable
class NavigatorProtocol(Protocol):
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
class PlumeModelProtocol(Protocol):
    """
    Protocol defining the interface for pluggable odor plume modeling implementations.
    
    This protocol enables seamless switching between different plume simulation approaches:
    - GaussianPlumeModel: Fast analytical dispersion calculations
    - TurbulentPlumeModel: Realistic filament-based turbulent physics
    - VideoPlumeAdapter: Backward-compatible video-based plume data
    
    All implementations must provide real-time concentration queries, temporal evolution,
    and state management while maintaining performance requirements for interactive simulation.
    
    Key Design Principles:
    - Spatial sampling via concentration_at() for flexible sensor integration
    - Temporal dynamics via step() for realistic environmental evolution
    - State management via reset() for reproducible experiment episodes
    - WindField integration for realistic transport physics (implementation-specific)
    
    Performance Requirements:
    - concentration_at(): <1ms for single query, <10ms for 100 concurrent agents
    - step(): <5ms per time step for real-time simulation compatibility
    - Memory efficiency: <100MB for typical simulation scenarios
    
    Examples:
        Basic concentration sampling:
        >>> plume_model = GaussianPlumeModel(source_position=(50, 50), strength=1000)
        >>> agent_positions = np.array([[45, 48], [52, 47]])
        >>> concentrations = plume_model.concentration_at(agent_positions)
        
        Temporal simulation:
        >>> for t in range(100):
        ...     plume_model.step(dt=1.0)  # Advance environmental state
        ...     current_concentrations = plume_model.concentration_at(agent_positions)
        
        Configuration-driven instantiation:
        >>> from hydra import compose, initialize
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        ...     plume_model = hydra.utils.instantiate(cfg.plume_model)
    """
    
    def concentration_at(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute odor concentrations at specified spatial locations.
        
        Args:
            positions: Agent positions as array with shape (n_agents, 2) for multiple
                agents or (2,) for single agent. Coordinates in environment units.
                
        Returns:
            np.ndarray: Concentration values with shape (n_agents,) or scalar for
                single agent. Values normalized to [0, 1] range representing
                relative odor intensity.
                
        Notes:
            Uses spatial interpolation for sub-pixel accuracy when applicable.
            Positions outside plume boundaries return 0.0 concentration.
            Implementation may cache results for performance optimization.
            
        Performance:
            Must execute in <1ms for single query, <10ms for 100 agents.
            
        Examples:
            Single agent query:
            >>> position = np.array([10.5, 20.3])
            >>> concentration = plume_model.concentration_at(position)
            
            Multi-agent batch query:
            >>> positions = np.array([[10, 20], [15, 25], [20, 30]])
            >>> concentrations = plume_model.concentration_at(positions)
        """
        ...
    
    def step(self, dt: float = 1.0) -> None:
        """
        Advance plume state by specified time delta.
        
        Args:
            dt: Time step size in seconds. Controls temporal resolution of
                environmental dynamics including dispersion, transport, and
                source evolution.
                
        Notes:
            Updates internal plume state including:
            - Dispersion dynamics and spatial evolution
            - Source strength variations (if applicable)
            - WindField integration for transport effects
            - Turbulent mixing and dissipation processes
            
            Implementation may optimize for performance by batching updates
            or using analytical solutions where possible.
            
        Performance:
            Must complete in <5ms per step for real-time simulation compatibility.
            
        Examples:
            Standard time step:
            >>> plume_model.step(dt=1.0)
            
            High-frequency simulation:
            >>> for _ in range(10):
            ...     plume_model.step(dt=0.1)  # 10x higher temporal resolution
        """
        ...
    
    def reset(self, **kwargs: Any) -> None:
        """
        Reset plume state to initial conditions.
        
        Args:
            **kwargs: Optional parameters to override initial settings.
                Common options include:
                - source_position: New source location (x, y)
                - source_strength: Initial emission rate
                - wind_conditions: WindField configuration updates
                - boundary_conditions: Spatial domain parameters
                
        Notes:
            Reinitializes all plume state while preserving model configuration.
            Parameter overrides are applied for this episode only unless
            explicitly configured for persistence.
            
            WindField integration is reset to initial conditions with any
            specified parameter updates applied.
            
        Performance:
            Should complete in <10ms to avoid blocking episode initialization.
            
        Examples:
            Reset to default initial state:
            >>> plume_model.reset()
            
            Reset with new source location:
            >>> plume_model.reset(source_position=(25, 75), source_strength=1500)
        """
        ...


@runtime_checkable
class WindFieldProtocol(Protocol):
    """
    Protocol defining environmental wind dynamics for realistic plume transport modeling.
    
    This protocol enables configurable wind field implementations supporting various
    levels of environmental realism:
    - ConstantWindField: Uniform directional flow with minimal computational overhead
    - TurbulentWindField: Realistic atmospheric boundary layer with gusty conditions
    - TimeVaryingWindField: Dynamic wind patterns with temporal evolution
    - MeasuredWindField: Data-driven wind from meteorological measurements
    
    WindField implementations integrate with PlumeModel components to provide realistic
    transport physics affecting odor dispersion patterns and navigation challenges.
    
    Key Design Principles:
    - Spatial velocity queries via velocity_at() for plume transport calculations
    - Temporal evolution via step() for dynamic environmental conditions
    - State management via reset() for reproducible experimental conditions
    - Performance optimization for real-time simulation requirements
    
    Performance Requirements:
    - velocity_at(): <0.5ms for single query, <5ms for spatial field evaluation
    - step(): <2ms per time step for minimal simulation overhead
    - Memory efficiency: <50MB for typical wind field representations
    
    Examples:
        Basic wind field query:
        >>> wind_field = ConstantWindField(velocity=(2.0, 0.5))  # East-northeast wind
        >>> positions = np.array([[10, 20], [15, 25]])
        >>> velocities = wind_field.velocity_at(positions)
        
        Turbulent wind simulation:
        >>> turbulent_wind = TurbulentWindField(mean_velocity=(3.0, 1.0), turbulence_intensity=0.2)
        >>> for t in range(100):
        ...     turbulent_wind.step(dt=1.0)
        ...     current_velocities = turbulent_wind.velocity_at(positions)
        
        Configuration-driven instantiation:
        >>> wind_field = hydra.utils.instantiate(cfg.wind_field)
    """
    
    def velocity_at(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute wind velocity vectors at specified spatial locations.
        
        Args:
            positions: Spatial positions as array with shape (n_positions, 2) for
                multiple locations or (2,) for single position. Coordinates in
                environment units.
                
        Returns:
            np.ndarray: Velocity vectors with shape (n_positions, 2) or (2,) for
                single position. Components represent [u_x, u_y] in environment
                units per time step.
                
        Notes:
            Velocity components follow standard meteorological conventions:
            - u_x: eastward wind component (positive = eastward)
            - u_y: northward wind component (positive = northward)
            
            Spatial interpolation provides smooth velocity fields for
            realistic plume transport physics.
            
        Performance:
            Must execute in <0.5ms for single query, <5ms for field evaluation.
            
        Examples:
            Single position query:
            >>> position = np.array([25.5, 35.2])
            >>> velocity = wind_field.velocity_at(position)
            
            Spatial field evaluation:
            >>> positions = np.array([[x, y] for x in range(0, 100, 10) 
            ...                                 for y in range(0, 100, 10)])
            >>> velocity_field = wind_field.velocity_at(positions)
        """
        ...
    
    def step(self, dt: float = 1.0) -> None:
        """
        Advance wind field temporal dynamics by specified time delta.
        
        Args:
            dt: Time step size in seconds. Controls temporal resolution of
                wind evolution including gust development, pressure changes,
                and atmospheric boundary layer dynamics.
                
        Notes:
            Updates wind field state including:
            - Turbulent eddy evolution and mixing processes
            - Pressure gradient effects and geostrophic adjustments
            - Boundary layer development and thermal effects
            - Gust formation and dissipation patterns
            
            Constant wind fields may have minimal or no temporal evolution,
            while turbulent implementations update stochastic flow patterns.
            
        Performance:
            Must complete in <2ms per step for minimal simulation overhead.
            
        Examples:
            Standard temporal evolution:
            >>> wind_field.step(dt=1.0)
            
            High-frequency dynamics:
            >>> for _ in range(10):
            ...     wind_field.step(dt=0.1)  # Resolve fast turbulent processes
        """
        ...
    
    def reset(self, **kwargs: Any) -> None:
        """
        Reset wind field to initial conditions with optional parameter updates.
        
        Args:
            **kwargs: Optional parameters to override initial settings.
                Common options include:
                - mean_velocity: Base wind vector (u_x, u_y)
                - turbulence_intensity: Relative turbulence strength [0, 1]
                - boundary_conditions: Spatial domain wind constraints
                - atmospheric_stability: Stability class for boundary layer physics
                
        Notes:
            Reinitializes wind field state while preserving model configuration.
            Parameter overrides apply to current episode only unless configured
            for persistence across episodes.
            
            Stochastic wind fields regenerate random components with controlled
            statistical properties matching specified parameters.
            
        Performance:
            Should complete in <5ms to avoid blocking episode initialization.
            
        Examples:
            Reset to default conditions:
            >>> wind_field.reset()
            
            Reset with stronger wind:
            >>> wind_field.reset(mean_velocity=(5.0, 2.0), turbulence_intensity=0.3)
        """
        ...


@runtime_checkable
class SensorProtocol(Protocol):
    """
    Protocol defining configurable sensor interfaces for flexible agent perception modeling.
    
    This protocol enables diverse sensing modalities without modifying core navigation logic:
    - BinarySensor: Threshold-based detection with configurable false positive/negative rates
    - ConcentrationSensor: Quantitative measurements with dynamic range and noise modeling
    - GradientSensor: Spatial derivative computation for directional navigation cues
    - HistoricalSensor: Temporal integration wrapper for memory-based sensing strategies
    
    SensorProtocol implementations interact with PlumeModel components to provide realistic
    sensing limitations, noise characteristics, and temporal dynamics matching real-world
    chemical sensors and biological detection systems.
    
    Key Design Principles:
    - Modular sensing via detect()/measure()/compute() for different sensor types
    - Plume model integration via standardized concentration field sampling
    - Configurable noise and response characteristics for realistic sensor modeling
    - Optional temporal history for memory-based navigation strategies
    
    Performance Requirements:
    - Sensor operations: <0.1ms per agent per sensor for minimal overhead
    - Batch processing: <1ms for 100 agents with multiple sensors
    - Memory efficiency: <10MB for historical data with configurable limits
    
    Examples:
        Binary odor detection:
        >>> binary_sensor = BinarySensor(threshold=0.1, false_positive_rate=0.02)
        >>> detections = binary_sensor.detect(plume_state, agent_positions)
        
        Quantitative concentration measurement:
        >>> conc_sensor = ConcentrationSensor(dynamic_range=(0, 1), resolution=0.001)
        >>> concentrations = conc_sensor.measure(plume_state, agent_positions)
        
        Spatial gradient computation:
        >>> gradient_sensor = GradientSensor(spatial_resolution=(0.5, 0.5))
        >>> gradients = gradient_sensor.compute_gradient(plume_state, agent_positions)
        
        Configuration-driven sensor setup:
        >>> sensors = [hydra.utils.instantiate(cfg) for cfg in cfg.sensors]
    """
    
    def detect(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        """
        Perform binary detection at specified agent positions (BinarySensor implementation).
        
        Args:
            plume_state: Current plume model state providing concentration field access.
                Typically a PlumeModel instance or spatial concentration array.
            positions: Agent positions as array with shape (n_agents, 2) or (2,) for
                single agent. Coordinates in environment units.
                
        Returns:
            np.ndarray: Boolean detection results with shape (n_agents,) or scalar
                for single agent. True indicates odor detection above threshold.
                
        Notes:
            Binary sensors apply configurable thresholds with optional hysteresis
            to prevent detection oscillations. Noise modeling includes false
            positive and false negative rates matching real sensor characteristics.
            
            Non-binary sensors may return detection status based on measurement
            confidence or signal-to-noise ratio criteria.
            
        Performance:
            Must execute in <0.1ms per agent for minimal sensing overhead.
            
        Examples:
            Single agent detection:
            >>> position = np.array([15, 25])
            >>> detected = sensor.detect(plume_state, position)
            
            Multi-agent batch detection:
            >>> positions = np.array([[10, 20], [15, 25], [20, 30]])
            >>> detections = sensor.detect(plume_state, positions)
        """
        ...
    
    def measure(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        """
        Perform quantitative measurements at specified agent positions (ConcentrationSensor).
        
        Args:
            plume_state: Current plume model state providing concentration field access.
            positions: Agent positions as array with shape (n_agents, 2) or (2,) for
                single agent. Coordinates in environment units.
                
        Returns:
            np.ndarray: Quantitative measurement values with shape (n_agents,) or
                scalar for single agent. Values in sensor-specific units with
                configured dynamic range and resolution.
                
        Notes:
            Concentration sensors provide calibrated measurements with configurable
            dynamic range, resolution, and noise characteristics. Temporal filtering
            and response delays model realistic sensor dynamics.
            
            Measurements may include saturation effects and calibration drift
            for enhanced realism in long-duration experiments.
            
        Performance:
            Must execute in <0.1ms per agent for minimal sensing overhead.
            
        Examples:
            Single agent measurement:
            >>> position = np.array([15, 25])
            >>> concentration = sensor.measure(plume_state, position)
            
            Multi-agent batch measurement:
            >>> positions = np.array([[10, 20], [15, 25], [20, 30]])
            >>> concentrations = sensor.measure(plume_state, positions)
        """
        ...
    
    def compute_gradient(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        """
        Compute spatial gradients at specified agent positions (GradientSensor implementation).
        
        Args:
            plume_state: Current plume model state providing concentration field access.
            positions: Agent positions as array with shape (n_agents, 2) or (2,) for
                single agent. Coordinates in environment units.
                
        Returns:
            np.ndarray: Gradient vectors with shape (n_agents, 2) or (2,) for single
                agent. Components represent [∂c/∂x, ∂c/∂y] spatial derivatives in
                concentration units per distance unit.
                
        Notes:
            Gradient sensors use finite difference methods with configurable spatial
            resolution and derivative order. Multi-point sampling enables accurate
            gradient estimation with noise suppression.
            
            Adaptive step sizing and error estimation provide robust gradient
            computation in regions with sharp concentration variations.
            
        Performance:
            Must execute in <0.2ms per agent due to multi-point sampling requirements.
            
        Examples:
            Single agent gradient:
            >>> position = np.array([15, 25])
            >>> gradient = sensor.compute_gradient(plume_state, position)
            
            Multi-agent batch gradients:
            >>> positions = np.array([[10, 20], [15, 25], [20, 30]])
            >>> gradients = sensor.compute_gradient(plume_state, positions)
        """
        ...
    
    def configure(self, **kwargs: Any) -> None:
        """
        Update sensor configuration parameters during runtime.
        
        Args:
            **kwargs: Sensor-specific configuration parameters. Common options:
                - threshold: Detection threshold for binary sensors
                - dynamic_range: Measurement range for concentration sensors
                - spatial_resolution: Finite difference step size for gradient sensors
                - noise_parameters: False positive/negative rates, measurement noise
                - temporal_filtering: Response time constants and history length
                
        Notes:
            Configuration updates apply immediately to subsequent sensor operations.
            Parameter validation ensures physical consistency and performance requirements.
            
            Temporal parameters may trigger reset of internal state buffers
            for clean transition to new configuration.
            
        Examples:
            Update binary sensor threshold:
            >>> sensor.configure(threshold=0.05, false_positive_rate=0.01)
            
            Adjust concentration sensor range:
            >>> sensor.configure(dynamic_range=(0, 2.0), resolution=0.0001)
            
            Configure gradient sensor resolution:
            >>> sensor.configure(spatial_resolution=(0.2, 0.2), method='central')
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
    - Integration with SpacesFactory utilities for type-safe space construction
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
            if SCHEMAS_AVAILABLE:
                config = NavigatorConfig(**config)
            # If schemas not available yet, work with dict directly
        elif hasattr(config, 'to_container') and HYDRA_AVAILABLE:
            # Handle DictConfig from Hydra
            if SCHEMAS_AVAILABLE:
                config_dict = config.to_container(resolve=True)
                config = NavigatorConfig(**config_dict)
            else:
                # Work with DictConfig directly
                pass
        
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
        
        Integrates with SpacesFactory utilities to create type-safe observation
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
            
        return SpacesFactory.create_observation_space(
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
        
        Integrates with SpacesFactory utilities to create type-safe action
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
            
        return SpacesFactory.create_action_space(
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
    def create_modular_environment(
        navigator_config: Union[DictConfig, dict],
        plume_model_config: Union[DictConfig, dict],
        wind_field_config: Optional[Union[DictConfig, dict]] = None,
        sensor_configs: Optional[List[Union[DictConfig, dict]]] = None,
        **env_kwargs: Any
    ) -> 'NavigatorProtocol':
        """
        Create complete modular navigation environment with all components.
        
        Args:
            navigator_config: Navigator configuration.
            plume_model_config: Plume model configuration.
            wind_field_config: Optional wind field configuration.
            sensor_configs: Optional list of sensor configurations.
            **env_kwargs: Additional environment parameters.
            
        Returns:
            NavigatorProtocol: Complete navigation environment with all components.
            
        Examples:
            Complete modular environment:
            >>> env = NavigatorFactory.create_modular_environment(
            ...     navigator_config={'position': (0, 0), 'max_speed': 2.0},
            ...     plume_model_config={'type': 'GaussianPlumeModel', 'source_position': (50, 50)},
            ...     wind_field_config={'type': 'ConstantWindField', 'velocity': (1.0, 0.0)},
            ...     sensor_configs=[{'type': 'ConcentrationSensor', 'dynamic_range': (0, 1)}]
            ... )
        """
        # Create individual components
        navigator = NavigatorFactory.from_config(navigator_config)
        plume_model = NavigatorFactory.create_plume_model(plume_model_config)
        
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
        """
        return isinstance(component, protocol_type)


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

ObservationHookType = callable
"""Type alias for compute_additional_obs hook functions."""

RewardHookType = callable
"""Type alias for compute_extra_reward hook functions."""

EpisodeEndHookType = callable
"""Type alias for on_episode_end hook functions."""

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


# Re-export protocol and factory for public API
__all__ = [
    # Core protocol and factory
    "NavigatorProtocol",
    "NavigatorFactory",
    
    # Modular component protocols for pluggable architecture
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
    
    # API compatibility utilities
    "detect_api_compatibility_mode",
    "convert_step_return_format",
    "convert_reset_return_format",
]