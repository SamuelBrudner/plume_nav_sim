"""
Unified Navigator module providing comprehensive navigation capabilities for Gymnasium 0.29.x migration.

This module serves as the primary entry point for plume navigation functionality,
combining the core Navigator class with enhanced factory-based instantiation patterns
for configuration-driven component creation. Enhanced for Gymnasium 0.29.x with
modern extensibility hooks, dual API compatibility, and integrated frame caching.

Key Features:
    - Navigator: Unified navigation class supporting single and multi-agent scenarios
    - NavigatorFactory: Enhanced factory for Hydra-based configuration instantiation
    - Gymnasium 0.29.x compatibility with extensibility hooks
    - Dual API support for legacy OpenAI Gym and modern Gymnasium
    - Performance optimization with frame caching integration
    - Type-safe configuration management with comprehensive validation

Enhanced Extensibility Hooks:
    - compute_additional_obs(): Custom observation augmentation
    - compute_extra_reward(): Custom reward shaping logic
    - on_episode_end(): Episode completion handling and logging

Integration Features:
    - Frame caching with LRU eviction and memory monitoring
    - Hydra configuration system with structured parameter validation
    - SpacesFactory integration for type-safe observation/action spaces
    - Performance monitoring with sub-33ms step execution guarantees
    - Memory-efficient operations supporting 100+ concurrent agents

Examples:
    Basic single-agent navigation:
    
    >>> from plume_nav_sim.core.navigator import Navigator
    >>> navigator = Navigator.single(position=(10, 20), speed=1.5)
    >>> navigator.reset()
    >>> navigator.step(env_array, dt=1.0)
    
    Configuration-driven instantiation with Gymnasium features:
    
    >>> from plume_nav_sim.core.navigator import NavigatorFactory
    >>> factory = NavigatorFactory()
    >>> navigator = factory.from_config(config_dict)
    >>> obs_space = factory.create_observation_space(navigator)
    
    Multi-agent swarm with extensibility hooks:
    
    >>> positions = [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]
    >>> navigator = NavigatorFactory.multi_agent(
    ...     positions=positions,
    ...     enable_extensibility_hooks=True,
    ...     frame_cache_mode="lru"
    ... )
    
    Custom navigation with reward shaping:
    
    >>> class CustomNavigator(Navigator):
    ...     def compute_extra_reward(self, base_reward: float, info: dict) -> float:
    ...         return 0.1 * info.get('exploration_bonus', 0.0)
    ...     
    ...     def compute_additional_obs(self, base_obs: dict) -> dict:
    ...         return {"wind_direction": self.sample_wind_sensor()}

Integration:
    This module integrates NavigatorProtocol interface implementations with application-layer
    factory methods to support unified package architecture. It provides backward compatibility
    with existing Navigator imports while enabling new Gymnasium-based configuration workflows,
    extensibility hooks, and enhanced performance monitoring.
    
    The module maintains full compatibility with the enhanced frame caching system featuring
    configurable modes, memory pressure monitoring, and intelligent eviction policies for
    optimal performance in resource-constrained environments.
"""

from __future__ import annotations
import warnings
from typing import Optional, Union, Tuple, List, Any, Dict, ClassVar
import numpy as np

# Core protocol and controller imports
from plume_nav_sim.protocols.navigator import NavigatorProtocol
from .protocols import NavigatorFactory as BaseNavigatorFactory
from .protocols import (
    PositionType, PositionsType, OrientationType, OrientationsType,
    SpeedType, SpeedsType, ConfigType, ObservationHookType, 
    RewardHookType, EpisodeEndHookType
)

# Controller implementations
import logging

logger = logging.getLogger(__name__)

try:
    from .controllers import SingleAgentController, MultiAgentController
except ImportError as exc:
    logger.error(
        "Navigator requires 'plume_nav_sim.core.controllers' to be available.",
        exc_info=True,
    )
    raise ImportError(
        "plume_nav_sim.core.controllers could not be imported."
    ) from exc

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


class Navigator:
    """
    Unified Navigator class providing comprehensive navigation capabilities.
    
    This class serves as the primary entry point for navigation functionality,
    wrapping NavigatorProtocol implementations with enhanced Gymnasium 0.29.x
    features, extensibility hooks, and performance optimization. Designed to
    support both single-agent and multi-agent scenarios with consistent API.
    
    Enhanced Features for Gymnasium 0.29.x Migration:
    - Extensibility hooks for custom observations, rewards, and episode handling
    - Dual API compatibility with automatic format conversion
    - Frame caching integration with configurable modes and memory monitoring
    - Performance optimization with sub-33ms step execution guarantees
    - Type-safe configuration management with Hydra integration
    
    The Navigator class maintains backward compatibility while providing modern
    capabilities required for advanced research workflows and production deployment.
    It integrates seamlessly with the enhanced environment infrastructure including
    FrameCache performance systems and SpacesFactory utilities.
    
    Attributes:
        controller: Underlying NavigatorProtocol implementation
        factory: NavigatorFactory instance for space creation
        enable_extensibility_hooks: Flag for custom hook execution
        frame_cache_mode: Frame caching strategy ("none", "lru", "preload")
        custom_observation_keys: List of additional observation keys
        reward_shaping: Custom reward shaping configuration
    
    Performance Requirements:
    - Single agent step execution: <1ms per step
    - Multi-agent step execution: <10ms for 100 agents
    - Memory efficiency: <10MB overhead per 100 agents
    - Cache hit rate: >90% with optimal frame caching configuration
    
    Examples:
        Single-agent navigation with extensibility:
        >>> navigator = Navigator(
        ...     position=(10.0, 20.0),
        ...     speed=1.5,
        ...     enable_extensibility_hooks=True,
        ...     frame_cache_mode="lru"
        ... )
        >>> navigator.reset()
        >>> navigator.step(env_array, dt=1.0)
        
        Multi-agent swarm with performance monitoring:
        >>> positions = [[0, 0], [10, 10], [20, 20]]
        >>> navigator = Navigator(
        ...     positions=positions,
        ...     enable_vectorized_ops=True
        ... )
        >>> metrics = navigator.get_performance_metrics()
        
        Custom observation and reward integration:
        >>> class ResearchNavigator(Navigator):
        ...     def compute_additional_obs(self, base_obs: dict) -> dict:
        ...         return {"wind_direction": self.sample_wind_sensor()}
        ...     
        ...     def compute_extra_reward(self, base_reward: float, info: dict) -> float:
        ...         exploration_bonus = 0.1 * info.get('exploration_score', 0)
        ...         return exploration_bonus
    """
    
    def __init__(
        self,
        controller: Optional[NavigatorProtocol] = None,
        enable_extensibility_hooks: bool = False,
        frame_cache_mode: Optional[str] = None,
        custom_observation_keys: Optional[List[str]] = None,
        reward_shaping: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize Navigator with enhanced Gymnasium 0.29.x features.
        
        Args:
            controller: Pre-configured NavigatorProtocol implementation.
                If None, will be created from kwargs.
            enable_extensibility_hooks: Enable custom observation/reward hooks
            frame_cache_mode: Frame caching strategy ("none", "lru", "preload")
            custom_observation_keys: Additional observation keys to include
            reward_shaping: Custom reward shaping configuration
            **kwargs: Additional parameters for controller creation
        
        Notes:
            If controller is None, creates appropriate controller based on
            presence of 'positions' (multi-agent) vs 'position' (single-agent)
            in kwargs. All enhanced features are automatically configured.
        """
        self.enable_extensibility_hooks = enable_extensibility_hooks
        self.frame_cache_mode = frame_cache_mode
        self.custom_observation_keys = custom_observation_keys or []
        self.reward_shaping = reward_shaping
        
        # Create controller if not provided
        if controller is None:
            controller = self._create_controller_from_kwargs(**kwargs)
        
        self.controller = controller
        self.factory = NavigatorFactory()
    
    def _create_controller_from_kwargs(self, **kwargs: Any) -> NavigatorProtocol:
        """Create appropriate controller based on kwargs configuration."""
        # Determine if multi-agent based on positions structure
        if 'positions' in kwargs:
            positions = kwargs['positions']
            
            # Check if positions indicate multi-agent or single-agent
            is_multi_agent = self._detect_multi_agent_mode(positions)
            
            if is_multi_agent:
                return MultiAgentController(
                    enable_extensibility_hooks=self.enable_extensibility_hooks,
                    frame_cache_mode=self.frame_cache_mode,
                    custom_observation_keys=self.custom_observation_keys,
                    reward_shaping=self.reward_shaping,
                    **kwargs
                )
            else:
                # Convert single-agent positions format and use position/y instead
                if isinstance(positions, (list, tuple)) and len(positions) == 2:
                    kwargs_copy = kwargs.copy()
                    kwargs_copy.pop('positions')
                    kwargs_copy['position'] = positions
                    return SingleAgentController(
                        enable_extensibility_hooks=self.enable_extensibility_hooks,
                        frame_cache_mode=self.frame_cache_mode,
                        custom_observation_keys=self.custom_observation_keys,
                        reward_shaping=self.reward_shaping,
                        **kwargs_copy
                    )
                else:
                    return MultiAgentController(
                        enable_extensibility_hooks=self.enable_extensibility_hooks,
                        frame_cache_mode=self.frame_cache_mode,
                        custom_observation_keys=self.custom_observation_keys,
                        reward_shaping=self.reward_shaping,
                        **kwargs
                    )
        else:
            return SingleAgentController(
                enable_extensibility_hooks=self.enable_extensibility_hooks,
                frame_cache_mode=self.frame_cache_mode,
                custom_observation_keys=self.custom_observation_keys,
                reward_shaping=self.reward_shaping,
                **kwargs
            )
    
    # Delegate core NavigatorProtocol methods to controller
    
    @property
    def positions(self) -> np.ndarray:
        """Get current agent position(s) as numpy array."""
        return self.controller.positions
    
    @property
    def orientations(self) -> np.ndarray:
        """Get current agent orientation(s) in degrees."""
        return self.controller.orientations
    
    @property
    def speeds(self) -> np.ndarray:
        """Get current agent speed(s) in units per time step."""
        return self.controller.speeds
    
    @property
    def max_speeds(self) -> np.ndarray:
        """Get maximum allowed speed(s) for each agent."""
        return self.controller.max_speeds
    
    @property
    def angular_velocities(self) -> np.ndarray:
        """Get current agent angular velocity/velocities in degrees per second."""
        return self.controller.angular_velocities
    
    @property
    def num_agents(self) -> int:
        """Get the total number of agents managed by this navigator."""
        return self.controller.num_agents
    
    def reset(self, **kwargs: Any) -> None:
        """
        Reset navigator to initial state with optional parameter overrides.
        
        Enhanced for Gymnasium 0.29.x with extensibility hook integration and
        frame cache configuration support.
        
        Args:
            **kwargs: Optional parameters to override initial settings.
                Supports all NavigatorProtocol reset parameters plus enhanced
                configuration options for extensibility and caching.
        
        Examples:
            Basic reset:
            >>> navigator.reset()
            
            Reset with new configuration:
            >>> navigator.reset(
            ...     position=(20.0, 30.0),
            ...     frame_cache_mode="lru",
            ...     enable_extensibility_hooks=True
            ... )
        """
        # Update enhanced configuration if provided
        if 'enable_extensibility_hooks' in kwargs:
            self.enable_extensibility_hooks = kwargs.pop('enable_extensibility_hooks')
        if 'frame_cache_mode' in kwargs:
            self.frame_cache_mode = kwargs.pop('frame_cache_mode')
        if 'custom_observation_keys' in kwargs:
            self.custom_observation_keys = kwargs.pop('custom_observation_keys')
        if 'reward_shaping' in kwargs:
            self.reward_shaping = kwargs.pop('reward_shaping')
        
        # Delegate to controller
        self.controller.reset(**kwargs)
    
    def step(self, env_array: np.ndarray, dt: float = 1.0) -> None:
        """
        Execute one simulation time step with environment interaction.
        
        Enhanced for Gymnasium 0.29.x with performance monitoring and
        extensibility hook integration for custom navigation logic.
        
        Args:
            env_array: Environment data array (e.g., odor plume frame)
            dt: Time step size in seconds (default: 1.0)
        
        Notes:
            Automatically integrates with frame caching system if available
            and configured. Executes custom extensibility hooks if enabled.
        """
        self.controller.step(env_array, dt)
    
    def sample_odor(self, env_array: np.ndarray) -> Union[float, np.ndarray]:
        """
        Sample odor concentration(s) at current agent position(s).
        
        Args:
            env_array: Environment array containing odor concentration data
            
        Returns:
            Union[float, np.ndarray]: Odor concentration value(s)
        """
        return self.controller.sample_odor(env_array)
    
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
            env_array: Environment array containing odor concentration data
            sensor_distance: Distance from agent center to each sensor
            sensor_angle: Angular separation between sensors in degrees  
            num_sensors: Number of sensors per agent
            layout_name: Predefined sensor layout name
            
        Returns:
            np.ndarray: Sensor readings array
        """
        return self.controller.sample_multiple_sensors(
            env_array, 
            sensor_distance=sensor_distance,
            sensor_angle=sensor_angle,
            num_sensors=num_sensors,
            layout_name=layout_name
        )

    def read_single_antenna_odor(
        self,
        positions_or_plume: Union[np.ndarray, Any],
        plume: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        """Read odor concentration using a single antenna interface.

        This method acts as a small shim around the concentration sensor to
        preserve backwards compatibility with legacy code that expected a
        ``read_single_antenna_odor`` method on the navigator.  The method is
        intentionally lightweight and performs only minimal validation so that
        tests can exercise the concentration sensing pathway directly.

        Two invocation patterns are supported::

            navigator.read_single_antenna_odor(plume_array)
            navigator.read_single_antenna_odor(positions, plume_array)

        When a single argument is provided it is treated as the plume or
        environment array and the navigator's internal positions are used.  If
        two arguments are supplied the first is interpreted as an explicit array
        of positions (``N×2``) and the second as the plume/environment.

        Parameters
        ----------
        positions_or_plume:
            Either an ``(N, 2)`` array of positions or the plume/environment
            array.
        plume:
            Optional plume/environment array when ``positions_or_plume``
            contains explicit positions.

        Returns
        -------
        Union[float, np.ndarray]
            Odor concentration for each queried position.  A scalar ``float``
            is returned for single-agent navigators for convenience.
        """

        if plume is None:
            plume_state = positions_or_plume
            positions = self.positions
        else:
            positions = np.asarray(positions_or_plume, dtype=float)
            plume_state = plume

        # If the plume state exposes a concentration_at interface we can defer to
        # it directly.  Otherwise assume ``plume_state`` is an ndarray and sample
        # values via integer indexing.  This mirrors the behaviour relied upon by
        # legacy tests where environments are provided as 2‑D arrays.
        if hasattr(plume_state, "concentration_at"):
            values = plume_state.concentration_at(positions)
        else:
            env = np.asarray(plume_state)
            h, w = env.shape[:2]
            x = np.clip(np.floor(positions[:, 0]).astype(int), 0, w - 1)
            y = np.clip(np.floor(positions[:, 1]).astype(int), 0, h - 1)
            values = env[y, x]
            if getattr(env, "dtype", None) == np.uint8:
                values = values.astype(np.float64) / 255.0

        # Return a scalar for the single‑agent case to match historical API
        if self.num_agents == 1 and np.ndim(values) > 0:
            return float(values[0])
        return values

    def observe(self, plume_state: np.ndarray) -> Dict[str, Any]:
        """Generate a minimal observation dictionary.

        The full navigation stack supports a rich, multi‑modal observation
        space.  For the purposes of these tests we only expose odor
        concentration via a :class:`ConcentrationSensor` and basic kinematic
        information including angular velocity.  Downstream consumers can build
        more elaborate observations by subclassing this method.
        """

        concentration = self.read_single_antenna_odor(plume_state)

        obs: Dict[str, Any] = {
            "concentration": concentration,
            "kinematics": {
                "position": self.positions.copy(),
                "orientation": self.orientations.copy(),
                "speed": self.speeds.copy(),
                "angular_velocity": self.angular_velocities.copy(),
            },
        }

        # For single agent convenience unwrap 1‑element arrays
        if self.num_agents == 1:
            kin = obs["kinematics"]
            for key, val in kin.items():
                kin[key] = float(val[0])
            obs["concentration"] = float(np.atleast_1d(concentration)[0])

        return obs
    
    # Enhanced extensibility hooks for Gymnasium 0.29.x migration
    
    def compute_additional_obs(self, base_obs: dict) -> dict:
        """
        Compute additional observations for custom environment extensions.
        
        This extensibility hook allows downstream implementations to augment 
        the base observation with domain-specific data without modifying the 
        core navigation logic. Override this method to add custom sensors,
        derived metrics, or specialized data processing.
        
        Args:
            base_obs: Base observation dict containing standard navigation data
                
        Returns:
            dict: Additional observation components to merge with base_obs
            
        Notes:
            Default implementation delegates to controller if extensibility
            hooks are enabled, otherwise returns empty dict.
            
        Examples:
            Custom sensor integration:
            >>> def compute_additional_obs(self, base_obs: dict) -> dict:
            ...     return {
            ...         "wind_direction": self.sample_wind_direction(),
            ...         "distance_to_wall": self.compute_wall_distance(),
            ...         "energy_level": self.get_energy_remaining()
            ...     }
        """
        if self.enable_extensibility_hooks:
            return self.controller.compute_additional_obs(base_obs)
        return {}
    
    def compute_extra_reward(self, base_reward: float, info: dict) -> float:
        """
        Compute additional reward components for custom reward shaping.
        
        This extensibility hook enables reward shaping and custom reward
        function implementations without modifying core environment logic.
        Override this method to implement domain-specific incentives,
        exploration bonuses, or multi-objective reward functions.
        
        Args:
            base_reward: Base reward computed by the environment
            info: Environment info dict containing episode state and metrics
                
        Returns:
            float: Additional reward component to add to base_reward
            
        Notes:
            Default implementation delegates to controller if extensibility
            hooks are enabled, otherwise returns 0.0.
            
        Examples:
            Exploration bonus:
            >>> def compute_extra_reward(self, base_reward: float, info: dict) -> float:
            ...     if self.is_novel_position(self.positions[-1]):
            ...         return 0.1
            ...     return 0.0
        """
        if self.enable_extensibility_hooks:
            return self.controller.compute_extra_reward(base_reward, info)
        return 0.0
    
    def on_episode_end(self, final_info: dict) -> None:
        """
        Handle episode completion events for logging and cleanup.
        
        This extensibility hook is called when an episode terminates or
        truncates, providing an opportunity for custom logging, metric
        computation, state persistence, or cleanup operations.
        
        Args:
            final_info: Final environment info dict containing episode summary
                
        Notes:
            Default implementation delegates to controller if extensibility
            hooks are enabled, otherwise performs no action.
            
        Examples:
            Custom metric logging:
            >>> def on_episode_end(self, final_info: dict) -> None:
            ...     episode_length = final_info.get('episode_length', 0)
            ...     success_rate = final_info.get('success', False)
            ...     logger.info(f"Episode ended: length={episode_length}")
        """
        if self.enable_extensibility_hooks:
            self.controller.on_episode_end(final_info)
    
    # Enhanced factory methods for streamlined creation
    
    @classmethod
    def single(
        cls,
        position: Optional[Tuple[float, float]] = None,
        orientation: float = 0.0,
        speed: float = 0.0,
        max_speed: float = 1.0,
        angular_velocity: float = 0.0,
        enable_extensibility_hooks: bool = False,
        frame_cache_mode: Optional[str] = None,
        **kwargs: Any
    ) -> 'Navigator':
        """
        Create single-agent Navigator with enhanced Gymnasium 0.29.x features.
        
        Args:
            position: Initial (x, y) position (default: (0, 0))
            orientation: Initial orientation in degrees (default: 0.0)
            speed: Initial speed (default: 0.0)
            max_speed: Maximum allowed speed (default: 1.0)
            angular_velocity: Initial angular velocity in deg/s (default: 0.0)
            enable_extensibility_hooks: Enable custom observation/reward hooks
            frame_cache_mode: Frame caching strategy ("none", "lru", "preload")
            **kwargs: Additional configuration options
            
        Returns:
            Navigator: Single-agent navigator with modern features
            
        Examples:
            Basic single agent:
            >>> navigator = Navigator.single()
            
            Enhanced single agent with caching:
            >>> navigator = Navigator.single(
            ...     position=(50.0, 100.0),
            ...     max_speed=2.5,
            ...     enable_extensibility_hooks=True,
            ...     frame_cache_mode="lru"
            ... )
        """
        controller = SingleAgentController(
            position=position,
            orientation=orientation,
            speed=speed,
            max_speed=max_speed,
            angular_velocity=angular_velocity,
            enable_extensibility_hooks=enable_extensibility_hooks,
            frame_cache_mode=frame_cache_mode,
            **kwargs
        )
        
        return cls(
            controller=controller,
            enable_extensibility_hooks=enable_extensibility_hooks,
            frame_cache_mode=frame_cache_mode,
            **kwargs
        )
    
    @classmethod
    def multi(
        cls,
        positions: Union[List[List[float]], np.ndarray],
        orientations: Optional[Union[List[float], np.ndarray]] = None,
        speeds: Optional[Union[List[float], np.ndarray]] = None,
        max_speeds: Optional[Union[List[float], np.ndarray]] = None,
        angular_velocities: Optional[Union[List[float], np.ndarray]] = None,
        enable_extensibility_hooks: bool = False,
        enable_vectorized_ops: bool = True,
        frame_cache_mode: Optional[str] = None,
        **kwargs: Any
    ) -> 'Navigator':
        """
        Create multi-agent Navigator with enhanced Gymnasium 0.29.x features.
        
        Args:
            positions: Initial positions as list or array with shape (n_agents, 2)
            orientations: Initial orientations for each agent (optional)
            speeds: Initial speeds for each agent (optional)
            max_speeds: Maximum speeds for each agent (optional)
            angular_velocities: Initial angular velocities for each agent (optional)
            enable_extensibility_hooks: Enable custom observation/reward hooks
            enable_vectorized_ops: Enable vectorized operations for performance
            frame_cache_mode: Frame caching strategy for all agents
            **kwargs: Additional configuration options
            
        Returns:
            Navigator: Multi-agent navigator with modern features
            
        Examples:
            Two-agent navigator:
            >>> positions = [[0.0, 0.0], [10.0, 10.0]]
            >>> navigator = Navigator.multi(positions)
            
            Performance-optimized multi-agent:
            >>> navigator = Navigator.multi(
            ...     positions=[[0, 0], [20, 0], [40, 0]],
            ...     orientations=[0, 45, 90],
            ...     max_speeds=[1.0, 1.5, 2.0],
            ...     enable_vectorized_ops=True,
            ...     frame_cache_mode="lru"
            ... )
        """
        controller = MultiAgentController(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds,
            angular_velocities=angular_velocities,
            enable_extensibility_hooks=enable_extensibility_hooks,
            enable_vectorized_ops=enable_vectorized_ops,
            frame_cache_mode=frame_cache_mode,
            **kwargs
        )
        
        return cls(
            controller=controller,
            enable_extensibility_hooks=enable_extensibility_hooks,
            frame_cache_mode=frame_cache_mode,
            **kwargs
        )
    
    @classmethod
    def from_config(cls, config: Union[DictConfig, dict]) -> 'Navigator':
        """
        Create Navigator from Hydra configuration with enhanced validation.
        
        Args:
            config: Configuration object containing navigator parameters
            
        Returns:
            Navigator: Configured navigator instance with Gymnasium 0.29.x features
            
        Examples:
            From Hydra configuration:
            >>> @hydra.main(config_path="../conf", config_name="config")
            >>> def main(cfg: DictConfig) -> None:
            ...     navigator = Navigator.from_config(cfg.navigator)
            
            Enhanced configuration with extensibility:
            >>> config = {
            ...     "position": (0, 0),
            ...     "max_speed": 2.0,
            ...     "enable_extensibility_hooks": True,
            ...     "frame_cache_mode": "lru",
            ...     "custom_observation_keys": ["wind_direction", "energy_level"]
            ... }
            >>> navigator = Navigator.from_config(config)
        """
        factory = NavigatorFactory()
        controller = factory.from_config(config)
        
        # Extract enhanced configuration options
        enable_extensibility_hooks = _get_config_value(config, 'enable_extensibility_hooks', False)
        frame_cache_mode = _get_config_value(config, 'frame_cache_mode')
        custom_observation_keys = _get_config_value(config, 'custom_observation_keys', [])
        reward_shaping = _get_config_value(config, 'reward_shaping')
        
        return cls(
            controller=controller,
            enable_extensibility_hooks=enable_extensibility_hooks,
            frame_cache_mode=frame_cache_mode,
            custom_observation_keys=custom_observation_keys,
            reward_shaping=reward_shaping
        )
    
    # Gymnasium 0.29.x integration methods
    
    def create_observation_space(
        self,
        include_additional_obs: bool = True,
        **space_kwargs: Any
    ) -> Optional[spaces.Space]:
        """
        Create Gymnasium observation space for this navigator instance.
        
        Integrates with SpacesFactory utilities to create type-safe observation
        spaces that account for both base navigation observations and any
        additional observations from extensibility hooks.
        
        Args:
            include_additional_obs: Include space for additional observations
            **space_kwargs: Additional arguments for space construction
            
        Returns:
            Optional[spaces.Space]: Gymnasium observation space or None if
                Gymnasium is not available
                
        Examples:
            Basic observation space:
            >>> obs_space = navigator.create_observation_space()
            
            With additional observations:
            >>> obs_space = navigator.create_observation_space(
            ...     include_additional_obs=True
            ... )
        """
        return self.factory.create_observation_space(
            self.controller,
            include_additional_obs=include_additional_obs and self.enable_extensibility_hooks,
            **space_kwargs
        )
    
    def create_action_space(self, **space_kwargs: Any) -> Optional[spaces.Space]:
        """
        Create Gymnasium action space for this navigator instance.
        
        Integrates with SpacesFactory utilities to create type-safe action
        spaces appropriate for the navigator's configuration.
        
        Args:
            **space_kwargs: Additional arguments for space construction
            
        Returns:
            Optional[spaces.Space]: Gymnasium action space or None if
                Gymnasium is not available
                
        Examples:
            Basic action space:
            >>> action_space = navigator.create_action_space()
        """
        return self.factory.create_action_space(self.controller, **space_kwargs)
    
    # Performance and monitoring methods
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for monitoring and optimization.
        
        Returns comprehensive performance data including step execution times,
        memory usage, cache hit rates, and throughput statistics.
        
        Returns:
            Dict[str, Any]: Performance metrics dictionary containing:
                - step_execution_time_ms: Average step execution time
                - memory_usage_mb: Current memory usage
                - cache_hit_rate: Frame cache hit rate (if available)
                - throughput_agents_fps: Agent processing throughput
                - vectorized_ops_enabled: Whether vectorized operations are active
        
        Examples:
            Monitor performance:
            >>> metrics = navigator.get_performance_metrics()
            >>> if metrics['step_execution_time_ms'] > 10:
            ...     print("Performance warning: step time exceeds target")
        """
        metrics = {
            'num_agents': self.num_agents,
            'enable_extensibility_hooks': self.enable_extensibility_hooks,
            'frame_cache_mode': self.frame_cache_mode,
            'custom_observation_keys': len(self.custom_observation_keys),
        }
        
        # Add controller-specific metrics if available
        if hasattr(self.controller, 'get_performance_metrics'):
            metrics.update(self.controller.get_performance_metrics())
        
        return metrics
    
    def _detect_multi_agent_mode(self, positions) -> bool:
        """Detect if positions indicate multi-agent or single-agent mode.
        
        Args:
            positions: Position data in various formats
            
        Returns:
            bool: True if multi-agent mode should be used, False for single-agent
        """
        try:
            positions_array = np.array(positions)
            
            # If 1D array with length 2, treat as single agent (x, y)
            if positions_array.ndim == 1 and len(positions_array) == 2:
                return False
                
            # If 2D array, check if it's a single position or multiple positions
            if positions_array.ndim == 2:
                # If shape is (1, 2), it's a single agent
                if positions_array.shape == (1, 2):
                    return False
                # If shape is (n, 2) where n > 1, it's multi-agent
                elif positions_array.shape[1] == 2:
                    return True
                    
            # If it's a list/tuple of coordinate pairs, check the structure
            if isinstance(positions, (list, tuple)):
                # If it's just two numbers, treat as single agent
                if len(positions) == 2 and all(isinstance(x, (int, float)) for x in positions):
                    return False
                # If it's a list of coordinate pairs, it's multi-agent
                elif all(isinstance(x, (list, tuple)) and len(x) == 2 for x in positions):
                    return True
                    
            # Default to multi-agent for complex structures
            return True
            
        except Exception:
            # If we can't parse it, let the controller handle the error
            return True


class NavigatorFactory(BaseNavigatorFactory):
    """
    Enhanced NavigatorFactory providing Hydra-integrated navigator creation with Gymnasium 0.29.x features.
    
    This factory extends the base NavigatorFactory with additional methods for creating
    Navigator instances (rather than raw NavigatorProtocol implementations) with enhanced
    configuration support, extensibility hooks, and modern API integration.
    
    Enhanced for Gymnasium 0.29.x migration:
    - Navigator instance creation with unified API
    - Extensibility hook configuration and validation
    - Frame caching integration with configurable modes
    - Performance optimization with monitoring capabilities
    - Type-safe space creation with SpacesFactory integration
    
    The factory pattern enables configuration-driven instantiation while maintaining
    consistent initialization patterns across the research framework and providing
    seamless integration with modern ML frameworks.
    
    Examples:
        Configuration-driven Navigator creation:
        >>> factory = NavigatorFactory()
        >>> navigator = factory.create_navigator(config_dict)
        
        Enhanced Navigator with custom features:
        >>> navigator = factory.create_enhanced_navigator(
        ...     position=(10.0, 20.0),
        ...     enable_extensibility_hooks=True,
        ...     frame_cache_mode="lru"
        ... )
        
        Space creation for environment integration:
        >>> obs_space = factory.create_observation_space(navigator)
        >>> action_space = factory.create_action_space(navigator)
    """
    
    @staticmethod
    def create_navigator(
        config: Union[DictConfig, dict],
        enable_extensibility_hooks: Optional[bool] = None,
        frame_cache_mode: Optional[str] = None,
        **kwargs: Any
    ) -> Navigator:
        """
        Create Navigator instance from configuration with enhanced features.
        
        This method creates a Navigator (wrapper) rather than a raw NavigatorProtocol
        implementation, providing additional functionality for Gymnasium 0.29.x
        migration including extensibility hooks and performance monitoring.
        
        Args:
            config: Configuration object containing navigator parameters
            enable_extensibility_hooks: Override config value for extensibility
            frame_cache_mode: Override config value for frame caching
            **kwargs: Additional configuration overrides
            
        Returns:
            Navigator: Enhanced navigator instance with modern features
            
        Examples:
            From configuration dict:
            >>> config = {"position": (10, 20), "max_speed": 2.0}
            >>> navigator = NavigatorFactory.create_navigator(config)
            
            With enhanced features:
            >>> navigator = NavigatorFactory.create_navigator(
            ...     config,
            ...     enable_extensibility_hooks=True,
            ...     frame_cache_mode="lru"
            ... )
        """
        # Create base controller using parent factory
        controller = BaseNavigatorFactory.from_config(config)
        
        # Extract or override enhanced configuration
        final_extensibility = enable_extensibility_hooks
        if final_extensibility is None:
            final_extensibility = _get_config_value(config, 'enable_extensibility_hooks', False)
        
        final_cache_mode = frame_cache_mode
        if final_cache_mode is None:
            final_cache_mode = _get_config_value(config, 'frame_cache_mode')
        
        custom_observation_keys = _get_config_value(config, 'custom_observation_keys', [])
        reward_shaping = _get_config_value(config, 'reward_shaping')
        
        return Navigator(
            controller=controller,
            enable_extensibility_hooks=final_extensibility,
            frame_cache_mode=final_cache_mode,
            custom_observation_keys=custom_observation_keys,
            reward_shaping=reward_shaping,
            **kwargs
        )
    
    @staticmethod
    def create_enhanced_navigator(
        navigator_type: str = "single",
        enable_extensibility_hooks: bool = True,
        frame_cache_mode: str = "lru",
        custom_observation_keys: Optional[List[str]] = None,
        reward_shaping: Optional[str] = None,
        **kwargs: Any
    ) -> Navigator:
        """
        Create Navigator with all Gymnasium 0.29.x enhancements enabled.
        
        This convenience method creates a Navigator with modern features enabled
        by default, suitable for new research projects and applications that
        want to leverage the full capabilities of the enhanced system.
        
        Args:
            navigator_type: Type of navigator ("single" or "multi")
            enable_extensibility_hooks: Enable custom observation/reward hooks
            frame_cache_mode: Frame caching strategy ("none", "lru", "preload")
            custom_observation_keys: Additional observation keys to include
            reward_shaping: Custom reward shaping configuration
            **kwargs: Navigator-specific configuration parameters
            
        Returns:
            Navigator: Fully enhanced navigator instance
            
        Raises:
            ValueError: If navigator_type is not "single" or "multi"
            
        Examples:
            Enhanced single-agent navigator:
            >>> navigator = NavigatorFactory.create_enhanced_navigator(
            ...     navigator_type="single",
            ...     position=(10.0, 20.0),
            ...     max_speed=2.5
            ... )
            
            Enhanced multi-agent navigator:
            >>> navigator = NavigatorFactory.create_enhanced_navigator(
            ...     navigator_type="multi",
            ...     positions=[[0, 0], [10, 10], [20, 20]],
            ...     custom_observation_keys=["wind_direction", "energy_level"]
            ... )
        """
        if navigator_type == "single":
            return Navigator.single(
                enable_extensibility_hooks=enable_extensibility_hooks,
                frame_cache_mode=frame_cache_mode,
                custom_observation_keys=custom_observation_keys,
                reward_shaping=reward_shaping,
                **kwargs
            )
        elif navigator_type == "multi":
            return Navigator.multi(
                enable_extensibility_hooks=enable_extensibility_hooks,
                frame_cache_mode=frame_cache_mode,
                custom_observation_keys=custom_observation_keys,
                reward_shaping=reward_shaping,
                **kwargs
            )



# Utility functions for configuration processing (imported from protocols)
from .protocols import (
    _is_multi_agent_config,
    _get_config_value,
    _extract_extensibility_config,
    detect_api_compatibility_mode,
    convert_step_return_format,
    convert_reset_return_format
)


# Backward compatibility aliases and legacy imports

# Maintain compatibility with original simple import pattern
def create_navigator_from_config(config: Union[DictConfig, dict]) -> Navigator:
    """
    Legacy function for backward compatibility.
    
    Args:
        config: Configuration object containing navigator parameters
        
    Returns:
        Navigator: Configured navigator instance
        
    Notes:
        This function is deprecated. Use Navigator.from_config() or
        NavigatorFactory.create_navigator() instead.
    """
    warnings.warn(
        "create_navigator_from_config is deprecated. Use Navigator.from_config() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return Navigator.from_config(config)


# Re-export all important types and classes for public API
__all__ = [
    # Core classes
    "Navigator",
    "NavigatorFactory",
    
    # Legacy compatibility
    "create_navigator_from_config",
    
    # Type aliases from protocols
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
    
    # API compatibility utilities
    "detect_api_compatibility_mode",
    "convert_step_return_format", 
    "convert_reset_return_format",
]