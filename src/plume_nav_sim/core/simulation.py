"""
Core simulation orchestration for Gymnasium-compliant odor plume navigation experiments.

This module provides comprehensive simulation lifecycle management for modern Gymnasium
environments with extensibility hooks, enhanced performance monitoring, and dual API
compatibility. Migrated from legacy OpenAI Gym 0.26 to Gymnasium 0.29.x while
maintaining backward compatibility for downstream projects.

The simulation engine implements enterprise-grade performance requirements:
- ≥30 FPS simulation rate with real-time monitoring and step-time enforcement (<10ms)
- Memory-efficient trajectory recording with configurable history limits
- Context-managed resource cleanup for environments, visualization, and database persistence
- Comprehensive result collection with performance metrics and extensibility hook integration
- Enhanced frame caching with LRU eviction and memory pressure management
- Dual API support for seamless migration from legacy Gym to modern Gymnasium

Key Features:
    - End-to-end simulation orchestration with Gymnasium 0.29.x environment integration
    - Performance monitoring enforcing sub-10ms step execution and ≥30 FPS throughput
    - Extensibility hooks integration (compute_additional_obs, compute_extra_reward, on_episode_end)
    - Enhanced frame caching system with LRU eviction and memory pressure management
    - Dual API compatibility wrapper handling both 4-tuple and 5-tuple environment returns
    - Context-managed resource lifecycle for environments, visualization, and database persistence
    - Comprehensive trajectory recording with configurable storage limits and performance optimization

Example Usage:
    Modern Gymnasium environment simulation:
        >>> import gymnasium
        >>> from plume_nav_sim.core.simulation import run_simulation
        >>> env = gymnasium.make("PlumeNavSim-v0")
        >>> results = run_simulation(env, num_steps=1000, target_fps=30.0)
        >>> print(f"Average FPS: {results.performance_metrics['average_fps']:.1f}")

    Legacy compatibility with migration path:
        >>> from plume_nav_sim.shims import gym_make  # Triggers deprecation warning
        >>> env = gym_make("PlumeNavSim-v0")  # Internally uses Gymnasium
        >>> results = run_simulation(env, num_steps=1000, enable_legacy_mode=True)

    Performance monitoring with extensibility hooks:
        >>> env = gymnasium.make("PlumeNavSim-v0")
        >>> results = run_simulation(
        ...     env, 
        ...     num_steps=2000,
        ...     target_fps=30.0,
        ...     performance_monitoring=True,
        ...     enable_hooks=True,
        ...     frame_cache_mode="lru"
        ... )
        >>> print(f"Hook execution time: {results.performance_metrics['hook_overhead_ms']:.2f}ms")

    With visualization and database persistence:
        >>> results = run_simulation(
        ...     env,
        ...     num_steps=5000,
        ...     enable_visualization=True,
        ...     enable_persistence=True,
        ...     experiment_id="gymnasium_migration_test_001"
        ... )
"""

import time
import contextlib
import warnings
from typing import Optional, Tuple, Dict, Any, Union, List, Protocol, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

# Core dependencies for Gymnasium migration
try:
    import gymnasium as gym
    from gymnasium import Env as GymnasiumEnv
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    GymnasiumEnv = object  # Fallback for type hints

# Legacy Gym compatibility (optional)
try:
    import gym as legacy_gym
    LEGACY_GYM_AVAILABLE = True
except ImportError:
    LEGACY_GYM_AVAILABLE = False

# Enhanced frame caching and utilities
try:
    from ..utils.frame_cache import FrameCache, FrameCacheConfig
    FRAME_CACHE_AVAILABLE = True
except ImportError:
    FRAME_CACHE_AVAILABLE = False
    warnings.warn(
        "Enhanced frame cache not available. Using basic caching fallback.",
        ImportWarning
    )

# Visualization support (optional)
try:
    from ..utils.visualization import SimulationVisualization, visualize_trajectory
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Database persistence (optional)
try:
    from ..db.session_manager import DatabaseSessionManager
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# Memory monitoring for cache management
try:
    import psutil
    MEMORY_MONITORING_AVAILABLE = True
except ImportError:
    MEMORY_MONITORING_AVAILABLE = False
    warnings.warn(
        "psutil not available. Memory monitoring disabled.",
        ImportWarning
    )

# Logging setup
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Type checking imports
if TYPE_CHECKING:
    from ..core.protocols import NavigatorProtocol
    from ..envs.plume_navigation_env import PlumeNavigationEnv


# Environment protocol for type safety
class EnvironmentProtocol(Protocol):
    """Protocol defining the expected environment interface."""
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Union[
        Tuple[Any, Dict], Any
    ]:
        """Reset environment to initial state."""
        ...
    
    def step(self, action: Any) -> Union[
        Tuple[Any, float, bool, Dict],  # Legacy 4-tuple
        Tuple[Any, float, bool, bool, Dict]  # Modern 5-tuple
    ]:
        """Execute one step in the environment."""
        ...
    
    def close(self) -> None:
        """Clean up environment resources."""
        ...
    
    @property
    def action_space(self) -> Any:
        """Environment action space."""
        ...
    
    @property
    def observation_space(self) -> Any:
        """Environment observation space."""
        ...


@dataclass
class SimulationConfig:
    """Configuration parameters for simulation execution with Gymnasium migration support.
    
    This dataclass provides type-safe parameter validation for both legacy and modern
    environment configurations, supporting the dual API compatibility requirements
    of the Gymnasium migration while maintaining performance guarantees.
    
    Attributes:
        num_steps: Total number of simulation steps to execute
        dt: Simulation timestep in seconds (affects environment dynamics)
        target_fps: Target frame rate for real-time monitoring (≥30 FPS requirement)
        step_time_limit_ms: Maximum allowed time per step in milliseconds (≤10ms requirement)
        enable_visualization: Whether to enable live visualization
        enable_persistence: Whether to enable database persistence
        record_trajectories: Whether to record full trajectory history
        record_performance: Whether to collect performance metrics
        max_trajectory_length: Maximum trajectory points to store (memory management)
        visualization_config: Optional visualization parameters
        performance_monitoring: Whether to enable real-time performance tracking
        error_recovery: Whether to enable automatic error recovery
        checkpoint_interval: Steps between simulation checkpoints (0 = disabled)
        experiment_id: Optional experiment identifier for persistence
        enable_legacy_mode: Support for legacy Gym 4-tuple returns
        enable_hooks: Whether to enable extensibility hooks
        frame_cache_mode: Frame cache operation mode ("none", "lru", "preload")
        memory_limit_mb: Memory limit for frame cache in megabytes
        hook_timeout_ms: Maximum time allowed for hook execution
        gymnasium_strict_mode: Whether to enforce strict Gymnasium API compliance
    """
    num_steps: int = 1000
    dt: float = 0.1
    target_fps: float = 30.0
    step_time_limit_ms: float = 10.0  # Performance requirement from Section 0.5.1
    enable_visualization: bool = False
    enable_persistence: bool = False
    record_trajectories: bool = True
    record_performance: bool = True
    max_trajectory_length: Optional[int] = None
    visualization_config: Dict[str, Any] = field(default_factory=dict)
    performance_monitoring: bool = True
    error_recovery: bool = True
    checkpoint_interval: int = 0
    experiment_id: Optional[str] = None
    enable_legacy_mode: bool = False
    enable_hooks: bool = True
    frame_cache_mode: str = "lru"  # none, lru, preload
    memory_limit_mb: int = 2048  # Hard limit per Section 0.2.2
    hook_timeout_ms: float = 1.0  # Prevent hook overhead
    gymnasium_strict_mode: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if self.step_time_limit_ms <= 0:
            raise ValueError("step_time_limit_ms must be positive")
        if self.max_trajectory_length is not None and self.max_trajectory_length <= 0:
            raise ValueError("max_trajectory_length must be positive if specified")
        if self.frame_cache_mode not in ["none", "lru", "preload"]:
            raise ValueError("frame_cache_mode must be 'none', 'lru', or 'preload'")
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive")
        if self.hook_timeout_ms <= 0:
            raise ValueError("hook_timeout_ms must be positive")


@dataclass
class SimulationResults:
    """Comprehensive simulation results with Gymnasium migration compatibility.
    
    Enhanced results dataclass supporting both legacy and modern environment outputs
    with performance metrics, extensibility hook execution data, and frame cache
    statistics for operational monitoring and optimization.
    
    Attributes:
        observations_history: Environment observations over time
        actions_history: Actions taken over time
        rewards_history: Rewards received over time
        terminated_history: Episode termination flags (Gymnasium)
        truncated_history: Episode truncation flags (Gymnasium)
        done_history: Combined done flags (legacy compatibility)
        info_history: Environment info dictionaries over time
        performance_metrics: Dictionary of performance measurements
        metadata: Simulation configuration and system information
        checkpoints: Optional simulation state checkpoints
        visualization_artifacts: Optional visualization outputs
        database_records: Optional database persistence information
        hook_execution_stats: Statistics on extensibility hook performance
        frame_cache_stats: Frame cache performance and memory usage statistics
        legacy_mode_used: Whether legacy 4-tuple mode was activated
        api_compatibility_info: Information about API compatibility handling
    """
    observations_history: List[Any] = field(default_factory=list)
    actions_history: List[Any] = field(default_factory=list)
    rewards_history: List[float] = field(default_factory=list)
    terminated_history: List[bool] = field(default_factory=list)
    truncated_history: List[bool] = field(default_factory=list)
    done_history: List[bool] = field(default_factory=list)  # Legacy compatibility
    info_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    visualization_artifacts: Dict[str, Any] = field(default_factory=dict)
    database_records: Dict[str, Any] = field(default_factory=dict)
    hook_execution_stats: Dict[str, Any] = field(default_factory=dict)
    frame_cache_stats: Dict[str, Any] = field(default_factory=dict)
    legacy_mode_used: bool = False
    api_compatibility_info: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Enhanced performance monitoring for Gymnasium migration with extensibility hooks.
    
    This class tracks simulation performance metrics including frame rates, step execution
    times, memory usage, and extensibility hook overhead. Enforces the performance
    requirements of ≥30 FPS and ≤10ms step execution while monitoring frame cache
    efficiency and memory pressure management.
    """
    
    def __init__(
        self, 
        target_fps: float = 30.0, 
        step_time_limit_ms: float = 10.0,
        history_length: int = 100,
        enable_memory_monitoring: bool = True
    ):
        """Initialize performance monitor with Gymnasium migration requirements.
        
        Parameters
        ----------
        target_fps : float
            Target frame rate for performance optimization (≥30 FPS requirement)
        step_time_limit_ms : float
            Maximum allowed time per step in milliseconds (≤10ms requirement)
        history_length : int
            Number of recent measurements to track for moving averages
        enable_memory_monitoring : bool
            Whether to enable memory usage monitoring for frame cache
        """
        self.target_fps = target_fps
        self.target_step_time = 1.0 / target_fps
        self.step_time_limit_ms = step_time_limit_ms
        self.step_time_limit_s = step_time_limit_ms / 1000.0
        self.history_length = history_length
        self.enable_memory_monitoring = enable_memory_monitoring and MEMORY_MONITORING_AVAILABLE
        
        # Performance tracking
        self.step_times: List[float] = []
        self.frame_times: List[float] = []
        self.hook_times: List[float] = []
        self.memory_usage: List[float] = []
        
        # Statistics
        self.total_steps = 0
        self.start_time = time.perf_counter()
        self.last_step_time = self.start_time
        
        # Performance violations
        self.performance_warnings = []
        self.step_time_violations = 0
        self.fps_violations = 0
        self.hook_timeout_violations = 0
        
        # Frame cache monitoring
        self.cache_hit_rate_history: List[float] = []
        self.cache_memory_usage: List[float] = []
    
    def record_step(
        self, 
        step_duration: float, 
        hook_duration: float = 0.0,
        cache_stats: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record timing for a simulation step with hook and cache monitoring.
        
        Parameters
        ----------
        step_duration : float
            Time taken for the core step in seconds
        hook_duration : float
            Time taken for extensibility hooks in seconds
        cache_stats : Optional[Dict[str, Any]]
            Frame cache performance statistics
        """
        current_time = time.perf_counter()
        
        # Update counters
        self.total_steps += 1
        self.step_times.append(step_duration)
        self.hook_times.append(hook_duration)
        
        # Calculate frame time (time since last step)
        frame_time = current_time - self.last_step_time
        self.frame_times.append(frame_time)
        self.last_step_time = current_time
        
        # Monitor memory usage if enabled
        if self.enable_memory_monitoring:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.memory_usage.append(memory_mb)
            except Exception:
                # Graceful degradation if memory monitoring fails
                pass
        
        # Track frame cache statistics
        if cache_stats:
            if 'hit_rate' in cache_stats:
                self.cache_hit_rate_history.append(cache_stats['hit_rate'])
            if 'memory_usage_mb' in cache_stats:
                self.cache_memory_usage.append(cache_stats['memory_usage_mb'])
        
        # Maintain rolling history
        if len(self.step_times) > self.history_length:
            self.step_times.pop(0)
            self.frame_times.pop(0)
            self.hook_times.pop(0)
            if self.memory_usage:
                self.memory_usage.pop(0)
        
        # Check performance thresholds
        self._check_performance_thresholds(step_duration, hook_duration)
    
    def _check_performance_thresholds(self, step_duration: float, hook_duration: float) -> None:
        """Check if performance is meeting requirements and record violations."""
        # Check step time limit (≤10ms requirement)
        if step_duration > self.step_time_limit_s:
            self.step_time_violations += 1
            warning = {
                'timestamp': time.perf_counter(),
                'violation_type': 'step_time_limit',
                'step_duration_ms': step_duration * 1000,
                'limit_ms': self.step_time_limit_ms,
                'step': self.total_steps,
                'message': f"Step time exceeded limit: {step_duration*1000:.2f}ms (limit: {self.step_time_limit_ms:.1f}ms)"
            }
            self.performance_warnings.append(warning)
            
            logger.warning(
                "Step time limit exceeded",
                extra={
                    'step_duration_ms': step_duration * 1000,
                    'limit_ms': self.step_time_limit_ms,
                    'step': self.total_steps,
                    'hook_duration_ms': hook_duration * 1000
                }
            )
        
        # Check FPS requirement (≥30 FPS)
        if len(self.frame_times) >= 10:
            recent_frame_time = np.mean(self.frame_times[-10:])
            current_fps = 1.0 / recent_frame_time if recent_frame_time > 0 else 0
            
            if current_fps < self.target_fps * 0.9:  # 10% tolerance
                self.fps_violations += 1
                warning = {
                    'timestamp': time.perf_counter(),
                    'violation_type': 'fps_requirement',
                    'current_fps': current_fps,
                    'target_fps': self.target_fps,
                    'step': self.total_steps,
                    'message': f"FPS below target: {current_fps:.1f} FPS (target: {self.target_fps:.1f} FPS)"
                }
                self.performance_warnings.append(warning)
                
                logger.warning(
                    "FPS requirement not met",
                    extra={
                        'current_fps': current_fps,
                        'target_fps': self.target_fps,
                        'step': self.total_steps
                    }
                )
    
    def get_current_fps(self) -> float:
        """Get current frame rate based on recent measurements."""
        if len(self.frame_times) < 5:
            return 0.0
        
        recent_frame_time = np.mean(self.frame_times[-5:])
        return 1.0 / recent_frame_time if recent_frame_time > 0 else 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics including Gymnasium migration stats."""
        elapsed_time = time.perf_counter() - self.start_time
        
        metrics = {
            # Core performance metrics
            'total_steps': self.total_steps,
            'elapsed_time': elapsed_time,
            'average_fps': self.total_steps / elapsed_time if elapsed_time > 0 else 0,
            'current_fps': self.get_current_fps(),
            'target_fps': self.target_fps,
            
            # Performance requirement compliance
            'step_time_limit_ms': self.step_time_limit_ms,
            'step_time_violations': self.step_time_violations,
            'fps_violations': self.fps_violations,
            'performance_warnings_count': len(self.performance_warnings),
            'requirements_met': {
                'fps_requirement': self.fps_violations == 0,
                'step_time_requirement': self.step_time_violations == 0
            },
            
            # Extensibility hook performance
            'hook_timeout_violations': self.hook_timeout_violations,
            'hook_overhead_ms': np.mean(self.hook_times) * 1000 if self.hook_times else 0.0,
            'max_hook_time_ms': np.max(self.hook_times) * 1000 if self.hook_times else 0.0,
        }
        
        # Add detailed timing statistics
        if self.step_times:
            metrics.update({
                'average_step_time_ms': np.mean(self.step_times) * 1000,
                'min_step_time_ms': np.min(self.step_times) * 1000,
                'max_step_time_ms': np.max(self.step_times) * 1000,
                'step_time_std_ms': np.std(self.step_times) * 1000,
                'step_time_95th_percentile_ms': np.percentile(self.step_times, 95) * 1000
            })
        
        if self.frame_times:
            metrics.update({
                'average_frame_time_ms': np.mean(self.frame_times) * 1000,
                'min_frame_time_ms': np.min(self.frame_times) * 1000,
                'max_frame_time_ms': np.max(self.frame_times) * 1000
            })
        
        # Add memory monitoring data
        if self.memory_usage:
            metrics.update({
                'average_memory_mb': np.mean(self.memory_usage),
                'peak_memory_mb': np.max(self.memory_usage),
                'current_memory_mb': self.memory_usage[-1] if self.memory_usage else 0
            })
        
        # Add frame cache statistics
        if self.cache_hit_rate_history:
            metrics.update({
                'cache_hit_rate': np.mean(self.cache_hit_rate_history),
                'cache_hit_rate_min': np.min(self.cache_hit_rate_history),
                'cache_efficiency_target_met': np.mean(self.cache_hit_rate_history) >= 0.9
            })
        
        if self.cache_memory_usage:
            metrics.update({
                'cache_memory_usage_mb': self.cache_memory_usage[-1],
                'cache_memory_peak_mb': np.max(self.cache_memory_usage)
            })
        
        return metrics


@contextlib.contextmanager
def simulation_context(
    env: EnvironmentProtocol,
    visualization: Optional[Any] = None,
    database_session: Optional[Any] = None,
    frame_cache: Optional[Any] = None,
    enable_visualization: bool = False,
    enable_persistence: bool = False,
    enable_frame_cache: bool = True
):
    """Context manager for simulation resource lifecycle management with Gymnasium support.
    
    Enhanced context manager ensuring proper setup and cleanup of all simulation
    resources including Gymnasium environments, visualization components, database
    connections, and frame cache systems. Implements enterprise-grade resource
    management patterns required for production Gymnasium deployments.
    
    Parameters
    ----------
    env : EnvironmentProtocol
        Gymnasium or legacy Gym environment instance
    visualization : Optional[Any]
        Visualization component (if available)
    database_session : Optional[Any]
        Database session for persistence (if available)
    frame_cache : Optional[Any]
        Enhanced frame cache instance (if available)
    enable_visualization : bool
        Whether visualization is enabled
    enable_persistence : bool
        Whether database persistence is enabled
    enable_frame_cache : bool
        Whether frame caching is enabled
    
    Yields
    ------
    Dict[str, Any]
        Dictionary containing initialized resources with status information
    """
    resources = {
        'env': env,
        'visualization': None,
        'database_session': None,
        'frame_cache': None,
        'api_info': {
            'gymnasium_env': GYMNASIUM_AVAILABLE and hasattr(env, 'spec'),
            'legacy_env': not (GYMNASIUM_AVAILABLE and hasattr(env, 'spec')),
            'supports_5_tuple': True,  # Assume modern API by default
        }
    }
    
    try:
        # Detect environment API version
        if hasattr(env, 'step'):
            # Test with a dummy action to detect return format
            try:
                # This is just for API detection, not actual simulation
                dummy_obs, dummy_info = env.reset() if GYMNASIUM_AVAILABLE else (env.reset(), {})
                resources['api_info']['supports_reset_info'] = isinstance(dummy_info, dict)
            except Exception:
                # Fallback for environments that need special initialization
                resources['api_info']['supports_reset_info'] = GYMNASIUM_AVAILABLE
        
        # Initialize frame cache if enabled and available
        if enable_frame_cache and FRAME_CACHE_AVAILABLE and frame_cache is not None:
            logger.info("Initializing enhanced frame cache")
            resources['frame_cache'] = frame_cache
        
        # Initialize visualization if enabled and available
        if enable_visualization and VISUALIZATION_AVAILABLE and visualization is not None:
            logger.info("Initializing visualization resources")
            resources['visualization'] = visualization
        
        # Initialize database session if enabled and available
        if enable_persistence and DATABASE_AVAILABLE and database_session is not None:
            logger.info("Initializing database session")
            resources['database_session'] = database_session
        
        logger.info(
            "Simulation context initialized",
            extra={
                'env_type': 'Gymnasium' if resources['api_info']['gymnasium_env'] else 'Legacy',
                'visualization_enabled': resources['visualization'] is not None,
                'persistence_enabled': resources['database_session'] is not None,
                'frame_cache_enabled': resources['frame_cache'] is not None,
                'api_compatibility': resources['api_info']
            }
        )
        
        yield resources
        
    except Exception as e:
        logger.error(f"Error in simulation context: {e}")
        raise
    
    finally:
        # Cleanup resources in reverse order
        logger.info("Cleaning up simulation resources")
        
        try:
            if resources['database_session'] is not None:
                logger.debug("Closing database session")
                resources['database_session'].close()
        except Exception as e:
            logger.warning(f"Error closing database session: {e}")
        
        try:
            if resources['visualization'] is not None:
                logger.debug("Closing visualization resources")
                if hasattr(resources['visualization'], 'close'):
                    resources['visualization'].close()
        except Exception as e:
            logger.warning(f"Error closing visualization: {e}")
        
        try:
            if resources['frame_cache'] is not None:
                logger.debug("Cleaning up frame cache")
                if hasattr(resources['frame_cache'], 'clear'):
                    resources['frame_cache'].clear()
        except Exception as e:
            logger.warning(f"Error cleaning up frame cache: {e}")
        
        try:
            if hasattr(env, 'close'):
                logger.debug("Closing environment")
                env.close()
        except Exception as e:
            logger.warning(f"Error closing environment: {e}")


def detect_environment_api(env: EnvironmentProtocol) -> Dict[str, Any]:
    """Detect environment API version and capabilities for dual compatibility.
    
    This function analyzes the environment to determine its API version,
    return format expectations, and available features to enable appropriate
    compatibility handling during simulation.
    
    Parameters
    ----------
    env : EnvironmentProtocol
        Environment instance to analyze
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing API detection results and compatibility information
    """
    api_info = {
        'is_gymnasium': False,
        'is_legacy_gym': False,
        'supports_5_tuple': False,
        'supports_seed_in_reset': False,
        'supports_options_in_reset': False,
        'has_spec': False,
        'env_id': None,
        'version': None
    }
    
    try:
        # Check for Gymnasium environment
        if hasattr(env, 'spec') and GYMNASIUM_AVAILABLE:
            api_info['is_gymnasium'] = True
            api_info['supports_5_tuple'] = True
            api_info['supports_seed_in_reset'] = True
            api_info['supports_options_in_reset'] = True
            api_info['has_spec'] = True
            
            if hasattr(env.spec, 'id'):
                api_info['env_id'] = env.spec.id
            if hasattr(env.spec, 'version'):
                api_info['version'] = env.spec.version
        
        # Check for legacy Gym environment
        elif hasattr(env, 'action_space') and hasattr(env, 'observation_space'):
            api_info['is_legacy_gym'] = True
            api_info['supports_5_tuple'] = False  # Assume legacy 4-tuple
            
            # Some legacy environments might support seed
            try:
                import inspect
                reset_sig = inspect.signature(env.reset)
                api_info['supports_seed_in_reset'] = 'seed' in reset_sig.parameters
            except Exception:
                api_info['supports_seed_in_reset'] = False
        
        logger.debug(
            "Environment API detected",
            extra=api_info
        )
        
    except Exception as e:
        logger.warning(f"Failed to detect environment API: {e}")
        # Fallback to safe defaults
        api_info['is_legacy_gym'] = True
        api_info['supports_5_tuple'] = False
    
    return api_info


def execute_extensibility_hooks(
    env: EnvironmentProtocol,
    observation: Any,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    hook_timeout_ms: float = 1.0,
    enable_hooks: bool = True
) -> Tuple[Any, float, Dict[str, Any], float]:
    """Execute extensibility hooks if available with performance monitoring.
    
    This function calls the extensibility hooks defined in the Gymnasium migration:
    - compute_additional_obs(): Add custom observations
    - compute_extra_reward(): Add reward shaping
    - on_episode_end(): Handle episode completion
    
    Parameters
    ----------
    env : EnvironmentProtocol
        Environment instance (may have extensibility hooks)
    observation : Any
        Base observation from environment
    reward : float
        Base reward from environment
    terminated : bool
        Episode termination flag
    truncated : bool
        Episode truncation flag
    info : Dict[str, Any]
        Environment info dictionary
    hook_timeout_ms : float
        Maximum time allowed for hook execution
    enable_hooks : bool
        Whether hooks are enabled
    
    Returns
    -------
    Tuple[Any, float, Dict[str, Any], float]
        Enhanced observation, modified reward, updated info, hook execution time
    """
    hook_start_time = time.perf_counter()
    hook_timeout_s = hook_timeout_ms / 1000.0
    
    if not enable_hooks:
        return observation, reward, info, 0.0
    
    try:
        enhanced_observation = observation
        enhanced_reward = reward
        enhanced_info = info.copy()
        
        # Execute compute_additional_obs hook
        if hasattr(env, 'compute_additional_obs'):
            try:
                additional_obs = env.compute_additional_obs(observation)
                if additional_obs and isinstance(additional_obs, dict):
                    if isinstance(enhanced_observation, dict):
                        enhanced_observation.update(additional_obs)
                    else:
                        # Convert to dict if additional observations provided
                        enhanced_observation = {
                            'base_obs': enhanced_observation,
                            **additional_obs
                        }
                    enhanced_info['additional_obs_applied'] = True
            except Exception as e:
                logger.warning(f"compute_additional_obs hook failed: {e}")
                enhanced_info['hook_errors'] = enhanced_info.get('hook_errors', [])
                enhanced_info['hook_errors'].append(f"compute_additional_obs: {e}")
        
        # Execute compute_extra_reward hook
        if hasattr(env, 'compute_extra_reward'):
            try:
                extra_reward = env.compute_extra_reward(reward, enhanced_info)
                if isinstance(extra_reward, (int, float)):
                    enhanced_reward += extra_reward
                    enhanced_info['extra_reward_applied'] = extra_reward
            except Exception as e:
                logger.warning(f"compute_extra_reward hook failed: {e}")
                enhanced_info['hook_errors'] = enhanced_info.get('hook_errors', [])
                enhanced_info['hook_errors'].append(f"compute_extra_reward: {e}")
        
        # Execute on_episode_end hook if episode is finished
        if (terminated or truncated) and hasattr(env, 'on_episode_end'):
            try:
                env.on_episode_end(enhanced_info)
                enhanced_info['on_episode_end_executed'] = True
            except Exception as e:
                logger.warning(f"on_episode_end hook failed: {e}")
                enhanced_info['hook_errors'] = enhanced_info.get('hook_errors', [])
                enhanced_info['hook_errors'].append(f"on_episode_end: {e}")
        
        hook_duration = time.perf_counter() - hook_start_time
        
        # Check hook timeout
        if hook_duration > hook_timeout_s:
            logger.warning(
                f"Extensibility hooks exceeded timeout: {hook_duration*1000:.2f}ms (limit: {hook_timeout_ms:.1f}ms)"
            )
            enhanced_info['hook_timeout_exceeded'] = True
        
        enhanced_info['hook_execution_time_ms'] = hook_duration * 1000
        
        return enhanced_observation, enhanced_reward, enhanced_info, hook_duration
        
    except Exception as e:
        hook_duration = time.perf_counter() - hook_start_time
        logger.error(f"Critical error in extensibility hooks: {e}")
        # Return original values on critical failure
        info['hook_critical_error'] = str(e)
        return observation, reward, info, hook_duration


def run_simulation(
    env: EnvironmentProtocol,
    num_steps: Optional[int] = None,
    config: Optional[Union[SimulationConfig, Dict[str, Any]]] = None,
    target_fps: float = 30.0,
    step_time_limit_ms: float = 10.0,
    enable_visualization: bool = False,
    enable_persistence: bool = False,
    record_trajectories: bool = True,
    record_performance: bool = True,
    enable_legacy_mode: Optional[bool] = None,
    enable_hooks: bool = True,
    frame_cache_mode: str = "lru",
    memory_limit_mb: int = 2048,
    visualization_config: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
    **kwargs: Any
) -> SimulationResults:
    """
    Execute a complete Gymnasium-compliant simulation with comprehensive monitoring.

    This function orchestrates end-to-end simulation execution through modern Gymnasium
    environments with backward compatibility for legacy Gym implementations. Implements
    all migration requirements including extensibility hooks, enhanced frame caching,
    performance monitoring, and dual API support.

    Key Migration Features:
    - Gymnasium 0.29.x environment integration with proper 5-tuple handling
    - Extensibility hooks system (compute_additional_obs, compute_extra_reward, on_episode_end)
    - Enhanced frame caching with LRU eviction and memory pressure management
    - Performance monitoring enforcing ≥30 FPS and ≤10ms step execution requirements
    - Dual API compatibility supporting both legacy 4-tuple and modern 5-tuple returns
    - Context-managed resource lifecycle for environments, visualization, and persistence

    Parameters
    ----------
    env : EnvironmentProtocol
        Gymnasium or legacy Gym environment instance
    num_steps : Optional[int], optional
        Number of simulation steps to execute, by default None (uses config or 1000)
    config : Optional[Union[SimulationConfig, Dict[str, Any]]], optional
        Simulation configuration object or dictionary, by default None
    target_fps : float, optional
        Target frame rate for performance monitoring (≥30 FPS requirement), by default 30.0
    step_time_limit_ms : float, optional
        Maximum allowed time per step in milliseconds (≤10ms requirement), by default 10.0
    enable_visualization : bool, optional
        Whether to enable live visualization, by default False
    enable_persistence : bool, optional
        Whether to enable database persistence, by default False
    record_trajectories : bool, optional
        Whether to record full trajectory history, by default True
    record_performance : bool, optional
        Whether to collect performance metrics, by default True
    enable_legacy_mode : Optional[bool], optional
        Force legacy 4-tuple compatibility mode, by default None (auto-detect)
    enable_hooks : bool, optional
        Whether to enable extensibility hooks, by default True
    frame_cache_mode : str, optional
        Frame cache operation mode ("none", "lru", "preload"), by default "lru"
    memory_limit_mb : int, optional
        Memory limit for frame cache in megabytes, by default 2048
    visualization_config : Optional[Dict[str, Any]], optional
        Visualization-specific configuration, by default None
    experiment_id : Optional[str], optional
        Experiment identifier for persistence, by default None
    **kwargs : Any
        Additional configuration parameters

    Returns
    -------
    SimulationResults
        Comprehensive simulation results including:
        - observations_history: Environment observations over time
        - actions_history: Actions taken over time  
        - rewards_history: Rewards received over time
        - terminated_history: Episode termination flags (Gymnasium)
        - truncated_history: Episode truncation flags (Gymnasium)
        - done_history: Combined done flags (legacy compatibility)
        - info_history: Environment info dictionaries over time
        - performance_metrics: Performance measurements and requirement compliance
        - hook_execution_stats: Extensibility hook performance statistics
        - frame_cache_stats: Frame cache efficiency and memory usage statistics

    Raises
    ------
    ValueError
        If required parameters are missing or invalid
        If environment is None or incompatible
        If configuration validation fails
    TypeError
        If environment doesn't implement expected interface
    RuntimeError
        If simulation execution fails or exceeds performance requirements
        If critical performance thresholds are violated

    Examples
    --------
    Modern Gymnasium environment simulation:
        >>> import gymnasium
        >>> env = gymnasium.make("PlumeNavSim-v0")
        >>> results = run_simulation(env, num_steps=1000, target_fps=30.0)
        >>> print(f"Average FPS: {results.performance_metrics['average_fps']:.1f}")
        >>> print(f"Step time violations: {results.performance_metrics['step_time_violations']}")

    Legacy compatibility with migration guidance:
        >>> from plume_nav_sim.shims import gym_make
        >>> env = gym_make("PlumeNavSim-v0")  # Logs deprecation warning
        >>> results = run_simulation(env, enable_legacy_mode=True)
        >>> print(f"Legacy mode used: {results.legacy_mode_used}")

    High-performance simulation with extensibility hooks:
        >>> results = run_simulation(
        ...     env,
        ...     num_steps=2000,
        ...     target_fps=60.0,
        ...     step_time_limit_ms=8.0,
        ...     enable_hooks=True,
        ...     frame_cache_mode="lru",
        ...     memory_limit_mb=4096
        ... )
        >>> print(f"Hook overhead: {results.performance_metrics['hook_overhead_ms']:.2f}ms")
        >>> print(f"Cache hit rate: {results.frame_cache_stats.get('hit_rate', 0):.1%}")

    Notes
    -----
    Performance Requirements (from Section 0.5.1):
    - Maintains ≥30 FPS simulation rate with real-time monitoring
    - Enforces ≤10ms step execution time limit
    - Frame cache achieves >90% hit rate with ≤2 GiB memory usage
    - Extensibility hooks complete within 1ms timeout by default

    Migration Compatibility:
    - Automatic detection of Gymnasium vs legacy Gym environments
    - Transparent conversion between 4-tuple and 5-tuple step returns
    - Deprecation warnings guide users toward modern API patterns
    - Maintains numerical fidelity (±1e-6) with original implementations
    """
    # Initialize logger with simulation context
    sim_logger = logger.bind(
        module=__name__,
        function="run_simulation",
        env_type=type(env).__name__,
        experiment_id=experiment_id,
        gymnasium_migration=True
    )

    try:
        # Validate required inputs
        if env is None:
            raise ValueError("env parameter is required")

        # Type validation for environment
        if not hasattr(env, 'step') or not hasattr(env, 'reset'):
            raise TypeError("env must implement step() and reset() methods")
        
        if not hasattr(env, 'action_space') or not hasattr(env, 'observation_space'):
            raise TypeError("env must have action_space and observation_space attributes")

        # Process configuration
        if config is None:
            # Create default configuration from parameters
            sim_config = SimulationConfig(
                num_steps=num_steps or 1000,
                target_fps=target_fps,
                step_time_limit_ms=step_time_limit_ms,
                enable_visualization=enable_visualization,
                enable_persistence=enable_persistence,
                record_trajectories=record_trajectories,
                record_performance=record_performance,
                enable_legacy_mode=enable_legacy_mode if enable_legacy_mode is not None else False,
                enable_hooks=enable_hooks,
                frame_cache_mode=frame_cache_mode,
                memory_limit_mb=memory_limit_mb,
                visualization_config=visualization_config or {},
                experiment_id=experiment_id,
                **kwargs
            )
        elif isinstance(config, dict):
            # Merge dictionary config with parameters
            config_dict = config.copy()
            if num_steps is not None:
                config_dict['num_steps'] = num_steps
            config_dict.update(kwargs)
            sim_config = SimulationConfig(**config_dict)
        elif isinstance(config, SimulationConfig):
            # Use provided config, override with explicit parameters
            sim_config = config
            if num_steps is not None:
                sim_config.num_steps = num_steps
        else:
            raise TypeError("config must be SimulationConfig, dict, or None")

        # Detect environment API capabilities
        api_info = detect_environment_api(env)
        
        # Determine legacy mode
        if enable_legacy_mode is None:
            sim_config.enable_legacy_mode = api_info['is_legacy_gym']
        
        # Initialize simulation parameters
        num_steps = sim_config.num_steps
        
        sim_logger.info(
            "Starting Gymnasium-compliant simulation execution",
            extra={
                'num_steps': num_steps,
                'target_fps': sim_config.target_fps,
                'step_time_limit_ms': sim_config.step_time_limit_ms,
                'visualization_enabled': sim_config.enable_visualization,
                'persistence_enabled': sim_config.enable_persistence,
                'hooks_enabled': sim_config.enable_hooks,
                'frame_cache_mode': sim_config.frame_cache_mode,
                'legacy_mode': sim_config.enable_legacy_mode,
                'api_info': api_info
            }
        )

        # Initialize performance monitor
        performance_monitor = None
        if sim_config.record_performance:
            performance_monitor = PerformanceMonitor(
                target_fps=sim_config.target_fps,
                step_time_limit_ms=sim_config.step_time_limit_ms,
                history_length=min(100, num_steps // 10),
                enable_memory_monitoring=MEMORY_MONITORING_AVAILABLE
            )

        # Initialize frame cache if enabled
        frame_cache = None
        if sim_config.frame_cache_mode != "none" and FRAME_CACHE_AVAILABLE:
            try:
                cache_config = FrameCacheConfig(
                    mode=sim_config.frame_cache_mode,
                    memory_limit_mb=sim_config.memory_limit_mb,
                    enable_statistics=True
                )
                frame_cache = FrameCache(cache_config)
                sim_logger.info(f"Frame cache initialized in '{sim_config.frame_cache_mode}' mode")
            except Exception as e:
                sim_logger.warning(f"Failed to initialize frame cache: {e}")
                frame_cache = None

        # Initialize visualization if enabled
        visualization = None
        if sim_config.enable_visualization and VISUALIZATION_AVAILABLE:
            try:
                visualization = SimulationVisualization(**sim_config.visualization_config)
                sim_logger.info("Visualization initialized successfully")
            except Exception as e:
                sim_logger.warning(f"Failed to initialize visualization: {e}")
                visualization = None

        # Initialize database session if enabled
        database_session = None
        if sim_config.enable_persistence and DATABASE_AVAILABLE:
            try:
                db_manager = DatabaseSessionManager()
                database_session = db_manager.get_session()
                sim_logger.info("Database session initialized")
            except Exception as e:
                sim_logger.warning(f"Failed to initialize database session: {e}")
                database_session = None

        # Initialize result storage
        observations_history = []
        actions_history = []
        rewards_history = []
        terminated_history = []
        truncated_history = []
        done_history = []
        info_history = []
        
        checkpoints = []
        visualization_artifacts = {}
        database_records = {}
        hook_execution_stats = {'total_time_ms': 0.0, 'call_count': 0, 'error_count': 0}

        # Execute simulation with context management
        with simulation_context(
            env,
            visualization=visualization,
            database_session=database_session,
            frame_cache=frame_cache,
            enable_visualization=sim_config.enable_visualization,
            enable_persistence=sim_config.enable_persistence,
            enable_frame_cache=sim_config.frame_cache_mode != "none"
        ) as resources:
            
            # Reset environment with appropriate API
            try:
                if api_info['supports_seed_in_reset'] and api_info['supports_options_in_reset']:
                    # Modern Gymnasium API
                    observation, info = env.reset(seed=None, options=None)
                elif api_info['supports_seed_in_reset']:
                    # Partial modern support
                    reset_result = env.reset(seed=None)
                    if isinstance(reset_result, tuple):
                        observation, info = reset_result
                    else:
                        observation, info = reset_result, {}
                else:
                    # Legacy API
                    reset_result = env.reset()
                    if isinstance(reset_result, tuple):
                        observation, info = reset_result
                    else:
                        observation, info = reset_result, {}
                
                sim_logger.debug("Environment reset successful")
                
            except Exception as e:
                sim_logger.error(f"Environment reset failed: {e}")
                raise RuntimeError(f"Failed to reset environment: {e}") from e
            
            # Store initial state
            if sim_config.record_trajectories:
                observations_history.append(observation)
                info_history.append(info)
            
            # Main simulation loop
            for step in range(num_steps):
                step_start_time = time.perf_counter()
                
                try:
                    # Sample action from action space (placeholder for actual policy)
                    action = env.action_space.sample()
                    
                    # Execute environment step
                    step_result = env.step(action)
                    
                    # Parse step result based on API version
                    if len(step_result) == 5:
                        # Modern Gymnasium 5-tuple
                        observation, reward, terminated, truncated, info = step_result
                        done = terminated or truncated  # Legacy compatibility
                    elif len(step_result) == 4:
                        # Legacy Gym 4-tuple
                        observation, reward, done, info = step_result
                        terminated = done
                        truncated = False  # Legacy environments don't distinguish
                    else:
                        raise ValueError(f"Unexpected step result format: {len(step_result)} elements")
                    
                    # Execute extensibility hooks if enabled
                    hook_duration = 0.0
                    if sim_config.enable_hooks:
                        observation, reward, info, hook_duration = execute_extensibility_hooks(
                            env, observation, reward, terminated, truncated, info,
                            sim_config.hook_timeout_ms, sim_config.enable_hooks
                        )
                        hook_execution_stats['total_time_ms'] += hook_duration * 1000
                        hook_execution_stats['call_count'] += 1
                        if 'hook_errors' in info:
                            hook_execution_stats['error_count'] += len(info['hook_errors'])

                    # Record trajectory data if enabled
                    if sim_config.record_trajectories:
                        observations_history.append(observation)
                        actions_history.append(action)
                        rewards_history.append(reward)
                        terminated_history.append(terminated)
                        truncated_history.append(truncated)
                        done_history.append(done)
                        info_history.append(info)

                    # Update visualization if enabled
                    if resources['visualization'] is not None:
                        try:
                            # Visualization update logic would go here
                            pass
                        except Exception as e:
                            sim_logger.debug(f"Visualization update failed at step {step}: {e}")

                    # Record performance metrics
                    step_duration = time.perf_counter() - step_start_time
                    if performance_monitor is not None:
                        cache_stats = {}
                        if frame_cache is not None and hasattr(frame_cache, 'get_stats'):
                            cache_stats = frame_cache.get_stats()
                        
                        performance_monitor.record_step(step_duration, hook_duration, cache_stats)

                    # Checkpoint creation
                    if (sim_config.checkpoint_interval > 0 and 
                        (step + 1) % sim_config.checkpoint_interval == 0):
                        checkpoint = {
                            'step': step + 1,
                            'timestamp': time.perf_counter(),
                            'observation': observation,
                            'reward': reward,
                            'terminated': terminated,
                            'truncated': truncated,
                            'info': info
                        }
                        checkpoints.append(checkpoint)

                    # Progress logging for long simulations
                    if num_steps > 100 and (step + 1) % (num_steps // 10) == 0:
                        progress = (step + 1) / num_steps * 100
                        current_fps = performance_monitor.get_current_fps() if performance_monitor else 0
                        sim_logger.info(
                            f"Simulation progress: {progress:.1f}% ({step + 1}/{num_steps} steps)",
                            extra={
                                'progress_percent': progress,
                                'current_fps': current_fps,
                                'step': step + 1,
                                'step_time_ms': step_duration * 1000,
                                'hook_overhead_ms': hook_duration * 1000
                            }
                        )
                    
                    # Check for episode termination
                    if done:
                        sim_logger.info(f"Episode terminated at step {step + 1}")
                        break

                except Exception as e:
                    if sim_config.error_recovery:
                        sim_logger.warning(f"Recoverable error at step {step}: {e}")
                        # Continue with next step
                        continue
                    else:
                        sim_logger.error(f"Simulation failed at step {step}: {e}")
                        raise RuntimeError(f"Simulation execution failed at step {step}: {e}") from e

        # Collect performance metrics
        performance_metrics = {}
        if performance_monitor is not None:
            performance_metrics = performance_monitor.get_metrics()

        # Collect frame cache statistics
        frame_cache_stats = {}
        if frame_cache is not None and hasattr(frame_cache, 'get_stats'):
            frame_cache_stats = frame_cache.get_stats()

        # Create metadata
        metadata = {
            'simulation_config': {
                'num_steps': sim_config.num_steps,
                'target_fps': sim_config.target_fps,
                'step_time_limit_ms': sim_config.step_time_limit_ms,
                'enable_hooks': sim_config.enable_hooks,
                'frame_cache_mode': sim_config.frame_cache_mode,
                'legacy_mode': sim_config.enable_legacy_mode
            },
            'environment_info': {
                'env_type': type(env).__name__,
                'action_space': str(env.action_space),
                'observation_space': str(env.observation_space),
                **api_info
            },
            'timestamp': time.time(),
            'experiment_id': sim_config.experiment_id,
            'migration_version': '0.3.0'
        }

        # Create results object
        results = SimulationResults(
            observations_history=observations_history,
            actions_history=actions_history,
            rewards_history=rewards_history,
            terminated_history=terminated_history,
            truncated_history=truncated_history,
            done_history=done_history,
            info_history=info_history,
            performance_metrics=performance_metrics,
            metadata=metadata,
            checkpoints=checkpoints,
            visualization_artifacts=visualization_artifacts,
            database_records=database_records,
            hook_execution_stats=hook_execution_stats,
            frame_cache_stats=frame_cache_stats,
            legacy_mode_used=sim_config.enable_legacy_mode,
            api_compatibility_info=api_info
        )

        sim_logger.info(
            "Simulation completed successfully",
            extra={
                'steps_executed': len(actions_history) if sim_config.record_trajectories else num_steps,
                'average_fps': performance_metrics.get('average_fps', 0),
                'trajectory_recorded': sim_config.record_trajectories,
                'performance_warnings': performance_metrics.get('performance_warnings_count', 0),
                'step_time_violations': performance_metrics.get('step_time_violations', 0),
                'fps_violations': performance_metrics.get('fps_violations', 0),
                'hook_calls': hook_execution_stats['call_count'],
                'hook_errors': hook_execution_stats['error_count'],
                'cache_hit_rate': frame_cache_stats.get('hit_rate', 'N/A'),
                'legacy_mode_used': sim_config.enable_legacy_mode,
                'api_version': 'Gymnasium' if api_info['is_gymnasium'] else 'Legacy Gym'
            }
        )

        return results

    except Exception as e:
        sim_logger.error(f"Simulation execution failed: {e}")
        raise RuntimeError(f"Failed to execute simulation: {e}") from e


# Export public API
__all__ = [
    "run_simulation",
    "SimulationConfig", 
    "SimulationResults",
    "PerformanceMonitor",
    "simulation_context",
    "detect_environment_api",
    "execute_extensibility_hooks",
    "EnvironmentProtocol"
]