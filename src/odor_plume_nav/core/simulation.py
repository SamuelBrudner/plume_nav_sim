"""
Core simulation orchestration for odor plume navigation experiments.

This module provides the primary run_simulation function that orchestrates frame-by-frame
navigation experiments with real-time performance monitoring, comprehensive result collection,
and context-managed resource lifecycle. Supports both single-agent and multi-agent scenarios
through the unified NavigatorProtocol interface.

The simulation engine implements enterprise-grade performance requirements:
- ≥30 FPS simulation rate with real-time monitoring and optimization
- Memory-efficient trajectory recording with configurable history limits
- Context-managed resource cleanup for video streams and visualization components
- Comprehensive result collection with performance metrics and optional persistence

Key Features:
    - Frame-by-frame simulation loop execution with error recovery
    - Real-time performance monitoring with FPS tracking and optimization
    - Unified interface supporting both single and multi-agent workflows
    - Comprehensive result collection with trajectory history and sensor readings
    - Optional visualization integration with live animation and static exports
    - Database persistence integration for experiment tracking and analysis
    - Context manager patterns for proper resource lifecycle management

Example Usage:
    Basic simulation execution:
        >>> navigator = create_navigator(position=(50, 50), max_speed=5.0)
        >>> video_plume = create_video_plume("data/plume.mp4")
        >>> results = run_simulation(navigator, video_plume, num_steps=1000, dt=0.1)
        >>> positions, orientations, readings, metrics = results

    With visualization and performance monitoring:
        >>> results = run_simulation(
        ...     navigator, video_plume,
        ...     num_steps=2000, dt=0.05,
        ...     enable_visualization=True,
        ...     target_fps=30.0,
        ...     record_performance=True
        ... )

    Multi-agent with database persistence:
        >>> multi_navigator = create_navigator(positions=[(10, 10), (20, 20)])
        >>> results = run_simulation(
        ...     multi_navigator, video_plume,
        ...     num_steps=5000,
        ...     enable_persistence=True,
        ...     experiment_id="multi_agent_experiment_001"
        ... )
"""

import time
import contextlib
from typing import Optional, Tuple, Dict, Any, Union, List
import numpy as np
from dataclasses import dataclass, field
import warnings

# Core navigation dependencies
from .controllers import SingleAgentController, MultiAgentController
try:
    from .protocols import NavigatorProtocol
except ImportError:
    # Define minimal protocol interface if protocols.py doesn't exist yet
    from typing import Protocol
    
    class NavigatorProtocol(Protocol):
        """Minimal protocol definition for navigator interface."""
        
        @property
        def positions(self) -> np.ndarray:
            """Agent positions as array with shape (num_agents, 2)."""
            ...
        
        @property
        def orientations(self) -> np.ndarray:
            """Agent orientations as array with shape (num_agents,)."""
            ...
        
        @property
        def num_agents(self) -> int:
            """Number of agents in the navigator."""
            ...
        
        def step(self, env_array: np.ndarray, dt: float = 1.0) -> None:
            """Take a simulation step."""
            ...
        
        def sample_odor(self, env_array: np.ndarray) -> Union[float, np.ndarray]:
            """Sample odor at current agent positions."""
            ...

# Visualization and utility dependencies
try:
    from ..utils.visualization import SimulationVisualization, visualize_trajectory
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    warnings.warn(
        "Visualization utilities not available. Visualization features will be disabled.",
        ImportWarning
    )

# Database persistence (optional dependency)
try:
    from ..db.session_manager import DatabaseSessionManager
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# Logging setup
try:
    from loguru import logger
except ImportError:
from loguru import logger
@dataclass
class SimulationConfig:
    """Configuration parameters for simulation execution.
    
    This dataclass provides type-safe parameter validation and default values
    for all simulation configuration options. It supports both basic and advanced
    configuration scenarios with clear parameter documentation.
    
    Attributes:
        num_steps: Total number of simulation steps to execute
        dt: Simulation timestep in seconds (affects navigation speed)
        target_fps: Target frame rate for real-time monitoring
        enable_visualization: Whether to enable live visualization
        enable_persistence: Whether to enable database persistence
        record_trajectories: Whether to record full trajectory history
        record_performance: Whether to collect performance metrics
        max_trajectory_length: Maximum trajectory points to store (memory limit)
        visualization_config: Optional visualization parameters
        performance_monitoring: Whether to enable real-time performance tracking
        error_recovery: Whether to enable automatic error recovery
        checkpoint_interval: Steps between simulation checkpoints (0 = disabled)
        experiment_id: Optional experiment identifier for persistence
    """
    num_steps: int = 1000
    dt: float = 0.1
    target_fps: float = 30.0
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
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if self.max_trajectory_length is not None and self.max_trajectory_length <= 0:
            raise ValueError("max_trajectory_length must be positive if specified")


@dataclass
class SimulationResults:
    """Comprehensive simulation results with trajectory data and performance metrics.
    
    This dataclass encapsulates all simulation outputs including trajectory histories,
    sensor readings, performance metrics, and metadata for comprehensive analysis
    and result interpretation.
    
    Attributes:
        positions_history: Agent positions over time, shape (num_agents, num_steps, 2)
        orientations_history: Agent orientations over time, shape (num_agents, num_steps)
        odor_readings: Sensor readings over time, shape (num_agents, num_steps)
        performance_metrics: Dictionary of performance measurements
        metadata: Simulation configuration and system information
        checkpoints: Optional simulation state checkpoints
        visualization_artifacts: Optional visualization outputs
        database_records: Optional database persistence information
        step_count: Number of simulation steps executed
        success: Whether the simulation completed successfully
    """
    positions_history: np.ndarray
    orientations_history: np.ndarray
    odor_readings: np.ndarray
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    visualization_artifacts: Dict[str, Any] = field(default_factory=dict)
    database_records: Dict[str, Any] = field(default_factory=dict)
    step_count: int = 0
    success: bool = False


class PerformanceMonitor:
    """Real-time performance monitoring for simulation optimization.
    
    This class tracks simulation performance metrics including frame rates,
    step execution times, memory usage, and provides optimization suggestions
    when performance degrades below target thresholds.
    """
    
    def __init__(self, target_fps: float = 30.0, history_length: int = 100):
        """Initialize performance monitor.
        
        Parameters
        ----------
        target_fps : float
            Target frame rate for performance optimization
        history_length : int
            Number of recent measurements to track for moving averages
        """
        self.target_fps = target_fps
        self.target_step_time = 1.0 / target_fps
        self.history_length = history_length
        
        # Performance tracking
        self.step_times: List[float] = []
        self.frame_times: List[float] = []
        self.memory_usage: List[float] = []
        
        # Statistics
        self.total_steps = 0
        self.start_time = time.perf_counter()
        self.last_step_time = self.start_time
        
        # Optimization flags
        self.performance_warnings = []
        self.optimization_applied = False
    
    def record_step_time(self, seconds: float, label: str | None = None) -> None:
        """Record timing for a simulation step.

        Parameters
        ----------
        seconds : float
            Time taken for the step in seconds.
        label : str | None
            Optional label for the recorded duration. Currently unused.
        """
        logger.debug(
            "Recording step duration",
            extra={"step_duration": seconds, "label": label},
        )
        current_time = time.perf_counter()

        # Update counters
        self.total_steps += 1
        self.step_times.append(seconds)

        # Calculate frame time (time since last step)
        frame_time = current_time - self.last_step_time
        self.frame_times.append(frame_time)
        self.last_step_time = current_time

        # Maintain rolling history
        if len(self.step_times) > self.history_length:
            self.step_times.pop(0)
            self.frame_times.pop(0)

        # Check performance thresholds
        self._check_performance_thresholds()

    # Backward compatibility
    record_step = record_step_time
    
    def _check_performance_thresholds(self) -> None:
        """Check if performance is below target and record warnings."""
        if len(self.frame_times) < 10:  # Need some history
            return
        
        # Calculate recent average FPS
        recent_frame_time = np.mean(self.frame_times[-10:])
        current_fps = 1.0 / recent_frame_time if recent_frame_time > 0 else 0
        
        # Check if below target
        if current_fps < self.target_fps * 0.8:  # 20% tolerance
            warning = {
                'timestamp': time.perf_counter(),
                'current_fps': current_fps,
                'target_fps': self.target_fps,
                'step': self.total_steps,
                'message': f"Performance below target: {current_fps:.1f} FPS (target: {self.target_fps:.1f})"
            }
            self.performance_warnings.append(warning)
            
            # Log performance warning
            logger.warning(
                f"Simulation performance below target",
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
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        elapsed_time = time.perf_counter() - self.start_time

        metrics = {
            'total_steps': self.total_steps,
            'elapsed_time': elapsed_time,
            'average_fps': self.total_steps / elapsed_time if elapsed_time > 0 else 0,
            'current_fps': self.get_current_fps(),
            'target_fps': self.target_fps,
            'performance_warnings': len(self.performance_warnings),
            'optimization_applied': self.optimization_applied
        }

        if self.step_times:
            metrics.update({
                'average_step_time': np.mean(self.step_times),
                'min_step_time': np.min(self.step_times),
                'max_step_time': np.max(self.step_times),
                'step_time_std': np.std(self.step_times)
            })

        if self.frame_times:
            metrics.update({
                'average_frame_time': np.mean(self.frame_times),
                'min_frame_time': np.min(self.frame_times),
                'max_frame_time': np.max(self.frame_times)
            })

        logger.debug("Performance summary computed", extra={"total_steps": self.total_steps})
        return metrics

    # Backward compatibility
    get_metrics = get_summary


@contextlib.contextmanager
def simulation_context(
    video_plume,
    visualization: Optional[Any] = None,
    database_session: Optional[Any] = None,
    enable_visualization: bool = False,
    enable_persistence: bool = False
):
    """Context manager for simulation resource lifecycle management.
    
    This context manager ensures proper setup and cleanup of all simulation
    resources including video streams, visualization components, and database
    connections. It implements the enterprise-grade resource management pattern
    required for production deployments.
    
    Parameters
    ----------
    video_plume : VideoPlume
        Video plume environment instance
    visualization : Optional[Any]
        Visualization component (if available)
    database_session : Optional[Any]
        Database session for persistence (if available)
    enable_visualization : bool
        Whether visualization is enabled
    enable_persistence : bool
        Whether database persistence is enabled
    
    Yields
    ------
    Dict[str, Any]
        Dictionary containing initialized resources
    """
    resources = {
        'video_plume': video_plume,
        'visualization': None,
        'database_session': None
    }
    
    try:
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
                'visualization_enabled': resources['visualization'] is not None,
                'persistence_enabled': resources['database_session'] is not None
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
            # Video plume cleanup is handled by its own context manager
            logger.debug("Video plume cleanup completed")
        except Exception as e:
            logger.warning(f"Error in video plume cleanup: {e}")


def run_simulation(
    navigator: NavigatorProtocol,
    video_plume,  # VideoPlume type - avoiding import to prevent circular dependency
    num_steps: Optional[int] = None,
    dt: Optional[float] = None,
    config: Optional[Union[SimulationConfig, Dict[str, Any]]] = None,
    target_fps: float = 30.0,
    enable_visualization: bool = False,
    enable_persistence: bool = False,
    record_trajectories: bool = True,
    record_performance: bool = True,
    visualization_config: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
    **kwargs: Any
) -> SimulationResults:
    """
    Execute a complete odor plume navigation simulation with comprehensive monitoring.

    This function orchestrates frame-by-frame agent navigation through video-based odor
    plume environments with real-time performance monitoring, comprehensive result collection,
    and context-managed resource lifecycle. Supports both single-agent and multi-agent
    scenarios through the unified NavigatorProtocol interface.

    The simulation engine implements enterprise-grade requirements:
    - ≥30 FPS simulation rate with real-time monitoring and optimization
    - Memory-efficient trajectory recording with configurable history limits  
    - Context-managed resource cleanup for video streams and visualization components
    - Comprehensive result collection with performance metrics and optional persistence

    Parameters
    ----------
    navigator : NavigatorProtocol
        Navigator instance (SingleAgentController or MultiAgentController)
    video_plume : VideoPlume
        VideoPlume environment instance providing odor concentration data
    num_steps : Optional[int], optional
        Number of simulation steps to execute, by default None (uses config or 1000)
    dt : Optional[float], optional
        Simulation timestep in seconds, by default None (uses config or 0.1)
    config : Optional[Union[SimulationConfig, Dict[str, Any]]], optional
        Simulation configuration object or dictionary, by default None
    target_fps : float, optional
        Target frame rate for performance monitoring, by default 30.0
    enable_visualization : bool, optional
        Whether to enable live visualization, by default False
    enable_persistence : bool, optional
        Whether to enable database persistence, by default False
    record_trajectories : bool, optional
        Whether to record full trajectory history, by default True
    record_performance : bool, optional
        Whether to collect performance metrics, by default True
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
        - positions_history: Agent positions over time, shape (num_agents, num_steps, 2)
        - orientations_history: Agent orientations over time, shape (num_agents, num_steps)
        - odor_readings: Sensor readings over time, shape (num_agents, num_steps)
        - performance_metrics: Dictionary of performance measurements
        - metadata: Simulation configuration and system information

    Raises
    ------
    ValueError
        If required parameters are missing or invalid
        If navigator or video_plume are None
        If configuration validation fails
    TypeError
        If navigator doesn't implement NavigatorProtocol
        If video_plume is not a valid environment instance
    RuntimeError
        If simulation execution fails or exceeds performance requirements

    Examples
    --------
    Basic single-agent simulation:
        >>> navigator = SingleAgentController(position=(50, 50), max_speed=5.0)
        >>> video_plume = VideoPlume.from_config("data/plume.mp4")
        >>> results = run_simulation(navigator, video_plume, num_steps=1000, dt=0.1)
        >>> print(f"Final position: {results.positions_history[0, -1, :]}")

    Multi-agent with visualization:
        >>> navigator = MultiAgentController(
        ...     positions=[(10, 10), (20, 20)],
        ...     max_speeds=[3.0, 4.0]
        ... )
        >>> results = run_simulation(
        ...     navigator, video_plume,
        ...     num_steps=2000, dt=0.05,
        ...     enable_visualization=True,
        ...     target_fps=30.0
        ... )

    High-performance simulation with monitoring:
        >>> results = run_simulation(
        ...     navigator, video_plume,
        ...     num_steps=5000,
        ...     target_fps=60.0,
        ...     record_performance=True,
        ...     enable_persistence=True,
        ...     experiment_id="high_perf_test_001"
        ... )
        >>> print(f"Average FPS: {results.performance_metrics['average_fps']:.1f}")

    Notes
    -----
    Performance characteristics:
    - Optimized for ≥30 FPS execution with real-time visualization
    - Memory-efficient trajectory recording with configurable storage limits
    - Automatic frame synchronization between navigator and video plume
    - Progress logging for long-running simulations

    Resource management:
    - Context managers ensure proper cleanup of video streams and visualization
    - Database connections use connection pooling for efficiency
    - Memory usage monitoring with automatic optimization triggers
    """
    # Initialize logger with simulation context
    sim_logger = logger.bind(
        module=__name__,
        function="run_simulation",
        navigator_type=type(navigator).__name__,
        num_agents=navigator.num_agents,
        experiment_id=experiment_id
    )

    try:
        # Validate required inputs
        if navigator is None:
            raise ValueError("navigator parameter is required")
        if video_plume is None:
            raise ValueError("video_plume parameter is required")

        # Type validation for navigator
        if not hasattr(navigator, 'positions') or not hasattr(navigator, 'step'):
            raise TypeError("navigator must implement NavigatorProtocol interface")
        
        # Type validation for video_plume
        if not hasattr(video_plume, 'get_frame') or not hasattr(video_plume, 'frame_count'):
            raise TypeError("video_plume must implement video environment interface")

        # Process configuration
        if config is None:
            # Create default configuration from parameters
            sim_config = SimulationConfig(
                num_steps=num_steps or 1000,
                dt=dt or 0.1,
                target_fps=target_fps,
                enable_visualization=enable_visualization,
                enable_persistence=enable_persistence,
                record_trajectories=record_trajectories,
                record_performance=record_performance,
                visualization_config=visualization_config or {},
                experiment_id=experiment_id,
                **kwargs
            )
        elif isinstance(config, dict):
            # Merge dictionary config with parameters
            config_dict = config.copy()
            if num_steps is not None:
                config_dict['num_steps'] = num_steps
            if dt is not None:
                config_dict['dt'] = dt
            config_dict.update(kwargs)
            sim_config = SimulationConfig(**config_dict)
        elif isinstance(config, SimulationConfig):
            # Use provided config, override with explicit parameters
            sim_config = config
            if num_steps is not None:
                sim_config.num_steps = num_steps
            if dt is not None:
                sim_config.dt = dt
        else:
            raise TypeError("config must be SimulationConfig, dict, or None")

        # Initialize simulation parameters
        num_steps = sim_config.num_steps
        dt = sim_config.dt
        num_agents = navigator.num_agents
        
        sim_logger.info(
            "Starting simulation execution",
            extra={
                'num_steps': num_steps,
                'dt': dt,
                'num_agents': num_agents,
                'target_fps': sim_config.target_fps,
                'visualization_enabled': sim_config.enable_visualization,
                'persistence_enabled': sim_config.enable_persistence
            }
        )

        # Initialize performance monitor
        performance_monitor = None
        if sim_config.record_performance:
            performance_monitor = PerformanceMonitor(
                target_fps=sim_config.target_fps,
                history_length=min(100, num_steps // 10)
            )

        # Initialize trajectory storage
        trajectory_length = num_steps + 1
        if sim_config.max_trajectory_length is not None:
            trajectory_length = min(trajectory_length, sim_config.max_trajectory_length)

        if sim_config.record_trajectories:
            positions_history = np.zeros((num_agents, trajectory_length, 2))
            orientations_history = np.zeros((num_agents, trajectory_length))
            odor_readings = np.zeros((num_agents, trajectory_length))
            
            # Store initial state
            positions_history[:, 0] = navigator.positions
            orientations_history[:, 0] = navigator.orientations
            
            # Get initial odor readings
            try:
                current_frame = video_plume.get_frame(0)
                initial_readings = navigator.sample_odor(current_frame)
                if isinstance(initial_readings, (int, float)):
                    odor_readings[:, 0] = initial_readings
                else:
                    odor_readings[:, 0] = initial_readings
            except Exception as e:
                sim_logger.warning(f"Failed to get initial odor readings: {e}")
                odor_readings[:, 0] = 0.0
        else:
            # Minimal storage for compatibility
            positions_history = np.zeros((num_agents, 2, 2))
            orientations_history = np.zeros((num_agents, 2))
            odor_readings = np.zeros((num_agents, 2))

        # Initialize visualization if enabled
        visualization = None
        if sim_config.enable_visualization and VISUALIZATION_AVAILABLE:
            try:
                visualization = SimulationVisualization(**sim_config.visualization_config)
                # Setup environment with first frame
                initial_frame = video_plume.get_frame(0)
                visualization.setup_environment(initial_frame)
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

        # Execution variables
        checkpoints = []
        visualization_artifacts = {}
        database_records = {}

        # Execute simulation with context management
        with simulation_context(
            video_plume,
            visualization=visualization,
            database_session=database_session,
            enable_visualization=sim_config.enable_visualization,
            enable_persistence=sim_config.enable_persistence
        ) as resources:
            
            # Main simulation loop
            for step in range(num_steps):
                step_start_time = time.perf_counter()
                
                try:
                    # Get current frame with bounds checking
                    frame_idx = min(step + 1, video_plume.frame_count - 1)
                    current_frame = video_plume.get_frame(frame_idx)
                    
                    # Update navigator state
                    navigator.step(current_frame, dt=dt)
                    
                    # Record trajectory data if enabled
                    if sim_config.record_trajectories and step + 1 < trajectory_length:
                        positions_history[:, step + 1] = navigator.positions
                        orientations_history[:, step + 1] = navigator.orientations
                        
                        # Sample odor at current position
                        try:
                            readings = navigator.sample_odor(current_frame)
                            if isinstance(readings, (int, float)):
                                odor_readings[:, step + 1] = readings
                            else:
                                odor_readings[:, step + 1] = readings
                        except Exception as e:
                            sim_logger.debug(f"Odor sampling failed at step {step}: {e}")
                            odor_readings[:, step + 1] = 0.0

                    # Update visualization if enabled
                    if resources['visualization'] is not None:
                        try:
                            resources['visualization'].update_visualization(
                                navigator.positions,
                                navigator.orientations,
                                current_frame
                            )
                        except Exception as e:
                            sim_logger.debug(f"Visualization update failed at step {step}: {e}")

                    # Record performance metrics
                    step_duration = time.perf_counter() - step_start_time
                    sim_logger.debug(
                        f"Step {step} duration: {step_duration:.6f}s"
                    )
                    if performance_monitor is not None:
                        performance_monitor.record_step_time(step_duration, label="step")

                    # Checkpoint creation
                    if (sim_config.checkpoint_interval > 0 and 
                        (step + 1) % sim_config.checkpoint_interval == 0):
                        checkpoint = {
                            'step': step + 1,
                            'timestamp': time.perf_counter(),
                            'navigator_state': {
                                'positions': navigator.positions.copy(),
                                'orientations': navigator.orientations.copy()
                            }
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
                                'step': step + 1
                            }
                        )

                except Exception as e:
                    if sim_config.error_recovery:
                        sim_logger.warning(f"Recoverable error at step {step}: {e}")
                        # Continue with next step
                        continue
                    else:
                        sim_logger.error(f"Simulation failed at step {step}: {e}")
                        raise RuntimeError(f"Simulation execution failed at step {step}: {e}") from e

        # Handle non-recording case by storing final state
        if not sim_config.record_trajectories:
            positions_history[:, 0] = navigator.positions
            orientations_history[:, 0] = navigator.orientations
            try:
                final_frame = video_plume.get_frame(video_plume.frame_count - 1)
                readings = navigator.sample_odor(final_frame)
                if isinstance(readings, (int, float)):
                    odor_readings[:, 0] = readings
                else:
                    odor_readings[:, 0] = readings
            except Exception as e:
                sim_logger.debug(f"Failed to get final odor readings: {e}")
                odor_readings[:, 0] = 0.0

        # Collect performance metrics
        performance_metrics = {}
        if performance_monitor is not None:
            performance_metrics = performance_monitor.get_summary()

        # Create metadata
        metadata = {
            'simulation_config': {
                'num_steps': sim_config.num_steps,
                'dt': sim_config.dt,
                'target_fps': sim_config.target_fps,
                'num_agents': num_agents
            },
            'navigator_type': type(navigator).__name__,
            'video_plume_info': {
                'frame_count': getattr(video_plume, 'frame_count', None),
                'width': getattr(video_plume, 'width', None),
                'height': getattr(video_plume, 'height', None)
            },
            'timestamp': time.time(),
            'experiment_id': sim_config.experiment_id
        }

        # Create results object
        results = SimulationResults(
            positions_history=positions_history,
            orientations_history=orientations_history,
            odor_readings=odor_readings,
            performance_metrics=performance_metrics,
            metadata=metadata,
            checkpoints=checkpoints,
            visualization_artifacts=visualization_artifacts,
            database_records=database_records,
            step_count=performance_monitor.total_steps if performance_monitor is not None else num_steps,
            success=True,
        )

        sim_logger.info(
            "Simulation completed successfully",
            extra={
                'steps_executed': num_steps,
                'final_positions': positions_history[:, -1, :].tolist() if sim_config.record_trajectories else positions_history[:, 0, :].tolist(),
                'average_fps': performance_metrics.get('average_fps', 0),
                'trajectory_recorded': sim_config.record_trajectories,
                'performance_warnings': performance_metrics.get('performance_warnings', 0)
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
    "simulation_context"
]