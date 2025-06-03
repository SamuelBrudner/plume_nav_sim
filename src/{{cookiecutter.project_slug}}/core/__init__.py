"""
Core navigation module for odor plume navigation library.

This module provides the unified entry point for all core navigation functionality,
aggregating navigation controllers, protocols, and factory methods under a single
namespace. It supports both single-agent and multi-agent navigation scenarios
while maintaining backward compatibility and enabling Kedro-based project integration.

Key Features:
- Unified Navigator factory for controller instantiation
- Protocol-based interfaces for extensibility  
- Hydra configuration system integration
- Backward-compatible API for existing users
- Support for Kedro-based ML framework integration

Examples
--------
Basic single-agent navigation:

>>> from {{cookiecutter.project_slug}}.core import Navigator, run_simulation
>>> navigator = Navigator.single(max_speed=10.0, position=(50, 50))
>>> # ... simulation code

Multi-agent navigation with Hydra configuration:

>>> from hydra import compose, initialize
>>> from {{cookiecutter.project_slug}}.core import Navigator
>>> 
>>> with initialize(config_path="../conf", version_base=None):
...     cfg = compose(config_name="config")
...     navigator = Navigator.from_config(cfg.navigator)

Direct controller access for advanced users:

>>> from {{cookiecutter.project_slug}}.core import SingleAgentController, MultiAgentController
>>> controller = SingleAgentController(position=(0, 0), max_speed=5.0)

Protocol-based development:

>>> from {{cookiecutter.project_slug}}.core import NavigatorProtocol
>>> def process_navigator(nav: NavigatorProtocol) -> None:
...     nav.reset()
...     # ... navigation logic
"""

from typing import Union, Optional, Tuple, List, Dict, Any, TYPE_CHECKING
import numpy as np
from loguru import logger

# Import core navigation protocol and implementations
from .navigator import NavigatorProtocol
from .controllers import SingleAgentController, MultiAgentController

# Import configuration support for Hydra integration
try:
    from omegaconf import DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    if TYPE_CHECKING:
        from omegaconf import DictConfig

# Initialize module logger
_logger = logger.bind(module=__name__, component="core_init")


class Navigator:
    """
    Factory class providing unified access to navigation controllers.
    
    This class serves as the primary entry point for creating navigation instances,
    supporting both direct instantiation and Hydra-based configuration management.
    It provides static factory methods for different navigation scenarios while
    maintaining compatibility with existing NavigatorProtocol interfaces.
    
    The Navigator factory supports:
    - Single-agent navigation with optimized performance
    - Multi-agent navigation with vectorized operations
    - Hydra configuration integration for ML frameworks
    - Backward compatibility with existing API contracts
    
    Examples
    --------
    Create a single agent navigator:
    
    >>> navigator = Navigator.single(position=(10, 20), max_speed=5.0)
    >>> navigator.reset()
    >>> navigator.step(env_array)
    
    Create a multi-agent navigator:
    
    >>> positions = np.array([[0, 0], [10, 10], [20, 20]])
    >>> navigator = Navigator.multi(positions=positions, max_speeds=[5.0, 6.0, 7.0])
    
    Create from Hydra configuration:
    
    >>> from hydra import compose, initialize
    >>> with initialize(config_path="../conf", version_base=None):
    ...     cfg = compose(config_name="config")
    ...     navigator = Navigator.from_config(cfg.navigator)
    
    Notes
    -----
    All factory methods return objects that implement NavigatorProtocol,
    ensuring consistent interfaces across different navigation scenarios.
    The factory automatically selects the appropriate controller implementation
    based on the provided parameters.
    """
    
    @staticmethod
    def single(
        position: Optional[Tuple[float, float]] = None,
        orientation: float = 0.0,
        speed: float = 0.0,
        max_speed: float = 1.0,
        angular_velocity: float = 0.0,
        config: Optional[Union['DictConfig', Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> SingleAgentController:
        """
        Create a single-agent navigation controller.
        
        This factory method creates an optimized SingleAgentController instance
        for scenarios involving a single navigating agent. The controller provides
        minimal overhead and direct access to agent state and operations.
        
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
            Initial angular velocity in degrees/timestep, by default 0.0
        config : Optional[Union[DictConfig, Dict[str, Any]]], optional
            Hydra configuration object or dictionary, by default None
        **kwargs : Any
            Additional configuration parameters
            
        Returns
        -------
        SingleAgentController
            Configured single-agent navigation controller
            
        Raises
        ------
        ValueError
            If parameters are invalid or inconsistent
        RuntimeError
            If controller initialization fails
            
        Examples
        --------
        Create with basic parameters:
        
        >>> navigator = Navigator.single(
        ...     position=(50.0, 50.0),
        ...     max_speed=10.0,
        ...     orientation=45.0
        ... )
        
        Create with Hydra configuration:
        
        >>> config = DictConfig({
        ...     'position': [25.0, 25.0],
        ...     'max_speed': 8.0,
        ...     'orientation': 90.0
        ... })
        >>> navigator = Navigator.single(config=config)
        """
        try:
            _logger.debug(
                "Creating single-agent navigator",
                extra={
                    "position": position,
                    "orientation": orientation,
                    "speed": speed,
                    "max_speed": max_speed,
                    "has_config": config is not None
                }
            )
            
            controller = SingleAgentController(
                position=position,
                orientation=orientation,
                speed=speed,
                max_speed=max_speed,
                angular_velocity=angular_velocity,
                config=config,
                **kwargs
            )
            
            _logger.info(
                "Single-agent navigator created successfully",
                extra={
                    "controller_type": "SingleAgentController",
                    "final_position": controller.positions[0].tolist(),
                    "final_max_speed": float(controller.max_speeds[0])
                }
            )
            
            return controller
            
        except Exception as e:
            _logger.error(
                f"Failed to create single-agent navigator: {e}",
                extra={"error_type": type(e).__name__}
            )
            raise RuntimeError(f"Single-agent navigator creation failed: {e}") from e
    
    @staticmethod
    def multi(
        positions: Optional[np.ndarray] = None,
        orientations: Optional[np.ndarray] = None,
        speeds: Optional[np.ndarray] = None,
        max_speeds: Optional[Union[np.ndarray, List[float], float]] = None,
        angular_velocities: Optional[np.ndarray] = None,
        config: Optional[Union['DictConfig', Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> MultiAgentController:
        """
        Create a multi-agent navigation controller.
        
        This factory method creates a MultiAgentController instance optimized for
        vectorized operations on multiple agents simultaneously. It supports up to
        100 agents with efficient memory usage and computational performance.
        
        Parameters
        ----------
        positions : Optional[np.ndarray], optional
            Array of initial positions with shape (num_agents, 2), by default None
        orientations : Optional[np.ndarray], optional
            Array of initial orientations with shape (num_agents,), by default None
        speeds : Optional[np.ndarray], optional
            Array of initial speeds with shape (num_agents,), by default None
        max_speeds : Optional[Union[np.ndarray, List[float], float]], optional
            Maximum speeds per agent or single value for all agents, by default None
        angular_velocities : Optional[np.ndarray], optional
            Array of angular velocities with shape (num_agents,), by default None
        config : Optional[Union[DictConfig, Dict[str, Any]]], optional
            Hydra configuration object or dictionary, by default None
        **kwargs : Any
            Additional configuration parameters
            
        Returns
        -------
        MultiAgentController
            Configured multi-agent navigation controller
            
        Raises
        ------
        ValueError
            If array shapes are inconsistent or parameters invalid
        RuntimeError
            If controller initialization fails
            
        Examples
        --------
        Create with position array:
        
        >>> positions = np.array([[0, 0], [10, 10], [20, 20]])
        >>> navigator = Navigator.multi(
        ...     positions=positions,
        ...     max_speeds=5.0  # Same speed for all agents
        ... )
        
        Create with different speeds per agent:
        
        >>> navigator = Navigator.multi(
        ...     positions=positions,
        ...     max_speeds=[5.0, 6.0, 7.0]  # Different speeds
        ... )
        
        Create with Hydra configuration:
        
        >>> config = DictConfig({
        ...     'positions': [[0, 0], [5, 5]],
        ...     'max_speeds': [4.0, 6.0]
        ... })
        >>> navigator = Navigator.multi(config=config)
        """
        try:
            # Handle scalar max_speeds for convenience
            if isinstance(max_speeds, (int, float)):
                if positions is not None:
                    num_agents = positions.shape[0] if hasattr(positions, 'shape') else len(positions)
                    max_speeds = np.full(num_agents, max_speeds)
                else:
                    max_speeds = np.array([max_speeds])
            elif isinstance(max_speeds, list):
                max_speeds = np.array(max_speeds)
            
            _logger.debug(
                "Creating multi-agent navigator",
                extra={
                    "num_positions": positions.shape[0] if positions is not None else None,
                    "has_orientations": orientations is not None,
                    "has_speeds": speeds is not None,
                    "max_speeds_type": type(max_speeds).__name__,
                    "has_config": config is not None
                }
            )
            
            controller = MultiAgentController(
                positions=positions,
                orientations=orientations,
                speeds=speeds,
                max_speeds=max_speeds,
                angular_velocities=angular_velocities,
                config=config,
                **kwargs
            )
            
            _logger.info(
                "Multi-agent navigator created successfully",
                extra={
                    "controller_type": "MultiAgentController",
                    "num_agents": controller.num_agents,
                    "mean_max_speed": float(np.mean(controller.max_speeds)),
                    "position_bounds": {
                        "x_range": [float(np.min(controller.positions[:, 0])), 
                                  float(np.max(controller.positions[:, 0]))],
                        "y_range": [float(np.min(controller.positions[:, 1])), 
                                  float(np.max(controller.positions[:, 1]))]
                    }
                }
            )
            
            return controller
            
        except Exception as e:
            _logger.error(
                f"Failed to create multi-agent navigator: {e}",
                extra={"error_type": type(e).__name__}
            )
            raise RuntimeError(f"Multi-agent navigator creation failed: {e}") from e
    
    @staticmethod
    def from_config(config: Union['DictConfig', Dict[str, Any]]) -> Union[SingleAgentController, MultiAgentController]:
        """
        Create a navigator from Hydra configuration.
        
        This factory method automatically selects the appropriate controller type
        (single or multi-agent) based on the configuration structure and creates
        the corresponding navigator instance with all specified parameters.
        
        Parameters
        ----------
        config : Union[DictConfig, Dict[str, Any]]
            Hydra configuration object or configuration dictionary containing
            navigator parameters. The configuration structure determines whether
            a single-agent or multi-agent controller is created.
            
        Returns
        -------
        Union[SingleAgentController, MultiAgentController]
            Configured navigator instance based on configuration type
            
        Raises
        ------
        ValueError
            If configuration format is invalid or required fields are missing
        RuntimeError
            If navigator creation fails
        TypeError
            If configuration type is not supported
            
        Examples
        --------
        Create single-agent from config:
        
        >>> config = DictConfig({
        ...     'type': 'single',
        ...     'position': [50.0, 50.0],
        ...     'max_speed': 10.0,
        ...     'orientation': 45.0
        ... })
        >>> navigator = Navigator.from_config(config)
        
        Create multi-agent from config:
        
        >>> config = DictConfig({
        ...     'type': 'multi',
        ...     'positions': [[0, 0], [10, 10]],
        ...     'max_speeds': [5.0, 6.0]
        ... })
        >>> navigator = Navigator.from_config(config)
        
        Auto-detect type from structure:
        
        >>> config = DictConfig({
        ...     'position': [25.0, 25.0],  # Single position -> single agent
        ...     'max_speed': 8.0
        ... })
        >>> navigator = Navigator.from_config(config)
        
        Notes
        -----
        The factory method uses the following logic to determine navigator type:
        1. If 'type' field is specified, use that value ('single' or 'multi')
        2. If 'positions' field exists (array), create multi-agent controller
        3. If 'position' field exists (tuple/list), create single-agent controller
        4. Default to single-agent controller if structure is ambiguous
        """
        try:
            if not isinstance(config, (dict, DictConfig if HYDRA_AVAILABLE else dict)):
                raise TypeError(f"Configuration must be dict or DictConfig, got {type(config)}")
            
            # Convert DictConfig to dict for easier processing
            if HYDRA_AVAILABLE and hasattr(config, '_content'):
                config_dict = dict(config)
            else:
                config_dict = config
            
            _logger.debug(
                "Creating navigator from configuration",
                extra={
                    "config_type": type(config).__name__,
                    "config_keys": list(config_dict.keys()),
                    "has_type_field": 'type' in config_dict
                }
            )
            
            # Determine navigator type from configuration
            nav_type = config_dict.get('type', None)
            
            if nav_type == 'single':
                navigator = Navigator.single(config=config)
            elif nav_type == 'multi':
                navigator = Navigator.multi(config=config)
            elif 'positions' in config_dict:
                # Multi-agent: has positions array
                navigator = Navigator.multi(config=config)
            elif 'position' in config_dict:
                # Single-agent: has single position
                navigator = Navigator.single(config=config)
            else:
                # Default to single-agent for ambiguous configurations
                _logger.warning(
                    "Configuration type ambiguous, defaulting to single-agent",
                    extra={"config_keys": list(config_dict.keys())}
                )
                navigator = Navigator.single(config=config)
            
            _logger.info(
                "Navigator created from configuration",
                extra={
                    "navigator_type": type(navigator).__name__,
                    "num_agents": getattr(navigator, 'num_agents', 1),
                    "config_source": "hydra" if HYDRA_AVAILABLE and hasattr(config, '_content') else "dict"
                }
            )
            
            return navigator
            
        except Exception as e:
            _logger.error(
                f"Failed to create navigator from config: {e}",
                extra={"error_type": type(e).__name__, "config_type": type(config).__name__}
            )
            raise RuntimeError(f"Navigator creation from config failed: {e}") from e


def run_simulation(
    navigator: NavigatorProtocol,
    video_plume: Any,
    max_frames: Optional[int] = None,
    record_trajectory: bool = True,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Execute a complete navigation simulation.
    
    This function provides a simplified interface for running complete navigation
    simulations, coordinating frame-by-frame execution between the navigator and
    video plume environment. It automatically handles trajectory recording and
    provides comprehensive simulation results.
    
    This function serves as a convenient wrapper around the full simulation API
    available in the `{{cookiecutter.project_slug}}.api.navigation` module, providing
    a streamlined interface for common simulation scenarios.
    
    Parameters
    ----------
    navigator : NavigatorProtocol
        Navigation controller implementing the NavigatorProtocol interface
    video_plume : Any
        Video plume environment providing frame data (typically VideoPlume instance)
    max_frames : Optional[int], optional
        Maximum number of frames to process, by default None (process all frames)
    record_trajectory : bool, optional
        Whether to record trajectory data during simulation, by default True
    **kwargs : Any
        Additional simulation parameters passed to the underlying API
        
    Returns
    -------
    Dict[str, Any]
        Simulation results containing:
        - 'trajectory': Recorded agent positions over time (if record_trajectory=True)
        - 'orientations': Recorded agent orientations over time
        - 'speeds': Recorded agent speeds over time  
        - 'odor_readings': Recorded odor concentration samples
        - 'frame_count': Number of frames processed
        - 'simulation_time': Total simulation duration
        - 'metadata': Additional simulation metadata
        
    Raises
    ------
    ImportError
        If the required API module is not available
    ValueError
        If navigator or video_plume parameters are invalid
    RuntimeError
        If simulation execution fails
        
    Examples
    --------
    Basic simulation execution:
    
    >>> from {{cookiecutter.project_slug}}.core import Navigator, run_simulation
    >>> from {{cookiecutter.project_slug}}.data import VideoPlume
    >>> 
    >>> navigator = Navigator.single(max_speed=10.0)
    >>> video_plume = VideoPlume.from_file("example.mp4")
    >>> results = run_simulation(navigator, video_plume)
    >>> print(f"Processed {results['frame_count']} frames")
    
    Limited frame simulation:
    
    >>> results = run_simulation(
    ...     navigator, 
    ...     video_plume, 
    ...     max_frames=100,
    ...     record_trajectory=True
    ... )
    
    Multi-agent simulation:
    
    >>> navigator = Navigator.multi(positions=[[0, 0], [10, 10]])
    >>> results = run_simulation(navigator, video_plume)
    >>> trajectory = results['trajectory']  # Shape: (num_frames, num_agents, 2)
    
    Notes
    -----
    This function is designed for convenience and common use cases. For advanced
    simulation features such as real-time visualization, custom data recording,
    or complex parameter management, use the full API available in
    `{{cookiecutter.project_slug}}.api.navigation.run_plume_simulation()`.
    
    The function automatically handles:
    - Frame-by-frame navigation updates
    - Odor sampling at agent positions
    - Trajectory data accumulation
    - Performance monitoring and logging
    - Error handling and recovery
    """
    try:
        # Lazy import to avoid circular dependencies
        from ..api.navigation import run_plume_simulation
        
        _logger.info(
            "Starting navigation simulation",
            extra={
                "navigator_type": type(navigator).__name__,
                "navigator_agents": getattr(navigator, 'num_agents', 1),
                "video_plume_type": type(video_plume).__name__,
                "max_frames": max_frames,
                "record_trajectory": record_trajectory
            }
        )
        
        # Execute simulation using the full API
        results = run_plume_simulation(
            navigator=navigator,
            video_plume=video_plume,
            max_frames=max_frames,
            record_trajectory=record_trajectory,
            **kwargs
        )
        
        _logger.info(
            "Navigation simulation completed successfully",
            extra={
                "frames_processed": results.get('frame_count', 0),
                "simulation_duration": results.get('simulation_time', 0),
                "trajectory_recorded": 'trajectory' in results
            }
        )
        
        return results
        
    except ImportError as e:
        _logger.error(
            f"Failed to import simulation API: {e}",
            extra={"error_type": "ImportError"}
        )
        raise ImportError(
            f"Simulation API not available: {e}. "
            "Ensure the {{cookiecutter.project_slug}}.api.navigation module is properly installed."
        ) from e
    except Exception as e:
        _logger.error(
            f"Simulation execution failed: {e}",
            extra={
                "error_type": type(e).__name__,
                "navigator_type": type(navigator).__name__
            }
        )
        raise RuntimeError(f"Navigation simulation failed: {e}") from e


# Export public API
__all__ = [
    # Core Protocol Interface
    "NavigatorProtocol",
    
    # Controller Implementations  
    "SingleAgentController",
    "MultiAgentController",
    
    # Factory Interface
    "Navigator",
    
    # Simulation Interface
    "run_simulation",
]

# Module metadata
__version__ = "1.0.0"
__author__ = "Odor Plume Navigation Team"
__description__ = "Core navigation module for odor plume navigation library"

# Log module initialization
_logger.info(
    "Core navigation module initialized",
    extra={
        "exported_components": len(__all__),
        "hydra_available": HYDRA_AVAILABLE,
        "module_version": __version__
    }
)