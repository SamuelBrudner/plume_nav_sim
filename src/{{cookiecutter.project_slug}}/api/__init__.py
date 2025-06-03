"""
Public API module for odor plume navigation library.

This module serves as the primary entry point for external consumers of the odor plume 
navigation library, providing unified access to all high-level functionality through 
clean, stable interfaces. Re-exports key factory methods, simulation execution functions, 
and visualization capabilities to enable seamless integration with Kedro pipelines, 
reinforcement learning frameworks, and machine learning analysis tools.

The API is designed with Hydra-based configuration management at its core, supporting
sophisticated parameter composition, environment variable interpolation, and 
multi-run experiment execution. All functions accept both direct parameters and 
Hydra DictConfig objects, ensuring compatibility with diverse research workflows.

Key Features:
    - Unified factory methods for navigator and environment creation
    - Hydra-based configuration management with hierarchical parameter composition
    - Support for both single-agent and multi-agent navigation scenarios  
    - Integration with scientific Python ecosystem (NumPy, Matplotlib, OpenCV)
    - Protocol-based interfaces ensuring extensibility and algorithm compatibility
    - Publication-quality visualization with real-time animation capabilities

Supported Import Patterns:
    Kedro pipeline integration:
        >>> from {{cookiecutter.project_slug}}.api import create_navigator, create_video_plume
        >>> from {{cookiecutter.project_slug}}.api import run_plume_simulation
    
    Reinforcement learning frameworks:
        >>> from {{cookiecutter.project_slug}}.api import create_navigator
        >>> from {{cookiecutter.project_slug}}.core import NavigatorProtocol
    
    Machine learning analysis tools:
        >>> from {{cookiecutter.project_slug}}.api import create_video_plume, visualize_simulation_results
        >>> from {{cookiecutter.project_slug}}.utils import set_global_seed

Configuration Management:
    All API functions support Hydra-based configuration through DictConfig objects:
        >>> from hydra import compose, initialize
        >>> from {{cookiecutter.project_slug}}.api import create_navigator
        >>> 
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        ...     navigator = create_navigator(cfg.navigator)
        ...     plume = create_video_plume(cfg.video_plume)
        ...     results = run_plume_simulation(navigator, plume, cfg.simulation)

Performance Characteristics:
    - Factory method initialization: <10ms for typical configurations
    - Multi-agent support: Up to 100 simultaneous agents with vectorized operations
    - Real-time visualization: 30+ FPS animation performance
    - Memory efficiency: Optimized NumPy array usage for large-scale simulations

Backward Compatibility:
    The API maintains compatibility with legacy interfaces while providing enhanced
    Hydra-based functionality. Legacy parameter patterns are supported alongside
    new configuration-driven approaches.
"""

from typing import Union, Optional, Tuple, Any, Dict, List
import pathlib
import numpy as np

# Core dependency imports for type hints
try:
    from omegaconf import DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = Dict[str, Any]  # Fallback type hint

# Import core API functions from navigation module
from .navigation import (
    create_navigator,
    create_video_plume, 
    run_plume_simulation,
    visualize_plume_simulation,
    # Legacy compatibility aliases
    create_navigator_from_config,
    create_video_plume_from_config,
    run_simulation,
)

# Import visualization functions from utils module
from ..utils.visualization import (
    visualize_simulation_results,
    visualize_trajectory,
    SimulationVisualization,
    batch_visualize_trajectories,
    setup_headless_mode,
    get_available_themes,
    create_simulation_visualization,
    export_animation,
)

# Import core protocols for type hints and advanced usage
from ..core.navigator import NavigatorProtocol
from ..data.video_plume import VideoPlume


def create_navigator_instance(
    cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
    **kwargs: Any
) -> NavigatorProtocol:
    """
    Alias for create_navigator providing enhanced documentation.
    
    This function creates navigator instances with comprehensive Hydra configuration 
    support, automatic parameter validation, and performance optimization for both 
    single-agent and multi-agent scenarios.
    
    Parameters
    ----------
    cfg : Optional[Union[DictConfig, Dict[str, Any]]], optional
        Hydra configuration object containing navigator parameters, by default None
    **kwargs : Any
        Direct parameter specification (overrides cfg values)
    
    Returns
    -------
    NavigatorProtocol
        Configured navigator instance ready for simulation use
    
    Examples
    --------
    Create single-agent navigator:
        >>> navigator = create_navigator_instance(
        ...     position=(50.0, 50.0),
        ...     orientation=45.0,
        ...     max_speed=10.0
        ... )
    
    Create with Hydra configuration:
        >>> from hydra import compose, initialize
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        ...     navigator = create_navigator_instance(cfg.navigator)
    
    Create multi-agent navigator:
        >>> navigator = create_navigator_instance(
        ...     positions=[(10, 20), (30, 40)],
        ...     orientations=[0, 90],
        ...     max_speeds=[5.0, 8.0]
        ... )
    """
    return create_navigator(cfg=cfg, **kwargs)


def create_video_plume_instance(
    cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
    **kwargs: Any
) -> VideoPlume:
    """
    Alias for create_video_plume providing enhanced documentation.
    
    This function creates VideoPlume instances with comprehensive video processing
    capabilities, Hydra configuration integration, and automatic parameter validation.
    
    Parameters
    ----------
    cfg : Optional[Union[DictConfig, Dict[str, Any]]], optional
        Hydra configuration object containing video plume parameters, by default None
    **kwargs : Any
        Direct parameter specification (overrides cfg values)
    
    Returns
    -------
    VideoPlume
        Configured VideoPlume instance ready for simulation use
    
    Examples
    --------
    Create with direct parameters:
        >>> plume = create_video_plume_instance(
        ...     video_path="data/plume_video.mp4",
        ...     flip=True,
        ...     kernel_size=5
        ... )
    
    Create with Hydra configuration:
        >>> plume = create_video_plume_instance(cfg.video_plume)
    """
    return create_video_plume(cfg=cfg, **kwargs)


def run_navigation_simulation(
    navigator: NavigatorProtocol,
    video_plume: VideoPlume,
    cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
    **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute complete odor plume navigation simulation with comprehensive data collection.
    
    This function orchestrates frame-by-frame agent navigation through video-based 
    odor plume environments with automatic trajectory recording, performance monitoring,
    and Hydra configuration support.
    
    Parameters
    ----------
    navigator : NavigatorProtocol
        Navigator instance (SingleAgentController or MultiAgentController)
    video_plume : VideoPlume
        VideoPlume environment providing odor concentration data
    cfg : Optional[Union[DictConfig, Dict[str, Any]]], optional
        Hydra configuration object containing simulation parameters, by default None
    **kwargs : Any
        Direct parameter specification (overrides cfg values)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        positions_history : Agent positions (num_agents, num_steps + 1, 2)
        orientations_history : Agent orientations (num_agents, num_steps + 1)
        odor_readings : Sensor readings (num_agents, num_steps + 1)
    
    Examples
    --------
    Basic simulation execution:
        >>> positions, orientations, readings = run_navigation_simulation(
        ...     navigator, plume, num_steps=1000, dt=0.1
        ... )
    
    Hydra-configured simulation:
        >>> results = run_navigation_simulation(
        ...     navigator, plume, cfg.simulation
        ... )
    """
    return run_plume_simulation(navigator, video_plume, cfg=cfg, **kwargs)


def visualize_results(
    positions: np.ndarray,
    orientations: np.ndarray,
    odor_readings: Optional[np.ndarray] = None,
    plume_frames: Optional[np.ndarray] = None,
    cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
    animation: bool = False,
    **kwargs: Any
) -> "matplotlib.figure.Figure":
    """
    Create comprehensive visualizations of simulation results.
    
    This function provides unified access to both static trajectory plots and 
    animated visualizations with publication-quality formatting and Hydra 
    configuration support.
    
    Parameters
    ----------
    positions : np.ndarray
        Agent positions with shape (num_agents, num_steps, 2)
    orientations : np.ndarray
        Agent orientations with shape (num_agents, num_steps)
    odor_readings : Optional[np.ndarray], optional
        Sensor readings with shape (num_agents, num_steps), by default None
    plume_frames : Optional[np.ndarray], optional
        Video frames for background visualization, by default None
    cfg : Optional[Union[DictConfig, Dict[str, Any]]], optional
        Hydra configuration for visualization parameters, by default None
    animation : bool, optional
        Whether to create animated visualization, by default False
    **kwargs : Any
        Additional visualization parameters
    
    Returns
    -------
    matplotlib.figure.Figure
        The created matplotlib figure object
    
    Examples
    --------
    Static trajectory plot:
        >>> fig = visualize_results(positions, orientations)
    
    Animated visualization:
        >>> fig = visualize_results(
        ...     positions, orientations, 
        ...     plume_frames=frames,
        ...     animation=True
        ... )
    
    Publication-quality export:
        >>> fig = visualize_results(
        ...     positions, orientations,
        ...     cfg=viz_config,
        ...     output_path="results/trajectory.png",
        ...     show_plot=False
        ... )
    """
    if animation:
        return visualize_simulation_results(
            positions=positions,
            orientations=orientations,
            odor_readings=odor_readings,
            plume_frames=plume_frames,
            **kwargs
        )
    else:
        return visualize_trajectory(
            positions=positions,
            orientations=orientations,
            plume_frames=plume_frames,
            config=cfg,
            **kwargs
        )


# Legacy compatibility functions for backward compatibility
def create_navigator_legacy(config_path: Optional[str] = None, **kwargs: Any) -> NavigatorProtocol:
    """
    Legacy navigator creation interface for backward compatibility.
    
    This function maintains compatibility with pre-Hydra configuration patterns
    while providing access to enhanced functionality through parameter passing.
    
    Parameters
    ----------
    config_path : Optional[str], optional
        Path to configuration file (legacy parameter), by default None
    **kwargs : Any
        Navigator configuration parameters
    
    Returns
    -------
    NavigatorProtocol
        Configured navigator instance
    
    Notes
    -----
    This function is provided for backward compatibility. New code should use
    create_navigator() with Hydra configuration support.
    """
    # Convert legacy config_path to modern parameter pattern
    if config_path is not None:
        # In legacy mode, we rely on direct parameters only
        # since we can't dynamically load YAML files without Hydra context
        import warnings
        warnings.warn(
            "config_path parameter is deprecated. Use Hydra configuration or direct parameters.",
            DeprecationWarning,
            stacklevel=2
        )
    
    return create_navigator(**kwargs)


def create_video_plume_legacy(config_path: Optional[str] = None, **kwargs: Any) -> VideoPlume:
    """
    Legacy video plume creation interface for backward compatibility.
    
    Parameters
    ----------
    config_path : Optional[str], optional
        Path to configuration file (legacy parameter), by default None
    **kwargs : Any
        VideoPlume configuration parameters
    
    Returns
    -------
    VideoPlume
        Configured VideoPlume instance
    
    Notes
    -----
    This function is provided for backward compatibility. New code should use
    create_video_plume() with Hydra configuration support.
    """
    if config_path is not None:
        import warnings
        warnings.warn(
            "config_path parameter is deprecated. Use Hydra configuration or direct parameters.",
            DeprecationWarning,
            stacklevel=2
        )
    
    return create_video_plume(**kwargs)


# Module metadata and version information
__version__ = "1.0.0"
__author__ = "Odor Plume Navigation Team"
__description__ = "Public API for odor plume navigation library with Hydra configuration support"

# Export all public functions and classes
__all__ = [
    # Primary factory methods
    "create_navigator",
    "create_video_plume",
    "run_plume_simulation",
    
    # Enhanced API aliases
    "create_navigator_instance", 
    "create_video_plume_instance",
    "run_navigation_simulation",
    "visualize_results",
    
    # Visualization functions
    "visualize_simulation_results",
    "visualize_trajectory", 
    "visualize_plume_simulation",
    "SimulationVisualization",
    "batch_visualize_trajectories",
    "setup_headless_mode",
    "get_available_themes",
    "create_simulation_visualization",
    "export_animation",
    
    # Core protocols and classes for advanced usage
    "NavigatorProtocol",
    "VideoPlume",
    
    # Legacy compatibility functions
    "create_navigator_from_config",
    "create_video_plume_from_config", 
    "run_simulation",
    "create_navigator_legacy",
    "create_video_plume_legacy",
    
    # Module metadata
    "__version__",
    "__author__",
    "__description__",
]

# Conditional exports based on Hydra availability
if HYDRA_AVAILABLE:
    # Add Hydra-specific functionality to exports
    __all__.extend([
        "DictConfig",  # Re-export for type hints
    ])

# Package initialization message (optional, for development/debugging)
def _get_api_info() -> Dict[str, Any]:
    """Get API module information for debugging and introspection."""
    return {
        "version": __version__,
        "hydra_available": HYDRA_AVAILABLE,
        "public_functions": len(__all__),
        "primary_functions": [
            "create_navigator", 
            "create_video_plume", 
            "run_plume_simulation",
            "visualize_simulation_results"
        ],
        "legacy_support": True,
        "configuration_types": ["direct_parameters", "hydra_dictconfig"] + (
            ["yaml_files"] if HYDRA_AVAILABLE else []
        )
    }


# Optional: Expose API info for debugging
def get_api_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the API module.
    
    This function provides metadata about available functions, configuration
    options, and system capabilities for debugging and introspection.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing API module information
    
    Examples
    --------
    >>> from {{cookiecutter.project_slug}}.api import get_api_info
    >>> info = get_api_info()
    >>> print(f"API version: {info['version']}")
    >>> print(f"Hydra support: {info['hydra_available']}")
    """
    return _get_api_info()