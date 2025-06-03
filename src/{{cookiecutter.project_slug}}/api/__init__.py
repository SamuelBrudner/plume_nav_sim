"""
Public API module for the Odor Plume Navigation library.

This module provides unified access to all high-level functionality for odor plume navigation,
including navigator creation, video plume environments, simulation execution, and visualization.
The API is designed to support Kedro pipelines, reinforcement learning frameworks, and machine
learning analysis tools through clean, consistent interfaces with comprehensive Hydra-based
configuration management.

The API supports the following import patterns:

For Kedro projects:
    from {{cookiecutter.project_slug}} import Navigator, VideoPlume
    from {{cookiecutter.project_slug}}.config import NavigatorConfig

For RL projects:
    from {{cookiecutter.project_slug}}.core import NavigatorProtocol
    from {{cookiecutter.project_slug}}.api import create_navigator

For ML/neural network analyses:
    from {{cookiecutter.project_slug}}.utils import set_global_seed
    from {{cookiecutter.project_slug}}.data import VideoPlume

Example Usage:
    # Basic navigation simulation
    navigator = create_navigator(position=(10.0, 20.0), max_speed=5.0)
    video_plume = create_video_plume("path/to/video.mp4")
    results = run_plume_simulation(navigator, video_plume, num_steps=1000)
    visualize_simulation_results(results)
    
    # Using Hydra configuration (Kedro integration)
    from hydra import compose, initialize
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config")
        navigator = create_navigator(cfg=cfg.navigator)
        video_plume = create_video_plume(cfg=cfg.video_plume)
        results = run_plume_simulation(navigator, video_plume, cfg=cfg.simulation)
        visualize_simulation_results(results, cfg=cfg.visualization)
        
    # Multi-agent simulation with parameter overrides
    navigator = create_navigator(
        positions=[(0, 0), (10, 10), (20, 20)],
        max_speeds=[5, 6, 7],
        seed=42
    )
    results = run_plume_simulation(navigator, video_plume, num_steps=500)
    visualize_trajectory(results, export_path="trajectory.pdf")
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Import Hydra types with graceful fallback
try:
    from omegaconf import DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    DictConfig = dict
    HYDRA_AVAILABLE = False
    warnings.warn(
        "Hydra/OmegaConf not available. Some advanced configuration features may be limited.",
        ImportWarning
    )

# Import core API functions from navigation module
from {{cookiecutter.project_slug}}.api.navigation import (
    # Primary API functions
    create_navigator,
    create_video_plume,
    run_plume_simulation,
    
    # Factory methods for backward compatibility
    create_navigator_from_config,
    create_video_plume_from_config,
    
    # Validation and configuration utilities
    _validate_and_merge_config,
    _normalize_positions,
    
    # Exceptions
    ConfigurationError,
    SimulationError,
)

# Import visualization functions with graceful fallback
try:
    from {{cookiecutter.project_slug}}.utils.visualization import (
        # Real-time animation interface
        SimulationVisualization,
        
        # Static trajectory plotting
        visualize_trajectory,
        
        # High-level visualization function
        visualize_simulation_results,
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    # Provide fallback implementations for environments without visualization dependencies
    VISUALIZATION_AVAILABLE = False
    
    def visualize_simulation_results(*args, **kwargs):
        """Fallback implementation when visualization is not available."""
        warnings.warn(
            "Visualization module not available. Install matplotlib and related dependencies.",
            ImportWarning
        )
        return None
    
    def visualize_trajectory(*args, **kwargs):
        """Fallback implementation when visualization is not available."""
        warnings.warn(
            "Visualization module not available. Install matplotlib and related dependencies.",
            ImportWarning
        )
        return None
    
    class SimulationVisualization:
        """Fallback class when visualization is not available."""
        def __init__(self, *args, **kwargs):
            warnings.warn(
                "Visualization module not available. Install matplotlib and related dependencies.",
                ImportWarning
            )

# Import core types for convenience and type hints
from {{cookiecutter.project_slug}}.core.navigator import Navigator, NavigatorProtocol
from {{cookiecutter.project_slug}}.core.controllers import SingleAgentController, MultiAgentController
from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume

# Import configuration schemas for validation and type hints
from {{cookiecutter.project_slug}}.config.schemas import (
    NavigatorConfig,
    VideoPlumeConfig,
    SimulationConfig,
    SingleAgentConfig,
    MultiAgentConfig,
)

# Import seed management for reproducibility
from {{cookiecutter.project_slug}}.utils.seed_manager import set_global_seed, get_current_seed


def create_navigation_session(
    cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
    navigator_params: Optional[Dict[str, Any]] = None,
    video_plume_params: Optional[Dict[str, Any]] = None,
    simulation_params: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    **kwargs: Any
) -> Tuple[Navigator, VideoPlume]:
    """
    Create a complete navigation session with navigator and video plume components.
    
    This is a convenience function that combines navigator and video plume creation
    with unified configuration management, providing a one-stop setup for complete
    simulation sessions. It supports both Hydra configuration objects and direct
    parameter specification.
    
    Args:
        cfg: Hydra DictConfig or dictionary containing complete session configuration
        navigator_params: Parameters for navigator creation (overrides cfg.navigator)
        video_plume_params: Parameters for video plume creation (overrides cfg.video_plume)
        simulation_params: Parameters for simulation setup (overrides cfg.simulation)
        seed: Global random seed for session reproducibility
        **kwargs: Additional parameters distributed to appropriate components
        
    Returns:
        Tuple of (Navigator, VideoPlume): Ready-to-use navigation session components
        
    Raises:
        ConfigurationError: If required parameters are missing or invalid
        FileNotFoundError: If video file does not exist
        
    Examples:
        # Complete session from Hydra configuration
        navigator, video_plume = create_navigation_session(cfg=hydra_config)
        
        # Session with direct parameters
        navigator, video_plume = create_navigation_session(
            navigator_params={"position": (10, 20), "max_speed": 5.0},
            video_plume_params={"video_path": "video.mp4", "flip": True},
            seed=42
        )
        
        # Mixed configuration with overrides
        navigator, video_plume = create_navigation_session(
            cfg=base_config,
            navigator_params={"max_speed": 8.0},  # Override config
            seed=123
        )
    """
    # Set global seed if provided
    if seed is not None:
        set_global_seed(seed)
    
    # Extract configuration sections
    nav_config = None
    plume_config = None
    
    if cfg is not None:
        if HYDRA_AVAILABLE and isinstance(cfg, DictConfig):
            # Extract nested configurations from DictConfig
            nav_config = cfg.get('navigator', None)
            plume_config = cfg.get('video_plume', None)
        elif isinstance(cfg, dict):
            # Extract from dictionary
            nav_config = cfg.get('navigator', None)
            plume_config = cfg.get('video_plume', None)
    
    # Merge with direct parameter overrides
    final_nav_params = {}
    if nav_config is not None:
        final_nav_params.update(_validate_and_merge_config(nav_config))
    if navigator_params is not None:
        final_nav_params.update(navigator_params)
    
    final_plume_params = {}
    if plume_config is not None:
        final_plume_params.update(_validate_and_merge_config(plume_config))
    if video_plume_params is not None:
        final_plume_params.update(video_plume_params)
    
    # Create components
    try:
        navigator = create_navigator(cfg=final_nav_params, **kwargs)
        video_plume = create_video_plume(cfg=final_plume_params, **kwargs)
        
        return navigator, video_plume
        
    except Exception as e:
        raise ConfigurationError(f"Failed to create navigation session: {e}") from e


def run_complete_experiment(
    cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
    navigator: Optional[Navigator] = None,
    video_plume: Optional[VideoPlume] = None,
    auto_visualize: bool = True,
    save_results: bool = True,
    output_dir: Optional[str] = None,
    experiment_name: Optional[str] = None,
    seed: Optional[int] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Execute a complete odor plume navigation experiment with automatic setup and visualization.
    
    This high-level function orchestrates the entire experimental workflow, from component
    creation through simulation execution to result visualization and saving. It provides
    comprehensive experiment management with automatic result organization and metadata
    preservation.
    
    Args:
        cfg: Hydra DictConfig containing complete experiment configuration
        navigator: Pre-configured Navigator instance (optional, created from config if None)
        video_plume: Pre-configured VideoPlume instance (optional, created from config if None)
        auto_visualize: Whether to automatically generate visualizations
        save_results: Whether to save simulation results to disk
        output_dir: Directory for saving results (defaults to Hydra working directory)
        experiment_name: Name for the experiment (used in file naming)
        seed: Global random seed for experiment reproducibility
        **kwargs: Additional parameters passed to simulation and visualization
        
    Returns:
        Dict containing:
            - 'positions': Agent position trajectories
            - 'orientations': Agent orientation trajectories  
            - 'odor_readings': Sensor odor readings
            - 'metadata': Experiment metadata and configuration
            - 'visualization_paths': Paths to generated visualization files (if auto_visualize=True)
            - 'results_path': Path to saved results file (if save_results=True)
            
    Raises:
        ConfigurationError: If configuration is invalid or incomplete
        SimulationError: If simulation execution fails
        
    Examples:
        # Complete experiment from Hydra configuration
        results = run_complete_experiment(cfg=hydra_config, experiment_name="test_run")
        
        # Experiment with pre-configured components
        navigator, video_plume = create_navigation_session(cfg=config)
        results = run_complete_experiment(
            navigator=navigator,
            video_plume=video_plume,
            auto_visualize=True,
            seed=42
        )
        
        # Custom experiment with parameter overrides
        results = run_complete_experiment(
            cfg=base_config,
            auto_visualize=True,
            save_results=True,
            output_dir="experiments/",
            experiment_name="multi_agent_test",
            num_steps=2000,  # Override simulation parameters
            seed=123
        )
    """
    import pathlib
    import datetime
    
    # Set global seed if provided
    if seed is not None:
        set_global_seed(seed)
    
    # Generate experiment metadata
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name is None:
        experiment_name = f"experiment_{timestamp}"
    
    # Create components if not provided
    if navigator is None or video_plume is None:
        if cfg is None:
            raise ConfigurationError("Either cfg must be provided or both navigator and video_plume must be pre-configured")
        
        if navigator is None and video_plume is None:
            navigator, video_plume = create_navigation_session(cfg=cfg, seed=seed)
        elif navigator is None:
            navigator = create_navigator(cfg=cfg.get('navigator', {}), seed=seed)
        elif video_plume is None:
            video_plume = create_video_plume(cfg=cfg.get('video_plume', {}))
    
    # Execute simulation
    try:
        positions, orientations, odor_readings = run_plume_simulation(
            navigator=navigator,
            video_plume=video_plume,
            cfg=cfg.get('simulation', {}) if cfg else {},
            seed=seed,
            **kwargs
        )
    except Exception as e:
        raise SimulationError(f"Experiment simulation failed: {e}") from e
    
    # Prepare results dictionary
    results = {
        'positions': positions,
        'orientations': orientations,
        'odor_readings': odor_readings,
        'metadata': {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'seed': get_current_seed(),
            'num_agents': navigator.num_agents,
            'num_steps': positions.shape[1] - 1,
            'configuration': cfg if cfg is not None else {},
        }
    }
    
    # Determine output directory
    if output_dir is None:
        # Use current working directory or Hydra working directory
        output_path = pathlib.Path.cwd()
    else:
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Save results if requested
    if save_results:
        try:
            results_file = output_path / f"{experiment_name}_results.npz"
            np.savez_compressed(
                results_file,
                positions=positions,
                orientations=orientations,
                odor_readings=odor_readings,
                metadata=results['metadata']
            )
            results['results_path'] = str(results_file)
        except Exception as e:
            warnings.warn(f"Failed to save results: {e}", UserWarning)
    
    # Generate visualizations if requested
    if auto_visualize and VISUALIZATION_AVAILABLE:
        try:
            visualization_paths = []
            
            # Generate trajectory plot
            trajectory_file = output_path / f"{experiment_name}_trajectory.png"
            fig = visualize_trajectory(
                results,
                cfg=cfg.get('visualization', {}) if cfg else {},
                export_path=str(trajectory_file)
            )
            if fig is not None:
                visualization_paths.append(str(trajectory_file))
            
            # Generate animation if requested in config
            viz_config = cfg.get('visualization', {}) if cfg else {}
            if viz_config.get('generate_animation', False):
                animation_file = output_path / f"{experiment_name}_animation.mp4"
                visualize_simulation_results(
                    results,
                    cfg=viz_config,
                    export_path=str(animation_file),
                    show_animation=False  # Save only, don't display
                )
                visualization_paths.append(str(animation_file))
            
            results['visualization_paths'] = visualization_paths
            
        except Exception as e:
            warnings.warn(f"Failed to generate visualizations: {e}", UserWarning)
            results['visualization_paths'] = []
    
    return results


# Define public API exports
__all__ = [
    # Primary API functions
    'create_navigator',
    'create_video_plume',
    'run_plume_simulation',
    
    # High-level convenience functions
    'create_navigation_session',
    'run_complete_experiment',
    
    # Visualization functions
    'visualize_simulation_results',
    'visualize_trajectory',
    'SimulationVisualization',
    
    # Factory methods (backward compatibility)
    'create_navigator_from_config',
    'create_video_plume_from_config',
    
    # Core types for type hints and direct access
    'Navigator',
    'NavigatorProtocol',
    'VideoPlume',
    'SingleAgentController',
    'MultiAgentController',
    
    # Configuration schemas
    'NavigatorConfig',
    'VideoPlumeConfig',
    'SimulationConfig',
    'SingleAgentConfig',
    'MultiAgentConfig',
    
    # Utility functions
    'set_global_seed',
    'get_current_seed',
    
    # Exceptions
    'ConfigurationError',
    'SimulationError',
    
    # Constants and availability flags
    'HYDRA_AVAILABLE',
    'VISUALIZATION_AVAILABLE',
]

# Version and metadata information
__version__ = "1.0.0"
__author__ = "Odor Plume Navigation Team"
__email__ = "support@odorplumenav.org"
__description__ = "A reusable library for odor plume navigation simulation and analysis"