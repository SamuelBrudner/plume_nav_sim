"""
Public API module for odor plume navigation library.

This module serves as the primary entry point for external consumers of the odor plume 
navigation library, providing unified access to all high-level functionality through 
clean, stable interfaces. Re-exports key factory methods, simulation execution functions, 
visualization capabilities, and reinforcement learning framework integration to enable 
seamless integration with Kedro pipelines, 
stable-baselines3 algorithms, and machine learning analysis tools.

The API is designed with Hydra-based configuration management at its core, supporting
sophisticated parameter composition, environment variable interpolation, and 
multi-run experiment execution. All functions accept both direct parameters and 
Hydra DictConfig objects, ensuring compatibility with diverse research workflows.

Key Features:
    - Unified factory methods for navigator and environment creation
    - Gymnasium-compliant RL environment interface for modern ML frameworks
    - Hydra-based configuration management with hierarchical parameter composition
    - Support for both single-agent and multi-agent navigation scenarios  
    - Integration with scientific Python ecosystem (NumPy, Matplotlib, OpenCV)
    - Protocol-based interfaces ensuring extensibility and algorithm compatibility
    - Publication-quality visualization with real-time animation capabilities
    - Legacy API migration support for smooth workflow transitions

Supported Import Patterns:
    Kedro pipeline integration:
        >>> from odor_plume_nav.api import create_navigator, create_video_plume
        >>> from odor_plume_nav.api import run_plume_simulation
    
    Reinforcement learning frameworks:
        >>> from odor_plume_nav.api import create_navigator
        >>> from odor_plume_nav.core import NavigatorProtocol
        >>> import stable_baselines3 as sb3
        >>> 
        >>> # Create Gymnasium environment for RL training
        >>> env = create_gymnasium_environment(cfg.gymnasium)
        >>> model = sb3.PPO("MlpPolicy", env, verbose=1)
        >>> model.learn(total_timesteps=10000)
    
    Legacy API migration:
        >>> from odor_plume_nav.api import from_legacy
        >>> from odor_plume_nav.api import create_navigator, run_plume_simulation
        >>> 
        >>> # Migrate existing simulation to Gymnasium environment
        >>> navigator = create_navigator(cfg.navigator)
        >>> plume = create_video_plume(cfg.video_plume)
        >>> env = from_legacy(navigator, plume, cfg.simulation)
    
    Machine learning analysis tools:
        >>> from odor_plume_nav.api import create_video_plume, visualize_trajectory
        >>> from odor_plume_nav.utils import set_global_seed

Configuration Management:
    All API functions support Hydra-based configuration through DictConfig objects:
        >>> from hydra import compose, initialize
        >>> from odor_plume_nav.api import create_navigator, create_gymnasium_environment
        >>> 
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        ...     navigator = create_navigator(cfg.navigator)
        ...     plume = create_video_plume(cfg.video_plume)
        ...     results = run_plume_simulation(navigator, plume, cfg.simulation)
        ...     
        ...     # Or create RL environment
        ...     rl_env = create_gymnasium_environment(cfg.gymnasium)

Performance Characteristics:
    - Factory method initialization: <10ms for typical configurations
    - Multi-agent support: Up to 100 simultaneous agents with vectorized operations
    - Real-time visualization: 30+ FPS animation performance
    - Memory efficiency: Optimized NumPy array usage for large-scale simulations
    - RL environment step overhead: <1ms for step/reset operations

Backward Compatibility:
    The API maintains compatibility with legacy interfaces while providing enhanced
    Hydra-based functionality. Legacy parameter patterns are supported alongside
    new configuration-driven approaches through the from_legacy migration interface.
"""

from typing import Union, Optional, Tuple, Any, Dict, List
import pathlib
import warnings
import inspect
import numpy as np

# Enhanced logging setup for deprecation warnings and structured messaging
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    # Fallback to standard logging if Loguru not available
    import logging
    logger = logging.getLogger(__name__)

# Core dependency imports for type hints
try:
    from omegaconf import DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = Dict[str, Any]  # Fallback type hint


def _detect_legacy_gym_import() -> bool:
    """
    Detect if the calling code is using legacy gym imports.
    
    This function inspects the call stack to identify potential legacy gym
    import patterns and triggers appropriate deprecation warnings through
    structured logging as per Section 5.4.3.2 legacy API deprecation handling.
    
    Returns
    -------
    bool
        True if legacy gym import patterns are detected, False otherwise
    """
    try:
        # Inspect the call stack to detect gym imports
        frame = inspect.currentframe()
        if frame is None:
            return False
            
        # Look back through the call stack for indications of legacy gym usage
        for i in range(10):  # Check up to 10 frames back
            frame = frame.f_back
            if frame is None:
                break
                
            # Check for gym module in globals or locals
            frame_globals = frame.f_globals
            frame_locals = frame.f_locals
            
            # Look for gym imports in the calling module
            if 'gym' in frame_globals and 'gymnasium' not in frame_globals:
                # Check if it's the legacy gym package
                gym_module = frame_globals.get('gym')
                if gym_module and hasattr(gym_module, 'make'):
                    # This appears to be a legacy gym import
                    return True
                    
            # Check for gym.make calls in local variables or function calls
            if any('gym.make' in str(v) for v in frame_locals.values() if isinstance(v, str)):
                return True
                
    except Exception:
        # If inspection fails, err on the side of caution
        pass
    finally:
        del frame  # Prevent reference cycles
        
    return False


def _issue_legacy_gym_warning() -> None:
    """
    Issue structured deprecation warning for legacy gym usage.
    
    Uses Loguru logger.warning for structured guidance as specified in
    Section 5.4.3.2 Legacy API Deprecation Handling. Includes specific
    migration instructions and compatibility layer information.
    """
    warning_message = (
        "Legacy 'gym' import detected. Please migrate to 'gymnasium' for modern RL support. "
        "The legacy gym package is deprecated and may not receive future updates."
    )
    
    migration_guidance = {
        "legacy_pattern": "import gym; env = gym.make('OdorPlumeNavigation-v1')",
        "modern_pattern": "import gymnasium; env = gymnasium.make('PlumeNavSim-v0')",
        "migration_steps": [
            "1. Replace 'import gym' with 'import gymnasium'",
            "2. Use new environment ID 'PlumeNavSim-v0' instead of legacy IDs",
            "3. Update step() return handling for 5-tuple format",
            "4. Use seed parameter in reset() method for reproducibility"
        ],
        "compatibility_notes": [
            "Legacy environments continue to work with 4-tuple step() returns",
            "New Gymnasium environments provide enhanced API compliance",
            "Cross-compatibility maintained through automatic detection"
        ],
        "documentation_links": [
            "Migration guide: See project documentation for detailed migration examples",
            "Gymnasium documentation: https://gymnasium.farama.org/",
            "API reference: Check odor_plume_nav.api module documentation"
        ]
    }
    
    if LOGURU_AVAILABLE:
        # Use structured Loguru logging as per specification
        logger.warning(
            warning_message,
            extra={
                "warning_type": "legacy_api_deprecation",
                "component": "gym_import_detection",
                "migration_guidance": migration_guidance,
                "action_required": True,
                "breaking_change_timeline": "Next major version",
                "current_compatibility": "Maintained with warnings"
            }
        )
    else:
        # Fallback to standard warnings if Loguru not available
        warnings.warn(
            f"{warning_message}\nMigration guidance: {migration_guidance}",
            DeprecationWarning,
            stacklevel=3
        )


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
    visualize_trajectory,
    SimulationVisualization,
)

# Import core protocols for type hints and advanced usage
from ..core.protocols import NavigatorProtocol
from ..environments.video_plume import VideoPlume

# Import Gymnasium RL integration functions with enhanced error handling
try:
    from .navigation import create_gymnasium_environment, from_legacy
    GYMNASIUM_AVAILABLE = True
    
    # Detect legacy gym usage and issue warnings
    if _detect_legacy_gym_import():
        _issue_legacy_gym_warning()
        
except ImportError as e:
    # Enhanced error handling for new Gymnasium functions
    GYMNASIUM_AVAILABLE = False
    
    def create_gymnasium_environment(*args, **kwargs):
        """
        Enhanced error handler for create_gymnasium_environment.
        
        Provides informative error messages when the Gymnasium compatibility 
        layer is not yet available, as per the key changes requirements.
        """
        error_details = {
            "function": "create_gymnasium_environment",
            "requirement": "Gymnasium environment support",
            "install_command": "pip install 'odor_plume_nav[rl]'",
            "alternative_approach": "Use legacy simulation API or wait for RL integration",
            "documentation": "See project documentation for installation requirements",
            "import_error": str(e) if e else "Module not available"
        }
        
        if LOGURU_AVAILABLE:
            logger.error(
                "create_gymnasium_environment not available - ensure Gymnasium environment is created",
                extra=error_details
            )
        
        raise ImportError(
            "create_gymnasium_environment not available. "
            f"Install RL dependencies with: {error_details['install_command']} "
            "or use the legacy simulation API for non-RL workflows. "
            f"Original error: {error_details['import_error']}"
        )
    
    def from_legacy(*args, **kwargs):
        """
        Enhanced error handler for from_legacy migration function.
        
        Provides informative error messages when the legacy migration 
        compatibility layer is not yet available.
        """
        error_details = {
            "function": "from_legacy", 
            "requirement": "Legacy migration support",
            "install_command": "pip install 'odor_plume_nav[rl]'",
            "alternative_approach": "Use direct Gymnasium environment creation",
            "documentation": "See migration guide in project documentation",
            "import_error": str(e) if e else "Module not available"
        }
        
        if LOGURU_AVAILABLE:
            logger.error(
                "from_legacy not available - ensure legacy migration function is created",
                extra=error_details
            )
        
        raise ImportError(
            "from_legacy migration function not available. "
            f"Install RL dependencies with: {error_details['install_command']} "
            "or use create_gymnasium_environment directly with equivalent parameters. "
            f"Original error: {error_details['import_error']}"
        )


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
    # Both animation and static visualization use visualize_trajectory for now
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
__author__ = "Odor Plume Navigation Development Team"
__description__ = "Public API for odor plume navigation library with unified package structure and Hydra configuration support"

# Export all public functions and classes
__all__ = [
    # Primary factory methods
    "create_navigator",
    "create_video_plume",
    "run_plume_simulation",
    
    # Gymnasium RL integration functions
    "create_gymnasium_environment",
    "from_legacy",
    
    # Enhanced API aliases
    "create_navigator_instance", 
    "create_video_plume_instance",
    "run_navigation_simulation",
    "visualize_results",
    
    # Visualization functions
    "visualize_trajectory", 
    "SimulationVisualization",
    
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

# Conditional exports based on Hydra and Gymnasium availability
if HYDRA_AVAILABLE:
    # Add Hydra-specific functionality to exports
    __all__.extend([
        "DictConfig",  # Re-export for type hints
    ])

if GYMNASIUM_AVAILABLE:
    # Add Gymnasium-specific functionality when available
    __all__.extend([
        # Additional Gymnasium-specific exports can be added here
        # when new functionality becomes available
    ])

# Package initialization message (optional, for development/debugging)
def _get_api_info() -> Dict[str, Any]:
    """Get API module information for debugging and introspection."""
    return {
        "version": __version__,
        "hydra_available": HYDRA_AVAILABLE,
        "gymnasium_available": GYMNASIUM_AVAILABLE,
        "loguru_available": LOGURU_AVAILABLE,
        "public_functions": len(__all__),
        "primary_functions": [
            "create_navigator", 
            "create_video_plume", 
            "run_plume_simulation",
            "visualize_trajectory"
        ],
        "rl_functions": [
            "create_gymnasium_environment",
            "from_legacy"
        ] if GYMNASIUM_AVAILABLE else [],
        "legacy_support": True,
        "deprecation_warnings": True,
        "configuration_types": ["direct_parameters", "hydra_dictconfig"] + (
            ["yaml_files"] if HYDRA_AVAILABLE else []
        ),
        "compatibility_features": [
            "legacy_gym_detection",
            "structured_warnings",
            "enhanced_error_handling",
            "migration_guidance"
        ]
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
    >>> from odor_plume_nav.api import get_api_info
    >>> info = get_api_info()
    >>> print(f"API version: {info['version']}")
    >>> print(f"Hydra support: {info['hydra_available']}")
    >>> print(f"Gymnasium support: {info['gymnasium_available']}")
    """
    return _get_api_info()