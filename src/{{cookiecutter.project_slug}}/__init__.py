"""
{{cookiecutter.project_name}} - Odor Plume Navigation Library

A modular, research-oriented Python library for simulating agent navigation through odor plumes.
Supports both single-agent and multi-agent scenarios with sophisticated video-based environments,
real-time visualization, and comprehensive configuration management through Hydra.

This refactored package transforms the original standalone application into an importable library
that integrates seamlessly with Kedro pipelines, reinforcement learning frameworks, and 
machine learning analysis tools.

Key Features:
    - Protocol-based navigation interfaces for extensibility
    - Video-based odor plume environments with OpenCV processing
    - Hydra-based hierarchical configuration management
    - Real-time visualization and trajectory recording
    - Reproducible experiment setup with seed management
    - CLI interface for batch processing and automation
    - Database session management for future extensibility

Example Usage:
    # Kedro integration
    from {{cookiecutter.project_slug}} import Navigator, VideoPlume
    from {{cookiecutter.project_slug}}.config import NavigatorConfig
    
    # RL framework integration
    from {{cookiecutter.project_slug}}.core import NavigatorProtocol
    from {{cookiecutter.project_slug}}.api import create_navigator
    
    # ML/neural network analysis
    from {{cookiecutter.project_slug}}.utils import set_global_seed
    from {{cookiecutter.project_slug}}.data import VideoPlume
    
    # Complete simulation pipeline
    navigator = create_navigator(position=(50, 50), max_speed=10.0)
    plume = VideoPlume.from_config(video_path="odor_plume.mp4")
    results = run_plume_simulation(navigator, plume, num_steps=1000)
    visualize_simulation_results(results)
"""

# Import version information
try:
    # Try to import from package metadata (installed package)
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("{{cookiecutter.project_slug}}")
    except PackageNotFoundError:
        # Development installation or package not found
        __version__ = "0.1.0-dev"
except ImportError:
    # Python < 3.8 fallback
    try:
        from pkg_resources import get_distribution, DistributionNotFound
        try:
            __version__ = get_distribution("{{cookiecutter.project_slug}}").version
        except DistributionNotFound:
            __version__ = "0.1.0-dev"
    except ImportError:
        __version__ = "0.1.0-dev"

# Core API exports for unified import access
# These provide the primary interfaces for library consumers
try:
    # Primary navigation API
    from {{cookiecutter.project_slug}}.api.navigation import (
        create_navigator,
        create_video_plume,
        run_plume_simulation,
        visualize_simulation_results,
    )
    
    # Core navigation components
    from {{cookiecutter.project_slug}}.core.navigator import (
        Navigator,
        NavigatorProtocol,
    )
    
    # Navigation controllers
    from {{cookiecutter.project_slug}}.core.controllers import (
        SingleAgentController,
        MultiAgentController,
    )
    
    # Video plume environment
    from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
    
    # Configuration schemas
    from {{cookiecutter.project_slug}}.config.schemas import (
        NavigatorConfig,
        SingleAgentConfig,
        MultiAgentConfig,
        VideoPlumeConfig,
    )
    
    # Essential utilities
    from {{cookiecutter.project_slug}}.utils.seed_manager import set_global_seed
    from {{cookiecutter.project_slug}}.utils.visualization import visualize_trajectory
    
    # Simulation runner for backward compatibility
    run_simulation = run_plume_simulation
    
except ImportError as e:
    # Graceful degradation for development environments
    # where some modules may not be available yet
    import warnings
    warnings.warn(
        f"Some {{cookiecutter.project_slug}} components could not be imported: {e}. "
        "This may indicate a development environment setup issue or missing dependencies.",
        ImportWarning,
        stacklevel=2
    )
    
    # Provide minimal fallback interface
    __version__ = "0.1.0-dev"
    
    # Define empty stubs to prevent import errors
    def create_navigator(*args, **kwargs):
        raise NotImplementedError("Navigator components not available. Check installation.")
    
    def create_video_plume(*args, **kwargs):
        raise NotImplementedError("VideoPlume components not available. Check installation.")
    
    def run_plume_simulation(*args, **kwargs):
        raise NotImplementedError("Simulation components not available. Check installation.")
    
    def visualize_simulation_results(*args, **kwargs):
        raise NotImplementedError("Visualization components not available. Check installation.")
    
    def set_global_seed(*args, **kwargs):
        raise NotImplementedError("Seed management components not available. Check installation.")
    
    # Provide basic protocol stubs
    class NavigatorProtocol:
        pass
    
    class Navigator:
        pass
    
    class VideoPlume:
        pass
    
    # Configuration stubs
    class NavigatorConfig:
        pass
    
    class VideoPlumeConfig:
        pass

# Public API definition
# This defines what is available when using "from {{cookiecutter.project_slug}} import *"
__all__ = [
    # Version information
    "__version__",
    
    # Primary API functions
    "create_navigator",
    "create_video_plume", 
    "run_plume_simulation",
    "run_simulation",  # Backward compatibility alias
    "visualize_simulation_results",
    
    # Core components
    "Navigator",
    "NavigatorProtocol",
    "SingleAgentController",
    "MultiAgentController",
    "VideoPlume",
    
    # Configuration schemas
    "NavigatorConfig",
    "SingleAgentConfig", 
    "MultiAgentConfig",
    "VideoPlumeConfig",
    
    # Essential utilities
    "set_global_seed",
    "visualize_trajectory",
]

# Package metadata
__author__ = "{{cookiecutter.author_name}}"
__email__ = "{{cookiecutter.author_email}}"
__description__ = "A modular Python library for odor plume navigation simulation"
__url__ = "{{cookiecutter.repository_url}}"
__license__ = "{{cookiecutter.license}}"

# Development and debugging information
__package_name__ = "{{cookiecutter.project_slug}}"
__package_title__ = "{{cookiecutter.project_name}}"

# Configuration for library behavior
# These can be modified by users to customize behavior
import logging

# Set up default logging configuration
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Library-wide configuration
_LIBRARY_CONFIG = {
    "strict_mode": False,  # Whether to enforce strict validation
    "debug_mode": False,   # Enable debug logging and checks
    "numpy_random_seed": None,  # Default random seed for reproducibility
}

def configure_library(strict_mode=None, debug_mode=None, numpy_random_seed=None):
    """Configure library-wide behavior settings.
    
    Parameters
    ----------
    strict_mode : bool, optional
        If True, enables strict validation and error checking.
        If False, allows more permissive behavior for development.
    debug_mode : bool, optional
        If True, enables debug logging and additional runtime checks.
    numpy_random_seed : int, optional
        Sets the global NumPy random seed for reproducible results.
        
    Example
    -------
    >>> import {{cookiecutter.project_slug}}
    >>> {{cookiecutter.project_slug}}.configure_library(strict_mode=True, debug_mode=True)
    >>> {{cookiecutter.project_slug}}.set_global_seed(42)  # For reproducible experiments
    """
    global _LIBRARY_CONFIG
    
    if strict_mode is not None:
        _LIBRARY_CONFIG["strict_mode"] = strict_mode
    
    if debug_mode is not None:
        _LIBRARY_CONFIG["debug_mode"] = debug_mode
        
        # Configure logging level based on debug mode
        logger = logging.getLogger(__name__)
        if debug_mode:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
    
    if numpy_random_seed is not None:
        _LIBRARY_CONFIG["numpy_random_seed"] = numpy_random_seed
        try:
            # Try to set the seed immediately if possible
            if "set_global_seed" in globals():
                set_global_seed(numpy_random_seed)
        except Exception:
            # Seed will be set when set_global_seed is imported
            pass

def get_library_config():
    """Get current library configuration settings.
    
    Returns
    -------
    dict
        Dictionary containing current library configuration.
    """
    return _LIBRARY_CONFIG.copy()

# Add configuration functions to public API
__all__.extend(["configure_library", "get_library_config"])