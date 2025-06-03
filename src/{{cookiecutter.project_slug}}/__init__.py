"""
{{cookiecutter.project_name}} - Odor Plume Navigation Library

A refactored, production-ready library for simulating odor plume navigation with 
support for both single-agent and multi-agent scenarios. This library provides
clean API interfaces for Kedro pipeline integration, reinforcement learning 
frameworks, and machine learning analysis tools.

The library transforms the original standalone application into an importable
Python package with enhanced Hydra-based configuration management, improved
modularity, and standardized project template structure.

Key Features:
- Protocol-based navigator interface for extensibility  
- Factory methods with Hydra configuration support
- Comprehensive error handling and validation
- Production-ready logging and monitoring
- Deterministic behavior through seed management
- Support for both single and multi-agent navigation

Example Usage:
    # Basic usage for Kedro projects
    from {{cookiecutter.project_slug}} import Navigator, VideoPlume
    navigator = Navigator.single(position=(10.0, 20.0), max_speed=5.0)
    plume = VideoPlume("path/to/video.mp4")
    
    # Advanced usage with Hydra configuration
    from {{cookiecutter.project_slug}} import create_navigator, create_video_plume
    from {{cookiecutter.project_slug}}.config import NavigatorConfig
    navigator = create_navigator(cfg=hydra_config.navigator)
    plume = create_video_plume(cfg=hydra_config.video_plume)
    
    # RL framework integration
    from {{cookiecutter.project_slug}}.core import NavigatorProtocol
    from {{cookiecutter.project_slug}}.api import create_navigator
    navigator = create_navigator(position=(0, 0), max_speed=10.0)
    
    # ML analysis with seed management
    from {{cookiecutter.project_slug}}.utils import set_global_seed
    set_global_seed(42)  # Ensures reproducible experiments
"""

import importlib.metadata
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import warnings

try:
    __version__ = importlib.metadata.version("{{cookiecutter.project_slug}}")
except importlib.metadata.PackageNotFoundError:
    # Fallback for development installations
    __version__ = "0.1.0-dev"

# Package metadata
__author__ = "{{cookiecutter.author_name}}"
__email__ = "{{cookiecutter.author_email}}"
__description__ = "Refactored Odor Plume Navigation library with Hydra configuration support"
__license__ = "{{cookiecutter.license}}"

# Core API exports for simplified import patterns
try:
    # Main factory functions - Primary API for most users
    from .api.navigation import (
        create_navigator,
        create_video_plume, 
        run_plume_simulation,
        # Factory methods for backward compatibility
        create_navigator_from_config,
        create_video_plume_from_config,
        # Exception classes
        ConfigurationError,
        SimulationError,
    )
    
    # Core navigation components - For advanced users and type hints
    from .core.navigator import NavigatorProtocol
    from .core.controllers import SingleAgentController, MultiAgentController
    
    # Data processing components
    from .data.video_plume import VideoPlume
    
    # Configuration schemas - For type-safe configuration
    from .config.schemas import (
        NavigatorConfig,
        SingleAgentConfig, 
        MultiAgentConfig,
        VideoPlumeConfig,
        SimulationConfig,
    )
    
    # Utility functions - For reproducibility and advanced features  
    from .utils.seed_manager import set_global_seed, get_current_seed
    
except ImportError as e:
    # Graceful degradation for partial installations or circular imports
    warnings.warn(
        f"Some {{cookiecutter.project_slug}} components could not be imported: {e}. "
        "This may indicate a missing dependency or installation issue.",
        ImportWarning,
        stacklevel=2
    )
    
    # Provide empty fallbacks to prevent complete failure
    create_navigator = None
    create_video_plume = None
    run_plume_simulation = None
    NavigatorProtocol = None
    VideoPlume = None
    NavigatorConfig = None


# Navigator factory class for backward compatibility and convenience
# This provides the Navigator.single() and Navigator.multi() patterns mentioned in the spec
class Navigator:
    """
    Factory class for creating navigator instances with simplified interface.
    
    This class provides convenient static methods for creating single-agent and
    multi-agent navigators, supporting both direct parameter specification and
    Hydra configuration objects. It serves as the primary entry point for most
    navigation use cases.
    
    The factory methods delegate to the underlying create_navigator function
    while providing a more object-oriented interface that matches common usage
    patterns in machine learning and robotics frameworks.
    
    Examples:
        # Single-agent navigation
        navigator = Navigator.single(position=(10.0, 20.0), max_speed=5.0)
        
        # Multi-agent navigation
        positions = [(0, 0), (10, 10), (20, 20)]
        navigator = Navigator.multi(positions=positions, max_speeds=[5, 6, 7])
        
        # From Hydra configuration
        navigator = Navigator.from_config(hydra_config.navigator)
    """
    
    @staticmethod
    def single(
        position: Union[Tuple[float, float], List[float]] = (0.0, 0.0),
        orientation: float = 0.0,
        speed: float = 0.0,
        max_speed: float = 1.0,
        angular_velocity: float = 0.0,
        **kwargs: Any
    ) -> 'NavigatorProtocol':
        """
        Create a single-agent navigator with specified parameters.
        
        Args:
            position: Initial position as (x, y) coordinates
            orientation: Initial orientation in degrees
            speed: Initial speed
            max_speed: Maximum allowed speed
            angular_velocity: Angular velocity in degrees per second
            **kwargs: Additional parameters passed to create_navigator
            
        Returns:
            Navigator instance implementing NavigatorProtocol
            
        Raises:
            ConfigurationError: If parameters are invalid
            
        Examples:
            # Basic single agent
            nav = Navigator.single(position=(10, 20), max_speed=5.0)
            
            # With custom parameters
            nav = Navigator.single(
                position=(0, 0), 
                orientation=45.0, 
                max_speed=10.0,
                angular_velocity=2.0
            )
        """
        if create_navigator is None:
            raise ImportError("create_navigator function not available. Check installation.")
        
        return create_navigator(
            position=position,
            orientation=orientation,
            speed=speed,
            max_speed=max_speed,
            angular_velocity=angular_velocity,
            **kwargs
        )
    
    @staticmethod
    def multi(
        positions: Union[List[Tuple[float, float]], List[List[float]]] = None,
        orientations: Optional[Union[List[float], tuple]] = None,
        speeds: Optional[Union[List[float], tuple]] = None,
        max_speeds: Optional[Union[List[float], tuple]] = None,
        angular_velocities: Optional[Union[List[float], tuple]] = None,
        **kwargs: Any
    ) -> 'NavigatorProtocol':
        """
        Create a multi-agent navigator with specified parameters.
        
        Args:
            positions: List of (x, y) position tuples for each agent
            orientations: List of initial orientations in degrees
            speeds: List of initial speeds for each agent
            max_speeds: List of maximum speeds for each agent
            angular_velocities: List of angular velocities in degrees per second
            **kwargs: Additional parameters passed to create_navigator
            
        Returns:
            Navigator instance implementing NavigatorProtocol
            
        Raises:
            ConfigurationError: If parameters are invalid or inconsistent
            
        Examples:
            # Basic multi-agent with positions only
            positions = [(0, 0), (10, 10), (20, 20)]
            nav = Navigator.multi(positions=positions)
            
            # With full parameter specification
            nav = Navigator.multi(
                positions=[(0, 0), (10, 10)],
                orientations=[0.0, 90.0],
                max_speeds=[5.0, 7.0],
                angular_velocities=[1.0, 1.5]
            )
        """
        if create_navigator is None:
            raise ImportError("create_navigator function not available. Check installation.")
        
        # Set default positions if not provided
        if positions is None:
            positions = [(0.0, 0.0), (10.0, 10.0)]
        
        return create_navigator(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds,
            angular_velocities=angular_velocities,
            **kwargs
        )
    
    @staticmethod
    def from_config(cfg: Union['DictConfig', Dict[str, Any]]) -> 'NavigatorProtocol':
        """
        Create a navigator from Hydra configuration object.
        
        This method automatically detects whether to create a single-agent or
        multi-agent navigator based on the configuration structure, providing
        seamless integration with Hydra-based workflows.
        
        Args:
            cfg: Hydra DictConfig or dictionary containing navigator configuration
            
        Returns:
            Navigator instance implementing NavigatorProtocol
            
        Raises:
            ConfigurationError: If configuration is invalid
            
        Examples:
            # From Hydra configuration
            navigator = Navigator.from_config(hydra_config.navigator)
            
            # From dictionary configuration
            config = {
                'position': [10.0, 20.0],
                'max_speed': 5.0,
                'orientation': 45.0
            }
            navigator = Navigator.from_config(config)
        """
        if create_navigator is None:
            raise ImportError("create_navigator function not available. Check installation.")
        
        return create_navigator(cfg=cfg)


# Version information and metadata access
def get_version() -> str:
    """Get the current version of the library."""
    return __version__

def get_package_info() -> Dict[str, str]:
    """Get comprehensive package information."""
    return {
        "name": "{{cookiecutter.project_slug}}",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": __description__,
        "license": __license__,
    }

# Check for optional dependencies and provide informative warnings
def _check_optional_dependencies() -> Dict[str, bool]:
    """Check availability of optional dependencies."""
    dependencies = {}
    
    try:
        import hydra
        dependencies['hydra'] = True
    except ImportError:
        dependencies['hydra'] = False
        warnings.warn(
            "Hydra not available. Advanced configuration features will be limited.",
            ImportWarning,
            stacklevel=2
        )
    
    try:
        import cv2
        dependencies['opencv'] = True
    except ImportError:
        dependencies['opencv'] = False
        warnings.warn(
            "OpenCV not available. Video plume processing will be limited.",
            ImportWarning,
            stacklevel=2
        )
    
    try:
        import matplotlib
        dependencies['matplotlib'] = True
    except ImportError:
        dependencies['matplotlib'] = False
        warnings.warn(
            "Matplotlib not available. Visualization features will be limited.",
            ImportWarning,
            stacklevel=2
        )
    
    return dependencies

# Store dependency information for runtime checking
_optional_dependencies = _check_optional_dependencies()

def check_dependencies() -> Dict[str, bool]:
    """
    Check the availability of optional dependencies.
    
    Returns:
        Dictionary mapping dependency names to availability status
    """
    return _optional_dependencies.copy()

# Public API for library consumers
# This supports the import patterns specified in the requirements:
# - from {{cookiecutter.project_slug}} import Navigator, VideoPlume
# - from {{cookiecutter.project_slug}}.config import NavigatorConfig  
# - from {{cookiecutter.project_slug}}.core import NavigatorProtocol
# - from {{cookiecutter.project_slug}}.api import create_navigator
# - from {{cookiecutter.project_slug}}.utils import set_global_seed

__all__ = [
    # Version and metadata
    "__version__",
    "get_version", 
    "get_package_info",
    
    # Primary API - Main entry points for most users
    "Navigator",           # Factory class with .single(), .multi(), .from_config()
    "VideoPlume",         # Video-based environment class
    "create_navigator",   # Direct factory function
    "create_video_plume", # Direct factory function  
    "run_plume_simulation", # Simulation execution
    
    # Configuration support for type-safe usage
    "NavigatorConfig",
    "SingleAgentConfig",
    "MultiAgentConfig", 
    "VideoPlumeConfig",
    "SimulationConfig",
    
    # Advanced API - For framework integration and extension
    "NavigatorProtocol",  # Protocol for custom implementations
    "SingleAgentController", # Concrete single-agent implementation
    "MultiAgentController",  # Concrete multi-agent implementation
    
    # Factory methods for backward compatibility
    "create_navigator_from_config",
    "create_video_plume_from_config",
    
    # Utility functions for reproducibility
    "set_global_seed",
    "get_current_seed",
    
    # Exception classes for error handling
    "ConfigurationError",
    "SimulationError",
    
    # Diagnostic and utility functions
    "check_dependencies",
]

# Compatibility layer for common import patterns
# This ensures that legacy code continues to work
try:
    # Support legacy import pattern: from package import run_simulation
    from .api.navigation import run_plume_simulation as run_simulation
    __all__.append("run_simulation")
except ImportError:
    pass

# Initialize package-level configuration
def _initialize_package():
    """Initialize package-level configuration and logging."""
    try:
        # Configure default logging behavior for the package
        from .utils.logging import setup_package_logging
        setup_package_logging()
    except ImportError:
        # Graceful fallback if logging utilities are not available
        import logging
        logging.getLogger(__name__).addHandler(logging.NullHandler())

# Run package initialization
_initialize_package()