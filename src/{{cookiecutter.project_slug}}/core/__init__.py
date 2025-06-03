"""
Core navigation module providing unified API access to navigation components.

This module serves as the primary entry point for the odor plume navigation system's
core functionality, exposing NavigatorProtocol interfaces, controller implementations,
factory methods, and simulation orchestration under a unified namespace. It supports
the refactored template-based package structure while maintaining backward compatibility
with existing NavigatorProtocol interfaces.

The module integrates seamlessly with Hydra-based configuration management for enhanced
ML framework compatibility, supporting Kedro pipeline integration, reinforcement learning
frameworks, and machine learning analysis workflows through standardized interfaces.

Key Features:
    - Protocol-based interface design ensuring algorithm extensibility
    - Hydra configuration integration for hierarchical parameter management
    - Factory pattern implementation supporting both single and multi-agent scenarios
    - Enhanced error handling and performance monitoring for research workflows
    - Backward compatibility with existing NavigatorProtocol implementations
    - Optimized performance for 100+ agents at 30fps simulation throughput

Import Patterns:
    Kedro-based projects:
        >>> from {{cookiecutter.project_slug}}.core import Navigator, NavigatorProtocol
        >>> from {{cookiecutter.project_slug}}.core import run_simulation
        
    RL framework integration:
        >>> from {{cookiecutter.project_slug}}.core import NavigatorProtocol
        >>> from {{cookiecutter.project_slug}}.core import SingleAgentController, MultiAgentController
        
    ML/neural network analyses:
        >>> from {{cookiecutter.project_slug}}.core import Navigator, run_simulation
        >>> from {{cookiecutter.project_slug}}.core import create_controller_from_config

    Direct component access:
        >>> from {{cookiecutter.project_slug}}.core import (
        ...     NavigatorProtocol,
        ...     SingleAgentController,
        ...     MultiAgentController,
        ...     NavigatorFactory,
        ...     run_simulation
        ... )

Examples:
    Create navigator with factory method:
        >>> from {{cookiecutter.project_slug}}.core import NavigatorFactory
        >>> navigator = NavigatorFactory.single_agent(position=(10.0, 20.0), max_speed=2.0)
        
    Configuration-driven instantiation:
        >>> from hydra import compose, initialize
        >>> from {{cookiecutter.project_slug}}.core import NavigatorFactory, run_simulation
        >>> 
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        ...     navigator = NavigatorFactory.from_config(cfg.navigator)
        ...     # Simulation execution requires video_plume from data module
        
    Multi-agent scenario:
        >>> from {{cookiecutter.project_slug}}.core import MultiAgentController
        >>> positions = [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]
        >>> controller = MultiAgentController(positions=positions)
        >>> # controller implements NavigatorProtocol for consistent interface

Notes:
    The core module maintains strict adherence to the NavigatorProtocol interface while
    providing enhanced functionality through Hydra configuration integration. All components
    support both direct parameter initialization and configuration-driven instantiation
    for maximum flexibility across research workflows.
    
    Performance requirements are maintained with <33ms frame processing latency and
    support for 100+ agents at 30fps simulation throughput through vectorized operations
    and optimized memory management patterns.
"""

# Import core protocol and type definitions
from .navigator import (
    NavigatorProtocol,
    NavigatorFactory,
    PositionType,
    PositionsType,
    OrientationType,
    OrientationsType,
    SpeedType,
    SpeedsType,
    ConfigType,
)

# Import controller implementations
from .controllers import (
    SingleAgentController,
    MultiAgentController,
    SingleAgentParams,
    MultiAgentParams,
    create_controller_from_config,
)

# Import simulation orchestration from API module
from ..api.navigation import run_plume_simulation as run_simulation

# Create Navigator alias for backward compatibility and simplified imports
# This enables: from {{cookiecutter.project_slug}}.core import Navigator
Navigator = NavigatorFactory

# Define public API with comprehensive exports for maximum flexibility
__all__ = [
    # Core protocol and factory interfaces
    "NavigatorProtocol",
    "NavigatorFactory", 
    "Navigator",  # Alias for NavigatorFactory
    
    # Controller implementations
    "SingleAgentController",
    "MultiAgentController",
    
    # Configuration and utility classes
    "SingleAgentParams",
    "MultiAgentParams",
    "create_controller_from_config",
    
    # Simulation orchestration
    "run_simulation",
    
    # Type aliases for enhanced IDE support
    "PositionType",
    "PositionsType", 
    "OrientationType",
    "OrientationsType",
    "SpeedType",
    "SpeedsType",
    "ConfigType",
]

# Module metadata for documentation and introspection
__version__ = "1.0.0"
__author__ = "Odor Plume Navigation Team"
__description__ = "Core navigation module with Hydra-integrated configuration support"

# Performance and compatibility metrics for monitoring
__performance_requirements__ = {
    "frame_processing_latency_ms": 33,
    "max_agents_supported": 100,
    "target_fps": 30,
    "memory_efficiency_agents": 100,  # <10MB overhead per 100 agents
    "initialization_time_ms": 2000,  # <2s for complex configurations
}

__compatibility_features__ = {
    "hydra_configuration": True,
    "kedro_integration": True,
    "rl_framework_support": True,
    "ml_analysis_compatibility": True,
    "backward_compatibility": True,
    "protocol_based_extensibility": True,
}

# Validation function to ensure all required components are available
def _validate_core_module_integrity():
    """
    Validate that all core components are properly imported and functional.
    
    This function performs basic integrity checks to ensure the module is
    properly initialized and all required components are accessible.
    
    Raises:
        ImportError: If critical components are missing
        AttributeError: If components don't have required interfaces
    """
    # Validate protocol implementation
    if not hasattr(NavigatorProtocol, 'positions'):
        raise AttributeError("NavigatorProtocol missing required 'positions' property")
    
    # Validate controller implementations
    required_controller_methods = ['step', 'reset', 'sample_odor']
    for controller_class in [SingleAgentController, MultiAgentController]:
        for method_name in required_controller_methods:
            if not hasattr(controller_class, method_name):
                raise AttributeError(
                    f"{controller_class.__name__} missing required method '{method_name}'"
                )
    
    # Validate factory functionality
    if not hasattr(NavigatorFactory, 'from_config'):
        raise AttributeError("NavigatorFactory missing required 'from_config' method")
    
    # Validate simulation function availability
    if not callable(run_simulation):
        raise ImportError("run_simulation function not properly imported")

# Perform module integrity validation on import
try:
    _validate_core_module_integrity()
except Exception as e:
    # Log validation failure but don't prevent import
    # This allows for graceful degradation in testing environments
    import warnings
    warnings.warn(
        f"Core module integrity validation failed: {e}. "
        f"Some functionality may be unavailable.",
        ImportWarning,
        stacklevel=2
    )

# Configure module-level logging context for debugging
try:
    from loguru import logger
    
    # Bind core module context for enhanced logging
    _module_logger = logger.bind(
        module="core",
        version=__version__,
        performance_requirements=__performance_requirements__,
        compatibility_features=__compatibility_features__
    )
    
    _module_logger.debug(
        "Core navigation module initialized successfully",
        exported_components=len(__all__),
        protocol_available=hasattr(NavigatorProtocol, '__protocol_attrs__'),
        factory_methods=len([attr for attr in dir(NavigatorFactory) if not attr.startswith('_')])
    )
    
except ImportError:
    # Loguru not available - continue without enhanced logging
    pass

# Documentation strings for IDE autocompletion and help systems
_module_docs = {
    "NavigatorProtocol": "Core protocol defining navigation interface contract",
    "NavigatorFactory": "Factory class for creating navigator instances with configuration support", 
    "Navigator": "Alias for NavigatorFactory providing simplified access",
    "SingleAgentController": "Implementation for single-agent navigation scenarios",
    "MultiAgentController": "Implementation for multi-agent swarm navigation scenarios",
    "run_simulation": "Execute complete plume navigation simulation with data collection",
    "create_controller_from_config": "Create navigator from configuration with automatic type detection",
}

def get_component_documentation(component_name: str) -> str:
    """
    Get documentation string for a specific component.
    
    Parameters
    ----------
    component_name : str
        Name of the component to get documentation for
        
    Returns
    -------
    str
        Documentation string or empty string if component not found
    """
    return _module_docs.get(component_name, "")

def list_available_components() -> dict:
    """
    List all available components in the core module.
    
    Returns
    -------
    dict
        Dictionary mapping component names to their documentation strings
    """
    return _module_docs.copy()

# Support for dynamic component discovery in research environments
def get_navigator_implementations() -> list:
    """
    Get list of available NavigatorProtocol implementations.
    
    Returns
    -------
    list
        List of class objects implementing NavigatorProtocol
    """
    return [SingleAgentController, MultiAgentController]

def get_factory_methods() -> list:
    """
    Get list of available factory methods for navigator creation.
    
    Returns
    -------
    list
        List of method names available on NavigatorFactory
    """
    return [
        method for method in dir(NavigatorFactory) 
        if not method.startswith('_') and callable(getattr(NavigatorFactory, method))
    ]

# Integration support functions for external frameworks
def create_kedro_compatible_navigator(config_dict: dict):
    """
    Create navigator instance optimized for Kedro pipeline integration.
    
    This convenience function provides explicit Kedro integration support
    by handling configuration dictionaries in Kedro's preferred format.
    
    Parameters
    ----------
    config_dict : dict
        Configuration dictionary in Kedro catalog format
        
    Returns
    -------
    NavigatorProtocol
        Configured navigator instance
    """
    return NavigatorFactory.from_config(config_dict)

def create_rl_compatible_navigator(position=None, max_speed=1.0, **kwargs):
    """
    Create navigator instance optimized for reinforcement learning frameworks.
    
    This convenience function provides simplified parameter handling for
    common RL use cases with sensible defaults.
    
    Parameters
    ----------
    position : tuple or None, optional
        Initial position, by default None (becomes (0, 0))
    max_speed : float, optional
        Maximum agent speed, by default 1.0
    **kwargs
        Additional navigator parameters
        
    Returns
    -------
    NavigatorProtocol
        Configured navigator instance suitable for RL environments
    """
    if position is None:
        position = (0.0, 0.0)
    
    return NavigatorFactory.single_agent(
        position=position,
        max_speed=max_speed,
        **kwargs
    )