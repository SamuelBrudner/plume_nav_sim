"""
Odor Plume Navigation Package

A package for simulating navigation of odor plumes.

This package provides tools for simulating how agents navigate through odor plumes,
with support for both single and multi-agent simulations, enhanced configuration
management, database persistence, CLI tools, and comprehensive testing utilities.
"""

__version__ = "0.2.0"

# API exports for easy access
from odor_plume_nav.api import (
    create_navigator,
    create_video_plume,
    run_plume_simulation,
    visualize_simulation_results,
)

# Enhanced API factory functions from merged application-layer components
try:
    from odor_plume_nav.api import (
        create_simulation_runner,
        create_batch_processor,
        run_experiment_sweep,
    )
except ImportError:
    # Graceful degradation if enhanced API functions not available
    pass

# Import the protocol-based navigator implementation
from odor_plume_nav.core import (
    Navigator,
    SingleAgentController,
    MultiAgentController,
    NavigatorProtocol,
    run_simulation
)
from odor_plume_nav.environments import VideoPlume
from odor_plume_nav.visualization import visualize_trajectory
from odor_plume_nav.config import (
    NavigatorConfig, 
    SingleAgentConfig,
    MultiAgentConfig, 
    VideoPlumeConfig,
    load_config,
    save_config,
)
from odor_plume_nav.utils import (
    # IO utilities
    load_yaml,
    save_yaml,
    load_json, 
    save_json,
    load_numpy,
    save_numpy,
    
    # Logging utilities
    setup_logger,
    get_module_logger,
    DEFAULT_FORMAT,
    MODULE_FORMAT,
    LOG_LEVELS,
)

# Enhanced data processing capabilities from merged odor_plume_nav.data submodule
try:
    from odor_plume_nav.data import (
        VideoPlume,
        create_synthetic_plume,
        preprocess_video_data,
        validate_plume_data,
    )
except ImportError:
    # Graceful degradation if enhanced data processing not available
    pass

# CLI functionality from merged odor_plume_nav.cli submodule
try:
    from odor_plume_nav.cli import (
        cli,
        run_command,
        config_command,
        visualize_command,
        batch_command,
    )
except ImportError:
    # CLI functionality not available - use with caution in non-CLI environments
    cli = None

# Database persistence capabilities from merged odor_plume_nav.db submodule
try:
    from odor_plume_nav.db import (
        SessionManager,
        create_session,
        initialize_database,
        store_trajectory,
        store_experiment_metadata,
    )
except ImportError:
    # Database functionality not available - file-based persistence only
    SessionManager = None

# Enhanced testing utilities from merged odor_plume_nav.tests submodule
try:
    from odor_plume_nav.tests import (
        create_test_navigator,
        create_test_environment,
        generate_test_trajectory,
        validate_simulation_results,
    )
except ImportError:
    # Testing utilities not available
    pass

# Feature availability flags for backward compatibility and runtime checks
FEATURES = {
    'cli': cli is not None,
    'database': SessionManager is not None,
    'enhanced_api': True,  # Always available in unified package
    'enhanced_data': True,  # Always available in unified package
    'testing_utils': True,  # Always available in unified package
}

def get_available_features():
    """
    Get a dictionary of available feature flags.
    
    Returns:
        dict: Feature availability status
    """
    return FEATURES.copy()

def is_feature_available(feature_name):
    """
    Check if a specific feature is available.
    
    Args:
        feature_name (str): Name of the feature to check
        
    Returns:
        bool: True if feature is available, False otherwise
    """
    return FEATURES.get(feature_name, False)

# Define public API for wildcard imports
__all__ = [
    # Version and metadata
    '__version__',
    
    # Core API functions
    'create_navigator',
    'create_video_plume', 
    'run_plume_simulation',
    'visualize_simulation_results',
    
    # Core navigation classes
    'Navigator',
    'SingleAgentController',
    'MultiAgentController', 
    'NavigatorProtocol',
    'run_simulation',
    
    # Environment classes
    'VideoPlume',
    
    # Visualization functions
    'visualize_trajectory',
    
    # Configuration classes and functions
    'NavigatorConfig',
    'SingleAgentConfig', 
    'MultiAgentConfig',
    'VideoPlumeConfig',
    'load_config',
    'save_config',
    
    # Utility functions
    'load_yaml',
    'save_yaml',
    'load_json',
    'save_json', 
    'load_numpy',
    'save_numpy',
    'setup_logger',
    'get_module_logger',
    'DEFAULT_FORMAT',
    'MODULE_FORMAT',
    'LOG_LEVELS',
    
    # Feature availability functions
    'get_available_features',
    'is_feature_available',
    'FEATURES',
]
