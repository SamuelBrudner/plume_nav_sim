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

# RL integration features - conditional imports for optional RL capabilities per F-011 and F-015
try:
    from odor_plume_nav.environments.gymnasium_env import GymnasiumEnv
    from odor_plume_nav.environments.spaces import (
        ActionSpace,
        ObservationSpace,
    )
    from odor_plume_nav.environments.wrappers import (
        NormalizationWrapper,
        FrameStackWrapper,
        RewardShapingWrapper,
    )
    _gymnasium_available = True
except ImportError:
    # Gymnasium environment not available - requires gymnasium and stable-baselines3
    GymnasiumEnv = None
    ActionSpace = None
    ObservationSpace = None
    NormalizationWrapper = None
    FrameStackWrapper = None
    RewardShapingWrapper = None
    _gymnasium_available = False

# Enhanced API functions with RL integration per F-015 factory function requirements
try:
    from odor_plume_nav.api.navigation import create_gymnasium_environment
    _gymnasium_factory_available = True
except ImportError:
    # Gymnasium environment factory not available
    create_gymnasium_environment = None
    _gymnasium_factory_available = False

# RL training utilities from odor_plume_nav.rl submodule per Section 0.1.2 new components
try:
    from odor_plume_nav.rl.training import (
        train_policy,
        evaluate_policy,
        create_vectorized_env,
        save_trained_model,
        load_trained_model,
    )
    from odor_plume_nav.rl.policies import (
        CustomPolicy,
        MultiModalPolicy,
        create_policy_network,
    )
    _rl_training_available = True
except ImportError:
    # RL training utilities not available - requires stable-baselines3
    train_policy = None
    evaluate_policy = None
    create_vectorized_env = None
    save_trained_model = None
    load_trained_model = None
    CustomPolicy = None
    MultiModalPolicy = None
    create_policy_network = None
    _rl_training_available = False

# Check for stable-baselines3 availability for vectorized training per Section 0.2.4 dependencies
try:
    import stable_baselines3
    _stable_baselines3_available = True
except ImportError:
    _stable_baselines3_available = False

# Feature availability flags for backward compatibility and runtime checks
FEATURES = {
    'cli': cli is not None,
    'database': SessionManager is not None,
    'enhanced_api': True,  # Always available in unified package
    'enhanced_data': True,  # Always available in unified package
    'testing_utils': True,  # Always available in unified package
    # RL integration feature flags per Section 0.1.3 API surface changes
    'rl_integration': _gymnasium_available and _gymnasium_factory_available and _rl_training_available,
    'gymnasium_env': _gymnasium_available,
    'stable_baselines3': _stable_baselines3_available,
    'vectorized_training': _stable_baselines3_available and _rl_training_available,
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

# Build dynamic __all__ list based on available features
_base_exports = [
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

# Add RL-specific exports when available per public API requirements
_rl_exports = []
if _gymnasium_available:
    _rl_exports.extend([
        'GymnasiumEnv',
        'ActionSpace',
        'ObservationSpace',
        'NormalizationWrapper',
        'FrameStackWrapper',
        'RewardShapingWrapper',
    ])

if _gymnasium_factory_available:
    _rl_exports.append('create_gymnasium_environment')

if _rl_training_available:
    _rl_exports.extend([
        'train_policy',
        'evaluate_policy',
        'create_vectorized_env',
        'save_trained_model',
        'load_trained_model',
        'CustomPolicy',
        'MultiModalPolicy',
        'create_policy_network',
    ])

# Define public API for wildcard imports
__all__ = _base_exports + _rl_exports