"""
Reinforcement Learning Module for Odor Plume Navigation

This module provides comprehensive reinforcement learning capabilities for the
odor plume navigation system, including Gymnasium environment wrappers, training
utilities, custom policy implementations, and integration with stable-baselines3.

The module gracefully handles optional RL dependencies and provides feature
availability checking to ensure compatibility across different deployment scenarios.

Features:
    - Gymnasium-compliant environment wrappers
    - RL algorithm training utilities (PPO, SAC, TD3)
    - Custom policy implementations for multi-modal observations
    - Vectorized environment support for parallel training
    - Integration with modern RL frameworks (stable-baselines3, shimmy)
"""

__version__ = "0.2.0"
__author__ = "Odor Plume Navigation Team"

# Core RL training utilities
try:
    from odor_plume_nav.rl.training import (
        create_rl_algorithm,
        train_policy,
        create_vectorized_env,
        setup_training_callbacks,
        TrainingConfig,
        AlgorithmFactory,
    )
    _training_available = True
except ImportError:
    # Training utilities not available - likely missing RL dependencies
    _training_available = False
    create_rl_algorithm = None
    train_policy = None
    create_vectorized_env = None
    setup_training_callbacks = None
    TrainingConfig = None
    AlgorithmFactory = None

# Custom policy implementations
try:
    from odor_plume_nav.rl.policies import (
        MultiModalNavigationPolicy,
        OdorConcentrationNet,
        SpatialFeatureExtractor,
        NavigationPolicyConfig,
    )
    _policies_available = True
except ImportError:
    # Custom policies not available - using standard stable-baselines3 policies
    _policies_available = False
    MultiModalNavigationPolicy = None
    OdorConcentrationNet = None
    SpatialFeatureExtractor = None
    NavigationPolicyConfig = None

# Check for core RL framework dependencies
_rl_deps_available = True
_rl_deps_errors = []

try:
    import gymnasium
    _gymnasium_available = True
    _gymnasium_version = getattr(gymnasium, '__version__', 'unknown')
except ImportError as e:
    _gymnasium_available = False
    _gymnasium_version = None
    _rl_deps_available = False
    _rl_deps_errors.append(f"gymnasium: {e}")

try:
    import stable_baselines3
    _sb3_available = True
    _sb3_version = getattr(stable_baselines3, '__version__', 'unknown')
except ImportError as e:
    _sb3_available = False
    _sb3_version = None
    _rl_deps_available = False
    _rl_deps_errors.append(f"stable-baselines3: {e}")

try:
    import shimmy
    _shimmy_available = True
    _shimmy_version = getattr(shimmy, '__version__', 'unknown')
except ImportError as e:
    _shimmy_available = False
    _shimmy_version = None
    # shimmy is optional for basic RL functionality
    _rl_deps_errors.append(f"shimmy (optional): {e}")

# Feature availability tracking
RL_FEATURES = {
    'training_utilities': _training_available and _rl_deps_available,
    'custom_policies': _policies_available and _rl_deps_available,
    'gymnasium': _gymnasium_available,
    'stable_baselines3': _sb3_available,
    'shimmy': _shimmy_available,
    'vectorized_training': _training_available and _sb3_available,
    'rl_complete': _rl_deps_available and _training_available and _policies_available,
}

def is_rl_available():
    """
    Check if reinforcement learning capabilities are fully available.
    
    Returns:
        bool: True if core RL dependencies and modules are available, False otherwise
    """
    return RL_FEATURES['rl_complete']

def get_rl_features():
    """
    Get a dictionary of available RL feature flags.
    
    Returns:
        dict: RL feature availability status
    """
    return RL_FEATURES.copy()

def is_rl_feature_available(feature_name):
    """
    Check if a specific RL feature is available.
    
    Args:
        feature_name (str): Name of the RL feature to check
        
    Returns:
        bool: True if feature is available, False otherwise
    """
    return RL_FEATURES.get(feature_name, False)

def get_rl_dependency_info():
    """
    Get detailed information about RL dependency availability and versions.
    
    Returns:
        dict: Dictionary containing dependency versions and availability status
    """
    return {
        'gymnasium': {
            'available': _gymnasium_available,
            'version': _gymnasium_version,
        },
        'stable_baselines3': {
            'available': _sb3_available,
            'version': _sb3_version,
        },
        'shimmy': {
            'available': _shimmy_available,
            'version': _shimmy_version,
        },
        'errors': _rl_deps_errors,
    }

def require_rl_dependencies():
    """
    Raise an informative error if required RL dependencies are missing.
    
    Raises:
        ImportError: If core RL dependencies are not available with installation instructions
    """
    if not _rl_deps_available:
        error_msg = (
            "Reinforcement learning functionality requires additional dependencies.\n"
            "Missing dependencies:\n"
        )
        for error in _rl_deps_errors:
            error_msg += f"  - {error}\n"
        
        error_msg += (
            "\nTo install RL dependencies, run:\n"
            "  pip install 'odor-plume-nav[rl]'\n"
            "\nOr install individual packages:\n"
            "  pip install gymnasium>=1.0.0 stable-baselines3>=2.6.0 shimmy>=2.0.0"
        )
        raise ImportError(error_msg)

def create_training_environment(*args, **kwargs):
    """
    Factory function for creating RL training environments.
    
    This function provides a unified interface for environment creation with
    proper dependency checking and error handling.
    
    Args:
        *args: Positional arguments passed to environment factory
        **kwargs: Keyword arguments passed to environment factory
        
    Returns:
        gymnasium.Env: Configured training environment
        
    Raises:
        ImportError: If RL dependencies are not available
    """
    require_rl_dependencies()
    
    # Import here to avoid circular imports and ensure dependencies are available
    from odor_plume_nav.environments.gymnasium_env import create_gymnasium_environment
    return create_gymnasium_environment(*args, **kwargs)

# Define public API for wildcard imports
__all__ = [
    # Version and metadata
    '__version__',
    '__author__',
    
    # Core RL training utilities (if available)
    'create_rl_algorithm',
    'train_policy',
    'create_vectorized_env',
    'setup_training_callbacks',
    'TrainingConfig',
    'AlgorithmFactory',
    
    # Custom policy implementations (if available)
    'MultiModalNavigationPolicy',
    'OdorConcentrationNet',
    'SpatialFeatureExtractor',
    'NavigationPolicyConfig',
    
    # Feature availability functions
    'is_rl_available',
    'get_rl_features',
    'is_rl_feature_available',
    'get_rl_dependency_info',
    'require_rl_dependencies',
    'create_training_environment',
    
    # Feature tracking
    'RL_FEATURES',
]

# Convenience imports for backward compatibility and ease of use
# Only export if dependencies are available
if _rl_deps_available:
    try:
        # Re-export key gymnasium components for convenience
        from gymnasium import Env, Space
        from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
        __all__.extend(['Env', 'Space', 'Box', 'Dict', 'Discrete', 'MultiDiscrete'])
    except ImportError:
        pass
    
    try:
        # Re-export key stable-baselines3 algorithms for convenience
        from stable_baselines3 import PPO, SAC, TD3, DQN
        __all__.extend(['PPO', 'SAC', 'TD3', 'DQN'])
    except ImportError:
        pass