"""
Environments module for odor plume navigation.

This module contains environment classes for simulation, including VideoPlume
and new Gymnasium-compliant environments for reinforcement learning integration.

Features new Gymnasium 0.29.x-compliant API with dual compatibility support for
legacy gym usage. Includes centralized Loguru logging and structured environment
registration with comprehensive diagnostic capabilities.
"""

import warnings
from typing import Dict, Any, Optional, Union

# Core environment - always available
from odor_plume_nav.environments.video_plume import VideoPlume

# Import centralized logging setup first for structured diagnostics
try:
    from odor_plume_nav.utils.logging_setup import get_enhanced_logger
    logger = get_enhanced_logger(__name__)
    LOGGING_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    LOGGING_AVAILABLE = False

# List of all available exports - will be updated based on successful imports
__all__ = [
    "VideoPlume",
]

# Check for legacy gym import attempts and issue deprecation warnings
try:
    import gym
    warnings.warn(
        "Legacy 'gym' package detected. The gym package is deprecated and will be removed in a future version. "
        "Please migrate to 'gymnasium' package and use environment ID 'PlumeNavSim-v0' for new Gymnasium API compatibility. "
        "Existing 'OdorPlumeNavigation-v1' ID continues to work for backward compatibility.",
        DeprecationWarning,
        stacklevel=2
    )
    LEGACY_GYM_AVAILABLE = True
    logger.warning(
        "Legacy gym package import detected", 
        extra={
            "metric_type": "deprecation_warning",
            "deprecated_package": "gym",
            "recommended_package": "gymnasium",
            "recommended_env_id": "PlumeNavSim-v0"
        }
    ) if LOGGING_AVAILABLE else None
except ImportError:
    LEGACY_GYM_AVAILABLE = False

# Check for Gymnasium availability
try:
    import gymnasium as gym_modern
    GYMNASIUM_AVAILABLE = True
    logger.info(
        "Gymnasium package available for modern RL API", 
        extra={
            "metric_type": "environment_capability",
            "package": "gymnasium",
            "api_version": "0.29.x"
        }
    ) if LOGGING_AVAILABLE else None
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gym_modern = None

# Conditional imports for reinforcement learning features
# These require gymnasium and related RL dependencies
try:
    from odor_plume_nav.environments.gymnasium_env import GymnasiumEnv
    __all__.append("GymnasiumEnv")
    RL_ENV_AVAILABLE = True
except ImportError:
    # RL dependencies not available - GymnasiumEnv not available
    GymnasiumEnv = None
    RL_ENV_AVAILABLE = False

# Import compatibility layer for dual API support
try:
    from odor_plume_nav.environments.compat import (
        CompatibilityEnvWrapper,
        detect_api_context,
        create_environment_factory
    )
    __all__.extend([
        "CompatibilityEnvWrapper",
        "detect_api_context", 
        "create_environment_factory"
    ])
    COMPAT_LAYER_AVAILABLE = True
    logger.info(
        "Compatibility layer available for dual API support",
        extra={
            "metric_type": "environment_capability",
            "feature": "dual_api_support",
            "legacy_support": True,
            "modern_support": True
        }
    ) if LOGGING_AVAILABLE else None
except ImportError:
    # Compatibility layer not available - proceed without dual API support
    CompatibilityEnvWrapper = None
    detect_api_context = None
    create_environment_factory = None
    COMPAT_LAYER_AVAILABLE = False
    logger.warning(
        "Compatibility layer not available, single API mode only",
        extra={
            "metric_type": "environment_limitation",
            "missing_feature": "dual_api_support"
        }
    ) if LOGGING_AVAILABLE else None

try:
    from odor_plume_nav.environments.spaces import (
        create_action_space,
        create_observation_space,
        ActionSpaceConfig,
        ObservationSpaceConfig,
    )
    __all__.extend([
        "create_action_space",
        "create_observation_space", 
        "ActionSpaceConfig",
        "ObservationSpaceConfig",
    ])
    SPACES_AVAILABLE = True
except ImportError:
    # Spaces module not available
    create_action_space = None
    create_observation_space = None
    ActionSpaceConfig = None
    ObservationSpaceConfig = None
    SPACES_AVAILABLE = False

try:
    from odor_plume_nav.environments.wrappers import (
        NormalizationWrapper,
        FrameStackWrapper,
        ClippingWrapper,
        RewardShapingWrapper,
    )
    __all__.extend([
        "NormalizationWrapper",
        "FrameStackWrapper",
        "ClippingWrapper", 
        "RewardShapingWrapper",
    ])
    WRAPPERS_AVAILABLE = True
except ImportError:
    # Wrappers module not available
    NormalizationWrapper = None
    FrameStackWrapper = None
    ClippingWrapper = None
    RewardShapingWrapper = None
    WRAPPERS_AVAILABLE = False

# Environment registration functionality
def register_environments() -> Dict[str, bool]:
    """
    Register all available environment IDs with Gymnasium and legacy gym.
    
    Implements dual registration strategy:
    1. New PlumeNavSim-v0 for Gymnasium 0.29.x API compliance  
    2. Maintains OdorPlumeNavigation-v1 for backward compatibility
    
    Returns:
        Dict mapping environment IDs to registration success status
        
    Note:
        Registration diagnostics are logged with structured metadata for
        troubleshooting and environment discovery.
    """
    registration_results = {}
    
    if not RL_ENV_AVAILABLE:
        logger.warning(
            "RL environment unavailable, skipping environment registration",
            extra={
                "metric_type": "registration_skip",
                "reason": "missing_gymnasium_env"
            }
        ) if LOGGING_AVAILABLE else None
        return registration_results
    
    # Register new Gymnasium 0.29.x compliant environment
    if GYMNASIUM_AVAILABLE:
        try:
            gym_modern.register(
                id='PlumeNavSim-v0',
                entry_point='odor_plume_nav.environments.gymnasium_env:GymnasiumEnv',
                max_episode_steps=1000,
                reward_threshold=100.0,
                nondeterministic=False,
                kwargs={
                    'use_gymnasium_api': True,  # Explicit Gymnasium API mode
                    'api_version': '0.29.x'
                }
            )
            registration_results['PlumeNavSim-v0'] = True
            logger.info(
                "Successfully registered PlumeNavSim-v0 for Gymnasium API",
                extra={
                    "metric_type": "environment_registration",
                    "env_id": "PlumeNavSim-v0",
                    "api_type": "gymnasium",
                    "api_version": "0.29.x",
                    "compliance": "5-tuple",
                    "terminated_truncated_support": True
                }
            ) if LOGGING_AVAILABLE else None
        except Exception as e:
            registration_results['PlumeNavSim-v0'] = False
            logger.error(
                f"Failed to register PlumeNavSim-v0: {e}",
                extra={
                    "metric_type": "registration_error",
                    "env_id": "PlumeNavSim-v0",
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            ) if LOGGING_AVAILABLE else None
    
    # Register legacy environment for backward compatibility
    # Supports both legacy gym and gymnasium depending on availability
    target_gym = gym_modern if GYMNASIUM_AVAILABLE else (gym if LEGACY_GYM_AVAILABLE else None)
    
    if target_gym:
        try:
            target_gym.register(
                id='OdorPlumeNavigation-v1',
                entry_point='odor_plume_nav.environments.gymnasium_env:GymnasiumEnv',
                max_episode_steps=1000,
                reward_threshold=100.0,
                nondeterministic=False,
                kwargs={
                    'use_gymnasium_api': False,  # Legacy compatibility mode
                    'api_version': 'legacy'
                }
            )
            registration_results['OdorPlumeNavigation-v1'] = True
            api_type = "gymnasium" if GYMNASIUM_AVAILABLE else "legacy_gym"
            logger.info(
                "Successfully registered OdorPlumeNavigation-v1 for backward compatibility",
                extra={
                    "metric_type": "environment_registration",
                    "env_id": "OdorPlumeNavigation-v1",
                    "api_type": api_type,
                    "compliance": "4-tuple",
                    "backward_compatible": True,
                    "deprecated": LEGACY_GYM_AVAILABLE and not GYMNASIUM_AVAILABLE
                }
            ) if LOGGING_AVAILABLE else None
        except Exception as e:
            registration_results['OdorPlumeNavigation-v1'] = False
            logger.error(
                f"Failed to register OdorPlumeNavigation-v1: {e}",
                extra={
                    "metric_type": "registration_error", 
                    "env_id": "OdorPlumeNavigation-v1",
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            ) if LOGGING_AVAILABLE else None
    
    return registration_results


def make_environment(
    env_id: str, 
    config: Optional[Union[Dict[str, Any], Any]] = None,
    **kwargs
) -> Union["GymnasiumEnv", None]:
    """
    Factory function for creating environments with proper API detection.
    
    Args:
        env_id: Environment identifier ('PlumeNavSim-v0' or 'OdorPlumeNavigation-v1')
        config: Optional configuration dictionary or DictConfig
        **kwargs: Additional parameters for environment creation
        
    Returns:
        Configured environment instance or None if creation fails
        
    Examples:
        >>> # Modern Gymnasium API
        >>> env = make_environment('PlumeNavSim-v0', {'video_path': 'plume.mp4'})
        >>> obs, info = env.reset()
        >>> obs, reward, terminated, truncated, info = env.step(action)
        
        >>> # Legacy compatibility
        >>> env = make_environment('OdorPlumeNavigation-v1', {'video_path': 'plume.mp4'}) 
        >>> obs = env.reset()
        >>> obs, reward, done, info = env.step(action)
    """
    if not RL_ENV_AVAILABLE:
        logger.error(
            f"Cannot create environment {env_id}: RL environment not available",
            extra={
                "metric_type": "environment_creation_error",
                "env_id": env_id,
                "reason": "missing_gymnasium_env"
            }
        ) if LOGGING_AVAILABLE else None
        return None
    
    # Determine which gym implementation to use
    target_gym = None
    if env_id == 'PlumeNavSim-v0' and GYMNASIUM_AVAILABLE:
        target_gym = gym_modern
    elif env_id == 'OdorPlumeNavigation-v1':
        target_gym = gym_modern if GYMNASIUM_AVAILABLE else (gym if LEGACY_GYM_AVAILABLE else None)
    
    if target_gym is None:
        logger.error(
            f"No suitable gym implementation available for {env_id}",
            extra={
                "metric_type": "environment_creation_error",
                "env_id": env_id,
                "gymnasium_available": GYMNASIUM_AVAILABLE,
                "legacy_gym_available": LEGACY_GYM_AVAILABLE
            }
        ) if LOGGING_AVAILABLE else None
        return None
    
    try:
        # Create environment with configuration
        if config:
            merged_kwargs = dict(config) if isinstance(config, dict) else dict(config._content if hasattr(config, '_content') else config)
            merged_kwargs.update(kwargs)
            env = target_gym.make(env_id, **merged_kwargs)
        else:
            env = target_gym.make(env_id, **kwargs)
        
        logger.info(
            f"Successfully created environment {env_id}",
            extra={
                "metric_type": "environment_creation_success",
                "env_id": env_id,
                "api_type": "gymnasium" if target_gym == gym_modern else "legacy_gym",
                "config_provided": config is not None,
                "kwargs_count": len(kwargs)
            }
        ) if LOGGING_AVAILABLE else None
        
        return env
        
    except Exception as e:
        logger.error(
            f"Failed to create environment {env_id}: {e}",
            extra={
                "metric_type": "environment_creation_error",
                "env_id": env_id,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        ) if LOGGING_AVAILABLE else None
        return None


def get_available_environments() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available environment IDs and their capabilities.
    
    Returns:
        Dictionary mapping environment IDs to their metadata and capabilities
    """
    environments = {}
    
    if RL_ENV_AVAILABLE:
        environments['PlumeNavSim-v0'] = {
            'api_type': 'gymnasium',
            'api_version': '0.29.x',
            'step_returns': '5-tuple',
            'supports_terminated_truncated': True,
            'available': GYMNASIUM_AVAILABLE,
            'description': 'Modern Gymnasium API compliant environment',
            'recommended': True
        }
        
        environments['OdorPlumeNavigation-v1'] = {
            'api_type': 'legacy_compatible',
            'api_version': 'legacy',
            'step_returns': '4-tuple',
            'supports_terminated_truncated': False,
            'available': GYMNASIUM_AVAILABLE or LEGACY_GYM_AVAILABLE,
            'description': 'Backward compatible environment for legacy code',
            'recommended': False,
            'deprecated': LEGACY_GYM_AVAILABLE and not GYMNASIUM_AVAILABLE
        }
    
    return environments


def diagnose_environment_setup() -> Dict[str, Any]:
    """
    Comprehensive diagnostic information about environment setup and capabilities.
    
    Returns:
        Dictionary containing diagnostic information for troubleshooting
    """
    diagnostics = {
        'packages': {
            'gymnasium_available': GYMNASIUM_AVAILABLE,
            'legacy_gym_available': LEGACY_GYM_AVAILABLE,
            'logging_available': LOGGING_AVAILABLE,
            'rl_env_available': RL_ENV_AVAILABLE,
            'compat_layer_available': COMPAT_LAYER_AVAILABLE,
            'spaces_available': SPACES_AVAILABLE,
            'wrappers_available': WRAPPERS_AVAILABLE
        },
        'environments': get_available_environments(),
        'recommendations': []
    }
    
    # Generate recommendations based on setup
    if not GYMNASIUM_AVAILABLE and LEGACY_GYM_AVAILABLE:
        diagnostics['recommendations'].append(
            "Install gymnasium package for modern RL API: pip install gymnasium==0.29.*"
        )
    
    if not RL_ENV_AVAILABLE:
        diagnostics['recommendations'].append(
            "Install RL dependencies for environment support: pip install 'odor_plume_nav[rl]'"
        )
    
    if LEGACY_GYM_AVAILABLE:
        diagnostics['recommendations'].append(
            "Consider migrating from legacy 'gym' to 'gymnasium' package for future compatibility"
        )
    
    if not LOGGING_AVAILABLE:
        diagnostics['recommendations'].append(
            "Install enhanced logging for better diagnostics: pip install loguru>=0.7.0"
        )
    
    return diagnostics


# Add new factory and diagnostic functions to exports
__all__.extend([
    "register_environments",
    "make_environment", 
    "get_available_environments",
    "diagnose_environment_setup"
])

# Module initialization: Automatically register environments on import
# This ensures environments are available via gymnasium.make() immediately
_registration_results = {}

try:
    _registration_results = register_environments()
    
    # Log module initialization status
    if LOGGING_AVAILABLE:
        successful_registrations = [env_id for env_id, success in _registration_results.items() if success]
        failed_registrations = [env_id for env_id, success in _registration_results.items() if not success]
        
        logger.info(
            "Environment module initialization complete",
            extra={
                "metric_type": "module_initialization",
                "successful_registrations": successful_registrations,
                "failed_registrations": failed_registrations,
                "total_environments": len(_registration_results),
                "gymnasium_available": GYMNASIUM_AVAILABLE,
                "legacy_gym_available": LEGACY_GYM_AVAILABLE,
                "compat_layer_available": COMPAT_LAYER_AVAILABLE
            }
        )
        
        # Issue warnings for failed registrations
        for env_id in failed_registrations:
            logger.warning(
                f"Environment {env_id} registration failed",
                extra={
                    "metric_type": "registration_failure",
                    "env_id": env_id
                }
            )
    
    # Provide user-friendly summary for common import scenarios
    if successful_registrations:
        if 'PlumeNavSim-v0' in successful_registrations:
            # Modern setup - recommend new environment
            pass  # Already logged above
        elif 'OdorPlumeNavigation-v1' in successful_registrations:
            # Legacy compatibility mode
            if LOGGING_AVAILABLE:
                logger.info(
                    "Running in legacy compatibility mode",
                    extra={
                        "metric_type": "compatibility_mode",
                        "recommendation": "consider_upgrading_to_gymnasium"
                    }
                )

except Exception as e:
    # Handle initialization errors gracefully
    if LOGGING_AVAILABLE:
        logger.error(
            f"Environment module initialization failed: {e}",
            extra={
                "metric_type": "module_initialization_error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )
    else:
        # Fallback warning if logging not available
        warnings.warn(
            f"Environment registration failed: {e}. "
            "Some environments may not be available via gymnasium.make()",
            RuntimeWarning,
            stacklevel=2
        )

# Module provides the following environment IDs for registration:
#
# Modern Gymnasium API (recommended):
#   PlumeNavSim-v0 - Returns 5-tuple (obs, reward, terminated, truncated, info)
#   Usage: env = gymnasium.make('PlumeNavSim-v0', video_path='plume.mp4')
#
# Legacy Compatibility:
#   OdorPlumeNavigation-v1 - Returns 4-tuple (obs, reward, done, info) 
#   Usage: env = gym.make('OdorPlumeNavigation-v1', video_path='plume.mp4')
#
# Factory Functions:
#   make_environment() - Recommended factory with configuration support
#   get_available_environments() - Discovery of available environment IDs
#   diagnose_environment_setup() - Troubleshooting and setup validation
#
# For detailed usage examples and migration guidance, see the project documentation.