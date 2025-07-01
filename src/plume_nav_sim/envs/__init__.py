"""
Environments module for plume navigation simulation with modular architecture support.

This module contains environment classes for Gymnasium 0.29.x-compliant RL simulation,
including VideoPlume and modern Gymnasium environments for reinforcement learning integration
with enhanced extensibility hooks, frame caching capabilities, and pluggable component support.

Features modern Gymnasium 0.29.x API with dual compatibility support for legacy gym usage
through the compatibility shim layer. Includes centralized Loguru logging, structured
environment registration, comprehensive diagnostic capabilities, and configurable frame
caching with memory management.

Key Components:
- PlumeNavigationEnv: Core Gymnasium-compliant environment with extensibility hooks
- VideoPlume: High-performance video processing with frame caching
- SpacesFactory: Type-safe observation and action space creation utilities
- CompatibilityWrapper: Legacy gym API support with automatic format conversion
- Environment wrappers: Normalization, frame stacking, clipping, reward shaping

Modular Architecture Components:
- PlumeModelProtocol: Pluggable plume modeling (Gaussian, Turbulent, Video-based)
- WindFieldProtocol: Environmental wind dynamics with configurable complexity
- SensorProtocol: Flexible sensing modalities (Binary, Concentration, Gradient)
- ModularEnvironmentFactory: Configuration-driven component instantiation
- Component diagnostics: Availability reporting for all modular simulation components
"""

import warnings
from typing import Dict, Any, Optional, Union

# Core environment - always available
try:
    from plume_nav_sim.envs.video_plume import VideoPlume
    VIDEO_PLUME_AVAILABLE = True
except ImportError:
    VideoPlume = None
    VIDEO_PLUME_AVAILABLE = False

# Import centralized logging setup first for structured diagnostics
try:
    from plume_nav_sim.utils.logging_setup import get_enhanced_logger
    logger = get_enhanced_logger(__name__)
    LOGGING_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    LOGGING_AVAILABLE = False

# List of all available exports - will be updated based on successful imports
__all__ = []

if VIDEO_PLUME_AVAILABLE:
    __all__.append("VideoPlume")

# Check for legacy gym import attempts and issue deprecation warnings
try:
    import gym
    warnings.warn(
        "Legacy 'gym' package detected. The gym package is deprecated and will be removed in v1.0. "
        "Please migrate to 'gymnasium' package and use environment ID 'PlumeNavSim-v0' for new Gymnasium API compliance. "
        "Existing 'OdorPlumeNavigation-v1' ID continues to work for backward compatibility. "
        "For migration guidance, see: https://gymnasium.farama.org/content/migration-guide/",
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
            "recommended_env_id": "PlumeNavSim-v0",
            "migration_guide": "https://gymnasium.farama.org/content/migration-guide/"
        }
    ) if LOGGING_AVAILABLE else None
except ImportError:
    LEGACY_GYM_AVAILABLE = False

# Check for Gymnasium availability - required for modern API
try:
    import gymnasium as gym_modern
    GYMNASIUM_AVAILABLE = True
    logger.info(
        "Gymnasium package available for modern RL API", 
        extra={
            "metric_type": "environment_capability",
            "package": "gymnasium",
            "api_version": "0.29.x",
            "supports_terminated_truncated": True
        }
    ) if LOGGING_AVAILABLE else None
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gym_modern = None
    logger.warning(
        "Gymnasium package not available - modern RL environments disabled",
        extra={
            "metric_type": "environment_limitation",
            "missing_package": "gymnasium",
            "recommendation": "pip install gymnasium==0.29.*"
        }
    ) if LOGGING_AVAILABLE else None

# Core environment implementation - requires gymnasium
try:
    from plume_nav_sim.envs.plume_navigation_env import PlumeNavigationEnv
    __all__.append("PlumeNavigationEnv")
    PLUME_ENV_AVAILABLE = True
    logger.info(
        "PlumeNavigationEnv available with extensibility hooks",
        extra={
            "metric_type": "environment_capability",
            "component": "plume_navigation_env",
            "hooks": ["compute_additional_obs", "compute_extra_reward", "on_episode_end"],
            "frame_cache_support": True
        }
    ) if LOGGING_AVAILABLE else None
except ImportError as e:
    PlumeNavigationEnv = None
    PLUME_ENV_AVAILABLE = False
    logger.warning(
        f"PlumeNavigationEnv not available: {e}",
        extra={
            "metric_type": "environment_limitation",
            "missing_component": "plume_navigation_env",
            "error": str(e)
        }
    ) if LOGGING_AVAILABLE else None

# Import compatibility layer for dual API support
try:
    from plume_nav_sim.envs.compat import (
        CompatibilityEnvWrapper,
        detect_api_context,
        create_environment_factory,
        LegacyWrapper
    )
    __all__.extend([
        "CompatibilityEnvWrapper",
        "detect_api_context", 
        "create_environment_factory",
        "LegacyWrapper"
    ])
    COMPAT_LAYER_AVAILABLE = True
    logger.info(
        "Compatibility layer available for dual API support",
        extra={
            "metric_type": "environment_capability",
            "feature": "dual_api_support",
            "legacy_support": True,
            "modern_support": True,
            "auto_detection": True
        }
    ) if LOGGING_AVAILABLE else None
except ImportError as e:
    CompatibilityEnvWrapper = None
    detect_api_context = None
    create_environment_factory = None
    LegacyWrapper = None
    COMPAT_LAYER_AVAILABLE = False
    logger.warning(
        f"Compatibility layer not available: {e}, single API mode only",
        extra={
            "metric_type": "environment_limitation",
            "missing_feature": "dual_api_support",
            "error": str(e)
        }
    ) if LOGGING_AVAILABLE else None

# Import space factory utilities for type-safe space creation
try:
    from plume_nav_sim.envs.spaces import (
        SpacesFactory,
        create_action_space,
        create_observation_space,
        ActionSpaceConfig,
        ObservationSpaceConfig,
        validate_space_compliance
    )
    __all__.extend([
        "SpacesFactory",
        "create_action_space",
        "create_observation_space", 
        "ActionSpaceConfig",
        "ObservationSpaceConfig",
        "validate_space_compliance"
    ])
    SPACES_AVAILABLE = True
    logger.info(
        "Space factory utilities available for type-safe space creation",
        extra={
            "metric_type": "environment_capability",
            "component": "spaces_factory",
            "features": ["Box", "Discrete", "Dict", "validation"]
        }
    ) if LOGGING_AVAILABLE else None
except ImportError as e:
    SpacesFactory = None
    create_action_space = None
    create_observation_space = None
    ActionSpaceConfig = None
    ObservationSpaceConfig = None
    validate_space_compliance = None
    SPACES_AVAILABLE = False
    logger.warning(
        f"Space factory utilities not available: {e}",
        extra={
            "metric_type": "environment_limitation",
            "missing_component": "spaces_factory",
            "error": str(e)
        }
    ) if LOGGING_AVAILABLE else None

# Import environment wrappers for enhanced functionality
try:
    from plume_nav_sim.envs.wrappers import (
        NormalizationWrapper,
        FrameStackWrapper,
        ClippingWrapper,
        RewardShapingWrapper,
        PerformanceMonitoringWrapper
    )
    __all__.extend([
        "NormalizationWrapper",
        "FrameStackWrapper",
        "ClippingWrapper", 
        "RewardShapingWrapper",
        "PerformanceMonitoringWrapper"
    ])
    WRAPPERS_AVAILABLE = True
    logger.info(
        "Environment wrappers available for enhanced functionality",
        extra={
            "metric_type": "environment_capability",
            "component": "wrappers",
            "types": ["normalization", "frame_stacking", "clipping", "reward_shaping", "monitoring"]
        }
    ) if LOGGING_AVAILABLE else None
except ImportError as e:
    NormalizationWrapper = None
    FrameStackWrapper = None
    ClippingWrapper = None
    RewardShapingWrapper = None
    PerformanceMonitoringWrapper = None
    WRAPPERS_AVAILABLE = False
    logger.warning(
        f"Environment wrappers not available: {e}",
        extra={
            "metric_type": "environment_limitation",
            "missing_component": "wrappers",
            "error": str(e)
        }
    ) if LOGGING_AVAILABLE else None

# Import modular component protocols for pluggable architecture
try:
    from plume_nav_sim.core.protocols import (
        PlumeModelProtocol,
        WindFieldProtocol,
        SensorProtocol,
        AgentObservationProtocol,
        AgentActionProtocol,
        NavigatorFactory
    )
    __all__.extend([
        "PlumeModelProtocol",
        "WindFieldProtocol",
        "SensorProtocol",
        "AgentObservationProtocol", 
        "AgentActionProtocol",
        "NavigatorFactory"
    ])
    PROTOCOLS_AVAILABLE = True
    logger.info(
        "Modular component protocols available for pluggable architecture",
        extra={
            "metric_type": "environment_capability",
            "component": "protocols",
            "protocols": ["plume_model", "wind_field", "sensor", "observation", "action"]
        }
    ) if LOGGING_AVAILABLE else None
except ImportError as e:
    PlumeModelProtocol = None
    WindFieldProtocol = None
    SensorProtocol = None
    AgentObservationProtocol = None
    AgentActionProtocol = None
    NavigatorFactory = None
    PROTOCOLS_AVAILABLE = False
    logger.warning(
        f"Modular component protocols not available: {e}",
        extra={
            "metric_type": "environment_limitation",
            "missing_component": "protocols",
            "error": str(e)
        }
    ) if LOGGING_AVAILABLE else None

# Import plume model implementations
try:
    from plume_nav_sim.models.plume import (
        GaussianPlumeModel,
        TurbulentPlumeModel,
        VideoPlumeAdapter
    )
    __all__.extend([
        "GaussianPlumeModel",
        "TurbulentPlumeModel", 
        "VideoPlumeAdapter"
    ])
    PLUME_MODELS_AVAILABLE = True
    logger.info(
        "Plume model implementations available",
        extra={
            "metric_type": "environment_capability",
            "component": "plume_models",
            "models": ["gaussian", "turbulent", "video_adapter"]
        }
    ) if LOGGING_AVAILABLE else None
except ImportError as e:
    GaussianPlumeModel = None
    TurbulentPlumeModel = None
    VideoPlumeAdapter = None
    PLUME_MODELS_AVAILABLE = False
    logger.info(
        f"Plume model implementations not available yet: {e}",
        extra={
            "metric_type": "environment_info",
            "missing_component": "plume_models",
            "note": "Models will be available after other agents complete implementation"
        }
    ) if LOGGING_AVAILABLE else None

# Import wind field implementations  
try:
    from plume_nav_sim.models.wind import (
        ConstantWindField,
        TurbulentWindField,
        TimeVaryingWindField
    )
    __all__.extend([
        "ConstantWindField",
        "TurbulentWindField",
        "TimeVaryingWindField"
    ])
    WIND_FIELDS_AVAILABLE = True
    logger.info(
        "Wind field implementations available",
        extra={
            "metric_type": "environment_capability",
            "component": "wind_fields",
            "fields": ["constant", "turbulent", "time_varying"]
        }
    ) if LOGGING_AVAILABLE else None
except ImportError as e:
    ConstantWindField = None
    TurbulentWindField = None
    TimeVaryingWindField = None
    WIND_FIELDS_AVAILABLE = False
    logger.info(
        f"Wind field implementations not available yet: {e}",
        extra={
            "metric_type": "environment_info",
            "missing_component": "wind_fields",
            "note": "Wind fields will be available after other agents complete implementation"
        }
    ) if LOGGING_AVAILABLE else None

# Import sensor implementations
try:
    from plume_nav_sim.core.sensors import (
        BinarySensor,
        ConcentrationSensor,
        GradientSensor,
        HistoricalSensor
    )
    __all__.extend([
        "BinarySensor",
        "ConcentrationSensor",
        "GradientSensor",
        "HistoricalSensor"
    ])
    SENSORS_AVAILABLE = True
    logger.info(
        "Sensor implementations available",
        extra={
            "metric_type": "environment_capability",
            "component": "sensors",
            "sensors": ["binary", "concentration", "gradient", "historical"]
        }
    ) if LOGGING_AVAILABLE else None
except ImportError as e:
    BinarySensor = None
    ConcentrationSensor = None
    GradientSensor = None
    HistoricalSensor = None
    SENSORS_AVAILABLE = False
    logger.info(
        f"Sensor implementations not available yet: {e}",
        extra={
            "metric_type": "environment_info",
            "missing_component": "sensors",
            "note": "Sensors will be available after other agents complete implementation"
        }
    ) if LOGGING_AVAILABLE else None

# Overall RL environment availability check
RL_ENV_AVAILABLE = PLUME_ENV_AVAILABLE and GYMNASIUM_AVAILABLE

# Modular architecture availability summary
MODULAR_ARCHITECTURE_AVAILABLE = PROTOCOLS_AVAILABLE
MODULAR_COMPONENTS_PARTIAL = PLUME_MODELS_AVAILABLE or WIND_FIELDS_AVAILABLE or SENSORS_AVAILABLE
MODULAR_COMPONENTS_COMPLETE = PLUME_MODELS_AVAILABLE and WIND_FIELDS_AVAILABLE and SENSORS_AVAILABLE

# Environment registration functionality
def register_environments() -> Dict[str, bool]:
    """
    Register all available environment IDs with Gymnasium and legacy gym.
    
    Implements dual registration strategy per Section 0.2.2 requirements:
    1. New PlumeNavSim-v0 for Gymnasium 0.29.x API compliance  
    2. Maintains OdorPlumeNavigation-v1 for backward compatibility
    3. Modular PlumeNavSim-v1 for new pluggable architecture support
    
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
                "reason": "missing_gymnasium_env_or_plume_env",
                "gymnasium_available": GYMNASIUM_AVAILABLE,
                "plume_env_available": PLUME_ENV_AVAILABLE
            }
        ) if LOGGING_AVAILABLE else None
        return registration_results
    
    # Register new Gymnasium 0.29.x compliant environment
    if GYMNASIUM_AVAILABLE:
        try:
            gym_modern.register(
                id='PlumeNavSim-v0',
                entry_point='plume_nav_sim.envs.plume_navigation_env:PlumeNavigationEnv',
                max_episode_steps=1000,
                reward_threshold=100.0,
                nondeterministic=False,
                kwargs={
                    'use_gymnasium_api': True,  # Explicit Gymnasium API mode
                    'api_version': '0.29.x',
                    'enable_extensibility_hooks': True,
                    'frame_cache_mode': 'lru'  # Default to LRU caching
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
                    "terminated_truncated_support": True,
                    "extensibility_hooks": True,
                    "frame_cache_default": "lru"
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
                entry_point='plume_nav_sim.envs.plume_navigation_env:PlumeNavigationEnv',
                max_episode_steps=1000,
                reward_threshold=100.0,
                nondeterministic=False,
                kwargs={
                    'use_gymnasium_api': False,  # Legacy compatibility mode
                    'api_version': 'legacy',
                    'enable_extensibility_hooks': True,
                    'frame_cache_mode': 'lru'
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
                    "deprecated": LEGACY_GYM_AVAILABLE and not GYMNASIUM_AVAILABLE,
                    "extensibility_hooks": True
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
    
    # Register new modular environment with pluggable architecture support
    if GYMNASIUM_AVAILABLE and MODULAR_ARCHITECTURE_AVAILABLE:
        try:
            gym_modern.register(
                id='PlumeNavSim-v1',
                entry_point='plume_nav_sim.envs.plume_navigation_env:PlumeNavigationEnv',
                max_episode_steps=1000,
                reward_threshold=100.0,
                nondeterministic=False,
                kwargs={
                    'use_gymnasium_api': True,  # Explicit Gymnasium API mode
                    'api_version': '0.29.x',
                    'enable_extensibility_hooks': True,
                    'frame_cache_mode': 'lru',  # Default to LRU caching
                    'modular_architecture': True,  # Enable modular component support
                    'support_plume_models': PLUME_MODELS_AVAILABLE,
                    'support_wind_fields': WIND_FIELDS_AVAILABLE,
                    'support_sensors': SENSORS_AVAILABLE
                }
            )
            registration_results['PlumeNavSim-v1'] = True
            logger.info(
                "Successfully registered PlumeNavSim-v1 for modular architecture",
                extra={
                    "metric_type": "environment_registration",
                    "env_id": "PlumeNavSim-v1",
                    "api_type": "gymnasium",
                    "api_version": "0.29.x",
                    "compliance": "5-tuple",
                    "modular_architecture": True,
                    "plume_models_available": PLUME_MODELS_AVAILABLE,
                    "wind_fields_available": WIND_FIELDS_AVAILABLE,
                    "sensors_available": SENSORS_AVAILABLE,
                    "protocols_available": PROTOCOLS_AVAILABLE
                }
            ) if LOGGING_AVAILABLE else None
        except Exception as e:
            registration_results['PlumeNavSim-v1'] = False
            logger.error(
                f"Failed to register PlumeNavSim-v1: {e}",
                extra={
                    "metric_type": "registration_error",
                    "env_id": "PlumeNavSim-v1",
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            ) if LOGGING_AVAILABLE else None
    
    return registration_results


def make_environment(
    env_id: str, 
    config: Optional[Union[Dict[str, Any], Any]] = None,
    **kwargs
) -> Union["PlumeNavigationEnv", None]:
    """
    Factory function for creating environments with proper API detection.
    
    Supports modern environment creation patterns per Section 0.2.2 requirements
    with enhanced configuration management and automatic API adaptation.
    
    Args:
        env_id: Environment identifier ('PlumeNavSim-v0', 'PlumeNavSim-v1', or 'OdorPlumeNavigation-v1')
        config: Optional configuration dictionary or DictConfig with frame cache settings
        **kwargs: Additional parameters for environment creation
        
    Returns:
        Configured environment instance or None if creation fails
        
    Examples:
        >>> # Modern Gymnasium API with frame cache configuration
        >>> env = make_environment('PlumeNavSim-v0', {
        ...     'video_path': 'plume.mp4',
        ...     'frame_cache_mode': 'lru',
        ...     'memory_limit_mb': 2048
        ... })
        >>> obs, info = env.reset()
        >>> obs, reward, terminated, truncated, info = env.step(action)
        
        >>> # Legacy compatibility with automatic format conversion
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
                "reason": "missing_gymnasium_env_or_plume_env",
                "gymnasium_available": GYMNASIUM_AVAILABLE,
                "plume_env_available": PLUME_ENV_AVAILABLE
            }
        ) if LOGGING_AVAILABLE else None
        return None
    
    # Determine which gym implementation to use
    target_gym = None
    if env_id == 'PlumeNavSim-v0' and GYMNASIUM_AVAILABLE:
        target_gym = gym_modern
    elif env_id == 'PlumeNavSim-v1' and GYMNASIUM_AVAILABLE:
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
                "kwargs_count": len(kwargs),
                "frame_cache_enabled": kwargs.get('frame_cache_mode', 'default') != 'none'
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


def create_modular_environment(
    navigator_config: Optional[Dict[str, Any]] = None,
    plume_model_config: Optional[Dict[str, Any]] = None,
    wind_field_config: Optional[Dict[str, Any]] = None,
    sensor_configs: Optional[List[Dict[str, Any]]] = None,
    **env_kwargs
) -> Optional["PlumeNavigationEnv"]:
    """
    Factory function for creating environments with modular component support.
    
    Creates a fully configured environment using the new modular architecture with
    pluggable plume models, wind fields, and sensor systems. Automatically handles
    component instantiation and integration based on configuration.
    
    Args:
        navigator_config: Navigator configuration for agent behavior
        plume_model_config: Plume model configuration (Gaussian, Turbulent, Video)
        wind_field_config: Wind field configuration (Constant, Turbulent, TimeVarying)
        sensor_configs: List of sensor configurations (Binary, Concentration, Gradient)
        **env_kwargs: Additional environment parameters
        
    Returns:
        Configured modular environment instance or None if creation fails
        
    Examples:
        >>> # Create environment with Gaussian plume and constant wind
        >>> env = create_modular_environment(
        ...     plume_model_config={'type': 'GaussianPlumeModel', 'source_position': (50, 50)},
        ...     wind_field_config={'type': 'ConstantWindField', 'velocity': (1.0, 0.0)},
        ...     sensor_configs=[{'type': 'ConcentrationSensor', 'dynamic_range': (0, 1)}]
        ... )
        
        >>> # Advanced turbulent environment
        >>> env = create_modular_environment(
        ...     plume_model_config={
        ...         'type': 'TurbulentPlumeModel',
        ...         'filament_count': 500,
        ...         'turbulence_intensity': 0.3
        ...     },
        ...     wind_field_config={
        ...         'type': 'TurbulentWindField',
        ...         'mean_velocity': (3.0, 1.0),
        ...         'turbulence_intensity': 0.2
        ...     },
        ...     sensor_configs=[
        ...         {'type': 'BinarySensor', 'threshold': 0.1},
        ...         {'type': 'GradientSensor', 'spatial_resolution': (0.5, 0.5)}
        ...     ]
        ... )
    """
    if not MODULAR_ARCHITECTURE_AVAILABLE:
        logger.error(
            "Modular architecture not available for environment creation",
            extra={
                "protocols_available": PROTOCOLS_AVAILABLE,
                "recommendation": "Ensure core protocols are available"
            }
        ) if LOGGING_AVAILABLE else None
        return None
    
    try:
        # Use NavigatorFactory to create modular environment if available
        if NavigatorFactory and hasattr(NavigatorFactory, 'create_modular_environment'):
            return NavigatorFactory.create_modular_environment(
                navigator_config=navigator_config or {},
                plume_model_config=plume_model_config or {'type': 'GaussianPlumeModel'},
                wind_field_config=wind_field_config,
                sensor_configs=sensor_configs,
                **env_kwargs
            )
        else:
            # Fallback to standard environment creation with modular support
            env_config = {
                'modular_architecture': True,
                'plume_model': plume_model_config,
                'wind_field': wind_field_config,
                'sensors': sensor_configs,
                'navigator': navigator_config,
                **env_kwargs
            }
            return make_environment('PlumeNavSim-v1', env_config)
            
    except Exception as e:
        logger.error(f"Failed to create modular environment: {e}")
        return None


def create_plume_model(config: Dict[str, Any]):
    """
    Factory function for creating plume model instances.
    
    Args:
        config: Configuration dictionary with 'type' key and model parameters
        
    Returns:
        Plume model instance implementing PlumeModelProtocol
        
    Examples:
        >>> gaussian_model = create_plume_model({
        ...     'type': 'GaussianPlumeModel',
        ...     'source_position': (50, 50),
        ...     'source_strength': 1000.0
        ... })
        
        >>> turbulent_model = create_plume_model({
        ...     'type': 'TurbulentPlumeModel',
        ...     'filament_count': 500,
        ...     'turbulence_intensity': 0.3
        ... })
    """
    if NavigatorFactory and hasattr(NavigatorFactory, 'create_plume_model'):
        try:
            return NavigatorFactory.create_plume_model(config)
        except Exception as e:
            logger.error(f"Failed to create plume model: {e}")
            return None
    
    model_type = config.get('type', 'GaussianPlumeModel')
    logger.warning(f"NavigatorFactory not available, cannot create {model_type}")
    return None


def create_wind_field(config: Dict[str, Any]):
    """
    Factory function for creating wind field instances.
    
    Args:
        config: Configuration dictionary with 'type' key and field parameters
        
    Returns:
        Wind field instance implementing WindFieldProtocol
        
    Examples:
        >>> constant_wind = create_wind_field({
        ...     'type': 'ConstantWindField',
        ...     'velocity': (2.0, 0.5)
        ... })
        
        >>> turbulent_wind = create_wind_field({
        ...     'type': 'TurbulentWindField',
        ...     'mean_velocity': (3.0, 1.0),
        ...     'turbulence_intensity': 0.2
        ... })
    """
    if NavigatorFactory and hasattr(NavigatorFactory, 'create_wind_field'):
        try:
            return NavigatorFactory.create_wind_field(config)
        except Exception as e:
            logger.error(f"Failed to create wind field: {e}")
            return None
    
    field_type = config.get('type', 'ConstantWindField')
    logger.warning(f"NavigatorFactory not available, cannot create {field_type}")
    return None


def create_sensors(sensor_configs: List[Dict[str, Any]]):
    """
    Factory function for creating sensor instances.
    
    Args:
        sensor_configs: List of sensor configuration dictionaries
        
    Returns:
        List of sensor instances implementing SensorProtocol
        
    Examples:
        >>> sensors = create_sensors([
        ...     {'type': 'BinarySensor', 'threshold': 0.1},
        ...     {'type': 'ConcentrationSensor', 'dynamic_range': (0, 1)},
        ...     {'type': 'GradientSensor', 'spatial_resolution': (0.5, 0.5)}
        ... ])
    """
    if NavigatorFactory and hasattr(NavigatorFactory, 'create_sensors'):
        try:
            return NavigatorFactory.create_sensors(sensor_configs)
        except Exception as e:
            logger.error(f"Failed to create sensors: {e}")
            return []
    
    logger.warning("NavigatorFactory not available, cannot create sensors")
    return []


def get_available_environments() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available environment IDs and their capabilities.
    
    Returns comprehensive metadata about registered environments including
    extensibility hooks, frame caching, and API compatibility information.
    
    Returns:
        Dictionary mapping environment IDs to their metadata and capabilities
    """
    environments = {}
    
    if RL_ENV_AVAILABLE:
        environments['PlumeNavSim-v0'] = {
            'api_type': 'gymnasium',
            'api_version': '0.29.x',
            'step_returns': '5-tuple',
            'reset_signature': 'reset(seed=None, options=None)',
            'supports_terminated_truncated': True,
            'extensibility_hooks': ['compute_additional_obs', 'compute_extra_reward', 'on_episode_end'],
            'frame_cache_modes': ['none', 'lru', 'all'],
            'modular_architecture': False,
            'available': GYMNASIUM_AVAILABLE,
            'description': 'Modern Gymnasium API compliant environment with extensibility hooks',
            'recommended': not MODULAR_ARCHITECTURE_AVAILABLE,
            'performance_target': '<10ms step time'
        }
        
        environments['PlumeNavSim-v1'] = {
            'api_type': 'gymnasium',
            'api_version': '0.29.x',
            'step_returns': '5-tuple',
            'reset_signature': 'reset(seed=None, options=None)',
            'supports_terminated_truncated': True,
            'extensibility_hooks': ['compute_additional_obs', 'compute_extra_reward', 'on_episode_end'],
            'frame_cache_modes': ['none', 'lru', 'all'],
            'modular_architecture': True,
            'pluggable_components': {
                'plume_models': ['gaussian', 'turbulent', 'video_adapter'],
                'wind_fields': ['constant', 'turbulent', 'time_varying'],
                'sensors': ['binary', 'concentration', 'gradient', 'historical']
            },
            'component_availability': {
                'plume_models': PLUME_MODELS_AVAILABLE,
                'wind_fields': WIND_FIELDS_AVAILABLE,  
                'sensors': SENSORS_AVAILABLE,
                'protocols': PROTOCOLS_AVAILABLE
            },
            'available': GYMNASIUM_AVAILABLE and MODULAR_ARCHITECTURE_AVAILABLE,
            'description': 'Advanced modular environment with pluggable plume models, wind fields, and sensors',
            'recommended': MODULAR_ARCHITECTURE_AVAILABLE,
            'performance_target': '<10ms step time',
            'configuration_driven': True
        }
        
        environments['OdorPlumeNavigation-v1'] = {
            'api_type': 'legacy_compatible',
            'api_version': 'legacy',
            'step_returns': '4-tuple',
            'reset_signature': 'reset()',
            'supports_terminated_truncated': False,
            'extensibility_hooks': ['compute_additional_obs', 'compute_extra_reward', 'on_episode_end'],
            'frame_cache_modes': ['none', 'lru', 'all'],
            'modular_architecture': False,
            'available': GYMNASIUM_AVAILABLE or LEGACY_GYM_AVAILABLE,
            'description': 'Backward compatible environment for legacy code with automatic format conversion',
            'recommended': False,
            'deprecated': LEGACY_GYM_AVAILABLE and not GYMNASIUM_AVAILABLE,
            'migration_note': 'Use PlumeNavSim-v1 for new projects with modular architecture'
        }
    
    return environments


def get_environment_flavors() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available environment flavors and configurations.
    
    Returns flavor configurations for different deployment scenarios per
    Section 0.2.3 requirements including memory and memoryless configurations.
    
    Returns:
        Dictionary mapping flavor names to their configuration specifications
    """
    return {
        'memory': {
            'description': 'Memory-optimized configuration with full frame caching',
            'frame_cache_mode': 'all',
            'memory_limit_mb': 2048,
            'use_case': 'Repeated training on same video with high memory availability',
            'performance': 'Highest step speed after preload',
            'config_file': 'conf/base/env/flavors/memory.yaml'
        },
        'memoryless': {
            'description': 'Memory-constrained configuration with minimal caching',
            'frame_cache_mode': 'none',
            'memory_limit_mb': 128,
            'use_case': 'Memory-constrained environments or streaming scenarios',
            'performance': 'Lower step speed but minimal memory usage',
            'config_file': 'conf/base/env/flavors/memoryless.yaml'
        },
        'balanced': {
            'description': 'Balanced configuration with LRU caching',
            'frame_cache_mode': 'lru',
            'memory_limit_mb': 1024,
            'use_case': 'Default configuration for most use cases',
            'performance': 'Good balance of speed and memory usage',
            'config_file': 'Built-in default'
        }
    }


def diagnose_environment_setup() -> Dict[str, Any]:
    """
    Comprehensive diagnostic information about environment setup and capabilities.
    
    Enhanced with frame cache diagnostics, extensibility hook information, and
    modular component availability per Section 0.2.2 requirements.
    
    Returns:
        Dictionary containing diagnostic information for troubleshooting
    """
    diagnostics = {
        'packages': {
            'gymnasium_available': GYMNASIUM_AVAILABLE,
            'legacy_gym_available': LEGACY_GYM_AVAILABLE,
            'logging_available': LOGGING_AVAILABLE,
            'plume_env_available': PLUME_ENV_AVAILABLE,
            'video_plume_available': VIDEO_PLUME_AVAILABLE,
            'compat_layer_available': COMPAT_LAYER_AVAILABLE,
            'spaces_available': SPACES_AVAILABLE,
            'wrappers_available': WRAPPERS_AVAILABLE,
            'rl_env_available': RL_ENV_AVAILABLE
        },
        'modular_architecture': {
            'protocols_available': PROTOCOLS_AVAILABLE,
            'plume_models_available': PLUME_MODELS_AVAILABLE,
            'wind_fields_available': WIND_FIELDS_AVAILABLE,
            'sensors_available': SENSORS_AVAILABLE,
            'modular_architecture_available': MODULAR_ARCHITECTURE_AVAILABLE,
            'components_partial': MODULAR_COMPONENTS_PARTIAL,
            'components_complete': MODULAR_COMPONENTS_COMPLETE
        },
        'component_implementations': {
            'plume_models': {
                'gaussian': GaussianPlumeModel is not None,
                'turbulent': TurbulentPlumeModel is not None,
                'video_adapter': VideoPlumeAdapter is not None
            },
            'wind_fields': {
                'constant': ConstantWindField is not None,
                'turbulent': TurbulentWindField is not None,
                'time_varying': TimeVaryingWindField is not None
            },
            'sensors': {
                'binary': BinarySensor is not None,
                'concentration': ConcentrationSensor is not None,
                'gradient': GradientSensor is not None,
                'historical': HistoricalSensor is not None
            }
        },
        'environments': get_available_environments(),
        'flavors': get_environment_flavors(),
        'capabilities': {
            'extensibility_hooks': PLUME_ENV_AVAILABLE,
            'frame_caching': PLUME_ENV_AVAILABLE,
            'dual_api_support': COMPAT_LAYER_AVAILABLE,
            'space_validation': SPACES_AVAILABLE,
            'environment_wrappers': WRAPPERS_AVAILABLE,
            'modular_components': MODULAR_ARCHITECTURE_AVAILABLE,
            'pluggable_plume_models': PLUME_MODELS_AVAILABLE,
            'configurable_wind_fields': WIND_FIELDS_AVAILABLE,
            'flexible_sensors': SENSORS_AVAILABLE
        },
        'recommendations': []
    }
    
    # Generate recommendations based on setup
    if not GYMNASIUM_AVAILABLE and LEGACY_GYM_AVAILABLE:
        diagnostics['recommendations'].append(
            "Install gymnasium package for modern RL API: pip install gymnasium==0.29.*"
        )
    
    if not RL_ENV_AVAILABLE:
        if not GYMNASIUM_AVAILABLE:
            diagnostics['recommendations'].append(
                "Install gymnasium for environment support: pip install gymnasium==0.29.*"
            )
        if not PLUME_ENV_AVAILABLE:
            diagnostics['recommendations'].append(
                "Install plume navigation environment dependencies"
            )
    
    if LEGACY_GYM_AVAILABLE:
        diagnostics['recommendations'].append(
            "Consider migrating from legacy 'gym' to 'gymnasium' package for future compatibility"
        )
    
    if not LOGGING_AVAILABLE:
        diagnostics['recommendations'].append(
            "Install enhanced logging for better diagnostics: pip install loguru>=0.7.0"
        )
    
    if not SPACES_AVAILABLE:
        diagnostics['recommendations'].append(
            "Install space factory utilities for type-safe environment creation"
        )
    
    if not WRAPPERS_AVAILABLE:
        diagnostics['recommendations'].append(
            "Install environment wrappers for enhanced functionality"
        )
    
    # Modular architecture recommendations
    if not PROTOCOLS_AVAILABLE:
        diagnostics['recommendations'].append(
            "Core protocols not available - modular architecture features disabled"
        )
    elif PROTOCOLS_AVAILABLE and not MODULAR_COMPONENTS_COMPLETE:
        missing_components = []
        if not PLUME_MODELS_AVAILABLE:
            missing_components.append("plume models")
        if not WIND_FIELDS_AVAILABLE:
            missing_components.append("wind fields")
        if not SENSORS_AVAILABLE:
            missing_components.append("sensors")
        
        diagnostics['recommendations'].append(
            f"Modular architecture partially available - missing: {', '.join(missing_components)}"
        )
    elif MODULAR_COMPONENTS_COMPLETE:
        diagnostics['recommendations'].append(
            "Full modular architecture available - use PlumeNavSim-v1 for advanced features"
        )
    
    if MODULAR_ARCHITECTURE_AVAILABLE and MODULAR_COMPONENTS_COMPLETE:
        diagnostics['recommendations'].append(
            "Consider using create_modular_environment() for configuration-driven component selection"
        )
    
    return diagnostics


# Add factory and diagnostic functions to exports
__all__.extend([
    "register_environments",
    "make_environment", 
    "get_available_environments",
    "get_environment_flavors",
    "diagnose_environment_setup",
    # Modular architecture factory functions
    "create_modular_environment",
    "create_plume_model",
    "create_wind_field", 
    "create_sensors"
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
                "module": "plume_nav_sim.envs",
                "successful_registrations": successful_registrations,
                "failed_registrations": failed_registrations,
                "total_environments": len(_registration_results),
                "gymnasium_available": GYMNASIUM_AVAILABLE,
                "legacy_gym_available": LEGACY_GYM_AVAILABLE,
                "compat_layer_available": COMPAT_LAYER_AVAILABLE,
                "extensibility_hooks_available": PLUME_ENV_AVAILABLE,
                "frame_cache_available": PLUME_ENV_AVAILABLE,
                "spaces_factory_available": SPACES_AVAILABLE,
                "wrappers_available": WRAPPERS_AVAILABLE,
                "modular_architecture_available": MODULAR_ARCHITECTURE_AVAILABLE,
                "protocols_available": PROTOCOLS_AVAILABLE,
                "plume_models_available": PLUME_MODELS_AVAILABLE,
                "wind_fields_available": WIND_FIELDS_AVAILABLE,
                "sensors_available": SENSORS_AVAILABLE,
                "modular_components_complete": MODULAR_COMPONENTS_COMPLETE
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
        if 'PlumeNavSim-v1' in successful_registrations:
            # Advanced modular setup - recommend new environment
            if LOGGING_AVAILABLE:
                logger.info(
                    "Advanced modular Gymnasium environment available",
                    extra={
                        "metric_type": "setup_success",
                        "recommended_env_id": "PlumeNavSim-v1",
                        "api_version": "0.29.x",
                        "modular_architecture": True,
                        "component_availability": {
                            "plume_models": PLUME_MODELS_AVAILABLE,
                            "wind_fields": WIND_FIELDS_AVAILABLE,
                            "sensors": SENSORS_AVAILABLE
                        }
                    }
                )
        elif 'PlumeNavSim-v0' in successful_registrations:
            # Modern setup - recommend standard environment
            if LOGGING_AVAILABLE:
                logger.info(
                    "Modern Gymnasium environment available",
                    extra={
                        "metric_type": "setup_success",
                        "recommended_env_id": "PlumeNavSim-v0",
                        "api_version": "0.29.x",
                        "note": "For modular architecture features, ensure protocols are available"
                    }
                )
        elif 'OdorPlumeNavigation-v1' in successful_registrations:
            # Legacy compatibility mode
            if LOGGING_AVAILABLE:
                logger.info(
                    "Running in legacy compatibility mode",
                    extra={
                        "metric_type": "compatibility_mode",
                        "recommendation": "consider_upgrading_to_gymnasium",
                        "available_env_id": "OdorPlumeNavigation-v1"
                    }
                )

except Exception as e:
    # Handle initialization errors gracefully
    if LOGGING_AVAILABLE:
        logger.error(
            f"Environment module initialization failed: {e}",
            extra={
                "metric_type": "module_initialization_error",
                "module": "plume_nav_sim.envs",
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )
    else:
        # Fallback warning if logging not available
        warnings.warn(
            f"Environment registration failed: {e}. "
            "Some environments may not be available via gymnasium.make(). "
            "Check dependencies: gymnasium==0.29.*, numpy>=1.26.0, psutil>=5.9.0",
            RuntimeWarning,
            stacklevel=2
        )

# Module provides the following environment IDs for registration:
#
# Advanced Modular Architecture (recommended):
#   PlumeNavSim-v1 - Returns 5-tuple (obs, reward, terminated, truncated, info)
#   Usage: env = gymnasium.make('PlumeNavSim-v1', plume_model={'type': 'GaussianPlumeModel'})
#   Features: Pluggable plume models, wind fields, sensors, configuration-driven component selection
#   Components: GaussianPlumeModel, TurbulentPlumeModel, ConstantWindField, BinarySensor, etc.
#
# Modern Gymnasium API (standard):
#   PlumeNavSim-v0 - Returns 5-tuple (obs, reward, terminated, truncated, info)
#   Usage: env = gymnasium.make('PlumeNavSim-v0', video_path='plume.mp4', frame_cache_mode='lru')
#   Features: Extensibility hooks, configurable frame caching, performance monitoring
#
# Legacy Compatibility:
#   OdorPlumeNavigation-v1 - Returns 4-tuple (obs, reward, done, info) with auto-conversion
#   Usage: env = gym.make('OdorPlumeNavigation-v1', video_path='plume.mp4')
#   Features: Backward compatibility, automatic format detection, deprecation guidance
#
# Modular Component Factory Functions:
#   create_modular_environment() - Create environment with pluggable components
#   create_plume_model() - Instantiate plume models (Gaussian, Turbulent, Video)
#   create_wind_field() - Instantiate wind fields (Constant, Turbulent, TimeVarying)
#   create_sensors() - Instantiate sensors (Binary, Concentration, Gradient, Historical)
#
# Standard Factory Functions:
#   make_environment() - Recommended factory with enhanced configuration support
#   get_available_environments() - Discovery of available environment IDs with capabilities
#   get_environment_flavors() - Available configuration flavors (memory, memoryless, balanced)
#   diagnose_environment_setup() - Comprehensive troubleshooting and modular component validation
#
# Extensibility Hooks (available in all environments):
#   compute_additional_obs(base_obs: dict) -> dict  # Add custom observations
#   compute_extra_reward(base_reward: float, info: dict) -> float  # Reward shaping
#   on_episode_end(final_info: dict) -> None  # Episode-level processing
#
# Modular Component Protocols:
#   PlumeModelProtocol: concentration_at(), step(), reset()
#   WindFieldProtocol: velocity_at(), step(), reset()
#   SensorProtocol: detect(), measure(), compute_gradient()
#   AgentObservationProtocol: construct_observation(), get_observation_space()
#   AgentActionProtocol: validate_action(), process_action(), get_action_space()
#
# Frame Cache Modes:
#   'none' - No caching, minimal memory usage
#   'lru' - LRU caching with configurable memory limits (default)
#   'all' - Full preload caching for maximum performance
#
# For detailed usage examples, modular architecture guide, and component development,
# see the project documentation at docs/extending_plume_nav_sim.md