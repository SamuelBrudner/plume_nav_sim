"""
Plume Navigation Simulation Package (v1.0.0)

A general-purpose, extensible simulation toolkit for odor plume navigation research 
with protocol-based architecture, zero-code extensibility, and comprehensive 
recording and analysis capabilities. Maintains full Gymnasium 0.29.x compatibility 
and backward compatibility for legacy Gym APIs.

This package provides enhanced tools for simulating how agents navigate through 
odor plumes, with support for both single and multi-agent simulations, 
extensible hook system, configurable frame caching, and comprehensive 
reinforcement learning integration.

Migration from legacy gym to modern Gymnasium is seamlessly supported through
automatic detection and compatibility shims, ensuring zero breaking changes
while providing clear migration guidance.
"""

import warnings
import inspect
import sys
import atexit

__version__ = "1.0.0"

# =============================================================================
# LEGACY GYM DETECTION AND DEPRECATION WARNING SYSTEM
# =============================================================================

def _detect_legacy_gym_import():
    """
    Detect if legacy gym package is being imported in calling context.
    
    This function inspects the calling stack to determine if the legacy
    'gym' package is being used instead of the modern 'gymnasium' package.
    Used to emit appropriate deprecation warnings while maintaining
    backward compatibility.
    
    Returns:
        bool: True if legacy gym usage detected, False otherwise
    """
    try:
        # Check if gym is imported in any frame of the calling stack
        frame = inspect.currentframe()
        while frame:
            frame_globals = frame.f_globals
            frame_locals = frame.f_locals
            
            # Check if 'gym' is in the namespace (indicating legacy usage)
            if ('gym' in frame_globals and 
                hasattr(frame_globals.get('gym'), 'make') and
                'gymnasium' not in frame_globals):
                return True
                
            # Check for legacy gym import patterns in locals
            if ('gym' in frame_locals and 
                hasattr(frame_locals.get('gym'), 'make')):
                return True
                
            frame = frame.f_back
            
        # Also check sys.modules for gym without gymnasium
        if 'gym' in sys.modules and 'gymnasium' not in sys.modules:
            # Additional check to ensure it's actually the legacy gym
            gym_module = sys.modules.get('gym')
            if gym_module and hasattr(gym_module, 'make'):
                # Check if it's the legacy gym by looking for new gymnasium features
                if not hasattr(gym_module, 'error') or not hasattr(gym_module.error, 'DeprecatedWarning'):
                    return True
                    
    except Exception:
        # If detection fails, default to False to avoid breaking functionality
        pass
    finally:
        # Clean up frame references to prevent memory leaks
        if 'frame' in locals():
            del frame
        
    return False

def _emit_legacy_gym_warning():
    """
    Emit deprecation warning for legacy gym usage.
    
    This function emits a comprehensive deprecation warning when legacy
    gym usage is detected, providing clear migration guidance while
    ensuring the warning is only shown once per session.
    """
    if not hasattr(_emit_legacy_gym_warning, '_warning_emitted'):
        warnings.warn(
            "\n" + "="*80 + "\n"
            "DEPRECATION WARNING: Legacy 'gym' package detected\n"
            "="*80 + "\n"
            "You are using the legacy 'gym' package which returns 4-tuple step() results.\n"
            "This usage pattern is deprecated and will be removed in v1.0.\n\n"
            "RECOMMENDED MIGRATION:\n"
            "- Replace 'import gym' with 'import gymnasium'\n"
            "- Update step() handling: (obs, reward, done, info) → (obs, reward, terminated, truncated, info)\n"
            "- Use new environment ID: 'PlumeNavSim-v0' instead of legacy IDs\n"
            "- Use shim layer: from plume_nav_sim.shims import gym_make\n\n"
            "CURRENT COMPATIBILITY:\n"
            "- Legacy gym API continues to work with existing environment IDs\n"
            "- 4-tuple step() returns are maintained for backward compatibility\n"
            "- All existing functionality remains unchanged\n\n"
            "For more information, see: https://gymnasium.farama.org/content/migration-guide/\n"
            "="*80,
            DeprecationWarning,
            stacklevel=3
        )
        _emit_legacy_gym_warning._warning_emitted = True

# Perform legacy gym detection on import
_legacy_gym_detected = _detect_legacy_gym_import()
if _legacy_gym_detected:
    _emit_legacy_gym_warning()

# =============================================================================
# GYMNASIUM ENVIRONMENT REGISTRATION
# =============================================================================

def _register_gymnasium_environments():
    """
    Register Gymnasium 0.29.x environments with proper entry points.
    
    Registers both the primary PlumeNavSim-v0 environment and the legacy
    OdorPlumeNavigation-v1 environment for backward compatibility.
    """
    try:
        import gymnasium
        from gymnasium.envs.registration import register
        
        # Register primary Gymnasium environment
        register(
            id='PlumeNavSim-v0',
            entry_point='plume_nav_sim.envs:PlumeNavigationEnv',
            max_episode_steps=1000,
            reward_threshold=500.0,
            kwargs={
                'api_version': 'gymnasium',
                'return_format': '5-tuple',
                'enable_extensibility_hooks': True,
            }
        )
        
        # Register legacy compatibility environment
        register(
            id='OdorPlumeNavigation-v1',
            entry_point='plume_nav_sim.envs:PlumeNavigationEnv',
            max_episode_steps=1000,
            reward_threshold=500.0,
            kwargs={
                'api_version': 'legacy',
                'return_format': '4-tuple',
                'enable_extensibility_hooks': False,
            }
        )
        
        return True
        
    except ImportError:
        # Gymnasium not available - environments will not be registered
        return False
    except Exception:
        # Registration failed - continue without breaking package import
        return False

# Attempt environment registration
_gymnasium_registered = _register_gymnasium_environments()

# =============================================================================
# CONDITIONAL IMPORTS AND API EXPORTS
# =============================================================================

# Core API functions - always available
try:
    from plume_nav_sim.api import (
        create_navigator,
        create_video_plume,
        run_plume_simulation,
        visualize_simulation_results,
    )
    _core_api_available = True
except ImportError:
    # Core API not available yet
    create_navigator = None
    create_video_plume = None
    run_plume_simulation = None
    visualize_simulation_results = None
    _core_api_available = False

# Enhanced API factory functions
try:
    from plume_nav_sim.api import (
        create_simulation_runner,
        create_batch_processor,
        run_experiment_sweep,
    )
    _enhanced_api_available = True
except ImportError:
    # Enhanced API not available
    create_simulation_runner = None
    create_batch_processor = None
    run_experiment_sweep = None
    _enhanced_api_available = False

# Core navigation components
try:
    from plume_nav_sim.core import (
        Navigator,
        SingleAgentController,
        MultiAgentController,
        NavigatorProtocol,
        run_simulation
    )
    _core_navigation_available = True
except ImportError:
    Navigator = None
    SingleAgentController = None
    MultiAgentController = None
    NavigatorProtocol = None
    run_simulation = None
    _core_navigation_available = False

# Environment components
try:
    from plume_nav_sim.envs import VideoPlume
    from plume_nav_sim.envs import PlumeNavigationEnv
    _env_components_available = True
except ImportError:
    VideoPlume = None
    PlumeNavigationEnv = None
    _env_components_available = False

# Configuration management
try:
    from plume_nav_sim.config import (
        NavigatorConfig, 
        SingleAgentConfig,
        MultiAgentConfig, 
        VideoPlumeConfig,
        load_config,
        save_config,
    )
    _config_available = True
except ImportError:
    NavigatorConfig = None
    SingleAgentConfig = None
    MultiAgentConfig = None
    VideoPlumeConfig = None
    load_config = None
    save_config = None
    _config_available = False

# Utility functions
try:
    from plume_nav_sim.utils import (
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
    _utils_available = True
except ImportError:
    # Utilities not available
    load_yaml = None
    save_yaml = None
    load_json = None
    save_json = None
    load_numpy = None
    save_numpy = None
    setup_logger = None
    get_module_logger = None
    DEFAULT_FORMAT = None
    MODULE_FORMAT = None
    LOG_LEVELS = None
    _utils_available = False

# Gymnasium and RL integration features
try:
    from plume_nav_sim.envs.gymnasium_env import GymnasiumEnv
    from plume_nav_sim.envs.spaces import (
        ActionSpace,
        ObservationSpace,
    )
    from plume_nav_sim.envs.wrappers import (
        NormalizationWrapper,
        FrameStackWrapper,
        RewardShapingWrapper,
    )
    _gymnasium_components_available = True
except ImportError:
    GymnasiumEnv = None
    ActionSpace = None
    ObservationSpace = None
    NormalizationWrapper = None
    FrameStackWrapper = None
    RewardShapingWrapper = None
    _gymnasium_components_available = False

# Gymnasium environment factory
try:
    from plume_nav_sim.api.navigation import create_gymnasium_environment
    _gymnasium_factory_available = True
except ImportError:
    create_gymnasium_environment = None
    _gymnasium_factory_available = False

# Shim compatibility layer
try:
    from plume_nav_sim.shims import gym_make
    _shim_available = True
except ImportError:
    gym_make = None
    _shim_available = False

# Check for stable-baselines3 availability
try:
    import stable_baselines3
    _stable_baselines3_available = True
except ImportError:
    _stable_baselines3_available = False

# Check for Gymnasium availability
try:
    import gymnasium
    _gymnasium_available = True
except ImportError:
    _gymnasium_available = False

# =============================================================================
# FEATURE AVAILABILITY MAPPING
# =============================================================================

FEATURES = {
    # Core functionality
    'core_api': _core_api_available,
    'enhanced_api': _enhanced_api_available,
    'core_navigation': _core_navigation_available,
    'environment_components': _env_components_available,
    'configuration': _config_available,
    'utilities': _utils_available,
    
    # Gymnasium and RL integration features
    'gymnasium_integration': _gymnasium_components_available and _gymnasium_registered,
    'gymnasium_env': _gymnasium_components_available,
    'gymnasium_factory': _gymnasium_factory_available,
    'stable_baselines3': _stable_baselines3_available,
    'vectorized_training': _stable_baselines3_available and _gymnasium_components_available,
    
    # Compatibility and migration features
    'shim_compatibility_layer': _shim_available,
    'legacy_gym_detected': _legacy_gym_detected,
    'dual_api_support': _shim_available,  # Available when shims work
    'gymnasium_0_29_compliance': _gymnasium_available and _gymnasium_registered,
    
    # Migration and deprecation features
    'deprecation_warnings': True,  # Always available
    'migration_guidance': True,    # Always available
    'backward_compatibility': _shim_available,
}

def get_available_features():
    """
    Get a dictionary of available feature flags.
    
    Returns:
        dict: Feature availability status with detailed boolean flags
        
    Examples:
        >>> from plume_nav_sim import get_available_features
        >>> features = get_available_features()
        >>> print(f"Gymnasium integration: {features['gymnasium_integration']}")
        >>> print(f"Legacy compatibility: {features['backward_compatibility']}")
    """
    return FEATURES.copy()

def is_feature_available(feature_name):
    """
    Check if a specific feature is available.
    
    Args:
        feature_name (str): Name of the feature to check
        
    Returns:
        bool: True if feature is available, False otherwise
        
    Examples:
        >>> from plume_nav_sim import is_feature_available
        >>> if is_feature_available('gymnasium_integration'):
        ...     print("Gymnasium integration is available")
    """
    return FEATURES.get(feature_name, False)

# =============================================================================
# API COMPATIBILITY AND MIGRATION FUNCTIONS
# =============================================================================

def check_api_compatibility():
    """
    Check API compatibility and provide migration guidance.
    
    This function analyzes the current environment setup and provides
    detailed information about API compatibility, detected usage patterns,
    and migration recommendations.
    
    Returns:
        dict: Comprehensive API compatibility report
        
    Examples:
        >>> from plume_nav_sim import check_api_compatibility
        >>> report = check_api_compatibility()
        >>> print(f"Legacy gym detected: {report['legacy_gym_detected']}")
        >>> print(f"Recommended action: {report['recommendation']}")
    """
    legacy_detected = FEATURES.get('legacy_gym_detected', False)
    gymnasium_available = FEATURES.get('gymnasium_0_29_compliance', False)
    
    report = {
        'legacy_gym_detected': legacy_detected,
        'gymnasium_available': gymnasium_available,
        'dual_api_support': FEATURES.get('dual_api_support', False),
        'current_api_version': 'legacy_gym' if legacy_detected else 'gymnasium',
        'recommendation': None,
        'migration_steps': [],
        'compatibility_status': 'compatible'
    }
    
    if legacy_detected:
        report['recommendation'] = 'migrate_to_gymnasium'
        report['migration_steps'] = [
            "Replace 'import gym' with 'import gymnasium'",
            "Update environment creation to use 'PlumeNavSim-v0'",
            "Modify step() handling: (obs, reward, done, info) → (obs, reward, terminated, truncated, info)",
            "Update reset() calls to handle new seed parameter",
            "Test with gymnasium.utils.env_checker for validation",
            "Use shim layer temporarily: from plume_nav_sim.shims import gym_make"
        ]
        report['compatibility_status'] = 'deprecated_but_functional'
    elif gymnasium_available:
        report['recommendation'] = 'use_modern_api'
        report['compatibility_status'] = 'optimal'
    else:
        report['recommendation'] = 'install_gymnasium'
        report['migration_steps'] = [
            "Install gymnasium: pip install 'gymnasium>=0.29.0'",
            "Follow migration steps above"
        ]
        report['compatibility_status'] = 'missing_dependencies'
    
    return report

def warn_if_legacy_env_usage(env_id):
    """
    Check and warn about legacy environment ID usage.
    
    This function detects when legacy environment IDs are being used
    and provides specific warnings with migration guidance for each
    legacy environment identifier.
    
    Args:
        env_id (str): Environment ID being accessed
        
    Returns:
        str: Recommended modern environment ID, or original if already modern
        
    Examples:
        >>> modern_id = warn_if_legacy_env_usage('OdorPlumeNavigation-v1')
        >>> # Emits deprecation warning and returns 'PlumeNavSim-v0'
    """
    legacy_env_mappings = {
        'OdorPlumeNavigation-v0': 'PlumeNavSim-v0',
        'OdorPlumeNavigation-v1': 'PlumeNavSim-v0', 
        'OdorPlumeNav-v0': 'PlumeNavSim-v0',
        'OdorPlumeNav-v1': 'PlumeNavSim-v0',
    }
    
    if env_id in legacy_env_mappings:
        modern_id = legacy_env_mappings[env_id]
        warnings.warn(
            f"Legacy environment ID '{env_id}' is deprecated. "
            f"Use '{modern_id}' for Gymnasium 0.29.x compatibility and new features. "
            f"Legacy ID will continue to work but may be removed in v1.0.",
            DeprecationWarning,
            stacklevel=3
        )
        return modern_id
    
    return env_id

def get_api_migration_guide():
    """
    Get comprehensive API migration guide for legacy users.
    
    This function provides detailed migration guidance for users
    transitioning from legacy gym to modern Gymnasium APIs, including
    code examples and common migration patterns.
    
    Returns:
        dict: Comprehensive migration guide with examples and patterns
        
    Examples:
        >>> guide = get_api_migration_guide()
        >>> print(guide['step_handling']['before'])
        >>> print(guide['step_handling']['after'])
    """
    return {
        'imports': {
            'before': 'import gym',
            'after': 'import gymnasium',
            'explanation': 'Replace legacy gym with modern gymnasium package'
        },
        'environment_creation': {
            'before': "env = gym.make('OdorPlumeNavigation-v1')",
            'after': "env = gymnasium.make('PlumeNavSim-v0')",
            'explanation': 'Use new environment ID with gymnasium.make()'
        },
        'reset_handling': {
            'before': 'obs = env.reset()',
            'after': 'obs, info = env.reset(seed=42)',
            'explanation': 'Modern reset() returns (obs, info) tuple and accepts seed parameter'
        },
        'step_handling': {
            'before': 'obs, reward, done, info = env.step(action)',
            'after': 'obs, reward, terminated, truncated, info = env.step(action)',
            'explanation': 'Modern step() returns 5-tuple with separate terminated/truncated flags'
        },
        'done_flag_migration': {
            'before': 'if done: ...',
            'after': 'if terminated or truncated: ...',
            'explanation': 'Replace single done flag with terminated OR truncated logic'
        },
        'shim_usage': {
            'temporary': 'from plume_nav_sim.shims import gym_make; env = gym_make("PlumeNavSim-v0")',
            'explanation': 'Use shim layer for gradual migration while maintaining legacy patterns'
        },
        'compatibility_note': (
            'The plume_nav_sim package maintains backward compatibility. '
            'Legacy patterns continue to work but emit deprecation warnings. '
            'Migration is recommended for new code and future compatibility.'
        )
    }

def _setup_runtime_api_monitoring():
    """
    Set up runtime monitoring for API usage patterns.
    
    This function configures runtime monitoring to detect and warn about
    legacy API usage patterns as they occur, providing just-in-time
    migration guidance while maintaining full backward compatibility.
    
    Note: This is called automatically on package import.
    """
    # Monitor for legacy environment creation patterns
    if _legacy_gym_detected:
        # Additional runtime setup for legacy gym detection
        # This ensures warnings are contextual and helpful
        def _cleanup_legacy_monitoring():
            """Clean up legacy monitoring on exit."""
            if hasattr(_emit_legacy_gym_warning, '_warning_emitted'):
                # Optional: log final migration reminder
                pass
        
        atexit.register(_cleanup_legacy_monitoring)

# Initialize runtime API monitoring
_setup_runtime_api_monitoring()

# =============================================================================
# DYNAMIC EXPORTS BASED ON FEATURE AVAILABILITY
# =============================================================================

# Build dynamic __all__ list based on available features
_base_exports = [
    # Version and metadata
    '__version__',
    
    # Feature availability functions
    'get_available_features',
    'is_feature_available',
    'FEATURES',
    
    # API compatibility and migration functions
    'check_api_compatibility',
    'warn_if_legacy_env_usage',
    'get_api_migration_guide',
]

# Add core API exports when available
if _core_api_available:
    _base_exports.extend([
        'create_navigator',
        'create_video_plume', 
        'run_plume_simulation',
        'visualize_simulation_results',
    ])

# Add enhanced API exports when available
if _enhanced_api_available:
    _base_exports.extend([
        'create_simulation_runner',
        'create_batch_processor',
        'run_experiment_sweep',
    ])

# Add core navigation exports when available
if _core_navigation_available:
    _base_exports.extend([
        'Navigator',
        'SingleAgentController',
        'MultiAgentController', 
        'NavigatorProtocol',
        'run_simulation',
    ])

# Add environment exports when available
if _env_components_available:
    _base_exports.extend([
        'VideoPlume',
        'PlumeNavigationEnv',
    ])

# Add configuration exports when available
if _config_available:
    _base_exports.extend([
        'NavigatorConfig',
        'SingleAgentConfig', 
        'MultiAgentConfig',
        'VideoPlumeConfig',
        'load_config',
        'save_config',
    ])

# Add utility exports when available
if _utils_available:
    _base_exports.extend([
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
    ])

# Add Gymnasium-specific exports when available
_gymnasium_exports = []
if _gymnasium_components_available:
    _gymnasium_exports.extend([
        'GymnasiumEnv',
        'ActionSpace',
        'ObservationSpace',
        'NormalizationWrapper',
        'FrameStackWrapper',
        'RewardShapingWrapper',
    ])

if _gymnasium_factory_available:
    _gymnasium_exports.append('create_gymnasium_environment')

# Add shim compatibility exports when available
if _shim_available:
    _gymnasium_exports.append('gym_make')

# Define public API for wildcard imports
__all__ = _base_exports + _gymnasium_exports

# =============================================================================
# PACKAGE INITIALIZATION SUMMARY
# =============================================================================
#
# The plume_nav_sim package (v0.3.0) implements a comprehensive migration
# strategy from legacy Gym 0.26 to modern Gymnasium 0.29.x while maintaining
# complete backward compatibility and providing clear migration guidance.
#
# KEY FEATURES:
# - Automatic legacy gym detection via stack introspection
# - Gymnasium 0.29.x environment registration (PlumeNavSim-v0, OdorPlumeNavigation-v1)
# - Comprehensive deprecation warning system with migration guidance
# - Dual API support through compatibility shim layer
# - Enhanced frame caching with configurable modes and memory management
# - Extensibility hooks for custom observations, rewards, and episode handling
#
# ENVIRONMENT REGISTRATION:
# - PlumeNavSim-v0: Primary Gymnasium environment with 5-tuple returns
# - OdorPlumeNavigation-v1: Legacy compatibility environment with 4-tuple returns
#
# MIGRATION SUPPORT:
# - Automatic detection and warnings for legacy usage patterns
# - Comprehensive migration guide with code examples
# - Compatibility shim: plume_nav_sim.shims.gym_make
# - Feature flags for runtime capability detection
#
# BACKWARD COMPATIBILITY:
# - Zero breaking changes for existing code
# - Legacy environment IDs continue to work
# - 4-tuple step() returns maintained for legacy callers
# - Graceful degradation when dependencies unavailable
#
# =============================================================================