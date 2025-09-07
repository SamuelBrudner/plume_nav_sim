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
from pathlib import Path
from importlib.metadata import PackageNotFoundError, distribution

try:  # Prefer Loguru when available
    from loguru import logger as _logger
    _LOGGER_IS_STUB = not hasattr(_logger, "configure")
    logger = _logger
except Exception:  # pragma: no cover - defensive fallback
from loguru import logger
    logger = _logger
    _LOGGER_IS_STUB = True


def _configure_logger() -> None:
    """Configure logging from logger.yaml or warn if running in stub mode."""
    if _LOGGER_IS_STUB:
        try:
            logger.warning(
                "Logging bootstrap not available; running in limited mode with lightweight stubs."
            )
        except Exception:  # pragma: no cover - safety
            pass
        return

    from .utils.logging_setup import setup_logger

    config_path = Path(__file__).resolve().parents[2] / "logger.yaml"
    setup_logger(logging_config_path=config_path)


_configure_logger()

__version__ = "1.0.0"

# =============================================================================
# INSTALLATION VALIDATION
# =============================================================================
try:  # pragma: no cover - exercised in tests via monkeypatch
    distribution("plume_nav_sim")
except PackageNotFoundError as e:  # pragma: no cover - executed when not installed
    msg = (
        "plume_nav_sim must be installed before use. "
        "Run 'pip install -e .' or './setup_env.sh --dev'."
    )
    logger.error(msg)
    raise ImportError(msg) from e

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
# Suppress warnings during test execution to prevent spam
import os
import sys
_is_testing = any(x in sys.modules for x in ['pytest', '_pytest']) or 'PYTEST_RUNNING' in os.environ
if _legacy_gym_detected and not _is_testing:
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
    import gymnasium
    from gymnasium.envs.registration import register

    env_kwargs = {
        'api_version': 'gymnasium',
        'return_format': '5-tuple',
        'enable_extensibility_hooks': True,
    }

    # Register primary Gymnasium environment
    register(
        id='PlumeNavSim-v0',
        entry_point='plume_nav_sim.envs:PlumeNavigationEnv',
        max_episode_steps=1000,
        reward_threshold=500.0,
        kwargs=dict(env_kwargs),
    )
    logger.info(
        "Registered PlumeNavSim-v0 environment",
        extra={"metric_type": "environment_registration", "env_id": "PlumeNavSim-v0"},
    )

    # Register snake_case alias
    register(
        id='plume_nav_sim_v0',
        entry_point='plume_nav_sim.envs:PlumeNavigationEnv',
        max_episode_steps=1000,
        reward_threshold=500.0,
        kwargs=dict(env_kwargs),
    )
    logger.info(
        "Registered plume_nav_sim_v0 alias for PlumeNavSim-v0",
        extra={
            "metric_type": "environment_registration",
            "env_id": "plume_nav_sim_v0",
            "alias_for": "PlumeNavSim-v0",
        },
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

# Attempt environment registration
_register_gymnasium_environments()
_gymnasium_registered = True

# =============================================================================
# CONDITIONAL IMPORTS AND API EXPORTS
# =============================================================================

# Core API functions - optional during testing
try:
    from plume_nav_sim.api import (
        create_navigator,
        create_video_plume,
        run_plume_simulation,
        visualize_simulation_results,
    )
    _core_api_available = True
except Exception as e:  # pragma: no cover - optional
    _core_api_available = False
    create_navigator = create_video_plume = run_plume_simulation = visualize_simulation_results = None  # type: ignore
    logger.error("Failed to import core API functions: %s", e)

# Enhanced API factory functions
try:
    from plume_nav_sim.api import (
        create_simulation_runner,
        create_batch_processor,
        run_experiment_sweep,
    )
    _enhanced_api_available = True
except Exception as e:  # pragma: no cover - optional
    _enhanced_api_available = False
    create_simulation_runner = create_batch_processor = run_experiment_sweep = None  # type: ignore
    logger.error("Failed to import enhanced API factory functions: %s", e)

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
except Exception as e:  # pragma: no cover - optional
    _core_navigation_available = False
    Navigator = SingleAgentController = MultiAgentController = NavigatorProtocol = run_simulation = None  # type: ignore
    logger.error("Failed to import core navigation components: %s", e)

# New v1.0 protocol interfaces for pluggable architecture
try:
    from plume_nav_sim.core.protocols import (
        SourceProtocol,
        BoundaryPolicyProtocol,
        ActionInterfaceProtocol,
        RecorderProtocol,
        StatsAggregatorProtocol,
    )
    _v1_protocols_available = True
except Exception as e:  # pragma: no cover - optional
    _v1_protocols_available = False
    SourceProtocol = BoundaryPolicyProtocol = ActionInterfaceProtocol = RecorderProtocol = StatsAggregatorProtocol = None  # type: ignore
    logger.error("Failed to import protocol interfaces: %s", e)

# Environment components
try:
    from plume_nav_sim.envs import VideoPlume
    from plume_nav_sim.envs import PlumeNavigationEnv
    _env_components_available = True
except Exception as e:  # pragma: no cover - optional
    _env_components_available = False
    VideoPlume = PlumeNavigationEnv = None  # type: ignore
    logger.error("Failed to import environment components: %s", e)

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
except Exception as e:  # pragma: no cover - optional
    _config_available = False
    NavigatorConfig = SingleAgentConfig = MultiAgentConfig = VideoPlumeConfig = load_config = save_config = None  # type: ignore
    logger.error("Failed to import configuration management components: %s", e)

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
except Exception as e:  # pragma: no cover - optional
    _utils_available = False
    load_yaml = save_yaml = load_json = save_json = load_numpy = save_numpy = None  # type: ignore
    setup_logger = get_module_logger = DEFAULT_FORMAT = MODULE_FORMAT = LOG_LEVELS = None  # type: ignore
    logger.error("Failed to import utility functions: %s", e)

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
except Exception as e:  # pragma: no cover - optional
    _gymnasium_components_available = False
    GymnasiumEnv = ActionSpace = ObservationSpace = NormalizationWrapper = FrameStackWrapper = RewardShapingWrapper = None  # type: ignore
    logger.error("Failed to import Gymnasium integration features: %s", e)

# Gymnasium environment factory
try:
    from plume_nav_sim.api.navigation import create_gymnasium_environment
    _gymnasium_factory_available = True
except Exception as e:  # pragma: no cover - optional
    _gymnasium_factory_available = False
    create_gymnasium_environment = None  # type: ignore
    logger.error("Failed to import Gymnasium environment factory: %s", e)

# Shim compatibility layer
try:
    from plume_nav_sim.shims import gym_make
    _shim_available = True
except Exception as e:  # pragma: no cover - optional
    _shim_available = False
    gym_make = None  # type: ignore
    logger.error("Failed to import shim compatibility layer: %s", e)

# Recording framework components for v1.0 architecture
try:
    from plume_nav_sim.recording import (
        BaseRecorder,
        RecorderFactory,
        RecorderManager
    )
    _recording_components_available = True
except Exception as e:  # pragma: no cover - optional
    _recording_components_available = False
    BaseRecorder = RecorderFactory = RecorderManager = None  # type: ignore
    logger.error("Failed to import recording framework components: %s", e)

# Analysis framework components for v1.0 architecture
try:
    from plume_nav_sim.analysis import (
        StatsAggregator,
        generate_summary
    )
    _analysis_components_available = True
except Exception as e:  # pragma: no cover - optional
    _analysis_components_available = False
    StatsAggregator = generate_summary = None  # type: ignore
    logger.error("Failed to import analysis framework components: %s", e)

# Debug framework components for v1.0 architecture
try:
    from plume_nav_sim.debug import (
        DebugGUI,
        plot_initial_state
    )
    _debug_components_available = True
except Exception as e:  # pragma: no cover - optional
    _debug_components_available = False
    DebugGUI = plot_initial_state = None  # type: ignore
    logger.error("Failed to import debug framework components: %s", e)

# Check for stable-baselines3 availability
try:
    import stable_baselines3
    _stable_baselines3_available = True
except Exception as e:  # pragma: no cover - optional
    _stable_baselines3_available = False
    logger.error("stable-baselines3 is required but failed to import: %s", e)

# Check for Gymnasium availability
try:
    import gymnasium
    _gymnasium_available = True
except Exception as e:  # pragma: no cover - optional
    _gymnasium_available = False
    logger.error("Gymnasium is required but failed to import: %s", e)

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
    
    # New v1.0 pluggable component architecture
    'pluggable_components': _v1_protocols_available,
    'protocol_based_architecture': _v1_protocols_available,
    'source_protocol': _v1_protocols_available,
    'boundary_policy_protocol': _v1_protocols_available,
    'action_interface_protocol': _v1_protocols_available,
    'zero_code_extensibility': _v1_protocols_available,
    
    # v1.0 recording framework
    'recording_framework': _recording_components_available,
    'multi_backend_recording': _recording_components_available,
    'parquet_backend': _recording_components_available,
    'hdf5_backend': _recording_components_available,
    'sqlite_backend': _recording_components_available,
    'data_compression': _recording_components_available,
    'performance_monitoring': _recording_components_available,
    
    # v1.0 analysis framework  
    'statistics_aggregation': _analysis_components_available,
    'automated_metrics': _analysis_components_available,
    'research_summaries': _analysis_components_available,
    'summary_generation': _analysis_components_available,
    'reproducible_analysis': _analysis_components_available,
    
    # v1.0 interactive debugging
    'interactive_debugging': _debug_components_available,
    'step_through_debugging': _debug_components_available,
    'qt_debug_gui': _debug_components_available,
    'streamlit_debug_gui': _debug_components_available,
    'initial_state_visualization': _debug_components_available,
    'debug_screenshots': _debug_components_available,
    
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

# Add new v1.0 protocol interface exports when available
if _v1_protocols_available:
    _base_exports.extend([
        'SourceProtocol',
        'BoundaryPolicyProtocol',
        'ActionInterfaceProtocol',
        'RecorderProtocol',
        'StatsAggregatorProtocol',
    ])

# Add recording framework exports when available
if _recording_components_available:
    _base_exports.extend([
        'BaseRecorder',
        'RecorderFactory',
        'RecorderManager',
    ])

# Add analysis framework exports when available
if _analysis_components_available:
    _base_exports.extend([
        'StatsAggregator',
        'generate_summary',
    ])

# Add debug framework exports when available
if _debug_components_available:
    _base_exports.extend([
        'DebugGUI',
        'plot_initial_state',
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
# The plume_nav_sim package (v1.0.0) implements a comprehensive transformation
# from a project-specific implementation into a general-purpose, extensible 
# simulation toolkit that serves as the definitive backbone for odor plume
# navigation research across the ecosystem.
#
# KEY v1.0 FEATURES:
# - Protocol-based pluggable component architecture with zero-code extensibility
# - Advanced recording framework with multi-backend support (parquet, HDF5, SQLite)
# - Automated statistics collection with research-focused metrics calculation
# - Interactive debugging tools with Qt/Streamlit GUI and step-through capabilities
# - Enhanced performance achieving ≤33ms per simulation step with 100+ agents
# - Scientific reproducibility through deterministic seeding and configuration snapshots
# - Comprehensive backward compatibility maintaining legacy Gym and v0.3.0 APIs
#
# PROTOCOL-BASED ARCHITECTURE:
# - SourceProtocol: Pluggable odor source implementations (PointSource, MultiSource, DynamicSource)
# - BoundaryPolicyProtocol: Configurable domain edge handling (terminate, bounce, wrap, clip)
# - ActionInterfaceProtocol: Standardized action space translation for RL frameworks
# - RecorderProtocol: Data persistence with performance-aware buffering and compression
# - StatsAggregatorProtocol: Automated research metrics with standardized summary generation
#
# ADVANCED RECORDING FRAMEWORK:
# - Multi-backend support: parquet (columnar), HDF5 (hierarchical), SQLite (transactional)
# - Performance-optimized with <1ms overhead when disabled, async I/O when enabled
# - Structured output organization with run_id/episode_id hierarchical directories
# - Data compression and validation with configurable quality and performance tradeoffs
#
# INTERACTIVE DEBUGGING SYSTEM:
# - Qt-based debug GUI for step-through debugging and real-time visualization
# - Streamlit web interface for browser-based debugging and collaborative sessions
# - Initial state visualization showing source location, boundaries, and agent positions
# - Screenshot export capabilities for documentation and presentation
#
# ENVIRONMENT REGISTRATION:
# - PlumeNavSim-v0: Primary Gymnasium environment with 5-tuple returns and v1.0 features
# - OdorPlumeNavigation-v1: Legacy compatibility environment with 4-tuple returns
#
# MIGRATION SUPPORT:
# - Automatic detection and warnings for legacy usage patterns
# - Comprehensive migration guide with code examples for v0.3.0 → v1.0 transition
# - Compatibility shim: plume_nav_sim.shims.gym_make for gradual migration
# - Feature flags for runtime capability detection and graceful degradation
#
# BACKWARD COMPATIBILITY:
# - Zero breaking changes for existing v0.3.0 code
# - Legacy environment IDs and APIs continue to work with deprecation guidance
# - 4-tuple step() returns maintained for legacy callers
# - Graceful degradation when optional v1.0 dependencies unavailable
#
# =============================================================================