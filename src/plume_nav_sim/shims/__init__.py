"""
Compatibility Shims Module for Legacy Gym API Support (plume_nav_sim v0.3.0)

This module provides backward compatibility for projects using legacy OpenAI Gym APIs,
enabling seamless migration to modern Gymnasium 0.29.x while maintaining full backward
compatibility. The shims in this module proxy legacy gym.make() calls to gymnasium.make()
with automatic detection and conversion between 4-tuple and 5-tuple step/reset formats.

MIGRATION PATH:
This compatibility layer is designed as a stepping stone to help projects migrate from
legacy gym to modern gymnasium. While the shims ensure existing code continues to work
without modification, users are strongly encouraged to migrate to native Gymnasium APIs
for optimal performance and future compatibility.

EXAMPLE USAGE:
    # Legacy compatibility (with deprecation warning)
    from plume_nav_sim.shims import gym_make
    env = gym_make("PlumeNavSim-v0")  # Proxies to gymnasium internally
    
    # Recommended modern approach
    import gymnasium
    env = gymnasium.make("PlumeNavSim-v0")

DEPRECATION TIMELINE:
- v0.3.0: Shims introduced for compatibility
- v0.4.0: Enhanced migration guidance and warnings
- v1.0.0: Shims will be removed (full gymnasium-only)

For complete migration guidance, see the package documentation and use
plume_nav_sim.get_api_migration_guide() for detailed migration steps.
"""

import warnings

# =============================================================================
# COMPATIBILITY FUNCTION IMPORTS
# =============================================================================

try:
    # Import the primary gym_make compatibility function
    from plume_nav_sim.shims.gym_make import gym_make
    _gym_make_available = True
    
except ImportError as e:
    # gym_make.py not available - provide a fallback with clear error
    warnings.warn(
        f"Failed to import gym_make compatibility function: {e}. "
        "Legacy gym.make() compatibility is not available. "
        "Please use gymnasium.make() directly for environment creation.",
        ImportWarning,
        stacklevel=2
    )
    
    def gym_make(*args, **kwargs):
        """
        Fallback gym_make function when compatibility layer is unavailable.
        
        This function provides a clear error message when the main gym_make
        compatibility function cannot be imported, guiding users to use
        the modern gymnasium.make() API directly.
        """
        raise ImportError(
            "The gym_make compatibility function is not available. "
            "This typically means the gym_make.py module is missing or has import errors. "
            "Please use gymnasium.make() directly:\n\n"
            "  import gymnasium\n"
            "  env = gymnasium.make('PlumeNavSim-v0')\n\n"
            "For migration guidance, see plume_nav_sim.get_api_migration_guide()"
        )
    
    _gym_make_available = False

# =============================================================================
# SHIMS MODULE FEATURE DETECTION
# =============================================================================

def is_shim_available():
    """
    Check if the compatibility shim layer is fully available.
    
    Returns:
        bool: True if gym_make and related compatibility functions are available
        
    Examples:
        >>> from plume_nav_sim.shims import is_shim_available
        >>> if is_shim_available():
        ...     print("Legacy compatibility layer is available")
        ... else:
        ...     print("Use gymnasium.make() directly")
    """
    return _gym_make_available

def get_shim_status():
    """
    Get detailed status information about the compatibility shim layer.
    
    Returns:
        dict: Detailed status including availability, version info, and recommendations
        
    Examples:
        >>> from plume_nav_sim.shims import get_shim_status
        >>> status = get_shim_status()
        >>> print(f"Shim available: {status['available']}")
        >>> print(f"Recommendation: {status['recommendation']}")
    """
    return {
        'available': _gym_make_available,
        'module_version': '0.3.0',
        'compatibility_target': 'OpenAI Gym 0.26.x → Gymnasium 0.29.x',
        'deprecation_status': 'active_with_warnings',
        'removal_timeline': 'v1.0.0',
        'recommendation': 'migrate_to_gymnasium' if _gym_make_available else 'use_gymnasium_directly',
        'migration_guide_available': True,
    }

# =============================================================================
# DEPRECATION AND MIGRATION GUIDANCE
# =============================================================================

def emit_shim_usage_warning():
    """
    Emit a comprehensive deprecation warning for shim usage.
    
    This function provides detailed migration guidance and is called automatically
    when shim functions are used. It includes specific code examples and migration
    steps to help users transition to modern Gymnasium APIs.
    """
    if not hasattr(emit_shim_usage_warning, '_warning_emitted'):
        warnings.warn(
            "\n" + "="*80 + "\n"
            "COMPATIBILITY SHIM USAGE DETECTED\n"
            "="*80 + "\n"
            "You are using the legacy compatibility shim layer. While this maintains\n"
            "backward compatibility, it is deprecated and will be removed in v1.0.\n\n"
            "RECOMMENDED MIGRATION STEPS:\n"
            "1. Replace shim imports:\n"
            "   BEFORE: from plume_nav_sim.shims import gym_make\n"
            "   AFTER:  import gymnasium\n\n"
            "2. Update environment creation:\n"
            "   BEFORE: env = gym_make('PlumeNavSim-v0')\n"
            "   AFTER:  env = gymnasium.make('PlumeNavSim-v0')\n\n"
            "3. Update step() handling:\n"
            "   BEFORE: obs, reward, done, info = env.step(action)\n"
            "   AFTER:  obs, reward, terminated, truncated, info = env.step(action)\n\n"
            "4. Update reset() handling:\n"
            "   BEFORE: obs = env.reset()\n"
            "   AFTER:  obs, info = env.reset(seed=42)\n\n"
            "MIGRATION RESOURCES:\n"
            "- Use plume_nav_sim.get_api_migration_guide() for detailed guidance\n"
            "- See https://gymnasium.farama.org/content/migration-guide/\n"
            "- Use gymnasium.utils.env_checker for validation\n"
            "="*80,
            DeprecationWarning,
            stacklevel=3
        )
        emit_shim_usage_warning._warning_emitted = True

# =============================================================================
# PUBLIC API EXPORTS
# =============================================================================

# Define public exports for the shims module
__all__ = [
    # Primary compatibility function
    'gym_make',
    
    # Shim status and feature detection
    'is_shim_available',
    'get_shim_status',
    
    # Migration guidance
    'emit_shim_usage_warning',
]

# =============================================================================
# MODULE INITIALIZATION SUMMARY
# =============================================================================
#
# The plume_nav_sim.shims module provides backward compatibility for legacy
# OpenAI Gym APIs while facilitating migration to modern Gymnasium 0.29.x.
#
# KEY COMPONENTS:
# - gym_make(): Primary compatibility function proxying to gymnasium.make()
# - Automatic 4-tuple ↔ 5-tuple conversion for step()/reset() returns
# - Comprehensive deprecation warnings with migration guidance
# - Feature detection for runtime capability assessment
#
# DESIGN PRINCIPLES:
# - Zero breaking changes for existing legacy code
# - Clear migration path with detailed guidance
# - Graceful degradation when dependencies unavailable
# - Comprehensive warnings without disrupting functionality
#
# DEPRECATION STRATEGY:
# - v0.3.0: Shims introduced with active deprecation warnings
# - v0.4.0: Enhanced migration tooling and guidance
# - v1.0.0: Complete removal of compatibility layer
#
# USAGE PATTERNS:
# - Temporary compatibility: Use shims for immediate compatibility
# - Gradual migration: Replace shim usage incrementally with native gymnasium
# - New development: Use gymnasium.make() directly without shims
#
# =============================================================================