"""
Configuration module for plume_nav_sim.

This module provides configuration schemas and utilities for the
plume navigation simulation system.
"""

from pathlib import Path
from .schemas import (
    NavigatorConfig,
    SingleAgentConfig,
    MultiAgentConfig,
    VideoPlumeConfig,
    SimulationConfig,
    cs
)

from .utils import (
    validate_config,
    load_environment_variables,
    initialize_hydra_config_store,
    compose_config_from_overrides,
    create_default_config,
    get_config_schema,
    register_config_schemas,
    validate_env_interpolation,
    resolve_env_value
)

__version__ = "1.0.0"
__all__ = [
    # Configuration schemas
    "NavigatorConfig",
    "SingleAgentConfig",
    "MultiAgentConfig",
    "VideoPlumeConfig", 
    "SimulationConfig",
    "cs",
    
    # Utility functions
    "validate_config",
    "load_environment_variables",
    "initialize_hydra_config_store",
    "compose_config_from_overrides",
    "create_default_config",
    "get_config_schema",
    "register_config_schemas",
    "validate_env_interpolation",
    "resolve_env_value",
    # dotenv helper
    "find_dotenv",
    # Expose Path for tests that monkey-patch it
    "Path",
]

# --------------------------------------------------------------------------- #
# Minimal shim for python-dotenv compatibility used in test-suite
# --------------------------------------------------------------------------- #

def find_dotenv(*args, **kwargs):  # noqa: D401, E501  (simple stub, no docstring needed)
    """
    Dummy implementation of python-dotenv's `find_dotenv`.

    The real implementation searches parent directories for a ``.env`` file.
    For testing purposes we only need the symbol to exist so that it can be
    monkey-patched.  Returning an empty string is sufficient and side-effect-free.
    """
    return ""