"""
Configuration module for plume_nav_sim.

This module provides configuration schemas and utilities for the
plume navigation simulation system.
"""

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
    "resolve_env_value"
]