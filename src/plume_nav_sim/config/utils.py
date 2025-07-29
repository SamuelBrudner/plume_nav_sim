"""
Configuration utilities for plume_nav_sim.

This module provides utility functions for configuration validation and management.
"""

import os
from typing import Dict, Any, Optional, Union, Type
from pydantic import BaseModel, ValidationError

try:
    from hydra.core.config_store import ConfigStore
    from hydra import initialize, compose
    from omegaconf import DictConfig, OmegaConf
    HAS_HYDRA = True
except ImportError:
    HAS_HYDRA = False
    DictConfig = dict
    OmegaConf = None

from .schemas import NavigatorConfig, SingleAgentConfig, MultiAgentConfig, VideoPlumeConfig, SimulationConfig


def validate_config(config_data: Union[Dict[str, Any], DictConfig], config_class: Type[BaseModel]) -> BaseModel:
    """
    Validate configuration data against a Pydantic model.
    
    Args:
        config_data: Configuration data to validate
        config_class: Pydantic model class to validate against
        
    Returns:
        Validated configuration instance
        
    Raises:
        ValidationError: If validation fails
    """
    if isinstance(config_data, DictConfig) and HAS_HYDRA:
        config_data = OmegaConf.to_container(config_data, resolve=True)
    
    return config_class(**config_data)


def load_environment_variables(prefix: str = "PLUME_NAV_") -> Dict[str, str]:
    """
    Load environment variables with a given prefix.
    
    Args:
        prefix: Environment variable prefix to filter by
        
    Returns:
        Dictionary of environment variables (without prefix)
    """
    env_vars = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            env_key = key[len(prefix):].lower()
            env_vars[env_key] = value
    return env_vars


def initialize_hydra_config_store() -> Optional[object]:
    """
    Initialize Hydra ConfigStore with default schemas.
    
    Returns:
        ConfigStore instance if Hydra is available, None otherwise
    """
    if not HAS_HYDRA:
        return None
        
    cs = ConfigStore.instance()
    
    # Register config schemas
    cs.store(name="navigator_config", node=NavigatorConfig)
    cs.store(name="single_agent_config", node=SingleAgentConfig)
    cs.store(name="multi_agent_config", node=MultiAgentConfig)
    cs.store(name="video_plume_config", node=VideoPlumeConfig)
    cs.store(name="simulation_config", node=SimulationConfig)
    
    return cs


def compose_config_from_overrides(config_name: str = "config", overrides: Optional[list] = None) -> DictConfig:
    """
    Compose configuration from Hydra with overrides.
    
    Args:
        config_name: Name of the config to compose
        overrides: List of override strings
        
    Returns:
        Composed configuration
    """
    if not HAS_HYDRA:
        raise ImportError("Hydra is required for config composition")
        
    if overrides is None:
        overrides = []
        
    with initialize(version_base=None, config_path="../../../conf"):
        cfg = compose(config_name=config_name, overrides=overrides)
    
    return cfg


def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration dictionary.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "navigator": SingleAgentConfig().model_dump(),
        "video_plume": VideoPlumeConfig(video_path="default.mp4").model_dump(),
        "simulation": SimulationConfig().model_dump()
    }


def get_config_schema(schema_name: str) -> Optional[Type[BaseModel]]:
    """
    Get a configuration schema by name.
    
    Args:
        schema_name: Name of the schema to retrieve
        
    Returns:
        Configuration schema class or None if not found
    """
    schemas = {
        "navigator": NavigatorConfig,
        "single_agent": SingleAgentConfig,
        "multi_agent": MultiAgentConfig,
        "video_plume": VideoPlumeConfig,
        "simulation": SimulationConfig
    }
    
    return schemas.get(schema_name)


def register_config_schemas():
    """Register all configuration schemas with Hydra ConfigStore."""
    initialize_hydra_config_store()


def validate_env_interpolation(value: str) -> bool:
    """
    Validate environment variable interpolation syntax.
    
    Args:
        value: String value to validate
        
    Returns:
        True if valid interpolation syntax, False otherwise
    """
    if not isinstance(value, str):
        return True
        
    # Check for ${env:VAR_NAME} pattern
    import re
    pattern = r'\$\{env:([A-Z_][A-Z0-9_]*)\}'
    return bool(re.search(pattern, value))


def resolve_env_value(value: str, default: str = "") -> str:
    """
    Resolve environment variable references in a string.
    
    Args:
        value: String with potential environment variable references
        default: Default value if environment variable is not set
        
    Returns:
        Resolved string value
    """
    if not isinstance(value, str):
        return str(value)
        
    import re
    
    def replacer(match):
        env_var = match.group(1)
        return os.environ.get(env_var, default)
    
    pattern = r'\$\{env:([A-Z_][A-Z0-9_]*)\}'
    return re.sub(pattern, replacer, value)


# Re-export schemas for convenience
__all__ = [
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