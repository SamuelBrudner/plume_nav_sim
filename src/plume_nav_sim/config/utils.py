"""
Configuration utilities for plume_nav_sim.

This module provides utility functions for configuration validation and management.
"""
from loguru import logger
import os
from typing import Dict, Any, Optional, Union, Type
from pydantic import BaseModel, ValidationError

from dotenv import load_dotenv, find_dotenv
try:
    from hydra.core.config_store import ConfigStore
    from hydra import initialize, compose
    from omegaconf import DictConfig, OmegaConf
except ImportError as exc:
    logger.error("Hydra and OmegaConf are required for configuration utilities: %s", exc)
    raise

from .schemas import (
    NavigatorConfig,
    SingleAgentConfig,
    MultiAgentConfig,
    VideoPlumeConfig,
    SimulationConfig,
)


def validate_config(config_class: Type[BaseModel], config_data: Union[Dict[str, Any], DictConfig]) -> BaseModel:
    """Validate configuration data against a Pydantic model.

    The ``config_class`` is supplied first to mirror ``pydantic``'s
    ``model_validate`` style and to simplify partial application.  Returns the
    instantiated and validated model instance.
    """
    if isinstance(config_data, DictConfig):
        config_data = OmegaConf.to_container(config_data, resolve=True)

    return config_class(**config_data)


def load_environment_variables(
    prefix: str = "PLUME_NAV_",
    dotenv_path: Optional[str] = None,
    verbose: bool = False,
) -> bool:
    """
    Load environment variables from a .env file and the current process.
    
    Args:
        prefix: Environment variable prefix to filter by
        dotenv_path: Explicit path to a .env file. If None, use find_dotenv()
        verbose: Verbosity flag passed to python-dotenv
        
    Returns:
        True if the dotenv file was located and successfully loaded, False otherwise
    """
    # Load .env first
    if dotenv_path:
        loaded = load_dotenv(dotenv_path, override=True, verbose=verbose)
    else:
        loaded = load_dotenv(find_dotenv(), override=True, verbose=verbose)

    # Nothing else to do for tests other than signal success
    return bool(loaded)


def initialize_hydra_config_store() -> ConfigStore:
    """Initialize Hydra ConfigStore with default schemas."""
    cs = ConfigStore.instance()

    # Register config schemas
    for name, node in [
        ("navigator_config", NavigatorConfig),
        ("single_agent_config", SingleAgentConfig),
        ("multi_agent_config", MultiAgentConfig),
        ("video_plume_config", VideoPlumeConfig),
        ("simulation_config", SimulationConfig),
    ]:
        try:
            cs.store(name=name, node=node)
        except Exception as exc:  # pragma: no cover - logging only
            logger.debug("Could not register %s: %s", name, exc)

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
        "navigator": NavigatorConfig().model_dump(),
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
        "NavigatorConfig": NavigatorConfig,
        "SingleAgentConfig": SingleAgentConfig,
        "MultiAgentConfig": MultiAgentConfig,
        "VideoPlumeConfig": VideoPlumeConfig,
        "SimulationConfig": SimulationConfig,
    }

    try:
        return schemas[schema_name]
    except KeyError as exc:
        raise KeyError(f"Unknown configuration schema: {schema_name}") from exc


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
        
    # Check for ${oc.env:VAR_NAME} or ${oc.env:VAR_NAME,default} pattern
    import re
    pattern = r"\$\{oc\.env:([A-Z_][A-Z0-9_]*)(,([^}]+))?\}"
    return bool(re.search(pattern, value))


def resolve_env_value(value: str, default: str = "") -> str:
    """Resolve environment variable references in a string without type casting."""
    if not isinstance(value, str):
        logger.debug("Non-string value provided; returning %s", value)
        return str(value)

    import re

    pattern = r'^\$\{oc\.env:([A-Z_][A-Z0-9_]*)(,([^}]*))?\}$'
    match = re.fullmatch(pattern, value)
    if match:
        env_var = match.group(1)
        inline_default = match.group(3)
        use_default = inline_default if inline_default is not None else default
        env_value = os.environ.get(env_var)
        if env_value is not None:
            logger.debug("Resolved env var %s as %s", env_var, env_value)
            return str(env_value)
        if use_default != "":
            logger.debug("Using default value for %s: %s", env_var, use_default)
            return str(use_default)
        raise KeyError(f"Environment variable '{env_var}' not found")

    if '${oc.env:' in value:
        logger.debug("Interpolation pattern detected but not full match; returning default: %s", default)
        return str(default)

    logger.debug("No interpolation; returning value unchanged: %s", value)
    return str(value)


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
