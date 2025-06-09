"""
Configuration module for odor plume navigation.

This module provides advanced Hydra-core configuration management with hierarchical
.env file loading, programmatic configuration generation, and Pydantic schema
registration for the unified odor plume navigation system.

Features:
- Hydra ConfigStore initialization and schema registration
- Hierarchical environment variable loading (.env.development, .env.local, .env)
- Programmatic configuration composition and validation
- Environment variable interpolation support
- Automatic module initialization with configuration loading
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from hydra import compose, initialize_config_store
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf, DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    ConfigStore = None
    DictConfig = None

# Export configuration models and utilities from this module
from odor_plume_nav.config.models import (
    NavigatorConfig,
    SingleAgentConfig,
    MultiAgentConfig,
    VideoPlumeConfig,
)
from odor_plume_nav.config.utils import (
    load_config,
    save_config,
    validate_config,
    update_config,
    get_config_dir,
    load_yaml_config,
    load_file_config,
)

# Import validation utilities
from odor_plume_nav.services.validator import (
    ConfigValidationError,
    validate_video_plume_config,
    validate_navigator_config,
)

# Module logger
logger = logging.getLogger(__name__)


def load_environment_variables(environment: Optional[str] = None) -> Dict[str, str]:
    """
    Load environment variables from hierarchical .env files.
    
    Loads variables in the following priority order:
    1. .env.development (if environment == 'development')
    2. .env.local (for local overrides)
    3. .env (base environment file)
    
    Args:
        environment: Environment name to load specific .env file
        
    Returns:
        Dictionary of loaded environment variables
        
    Raises:
        ImportError: If python-dotenv is not available
    """
    if load_dotenv is None:
        raise ImportError(
            "python-dotenv is required for environment variable loading. "
            "Install with: pip install python-dotenv"
        )
    
    loaded_vars = {}
    config_dir = get_config_dir()
    
    # Load .env files in reverse priority order (base first, overrides last)
    env_files = [
        config_dir / ".env",
        config_dir / ".env.local",
    ]
    
    # Add environment-specific file if specified
    if environment:
        env_files.insert(-1, config_dir / f".env.{environment}")
    
    # Also check current working directory
    cwd_env_files = [
        Path.cwd() / ".env",
        Path.cwd() / ".env.local",
    ]
    
    if environment:
        cwd_env_files.insert(-1, Path.cwd() / f".env.{environment}")
    
    env_files.extend(cwd_env_files)
    
    for env_file in env_files:
        if env_file.exists():
            logger.debug(f"Loading environment variables from {env_file}")
            load_dotenv(env_file, override=True)
            
            # Track what was loaded for debugging
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key = line.split('=', 1)[0]
                            if key in os.environ:
                                loaded_vars[key] = os.environ[key]
            except Exception as e:
                logger.warning(f"Could not parse environment file {env_file}: {e}")
    
    logger.info(f"Loaded {len(loaded_vars)} environment variables from .env files")
    return loaded_vars


def initialize_hydra_config_store() -> Optional[ConfigStore]:
    """
    Initialize Hydra ConfigStore and register Pydantic schemas.
    
    Registers configuration schemas for:
    - navigator: NavigatorConfig, SingleAgentConfig, MultiAgentConfig
    - video_plume: VideoPlumeConfig
    - simulation: Composite configuration schemas
    
    Returns:
        ConfigStore instance if Hydra is available, None otherwise
        
    Raises:
        ImportError: If Hydra is not available
    """
    if not HYDRA_AVAILABLE:
        logger.warning(
            "Hydra is not available. ConfigStore initialization skipped. "
            "Install with: pip install hydra-core"
        )
        return None
    
    try:
        cs = ConfigStore.instance()
        
        # Register navigator configurations
        cs.store(name="navigator_base", node=NavigatorConfig, group="navigator")
        cs.store(name="single_agent", node=SingleAgentConfig, group="navigator") 
        cs.store(name="multi_agent", node=MultiAgentConfig, group="navigator")
        
        # Register video plume configuration
        cs.store(name="video_plume_base", node=VideoPlumeConfig, group="video_plume")
        
        # Register composite schemas for common use cases
        cs.store(
            name="simulation_base",
            node={
                "navigator": NavigatorConfig(),
                "video_plume": VideoPlumeConfig(),
            },
            group="simulation"
        )
        
        cs.store(
            name="single_agent_simulation", 
            node={
                "navigator": SingleAgentConfig(),
                "video_plume": VideoPlumeConfig(),
            },
            group="simulation"
        )
        
        cs.store(
            name="multi_agent_simulation",
            node={
                "navigator": MultiAgentConfig(),
                "video_plume": VideoPlumeConfig(),
            },
            group="simulation"
        )
        
        logger.info("Hydra ConfigStore initialized with Pydantic schemas")
        return cs
        
    except Exception as e:
        logger.error(f"Failed to initialize Hydra ConfigStore: {e}")
        raise


def compose_config_from_overrides(
    config_name: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    return_hydra_config: bool = False
) -> Union[Dict[str, Any], DictConfig]:
    """
    Compose configuration programmatically from Hydra with overrides.
    
    Args:
        config_name: Name of the config to compose (defaults to 'config')
        overrides: List of override strings in Hydra format (e.g., ['navigator.speed=2.0'])
        return_hydra_config: Whether to return DictConfig or plain dict
        
    Returns:
        Composed configuration as dictionary or DictConfig
        
    Raises:
        ImportError: If Hydra is not available
        ValueError: If configuration composition fails
    """
    if not HYDRA_AVAILABLE:
        raise ImportError(
            "Hydra is required for configuration composition. "
            "Install with: pip install hydra-core"
        )
    
    try:
        # Use compose API for programmatic configuration
        cfg = compose(config_name=config_name or "config", overrides=overrides or [])
        
        if return_hydra_config:
            return cfg
        else:
            return OmegaConf.to_object(cfg)
            
    except Exception as e:
        logger.error(f"Failed to compose configuration: {e}")
        raise ValueError(f"Configuration composition failed: {e}")


def create_default_config(
    navigator_type: str = "single_agent",
    validate: bool = True
) -> Dict[str, Any]:
    """
    Create a default configuration for common use cases.
    
    Args:
        navigator_type: Type of navigator ('single_agent', 'multi_agent', or 'base')
        validate: Whether to validate the created configuration
        
    Returns:
        Default configuration dictionary
        
    Raises:
        ValueError: If navigator_type is invalid or validation fails
    """
    # Create default configurations based on type
    if navigator_type == "single_agent":
        config = {
            "navigator": SingleAgentConfig().dict(),
            "video_plume": VideoPlumeConfig().dict(),
        }
    elif navigator_type == "multi_agent":
        config = {
            "navigator": MultiAgentConfig().dict(),
            "video_plume": VideoPlumeConfig().dict(),
        }
    elif navigator_type == "base":
        config = {
            "navigator": NavigatorConfig().dict(),
            "video_plume": VideoPlumeConfig().dict(),
        }
    else:
        raise ValueError(
            f"Invalid navigator_type '{navigator_type}'. "
            f"Must be one of: 'single_agent', 'multi_agent', 'base'"
        )
    
    # Add metadata
    config["_metadata"] = {
        "created_by": "odor_plume_nav.config.create_default_config",
        "navigator_type": navigator_type,
        "version": "1.0",
    }
    
    # Validate if requested
    if validate:
        try:
            if "navigator" in config:
                validate_navigator_config(config)
            if "video_plume" in config:
                validate_video_plume_config(config)
        except ConfigValidationError as e:
            logger.error(f"Default configuration validation failed: {e}")
            raise ValueError(f"Default configuration is invalid: {e}")
    
    logger.debug(f"Created default {navigator_type} configuration")
    return config


def _initialize_module() -> None:
    """
    Initialize the configuration module automatically on import.
    
    This function:
    1. Loads environment variables from .env files
    2. Initializes Hydra ConfigStore with Pydantic schemas
    3. Sets up logging for configuration operations
    
    Called automatically when the module is imported.
    """
    try:
        # Determine environment from ENVIRONMENT variable
        environment = os.environ.get("ENVIRONMENT")
        
        # Load environment variables
        load_environment_variables(environment=environment)
        
        # Initialize Hydra ConfigStore if available
        if HYDRA_AVAILABLE:
            initialize_hydra_config_store()
        else:
            logger.info(
                "Hydra not available. Install with 'pip install hydra-core' "
                "for advanced configuration features."
            )
            
    except Exception as e:
        logger.warning(f"Configuration module initialization failed: {e}")


# Enhanced exports
__all__ = [
    # Configuration models
    "NavigatorConfig",
    "SingleAgentConfig", 
    "MultiAgentConfig",
    "VideoPlumeConfig",
    
    # Configuration utilities
    "load_config",
    "save_config",
    "validate_config",
    "update_config",
    "get_config_dir",
    "load_yaml_config",
    "load_file_config",
    
    # Validation utilities
    "ConfigValidationError",
    "validate_video_plume_config",
    "validate_navigator_config",
    
    # Advanced configuration functions
    "load_environment_variables",
    "initialize_hydra_config_store",
    "compose_config_from_overrides",
    "create_default_config",
]

# Automatically initialize the module when imported
_initialize_module()
