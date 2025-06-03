"""
Configuration package providing unified access to Hydra-integrated Pydantic schemas.

This module serves as the public API aggregator for the configuration system, enabling
sophisticated experiment orchestration through Hydra-core integration with comprehensive
Pydantic schema validation. The configuration architecture supports hierarchical
composition (base.yaml → config.yaml → local overrides), environment variable 
interpolation through python-dotenv, and automatic schema discovery via ConfigStore
registration.

The unified configuration system enables seamless integration with Kedro pipelines,
reinforcement learning frameworks, and machine learning workflows while maintaining
backward compatibility with existing PyYAML-based configurations.

Key Features:
    - Hydra ConfigStore integration for automatic schema discovery
    - Environment variable interpolation with secure credential management
    - Hierarchical configuration composition with validation
    - Protocol-based interfaces for external framework integration
    - Type-safe configuration validation through Pydantic models

Import Patterns:
    # Direct schema access
    from {{cookiecutter.project_slug}}.config import NavigatorConfig, VideoPlumeConfig
    
    # Configuration utilities
    from {{cookiecutter.project_slug}}.config import register_config_schemas
    
    # Environment management
    from {{cookiecutter.project_slug}}.config import load_environment_variables
    
    # Backward compatibility
    from {{cookiecutter.project_slug}}.config import validate_config
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type
from dotenv import load_dotenv, find_dotenv

# Import Hydra components for configuration management
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

# Import all configuration schemas from schemas module
from .schemas import (
    # Core configuration models
    NavigatorConfig,
    SingleAgentConfig, 
    MultiAgentConfig,
    VideoPlumeConfig,
    SimulationConfig,
    
    # ConfigStore registration function
    register_config_schemas,
    
    # Environment variable utilities
    validate_env_interpolation,
    resolve_env_value,
)

# Set up module logger
logger = logging.getLogger(__name__)

# Initialize ConfigStore for automatic schema discovery
cs = ConfigStore.instance()

# Module-level configuration state
_config_initialized = False
_environment_loaded = False


def load_environment_variables(
    dotenv_path: Optional[Union[str, Path]] = None,
    override: bool = False,
    verbose: bool = False
) -> bool:
    """
    Load environment variables from .env files with sophisticated hierarchy support.
    
    This function implements comprehensive environment variable management supporting
    multi-environment configurations (.env.development, .env.testing, .env.production)
    with automatic selection based on runtime context and secure credential handling.
    
    Args:
        dotenv_path: Specific .env file path. If None, uses automatic discovery
        override: Whether to override existing environment variables
        verbose: Enable detailed logging of environment loading process
        
    Returns:
        True if environment variables were successfully loaded
        
    Examples:
        # Automatic environment discovery
        >>> load_environment_variables()
        True
        
        # Load specific environment file
        >>> load_environment_variables(".env.development")
        True
        
        # Override existing variables with verbose logging
        >>> load_environment_variables(override=True, verbose=True)
        True
    """
    global _environment_loaded
    
    if _environment_loaded and not override:
        if verbose:
            logger.info("Environment variables already loaded. Use override=True to reload.")
        return True
    
    try:
        # Determine environment context for automatic file selection
        env_context = os.getenv('ENVIRONMENT', 'development')
        
        if dotenv_path is None:
            # Implement hierarchical .env file loading with fallback chain
            env_files = [
                f'.env.{env_context}',  # Environment-specific
                '.env.local',           # Local overrides
                '.env'                  # Base configuration
            ]
            
            loaded_files = []
            for env_file in env_files:
                env_path = find_dotenv(env_file, usecwd=True)
                if env_path:
                    load_dotenv(env_path, override=override, verbose=verbose)
                    loaded_files.append(env_file)
                    if verbose:
                        logger.info(f"Loaded environment file: {env_file}")
                else:
                    if verbose:
                        logger.debug(f"Environment file not found: {env_file}")
            
            if not loaded_files:
                logger.warning("No .env files found. Using system environment variables only.")
                
        else:
            # Load specific .env file
            env_path = Path(dotenv_path) if isinstance(dotenv_path, str) else dotenv_path
            
            if env_path.exists():
                load_dotenv(env_path, override=override, verbose=verbose)
                if verbose:
                    logger.info(f"Loaded environment file: {env_path}")
            else:
                logger.error(f"Specified .env file not found: {env_path}")
                return False
        
        # Validate critical environment variables for Hydra integration
        critical_vars = ['HYDRA_FULL_ERROR', 'OC_CAUSE']
        missing_vars = []
        
        for var in critical_vars:
            if var not in os.environ:
                # Set default values for missing Hydra variables
                if var == 'HYDRA_FULL_ERROR':
                    os.environ[var] = '1'  # Enable full error reporting
                elif var == 'OC_CAUSE':
                    os.environ[var] = '1'  # Enable error cause tracking
                
                if verbose:
                    logger.debug(f"Set default value for missing variable: {var}")
        
        _environment_loaded = True
        
        if verbose:
            logger.info("Environment variable loading completed successfully")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to load environment variables: {e}")
        return False


def initialize_hydra_config_store() -> bool:
    """
    Initialize Hydra ConfigStore with all configuration schemas.
    
    This function ensures that all Pydantic configuration schemas are properly
    registered with Hydra's ConfigStore for automatic discovery and validation
    within the structured configuration system. It handles initialization order
    dependencies and provides graceful fallback for missing components.
    
    Returns:
        True if ConfigStore initialization was successful
        
    Examples:
        >>> initialize_hydra_config_store()
        True
    """
    global _config_initialized
    
    if _config_initialized:
        logger.debug("Hydra ConfigStore already initialized")
        return True
    
    try:
        # Ensure environment variables are loaded first
        if not _environment_loaded:
            load_environment_variables(verbose=False)
        
        # Register all configuration schemas with ConfigStore
        register_config_schemas()
        
        # Verify critical schema registrations
        required_schemas = [
            'navigator/unified',
            'video_plume/default', 
            'simulation/standard'
        ]
        
        for schema_path in required_schemas:
            try:
                # Attempt to retrieve registered schema to verify registration
                group, name = schema_path.split('/')
                if not cs.repo.exists(f"{group}/{name}"):
                    logger.warning(f"Schema not found in ConfigStore: {schema_path}")
                else:
                    logger.debug(f"Verified schema registration: {schema_path}")
            except Exception as e:
                logger.warning(f"Could not verify schema registration {schema_path}: {e}")
        
        _config_initialized = True
        logger.info("Hydra ConfigStore initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize Hydra ConfigStore: {e}")
        return False


def validate_config(config: Union[Dict[str, Any], DictConfig]) -> bool:
    """
    Validate configuration against registered Pydantic schemas.
    
    This function provides comprehensive configuration validation by checking
    the configuration structure against the appropriate Pydantic models and
    ensuring compliance with Hydra's structured configuration requirements.
    
    Args:
        config: Configuration object to validate (dict or DictConfig)
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration fails validation
        TypeError: If configuration type is unsupported
        
    Examples:
        >>> config = {"navigator": {"mode": "single", "position": [50, 50]}}
        >>> validate_config(config)
        True
        
        >>> invalid_config = {"navigator": {"mode": "invalid"}}
        >>> validate_config(invalid_config)
        ValueError: mode must be one of ['single', 'multi', 'auto']
    """
    try:
        # Convert DictConfig to dict for Pydantic validation
        if isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        elif isinstance(config, dict):
            config_dict = config
        else:
            raise TypeError(f"Unsupported configuration type: {type(config)}")
        
        # Validate navigator configuration if present
        if 'navigator' in config_dict:
            navigator_config = NavigatorConfig.model_validate(config_dict['navigator'])
            logger.debug("Navigator configuration validation passed")
        
        # Validate video plume configuration if present
        if 'video_plume' in config_dict:
            video_config = VideoPlumeConfig.model_validate(config_dict['video_plume'])
            logger.debug("Video plume configuration validation passed")
        
        # Validate simulation configuration if present
        if 'simulation' in config_dict:
            sim_config = SimulationConfig.model_validate(config_dict['simulation'])
            logger.debug("Simulation configuration validation passed")
        
        logger.info("Configuration validation completed successfully")
        return True
        
    except Exception as e:
        error_msg = f"Configuration validation failed: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def get_config_schema(schema_name: str) -> Optional[Type]:
    """
    Retrieve a specific configuration schema class by name.
    
    This utility function provides programmatic access to configuration schemas
    for dynamic validation, documentation generation, and integration with
    external frameworks that require schema introspection.
    
    Args:
        schema_name: Name of the schema to retrieve
        
    Returns:
        Schema class if found, None otherwise
        
    Examples:
        >>> schema = get_config_schema("NavigatorConfig")
        >>> schema.__name__
        'NavigatorConfig'
        
        >>> schema = get_config_schema("InvalidSchema")
        >>> schema is None
        True
    """
    schema_mapping = {
        'NavigatorConfig': NavigatorConfig,
        'SingleAgentConfig': SingleAgentConfig,
        'MultiAgentConfig': MultiAgentConfig,
        'VideoPlumeConfig': VideoPlumeConfig,
        'SimulationConfig': SimulationConfig,
    }
    
    schema = schema_mapping.get(schema_name)
    if schema is None:
        logger.warning(f"Schema not found: {schema_name}")
        logger.debug(f"Available schemas: {list(schema_mapping.keys())}")
    
    return schema


def compose_config_from_overrides(
    config_name: str = "config",
    overrides: Optional[Dict[str, Any]] = None,
    return_hydra_cfg: bool = False
) -> Union[DictConfig, Dict[str, Any]]:
    """
    Compose configuration using Hydra with optional parameter overrides.
    
    This function provides programmatic access to Hydra's configuration composition
    system, enabling dynamic configuration creation with runtime parameter overrides.
    It supports both command-line style overrides and nested dictionary updates.
    
    Args:
        config_name: Base configuration name to load
        overrides: Dictionary of parameter overrides to apply
        return_hydra_cfg: Whether to return DictConfig (True) or dict (False)
        
    Returns:
        Composed configuration as DictConfig or dict
        
    Examples:
        >>> config = compose_config_from_overrides("config")
        >>> isinstance(config, dict)
        True
        
        >>> overrides = {"navigator.mode": "multi", "simulation.max_steps": 500}
        >>> config = compose_config_from_overrides("config", overrides)
        >>> config["navigator"]["mode"]
        'multi'
    """
    try:
        # Ensure ConfigStore is initialized
        if not _config_initialized:
            initialize_hydra_config_store()
        
        # Convert dict overrides to Hydra override format
        override_list = []
        if overrides:
            for key, value in overrides.items():
                if isinstance(value, str):
                    override_list.append(f"{key}={value}")
                else:
                    override_list.append(f"{key}={str(value)}")
        
        # Initialize Hydra context if not already initialized
        if not GlobalHydra().is_initialized():
            # Try to find config directory relative to package
            config_dir = Path(__file__).parent.parent.parent.parent / "conf"
            if not config_dir.exists():
                # Fallback to current directory
                config_dir = Path.cwd() / "conf"
            
            with initialize(version_base=None, config_path=str(config_dir)):
                cfg = compose(config_name=config_name, overrides=override_list)
        else:
            cfg = compose(config_name=config_name, overrides=override_list)
        
        # Return appropriate format
        if return_hydra_cfg:
            return cfg
        else:
            return OmegaConf.to_container(cfg, resolve=True)
            
    except Exception as e:
        logger.error(f"Failed to compose configuration: {e}")
        raise


def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration with sensible parameter values.
    
    This function generates a complete default configuration suitable for
    basic simulation scenarios, providing a starting point for configuration
    customization and serving as a reference for required parameters.
    
    Returns:
        Dictionary containing default configuration values
        
    Examples:
        >>> config = create_default_config()
        >>> config["navigator"]["mode"]
        'auto'
        
        >>> config["simulation"]["max_steps"]
        1000
    """
    try:
        # Create default configuration using schema defaults
        default_config = {
            "navigator": NavigatorConfig().model_dump(),
            "video_plume": VideoPlumeConfig(
                video_path="${oc.env:VIDEO_PATH,./data/default_plume.mp4}"
            ).model_dump(),
            "simulation": SimulationConfig().model_dump(),
        }
        
        logger.debug("Created default configuration with schema defaults")
        return default_config
        
    except Exception as e:
        logger.error(f"Failed to create default configuration: {e}")
        raise


# Automatic initialization on module import
def _initialize_module():
    """
    Perform automatic module initialization including environment loading and ConfigStore setup.
    
    This function is called automatically when the module is imported, ensuring that
    the configuration system is ready for use without requiring explicit initialization
    by the user. It handles initialization order dependencies and provides graceful
    fallback for missing components.
    """
    try:
        # Load environment variables with automatic discovery
        load_environment_variables(verbose=False)
        
        # Initialize Hydra ConfigStore with all schemas
        initialize_hydra_config_store()
        
        logger.info("Configuration module initialization completed successfully")
        
    except Exception as e:
        logger.warning(f"Module initialization completed with warnings: {e}")
        # Continue with partial initialization rather than failing completely


# Public API exports
__all__ = [
    # Configuration schema classes
    "NavigatorConfig",
    "SingleAgentConfig", 
    "MultiAgentConfig",
    "VideoPlumeConfig",
    "SimulationConfig",
    
    # Configuration management functions
    "load_environment_variables",
    "initialize_hydra_config_store",
    "validate_config",
    "get_config_schema",
    "compose_config_from_overrides",
    "create_default_config",
    
    # Schema registration utilities
    "register_config_schemas",
    
    # Environment variable utilities
    "validate_env_interpolation",
    "resolve_env_value",
    
    # ConfigStore instance for advanced usage
    "cs",
]

# Automatic module initialization
_initialize_module()

# Backward compatibility support for legacy imports
def validate_navigator_config(config: Dict[str, Any]) -> bool:
    """
    Legacy function for backward compatibility with existing code.
    
    Args:
        config: Navigator configuration dictionary
        
    Returns:
        True if configuration is valid
        
    Deprecated:
        Use validate_config() instead for comprehensive validation
    """
    logger.warning("validate_navigator_config is deprecated. Use validate_config() instead.")
    try:
        NavigatorConfig.model_validate(config)
        return True
    except Exception as e:
        logger.error(f"Navigator configuration validation failed: {e}")
        return False


def validate_video_plume_config(config: Dict[str, Any]) -> bool:
    """
    Legacy function for backward compatibility with existing code.
    
    Args:
        config: Video plume configuration dictionary
        
    Returns:
        True if configuration is valid
        
    Deprecated:
        Use validate_config() instead for comprehensive validation
    """
    logger.warning("validate_video_plume_config is deprecated. Use validate_config() instead.")
    try:
        VideoPlumeConfig.model_validate(config)
        return True
    except Exception as e:
        logger.error(f"Video plume configuration validation failed: {e}")
        return False


# Add legacy functions to exports for backward compatibility
__all__.extend([
    "validate_navigator_config",
    "validate_video_plume_config",
])