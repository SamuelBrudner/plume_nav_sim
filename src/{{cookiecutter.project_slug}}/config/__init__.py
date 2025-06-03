"""
Unified configuration management with Hydra integration and Pydantic validation.

This module provides a comprehensive configuration system supporting hierarchical 
composition through Hydra-core, environment variable interpolation via python-dotenv,
and type-safe validation through Pydantic schemas. The system enables sophisticated
experiment orchestration and configuration management for scientific computing workflows.

Key Features:
    - Hydra ConfigStore integration for automatic schema discovery
    - Environment variable interpolation with ${oc.env:VAR_NAME} syntax
    - Hierarchical configuration through conf/base.yaml → conf/config.yaml → local overrides
    - Type-safe configuration validation with comprehensive error reporting
    - Support for both programmatic and CLI-based configuration management
    - Seamless integration with Kedro, RL frameworks, and ML pipelines

Configuration Structure:
    conf/
    ├── base.yaml           # Foundation defaults and core parameters
    ├── config.yaml         # Environment-specific parameters and overrides
    └── local/              # Local development and testing overrides
        ├── credentials.yaml.template
        └── paths.yaml.template

Import Patterns:
    # Traditional schema imports
    from {{cookiecutter.project_slug}}.config import NavigatorConfig, VideoPlumeConfig
    
    # Factory-based configuration (Hydra-integrated)
    from {{cookiecutter.project_slug}}.config import create_navigator_config, load_config
    
    # Environment variable utilities
    from {{cookiecutter.project_slug}}.config import setup_environment, validate_environment
    
    # CLI integration (automatic with @hydra.main decorator)
    from {{cookiecutter.project_slug}}.config import register_configs

Environment Variable Examples:
    # Basic interpolation
    max_speed: ${oc.env:NAVIGATOR_MAX_SPEED,1.0}
    
    # Secure credentials
    database_url: ${oc.env:DATABASE_URL}
    
    # Development overrides
    video_path: ${oc.env:PLUME_VIDEO_PATH,./data/default_plume.mp4}
    
    # Multi-agent scaling
    num_agents: ${oc.env:SWARM_SIZE,5}
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union, Type
import logging

# Core configuration schemas - import all from schemas.py
from .schemas import (
    NavigatorConfig,
    SingleAgentConfig, 
    MultiAgentConfig,
    VideoPlumeConfig,
)

# Hydra and OmegaConf imports with graceful fallback
try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from hydra.core.global_hydra import GlobalHydra
    from hydra.initialize import initialize
    from hydra.compose import compose
    from omegaconf import DictConfig, OmegaConf, MISSING
    HYDRA_AVAILABLE = True
except ImportError:
    hydra = None
    ConfigStore = None
    GlobalHydra = None
    initialize = None
    compose = None
    DictConfig = dict
    OmegaConf = None
    MISSING = "???"
    HYDRA_AVAILABLE = False

# Environment variable management with python-dotenv
try:
    from dotenv import load_dotenv, find_dotenv, dotenv_values
    DOTENV_AVAILABLE = True
except ImportError:
    load_dotenv = None
    find_dotenv = None
    dotenv_values = None
    DOTENV_AVAILABLE = False

# Pydantic for validation
from pydantic import BaseModel, ValidationError

# Set up module logger
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """
    Comprehensive configuration error with context and suggestions.
    
    Provides detailed error information for configuration validation failures,
    missing environment variables, and Hydra composition issues.
    """
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, 
                 suggestions: Optional[list] = None):
        super().__init__(message)
        self.context = context or {}
        self.suggestions = suggestions or []
    
    def __str__(self):
        error_msg = super().__str__()
        
        if self.context:
            error_msg += f"\n\nContext: {self.context}"
        
        if self.suggestions:
            error_msg += f"\n\nSuggestions:"
            for suggestion in self.suggestions:
                error_msg += f"\n  - {suggestion}"
                
        return error_msg


def setup_environment(env_file: Optional[Union[str, Path]] = None, 
                     verbose: bool = False) -> Dict[str, str]:
    """
    Set up environment variables with python-dotenv integration.
    
    Loads environment variables from .env files with comprehensive search
    and precedence handling. Supports hierarchical environment files
    for development, testing, and production configurations.
    
    Args:
        env_file: Specific .env file to load (optional)
        verbose: Enable detailed logging of environment loading
        
    Returns:
        Dictionary of loaded environment variables
        
    Raises:
        ConfigurationError: If environment setup fails or required variables missing
        
    Examples:
        # Load default environment files
        env_vars = setup_environment()
        
        # Load specific environment file
        env_vars = setup_environment("./conf/local/.env.development")
        
        # Verbose environment loading
        env_vars = setup_environment(verbose=True)
    """
    if not DOTENV_AVAILABLE:
        logger.warning("python-dotenv not available - environment variables from system only")
        return dict(os.environ)
    
    loaded_vars = {}
    
    try:
        if env_file:
            # Load specific file
            env_path = Path(env_file)
            if env_path.exists():
                loaded = load_dotenv(env_path, verbose=verbose)
                if verbose and loaded:
                    logger.info(f"Loaded environment from: {env_path}")
                loaded_vars.update(dotenv_values(env_path))
            else:
                raise ConfigurationError(
                    f"Specified environment file not found: {env_path}",
                    context={"env_file": str(env_path)},
                    suggestions=[
                        "Check file path spelling and permissions",
                        "Create environment file from template",
                        "Use default environment loading (env_file=None)"
                    ]
                )
        else:
            # Auto-discover and load environment files with precedence
            env_files = [
                ".env.local",
                ".env.development", 
                ".env.testing",
                ".env.production",
                ".env"
            ]
            
            for env_name in env_files:
                env_path = find_dotenv(env_name)
                if env_path:
                    loaded = load_dotenv(env_path, verbose=verbose)
                    if verbose and loaded:
                        logger.info(f"Loaded environment from: {env_path}")
                    loaded_vars.update(dotenv_values(env_path))
        
        # Add system environment variables (highest priority)
        loaded_vars.update(dict(os.environ))
        
        if verbose:
            logger.info(f"Total environment variables loaded: {len(loaded_vars)}")
            
        return loaded_vars
        
    except Exception as e:
        raise ConfigurationError(
            f"Failed to set up environment: {str(e)}",
            context={"env_file": str(env_file) if env_file else "auto-discovery"},
            suggestions=[
                "Check .env file format and syntax",
                "Verify file permissions and accessibility",
                "Use absolute paths for environment files",
                "Check python-dotenv installation"
            ]
        ) from e


def validate_environment(required_vars: list, optional_vars: Optional[list] = None) -> Dict[str, str]:
    """
    Validate presence and format of required environment variables.
    
    Ensures all required environment variables are present and optionally
    validates their format. Provides comprehensive error reporting with
    suggestions for missing or invalid variables.
    
    Args:
        required_vars: List of required environment variable names
        optional_vars: List of optional environment variable names
        
    Returns:
        Dictionary of validated environment variables
        
    Raises:
        ConfigurationError: If required variables missing or invalid
        
    Examples:
        # Basic validation
        env_vars = validate_environment(["DATABASE_URL", "API_KEY"])
        
        # With optional variables
        env_vars = validate_environment(
            required_vars=["DATABASE_URL"],
            optional_vars=["DEBUG", "LOG_LEVEL"]
        )
    """
    optional_vars = optional_vars or []
    validated_vars = {}
    missing_vars = []
    
    # Check required variables
    for var_name in required_vars:
        value = os.getenv(var_name)
        if value is None:
            missing_vars.append(var_name)
        else:
            validated_vars[var_name] = value
    
    # Check optional variables
    for var_name in optional_vars:
        value = os.getenv(var_name)
        if value is not None:
            validated_vars[var_name] = value
    
    if missing_vars:
        raise ConfigurationError(
            f"Required environment variables missing: {missing_vars}",
            context={
                "missing_vars": missing_vars,
                "available_vars": list(validated_vars.keys())
            },
            suggestions=[
                "Create .env file with required variables",
                "Set environment variables in your shell",
                "Use conf/local/credentials.yaml.template as reference",
                "Check Hydra interpolation syntax: ${oc.env:VAR_NAME}"
            ]
        )
    
    logger.debug(f"Environment validation successful: {len(validated_vars)} variables")
    return validated_vars


def register_configs() -> None:
    """
    Register all configuration schemas with Hydra ConfigStore.
    
    Enables automatic configuration discovery and structured composition
    for CLI applications and factory methods. Registers both individual
    schemas and configuration groups for hierarchical composition.
    
    This function is automatically called when the module is imported
    if Hydra is available, ensuring seamless integration with @hydra.main
    decorators and compose() API.
    
    Raises:
        ConfigurationError: If Hydra not available or registration fails
        
    Examples:
        # Manual registration (normally automatic)
        register_configs()
        
        # Use in Hydra application
        @hydra.main(version_base=None, config_path="../conf", config_name="config")
        def my_app(cfg: DictConfig) -> None:
            # Configuration automatically available
            navigator_config = NavigatorConfig(**cfg.navigator)
    """
    if not HYDRA_AVAILABLE:
        logger.warning("Hydra not available - ConfigStore registration skipped")
        return
    
    try:
        cs = ConfigStore.instance()
        
        # Register individual configuration schemas
        cs.store(name="navigator_config", node=NavigatorConfig)
        cs.store(name="single_agent_config", node=SingleAgentConfig)
        cs.store(name="multi_agent_config", node=MultiAgentConfig)
        cs.store(name="video_plume_config", node=VideoPlumeConfig)
        
        # Register configuration groups for hierarchical composition
        cs.store(group="navigator", name="single_agent", node=SingleAgentConfig)
        cs.store(group="navigator", name="multi_agent", node=MultiAgentConfig)
        cs.store(group="navigator", name="unified", node=NavigatorConfig)
        cs.store(group="environment", name="video_plume", node=VideoPlumeConfig)
        
        # Register default configurations for common use cases
        cs.store(group="defaults", name="research", node={
            "navigator": "unified",
            "environment": "video_plume"
        })
        
        logger.debug("Hydra ConfigStore registration completed successfully")
        
    except Exception as e:
        raise ConfigurationError(
            f"Failed to register configurations with Hydra: {str(e)}",
            context={"hydra_available": HYDRA_AVAILABLE},
            suggestions=[
                "Check Hydra installation and version (≥1.3.2)",
                "Verify configuration schema definitions",
                "Import config module after Hydra initialization",
                "Check for circular import dependencies"
            ]
        ) from e


def load_config(config_path: Optional[str] = None, 
               config_name: str = "config",
               overrides: Optional[list] = None) -> DictConfig:
    """
    Load and compose Hydra configuration with environment interpolation.
    
    Provides programmatic access to Hydra configuration composition without
    requiring @hydra.main decorator. Supports runtime configuration loading
    for notebooks, testing, and library usage scenarios.
    
    Args:
        config_path: Path to configuration directory (defaults to ../conf)
        config_name: Name of primary config file (defaults to "config")
        overrides: List of configuration overrides in key=value format
        
    Returns:
        Composed and validated DictConfig object
        
    Raises:
        ConfigurationError: If configuration loading or validation fails
        
    Examples:
        # Basic configuration loading
        cfg = load_config()
        
        # Custom configuration path
        cfg = load_config(config_path="./custom_conf")
        
        # With overrides
        cfg = load_config(overrides=["navigator.max_speed=2.0", "hydra.job.name=experiment_1"])
        
        # Environment-specific configuration
        cfg = load_config(config_name="development")
    """
    if not HYDRA_AVAILABLE:
        raise ConfigurationError(
            "Hydra not available - cannot load configuration",
            suggestions=[
                "Install hydra-core: pip install hydra-core",
                "Use direct Pydantic schema instantiation",
                "Check dependencies in pyproject.toml"
            ]
        )
    
    overrides = overrides or []
    
    try:
        # Set default config path relative to package location
        if config_path is None:
            current_dir = Path(__file__).parent
            # Navigate up to project root and find conf directory
            config_path = str(current_dir.parent.parent.parent / "conf")
        
        # Initialize Hydra if not already initialized
        if not GlobalHydra().is_initialized():
            with initialize(version_base=None, config_path=config_path):
                cfg = compose(config_name=config_name, overrides=overrides)
        else:
            # Use existing Hydra instance
            cfg = compose(config_name=config_name, overrides=overrides)
        
        logger.debug(f"Configuration loaded successfully from: {config_path}/{config_name}.yaml")
        return cfg
        
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load configuration: {str(e)}",
            context={
                "config_path": config_path,
                "config_name": config_name,
                "overrides": overrides
            },
            suggestions=[
                "Check configuration file paths and permissions",
                "Verify YAML syntax in configuration files", 
                "Ensure environment variables are properly set",
                "Check Hydra interpolation syntax: ${oc.env:VAR_NAME}",
                "Validate configuration schema compatibility"
            ]
        ) from e


def create_navigator_config(config_data: Union[Dict[str, Any], DictConfig]) -> NavigatorConfig:
    """
    Create and validate NavigatorConfig from dictionary or DictConfig.
    
    Factory method for creating type-safe NavigatorConfig instances with
    comprehensive validation and error reporting. Supports both direct
    dictionary input and Hydra DictConfig composition.
    
    Args:
        config_data: Configuration data as dictionary or DictConfig
        
    Returns:
        Validated NavigatorConfig instance
        
    Raises:
        ConfigurationError: If validation fails with detailed error context
        
    Examples:
        # From dictionary
        nav_config = create_navigator_config({
            "position": [0.0, 0.0],
            "max_speed": 1.0,
            "orientation": 45.0
        })
        
        # From Hydra DictConfig
        cfg = load_config()
        nav_config = create_navigator_config(cfg.navigator)
        
        # Multi-agent configuration
        nav_config = create_navigator_config({
            "positions": [[0.0, 0.0], [1.0, 0.0]],
            "num_agents": 2,
            "max_speeds": [1.0, 1.2]
        })
    """
    try:
        # Convert DictConfig to dictionary if needed
        if hasattr(config_data, '_content'):  # DictConfig check
            config_dict = OmegaConf.to_container(config_data, resolve=True)
        else:
            config_dict = dict(config_data)
        
        # Create and validate NavigatorConfig
        navigator_config = NavigatorConfig(**config_dict)
        
        logger.debug("NavigatorConfig created and validated successfully")
        return navigator_config
        
    except ValidationError as e:
        raise ConfigurationError(
            f"NavigatorConfig validation failed: {str(e)}",
            context={
                "config_data": config_data,
                "validation_errors": e.errors()
            },
            suggestions=[
                "Check parameter types and value ranges",
                "Verify position coordinates are valid [x, y] pairs",
                "Ensure speed values are non-negative",
                "Check multi-agent parameter list lengths match",
                "Review NavigatorConfig documentation"
            ]
        ) from e
    except Exception as e:
        raise ConfigurationError(
            f"Failed to create NavigatorConfig: {str(e)}",
            context={"config_data": config_data},
            suggestions=[
                "Check configuration data format",
                "Verify all required parameters are present",
                "Use load_config() for Hydra integration",
                "Check environment variable interpolation"
            ]
        ) from e


def create_video_plume_config(config_data: Union[Dict[str, Any], DictConfig]) -> VideoPlumeConfig:
    """
    Create and validate VideoPlumeConfig from dictionary or DictConfig.
    
    Factory method for creating type-safe VideoPlumeConfig instances with
    path validation and preprocessing parameter verification. Supports
    environment variable interpolation for video paths and processing options.
    
    Args:
        config_data: Configuration data as dictionary or DictConfig
        
    Returns:
        Validated VideoPlumeConfig instance
        
    Raises:
        ConfigurationError: If validation fails or video file not accessible
        
    Examples:
        # Basic video configuration
        video_config = create_video_plume_config({
            "video_path": "./data/plume_video.mp4",
            "flip": False,
            "grayscale": True
        })
        
        # With environment variables
        video_config = create_video_plume_config({
            "video_path": "${oc.env:PLUME_VIDEO_PATH}",
            "threshold": 0.5,
            "kernel_size": 5
        })
        
        # Advanced preprocessing
        video_config = create_video_plume_config({
            "video_path": "./data/turbulent_plume.avi",
            "flip": True,
            "kernel_size": 7,
            "kernel_sigma": 2.0,
            "normalize": True
        })
    """
    try:
        # Convert DictConfig to dictionary if needed
        if hasattr(config_data, '_content'):  # DictConfig check
            config_dict = OmegaConf.to_container(config_data, resolve=True)
        else:
            config_dict = dict(config_data)
        
        # Create and validate VideoPlumeConfig
        video_config = VideoPlumeConfig(**config_dict)
        
        logger.debug("VideoPlumeConfig created and validated successfully")
        return video_config
        
    except ValidationError as e:
        raise ConfigurationError(
            f"VideoPlumeConfig validation failed: {str(e)}",
            context={
                "config_data": config_data,
                "validation_errors": e.errors()
            },
            suggestions=[
                "Check video file path exists and is accessible",
                "Verify kernel_size is odd and positive if specified",
                "Ensure kernel_sigma is positive when kernel_size provided",
                "Check threshold value is between 0.0 and 1.0",
                "Review VideoPlumeConfig documentation"
            ]
        ) from e
    except Exception as e:
        raise ConfigurationError(
            f"Failed to create VideoPlumeConfig: {str(e)}",
            context={"config_data": config_data},
            suggestions=[
                "Check configuration data format",
                "Verify video file path resolution",
                "Use load_config() for environment variable interpolation",
                "Check file permissions and accessibility"
            ]
        ) from e


# Automatic environment setup and config registration on module import
try:
    # Set up environment variables from .env files
    if DOTENV_AVAILABLE:
        setup_environment(verbose=False)
        logger.debug("Environment variables loaded automatically")
    
    # Register configurations with Hydra ConfigStore
    if HYDRA_AVAILABLE:
        register_configs()
        logger.debug("Hydra ConfigStore registration completed")
        
except Exception as e:
    logger.warning(f"Automatic configuration setup failed: {str(e)}")
    # Continue without automatic setup - manual setup still available


# Public API exports - comprehensive configuration interface
__all__ = [
    # Configuration schemas (from schemas.py)
    "NavigatorConfig",
    "SingleAgentConfig", 
    "MultiAgentConfig",
    "VideoPlumeConfig",
    
    # Factory methods for configuration creation
    "create_navigator_config",
    "create_video_plume_config",
    
    # Environment and configuration management
    "setup_environment",
    "validate_environment",
    "load_config",
    "register_configs",
    
    # Error handling
    "ConfigurationError",
    
    # Hydra integration utilities (when available)
    "HYDRA_AVAILABLE",
    "DOTENV_AVAILABLE",
]

# Version information and metadata
__version__ = "1.0.0"
__author__ = "Odor Plume Navigation Team"
__description__ = "Unified configuration management with Hydra and Pydantic integration"

# Configuration system metadata for introspection
CONFIG_METADATA = {
    "version": __version__,
    "hydra_available": HYDRA_AVAILABLE,
    "dotenv_available": DOTENV_AVAILABLE,
    "schemas": [
        "NavigatorConfig",
        "SingleAgentConfig", 
        "MultiAgentConfig",
        "VideoPlumeConfig"
    ],
    "config_groups": [
        "navigator",
        "environment", 
        "defaults"
    ],
    "supported_interpolations": [
        "${oc.env:VAR_NAME}",
        "${oc.env:VAR_NAME,default_value}"
    ]
}