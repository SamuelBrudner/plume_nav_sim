"""
Configuration module for odor plume navigation.

This module provides advanced Hydra-core configuration management with hierarchical
.env file loading, programmatic configuration generation, and dataclass schema
registration for the unified odor plume navigation system.

Features:
- Hydra ConfigStore initialization and dataclass schema registration
- Hierarchical environment variable loading (.env.development, .env.local, .env)
- Programmatic configuration composition and validation
- Environment variable interpolation support
- Dataclass-based structured configuration with Pydantic validation
- Automatic module initialization with configuration loading
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Tuple, Literal
from enum import Enum

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
    # Newly introduced Enum replacements for Literal types
    NavigatorMode,
    OutputFormat,
    LogLevel,
)
from odor_plume_nav.config.utils import (
    load_config,
    save_config,
    validate_config,
    update_config,
    get_config_dir,
    load_yaml_config,
    load_file_config,
    # Dataclass support utilities
    get_dataclass_fields_info,
    validate_dataclass_env_interpolation,
    create_dataclass_instance_with_env_resolution,
    validate_env_interpolation,
    resolve_env_value,
)

# Import validation utilities
from odor_plume_nav.services.validator import (
    ConfigValidationError,
    validate_video_plume_config,
    validate_navigator_config,
)

# Module logger
logger = logging.getLogger(__name__)


# Dataclass-based configuration schemas for Hydra structured configs
# These provide the dataclass implementations required by Hydra 1.3+ structured configuration

@dataclass
class DataclassSingleAgentConfig:
    """
    Dataclass configuration schema for single agent navigation parameters.
    
    Supports Hydra structured configuration with field validation and
    environment variable interpolation via ${oc.env:VAR,DEFAULT} syntax.
    """
    position: Optional[Tuple[float, float]] = field(
        default=None,
        metadata={"description": "Initial [x, y] position coordinates. Supports ${oc.env:AGENT_START_X,50.0} interpolation"}
    )
    orientation: Optional[float] = field(
        default=0.0,
        metadata={"ge": 0.0, "le": 360.0, "description": "Initial orientation in degrees (0=right, 90=up). Supports ${oc.env:AGENT_ORIENTATION,0.0}"}
    )
    speed: Optional[float] = field(
        default=0.0,
        metadata={"ge": 0.0, "description": "Initial speed in units per timestep. Supports ${oc.env:AGENT_SPEED,0.0}"}
    )
    max_speed: Optional[float] = field(
        default=1.0,
        metadata={"gt": 0.0, "description": "Maximum allowed speed constraint. Supports ${oc.env:AGENT_MAX_SPEED,1.0}"}
    )
    angular_velocity: Optional[float] = field(
        default=0.0,
        metadata={"description": "Angular velocity in degrees per timestep. Supports ${oc.env:AGENT_ANGULAR_VEL,0.0}"}
    )
    _target_: str = field(
        default="odor_plume_nav.core.controllers.SingleAgentController",
        metadata={"description": "Hydra target for automatic instantiation"}
    )


@dataclass
class DataclassMultiAgentConfig:
    """
    Dataclass configuration schema for multi-agent navigation scenarios.
    
    Defines parameters for coordinated multi-agent simulations with vectorized
    operations support and comprehensive validation. All list parameters must
    have consistent lengths equal to num_agents when specified.
    """
    positions: Optional[List[List[float]]] = field(
        default=None,
        metadata={"description": "List of [x, y] positions for each agent. Length must match num_agents"}
    )
    orientations: Optional[List[float]] = field(
        default=None,
        metadata={"description": "List of initial orientations in degrees for each agent"}
    )
    speeds: Optional[List[float]] = field(
        default=None,
        metadata={"description": "List of initial speeds for each agent"}
    )
    max_speeds: Optional[List[float]] = field(
        default=None,
        metadata={"description": "List of maximum speed constraints for each agent"}
    )
    angular_velocities: Optional[List[float]] = field(
        default=None,
        metadata={"description": "List of angular velocities in degrees per timestep for each agent"}
    )
    num_agents: Optional[int] = field(
        default=None,
        metadata={"gt": 0, "le": 100, "description": "Total number of agents in simulation. Supports ${oc.env:NUM_AGENTS,10}"}
    )
    _target_: str = field(
        default="odor_plume_nav.core.controllers.MultiAgentController",
        metadata={"description": "Hydra target for automatic instantiation"}
    )


@dataclass
class DataclassNavigatorConfig:
    """
    Dataclass unified configuration schema for navigation supporting both single and multi-agent scenarios.
    
    This is the primary configuration model that consolidates single and multi-agent parameters
    with intelligent mode detection, Hydra environment variable interpolation support, and
    hierarchical configuration composition for factory-driven component instantiation.
    """
    mode: NavigatorMode = field(
        default=NavigatorMode.AUTO,
        metadata={"description": "Navigation mode: 'single', 'multi', or 'auto' for automatic detection"}
    )
    
    # Single agent parameters
    position: Optional[Tuple[float, float]] = field(
        default=None,
        metadata={"description": "Single agent initial position. Supports ${oc.env:AGENT_POSITION} interpolation"}
    )
    orientation: Optional[float] = field(
        default=0.0,
        metadata={"ge": 0.0, "le": 360.0, "description": "Single agent initial orientation in degrees. Supports ${oc.env:AGENT_ORIENTATION,0.0}"}
    )
    speed: Optional[float] = field(
        default=0.0,
        metadata={"ge": 0.0, "description": "Single agent initial speed. Supports ${oc.env:AGENT_SPEED,0.0}"}
    )
    max_speed: Optional[float] = field(
        default=1.0,
        metadata={"gt": 0.0, "description": "Single agent maximum speed constraint. Supports ${oc.env:AGENT_MAX_SPEED,1.0}"}
    )
    angular_velocity: Optional[float] = field(
        default=0.0,
        metadata={"description": "Single agent angular velocity. Supports ${oc.env:AGENT_ANGULAR_VEL,0.0}"}
    )
    
    # Multi-agent parameters
    positions: Optional[List[List[float]]] = field(
        default=None,
        metadata={"description": "Multi-agent positions list. Presence triggers multi-agent mode"}
    )
    orientations: Optional[List[float]] = field(
        default=None,
        metadata={"description": "Multi-agent orientations list"}
    )
    speeds: Optional[List[float]] = field(
        default=None,
        metadata={"description": "Multi-agent speeds list"}
    )
    max_speeds: Optional[List[float]] = field(
        default=None,
        metadata={"description": "Multi-agent maximum speeds list"}
    )
    angular_velocities: Optional[List[float]] = field(
        default=None,
        metadata={"description": "Multi-agent angular velocities list"}
    )
    num_agents: Optional[int] = field(
        default=None,
        metadata={"gt": 0, "le": 100, "description": "Number of agents for multi-agent mode. Supports ${oc.env:NUM_AGENTS,10}"}
    )
    
    _target_: str = field(
        default="odor_plume_nav.api.create_navigator",
        metadata={"description": "Hydra target for factory instantiation"}
    )


@dataclass
class DataclassVideoPlumeConfig:
    """
    Dataclass configuration schema for video-based plume environment processing.
    
    Extends the domain VideoPlumeConfig with comprehensive validation, Hydra environment
    variable interpolation support, and advanced processing parameters for video file
    loading, preprocessing transformations, and frame analysis.
    """
    video_path: str = field(
        metadata={"description": "Path to video file. Supports ${oc.env:VIDEO_PATH} interpolation for deployment flexibility"}
    )
    flip: Optional[bool] = field(
        default=False,
        metadata={"description": "Apply horizontal flip transformation to video frames"}
    )
    grayscale: Optional[bool] = field(
        default=True,
        metadata={"description": "Convert video frames to grayscale for processing"}
    )
    kernel_size: Optional[int] = field(
        default=None,
        metadata={"description": "Gaussian kernel size for smoothing (must be odd and positive, or None to disable)"}
    )
    kernel_sigma: Optional[float] = field(
        default=None,
        metadata={"gt": 0.0, "description": "Gaussian kernel sigma parameter for smoothing intensity"}
    )
    threshold: Optional[float] = field(
        default=None,
        metadata={"ge": 0.0, "le": 1.0, "description": "Threshold value for binary frame processing (0.0-1.0 range)"}
    )
    normalize: Optional[bool] = field(
        default=True,
        metadata={"description": "Apply frame normalization for consistent value ranges"}
    )
    frame_skip: Optional[int] = field(
        default=0,
        metadata={"ge": 0, "description": "Number of frames to skip between processing steps"}
    )
    start_frame: Optional[int] = field(
        default=0,
        metadata={"ge": 0, "description": "Starting frame index for video processing"}
    )
    end_frame: Optional[int] = field(
        default=None,
        metadata={"ge": 0, "description": "Ending frame index (None for entire video)"}
    )
    fourcc: Optional[str] = field(
        default=None,
        metadata={"max_length": 4, "description": "Four-character code for video codec specification"}
    )
    fps_override: Optional[float] = field(
        default=None,
        metadata={"gt": 0.0, "description": "Override video FPS for simulation timing"}
    )
    _target_: str = field(
        default="odor_plume_nav.data.VideoPlume",
        metadata={"description": "Hydra target for automatic instantiation"}
    )


@dataclass 
class DataclassSimulationConfig:
    """
    Dataclass configuration schema for simulation execution parameters and output settings.
    
    Defines comprehensive simulation behavior including performance settings, output
    configuration, experiment tracking parameters, and Hydra integration for
    factory-driven component instantiation.
    """
    max_steps: int = field(
        default=1000,
        metadata={"gt": 0, "description": "Maximum number of simulation steps to execute"}
    )
    step_size: float = field(
        default=1.0,
        metadata={"gt": 0.0, "description": "Time step size for simulation progression"}
    )
    enable_gpu: bool = field(
        default=False,
        metadata={"description": "Enable GPU acceleration for computations when available"}
    )
    batch_size: int = field(
        default=1,
        metadata={"gt": 0, "description": "Batch size for parallel processing operations"}
    )
    num_workers: int = field(
        default=1,
        metadata={"ge": 1, "description": "Number of worker processes for parallel execution"}
    )
    record_trajectory: bool = field(
        default=True,
        metadata={"description": "Enable trajectory data recording during simulation"}
    )
    output_format: OutputFormat = field(
        default=OutputFormat.NUMPY,
        metadata={"description": "Output format for trajectory and results data"}
    )
    output_directory: str = field(
        default="./outputs",
        metadata={"description": "Directory for simulation output files. Supports ${oc.env:OUTPUT_DIR} interpolation"}
    )
    enable_visualization: bool = field(
        default=True,
        metadata={"description": "Enable real-time visualization during simulation"}
    )
    visualization_fps: float = field(
        default=30.0,
        metadata={"gt": 0.0, "le": 120.0, "description": "Target FPS for visualization rendering"}
    )
    save_animation: bool = field(
        default=False,
        metadata={"description": "Save simulation animation to video file"}
    )
    experiment_name: Optional[str] = field(
        default=None,
        metadata={"description": "Name for experiment tracking and output organization"}
    )
    random_seed: Optional[int] = field(
        default=None,
        metadata={"ge": 0, "description": "Random seed for reproducible simulations. Supports ${oc.env:RANDOM_SEED}"}
    )
    checkpoint_interval: int = field(
        default=100,
        metadata={"gt": 0, "description": "Number of steps between simulation checkpoints"}
    )
    enable_logging: bool = field(
        default=True,
        metadata={"description": "Enable detailed simulation logging"}
    )
    log_level: LogLevel = field(
        default=LogLevel.INFO,
        metadata={"description": "Logging verbosity level"}
    )
    _target_: str = field(
        default="odor_plume_nav.api.run_simulation",
        metadata={"description": "Hydra target for simulation execution"}
    )


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
    Initialize Hydra ConfigStore and register dataclass schemas.
    
    Registers dataclass-based configuration schemas with comprehensive validation for:
    - navigator: NavigatorConfig, SingleAgentConfig, MultiAgentConfig (dataclass schemas)
    - video_plume: VideoPlumeConfig (dataclass schema)
    - simulation: Composite configuration schemas with dataclass validation
    
    This function supports both dataclass instances and DictConfig handling while
    maintaining backward compatibility with existing YAML configuration loading.
    
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
        
        # Register dataclass-based navigator configurations with comprehensive validation
        cs.store(name="navigator_base", node=DataclassNavigatorConfig, group="navigator")
        cs.store(name="single_agent", node=DataclassSingleAgentConfig, group="navigator") 
        cs.store(name="multi_agent", node=DataclassMultiAgentConfig, group="navigator")
        
        # Register dataclass-based video plume configuration
        cs.store(name="video_plume_base", node=DataclassVideoPlumeConfig, group="video_plume")
        
        # Register dataclass-based simulation configuration
        cs.store(name="simulation_base", node=DataclassSimulationConfig, group="simulation")
        
        # Register composite schemas for common use cases with dataclass instances
        cs.store(
            name="simulation_unified",
            node={
                "navigator": DataclassNavigatorConfig(),
                "video_plume": DataclassVideoPlumeConfig(video_path="${oc.env:VIDEO_PATH,./data/default.mp4}"),
                "simulation": DataclassSimulationConfig(),
            },
            group="simulation"
        )
        
        cs.store(
            name="single_agent_simulation", 
            node={
                "navigator": DataclassSingleAgentConfig(),
                "video_plume": DataclassVideoPlumeConfig(video_path="${oc.env:VIDEO_PATH,./data/default.mp4}"),
                "simulation": DataclassSimulationConfig(),
            },
            group="simulation"
        )
        
        cs.store(
            name="multi_agent_simulation",
            node={
                "navigator": DataclassMultiAgentConfig(),
                "video_plume": DataclassVideoPlumeConfig(video_path="${oc.env:VIDEO_PATH,./data/default.mp4}"),
                "simulation": DataclassSimulationConfig(),
            },
            group="simulation"
        )
        
        # Register backward compatibility schemas with Pydantic model support
        cs.store(name="navigator_legacy", node=NavigatorConfig, group="navigator")
        cs.store(name="single_agent_legacy", node=SingleAgentConfig, group="navigator") 
        cs.store(name="multi_agent_legacy", node=MultiAgentConfig, group="navigator")
        cs.store(name="video_plume_legacy", node=VideoPlumeConfig, group="video_plume")
        
        logger.info("Hydra ConfigStore initialized with dataclass schemas and backward compatibility support")
        return cs
        
    except Exception as e:
        logger.error(f"Failed to initialize Hydra ConfigStore: {e}")
        raise


def compose_config_from_overrides(
    config_name: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    return_hydra_config: bool = False,
    validate_dataclass_fields: bool = True
) -> Union[Dict[str, Any], DictConfig]:
    """
    Compose configuration programmatically from Hydra with overrides and dataclass validation.
    
    This function supports both dataclass instances and DictConfig handling while maintaining
    backward compatibility with existing YAML configuration loading. It includes comprehensive
    dataclass field validation during configuration composition and override application.
    
    Args:
        config_name: Name of the config to compose (defaults to 'config')
        overrides: List of override strings in Hydra format (e.g., ['navigator.speed=2.0'])
        return_hydra_config: Whether to return DictConfig or plain dict
        validate_dataclass_fields: Whether to perform dataclass field validation
        
    Returns:
        Composed configuration as dictionary or DictConfig with dataclass validation applied
        
    Raises:
        ImportError: If Hydra is not available
        ValueError: If configuration composition fails or dataclass validation fails
    """
    if not HYDRA_AVAILABLE:
        raise ImportError(
            "Hydra is required for configuration composition. "
            "Install with: pip install hydra-core"
        )
    
    try:
        # Use compose API for programmatic configuration
        cfg = compose(config_name=config_name or "config", overrides=overrides or [])
        
        # Apply dataclass field validation if requested
        if validate_dataclass_fields:
            cfg = _apply_dataclass_validation_to_config(cfg)
        
        if return_hydra_config:
            return cfg
        else:
            return OmegaConf.to_object(cfg)
            
    except Exception as e:
        logger.error(f"Failed to compose configuration: {e}")
        raise ValueError(f"Configuration composition failed: {e}")


def _apply_dataclass_validation_to_config(cfg: DictConfig) -> DictConfig:
    """
    Apply dataclass field validation to a composed configuration.
    
    This function validates configuration sections against their corresponding
    dataclass schemas, including environment variable resolution and field
    constraint validation.
    
    Args:
        cfg: Hydra DictConfig to validate
        
    Returns:
        Validated DictConfig with environment variables resolved
        
    Raises:
        ValueError: If validation fails
    """
    try:
        # Map of configuration sections to their corresponding dataclass types
        dataclass_mapping = {
            "navigator": DataclassNavigatorConfig,
            "video_plume": DataclassVideoPlumeConfig,
            "simulation": DataclassSimulationConfig,
        }
        
        # Validate each section that has a corresponding dataclass
        for section_name, dataclass_type in dataclass_mapping.items():
            if section_name in cfg:
                section_config = OmegaConf.to_object(cfg[section_name])
                
                # Validate and resolve environment variables in this section
                validated_config = validate_dataclass_env_interpolation(
                    dataclass_type, section_config
                )
                
                # Update the configuration with validated values
                cfg[section_name] = OmegaConf.create(validated_config)
                
                logger.debug(f"Applied dataclass validation to {section_name} section")
        
        return cfg
        
    except Exception as e:
        logger.error(f"Dataclass validation failed: {e}")
        raise ValueError(f"Configuration validation failed: {e}")


def create_dataclass_config_with_env_resolution(
    config_type: str = "single_agent",
    env_overrides: Optional[Dict[str, str]] = None,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Create a configuration using dataclass schemas with environment variable resolution.
    
    This function provides seamless integration between dataclass-based validation
    and environment variable loading, supporting the new structured config approach
    while maintaining compatibility with existing workflows.
    
    Args:
        config_type: Type of configuration ('single_agent', 'multi_agent', 'unified', 'simulation')
        env_overrides: Optional environment variable overrides for testing
        validate: Whether to perform full dataclass validation
        
    Returns:
        Configuration dictionary with resolved environment variables and validated fields
        
    Raises:
        ValueError: If config_type is invalid or validation fails
    """
    # Temporarily set environment overrides for testing
    original_env = {}
    if env_overrides:
        for key, value in env_overrides.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
    
    try:
        # Create configuration based on type using dataclass schemas
        if config_type == "single_agent":
            config = {
                "navigator": create_dataclass_instance_with_env_resolution(
                    DataclassSingleAgentConfig, {}
                ),
                "video_plume": create_dataclass_instance_with_env_resolution(
                    DataclassVideoPlumeConfig, {"video_path": "${oc.env:VIDEO_PATH,./data/default.mp4}"}
                ),
                "simulation": create_dataclass_instance_with_env_resolution(
                    DataclassSimulationConfig, {}
                ),
            }
        elif config_type == "multi_agent":
            config = {
                "navigator": create_dataclass_instance_with_env_resolution(
                    DataclassMultiAgentConfig, {}
                ),
                "video_plume": create_dataclass_instance_with_env_resolution(
                    DataclassVideoPlumeConfig, {"video_path": "${oc.env:VIDEO_PATH,./data/default.mp4}"}
                ),
                "simulation": create_dataclass_instance_with_env_resolution(
                    DataclassSimulationConfig, {}
                ),
            }
        elif config_type == "unified":
            config = {
                "navigator": create_dataclass_instance_with_env_resolution(
                    DataclassNavigatorConfig, {}
                ),
                "video_plume": create_dataclass_instance_with_env_resolution(
                    DataclassVideoPlumeConfig, {"video_path": "${oc.env:VIDEO_PATH,./data/default.mp4}"}
                ),
                "simulation": create_dataclass_instance_with_env_resolution(
                    DataclassSimulationConfig, {}
                ),
            }
        elif config_type == "simulation":
            config = {
                "simulation": create_dataclass_instance_with_env_resolution(
                    DataclassSimulationConfig, {}
                ),
            }
        else:
            raise ValueError(
                f"Invalid config_type '{config_type}'. "
                f"Must be one of: 'single_agent', 'multi_agent', 'unified', 'simulation'"
            )
        
        # Convert dataclass instances to dictionaries for serialization
        def convert_dataclass_to_dict(obj):
            # Handle Enum values first to preserve primitive compatibility
            from enum import Enum as _Enum
            if hasattr(obj, "value") and isinstance(obj, _Enum):
                return obj.value
            if hasattr(obj, '__dataclass_fields__'):
                # This is a dataclass instance, convert to dict
                from dataclasses import asdict
                return asdict(obj)
            elif isinstance(obj, dict):
                return {k: convert_dataclass_to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        config = convert_dataclass_to_dict(config)
        
        # Add metadata
        config["_metadata"] = {
            "created_by": "odor_plume_nav.config.create_dataclass_config_with_env_resolution",
            "config_type": config_type,
            "version": "2.0",
            "schema_type": "dataclass",
        }
        
        logger.debug(f"Created dataclass-based {config_type} configuration with environment resolution")
        return config
        
    finally:
        # Restore original environment variables
        if env_overrides:
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value


def create_default_config(
    navigator_type: str = "single_agent",
    validate: bool = True,
    use_dataclass: bool = True
) -> Dict[str, Any]:
    """
    Create a default configuration for common use cases with backward compatibility.
    
    This function supports both dataclass-based and legacy Pydantic-based configuration
    creation to ensure seamless migration while maintaining existing functionality.
    
    Args:
        navigator_type: Type of navigator ('single_agent', 'multi_agent', 'base', or 'unified')
        validate: Whether to validate the created configuration
        use_dataclass: Whether to use dataclass schemas (True) or legacy Pydantic models (False)
        
    Returns:
        Default configuration dictionary
        
    Raises:
        ValueError: If navigator_type is invalid or validation fails
    """
    if use_dataclass:
        # Use new dataclass-based configuration with environment variable support
        try:
            config_type_mapping = {
                "single_agent": "single_agent",
                "multi_agent": "multi_agent", 
                "base": "unified",
                "unified": "unified"
            }
            
            if navigator_type not in config_type_mapping:
                raise ValueError(
                    f"Invalid navigator_type '{navigator_type}'. "
                    f"Must be one of: {list(config_type_mapping.keys())}"
                )
            
            config = create_dataclass_config_with_env_resolution(
                config_type=config_type_mapping[navigator_type],
                validate=validate
            )
            
            logger.debug(f"Created dataclass-based default {navigator_type} configuration")
            return config
            
        except Exception as e:
            logger.warning(f"Dataclass configuration creation failed, falling back to legacy: {e}")
            use_dataclass = False
    
    if not use_dataclass:
        # Legacy Pydantic-based configuration for backward compatibility
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
        elif navigator_type in ["base", "unified"]:
            config = {
                "navigator": NavigatorConfig().dict(),
                "video_plume": VideoPlumeConfig().dict(),
            }
        else:
            raise ValueError(
                f"Invalid navigator_type '{navigator_type}'. "
                f"Must be one of: 'single_agent', 'multi_agent', 'base', 'unified'"
            )
        
        # Add metadata
        config["_metadata"] = {
            "created_by": "odor_plume_nav.config.create_default_config",
            "navigator_type": navigator_type,
            "version": "1.0",
            "schema_type": "pydantic_legacy",
        }
        
        # Validate if requested using legacy validation
        if validate:
            try:
                if "navigator" in config:
                    validate_navigator_config(config)
                if "video_plume" in config:
                    validate_video_plume_config(config)
            except ConfigValidationError as e:
                logger.error(f"Default configuration validation failed: {e}")
                raise ValueError(f"Default configuration is invalid: {e}")
        
        logger.debug(f"Created legacy Pydantic-based default {navigator_type} configuration")
        
    return config


def _initialize_module() -> None:
    """
    Initialize the configuration module automatically on import.
    
    This function:
    1. Loads environment variables from .env files with enhanced hierarchy support
    2. Initializes Hydra ConfigStore with dataclass schemas and backward compatibility
    3. Sets up logging for configuration operations with dataclass validation support
    4. Validates dataclass schema registration for environment variable integration
    
    Called automatically when the module is imported.
    """
    try:
        # Determine environment from ENVIRONMENT variable
        environment = os.environ.get("ENVIRONMENT")
        
        # Load environment variables with enhanced hierarchy support
        loaded_env_vars = load_environment_variables(environment=environment)
        
        # Initialize Hydra ConfigStore if available
        if HYDRA_AVAILABLE:
            config_store = initialize_hydra_config_store()
            
            # Validate dataclass schemas are properly registered
            if config_store:
                logger.debug("Validating dataclass schema registration for environment variable support")
                try:
                    # Test environment variable resolution with a sample dataclass
                    test_config = {"speed": "${oc.env:TEST_SPEED,1.0}"}
                    validate_dataclass_env_interpolation(DataclassSingleAgentConfig, test_config)
                    logger.debug("Dataclass environment variable integration validated successfully")
                except Exception as e:
                    logger.warning(f"Dataclass environment variable validation failed: {e}")
            
        else:
            logger.info(
                "Hydra not available. Install with 'pip install hydra-core' "
                "for advanced dataclass configuration features."
            )
        
        # Log initialization summary
        logger.info(
            f"Configuration module initialized with {len(loaded_env_vars)} environment variables "
            f"and {'dataclass' if HYDRA_AVAILABLE else 'legacy'} schema support"
        )
            
    except Exception as e:
        logger.warning(f"Configuration module initialization failed: {e}")


# Enhanced exports with dataclass schema support
__all__ = [
    # Legacy Pydantic configuration models (backward compatibility)
    "NavigatorConfig",
    "SingleAgentConfig", 
    "MultiAgentConfig",
    "VideoPlumeConfig",
    
    # Dataclass configuration schemas (new structured config approach)
    "DataclassNavigatorConfig",
    "DataclassSingleAgentConfig",
    "DataclassMultiAgentConfig", 
    "DataclassVideoPlumeConfig",
    "DataclassSimulationConfig",
    
    # Configuration utilities
    "load_config",
    "save_config",
    "validate_config",
    "update_config",
    "get_config_dir",
    "load_yaml_config",
    "load_file_config",
    
    # Dataclass-enhanced configuration utilities
    "get_dataclass_fields_info",
    "validate_dataclass_env_interpolation",
    "create_dataclass_instance_with_env_resolution",
    "validate_env_interpolation",
    "resolve_env_value",
    
    # Validation utilities
    "ConfigValidationError",
    "validate_video_plume_config",
    "validate_navigator_config",
    
    # Advanced configuration functions with dataclass support
    "load_environment_variables",
    "initialize_hydra_config_store",
    "compose_config_from_overrides",
    "create_default_config",
    "create_dataclass_config_with_env_resolution",
]

# Automatically initialize the module when imported
_initialize_module()
