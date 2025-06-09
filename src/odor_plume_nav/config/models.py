"""
Enhanced Pydantic configuration models with Hydra integration and advanced validation.

This module provides comprehensive configuration management through enhanced Pydantic BaseModel
schemas integrated with Hydra's structured configuration system. It consolidates configuration
models from the domain layer while adding advanced Hydra capabilities including ConfigStore
registration, environment variable interpolation, and hierarchical configuration composition.

The models support both backward compatibility with existing Pydantic validation and forward
compatibility with modern Hydra-based configuration workflows for factory-driven component
instantiation and type-safe configuration management.
"""

from typing import List, Optional, Tuple, Union, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
import logging
from pathlib import Path
import os
import re

# Import base models from domain layer for backward compatibility
from odor_plume_nav.domain.models import (
    NavigatorConfig as BaseNavigatorConfig,
    SingleAgentConfig as BaseSingleAgentConfig,
    MultiAgentConfig as BaseMultiAgentConfig,
    VideoPlumeConfig as BaseVideoPlumeConfig,
)

# Set up module logger
logger = logging.getLogger(__name__)


class SingleAgentConfig(BaseSingleAgentConfig):
    """
    Enhanced configuration schema for single agent navigation parameters.
    
    This model extends the domain SingleAgentConfig with advanced Hydra integration,
    environment variable interpolation support, and comprehensive field validation
    for type-safe configuration management in production environments.
    """
    
    # Core navigation parameters with enhanced Field metadata
    position: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Initial [x, y] position coordinates. Supports ${oc.env:AGENT_START_X,50.0} interpolation"
    )
    orientation: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=360.0,
        description="Initial orientation in degrees (0=right, 90=up). Supports ${oc.env:AGENT_ORIENTATION,0.0}"
    )
    speed: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        description="Initial speed in units per timestep. Supports ${oc.env:AGENT_SPEED,0.0}"
    )
    max_speed: Optional[float] = Field(
        default=1.0,
        gt=0.0,
        description="Maximum allowed speed constraint. Supports ${oc.env:AGENT_MAX_SPEED,1.0}"
    )
    angular_velocity: Optional[float] = Field(
        default=0.0,
        description="Angular velocity in degrees per timestep. Supports ${oc.env:AGENT_ANGULAR_VEL,0.0}"
    )
    
    # Hydra-specific _target_ metadata for factory-driven component instantiation
    _target_: str = Field(
        default="odor_plume_nav.core.controllers.SingleAgentController",
        description="Hydra target for automatic instantiation"
    )
    
    @field_validator('speed', 'max_speed')
    @classmethod
    def validate_speed_constraints(cls, v, info):
        """Validate speed parameters are non-negative with enhanced error messaging."""
        if v is not None and v < 0:
            raise ValueError(f"{info.field_name} must be non-negative, got {v}")
        return v
    
    @field_validator('orientation')
    @classmethod
    def validate_orientation_range(cls, v):
        """Validate orientation is within valid degree range."""
        if v is not None and not (0.0 <= v <= 360.0):
            raise ValueError(f"Orientation must be between 0.0 and 360.0 degrees, got {v}")
        return v
    
    @model_validator(mode="after")
    def validate_speed_relationship(self):
        """Ensure speed does not exceed max_speed when both are specified."""
        if (self.speed is not None and 
            self.max_speed is not None and 
            self.speed > self.max_speed):
            raise ValueError(f"Initial speed ({self.speed}) cannot exceed max_speed ({self.max_speed})")
        return self
    
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "position": [25.0, 50.0],
                    "orientation": 90.0,
                    "speed": 0.5,
                    "max_speed": 2.0,
                    "angular_velocity": 0.1
                }
            ]
        }
    )


class MultiAgentConfig(BaseMultiAgentConfig):
    """
    Enhanced configuration schema for multi-agent navigation scenarios.
    
    Defines parameters for coordinated multi-agent simulations with vectorized
    operations support and comprehensive validation. All list parameters must
    have consistent lengths equal to num_agents when specified.
    """
    
    # Multi-agent parameters with enhanced Field metadata and validation
    positions: Optional[List[List[float]]] = Field(
        default=None,
        description="List of [x, y] positions for each agent. Length must match num_agents"
    )
    orientations: Optional[List[float]] = Field(
        default=None,
        description="List of initial orientations in degrees for each agent"
    )
    speeds: Optional[List[float]] = Field(
        default=None,
        description="List of initial speeds for each agent"
    )
    max_speeds: Optional[List[float]] = Field(
        default=None,
        description="List of maximum speed constraints for each agent"
    )
    angular_velocities: Optional[List[float]] = Field(
        default=None,
        description="List of angular velocities in degrees per timestep for each agent"
    )
    num_agents: Optional[int] = Field(
        default=None,
        gt=0,
        le=100,
        description="Total number of agents in simulation. Supports ${oc.env:NUM_AGENTS,10}"
    )
    
    # Hydra-specific _target_ metadata for factory-driven component instantiation
    _target_: str = Field(
        default="odor_plume_nav.core.controllers.MultiAgentController",
        description="Hydra target for automatic instantiation"
    )
    
    @field_validator('orientations', 'speeds', 'max_speeds', 'angular_velocities')
    @classmethod
    def validate_numeric_lists(cls, v, info):
        """Validate that numeric list parameters contain only valid numbers with enhanced checks."""
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValueError(f"{info.field_name} must be a list")
        
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(f"{info.field_name}[{i}] must be numeric")
            
            # Enhanced validation for specific fields
            if info.field_name == 'orientations' and not (0 <= val <= 360):
                raise ValueError(f"Orientation {i} must be between 0 and 360 degrees, got {val}")
            elif info.field_name in ['speeds', 'max_speeds'] and val < 0:
                raise ValueError(f"{info.field_name}[{i}] must be non-negative, got {val}")
        
        return v
    
    @model_validator(mode="after")
    def validate_agent_parameter_consistency(self):
        """Ensure all agent parameter lists have consistent lengths with enhanced validation."""
        if self.positions is not None:
            n_agents = len(self.positions)
            
            # Validate or set num_agents
            if self.num_agents is None:
                self.num_agents = n_agents
            elif self.num_agents != n_agents:
                raise ValueError(f"num_agents ({self.num_agents}) does not match positions length ({n_agents})")
            
            # Validate all other list parameters have matching lengths
            for field_name in ['orientations', 'speeds', 'max_speeds', 'angular_velocities']:
                field_value = getattr(self, field_name)
                if field_value is not None and len(field_value) != n_agents:
                    raise ValueError(f"{field_name} length ({len(field_value)}) does not match number of agents ({n_agents})")
        
        # Validate speed constraints for each agent
        if self.speeds is not None and self.max_speeds is not None:
            for i, (speed, max_speed) in enumerate(zip(self.speeds, self.max_speeds)):
                if speed > max_speed:
                    raise ValueError(f"Agent {i} speed ({speed}) exceeds max_speed ({max_speed})")
        
        return self
    
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "num_agents": 3,
                    "positions": [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
                    "orientations": [0.0, 90.0, 180.0],
                    "speeds": [1.0, 1.5, 0.8],
                    "max_speeds": [2.0, 2.0, 2.0],
                    "angular_velocities": [0.1, 0.1, 0.1]
                }
            ]
        }
    )


class NavigatorConfig(BaseNavigatorConfig):
    """
    Enhanced unified configuration schema for navigation supporting both single and multi-agent scenarios.
    
    This is the primary configuration model that consolidates single and multi-agent parameters
    with intelligent mode detection, Hydra environment variable interpolation support, and
    hierarchical configuration composition for factory-driven component instantiation.
    """
    
    # Navigation mode specification with enhanced validation
    mode: Literal["single", "multi", "auto"] = Field(
        default="auto",
        description="Navigation mode: 'single', 'multi', or 'auto' for automatic detection"
    )
    
    # Single agent parameters with environment variable interpolation support
    position: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Single agent initial position. Supports ${oc.env:AGENT_POSITION} interpolation"
    )
    orientation: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=360.0,
        description="Single agent initial orientation in degrees. Supports ${oc.env:AGENT_ORIENTATION,0.0}"
    )
    speed: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        description="Single agent initial speed. Supports ${oc.env:AGENT_SPEED,0.0}"
    )
    max_speed: Optional[float] = Field(
        default=1.0,
        gt=0.0,
        description="Single agent maximum speed constraint. Supports ${oc.env:AGENT_MAX_SPEED,1.0}"
    )
    angular_velocity: Optional[float] = Field(
        default=0.0,
        description="Single agent angular velocity. Supports ${oc.env:AGENT_ANGULAR_VEL,0.0}"
    )
    
    # Multi-agent parameters with enhanced validation
    positions: Optional[List[List[float]]] = Field(
        default=None,
        description="Multi-agent positions list. Presence triggers multi-agent mode"
    )
    orientations: Optional[List[float]] = Field(
        default=None,
        description="Multi-agent orientations list"
    )
    speeds: Optional[List[float]] = Field(
        default=None,
        description="Multi-agent speeds list"
    )
    max_speeds: Optional[List[float]] = Field(
        default=None,
        description="Multi-agent maximum speeds list"
    )
    angular_velocities: Optional[List[float]] = Field(
        default=None,
        description="Multi-agent angular velocities list"
    )
    num_agents: Optional[int] = Field(
        default=None,
        gt=0,
        le=100,
        description="Number of agents for multi-agent mode. Supports ${oc.env:NUM_AGENTS,10}"
    )
    
    # Hydra-specific _target_ metadata for factory-driven component instantiation
    _target_: str = Field(
        default="odor_plume_nav.api.create_navigator",
        description="Hydra target for factory instantiation"
    )
    
    @model_validator(mode="after")
    def validate_navigation_mode_consistency(self):
        """
        Validate configuration consistency and automatically detect navigation mode.
        
        This validator implements intelligent mode detection and ensures parameter
        consistency between single and multi-agent configurations with enhanced
        validation for hierarchical configuration composition.
        """
        has_multi_params = self.positions is not None
        has_single_params = self.position is not None
        
        # Automatic mode detection when mode="auto"
        if self.mode == "auto":
            if has_multi_params and has_single_params:
                raise ValueError("Cannot specify both single-agent and multi-agent parameters simultaneously")
            elif has_multi_params:
                self.mode = "multi"
            else:
                self.mode = "single"
        
        # Validate mode-specific parameter consistency
        if self.mode == "single":
            if has_multi_params:
                raise ValueError("Single-agent mode cannot have multi-agent parameters (positions, etc.)")
            # Validate single-agent speed constraints
            if (self.speed is not None and 
                self.max_speed is not None and 
                self.speed > self.max_speed):
                raise ValueError(f"Single agent speed ({self.speed}) cannot exceed max_speed ({self.max_speed})")
        
        elif self.mode == "multi":
            if has_single_params:
                logger.warning("Multi-agent mode specified but single-agent parameters present. Multi-agent parameters will take precedence.")
            
            if not has_multi_params:
                raise ValueError("Multi-agent mode requires positions parameter")
            
            # Apply enhanced multi-agent validation logic
            n_agents = len(self.positions)
            if self.num_agents is None:
                self.num_agents = n_agents
            elif self.num_agents != n_agents:
                raise ValueError(f"num_agents ({self.num_agents}) does not match positions length ({n_agents})")
            
            # Validate list parameter lengths
            for param_name in ['orientations', 'speeds', 'max_speeds', 'angular_velocities']:
                param_value = getattr(self, param_name)
                if param_value is not None and len(param_value) != n_agents:
                    raise ValueError(f"{param_name} length ({len(param_value)}) does not match number of agents ({n_agents})")
            
            # Validate multi-agent speed constraints
            if self.speeds is not None and self.max_speeds is not None:
                for i, (speed, max_speed) in enumerate(zip(self.speeds, self.max_speeds)):
                    if speed > max_speed:
                        raise ValueError(f"Agent {i} speed ({speed}) exceeds max_speed ({max_speed})")
        
        return self
    
    def get_single_agent_config(self) -> SingleAgentConfig:
        """Extract single agent configuration parameters."""
        if self.mode != "single":
            raise ValueError("Cannot extract single agent config from multi-agent mode")
        
        return SingleAgentConfig(
            position=self.position,
            orientation=self.orientation,
            speed=self.speed,
            max_speed=self.max_speed,
            angular_velocity=self.angular_velocity
        )
    
    def get_multi_agent_config(self) -> MultiAgentConfig:
        """Extract multi-agent configuration parameters."""
        if self.mode != "multi":
            raise ValueError("Cannot extract multi-agent config from single-agent mode")
        
        return MultiAgentConfig(
            positions=self.positions,
            orientations=self.orientations,
            speeds=self.speeds,
            max_speeds=self.max_speeds,
            angular_velocities=self.angular_velocities,
            num_agents=self.num_agents
        )
    
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "mode": "single",
                    "position": [50.0, 50.0],
                    "orientation": 0.0,
                    "speed": 1.0,
                    "max_speed": 2.0,
                    "angular_velocity": 0.1
                },
                {
                    "mode": "multi",
                    "num_agents": 2,
                    "positions": [[20.0, 30.0], [70.0, 80.0]],
                    "orientations": [45.0, 135.0],
                    "speeds": [1.2, 0.8],
                    "max_speeds": [2.0, 2.0],
                    "angular_velocities": [0.1, 0.1]
                }
            ]
        }
    )


class VideoPlumeConfig(BaseVideoPlumeConfig):
    """
    Enhanced configuration schema for video-based plume environment processing.
    
    Extends the domain VideoPlumeConfig with comprehensive validation, Hydra environment
    variable interpolation support, and advanced processing parameters for video file
    loading, preprocessing transformations, and frame analysis.
    """
    
    # Core video file configuration with environment variable interpolation
    video_path: Union[str, Path] = Field(
        description="Path to video file. Supports ${oc.env:VIDEO_PATH} interpolation for deployment flexibility"
    )
    
    # Video preprocessing parameters with enhanced metadata
    flip: Optional[bool] = Field(
        default=False,
        description="Apply horizontal flip transformation to video frames"
    )
    grayscale: Optional[bool] = Field(
        default=True,
        description="Convert video frames to grayscale for processing"
    )
    
    # Gaussian filtering configuration with enhanced validation
    kernel_size: Optional[int] = Field(
        default=None,
        description="Gaussian kernel size for smoothing (must be odd and positive, or None to disable)"
    )
    kernel_sigma: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Gaussian kernel sigma parameter for smoothing intensity"
    )
    
    # Advanced processing parameters
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Threshold value for binary frame processing (0.0-1.0 range)"
    )
    normalize: Optional[bool] = Field(
        default=True,
        description="Apply frame normalization for consistent value ranges"
    )
    
    # Frame selection and sampling parameters
    frame_skip: Optional[int] = Field(
        default=0,
        ge=0,
        description="Number of frames to skip between processing steps"
    )
    start_frame: Optional[int] = Field(
        default=0,
        ge=0,
        description="Starting frame index for video processing"
    )
    end_frame: Optional[int] = Field(
        default=None,
        ge=0,
        description="Ending frame index (None for entire video)"
    )
    
    # Video format and codec parameters
    fourcc: Optional[str] = Field(
        default=None,
        max_length=4,
        description="Four-character code for video codec specification"
    )
    fps_override: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Override video FPS for simulation timing"
    )
    
    # Hydra-specific _target_ metadata for factory-driven component instantiation
    _target_: str = Field(
        default="odor_plume_nav.data.VideoPlume",
        description="Hydra target for automatic instantiation"
    )
    
    @field_validator('video_path')
    @classmethod
    def validate_video_path(cls, v):
        """Validate video path format and perform basic existence checking with environment variable support."""
        if isinstance(v, str):
            # Handle Hydra environment variable interpolation patterns
            if v.startswith('${oc.env:'):
                return v  # Skip validation for interpolated paths
            
            path = Path(v)
        else:
            path = v
        
        # Validate file extension for supported formats
        supported_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v'}
        if path.suffix.lower() not in supported_extensions:
            logger.warning(f"Video file extension '{path.suffix}' may not be supported. Supported: {supported_extensions}")
        
        return str(path)
    
    @model_validator(mode="after")
    def validate_gaussian_parameters(self):
        """Validate consistency of Gaussian filtering parameters with enhanced error messaging."""
        has_kernel_size = self.kernel_size is not None
        has_kernel_sigma = self.kernel_sigma is not None
        
        if has_kernel_size and not has_kernel_sigma:
            logger.warning("kernel_size specified without kernel_sigma. Using default sigma=1.0")
            self.kernel_sigma = 1.0
        elif has_kernel_sigma and not has_kernel_size:
            logger.warning("kernel_sigma specified without kernel_size. Using default size=5")
            self.kernel_size = 5
        
        return self
    
    @model_validator(mode="after")
    def validate_frame_range(self):
        """Validate frame selection parameters are consistent with enhanced validation."""
        if self.end_frame is not None and self.start_frame is not None and self.end_frame <= self.start_frame:
            raise ValueError(f"end_frame ({self.end_frame}) must be greater than start_frame ({self.start_frame})")
        
        return self
    
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "video_path": "/path/to/plume_video.mp4",
                    "flip": False,
                    "grayscale": True,
                    "kernel_size": 5,
                    "kernel_sigma": 1.0,
                    "normalize": True,
                    "frame_skip": 0
                },
                {
                    "video_path": "${oc.env:VIDEO_PATH,./data/default_plume.mp4}",
                    "flip": True,
                    "grayscale": True,
                    "threshold": 0.5,
                    "start_frame": 100,
                    "end_frame": 1000
                }
            ]
        }
    )


class SimulationConfig(BaseModel):
    """
    Enhanced configuration schema for simulation execution parameters and output settings.
    
    Defines comprehensive simulation behavior including performance settings, output
    configuration, experiment tracking parameters, and Hydra integration for
    factory-driven component instantiation.
    """
    
    # Core simulation parameters with enhanced validation
    max_steps: int = Field(
        default=1000,
        gt=0,
        description="Maximum number of simulation steps to execute"
    )
    step_size: float = Field(
        default=1.0,
        gt=0.0,
        description="Time step size for simulation progression"
    )
    
    # Performance and optimization settings
    enable_gpu: bool = Field(
        default=False,
        description="Enable GPU acceleration for computations when available"
    )
    batch_size: int = Field(
        default=1,
        gt=0,
        description="Batch size for parallel processing operations"
    )
    num_workers: int = Field(
        default=1,
        ge=1,
        description="Number of worker processes for parallel execution"
    )
    
    # Output and recording configuration with environment variable support
    record_trajectory: bool = Field(
        default=True,
        description="Enable trajectory data recording during simulation"
    )
    output_format: Literal["numpy", "csv", "hdf5", "json"] = Field(
        default="numpy",
        description="Output format for trajectory and results data"
    )
    output_directory: Union[str, Path] = Field(
        default="./outputs",
        description="Directory for simulation output files. Supports ${oc.env:OUTPUT_DIR} interpolation"
    )
    
    # Visualization settings
    enable_visualization: bool = Field(
        default=True,
        description="Enable real-time visualization during simulation"
    )
    visualization_fps: float = Field(
        default=30.0,
        gt=0.0,
        le=120.0,
        description="Target FPS for visualization rendering"
    )
    save_animation: bool = Field(
        default=False,
        description="Save simulation animation to video file"
    )
    
    # Experiment tracking and reproducibility with environment variable support
    experiment_name: Optional[str] = Field(
        default=None,
        description="Name for experiment tracking and output organization"
    )
    random_seed: Optional[int] = Field(
        default=None,
        ge=0,
        description="Random seed for reproducible simulations. Supports ${oc.env:RANDOM_SEED}"
    )
    
    # Advanced simulation features
    checkpoint_interval: int = Field(
        default=100,
        gt=0,
        description="Number of steps between simulation checkpoints"
    )
    enable_logging: bool = Field(
        default=True,
        description="Enable detailed simulation logging"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging verbosity level"
    )
    
    # Hydra-specific _target_ metadata for factory-driven component instantiation
    _target_: str = Field(
        default="odor_plume_nav.api.run_simulation",
        description="Hydra target for simulation execution"
    )
    
    @field_validator('output_directory')
    @classmethod
    def validate_output_directory(cls, v):
        """Validate and normalize output directory path with environment variable support."""
        if isinstance(v, str):
            # Handle Hydra environment variable interpolation
            if v.startswith('${oc.env:'):
                return v
            
            path = Path(v)
        else:
            path = v
        
        return str(path.resolve())
    
    @model_validator(mode="after")
    def validate_performance_settings(self):
        """Validate performance configuration consistency with enhanced warnings."""
        if self.enable_gpu and self.num_workers > 1:
            logger.warning("GPU acceleration with multiple workers may cause resource conflicts")
        
        if self.batch_size > 1 and self.enable_visualization:
            logger.warning("Batch processing with real-time visualization may impact performance")
        
        return self
    
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "max_steps": 500,
                    "step_size": 1.0,
                    "record_trajectory": True,
                    "output_format": "numpy",
                    "enable_visualization": True,
                    "random_seed": 42
                }
            ]
        }
    )


# Environment variable interpolation utility functions
def validate_env_interpolation(value: str) -> bool:
    """
    Validate Hydra environment variable interpolation syntax (${oc.env:VAR,DEFAULT}).
    
    Args:
        value: String value that may contain ${oc.env:} interpolation
        
    Returns:
        True if interpolation syntax is valid
        
    Examples:
        >>> validate_env_interpolation("${oc.env:VIDEO_PATH}")
        True
        >>> validate_env_interpolation("${oc.env:VIDEO_PATH,./default.mp4}")
        True
        >>> validate_env_interpolation("regular_string")
        False
    """
    pattern = r'\$\{oc\.env:([A-Z_][A-Z0-9_]*)(,.*?)?\}'
    return bool(re.match(pattern, value))


def resolve_env_value(value: str, default: Any = None) -> Any:
    """
    Resolve environment variable interpolation for testing and validation.
    
    Args:
        value: String value with potential ${oc.env:} interpolation
        default: Default value if environment variable not found
        
    Returns:
        Resolved value from environment or default
        
    Examples:
        >>> os.environ['TEST_VAR'] = 'test_value'
        >>> resolve_env_value("${oc.env:TEST_VAR}")
        'test_value'
        >>> resolve_env_value("${oc.env:MISSING_VAR,default_val}")
        'default_val'
    """
    pattern = r'\$\{oc\.env:([A-Z_][A-Z0-9_]*)(,.*?)?\}'
    match = re.match(pattern, value)
    
    if match:
        env_var = match.group(1)
        env_default = match.group(2)[1:] if match.group(2) else default
        return os.getenv(env_var, env_default)
    
    return value


def register_config_schemas():
    """
    Register all configuration schemas with Hydra ConfigStore for structured configuration.
    
    This function enables automatic schema discovery and validation within Hydra's
    configuration composition system. It registers schemas under appropriate groups:
    - navigator: single_agent, multi_agent, unified
    - video_plume: default  
    - simulation: standard
    
    Supports hierarchical configuration composition and factory-driven component
    instantiation per structured configuration requirements.
    """
    try:
        # Import ConfigStore only when needed to avoid circular imports
        from hydra.core.config_store import ConfigStore
        
        cs = ConfigStore.instance()
        
        # Register individual component schemas under appropriate groups
        cs.store(
            group="navigator",
            name="single_agent",
            node=SingleAgentConfig,
            package="navigator"
        )
        
        cs.store(
            group="navigator", 
            name="multi_agent",
            node=MultiAgentConfig,
            package="navigator"
        )
        
        cs.store(
            group="navigator",
            name="unified",
            node=NavigatorConfig,
            package="navigator"
        )
        
        cs.store(
            group="video_plume",
            name="default",
            node=VideoPlumeConfig,
            package="video_plume"
        )
        
        cs.store(
            group="simulation",
            name="standard",
            node=SimulationConfig,
            package="simulation"
        )
        
        # Register base configuration schema combining all components
        cs.store(
            name="base_config",
            node={
                "navigator": NavigatorConfig,
                "video_plume": VideoPlumeConfig, 
                "simulation": SimulationConfig
            }
        )
        
        logger.info("Successfully registered all configuration schemas with Hydra ConfigStore")
        
    except ImportError as e:
        logger.warning(f"Hydra not available, skipping ConfigStore registration: {e}")
    except Exception as e:
        logger.error(f"Failed to register configuration schemas: {e}")
        raise


# Enhanced exports for backward compatibility and new functionality
__all__ = [
    "NavigatorConfig",
    "SingleAgentConfig",
    "MultiAgentConfig", 
    "VideoPlumeConfig",
    "SimulationConfig",
    "register_config_schemas",
    "validate_env_interpolation",
    "resolve_env_value",
]
