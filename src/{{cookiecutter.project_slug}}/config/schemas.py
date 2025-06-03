"""
Unified Pydantic configuration schemas with Hydra integration.

This module provides comprehensive configuration management through Pydantic BaseModel
schemas integrated with Hydra's structured configuration system. It consolidates all
configuration models from the original domain/models.py structure while adding 
advanced Hydra capabilities including ConfigStore registration, environment variable
interpolation, and hierarchical configuration composition.

The schemas support both backward compatibility with existing Pydantic validation
and forward compatibility with modern Hydra-based configuration workflows.
"""

from typing import List, Optional, Tuple, Union, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from hydra.core.config_store import ConfigStore
from hydra.core.defaults_list import create_defaults_list
from hydra.types import ConfigGroup
import logging
from pathlib import Path
import os

# Set up module logger
logger = logging.getLogger(__name__)

# Initialize Hydra ConfigStore for automatic schema registration
cs = ConfigStore.instance()


class SingleAgentConfig(BaseModel):
    """
    Configuration schema for single agent navigation parameters.
    
    This model defines the complete parameter set for single-agent scenarios
    including position, orientation, speed, and angular velocity parameters.
    Supports Hydra environment variable interpolation for deployment flexibility.
    """
    
    # Core navigation parameters
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
    
    # Hydra-specific configuration metadata
    _target_: str = Field(
        default="{{cookiecutter.project_slug}}.core.controllers.SingleAgentController",
        description="Hydra target for automatic instantiation"
    )
    
    @field_validator('speed', 'max_speed')
    @classmethod
    def validate_speed_constraints(cls, v, info):
        """Validate speed parameters are non-negative."""
        if v is not None and v < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
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


class MultiAgentConfig(BaseModel):
    """
    Configuration schema for multi-agent navigation scenarios.
    
    Defines parameters for coordinated multi-agent simulations with vectorized
    operations support. All list parameters must have consistent lengths equal
    to num_agents when specified.
    """
    
    # Multi-agent parameters with length validation
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
    
    # Hydra-specific configuration metadata
    _target_: str = Field(
        default="{{cookiecutter.project_slug}}.core.controllers.MultiAgentController",
        description="Hydra target for automatic instantiation"
    )
    
    @field_validator('positions')
    @classmethod
    def validate_positions_format(cls, v):
        """Validate that positions is a properly formatted list of [x, y] coordinates."""
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValueError("positions must be a list of coordinate pairs")
        
        for i, pos in enumerate(v):
            if not isinstance(pos, (list, tuple)) or len(pos) != 2:
                raise ValueError(f"Position {i} must be a [x, y] coordinate pair")
            if not all(isinstance(coord, (int, float)) for coord in pos):
                raise ValueError(f"Position {i} coordinates must be numeric")
        return v
    
    @field_validator('orientations', 'speeds', 'max_speeds', 'angular_velocities')
    @classmethod
    def validate_numeric_lists(cls, v, info):
        """Validate that numeric list parameters contain only valid numbers."""
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValueError(f"{info.field_name} must be a list")
        
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(f"{info.field_name}[{i}] must be numeric")
            
            # Additional validation for specific fields
            if info.field_name == 'orientations' and not (0 <= val <= 360):
                raise ValueError(f"Orientation {i} must be between 0 and 360 degrees")
            elif info.field_name in ['speeds', 'max_speeds'] and val < 0:
                raise ValueError(f"{info.field_name}[{i}] must be non-negative")
        
        return v
    
    @model_validator(mode="after")
    def validate_agent_parameter_consistency(self):
        """Ensure all agent parameter lists have consistent lengths."""
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


class NavigatorConfig(BaseModel):
    """
    Unified configuration schema for navigation that supports both single and multi-agent scenarios.
    
    This is the primary configuration model that consolidates single and multi-agent
    parameters with intelligent mode detection. It supports Hydra environment variable
    interpolation and hierarchical configuration composition.
    """
    
    # Navigation mode specification
    mode: Literal["single", "multi", "auto"] = Field(
        default="auto",
        description="Navigation mode: 'single', 'multi', or 'auto' for automatic detection"
    )
    
    # Single agent parameters (when mode="single" or auto-detected)
    position: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Single agent initial position. Supports ${oc.env:AGENT_POSITION} interpolation"
    )
    orientation: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=360.0,
        description="Single agent initial orientation in degrees"
    )
    speed: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        description="Single agent initial speed"
    )
    max_speed: Optional[float] = Field(
        default=1.0,
        gt=0.0,
        description="Single agent maximum speed constraint"
    )
    angular_velocity: Optional[float] = Field(
        default=0.0,
        description="Single agent angular velocity in degrees per timestep"
    )
    
    # Multi-agent parameters (when mode="multi" or auto-detected)
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
        description="Number of agents for multi-agent mode"
    )
    
    # Hydra-specific configuration metadata
    _target_: str = Field(
        default="{{cookiecutter.project_slug}}.api.navigation.create_navigator",
        description="Hydra target for factory instantiation"
    )
    
    @model_validator(mode="after")
    def validate_navigation_mode_consistency(self):
        """
        Validate configuration consistency and automatically detect navigation mode.
        
        This validator implements intelligent mode detection and ensures parameter
        consistency between single and multi-agent configurations.
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
            
            # Apply multi-agent validation logic
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


class VideoPlumeConfig(BaseModel):
    """
    Comprehensive configuration schema for video-based plume environment processing.
    
    Defines parameters for video file loading, preprocessing transformations,
    and frame analysis. Supports Hydra environment variable interpolation for
    secure path and credential management.
    """
    
    # Core video file configuration
    video_path: Union[str, Path] = Field(
        description="Path to video file. Supports ${oc.env:VIDEO_PATH} interpolation for deployment flexibility"
    )
    
    # Video preprocessing parameters
    flip: bool = Field(
        default=False,
        description="Apply horizontal flip transformation to video frames"
    )
    grayscale: bool = Field(
        default=True,
        description="Convert video frames to grayscale for processing"
    )
    
    # Gaussian filtering configuration
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
    normalize: bool = Field(
        default=True,
        description="Apply frame normalization for consistent value ranges"
    )
    
    # Frame selection and sampling
    frame_skip: int = Field(
        default=0,
        ge=0,
        description="Number of frames to skip between processing steps"
    )
    start_frame: int = Field(
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
    
    # Hydra-specific configuration metadata
    _target_: str = Field(
        default="{{cookiecutter.project_slug}}.data.video_plume.VideoPlume",
        description="Hydra target for automatic instantiation"
    )
    
    @field_validator('video_path')
    @classmethod
    def validate_video_path(cls, v):
        """Validate video path format and perform basic existence checking."""
        if isinstance(v, str):
            # Handle Hydra environment variable interpolation patterns
            if v.startswith('${oc.env:'):
                return v  # Skip validation for interpolated paths
            
            path = Path(v)
        else:
            path = v
        
        # Validate file extension for supported formats
        supported_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        if path.suffix.lower() not in supported_extensions:
            logger.warning(f"Video file extension '{path.suffix}' may not be supported. Supported: {supported_extensions}")
        
        return str(path)
    
    @field_validator('kernel_size')
    @classmethod
    def validate_kernel_size(cls, v):
        """Validate Gaussian kernel size is odd and positive."""
        if v is None:
            return v
        
        if v <= 0:
            raise ValueError("kernel_size must be positive")
        if v % 2 == 0:
            raise ValueError("kernel_size must be odd for proper Gaussian filtering")
        
        return v
    
    @model_validator(mode="after")
    def validate_gaussian_parameters(self):
        """Validate consistency of Gaussian filtering parameters."""
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
        """Validate frame selection parameters are consistent."""
        if self.end_frame is not None and self.end_frame <= self.start_frame:
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
    Configuration schema for simulation execution parameters and output settings.
    
    Defines comprehensive simulation behavior including performance settings,
    output configuration, and experiment tracking parameters.
    """
    
    # Core simulation parameters
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
    
    # Output and recording configuration
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
    
    # Experiment tracking and reproducibility
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
    
    # Hydra-specific configuration metadata
    _target_: str = Field(
        default="{{cookiecutter.project_slug}}.api.navigation.run_simulation",
        description="Hydra target for simulation execution"
    )
    
    @field_validator('output_directory')
    @classmethod
    def validate_output_directory(cls, v):
        """Validate and normalize output directory path."""
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
        """Validate performance configuration consistency."""
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


# Hydra ConfigStore registration for structured configuration
def register_config_schemas():
    """
    Register all configuration schemas with Hydra ConfigStore.
    
    This function enables automatic schema discovery and validation within
    Hydra's configuration composition system. It should be called during
    module initialization to ensure schemas are available for use.
    """
    
    try:
        # Register individual component schemas
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
        
    except Exception as e:
        logger.error(f"Failed to register configuration schemas: {e}")
        raise


# Environment variable interpolation helpers
def validate_env_interpolation(value: str) -> bool:
    """
    Validate Hydra environment variable interpolation syntax.
    
    Args:
        value: String value that may contain ${oc.env:} interpolation
        
    Returns:
        True if interpolation syntax is valid
    """
    import re
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
    """
    import re
    
    pattern = r'\$\{oc\.env:([A-Z_][A-Z0-9_]*)(,.*?)?\}'
    match = re.match(pattern, value)
    
    if match:
        env_var = match.group(1)
        env_default = match.group(2)[1:] if match.group(2) else default
        return os.getenv(env_var, env_default)
    
    return value


# Backward compatibility exports
__all__ = [
    "NavigatorConfig",
    "SingleAgentConfig", 
    "MultiAgentConfig",
    "VideoPlumeConfig",
    "SimulationConfig",
    "register_config_schemas",
    "validate_env_interpolation",
    "resolve_env_value"
]

# Automatic schema registration on module import
if cs is not None:
    try:
        register_config_schemas()
    except Exception as e:
        logger.warning(f"Schema registration deferred due to initialization order: {e}")