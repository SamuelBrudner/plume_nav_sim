"""
Unified Pydantic configuration schemas with Hydra ConfigStore integration.

This module consolidates all configuration models for the odor plume navigation
system, providing type-safe parameter validation, environment variable interpolation,
and automatic Hydra registration for structured configuration composition.

The schemas support hierarchical configuration through base.yaml → config.yaml → 
local overrides with environment variable substitution via ${oc.env:VAR_NAME} syntax.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

# Hydra imports for ConfigStore integration
try:
    from hydra.core.config_store import ConfigStore
    from omegaconf import MISSING, DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    # Fallback for environments without Hydra
    ConfigStore = None
    MISSING = "???"
    DictConfig = dict


class SingleAgentConfig(BaseModel):
    """
    Configuration schema for single agent navigation with Hydra integration.
    
    Supports environment variable interpolation through ${oc.env:VAR_NAME} syntax
    for secure credential management and deployment flexibility.
    
    Examples:
        Basic configuration:
            position: [0.0, 0.0]
            orientation: 0.0
            speed: 1.0
            max_speed: 2.0
            
        With environment variables:
            max_speed: ${oc.env:AGENT_MAX_SPEED,1.0}
            angular_velocity: ${oc.env:AGENT_ANGULAR_VEL,0.0}
    """
    
    # Position coordinates [x, y]
    position: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Initial agent position coordinates [x, y]"
    )
    
    # Orientation in degrees (0 = right, 90 = up)
    orientation: Optional[float] = Field(
        default=0.0,
        description="Initial orientation in degrees (0 = right, 90 = up)"
    )
    
    # Movement parameters
    speed: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        description="Initial speed in units per time step"
    )
    
    max_speed: Optional[float] = Field(
        default=1.0,
        gt=0.0,
        description="Maximum allowed speed in units per time step"
    )
    
    angular_velocity: Optional[float] = Field(
        default=0.0,
        description="Angular velocity in degrees per second"
    )

    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
        validate_assignment=True,
        # Support for Hydra interpolation
        arbitrary_types_allowed=True
    )

    @field_validator('orientation')
    @classmethod
    def normalize_orientation(cls, v):
        """Normalize orientation to [0, 360) range."""
        if v is not None:
            return v % 360.0
        return v

    @model_validator(mode="after")
    def validate_speed_constraints(self):
        """Ensure speed does not exceed max_speed."""
        if (self.speed is not None and 
            self.max_speed is not None and 
            self.speed > self.max_speed):
            raise ValueError(
                f"speed ({self.speed}) cannot exceed max_speed ({self.max_speed})"
            )
        return self


class MultiAgentConfig(BaseModel):
    """
    Configuration schema for multi-agent navigation with enhanced validation.
    
    Supports environment variable interpolation for scalable deployment scenarios
    and hierarchical configuration composition through Hydra config groups.
    
    Examples:
        Basic multi-agent setup:
            num_agents: 3
            positions: [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
            orientations: [0.0, 45.0, 90.0]
            speeds: [1.0, 1.0, 1.0]
            
        With environment scaling:
            num_agents: ${oc.env:SWARM_SIZE,5}
            max_speeds: ${oc.env:AGENT_MAX_SPEEDS,[1.0,1.0,1.0,1.0,1.0]}
    """
    
    # Agent positions as list of [x, y] coordinates
    positions: Optional[List[List[float]]] = Field(
        default=None,
        description="List of initial agent positions as [x, y] coordinates"
    )
    
    # Per-agent orientations in degrees
    orientations: Optional[List[float]] = Field(
        default=None,
        description="List of initial orientations in degrees for each agent"
    )
    
    # Per-agent speeds
    speeds: Optional[List[float]] = Field(
        default=None,
        description="List of initial speeds for each agent"
    )
    
    # Per-agent maximum speeds
    max_speeds: Optional[List[float]] = Field(
        default=None,
        description="List of maximum speeds for each agent"
    )
    
    # Per-agent angular velocities
    angular_velocities: Optional[List[float]] = Field(
        default=None,
        description="List of angular velocities in degrees per second"
    )
    
    # Number of agents (for validation)
    num_agents: Optional[int] = Field(
        default=None,
        gt=0,
        description="Total number of agents in the simulation"
    )

    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
        validate_assignment=True,
        arbitrary_types_allowed=True
    )

    @field_validator('positions')
    @classmethod
    def validate_positions_structure(cls, v):
        """Validate that positions is a list of [x, y] coordinates."""
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValueError("positions must be a list")
        for i, pos in enumerate(v):
            if not isinstance(pos, (list, tuple)) or len(pos) != 2:
                raise ValueError(
                    f"Position {i} must be a list/tuple of [x, y] coordinates"
                )
            if not all(isinstance(coord, (int, float)) for coord in pos):
                raise ValueError(
                    f"Position {i} coordinates must be numeric"
                )
        return v

    @field_validator('orientations')
    @classmethod
    def normalize_orientations(cls, v):
        """Normalize all orientations to [0, 360) range."""
        if v is not None:
            return [orientation % 360.0 for orientation in v]
        return v

    @field_validator('speeds', 'max_speeds')
    @classmethod
    def validate_positive_speeds(cls, v):
        """Ensure all speeds are non-negative."""
        if v is not None:
            for i, speed in enumerate(v):
                if speed < 0:
                    raise ValueError(f"Speed at index {i} must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_agent_parameter_consistency(self):
        """Ensure all agent parameter lists have consistent lengths."""
        if self.positions is not None:
            n_agents = len(self.positions)
            
            # Validate number of agents matches positions
            if self.num_agents is not None and self.num_agents != n_agents:
                raise ValueError(
                    f"num_agents ({self.num_agents}) does not match "
                    f"number of positions ({n_agents})"
                )
            
            # Validate all parameter lists have matching lengths
            param_lists = {
                'orientations': self.orientations,
                'speeds': self.speeds,
                'max_speeds': self.max_speeds,
                'angular_velocities': self.angular_velocities
            }
            
            for param_name, param_list in param_lists.items():
                if param_list is not None and len(param_list) != n_agents:
                    raise ValueError(
                        f"{param_name} length ({len(param_list)}) does not match "
                        f"number of agents ({n_agents})"
                    )
        
        # Validate speed constraints for each agent
        if self.speeds is not None and self.max_speeds is not None:
            for i, (speed, max_speed) in enumerate(zip(self.speeds, self.max_speeds)):
                if speed > max_speed:
                    raise ValueError(
                        f"Agent {i}: speed ({speed}) cannot exceed "
                        f"max_speed ({max_speed})"
                    )
        
        return self


class NavigatorConfig(BaseModel):
    """
    Unified configuration schema for navigator supporting both single and multi-agent modes.
    
    This schema automatically detects the navigation mode based on provided parameters
    and delegates to appropriate validation logic. Supports environment variable 
    interpolation and hierarchical Hydra configuration composition.
    
    Examples:
        Single agent mode:
            position: [0.0, 0.0]
            orientation: 0.0
            speed: 1.0
            max_speed: 2.0
            
        Multi-agent mode:
            positions: [[0.0, 0.0], [1.0, 0.0]]
            orientations: [0.0, 90.0]
            speeds: [1.0, 1.2]
            num_agents: 2
            
        With environment variables:
            max_speed: ${oc.env:NAVIGATOR_MAX_SPEED,1.0}
            num_agents: ${oc.env:SWARM_SIZE,1}
    """
    
    # Single agent parameters
    position: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Single agent initial position [x, y]"
    )
    orientation: Optional[float] = Field(
        default=0.0,
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
        description="Single agent maximum speed"
    )
    angular_velocity: Optional[float] = Field(
        default=0.0,
        description="Single agent angular velocity in degrees per second"
    )
    
    # Multi-agent parameters
    positions: Optional[List[List[float]]] = Field(
        default=None,
        description="Multi-agent initial positions as [x, y] coordinates"
    )
    orientations: Optional[List[float]] = Field(
        default=None,
        description="Multi-agent initial orientations in degrees"
    )
    speeds: Optional[List[float]] = Field(
        default=None,
        description="Multi-agent initial speeds"
    )
    max_speeds: Optional[List[float]] = Field(
        default=None,
        description="Multi-agent maximum speeds"
    )
    angular_velocities: Optional[List[float]] = Field(
        default=None,
        description="Multi-agent angular velocities in degrees per second"
    )
    num_agents: Optional[int] = Field(
        default=None,
        gt=0,
        description="Total number of agents (multi-agent mode)"
    )

    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
        validate_assignment=True,
        arbitrary_types_allowed=True
    )

    @field_validator('orientation')
    @classmethod
    def normalize_single_orientation(cls, v):
        """Normalize single agent orientation to [0, 360) range."""
        if v is not None:
            return v % 360.0
        return v

    @field_validator('orientations')
    @classmethod
    def normalize_multi_orientations(cls, v):
        """Normalize all multi-agent orientations to [0, 360) range."""
        if v is not None:
            return [orientation % 360.0 for orientation in v]
        return v

    @field_validator('positions')
    @classmethod
    def validate_multi_positions(cls, v):
        """Validate multi-agent positions structure."""
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValueError("positions must be a list")
        for i, pos in enumerate(v):
            if not isinstance(pos, (list, tuple)) or len(pos) != 2:
                raise ValueError(
                    f"Position {i} must be a list/tuple of [x, y] coordinates"
                )
        return v

    @model_validator(mode="after")
    def validate_navigation_mode_consistency(self):
        """
        Validate consistency between single-agent and multi-agent parameters.
        Ensures only one navigation mode is specified.
        """
        # Check for multi-agent indicators
        has_multi = any([
            self.positions is not None,
            self.num_agents is not None and self.num_agents > 1,
            isinstance(self.orientations, list),
            isinstance(self.speeds, list),
            isinstance(self.max_speeds, list),
            isinstance(self.angular_velocities, list)
        ])
        
        # Check for single-agent indicators
        has_single = self.position is not None
        
        # Validate mode exclusivity
        if has_multi and has_single:
            raise ValueError(
                "Cannot specify both single-agent (position) and "
                "multi-agent (positions, num_agents > 1) parameters"
            )
        
        # Multi-agent validation
        if has_multi:
            if self.positions is not None:
                n_agents = len(self.positions)
                
                # Validate num_agents consistency
                if self.num_agents is not None and self.num_agents != n_agents:
                    raise ValueError(
                        f"num_agents ({self.num_agents}) does not match "
                        f"number of positions ({n_agents})"
                    )
                
                # Validate parameter list lengths
                param_lists = {
                    'orientations': self.orientations,
                    'speeds': self.speeds,
                    'max_speeds': self.max_speeds,
                    'angular_velocities': self.angular_velocities
                }
                
                for param_name, param_list in param_lists.items():
                    if (isinstance(param_list, list) and 
                        len(param_list) != n_agents):
                        raise ValueError(
                            f"{param_name} length ({len(param_list)}) does not "
                            f"match number of agents ({n_agents})"
                        )
            
            # Validate multi-agent speed constraints
            if (isinstance(self.speeds, list) and 
                isinstance(self.max_speeds, list)):
                for i, (speed, max_speed) in enumerate(
                    zip(self.speeds, self.max_speeds)
                ):
                    if speed > max_speed:
                        raise ValueError(
                            f"Agent {i}: speed ({speed}) cannot exceed "
                            f"max_speed ({max_speed})"
                        )
        
        # Single-agent validation
        elif has_single or not has_multi:
            if (self.speed is not None and 
                self.max_speed is not None and 
                self.speed > self.max_speed):
                raise ValueError(
                    f"speed ({self.speed}) cannot exceed "
                    f"max_speed ({self.max_speed})"
                )
        
        return self


class VideoPlumeConfig(BaseModel):
    """
    Unified configuration schema for video-based plume environment processing.
    
    Consolidates video processing parameters with enhanced validation and 
    environment variable interpolation support. Integrates with Hydra 
    structured configuration for flexible video source management.
    
    Examples:
        Basic video configuration:
            video_path: "data/plume_video.mp4"
            flip: false
            grayscale: true
            kernel_size: 5
            kernel_sigma: 1.0
            
        With environment variables:
            video_path: ${oc.env:PLUME_VIDEO_PATH}
            flip: ${oc.env:VIDEO_FLIP,false}
            threshold: ${oc.env:PLUME_THRESHOLD,0.5}
            
        Advanced preprocessing:
            video_path: "data/turbulent_plume.avi"
            flip: true
            grayscale: true
            kernel_size: 7
            kernel_sigma: 2.0
            threshold: 0.3
            normalize: true
    """
    
    # Video source path (supports environment variable interpolation)
    video_path: Union[str, Path] = Field(
        description="Path to the video file containing plume data"
    )
    
    # Video preprocessing options
    flip: bool = Field(
        default=False,
        description="Whether to flip video frames horizontally"
    )
    
    grayscale: bool = Field(
        default=True,
        description="Convert video frames to grayscale"
    )
    
    # Gaussian smoothing parameters
    kernel_size: Optional[int] = Field(
        default=None,
        description="Gaussian kernel size for smoothing (must be odd and positive)"
    )
    
    kernel_sigma: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Gaussian kernel sigma for smoothing"
    )
    
    # Threshold and normalization
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Threshold value for binary plume detection"
    )
    
    normalize: bool = Field(
        default=True,
        description="Normalize frame values to [0, 1] range"
    )
    
    # Internal validation control
    _skip_validation: bool = Field(
        default=False,
        description="Skip file existence validation (for testing)"
    )

    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
        validate_assignment=True,
        arbitrary_types_allowed=True
    )

    @field_validator('video_path')
    @classmethod
    def validate_video_path(cls, v):
        """Validate video path format and convert to Path object."""
        if isinstance(v, str):
            # Handle environment variable interpolation placeholder
            if v.startswith('${') and v.endswith('}'):
                return v  # Return as-is for Hydra to resolve
            v = Path(v)
        return v

    @field_validator('kernel_size')
    @classmethod
    def validate_kernel_size(cls, v):
        """Validate that kernel_size is odd and positive if provided."""
        if v is not None:
            if v <= 0:
                raise ValueError("kernel_size must be positive")
            if v % 2 == 0:
                raise ValueError("kernel_size must be odd")
        return v

    @model_validator(mode="after")
    def validate_video_file_exists(self):
        """Validate video file existence unless validation is skipped."""
        if not self._skip_validation and not isinstance(self.video_path, str):
            # Only check existence for resolved paths
            if isinstance(self.video_path, Path):
                video_path = Path(self.video_path)
                if not video_path.exists():
                    raise ValueError(f"Video file not found: {video_path}")
                if not video_path.is_file():
                    raise ValueError(f"Video path is not a file: {video_path}")
        return self

    @model_validator(mode="after")
    def validate_gaussian_parameters(self):
        """Ensure Gaussian parameters are consistent."""
        if self.kernel_size is not None and self.kernel_sigma is None:
            raise ValueError(
                "kernel_sigma must be specified when kernel_size is provided"
            )
        if self.kernel_sigma is not None and self.kernel_size is None:
            raise ValueError(
                "kernel_size must be specified when kernel_sigma is provided"
            )
        return self


# Hydra ConfigStore registration for structured configuration
if HYDRA_AVAILABLE:
    cs = ConfigStore.instance()
    
    # Register configuration schemas for automatic discovery
    cs.store(name="single_agent_config", node=SingleAgentConfig)
    cs.store(name="multi_agent_config", node=MultiAgentConfig) 
    cs.store(name="navigator_config", node=NavigatorConfig)
    cs.store(name="video_plume_config", node=VideoPlumeConfig)
    
    # Register configuration groups for hierarchical composition
    cs.store(group="navigator", name="single_agent", node=SingleAgentConfig)
    cs.store(group="navigator", name="multi_agent", node=MultiAgentConfig)
    cs.store(group="navigator", name="unified", node=NavigatorConfig)
    cs.store(group="environment", name="video_plume", node=VideoPlumeConfig)


# Re-export all configuration models for public API
__all__ = [
    "NavigatorConfig",
    "SingleAgentConfig", 
    "MultiAgentConfig",
    "VideoPlumeConfig",
]