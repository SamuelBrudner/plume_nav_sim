"""
Configuration schemas for plume_nav_sim.

This module provides Pydantic models for configuration validation.
"""

from typing import List, Optional, Tuple, Union, Dict, Any
from enum import Enum
from pathlib import Path
import logging
import re
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
try:
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore.instance()
except ImportError:
    cs = None


logger = logging.getLogger(__name__)
ENV_VAR_PATTERN = re.compile(r"^\$\{[^}]+\}$")


class SingleAgentConfig(BaseModel):
    """Configuration for single agent navigator."""
    position: Optional[Tuple[float, float]] = None
    orientation: Optional[float] = 0.0  # degrees
    speed: Optional[float] = Field(0.0)  # Removed ge constraint for custom validator
    max_speed: Optional[float] = Field(1.0)  # Removed ge constraint for custom validator
    angular_velocity: Optional[float] = Field(0.0)  # Removed ge constraint for custom validator

    @field_validator('orientation')
    @classmethod
    def normalize_orientation(cls, v):
        if v is None:
            return v
        return v % 360

    @field_validator('speed', 'max_speed', 'angular_velocity')
    @classmethod
    def validate_non_negative(cls, v, info):
        """Validate that values are non-negative."""
        if v is not None and v < 0:
            raise ValueError(f"ensure this value is greater than or equal to 0")
        return v

    @model_validator(mode="after")
    def check_speed_constraints(cls, values):
        """Verify speed constraints."""
        if values.speed is not None and values.max_speed is not None and values.speed > values.max_speed:
            raise ValueError(f"speed ({values.speed}) cannot exceed max_speed ({values.max_speed})")
        return values

    model_config = ConfigDict(extra="allow")


class MultiAgentConfig(BaseModel):
    """Configuration for multi-agent navigator."""
    positions: List[List[float]] = Field(default_factory=list)  # List of [x, y] positions
    orientations: Optional[List[float]] = None
    speeds: Optional[List[float]] = None
    max_speeds: Optional[List[float]] = None
    angular_velocities: Optional[List[float]] = None
    num_agents: Optional[int] = None

    @field_validator('orientations')
    @classmethod
    def normalize_orientations(cls, v):
        if v is not None:
            return [orient % 360 for orient in v]
        return v

    @field_validator('positions')
    @classmethod
    def validate_positions(cls, v):
        for i, pos in enumerate(v):
            if not isinstance(pos, (list, tuple)) or len(pos) != 2:
                raise ValueError(f"Position {i} must be a list/tuple of [x, y] coordinates")
            x, y = pos
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise ValueError(f"Position {i} coordinates must be numeric")
        return v

    @field_validator('speeds')
    @classmethod
    def validate_speeds(cls, v):
        if v is not None:
            for i, s in enumerate(v):
                if s < 0:
                    raise ValueError(f"Speed at index {i} must be non-negative")
        return v

    @model_validator(mode="after")
    def check_agent_params(cls, values):
        """Verify multi-agent parameter consistency."""
        if values.positions:
            n_agents = len(values.positions)
            for param_name, param_val in [
                ('orientations', values.orientations),
                ('speeds', values.speeds),
                ('max_speeds', values.max_speeds),
                ('angular_velocities', values.angular_velocities)
            ]:
                if param_val is not None and len(param_val) != n_agents:
                    raise ValueError(f"{param_name} length ({len(param_val)}) does not match number of agents ({n_agents})")
            
            # Set num_agents if not provided
            if values.num_agents is None:
                values.num_agents = n_agents
            elif values.num_agents != n_agents:
                raise ValueError(f"num_agents ({values.num_agents}) does not match positions length ({n_agents})")

        return values

    model_config = ConfigDict(extra="allow")


class NavigatorMode(str, Enum):
    single = "single"
    multi = "multi"
    auto = "auto"


class NavigatorConfig(BaseModel):
    """
    Unified configuration for navigator that can handle single or multi-agent scenarios.
    
    This is maintained for backward compatibility.
    """
    # Mode selection - can be 'single', 'multi', or 'auto' for auto-detection
    mode: NavigatorMode = NavigatorMode.auto
    
    # Single agent parameters
    position: Optional[Tuple[float, float]] = None
    orientation: Optional[float] = 0.0
    speed: Optional[float] = Field(0.0)  # Removed ge constraint for custom validator
    max_speed: Optional[float] = Field(1.0)  # Removed ge constraint for custom validator
    angular_velocity: Optional[float] = Field(0.0)  # Removed ge constraint for custom validator
    
    # Multi-agent parameters
    positions: Optional[List[List[float]]] = None  # List of [x, y] positions
    orientations: Optional[List[float]] = None
    speeds: Optional[List[float]] = None
    max_speeds: Optional[List[float]] = None
    angular_velocities: Optional[List[float]] = None
    num_agents: Optional[int] = None

    @field_validator('orientation')
    @classmethod
    def normalize_orientation(cls, v):
        if v is None:
            return v
        return v % 360

    @field_validator('speed', 'max_speed', 'angular_velocity')
    @classmethod
    def validate_non_negative(cls, v, info):
        """Validate that values are non-negative with specific error message."""
        if v is not None and v < 0:
            raise ValueError("ensure this value is greater than or equal to 0")
        return v

    @field_validator('orientations')
    @classmethod
    def normalize_orientations(cls, v):
        if v is not None:
            return [orient % 360 for orient in v]
        return v

    @model_validator(mode="after")
    def check_agent_params(cls, values):
        """Verify that either single agent or multi-agent parameters are provided and auto-detect mode."""
        has_multi = values.positions is not None
        has_single = values.position is not None
        
        # Auto-detect mode if set to "auto"
        if values.mode == "auto":
            if has_multi:
                values.mode = "multi"
            else:
                values.mode = "single"
        
        # Validate mode consistency
        if values.mode == "multi" and not has_multi:
            if has_single:
                raise ValueError("Cannot use single-agent parameters with multi mode")
        elif values.mode == "single" and has_multi:
            raise ValueError("Cannot use multi-agent parameters with single mode")
        
        if has_multi and has_single:
            raise ValueError("Cannot specify both single-agent and multi-agent parameters")
        if values.positions is not None:
            n_agents = len(values.positions)
            for param in ['orientations', 'speeds', 'max_speeds', 'angular_velocities']:
                param_val = getattr(values, param)
                if param_val is not None and len(param_val) != n_agents:
                    raise ValueError(f"{param} length ({len(param_val)}) does not match number of agents ({n_agents})")
            if values.num_agents is None:
                values.num_agents = n_agents
            elif values.num_agents != n_agents:
                raise ValueError(f"num_agents ({values.num_agents}) does not match positions length ({n_agents})")
            if values.speeds is not None and values.max_speeds is not None:
                for i, (s, m) in enumerate(zip(values.speeds, values.max_speeds)):
                    if s > m:
                        raise ValueError(f"Agent {i}: speed ({s}) cannot exceed max_speed ({m})")

        if values.speed is not None and values.max_speed is not None and values.speed > values.max_speed:
            raise ValueError(f"speed ({values.speed}) cannot exceed max_speed ({values.max_speed})")
        return values

    model_config = ConfigDict(extra="allow")


class VideoPlumeConfig(BaseModel):
    """Configuration for video-based plume environment."""
    # Path to the video file (kept internally as Path but exposed as string)
    video_path: Union[Path, str]

    # Internal flag to optionally skip validation
    skip_validation: bool = Field(default=False, repr=False, exclude=True)

    # Optional parameters for video processing
    flip: Optional[bool] = False
    grayscale: Optional[bool] = True
    kernel_size: Optional[int] = None
    kernel_sigma: Optional[float] = None
    threshold: Optional[float] = Field(None)  # Removed ge/le constraints for custom validator
    normalize: Optional[bool] = True

    # Frame range parameters
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None

    @field_validator('threshold')
    @classmethod
    def validate_threshold(cls, v):
        """Validate threshold with specific error messages."""
        if v is not None:
            if v > 1.0:
                raise ValueError("ensure this value is less than or equal to 1")
            if v < 0.0:
                raise ValueError("ensure this value is greater than or equal to 0")
        return v

    @field_validator('video_path', mode='before')
    @classmethod
    def parse_video_path(cls, v):
        """Convert to Path unless value is an env-variable pattern."""
        if isinstance(v, str) and ENV_VAR_PATTERN.fullmatch(v):
            return v
        return Path(v)

    @field_validator('video_path')
    @classmethod
    def validate_video_path(cls, v, info):
        skip = info.data.get('skip_validation', False)
        if isinstance(v, str):
            return v
        if not skip:
            if not v.exists():
                logger.error("Video file not found: %s", v)
                raise ValueError('Video file not found')
            if not v.is_file():
                logger.error("Video path is not a file: %s", v)
                raise ValueError('Video path is not a file')
        return v

    @field_validator('kernel_size')
    @classmethod
    def validate_kernel_size(cls, v):
        if v is not None:
            if v < 0:
                raise ValueError('kernel_size must be positive')
            if v % 2 == 0:
                raise ValueError('kernel_size must be odd')
        return v

    @field_validator('kernel_sigma')
    @classmethod
    def validate_kernel_sigma(cls, v):
        if v is not None and v <= 0:
            raise ValueError('kernel_sigma must be positive')
        return v

    @model_validator(mode="after")
    def validate_parameters(cls, values):
        if (values.kernel_size is None) != (values.kernel_sigma is None):
            if values.kernel_size is None:
                raise ValueError('kernel_size must be specified when kernel_sigma is provided')
            else:
                raise ValueError('kernel_sigma must be specified when kernel_size is provided')
        if values.start_frame is not None and values.end_frame is not None:
            if values.end_frame <= values.start_frame:
                raise ValueError('end_frame must be greater than start_frame')
        return values

    @property
    def video_path_str(self) -> str:
        return str(self.video_path)

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data['video_path'] = self.video_path_str
        return data

    model_config = ConfigDict(extra="allow")


class SimulationConfig(BaseModel):
    """Configuration for simulation parameters."""
    # Simulation timing
    max_steps: Optional[int] = 1000
    dt: Optional[float] = 0.1
    seed: Optional[int] = None
    
    # Environment parameters
    width: Optional[float] = 100.0
    height: Optional[float] = 100.0
    
    # Recording parameters
    record_trajectory: Optional[bool] = True
    record_odor_readings: Optional[bool] = True
    record_performance: Optional[bool] = False
    
    # Performance parameters
    performance_targets: Optional[Dict[str, float]] = None
    enable_profiling: Optional[bool] = False
    
    # Visualization parameters
    enable_visualization: Optional[bool] = False
    save_animation: Optional[bool] = False
    animation_path: Optional[str] = None
    # Runtime/IO parameters for compatibility with tests
    max_duration: Optional[float] = Field(None, gt=0)
    fps: Optional[int] = Field(None, gt=0)
    real_time: Optional[bool] = False
    output_directory: Optional[str] = None
    random_seed: Optional[Union[int, str]] = None
    
    # Debugging parameters
    debug_mode: Optional[bool] = False
    log_level: Optional[str] = "INFO"

    @field_validator('max_steps')
    @classmethod
    def validate_max_steps(cls, v):
        """Validate that max_steps is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("max_steps must be positive")
        return v
    
    @field_validator('dt')
    @classmethod 
    def validate_dt(cls, v):
        """Validate that dt is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("dt (timestep) must be positive")
        return v
    
    @field_validator('width', 'height')
    @classmethod
    def validate_dimensions(cls, v):
        """Validate that width and height are positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("Environment dimensions must be positive")
        return v

    model_config = ConfigDict(extra="allow")


# Re-export the models
__all__ = [
    "NavigatorConfig",
    "SingleAgentConfig",
    "MultiAgentConfig",
    "VideoPlumeConfig",
    "SimulationConfig",
    "cs"
]
