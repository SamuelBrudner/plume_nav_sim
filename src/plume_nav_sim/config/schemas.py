"""
Configuration schemas for plume_nav_sim.

This module provides Pydantic models for configuration validation.
"""

from typing import List, Optional, Tuple, Union, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
try:
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore.instance()
except ImportError:
    cs = None


class SingleAgentConfig(BaseModel):
    """Configuration for single agent navigator."""
    position: Optional[Tuple[float, float]] = None
    orientation: Optional[float] = 0.0  # degrees
    speed: Optional[float] = 0.0
    max_speed: Optional[float] = 1.0
    angular_velocity: Optional[float] = 0.0  # degrees per second

    @field_validator('speed', 'max_speed', 'angular_velocity')
    @classmethod
    def validate_speeds(cls, v):
        """Validate that speeds are non-negative."""
        if v is not None and v < 0:
            raise ValueError("Speeds must be non-negative")
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


class NavigatorConfig(BaseModel):
    """
    Unified configuration for navigator that can handle single or multi-agent scenarios.
    
    This is maintained for backward compatibility.
    """
    # Mode selection - can be 'single', 'multi', or 'auto' for auto-detection
    mode: Optional[str] = "auto"
    
    # Single agent parameters
    position: Optional[Tuple[float, float]] = None
    orientation: Optional[float] = 0.0  # degrees
    speed: Optional[float] = 0.0
    max_speed: Optional[float] = 1.0
    angular_velocity: Optional[float] = 0.0  # degrees per second
    
    # Multi-agent parameters
    positions: Optional[List[List[float]]] = None  # List of [x, y] positions
    orientations: Optional[List[float]] = None
    speeds: Optional[List[float]] = None
    max_speeds: Optional[List[float]] = None
    angular_velocities: Optional[List[float]] = None
    num_agents: Optional[int] = None

    @field_validator('orientation')
    @classmethod
    def validate_orientation(cls, v):
        """Validate that orientation is within valid range."""
        if v is not None and (v < 0 or v >= 360):
            raise ValueError("orientation must be between 0 and 360 degrees")
        return v

    @field_validator('orientations')
    @classmethod
    def validate_orientations(cls, v):
        """Validate that all orientations are within valid range."""
        if v is not None:
            for orient in v:
                if orient < 0 or orient >= 360:
                    raise ValueError("orientation must be between 0 and 360 degrees")
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
        if values.speed is not None and values.max_speed is not None and values.speed > values.max_speed:
            raise ValueError(f"speed ({values.speed}) cannot exceed max_speed ({values.max_speed})")
        return values

    model_config = ConfigDict(extra="allow")


class VideoPlumeConfig(BaseModel):
    """Configuration for video-based plume environment."""
    # Path to the video file
    video_path: str
    
    # Optional parameters for video processing
    flip: Optional[bool] = False
    grayscale: Optional[bool] = True
    kernel_size: Optional[int] = None
    kernel_sigma: Optional[float] = None
    threshold: Optional[float] = None
    normalize: Optional[bool] = True
    
    # Frame range parameters
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None

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

    @field_validator('threshold')
    @classmethod
    def validate_threshold(cls, v):
        """Validate that threshold is within valid range [0, 1]."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError("ensure this value is less than or equal to 1")
        return v

    @model_validator(mode="after")
    def validate_frame_range(cls, values):
        """Validate that end_frame is greater than start_frame."""
        if values.start_frame is not None and values.end_frame is not None:
            if values.end_frame <= values.start_frame:
                raise ValueError("end_frame must be greater than start_frame")
        return values

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