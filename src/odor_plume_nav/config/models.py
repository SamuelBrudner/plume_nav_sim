"""
Configuration models for odor plume navigation.

This module provides Pydantic models for configuration validation.
"""

from typing import List, Optional, Tuple, Union, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class SingleAgentConfig(BaseModel):
    """Configuration for a single agent navigation."""
    position: Optional[Tuple[float, float]] = None
    orientation: Optional[float] = None  # degrees
    speed: Optional[float] = None
    max_speed: Optional[float] = None
    angular_velocity: Optional[float] = None  # degrees per second

    model_config = ConfigDict(extra="allow")


class MultiAgentConfig(BaseModel):
    """Configuration for multiple agent navigation."""
    positions: Optional[List[List[float]]] = None  # List of [x, y] positions
    orientations: Optional[List[float]] = None
    speeds: Optional[List[float]] = None
    max_speeds: Optional[List[float]] = None
    angular_velocities: Optional[List[float]] = None
    num_agents: Optional[int] = None

    @field_validator('positions')
    @classmethod
    def validate_positions(cls, v):
        """Validate that positions is a list of [x, y] positions."""
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValueError("positions must be a list")
        for pos in v:
            if not isinstance(pos, list) or len(pos) != 2:
                raise ValueError("Each position must be a list of [x, y]")
        return v

    model_config = ConfigDict(extra="allow")


class NavigatorConfig(BaseModel):
    """
    Configuration for navigator.
    
    This can represent either a single agent or multiple agents.
    """
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

    @model_validator(mode="after")
    def check_agent_params(cls, values):
        """Verify that either single agent or multi-agent parameters are provided."""
        has_multi = values.positions is not None
        has_single = values.position is not None
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

    model_config = ConfigDict(extra="allow")


# Re-export the models
__all__ = [
    "NavigatorConfig",
    "SingleAgentConfig",
    "MultiAgentConfig",
    "VideoPlumeConfig",
]
