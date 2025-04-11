"""
Configuration models for odor plume navigation.

This module provides Pydantic models for configuration validation.
"""

from typing import List, Optional, Tuple, Union, Dict, Any
from pydantic import BaseModel, root_validator, validator, Field


class SingleAgentConfig(BaseModel):
    """Configuration for a single agent navigation."""
    position: Optional[Tuple[float, float]] = None
    orientation: Optional[float] = None  # degrees
    speed: Optional[float] = None
    max_speed: Optional[float] = None
    angular_velocity: Optional[float] = None  # degrees per second

    class Config:
        extra = "allow"  # Allow extra fields


class MultiAgentConfig(BaseModel):
    """Configuration for multiple agent navigation."""
    positions: Optional[List[List[float]]] = None  # List of [x, y] positions
    orientations: Optional[List[float]] = None
    speeds: Optional[List[float]] = None
    max_speeds: Optional[List[float]] = None
    angular_velocities: Optional[List[float]] = None
    num_agents: Optional[int] = None

    @validator('positions')
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

    class Config:
        extra = "allow"  # Allow extra fields


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

    @root_validator(skip_on_failure=True)
    def check_agent_params(cls, values):
        """Verify that either single agent or multi-agent parameters are provided."""
        # Check if we have multi-agent parameters
        has_multi = values.get('positions') is not None
        
        # Check if we have single agent parameters
        has_single = values.get('position') is not None
        
        # Only one type of parameters should be provided
        if has_multi and has_single:
            raise ValueError("Cannot specify both single-agent and multi-agent parameters")
            
        if values.get('positions') is not None:
            # If providing multi-agent positions, ensure compatible array sizes
            positions = values.get('positions')
            n_agents = len(positions)
            
            # Update num_agents if provided
            if values.get('num_agents') is None:
                values['num_agents'] = n_agents
            elif values.get('num_agents') != n_agents:
                raise ValueError(f"num_agents ({values.get('num_agents')}) does not match length of positions ({n_agents})")
            
            # Ensure other multi-agent parameters have the same length if provided
            for param in ['orientations', 'speeds', 'max_speeds', 'angular_velocities']:
                if values.get(param) is not None and len(values.get(param)) != n_agents:
                    raise ValueError(f"{param} length ({len(values.get(param))}) does not match number of agents ({n_agents})")
        
        # Speed should not exceed max_speed
        if values.get('speed') is not None and values.get('max_speed') is not None:
            if values.get('speed') > values.get('max_speed'):
                raise ValueError(f"speed ({values.get('speed')}) cannot exceed max_speed ({values.get('max_speed')})")
        
        return values

    class Config:
        extra = "allow"  # Allow extra fields


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
    
    @validator('kernel_size')
    def validate_kernel_size(cls, v):
        """Validate that kernel_size is odd and positive if provided."""
        if v is not None:
            if v <= 0:
                raise ValueError("kernel_size must be positive")
            if v % 2 == 0:
                raise ValueError("kernel_size must be odd")
        return v
    
    class Config:
        extra = "allow"  # Allow extra fields


# Re-export the models
__all__ = [
    "NavigatorConfig",
    "SingleAgentConfig",
    "MultiAgentConfig",
    "VideoPlumeConfig",
]
