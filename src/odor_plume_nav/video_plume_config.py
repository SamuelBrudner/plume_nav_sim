"""
Configuration models for VideoPlume.

This module provides Pydantic models for validating VideoPlume configuration.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
from pydantic import BaseModel, ConfigDict, field_validator


class VideoPlumeConfig(BaseModel):
    """Configuration model for VideoPlume."""
    video_path: Union[str, Path]
    flip: bool = False
    kernel_size: int = 0
    kernel_sigma: float = 1.0
    _skip_validation: bool = False
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_validator('kernel_size')
    @classmethod
    def validate_kernel_size(cls, v):
        """Validate kernel size is non-negative."""
        if v < 0:
            raise ValueError("kernel_size must be non-negative")
        return v
    
    @field_validator('kernel_sigma')
    @classmethod
    def validate_kernel_sigma(cls, v):
        """Validate kernel sigma is positive."""
        if v <= 0:
            raise ValueError("kernel_sigma must be positive")
        return v
