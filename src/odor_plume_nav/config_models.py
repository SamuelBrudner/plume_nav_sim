"""
Configuration models for odor plume navigation module.

This module provides Pydantic models for validating configuration dictionaries.
"""

from typing import Optional, Union, List, Tuple, Dict, Any
from pydantic import BaseModel, Field, validator, model_validator, field_validator, ConfigDict
import numpy as np


class SingleAgentConfig(BaseModel):
    """Configuration model for single-agent Navigator."""
    position: Tuple[float, float] = (0.0, 0.0)
    orientation: float = 0.0
    speed: float = 0.0
    max_speed: float = 1.0
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class MultiAgentConfig(BaseModel):
    """Configuration model for multi-agent Navigator."""
    positions: np.ndarray
    orientations: Optional[np.ndarray] = None
    speeds: Optional[np.ndarray] = None
    max_speeds: Optional[np.ndarray] = None
    num_agents: Optional[int] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_validator('positions')
    @classmethod
    def validate_positions(cls, v):
        """Validate positions array shape and type."""
        if not isinstance(v, np.ndarray):
            raise ValueError("Positions must be a numpy array")
        
        if len(v.shape) != 2 or v.shape[1] != 2:
            raise ValueError("Positions must be a numpy array with shape (n, 2)")
        
        return v
    
    @field_validator('orientations', 'speeds', 'max_speeds')
    @classmethod
    def validate_arrays(cls, v, info):
        """Validate array shapes and check they match positions length."""
        if v is None:
            return v
        
        field_name = info.field_name
        
        if not isinstance(v, np.ndarray):
            raise ValueError(f"{field_name} must be a numpy array")
        
        if len(v.shape) != 1:
            raise ValueError(f"{field_name} must be a numpy array with shape (n,)")
        
        # We can't check length consistency here because we don't have access
        # to other fields in a field_validator. We'll do that in the model validator.
        
        return v
    
    @model_validator(mode='after')
    def check_array_lengths(self):
        """Ensure all arrays have consistent lengths."""
        if self.positions is not None:
            num_agents = self.positions.shape[0]
            
            # Check orientations length
            if self.orientations is not None and self.orientations.shape[0] != num_agents:
                raise ValueError("Array lengths must match: positions.shape[0] != orientations.shape[0]")
            
            # Check speeds length
            if self.speeds is not None and self.speeds.shape[0] != num_agents:
                raise ValueError("Array lengths must match: positions.shape[0] != speeds.shape[0]")
            
            # Check max_speeds length
            if self.max_speeds is not None and self.max_speeds.shape[0] != num_agents:
                raise ValueError("Array lengths must match: positions.shape[0] != max_speeds.shape[0]")
            
            # If num_agents is provided, ensure it matches positions length
            if self.num_agents is not None and self.num_agents != num_agents:
                raise ValueError("num_agents must match positions.shape[0]")
        
        return self


class NavigatorConfig(BaseModel):
    """Configuration model for Navigator class."""
    config: Union[SingleAgentConfig, MultiAgentConfig]
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @model_validator(mode='before')
    @classmethod
    def check_single_or_multi(cls, data):
        """Determine whether this is a single-agent or multi-agent config."""
        if not isinstance(data, dict):
            return {"config": data}
            
        single_agent_params = {'position', 'orientation', 'speed', 'max_speed'}
        multi_agent_params = {'positions', 'orientations', 'speeds', 'max_speeds', 'num_agents'}
        
        has_single = any(param in data for param in single_agent_params)
        has_multi = any(param in data for param in multi_agent_params)
        
        if has_single and has_multi:
            raise ValueError("Cannot specify both single-agent and multi-agent parameters")
            
        if not (has_single or has_multi):
            raise ValueError("Config must contain either 'position' or 'positions'")
            
        # Determine the type of config and return the appropriate structure
        config_type = MultiAgentConfig if has_multi else SingleAgentConfig
        return {"config": config_type(**data)}
    
    def dict(self, **kwargs):
        """Override dict method to return the nested config dict."""
        return self.config.dict(**kwargs)
