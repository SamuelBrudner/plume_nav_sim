"""
Enhanced dataclass configuration models with Hydra integration and structured validation.

This module provides comprehensive configuration management through dataclass-based schemas
integrated with Hydra's structured configuration system. It consolidates configuration
models from the domain layer while adding advanced Hydra capabilities including ConfigStore
registration, environment variable interpolation, and hierarchical configuration composition.

The models support both backward compatibility with existing YAML configurations and forward
compatibility with modern Hydra-based dataclass workflows for factory-driven component
instantiation and type-safe configuration management.
"""

from typing import List, Optional, Tuple, Union, Dict, Any, Literal
from dataclasses import dataclass, field
import logging
from pathlib import Path
import os
import re

# Pydantic for external validation compatibility
from pydantic import BaseModel, Field, field_validator, model_validator

# Import base models from domain layer for backward compatibility
from odor_plume_nav.domain.models import (
    NavigatorConfig as BaseNavigatorConfig,
    SingleAgentConfig as BaseSingleAgentConfig,
    MultiAgentConfig as BaseMultiAgentConfig,
    VideoPlumeConfig as BaseVideoPlumeConfig,
)

# Set up module logger
logger = logging.getLogger(__name__)


# Dataclass validation utilities to replace Pydantic validators
def validate_speed_constraints(value: float, field_name: str) -> float:
    """Validate speed parameters are non-negative with enhanced error messaging."""
    if value is None:
        return value
    
    # Convert string values to float (handles environment variable resolution)
    if isinstance(value, str):
        try:
            value = float(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"{field_name} must be a valid numeric value, got '{value}': {e}")
    
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value}")
    return value


def validate_orientation_range(value: float) -> float:
    """Validate orientation is within valid degree range."""
    if value is None:
        return value
    
    # Convert string values to float (handles environment variable resolution)
    if isinstance(value, str):
        try:
            value = float(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Orientation must be a valid numeric value, got '{value}': {e}")
    
    if not (0.0 <= value <= 360.0):
        raise ValueError(f"Orientation must be between 0.0 and 360.0 degrees, got {value}")
    return value


def validate_numeric_list(value: List[Union[int, float]], field_name: str) -> List[Union[int, float]]:
    """Validate that numeric list parameters contain only valid numbers with enhanced checks."""
    if value is None:
        return value
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    
    validated_list = []
    for i, val in enumerate(value):
        # Convert string values to float (handles environment variable resolution)
        if isinstance(val, str):
            try:
                val = float(val)
            except (ValueError, TypeError) as e:
                raise ValueError(f"{field_name}[{i}] must be a valid numeric value, got '{val}': {e}")
        elif not isinstance(val, (int, float)):
            raise ValueError(f"{field_name}[{i}] must be numeric, got {type(val).__name__}")
        
        # Enhanced validation for specific fields
        if field_name == 'orientations' and not (0 <= val <= 360):
            raise ValueError(f"Orientation {i} must be between 0 and 360 degrees, got {val}")
        elif field_name in ['speeds', 'max_speeds'] and val < 0:
            raise ValueError(f"{field_name}[{i}] must be non-negative, got {val}")
        
        validated_list.append(val)
    
    return validated_list


def validate_position(position_value: Union[Tuple[float, float], List[float], List[str]]) -> Tuple[float, float]:
    """Validate position tuple with string-to-float conversion for environment variable support."""
    if position_value is None:
        return None
    
    # Convert various input formats to list for processing
    if isinstance(position_value, (tuple, list)):
        pos_list = list(position_value)
    else:
        raise ValueError(f"Position must be a tuple or list, got {type(position_value).__name__}")
    
    if len(pos_list) != 2:
        raise ValueError(f"Position must have exactly 2 coordinates [x, y], got {len(pos_list)}")
    
    validated_coords = []
    for i, coord in enumerate(pos_list):
        # Convert string values to float (handles environment variable resolution)
        if isinstance(coord, str):
            try:
                coord = float(coord)
            except (ValueError, TypeError) as e:
                coord_name = "x" if i == 0 else "y"
                raise ValueError(f"Position {coord_name} coordinate must be a valid numeric value, got '{coord}': {e}")
        elif not isinstance(coord, (int, float)):
            coord_name = "x" if i == 0 else "y"
            raise ValueError(f"Position {coord_name} coordinate must be numeric, got {type(coord).__name__}")
        
        validated_coords.append(coord)
    
    return tuple(validated_coords)


def validate_positions(positions_value: List[List[float]]) -> List[List[float]]:
    """Validate list of positions with string-to-float conversion for environment variable support."""
    if positions_value is None:
        return None
    
    # Handle numpy arrays by converting to list
    if hasattr(positions_value, 'tolist'):  # numpy array or similar
        positions_value = positions_value.tolist()
    elif not isinstance(positions_value, list):
        raise ValueError(f"Positions must be a list or numpy array, got {type(positions_value).__name__}")
    
    validated_positions = []
    for i, position in enumerate(positions_value):
        try:
            validated_position = validate_position(position)
            validated_positions.append(list(validated_position))
        except ValueError as e:
            raise ValueError(f"Invalid position {i}: {e}")
    
    return validated_positions


def validate_video_path(path_value: Union[str, Path]) -> str:
    """Validate video path format and perform basic existence checking with environment variable support."""
    if isinstance(path_value, str):
        # Handle Hydra environment variable interpolation patterns
        if path_value.startswith('${oc.env:'):
            return path_value  # Skip validation for interpolated paths
        
        path = Path(path_value)
    else:
        path = path_value
    
    # Validate file extension for supported formats
    supported_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v'}
    if path.suffix.lower() not in supported_extensions:
        logger.warning(f"Video file extension '{path.suffix}' may not be supported. Supported: {supported_extensions}")
    
    return str(path)


def validate_output_directory(path_value: Union[str, Path]) -> str:
    """Validate and normalize output directory path with environment variable support."""
    if isinstance(path_value, str):
        # Handle Hydra environment variable interpolation
        if path_value.startswith('${oc.env:'):
            return path_value
        
        path = Path(path_value)
    else:
        path = path_value
    
    return str(path.resolve())


def validate_cache_mode(cache_mode: str) -> str:
    """Validate frame cache mode enumeration values."""
    valid_modes = {"none", "lru", "all"}
    if cache_mode not in valid_modes:
        raise ValueError(f"Cache mode must be one of {valid_modes}, got '{cache_mode}'")
    return cache_mode


def validate_memory_size(memory_size: Union[str, int, float]) -> str:
    """
    Validate and normalize memory size configuration with unit support.
    
    Args:
        memory_size: Memory size specification (e.g., "2GiB", "1024MB", 2147483648)
        
    Returns:
        Normalized memory size string with proper units
        
    Raises:
        ValueError: If memory size format is invalid or value is negative
    """
    if isinstance(memory_size, (int, float)):
        if memory_size < 0:
            raise ValueError(f"Memory size must be non-negative, got {memory_size}")
        return f"{int(memory_size)}B"
    
    if not isinstance(memory_size, str):
        raise ValueError(f"Memory size must be string or numeric, got {type(memory_size)}")
    
    # Handle environment variable interpolation
    if memory_size.startswith('${oc.env:'):
        return memory_size
    
    # Validate memory size format with units
    import re
    pattern = r'^(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|GiB|MiB|KiB|TB)?$'
    match = re.match(pattern, memory_size.strip(), re.IGNORECASE)
    
    if not match:
        raise ValueError(f"Invalid memory size format: {memory_size}. Use format like '2GiB', '1024MB', or numeric bytes")
    
    value, unit = match.groups()
    value = float(value)
    
    if value < 0:
        raise ValueError(f"Memory size must be non-negative, got {value}")
    
    # Normalize unit to standard format
    if unit is None:
        unit = "B"
    else:
        unit = unit.upper()
        # Normalize binary units
        if unit in ["GIB", "MIB", "KIB"]:
            unit = unit.replace("IB", "iB")
    
    return f"{value:.0f}{unit}" if value.is_integer() else f"{value}{unit}"


def validate_eviction_policy(policy: str) -> str:
    """Validate cache eviction policy enumeration values."""
    valid_policies = {"lru", "fifo", "random"}
    if policy not in valid_policies:
        raise ValueError(f"Eviction policy must be one of {valid_policies}, got '{policy}'")
    return policy


# --------------------------------------------------------------------------- #
# Pydantic compatibility layer for VideoPlumeConfig                           #
# --------------------------------------------------------------------------- #

class _VideoPlumeConfigModel(BaseModel):
    """
    Internal Pydantic model that mirrors VideoPlumeConfig for external libraries
    expecting `model_validate` / `parse_obj` style APIs (e.g., tests).
    """
    video_path: Union[str, Path]
    flip: Optional[bool] = False
    grayscale: Optional[bool] = True
    kernel_size: Optional[int] = None
    kernel_sigma: Optional[float] = None
    threshold: Optional[float] = None
    normalize: Optional[bool] = True
    frame_skip: Optional[int] = 0
    start_frame: Optional[int] = 0
    end_frame: Optional[int] = None
    fourcc: Optional[str] = None
    fps_override: Optional[float] = None

    # Sentinel used by VideoPlume.from_config to bypass certain validations
    _skip_validation: Optional[bool] = False

    # --------------------------------------------------------------------- #
    # Field-level validators                                                #
    # --------------------------------------------------------------------- #
    @field_validator('threshold')
    @classmethod
    def _validate_threshold(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("threshold must be between 0.0 and 1.0")
        return v

    # --------------------------------------------------------------------- #
    # Model-level validation                                                #
    # --------------------------------------------------------------------- #
    @model_validator(mode='after')
    def _validate_gaussian_and_frames(cls, values):
        if values.get("_skip_validation"):
            # Skip heavy validation when requested (e.g., file existence)
            return values

        kernel_size, kernel_sigma = values.get("kernel_size"), values.get("kernel_sigma")
        if (kernel_size is not None) != (kernel_sigma is not None):
            raise ValueError("Both kernel_size and kernel_sigma must be specified together")

        sf, ef = values.get("start_frame"), values.get("end_frame")
        if sf is not None and ef is not None and ef <= sf:
            raise ValueError("end_frame must be greater than start_frame")
        return values

    class Config:
        extra = "allow"


def __post_init_single_agent_config__(self):
    """Post-initialization validation for SingleAgentConfig."""
    if hasattr(self, 'speed') and hasattr(self, 'max_speed'):
        if self.speed is not None and self.max_speed is not None:
            # Convert to float if they are strings (from environment variable resolution)
            speed_val = float(self.speed) if isinstance(self.speed, str) else self.speed
            max_speed_val = float(self.max_speed) if isinstance(self.max_speed, str) else self.max_speed
            
            if speed_val > max_speed_val:
                raise ValueError(f"Initial speed ({speed_val}) cannot exceed max_speed ({max_speed_val})")


def __post_init_multi_agent_config__(self):
    """Post-initialization validation for MultiAgentConfig."""
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
            # Convert to float if they are strings (from environment variable resolution)
            speed_val = float(speed) if isinstance(speed, str) else speed
            max_speed_val = float(max_speed) if isinstance(max_speed, str) else max_speed
            
            if speed_val > max_speed_val:
                raise ValueError(f"Agent {i} speed ({speed_val}) exceeds max_speed ({max_speed_val})")


def __post_init_navigator_config__(self):
    """Post-initialization validation for NavigatorConfig."""
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


def __post_init_video_plume_config__(self):
    """Post-initialization validation for VideoPlumeConfig."""
    # Validate boolean parameters
    if self.flip is not None and not isinstance(self.flip, bool):
        raise ValueError(f"flip must be a boolean value, got {type(self.flip).__name__}: {self.flip}")
    
    if self.grayscale is not None and not isinstance(self.grayscale, bool):
        raise ValueError(f"grayscale must be a boolean value, got {type(self.grayscale).__name__}: {self.grayscale}")
    
    if self.normalize is not None and not isinstance(self.normalize, bool):
        raise ValueError(f"normalize must be a boolean value, got {type(self.normalize).__name__}: {self.normalize}")
    
    # Validate numeric parameters
    if self.kernel_size is not None and not isinstance(self.kernel_size, int):
        raise ValueError(f"kernel_size must be an integer, got {type(self.kernel_size).__name__}: {self.kernel_size}")
    
    if self.kernel_sigma is not None and not isinstance(self.kernel_sigma, (int, float)):
        raise ValueError(f"kernel_sigma must be a numeric value, got {type(self.kernel_sigma).__name__}: {self.kernel_sigma}")
    
    if self.threshold is not None and not isinstance(self.threshold, (int, float)):
        raise ValueError(f"threshold must be a numeric value, got {type(self.threshold).__name__}: {self.threshold}")
    
    has_kernel_size = self.kernel_size is not None
    has_kernel_sigma = self.kernel_sigma is not None
    
    if has_kernel_size and not has_kernel_sigma:
        logger.warning("kernel_size specified without kernel_sigma. Using default sigma=1.0")
        self.kernel_sigma = 1.0
    elif has_kernel_sigma and not has_kernel_size:
        logger.warning("kernel_sigma specified without kernel_size. Using default size=5")
        self.kernel_size = 5
    
    # Validate frame range
    if self.end_frame is not None and self.start_frame is not None and self.end_frame <= self.start_frame:
        raise ValueError(f"end_frame ({self.end_frame}) must be greater than start_frame ({self.start_frame})")


def __post_init_simulation_config__(self):
    """Post-initialization validation for SimulationConfig."""
    if self.enable_gpu and self.num_workers > 1:
        logger.warning("GPU acceleration with multiple workers may cause resource conflicts")
    
    if self.batch_size > 1 and self.enable_visualization:
        logger.warning("Batch processing with real-time visualization may impact performance")


def __post_init_frame_cache_config__(self):
    """Post-initialization validation for FrameCacheConfig."""
    # Validate cache mode specific constraints
    if self.mode == "none" and (self.memory_limit_mb is not None or self.eviction_policy != "lru"):
        logger.warning("Cache mode 'none' specified with cache-specific parameters. Parameters will be ignored.")
    
    if self.mode == "all" and self.eviction_policy != "lru":
        logger.warning("Eviction policy is not applicable for 'all' cache mode (full preload)")
    
    # Validate memory limit for non-none modes
    if self.mode in ["lru", "all"] and self.memory_limit_mb is not None:
        try:
            # Parse memory limit to ensure it's reasonable
            if isinstance(self.memory_limit_mb, str) and not self.memory_limit_mb.startswith('${oc.env:'):
                import re
                pattern = r'^(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|GiB|MiB|KiB|TB)?$'
                match = re.match(pattern, self.memory_limit_mb.strip(), re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    unit = (match.group(2) or "B").upper()
                    
                    # Convert to MB for validation
                    multipliers = {
                        "B": 1e-6, "KB": 1e-3, "MB": 1, "GB": 1e3, "GIB": 1073.741824,
                        "MIB": 1.048576, "KIB": 1.024e-3, "TB": 1e6
                    }
                    mb_value = value * multipliers.get(unit.replace("IB", "iB"), 1)
                    
                    if mb_value < 10:  # Minimum 10MB
                        logger.warning(f"Memory limit {self.memory_limit_mb} is very small (<10MB). Consider increasing for better performance.")
                    elif mb_value > 32768:  # 32GB warning
                        logger.warning(f"Memory limit {self.memory_limit_mb} is very large (>32GB). Ensure sufficient system memory.")
        except Exception:
            pass  # Validation will happen during runtime resolution


@dataclass
class SingleAgentConfig:
    """
    Enhanced dataclass configuration schema for single agent navigation parameters.
    
    This model provides advanced Hydra integration with environment variable 
    interpolation support and comprehensive field validation for type-safe 
    configuration management in production environments.
    """
    
    # Core navigation parameters with enhanced field metadata
    position: Optional[Tuple[float, float]] = field(
        default=None,
        metadata={
            "description": "Initial [x, y] position coordinates. Supports ${oc.env:AGENT_START_X,50.0} interpolation"
        }
    )
    orientation: Optional[float] = field(
        default=0.0,
        metadata={
            "ge": 0.0,
            "le": 360.0,
            "description": "Initial orientation in degrees (0=right, 90=up). Supports ${oc.env:AGENT_ORIENTATION,0.0}"
        }
    )
    speed: Optional[float] = field(
        default=0.0,
        metadata={
            "ge": 0.0,
            "description": "Initial speed in units per timestep. Supports ${oc.env:AGENT_SPEED,0.0}"
        }
    )
    max_speed: Optional[float] = field(
        default=1.0,
        metadata={
            "gt": 0.0,
            "description": "Maximum allowed speed constraint. Supports ${oc.env:AGENT_MAX_SPEED,1.0}"
        }
    )
    angular_velocity: Optional[float] = field(
        default=0.0,
        metadata={
            "description": "Angular velocity in degrees per timestep. Supports ${oc.env:AGENT_ANGULAR_VEL,0.0}"
        }
    )
    
    # Hydra-specific _target_ metadata for factory-driven component instantiation
    _target_: str = field(
        default="odor_plume_nav.core.controllers.SingleAgentController",
        metadata={
            "description": "Hydra target for automatic instantiation"
        }
    )
    
    def __post_init__(self):
        """Post-initialization validation for all field constraints."""
        # Validate position coordinates
        if self.position is not None:
            self.position = validate_position(self.position)
        
        # Validate speed constraints
        if self.speed is not None:
            self.speed = validate_speed_constraints(self.speed, "speed")
        if self.max_speed is not None:
            self.max_speed = validate_speed_constraints(self.max_speed, "max_speed")
        
        # Validate orientation range
        if self.orientation is not None:
            self.orientation = validate_orientation_range(self.orientation)
        
        # Validate speed relationship
        __post_init_single_agent_config__(self)


@dataclass
class MultiAgentConfig:
    """
    Enhanced dataclass configuration schema for multi-agent navigation scenarios.
    
    Defines parameters for coordinated multi-agent simulations with vectorized
    operations support and comprehensive validation. All list parameters must
    have consistent lengths equal to num_agents when specified.
    """
    
    # Multi-agent parameters with enhanced field metadata and validation
    positions: Optional[List[List[float]]] = field(
        default=None,
        metadata={
            "description": "List of [x, y] positions for each agent. Length must match num_agents"
        }
    )
    orientations: Optional[List[float]] = field(
        default=None,
        metadata={
            "description": "List of initial orientations in degrees for each agent"
        }
    )
    speeds: Optional[List[float]] = field(
        default=None,
        metadata={
            "description": "List of initial speeds for each agent"
        }
    )
    max_speeds: Optional[List[float]] = field(
        default=None,
        metadata={
            "description": "List of maximum speed constraints for each agent"
        }
    )
    angular_velocities: Optional[List[float]] = field(
        default=None,
        metadata={
            "description": "List of angular velocities in degrees per timestep for each agent"
        }
    )
    num_agents: Optional[int] = field(
        default=None,
        metadata={
            "gt": 0,
            "le": 100,
            "description": "Total number of agents in simulation. Supports ${oc.env:NUM_AGENTS,10}"
        }
    )
    
    # Hydra-specific _target_ metadata for factory-driven component instantiation
    _target_: str = field(
        default="odor_plume_nav.core.controllers.MultiAgentController",
        metadata={
            "description": "Hydra target for automatic instantiation"
        }
    )
    
    def __post_init__(self):
        """Post-initialization validation for all field constraints."""
        # Validate positions list
        if self.positions is not None:
            self.positions = validate_positions(self.positions)
        
        # Validate numeric lists
        if self.orientations is not None:
            self.orientations = validate_numeric_list(self.orientations, "orientations")
        if self.speeds is not None:
            self.speeds = validate_numeric_list(self.speeds, "speeds")
        if self.max_speeds is not None:
            self.max_speeds = validate_numeric_list(self.max_speeds, "max_speeds")
        if self.angular_velocities is not None:
            self.angular_velocities = validate_numeric_list(self.angular_velocities, "angular_velocities")
        
        # Apply multi-agent specific validation
        __post_init_multi_agent_config__(self)


@dataclass
class NavigatorConfig:
    """
    Enhanced unified dataclass configuration schema for navigation supporting both single and multi-agent scenarios.
    
    This is the primary configuration model that consolidates single and multi-agent parameters
    with intelligent mode detection, Hydra environment variable interpolation support, and
    hierarchical configuration composition for factory-driven component instantiation.
    """
    
    # Navigation mode specification with enhanced validation
    mode: Literal["single", "multi", "auto"] = field(
        default="auto",
        metadata={
            "description": "Navigation mode: 'single', 'multi', or 'auto' for automatic detection"
        }
    )
    
    # Single agent parameters with environment variable interpolation support
    position: Optional[Tuple[float, float]] = field(
        default=None,
        metadata={
            "description": "Single agent initial position. Supports ${oc.env:AGENT_POSITION} interpolation"
        }
    )
    orientation: Optional[float] = field(
        default=0.0,
        metadata={
            "ge": 0.0,
            "le": 360.0,
            "description": "Single agent initial orientation in degrees. Supports ${oc.env:AGENT_ORIENTATION,0.0}"
        }
    )
    speed: Optional[float] = field(
        default=0.0,
        metadata={
            "ge": 0.0,
            "description": "Single agent initial speed. Supports ${oc.env:AGENT_SPEED,0.0}"
        }
    )
    max_speed: Optional[float] = field(
        default=1.0,
        metadata={
            "gt": 0.0,
            "description": "Single agent maximum speed constraint. Supports ${oc.env:AGENT_MAX_SPEED,1.0}"
        }
    )
    angular_velocity: Optional[float] = field(
        default=0.0,
        metadata={
            "description": "Single agent angular velocity. Supports ${oc.env:AGENT_ANGULAR_VEL,0.0}"
        }
    )
    
    # Multi-agent parameters with enhanced validation
    positions: Optional[List[List[float]]] = field(
        default=None,
        metadata={
            "description": "Multi-agent positions list. Presence triggers multi-agent mode"
        }
    )
    orientations: Optional[List[float]] = field(
        default=None,
        metadata={
            "description": "Multi-agent orientations list"
        }
    )
    speeds: Optional[List[float]] = field(
        default=None,
        metadata={
            "description": "Multi-agent speeds list"
        }
    )
    max_speeds: Optional[List[float]] = field(
        default=None,
        metadata={
            "description": "Multi-agent maximum speeds list"
        }
    )
    angular_velocities: Optional[List[float]] = field(
        default=None,
        metadata={
            "description": "Multi-agent angular velocities list"
        }
    )
    num_agents: Optional[int] = field(
        default=None,
        metadata={
            "gt": 0,
            "le": 100,
            "description": "Number of agents for multi-agent mode. Supports ${oc.env:NUM_AGENTS,10}"
        }
    )
    
    # Hydra-specific _target_ metadata for factory-driven component instantiation
    _target_: str = field(
        default="odor_plume_nav.api.create_navigator",
        metadata={
            "description": "Hydra target for factory instantiation"
        }
    )
    
    def __post_init__(self):
        """Post-initialization validation for navigation mode consistency."""
        __post_init_navigator_config__(self)
    
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


@dataclass
class VideoPlumeConfig:
    """
    Enhanced dataclass configuration schema for video-based plume environment processing.
    
    Extends the domain VideoPlumeConfig with comprehensive validation, Hydra environment
    variable interpolation support, and advanced processing parameters for video file
    loading, preprocessing transformations, and frame analysis.
    """
    
    # Core video file configuration with environment variable interpolation
    video_path: Union[str, Path] = field(
        default="${oc.env:VIDEO_PATH,./data/default.mp4}",
        metadata={
            "description": "Path to video file. Supports ${oc.env:VIDEO_PATH} interpolation for deployment flexibility"
        }
    )
    
    # Video preprocessing parameters with enhanced metadata
    flip: Optional[bool] = field(
        default=False,
        metadata={
            "description": "Apply horizontal flip transformation to video frames"
        }
    )
    grayscale: Optional[bool] = field(
        default=True,
        metadata={
            "description": "Convert video frames to grayscale for processing"
        }
    )
    
    # Gaussian filtering configuration with enhanced validation
    kernel_size: Optional[int] = field(
        default=None,
        metadata={
            "description": "Gaussian kernel size for smoothing (must be odd and positive, or None to disable)"
        }
    )
    kernel_sigma: Optional[float] = field(
        default=None,
        metadata={
            "gt": 0.0,
            "description": "Gaussian kernel sigma parameter for smoothing intensity"
        }
    )
    
    # Advanced processing parameters
    threshold: Optional[float] = field(
        default=None,
        metadata={
            "ge": 0.0,
            "le": 1.0,
            "description": "Threshold value for binary frame processing (0.0-1.0 range)"
        }
    )
    normalize: Optional[bool] = field(
        default=True,
        metadata={
            "description": "Apply frame normalization for consistent value ranges"
        }
    )
    
    # Frame selection and sampling parameters
    frame_skip: Optional[int] = field(
        default=0,
        metadata={
            "ge": 0,
            "description": "Number of frames to skip between processing steps"
        }
    )
    start_frame: Optional[int] = field(
        default=0,
        metadata={
            "ge": 0,
            "description": "Starting frame index for video processing"
        }
    )
    end_frame: Optional[int] = field(
        default=None,
        metadata={
            "ge": 0,
            "description": "Ending frame index (None for entire video)"
        }
    )
    
    # Video format and codec parameters
    fourcc: Optional[str] = field(
        default=None,
        metadata={
            "max_length": 4,
            "description": "Four-character code for video codec specification"
        }
    )
    fps_override: Optional[float] = field(
        default=None,
        metadata={
            "gt": 0.0,
            "description": "Override video FPS for simulation timing"
        }
    )
    
    # Hydra-specific _target_ metadata for factory-driven component instantiation
    _target_: str = field(
        default="odor_plume_nav.data.VideoPlume",
        metadata={
            "description": "Hydra target for automatic instantiation"
        }
    )
    
    def __post_init__(self):
        """Post-initialization validation for all field constraints."""
        # Validate video path
        if self.video_path:
            self.video_path = validate_video_path(self.video_path)
        
        # Apply video plume specific validation
        __post_init_video_plume_config__(self)

    # ----------------------------------------------------------------- #
    # Pydantic compatibility helpers                                    #
    # ----------------------------------------------------------------- #
    @classmethod
    def model_validate(cls, data):
        """
        Proxy to internal Pydantic model validation so external code/tests
        can call `VideoPlumeConfig.model_validate(...)` directly.
        """
        return _VideoPlumeConfigModel.model_validate(data)

    @classmethod
    def parse_obj(cls, obj):
        """
        Backwards-compatibility for Pydantic v1 style `parse_obj` API that may
        be used by downstream libraries or legacy code paths.
        """
        return _VideoPlumeConfigModel.model_validate(obj)


@dataclass
class SimulationConfig:
    """
    Enhanced dataclass configuration schema for simulation execution parameters and output settings.
    
    Defines comprehensive simulation behavior including performance settings, output
    configuration, experiment tracking parameters, and Hydra integration for
    factory-driven component instantiation.
    """
    
    # Core simulation parameters with enhanced validation
    max_steps: int = field(
        default=1000,
        metadata={
            "gt": 0,
            "description": "Maximum number of simulation steps to execute"
        }
    )
    step_size: float = field(
        default=1.0,
        metadata={
            "gt": 0.0,
            "description": "Time step size for simulation progression"
        }
    )
    
    # Performance and optimization settings
    enable_gpu: bool = field(
        default=False,
        metadata={
            "description": "Enable GPU acceleration for computations when available"
        }
    )
    batch_size: int = field(
        default=1,
        metadata={
            "gt": 0,
            "description": "Batch size for parallel processing operations"
        }
    )
    num_workers: int = field(
        default=1,
        metadata={
            "ge": 1,
            "description": "Number of worker processes for parallel execution"
        }
    )
    
    # Output and recording configuration with environment variable support
    record_trajectory: bool = field(
        default=True,
        metadata={
            "description": "Enable trajectory data recording during simulation"
        }
    )
    output_format: Literal["numpy", "csv", "hdf5", "json"] = field(
        default="numpy",
        metadata={
            "description": "Output format for trajectory and results data"
        }
    )
    output_directory: Union[str, Path] = field(
        default="./outputs",
        metadata={
            "description": "Directory for simulation output files. Supports ${oc.env:OUTPUT_DIR} interpolation"
        }
    )
    
    # Visualization settings
    enable_visualization: bool = field(
        default=True,
        metadata={
            "description": "Enable real-time visualization during simulation"
        }
    )
    visualization_fps: float = field(
        default=30.0,
        metadata={
            "gt": 0.0,
            "le": 120.0,
            "description": "Target FPS for visualization rendering"
        }
    )
    save_animation: bool = field(
        default=False,
        metadata={
            "description": "Save simulation animation to video file"
        }
    )
    
    # Experiment tracking and reproducibility with environment variable support
    experiment_name: Optional[str] = field(
        default=None,
        metadata={
            "description": "Name for experiment tracking and output organization"
        }
    )
    random_seed: Optional[int] = field(
        default=None,
        metadata={
            "ge": 0,
            "description": "Random seed for reproducible simulations. Supports ${oc.env:RANDOM_SEED}"
        }
    )
    
    # Advanced simulation features
    checkpoint_interval: int = field(
        default=100,
        metadata={
            "gt": 0,
            "description": "Number of steps between simulation checkpoints"
        }
    )
    enable_logging: bool = field(
        default=True,
        metadata={
            "description": "Enable detailed simulation logging"
        }
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = field(
        default="INFO",
        metadata={
            "description": "Logging verbosity level"
        }
    )
    
    # Hydra-specific _target_ metadata for factory-driven component instantiation
    _target_: str = field(
        default="odor_plume_nav.api.run_simulation",
        metadata={
            "description": "Hydra target for simulation execution"
        }
    )
    
    def __post_init__(self):
        """Post-initialization validation for all field constraints."""
        # Validate output directory
        if self.output_directory:
            self.output_directory = validate_output_directory(self.output_directory)
        
        # Apply simulation-specific validation
        __post_init_simulation_config__(self)


@dataclass
class FrameCacheConfig:
    """
    Enhanced dataclass configuration schema for frame caching system in video processing.
    
    Defines comprehensive frame cache parameters supporting dual-mode operation (LRU/full-preload)
    with intelligent memory management, performance optimization, and deployment-specific
    configuration through environment variable interpolation.
    
    This configuration enables sub-10ms frame retrieval performance for RL training workflows
    while providing flexible memory usage controls and cache behavior customization.
    """
    
    # Core cache mode configuration with enhanced validation
    mode: Literal["none", "lru", "all"] = field(
        default="${oc.env:FRAME_CACHE_MODE,lru}",
        metadata={
            "description": "Frame cache mode: 'none' (disabled), 'lru' (memory-efficient), 'all' (full preload). Supports ${oc.env:FRAME_CACHE_MODE,lru}"
        }
    )
    
    # Memory management configuration with environment variable support
    memory_limit_mb: Optional[Union[str, int, float]] = field(
        default="${oc.env:FRAME_CACHE_SIZE_MB,2GiB}",
        metadata={
            "description": "Maximum cache memory usage. Supports units (B, KB, MB, GB, GiB, MiB). Supports ${oc.env:FRAME_CACHE_SIZE_MB,2GiB}"
        }
    )
    
    # Cache eviction policy for LRU mode
    eviction_policy: Literal["lru", "fifo", "random"] = field(
        default="lru",
        metadata={
            "description": "Cache eviction policy for LRU mode. Options: 'lru' (least recently used), 'fifo' (first in, first out), 'random'"
        }
    )
    
    # Performance optimization settings
    preload_strategy: Literal["sequential", "random", "adaptive"] = field(
        default="sequential",
        metadata={
            "description": "Preload strategy for 'all' mode: 'sequential' (ordered), 'random' (shuffled), 'adaptive' (usage-based)"
        }
    )
    
    # Memory pressure management
    memory_pressure_threshold: float = field(
        default=0.9,
        metadata={
            "ge": 0.1,
            "le": 1.0,
            "description": "Memory usage threshold (0.0-1.0) that triggers aggressive eviction or cache size reduction"
        }
    )
    
    # Cache warming and preload settings
    enable_cache_warming: bool = field(
        default=True,
        metadata={
            "description": "Enable cache warming during environment initialization for predictable performance"
        }
    )
    
    warm_cache_percentage: float = field(
        default=0.1,
        metadata={
            "ge": 0.0,
            "le": 1.0,
            "description": "Percentage of video frames to preload during cache warming (0.0-1.0)"
        }
    )
    
    # Performance monitoring and statistics
    enable_statistics: bool = field(
        default=True,
        metadata={
            "description": "Enable cache hit/miss statistics collection and performance metrics tracking"
        }
    )
    
    statistics_update_interval: int = field(
        default=100,
        metadata={
            "gt": 0,
            "description": "Number of cache operations between statistics updates for performance logging"
        }
    )
    
    # Thread safety and concurrency settings
    enable_thread_safety: bool = field(
        default=True,
        metadata={
            "description": "Enable thread-safe cache operations for multi-agent scenarios (slight performance overhead)"
        }
    )
    
    max_concurrent_access: int = field(
        default=10,
        metadata={
            "gt": 0,
            "le": 100,
            "description": "Maximum number of concurrent cache access operations"
        }
    )
    
    # Advanced configuration options
    enable_zero_copy: bool = field(
        default=True,
        metadata={
            "description": "Enable zero-copy frame operations for minimal memory allocation overhead"
        }
    )
    
    compression_enabled: bool = field(
        default=False,
        metadata={
            "description": "Enable frame compression in cache to reduce memory usage (CPU overhead trade-off)"
        }
    )
    
    compression_quality: float = field(
        default=0.9,
        metadata={
            "ge": 0.1,
            "le": 1.0,
            "description": "Compression quality factor (0.1-1.0) when compression is enabled"
        }
    )
    
    # Debugging and development settings
    enable_debug_logging: bool = field(
        default=False,
        metadata={
            "description": "Enable detailed debug logging for cache operations (performance impact)"
        }
    )
    
    cache_validation_enabled: bool = field(
        default=False,
        metadata={
            "description": "Enable cache integrity validation (significant performance impact, use for debugging)"
        }
    )
    
    # Hydra-specific _target_ metadata for factory-driven component instantiation
    _target_: str = field(
        default="odor_plume_nav.cache.frame_cache.FrameCache",
        metadata={
            "description": "Hydra target for automatic FrameCache instantiation"
        }
    )
    
    def __post_init__(self):
        """Post-initialization validation for all field constraints."""
        # Validate cache mode
        if isinstance(self.mode, str) and not self.mode.startswith('${oc.env:'):
            validate_cache_mode(self.mode)
        
        # Validate memory limit format
        if (self.memory_limit_mb is not None and 
            isinstance(self.memory_limit_mb, str) and 
            not self.memory_limit_mb.startswith('${oc.env:')):
            validate_memory_size(self.memory_limit_mb)
        
        # Validate eviction policy
        if isinstance(self.eviction_policy, str):
            validate_eviction_policy(self.eviction_policy)
        
        # Apply frame cache specific validation
        __post_init_frame_cache_config__(self)
    
    def get_memory_limit_bytes(self) -> Optional[int]:
        """
        Convert memory limit to bytes for runtime use.
        
        Returns:
            Memory limit in bytes, or None if not specified or using environment interpolation
        """
        if self.memory_limit_mb is None:
            return None
        
        if isinstance(self.memory_limit_mb, str):
            if self.memory_limit_mb.startswith('${oc.env:'):
                # Cannot resolve environment variables in dataclass
                return None
            
            # Parse memory size string
            import re
            pattern = r'^(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|GiB|MiB|KiB|TB)?$'
            match = re.match(pattern, self.memory_limit_mb.strip(), re.IGNORECASE)
            
            if not match:
                return None
            
            value = float(match.group(1))
            unit = (match.group(2) or "B").upper()
            
            # Convert to bytes
            multipliers = {
                "B": 1,
                "KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4,
                "KIB": 1024, "MIB": 1024**2, "GIB": 1024**3, "TIB": 1024**4
            }
            
            unit = unit.replace("IB", "iB")  # Normalize binary units
            multiplier = multipliers.get(unit, 1)
            
            return int(value * multiplier)
        
        if isinstance(self.memory_limit_mb, (int, float)):
            return int(self.memory_limit_mb)
        
        return None
    
    def is_cache_enabled(self) -> bool:
        """
        Check if caching is enabled based on mode configuration.
        
        Returns:
            True if cache mode is not 'none'
        """
        if isinstance(self.mode, str) and not self.mode.startswith('${oc.env:'):
            return self.mode != "none"
        return True  # Assume enabled if using environment interpolation


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


# Utility functions for dataclass configuration management

def create_default_single_agent_config() -> SingleAgentConfig:
    """Create a default SingleAgentConfig with sensible defaults."""
    return SingleAgentConfig()


def create_default_multi_agent_config(num_agents: int = 3) -> MultiAgentConfig:
    """Create a default MultiAgentConfig with specified number of agents."""
    positions = [[float(i * 20), float(i * 20)] for i in range(num_agents)]
    orientations = [0.0] * num_agents
    speeds = [1.0] * num_agents
    max_speeds = [2.0] * num_agents
    angular_velocities = [0.1] * num_agents
    
    return MultiAgentConfig(
        positions=positions,
        orientations=orientations,
        speeds=speeds,
        max_speeds=max_speeds,
        angular_velocities=angular_velocities,
        num_agents=num_agents
    )


def create_default_navigator_config(mode: str = "single") -> NavigatorConfig:
    """Create a default NavigatorConfig for specified mode."""
    if mode == "single":
        return NavigatorConfig(
            mode="single",
            position=(50.0, 50.0),
            orientation=0.0,
            speed=1.0,
            max_speed=2.0,
            angular_velocity=0.1
        )
    elif mode == "multi":
        return NavigatorConfig(
            mode="multi",
            positions=[[20.0, 30.0], [70.0, 80.0]],
            orientations=[45.0, 135.0],
            speeds=[1.2, 0.8],
            max_speeds=[2.0, 2.0],
            angular_velocities=[0.1, 0.1],
            num_agents=2
        )
    else:
        return NavigatorConfig()


def create_default_video_plume_config(video_path: str = "${oc.env:VIDEO_PATH,./data/default.mp4}") -> VideoPlumeConfig:
    """Create a default VideoPlumeConfig with specified video path."""
    return VideoPlumeConfig(video_path=video_path)


def create_default_simulation_config() -> SimulationConfig:
    """Create a default SimulationConfig with sensible defaults."""
    return SimulationConfig()


def create_default_frame_cache_config(mode: str = "lru") -> FrameCacheConfig:
    """
    Create a default FrameCacheConfig with sensible defaults for specified mode.
    
    Args:
        mode: Cache mode ("none", "lru", "all")
        
    Returns:
        FrameCacheConfig instance with appropriate defaults
        
    Examples:
        >>> config = create_default_frame_cache_config("lru")
        >>> config.mode
        'lru'
        >>> config.memory_limit_mb
        '2GiB'
    """
    if mode == "none":
        return FrameCacheConfig(
            mode="none",
            memory_limit_mb=None,
            enable_cache_warming=False,
            enable_statistics=False
        )
    elif mode == "all":
        return FrameCacheConfig(
            mode="all",
            memory_limit_mb="4GiB",  # Larger default for full preload
            preload_strategy="sequential",
            enable_cache_warming=True,
            warm_cache_percentage=1.0,  # Preload everything
            eviction_policy="lru"  # Not used but kept for consistency
        )
    else:  # Default to LRU mode
        return FrameCacheConfig(
            mode="lru",
            memory_limit_mb="2GiB",
            eviction_policy="lru",
            preload_strategy="sequential",
            enable_cache_warming=True,
            warm_cache_percentage=0.1
        )


def dataclass_to_dict(config_obj) -> Dict[str, Any]:
    """Convert a dataclass configuration to dictionary, handling nested structures."""
    from dataclasses import asdict, is_dataclass
    
    if is_dataclass(config_obj):
        return asdict(config_obj)
    elif isinstance(config_obj, dict):
        return {k: dataclass_to_dict(v) for k, v in config_obj.items()}
    elif isinstance(config_obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in config_obj]
    else:
        return config_obj


def dict_to_dataclass(config_dict: Dict[str, Any], config_class):
    """Convert a dictionary to a dataclass configuration, with validation."""
    from dataclasses import fields, is_dataclass
    
    if not is_dataclass(config_class):
        raise ValueError(f"Target class {config_class} is not a dataclass")
    
    # Filter dict to only include fields that exist in the dataclass
    field_names = {f.name for f in fields(config_class)}
    filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
    
    try:
        return config_class(**filtered_dict)
    except Exception as e:
        raise ValueError(f"Failed to create {config_class.__name__} from dictionary: {e}")


def register_config_schemas():
    """
    Register all dataclass configuration schemas with Hydra ConfigStore for structured configuration.
    
    This function enables automatic schema discovery and validation within Hydra's
    configuration composition system. It registers dataclass schemas under appropriate groups:
    - navigator: single_agent, multi_agent, unified
    - video_plume: default  
    - simulation: standard
    
    Supports hierarchical configuration composition and factory-driven component
    instantiation per structured configuration requirements with dataclass validation.
    """
    try:
        # Import ConfigStore only when needed to avoid circular imports
        from hydra.core.config_store import ConfigStore
        
        cs = ConfigStore.instance()
        
        # Register individual dataclass component schemas under appropriate groups
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
        
        # Register FrameCacheConfig schemas under frame_cache group
        cs.store(
            group="frame_cache",
            name="none",
            node=FrameCacheConfig(
                mode="none",
                memory_limit_mb=None,
                enable_cache_warming=False,
                enable_statistics=False
            ),
            package="frame_cache"
        )
        
        cs.store(
            group="frame_cache",
            name="lru",
            node=FrameCacheConfig(
                mode="lru",
                memory_limit_mb="2GiB",
                eviction_policy="lru",
                enable_cache_warming=True
            ),
            package="frame_cache"
        )
        
        cs.store(
            group="frame_cache",
            name="all",
            node=FrameCacheConfig(
                mode="all",
                memory_limit_mb="4GiB",
                preload_strategy="sequential",
                warm_cache_percentage=1.0,
                enable_cache_warming=True
            ),
            package="frame_cache"
        )
        
        # Register default FrameCacheConfig schema
        cs.store(
            group="frame_cache",
            name="default",
            node=FrameCacheConfig,
            package="frame_cache"
        )
        
        # Register base configuration schema combining all dataclass components
        cs.store(
            name="base_config",
            node={
                "navigator": NavigatorConfig,
                "video_plume": VideoPlumeConfig, 
                "simulation": SimulationConfig,
                "frame_cache": FrameCacheConfig
            }
        )
        
        # Register dataclass instances for immediate use
        cs.store(
            name="base_config_instance",
            node={
                "navigator": NavigatorConfig(),
                "video_plume": VideoPlumeConfig(video_path="${oc.env:VIDEO_PATH,./data/default.mp4}"), 
                "simulation": SimulationConfig(),
                "frame_cache": FrameCacheConfig()
            }
        )
        
        logger.info("Successfully registered all dataclass configuration schemas with Hydra ConfigStore")
        
    except ImportError as e:
        logger.warning(f"Hydra not available, skipping ConfigStore registration: {e}")
    except Exception as e:
        logger.error(f"Failed to register dataclass configuration schemas: {e}")
        raise


# Enhanced exports for dataclass-based configurations and utilities
__all__ = [
    # Dataclass configuration models (primary interface)
    "NavigatorConfig",
    "SingleAgentConfig",
    "MultiAgentConfig", 
    "VideoPlumeConfig",
    "SimulationConfig",
    "FrameCacheConfig",
    
    # Configuration registration and utilities
    "register_config_schemas",
    "validate_env_interpolation",
    "resolve_env_value",
    
    # Validation utilities for dataclass fields
    "validate_speed_constraints",
    "validate_orientation_range",
    "validate_numeric_list",
    "validate_video_path",
    "validate_output_directory",
    "validate_cache_mode",
    "validate_memory_size",
    "validate_eviction_policy",
    
    # Post-initialization validators
    "__post_init_single_agent_config__",
    "__post_init_multi_agent_config__",
    "__post_init_navigator_config__",
    "__post_init_video_plume_config__",
    "__post_init_simulation_config__",
    "__post_init_frame_cache_config__",
    
    # Configuration factory functions for convenience
    "create_default_single_agent_config",
    "create_default_multi_agent_config",
    "create_default_navigator_config",
    "create_default_video_plume_config",
    "create_default_simulation_config",
    "create_default_frame_cache_config",
    
    # Dataclass conversion utilities
    "dataclass_to_dict",
    "dict_to_dataclass",
]
