"""
Core types and data structures module for plume_nav_sim providing comprehensive type definitions, 
data classes, enumerations, and factory functions for the Gymnasium-compatible reinforcement learning 
environment with full type safety, validation integration, and cross-component compatibility ensuring 
mathematical consistency and performance optimization.

This module establishes the foundational type system for the entire plume_nav_sim package, providing
immutable data structures, comprehensive validation, factory functions, and utility operations that
ensure type safety, performance optimization, and mathematical consistency across all components.
"""

# External imports with version comments
import numpy as np  # >=2.1.0 - Array operations, dtype definitions, mathematical calculations, and coordinate arithmetic
from typing import Union, Optional, Literal, TypeVar, Generic, Dict, List, Tuple, Any  # >=3.10 - Advanced type hints for comprehensive type safety
from dataclasses import dataclass, field  # >=3.10 - Data class decorators for structured type definitions with automatic method generation and validation
from enum import IntEnum, Enum  # >=3.10 - Enumeration classes for Action and RenderMode with proper value definitions and type safety
from abc import ABC, abstractmethod  # >=3.10 - Abstract base classes for type interfaces and protocol definitions ensuring consistent implementation
import copy  # >=3.10 - Deep copying operations for immutable type operations and state cloning in data class methods
import math  # >=3.10 - Mathematical operations including sqrt, floor, ceil for coordinate calculations and distance computations
import uuid  # >=3.10 - Unique identifier generation for tracking types like StateSnapshot and episode identification
import time  # >=3.10 - Timestamp generation for state snapshots, performance tracking, and temporal analysis

# Internal imports from constants module
from .constants import (
    DEFAULT_GRID_SIZE, DEFAULT_SOURCE_LOCATION, DEFAULT_PLUME_SIGMA, DEFAULT_GOAL_RADIUS, DEFAULT_MAX_STEPS,
    ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT, MOVEMENT_VECTORS, SUPPORTED_RENDER_MODES,
    FIELD_DTYPE, OBSERVATION_DTYPE, RGB_DTYPE, COORDINATE_DTYPE, MIN_PLUME_SIGMA, MAX_PLUME_SIGMA,
    MEMORY_LIMIT_PLUME_FIELD_MB
)

# Internal imports from exceptions module
from ..utils.exceptions import ValidationError, StateError, ConfigurationError

# Global constants for type validation and precision
COORDINATE_PRECISION = 1e-10
DISTANCE_CALCULATION_EPSILON = 1e-12
STATE_VALIDATION_ENABLED = True
TYPE_VALIDATION_STRICT_MODE = True
PERFORMANCE_TRACKING_ENABLED = True


class Action(IntEnum):
    """Enumeration class for discrete agent actions in cardinal directions providing type-safe action 
    representation, movement vector calculation, and integration with Gymnasium Discrete action space 
    for environment step processing."""
    
    UP = ACTION_UP      # 0 - Upward movement in coordinate system
    RIGHT = ACTION_RIGHT # 1 - Rightward movement in coordinate system  
    DOWN = ACTION_DOWN   # 2 - Downward movement in coordinate system
    LEFT = ACTION_LEFT   # 3 - Leftward movement in coordinate system
    
    def to_vector(self) -> Tuple[int, int]:
        """Convert action to movement vector (dx, dy) for position arithmetic and coordinate updates.
        
        Returns:
            tuple[int, int]: Movement vector (dx, dy) for coordinate calculations
            
        Raises:
            ValidationError: If action value is invalid or not found in MOVEMENT_VECTORS
        """
        try:
            return MOVEMENT_VECTORS[self.value]
        except KeyError:
            raise ValidationError(
                f"Invalid action value {self.value} not found in movement vectors",
                parameter_name="action",
                invalid_value=self.value,
                expected_format="Action enum value in range [0, 3]"
            )
    
    def opposite(self) -> 'Action':
        """Get opposite action for movement reversal and trajectory analysis.
        
        Returns:
            Action: Action enum representing opposite direction
        """
        opposite_map = {
            Action.UP: Action.DOWN,
            Action.DOWN: Action.UP,
            Action.RIGHT: Action.LEFT,
            Action.LEFT: Action.RIGHT
        }
        return opposite_map[self]
    
    def is_horizontal(self) -> bool:
        """Check if action represents horizontal movement (LEFT or RIGHT).
        
        Returns:
            bool: True if action is horizontal (LEFT/RIGHT), False if vertical (UP/DOWN)
        """
        return self in (Action.LEFT, Action.RIGHT)
    
    def is_vertical(self) -> bool:
        """Check if action represents vertical movement (UP or DOWN).
        
        Returns:
            bool: True if action is vertical (UP/DOWN), False if horizontal (LEFT/RIGHT)
        """
        return self in (Action.UP, Action.DOWN)


class RenderMode(Enum):
    """Enumeration class for visualization modes supporting dual-mode rendering with RGB array generation 
    for programmatic processing and human mode for interactive visualization with backend compatibility."""
    
    RGB_ARRAY = 'rgb_array'  # Programmatic NumPy array generation
    HUMAN = 'human'          # Interactive matplotlib visualization display
    
    def is_programmatic(self) -> bool:
        """Check if render mode is for programmatic processing (RGB_ARRAY).
        
        Returns:
            bool: True if mode is RGB_ARRAY for automated processing, False otherwise
        """
        return self == RenderMode.RGB_ARRAY
    
    def requires_display(self) -> bool:
        """Check if render mode requires display capabilities (HUMAN mode).
        
        Returns:
            bool: True if mode is HUMAN requiring display, False for RGB_ARRAY
        """
        return self == RenderMode.HUMAN
    
    def get_output_format(self) -> str:
        """Get expected output format description for render mode validation and documentation.
        
        Returns:
            str: Description of expected output format for the render mode
        """
        if self == RenderMode.RGB_ARRAY:
            return 'np.ndarray[H,W,3] uint8'
        else:  # HUMAN
            return 'Interactive matplotlib window (returns None)'


@dataclass(frozen=True)
class Coordinates:
    """Immutable data class representing 2D grid coordinates with utility methods for distance calculation, 
    movement operations, bounds checking, and coordinate arithmetic with numerical precision and validation support."""
    
    x: int
    y: int
    
    def __post_init__(self):
        """Initialize immutable coordinates with integer x,y values and validation."""
        # Validate x and y are integer types or safely convertible to integers
        if not isinstance(self.x, int) or not isinstance(self.y, int):
            raise ValidationError(
                f"Coordinates must be integers, got x={type(self.x).__name__}, y={type(self.y).__name__}",
                parameter_name="coordinates",
                invalid_value=(self.x, self.y),
                expected_format="tuple[int, int] with non-negative integers"
            )
        
        # Ensure coordinates are non-negative for grid system compatibility
        if self.x < 0 or self.y < 0:
            raise ValidationError(
                f"Coordinates must be non-negative, got ({self.x}, {self.y})",
                parameter_name="coordinates",
                invalid_value=(self.x, self.y),
                expected_format="Non-negative integer coordinates"
            )
    
    def distance_to(self, other: 'Coordinates', high_precision: bool = False) -> float:
        """Calculate Euclidean distance to another coordinate point with numerical precision handling.
        
        Args:
            other: Target coordinates for distance calculation
            high_precision: Enable high precision calculation with numerical stability
            
        Returns:
            float: Euclidean distance between coordinates with appropriate precision
            
        Raises:
            ValidationError: If other is not a valid Coordinates instance
        """
        if not isinstance(other, Coordinates):
            raise ValidationError(
                f"Distance calculation requires Coordinates instance, got {type(other).__name__}",
                parameter_name="other",
                invalid_value=other,
                expected_format="Coordinates instance"
            )
        
        return calculate_euclidean_distance(self, other, high_precision=high_precision)
    
    def move(self, movement: Union[Action, Tuple[int, int]], bounds: Optional['GridSize'] = None) -> 'Coordinates':
        """Create new Coordinates by applying movement vector or Action with immutability preservation.
        
        Args:
            movement: Action enum or movement vector (dx, dy) to apply
            bounds: Optional grid bounds for movement constraint validation
            
        Returns:
            Coordinates: New Coordinates object after applying movement with optional bounds checking
            
        Raises:
            ValidationError: If movement would result in invalid coordinates
        """
        # Convert Action to movement vector using to_vector() if Action provided
        if isinstance(movement, Action):
            dx, dy = movement.to_vector()
        elif isinstance(movement, tuple) and len(movement) == 2:
            dx, dy = movement
        else:
            raise ValidationError(
                f"Movement must be Action enum or tuple[int, int], got {type(movement).__name__}",
                parameter_name="movement",
                invalid_value=movement,
                expected_format="Action enum or tuple[int, int]"
            )
        
        # Calculate new position coordinates
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Apply bounds checking if bounds provided to constrain movement
        if bounds is not None:
            new_x = max(0, min(new_x, bounds.width - 1))
            new_y = max(0, min(new_y, bounds.height - 1))
        
        # Ensure new coordinates are non-negative
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        
        # Create new Coordinates preserving immutability
        return Coordinates(new_x, new_y)
    
    def is_within_bounds(self, grid_bounds: 'GridSize') -> bool:
        """Check if coordinates are within specified grid boundaries for bounds validation.
        
        Args:
            grid_bounds: GridSize object defining valid coordinate bounds
            
        Returns:
            bool: True if coordinates are within grid bounds, False otherwise
        """
        return (0 <= self.x < grid_bounds.width and 0 <= self.y < grid_bounds.height)
    
    def manhattan_distance_to(self, other: 'Coordinates') -> int:
        """Calculate Manhattan (L1) distance to another coordinate point for grid-based distance metrics.
        
        Args:
            other: Target coordinates for Manhattan distance calculation
            
        Returns:
            int: Manhattan distance (sum of absolute differences in coordinates)
        """
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def to_tuple(self) -> Tuple[int, int]:
        """Convert coordinates to tuple format for serialization and external interfaces.
        
        Returns:
            tuple[int, int]: Coordinates as (x, y) tuple
        """
        return (self.x, self.y)
    
    def clone(self) -> 'Coordinates':
        """Create identical copy of coordinates (for API consistency, returns self due to immutability).
        
        Returns:
            Coordinates: Identical Coordinates object (self for immutable types)
        """
        return self


@dataclass(frozen=True)
class GridSize:
    """Immutable data class representing 2D grid dimensions with utility methods for memory estimation, 
    center calculation, coordinate validation, and resource constraint analysis for environment configuration."""
    
    width: int
    height: int
    
    def __post_init__(self):
        """Initialize immutable grid dimensions with positive integer width,height and validation."""
        # Validate width and height are positive integers
        if not isinstance(self.width, int) or not isinstance(self.height, int):
            raise ValidationError(
                f"Grid dimensions must be integers, got width={type(self.width).__name__}, height={type(self.height).__name__}",
                parameter_name="grid_size",
                invalid_value=(self.width, self.height),
                expected_format="tuple[int, int] with positive integers"
            )
        
        if self.width <= 0 or self.height <= 0:
            raise ValidationError(
                f"Grid dimensions must be positive, got ({self.width}, {self.height})",
                parameter_name="grid_size",
                invalid_value=(self.width, self.height),
                expected_format="Positive integer dimensions"
            )
        
        # Ensure dimensions are within reasonable system limits for performance
        max_dimension = 1024  # Reasonable upper bound for memory management
        if self.width > max_dimension or self.height > max_dimension:
            raise ValidationError(
                f"Grid dimensions exceed maximum size {max_dimension}, got ({self.width}, {self.height})",
                parameter_name="grid_size",
                invalid_value=(self.width, self.height),
                expected_format=f"Dimensions <= {max_dimension}"
            )
    
    def total_cells(self) -> int:
        """Calculate total number of cells in grid for memory and performance analysis.
        
        Returns:
            int: Total cells (width × height) in grid for resource calculations
        """
        return self.width * self.height
    
    def center(self) -> Coordinates:
        """Calculate center coordinates of grid for source location defaults and centering operations.
        
        Returns:
            Coordinates: Center coordinates of grid using floor division for integer results
        """
        center_x = self.width // 2
        center_y = self.height // 2
        return Coordinates(center_x, center_y)
    
    def estimate_memory_mb(self, field_dtype: Optional[np.dtype] = None) -> float:
        """Estimate memory usage for plume field storage and system resource planning.
        
        Args:
            field_dtype: NumPy dtype for memory calculation, defaults to FIELD_DTYPE
            
        Returns:
            float: Estimated memory usage in megabytes for plume field storage
        """
        if field_dtype is None:
            field_dtype = FIELD_DTYPE
        
        # Calculate bytes per cell using NumPy dtype itemsize
        bytes_per_cell = np.dtype(field_dtype).itemsize
        
        # Calculate total bytes required
        total_bytes = self.total_cells() * bytes_per_cell
        
        # Convert to megabytes
        memory_mb = total_bytes / (1024 * 1024)
        
        return memory_mb
    
    def contains_coordinates(self, coordinates: Coordinates) -> bool:
        """Check if given coordinates are within grid boundaries for coordinate validation.
        
        Args:
            coordinates: Coordinates to validate against grid bounds
            
        Returns:
            bool: True if coordinates are within grid bounds, False otherwise
        """
        return coordinates.is_within_bounds(self)
    
    def is_performance_feasible(self, performance_targets: Optional[Dict[str, Any]] = None) -> bool:
        """Check if grid size meets performance targets for environment operations.
        
        Args:
            performance_targets: Optional performance constraints dictionary
            
        Returns:
            bool: True if grid size is within performance targets, False otherwise
        """
        # Check memory estimate against system limits
        memory_estimate = self.estimate_memory_mb()
        if memory_estimate > MEMORY_LIMIT_PLUME_FIELD_MB:
            return False
        
        # Check against performance targets if provided
        if performance_targets:
            if 'max_total_cells' in performance_targets:
                if self.total_cells() > performance_targets['max_total_cells']:
                    return False
            
            if 'max_memory_mb' in performance_targets:
                if memory_estimate > performance_targets['max_memory_mb']:
                    return False
        
        return True
    
    def to_tuple(self) -> Tuple[int, int]:
        """Convert grid size to tuple format for serialization and external interfaces.
        
        Returns:
            tuple[int, int]: Grid dimensions as (width, height) tuple
        """
        return (self.width, self.height)


@dataclass
class AgentState:
    """Mutable data class for tracking agent state including position, rewards, step count, movement history, 
    and performance metrics with state transition validation and trajectory analysis for episode management."""
    
    position: Coordinates
    step_count: int = 0
    total_reward: float = 0.0
    movement_history: List[Coordinates] = field(default_factory=list)
    goal_reached: bool = False
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize agent state with validation and performance tracking setup."""
        # Validate position is valid Coordinates instance
        if not isinstance(self.position, Coordinates):
            raise ValidationError(
                f"Agent position must be Coordinates instance, got {type(self.position).__name__}",
                parameter_name="position",
                invalid_value=self.position,
                expected_format="Coordinates instance"
            )
        
        # Initialize performance tracking if enabled
        if PERFORMANCE_TRACKING_ENABLED:
            self.performance_metrics.setdefault('state_updates', 0)
            self.performance_metrics.setdefault('position_changes', 0)
            self.performance_metrics.setdefault('reward_changes', 0)
    
    def update_position(self, new_position: Coordinates, record_history: bool = True) -> None:
        """Update agent position with movement validation and history tracking.
        
        Args:
            new_position: New position coordinates for agent
            record_history: Whether to record current position in movement history
            
        Raises:
            ValidationError: If new_position is not valid Coordinates
        """
        if not isinstance(new_position, Coordinates):
            raise ValidationError(
                f"New position must be Coordinates instance, got {type(new_position).__name__}",
                parameter_name="new_position",
                invalid_value=new_position,
                expected_format="Coordinates instance"
            )
        
        # Add current position to movement history if requested
        if record_history:
            self.movement_history.append(self.position)
        
        # Update position
        self.position = new_position
        
        # Update performance metrics
        if PERFORMANCE_TRACKING_ENABLED:
            self.performance_metrics['position_changes'] += 1
            self.performance_metrics['state_updates'] += 1
    
    def add_reward(self, reward: float, validate_reward: bool = True) -> None:
        """Add reward to total with validation and performance tracking.
        
        Args:
            reward: Reward value to add to total
            validate_reward: Whether to validate reward is numeric
            
        Raises:
            ValidationError: If reward is not a valid numeric type
        """
        if validate_reward:
            if not isinstance(reward, (int, float)):
                raise ValidationError(
                    f"Reward must be numeric type, got {type(reward).__name__}",
                    parameter_name="reward",
                    invalid_value=reward,
                    expected_format="float or int value"
                )
        
        # Add reward with numerical stability
        self.total_reward += float(reward)
        
        # Update performance metrics
        if PERFORMANCE_TRACKING_ENABLED:
            self.performance_metrics['reward_changes'] += 1
            self.performance_metrics['state_updates'] += 1
        
        # Check for goal achievement indication
        if reward > 0:
            self.goal_reached = True
    
    def increment_step(self) -> None:
        """Increment step count with validation and performance tracking."""
        self.step_count += 1
        
        # Validate step count is within reasonable limits
        if self.step_count > DEFAULT_MAX_STEPS * 10:  # Allow some flexibility
            raise StateError(
                f"Step count {self.step_count} exceeds reasonable episode limits",
                current_state=f"step_count={self.step_count}",
                expected_state=f"step_count <= {DEFAULT_MAX_STEPS * 10}",
                component_name="agent_state"
            )
        
        # Update performance metrics
        if PERFORMANCE_TRACKING_ENABLED:
            self.performance_metrics['state_updates'] += 1
    
    def reset(self, new_position: Optional[Coordinates] = None, preserve_performance_metrics: bool = False) -> None:
        """Reset agent state for new episode with optional position preservation.
        
        Args:
            new_position: Optional new position for reset, keeps current if None
            preserve_performance_metrics: Whether to preserve performance data
        """
        # Update position if provided
        if new_position is not None:
            if not isinstance(new_position, Coordinates):
                raise ValidationError(
                    f"New position must be Coordinates instance, got {type(new_position).__name__}",
                    parameter_name="new_position",
                    invalid_value=new_position,
                    expected_format="Coordinates instance"
                )
            self.position = new_position
        
        # Reset episode-specific state
        self.step_count = 0
        self.total_reward = 0.0
        self.movement_history.clear()
        self.goal_reached = False
        
        # Handle performance metrics
        if not preserve_performance_metrics:
            self.performance_metrics.clear()
        elif PERFORMANCE_TRACKING_ENABLED:
            # Reset counters but preserve structure
            self.performance_metrics['state_updates'] = 0
            self.performance_metrics['position_changes'] = 0
            self.performance_metrics['reward_changes'] = 0
    
    def get_trajectory(self, include_current_position: bool = True) -> List[Coordinates]:
        """Get complete movement trajectory including current position for analysis.
        
        Args:
            include_current_position: Whether to include current position in trajectory
            
        Returns:
            list[Coordinates]: Complete trajectory of agent movement
        """
        # Copy movement history to avoid external modification
        trajectory = self.movement_history.copy()
        
        # Append current position if requested
        if include_current_position:
            trajectory.append(self.position)
        
        return trajectory
    
    def calculate_trajectory_length(self) -> float:
        """Calculate total distance traveled by agent for performance analysis.
        
        Returns:
            float: Total distance traveled by agent during episode
        """
        if len(self.movement_history) < 2:
            return 0.0
        
        total_distance = 0.0
        
        # Calculate distance between consecutive positions
        for i in range(len(self.movement_history) - 1):
            current_pos = self.movement_history[i]
            next_pos = self.movement_history[i + 1]
            total_distance += current_pos.distance_to(next_pos)
        
        # Add distance from last history position to current position
        if self.movement_history:
            total_distance += self.movement_history[-1].distance_to(self.position)
        
        return total_distance
    
    def to_dict(self, include_history: bool = False, include_performance_metrics: bool = False) -> Dict[str, Any]:
        """Convert agent state to dictionary for serialization and external analysis.
        
        Args:
            include_history: Whether to include movement history
            include_performance_metrics: Whether to include performance data
            
        Returns:
            dict: Agent state as dictionary with optional history and performance data
        """
        # Create base dictionary
        state_dict = {
            'position': self.position.to_tuple(),
            'step_count': self.step_count,
            'total_reward': self.total_reward,
            'goal_reached': self.goal_reached
        }
        
        # Include movement history if requested
        if include_history:
            state_dict['movement_history'] = [pos.to_tuple() for pos in self.movement_history]
        
        # Include performance metrics if requested
        if include_performance_metrics:
            state_dict['performance_metrics'] = self.performance_metrics.copy()
        
        return state_dict


@dataclass
class EpisodeState:
    """Data class for comprehensive episode state management including agent state, termination flags, 
    episode tracking, and performance analysis with history management and statistical analysis for 
    research reproducibility."""
    
    agent_state: AgentState
    terminated: bool = False
    truncated: bool = False
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    episode_summary: Dict[str, Any] = field(default_factory=dict)
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize episode state with validation and tracking setup."""
        # Validate agent_state is valid AgentState
        if not isinstance(self.agent_state, AgentState):
            raise ValidationError(
                f"Agent state must be AgentState instance, got {type(self.agent_state).__name__}",
                parameter_name="agent_state",
                invalid_value=self.agent_state,
                expected_format="AgentState instance"
            )
        
        # Validate termination flags are boolean
        if not isinstance(self.terminated, bool) or not isinstance(self.truncated, bool):
            raise ValidationError(
                "Termination flags must be boolean values",
                parameter_name="termination_flags",
                invalid_value=(self.terminated, self.truncated),
                expected_format="bool, bool"
            )
        
        # Validate mutually exclusive termination states
        if self.terminated and self.truncated:
            raise ValidationError(
                "Episode cannot be both terminated and truncated simultaneously",
                parameter_name="termination_flags",
                invalid_value=(self.terminated, self.truncated),
                expected_format="Only one termination flag can be True"
            )
    
    def is_done(self) -> bool:
        """Check if episode is complete (terminated or truncated) for control flow.
        
        Returns:
            bool: True if episode is terminated or truncated, False if active
        """
        return self.terminated or self.truncated
    
    def set_termination(self, terminated: bool, truncated: bool, reason: Optional[str] = None) -> None:
        """Set episode termination with reason and timing for proper episode closure.
        
        Args:
            terminated: Whether episode terminated successfully (goal reached)
            truncated: Whether episode was truncated (time/step limit)
            reason: Optional reason for termination
            
        Raises:
            ValidationError: If termination flags are invalid
        """
        # Validate termination flags
        if terminated and truncated:
            raise ValidationError(
                "Episode cannot be both terminated and truncated simultaneously",
                parameter_name="termination_flags",
                invalid_value=(terminated, truncated),
                expected_format="Only one termination flag can be True"
            )
        
        self.terminated = terminated
        self.truncated = truncated
        
        # Record end time if episode is now complete
        if self.is_done() and self.end_time is None:
            self.end_time = time.time()
        
        # Add termination reason to episode summary
        if reason:
            self.episode_summary['termination_reason'] = reason
        
        # Update agent state goal reached status
        if terminated:
            self.agent_state.goal_reached = True
    
    def get_episode_duration(self) -> float:
        """Calculate episode duration for performance analysis and statistics.
        
        Returns:
            float: Episode duration in seconds, or current duration if episode active
        """
        end_time = self.end_time if self.end_time is not None else time.time()
        return end_time - self.start_time
    
    def get_episode_summary(self, include_trajectory_analysis: bool = False, 
                           include_performance_metrics: bool = False) -> Dict[str, Any]:
        """Generate comprehensive episode summary with statistics and performance metrics.
        
        Args:
            include_trajectory_analysis: Whether to include trajectory analysis
            include_performance_metrics: Whether to include performance data
            
        Returns:
            dict: Comprehensive episode summary with statistics and analysis
        """
        # Compile basic episode information
        summary = {
            'episode_id': self.episode_id,
            'duration_seconds': self.get_episode_duration(),
            'step_count': self.agent_state.step_count,
            'total_reward': self.agent_state.total_reward,
            'terminated': self.terminated,
            'truncated': self.truncated,
            'goal_reached': self.agent_state.goal_reached,
            'final_position': self.agent_state.position.to_tuple()
        }
        
        # Include trajectory analysis if requested
        if include_trajectory_analysis:
            trajectory = self.agent_state.get_trajectory(include_current_position=True)
            summary['trajectory_analysis'] = {
                'total_distance': self.agent_state.calculate_trajectory_length(),
                'position_count': len(trajectory),
                'start_position': trajectory[0].to_tuple() if trajectory else None,
                'end_position': trajectory[-1].to_tuple() if trajectory else None
            }
        
        # Include performance metrics if requested
        if include_performance_metrics:
            summary['performance_metrics'] = self.agent_state.performance_metrics.copy()
        
        # Include episode summary data
        summary.update(self.episode_summary)
        
        return summary
    
    def record_state(self, additional_context: Optional[Dict[str, Any]] = None) -> None:
        """Record current state in history for episode replay and analysis.
        
        Args:
            additional_context: Optional additional context information
        """
        # Create state snapshot
        snapshot = {
            'timestamp': time.time(),
            'step_count': self.agent_state.step_count,
            'agent_position': self.agent_state.position.to_tuple(),
            'total_reward': self.agent_state.total_reward,
            'terminated': self.terminated,
            'truncated': self.truncated
        }
        
        # Include additional context if provided
        if additional_context:
            snapshot['additional_context'] = additional_context.copy()
        
        # Add to state history
        self.state_history.append(snapshot)
        
        # Limit history size to prevent memory issues
        max_history_size = 10000  # Reasonable limit for episode history
        if len(self.state_history) > max_history_size:
            # Remove oldest entries, keeping most recent
            self.state_history = self.state_history[-max_history_size:]
    
    def get_state_history(self, max_entries: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve episode state history for replay, analysis, and debugging.
        
        Args:
            max_entries: Optional limit on number of entries returned
            
        Returns:
            list: Episode state history with optional entry limit
        """
        # Return copy to prevent external modification
        history = self.state_history.copy()
        
        # Apply max entries limit if specified
        if max_entries is not None and max_entries > 0:
            history = history[-max_entries:]
        
        return history
    
    def reset(self, new_agent_state: AgentState, preserve_episode_id: bool = False) -> None:
        """Reset episode state for new episode while preserving configuration.
        
        Args:
            new_agent_state: New agent state for fresh episode
            preserve_episode_id: Whether to keep same episode ID
            
        Raises:
            ValidationError: If new_agent_state is invalid
        """
        if not isinstance(new_agent_state, AgentState):
            raise ValidationError(
                f"New agent state must be AgentState instance, got {type(new_agent_state).__name__}",
                parameter_name="new_agent_state",
                invalid_value=new_agent_state,
                expected_format="AgentState instance"
            )
        
        # Update agent state
        self.agent_state = new_agent_state
        
        # Reset episode flags
        self.terminated = False
        self.truncated = False
        
        # Reset episode tracking
        if not preserve_episode_id:
            self.episode_id = str(uuid.uuid4())
        
        self.start_time = time.time()
        self.end_time = None
        
        # Clear episode data
        self.episode_summary.clear()
        self.state_history.clear()


@dataclass
class PlumeParameters:
    """Data class for plume model configuration including source location, dispersion parameters, 
    mathematical validation, and Gaussian model consistency checking with grid compatibility and 
    numerical stability analysis."""
    
    source_location: Coordinates
    sigma: float
    grid_compatibility: Optional[GridSize] = None
    validation_results: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize plume parameters with validation setup."""
        # Validate source_location is valid Coordinates
        if not isinstance(self.source_location, Coordinates):
            raise ValidationError(
                f"Source location must be Coordinates instance, got {type(self.source_location).__name__}",
                parameter_name="source_location",
                invalid_value=self.source_location,
                expected_format="Coordinates instance"
            )
        
        # Validate sigma is within acceptable range
        if not isinstance(self.sigma, (int, float)):
            raise ValidationError(
                f"Sigma must be numeric type, got {type(self.sigma).__name__}",
                parameter_name="sigma",
                invalid_value=self.sigma,
                expected_format="float value"
            )
        
        if not (MIN_PLUME_SIGMA <= self.sigma <= MAX_PLUME_SIGMA):
            raise ValidationError(
                f"Sigma {self.sigma} outside valid range [{MIN_PLUME_SIGMA}, {MAX_PLUME_SIGMA}]",
                parameter_name="sigma",
                invalid_value=self.sigma,
                expected_format=f"float in range [{MIN_PLUME_SIGMA}, {MAX_PLUME_SIGMA}]"
            )
    
    def validate(self, grid_size: Optional[GridSize] = None, strict_validation: bool = False) -> bool:
        """Comprehensive validation of plume parameters including mathematical consistency and grid compatibility.
        
        Args:
            grid_size: Optional grid size for compatibility validation
            strict_validation: Enable additional precision and bounds checking
            
        Returns:
            bool: True if parameters are valid, raises ValidationError if invalid
            
        Raises:
            ValidationError: If parameters fail validation
        """
        validation_errors = []
        
        # Validate sigma is within allowed range
        if not (MIN_PLUME_SIGMA <= self.sigma <= MAX_PLUME_SIGMA):
            validation_errors.append(f"Sigma {self.sigma} outside valid range")
        
        # Check source location is within grid bounds if grid_size provided
        if grid_size is not None:
            if not self.source_location.is_within_bounds(grid_size):
                validation_errors.append(f"Source location {self.source_location.to_tuple()} outside grid bounds")
            
            # Store grid compatibility for future reference
            self.grid_compatibility = grid_size
        
        # Validate mathematical consistency in strict mode
        if strict_validation and grid_size is not None:
            # Check if plume will have reasonable spread given grid size
            grid_diagonal = math.sqrt(grid_size.width**2 + grid_size.height**2)
            if self.sigma > grid_diagonal / 2:
                validation_errors.append(f"Sigma {self.sigma} too large for grid diagonal {grid_diagonal}")
        
        # Store validation results
        self.validation_results = {
            'validated_at': time.time(),
            'strict_mode': strict_validation,
            'grid_compatible': grid_size is not None,
            'validation_errors': validation_errors
        }
        
        if validation_errors:
            raise ValidationError(
                f"Plume parameter validation failed: {'; '.join(validation_errors)}",
                parameter_name="plume_parameters",
                invalid_value=(self.source_location.to_tuple(), self.sigma),
                expected_format="Valid source location and sigma within acceptable range"
            )
        
        return True
    
    def estimate_field_generation_time(self, grid_size: GridSize) -> float:
        """Estimate computational time for plume field generation based on grid size and sigma.
        
        Args:
            grid_size: Grid size for field generation
            
        Returns:
            float: Estimated field generation time in milliseconds
        """
        # Base computation complexity scales with grid size
        base_complexity = grid_size.total_cells()
        
        # Sigma affects precision requirements (smaller sigma needs more precision)
        sigma_factor = max(1.0, MIN_PLUME_SIGMA / self.sigma)
        
        # Estimate based on empirical measurements (placeholder values)
        estimated_ms = (base_complexity * sigma_factor) / 100000  # Rough estimate
        
        return estimated_ms
    
    def get_concentration_at(self, position: Coordinates, high_precision: bool = False) -> float:
        """Calculate theoretical concentration at given coordinates using Gaussian formula.
        
        Args:
            position: Position coordinates for concentration calculation
            high_precision: Enable high precision calculations
            
        Returns:
            float: Theoretical concentration value [0,1] at specified position
        """
        # Calculate distance from position to source location
        distance = self.source_location.distance_to(position, high_precision=high_precision)
        
        # Apply Gaussian formula: exp(-distance² / (2 * sigma²))
        concentration = math.exp(-distance**2 / (2 * self.sigma**2))
        
        # Clamp result to [0.0, 1.0] range for consistency
        return max(0.0, min(1.0, concentration))
    
    def get_effective_radius(self, concentration_threshold: float = 0.01) -> float:
        """Calculate effective radius where concentration drops below threshold for plume extent analysis.
        
        Args:
            concentration_threshold: Concentration threshold for radius calculation
            
        Returns:
            float: Radius beyond which concentration falls below threshold
            
        Raises:
            ValidationError: If concentration_threshold is invalid
        """
        if not (0.0 < concentration_threshold < 1.0):
            raise ValidationError(
                f"Concentration threshold {concentration_threshold} must be in range (0, 1)",
                parameter_name="concentration_threshold",
                invalid_value=concentration_threshold,
                expected_format="float in range (0.0, 1.0)"
            )
        
        # Solve Gaussian equation for radius: exp(-r²/(2σ²)) = threshold
        # r = σ * sqrt(-2 * ln(threshold))
        radius = self.sigma * math.sqrt(-2 * math.log(concentration_threshold))
        
        return radius
    
    def clone(self, new_source_location: Optional[Coordinates] = None, 
              new_sigma: Optional[float] = None) -> 'PlumeParameters':
        """Create deep copy of plume parameters with optional modifications.
        
        Args:
            new_source_location: Optional new source location
            new_sigma: Optional new sigma value
            
        Returns:
            PlumeParameters: New PlumeParameters instance with optional modifications
        """
        # Use new values if provided, otherwise copy current values
        source_loc = new_source_location if new_source_location is not None else self.source_location
        sigma_val = new_sigma if new_sigma is not None else self.sigma
        
        # Create new instance
        new_params = PlumeParameters(source_location=source_loc, sigma=sigma_val)
        
        # Copy compatibility and validation data if no modifications
        if new_source_location is None and new_sigma is None:
            new_params.grid_compatibility = self.grid_compatibility
            new_params.validation_results = self.validation_results.copy()
        
        return new_params
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plume parameters to dictionary for serialization and configuration storage.
        
        Returns:
            dict: Plume parameters as dictionary with source location and sigma
        """
        result = {
            'source_location': self.source_location.to_tuple(),
            'sigma': self.sigma
        }
        
        # Add validation results if available
        if self.validation_results:
            result['validation_results'] = self.validation_results.copy()
        
        # Add grid compatibility if available
        if self.grid_compatibility:
            result['grid_compatibility'] = self.grid_compatibility.to_tuple()
        
        return result


@dataclass
class EnvironmentConfig:
    """Comprehensive environment configuration data class combining all environment parameters with 
    cross-parameter validation, resource estimation, and configuration consistency checking for complete 
    environment setup."""
    
    grid_size: GridSize
    plume_params: PlumeParameters
    max_steps: int = DEFAULT_MAX_STEPS
    goal_radius: float = DEFAULT_GOAL_RADIUS
    enable_validation: bool = True
    enable_performance_monitoring: bool = True
    resource_estimates: Dict[str, Any] = field(default_factory=dict)
    configuration_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize comprehensive environment configuration with validation setup."""
        # Validate grid_size is valid GridSize
        if not isinstance(self.grid_size, GridSize):
            raise ConfigurationError(
                f"Grid size must be GridSize instance, got {type(self.grid_size).__name__}",
                config_parameter="grid_size",
                invalid_value=self.grid_size
            )
        
        # Validate plume_params is valid PlumeParameters
        if not isinstance(self.plume_params, PlumeParameters):
            raise ConfigurationError(
                f"Plume params must be PlumeParameters instance, got {type(self.plume_params).__name__}",
                config_parameter="plume_params",
                invalid_value=self.plume_params
            )
        
        # Validate max_steps is positive integer
        if not isinstance(self.max_steps, int) or self.max_steps <= 0:
            raise ConfigurationError(
                f"Max steps must be positive integer, got {self.max_steps}",
                config_parameter="max_steps",
                invalid_value=self.max_steps
            )
        
        # Validate goal_radius is non-negative float
        if not isinstance(self.goal_radius, (int, float)) or self.goal_radius < 0:
            raise ConfigurationError(
                f"Goal radius must be non-negative number, got {self.goal_radius}",
                config_parameter="goal_radius",
                invalid_value=self.goal_radius
            )
        
        # Initialize metadata
        self.configuration_metadata['created_at'] = time.time()
        self.configuration_metadata['config_version'] = '1.0'
    
    def validate(self, strict_mode: bool = False, check_resource_constraints: bool = True) -> bool:
        """Comprehensive validation of all configuration parameters with cross-parameter consistency checking.
        
        Args:
            strict_mode: Enable additional validation rules
            check_resource_constraints: Whether to validate resource usage
            
        Returns:
            bool: True if configuration is valid, raises ConfigurationError if invalid
            
        Raises:
            ConfigurationError: If configuration validation fails
        """
        validation_errors = []
        
        # Validate grid size and memory feasibility
        if check_resource_constraints:
            memory_estimate = self.grid_size.estimate_memory_mb()
            if memory_estimate > MEMORY_LIMIT_PLUME_FIELD_MB:
                validation_errors.append(f"Grid memory estimate {memory_estimate:.1f}MB exceeds limit")
        
        # Validate plume parameters with grid compatibility
        try:
            self.plume_params.validate(grid_size=self.grid_size, strict_validation=strict_mode)
        except ValidationError as e:
            validation_errors.append(f"Plume validation failed: {e.message}")
        
        # Check source location is within grid boundaries
        if not self.plume_params.source_location.is_within_bounds(self.grid_size):
            validation_errors.append("Source location outside grid boundaries")
        
        # Validate step and goal parameters
        if strict_mode:
            if self.max_steps > 50000:  # Very long episodes
                validation_errors.append(f"Max steps {self.max_steps} may cause performance issues")
            
            # Check goal radius makes sense for grid size
            grid_diagonal = math.sqrt(self.grid_size.width**2 + self.grid_size.height**2)
            if self.goal_radius > grid_diagonal / 2:
                validation_errors.append(f"Goal radius {self.goal_radius} too large for grid")
        
        if validation_errors:
            raise ConfigurationError(
                f"Configuration validation failed: {'; '.join(validation_errors)}",
                config_parameter="environment_config",
                invalid_value="multiple_parameters"
            )
        
        return True
    
    def estimate_resources(self) -> Dict[str, Any]:
        """Estimate computational and memory resources required for environment configuration.
        
        Returns:
            dict: Resource estimates including memory, computation time, and performance projections
        """
        # Memory estimates
        plume_memory = self.grid_size.estimate_memory_mb()
        base_memory = 10.0  # Base environment overhead
        total_memory = plume_memory + base_memory
        
        # Computation time estimates
        plume_generation_time = self.plume_params.estimate_field_generation_time(self.grid_size)
        episode_time_estimate = (self.max_steps * 1.0) / 1000  # 1ms per step target
        
        # Compile resource estimates
        estimates = {
            'memory_usage_mb': {
                'plume_field': plume_memory,
                'base_environment': base_memory,
                'total_estimated': total_memory
            },
            'computation_time_ms': {
                'plume_generation': plume_generation_time,
                'episode_estimated': episode_time_estimate,
                'reset_overhead': 10.0
            },
            'performance_projections': {
                'episodes_per_second': 1000 / (episode_time_estimate + 10.0),
                'memory_efficiency': 'good' if total_memory < MEMORY_LIMIT_PLUME_FIELD_MB else 'concerning'
            }
        }
        
        # Store estimates for future reference
        self.resource_estimates = estimates
        
        return estimates
    
    def get_gymnasium_config(self) -> Dict[str, Any]:
        """Generate Gymnasium-compatible configuration dictionary for environment registration.
        
        Returns:
            dict: Gymnasium registration configuration with entry points and parameters
        """
        config = {
            'id': 'PlumeNav-StaticGaussian-v0',
            'entry_point': 'plume_nav_sim.environment:PlumeSearchEnv',
            'max_episode_steps': self.max_steps,
            'kwargs': {
                'grid_size': self.grid_size.to_tuple(),
                'source_location': self.plume_params.source_location.to_tuple(),
                'sigma': self.plume_params.sigma,
                'goal_radius': self.goal_radius
            }
        }
        
        return config
    
    def clone(self, overrides: Optional[Dict[str, Any]] = None) -> 'EnvironmentConfig':
        """Create deep copy of environment configuration with optional parameter overrides.
        
        Args:
            overrides: Optional dictionary of parameter overrides
            
        Returns:
            EnvironmentConfig: New EnvironmentConfig with optional parameter modifications
        """
        # Create deep copies of mutable components
        new_grid_size = GridSize(self.grid_size.width, self.grid_size.height)
        new_plume_params = self.plume_params.clone()
        
        # Apply overrides if provided
        max_steps = self.max_steps
        goal_radius = self.goal_radius
        
        if overrides:
            if 'max_steps' in overrides:
                max_steps = overrides['max_steps']
            if 'goal_radius' in overrides:
                goal_radius = overrides['goal_radius']
            
            # Handle grid size override
            if 'grid_size' in overrides:
                gs = overrides['grid_size']
                if isinstance(gs, tuple) and len(gs) == 2:
                    new_grid_size = GridSize(gs[0], gs[1])
                elif isinstance(gs, GridSize):
                    new_grid_size = gs
        
        # Create new configuration
        new_config = EnvironmentConfig(
            grid_size=new_grid_size,
            plume_params=new_plume_params,
            max_steps=max_steps,
            goal_radius=goal_radius,
            enable_validation=self.enable_validation,
            enable_performance_monitoring=self.enable_performance_monitoring
        )
        
        return new_config
    
    def to_dict(self, include_resource_estimates: bool = False, 
                include_metadata: bool = False) -> Dict[str, Any]:
        """Convert complete configuration to dictionary for serialization and storage.
        
        Args:
            include_resource_estimates: Whether to include resource estimates
            include_metadata: Whether to include configuration metadata
            
        Returns:
            dict: Complete configuration as dictionary with optional estimates and metadata
        """
        config_dict = {
            'grid_size': self.grid_size.to_tuple(),
            'plume_params': self.plume_params.to_dict(),
            'max_steps': self.max_steps,
            'goal_radius': self.goal_radius,
            'enable_validation': self.enable_validation,
            'enable_performance_monitoring': self.enable_performance_monitoring
        }
        
        # Include resource estimates if requested
        if include_resource_estimates:
            config_dict['resource_estimates'] = self.resource_estimates.copy()
        
        # Include metadata if requested
        if include_metadata:
            config_dict['configuration_metadata'] = self.configuration_metadata.copy()
        
        return config_dict
    
    def get_configuration_hash(self, include_metadata: bool = False) -> str:
        """Generate hash of configuration parameters for caching and comparison.
        
        Args:
            include_metadata: Whether to include metadata in hash calculation
            
        Returns:
            str: Hash string representing configuration for caching and comparison
        """
        import hashlib
        
        # Create consistent string representation
        config_data = self.to_dict(include_metadata=include_metadata)
        config_str = str(sorted(config_data.items()))
        
        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(config_str.encode('utf-8'))
        return hash_obj.hexdigest()


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable snapshot of complete environment state including agent state, episode status, timestamp, 
    and performance metrics for debugging, analysis, state restoration, and research reproducibility with 
    comprehensive state preservation."""
    
    agent_state: AgentState
    episode_state: EpisodeState
    timestamp: float
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Create immutable state snapshot with deep copies and unique identification."""
        # Create deep copies using copy.deepcopy to ensure immutability
        object.__setattr__(self, 'agent_state', copy.deepcopy(self.agent_state))
        object.__setattr__(self, 'episode_state', copy.deepcopy(self.episode_state))
        
        # Validate timestamp is reasonable
        current_time = time.time()
        if self.timestamp > current_time + 60:  # Allow 1 minute future tolerance
            raise ValidationError(
                f"Snapshot timestamp {self.timestamp} is in the future",
                parameter_name="timestamp",
                invalid_value=self.timestamp,
                expected_format="timestamp <= current_time"
            )
    
    def compare_with(self, other_snapshot: 'StateSnapshot') -> Dict[str, Any]:
        """Compare this snapshot with another to identify state differences and transitions.
        
        Args:
            other_snapshot: Another state snapshot for comparison
            
        Returns:
            dict: Comprehensive comparison with differences, transitions, and analysis
        """
        if not isinstance(other_snapshot, StateSnapshot):
            raise ValidationError(
                f"Comparison requires StateSnapshot instance, got {type(other_snapshot).__name__}",
                parameter_name="other_snapshot",
                invalid_value=other_snapshot,
                expected_format="StateSnapshot instance"
            )
        
        # Compare agent state differences
        agent_diff = {
            'position_changed': self.agent_state.position != other_snapshot.agent_state.position,
            'position_delta': None,
            'step_count_change': self.agent_state.step_count - other_snapshot.agent_state.step_count,
            'reward_change': self.agent_state.total_reward - other_snapshot.agent_state.total_reward,
            'goal_status_change': self.agent_state.goal_reached != other_snapshot.agent_state.goal_reached
        }
        
        if agent_diff['position_changed']:
            agent_diff['position_delta'] = (
                self.agent_state.position.x - other_snapshot.agent_state.position.x,
                self.agent_state.position.y - other_snapshot.agent_state.position.y
            )
        
        # Compare episode state differences
        episode_diff = {
            'termination_changed': (
                self.episode_state.terminated != other_snapshot.episode_state.terminated or
                self.episode_state.truncated != other_snapshot.episode_state.truncated
            ),
            'time_elapsed': self.timestamp - other_snapshot.timestamp,
            'episode_status': {
                'previous': (other_snapshot.episode_state.terminated, other_snapshot.episode_state.truncated),
                'current': (self.episode_state.terminated, self.episode_state.truncated)
            }
        }
        
        return {
            'comparison_id': str(uuid.uuid4()),
            'comparison_timestamp': time.time(),
            'agent_differences': agent_diff,
            'episode_differences': episode_diff,
            'snapshot_ids': (other_snapshot.snapshot_id, self.snapshot_id),
            'significant_changes': any([
                agent_diff['position_changed'],
                agent_diff['goal_status_change'],
                episode_diff['termination_changed']
            ])
        }
    
    def validate_consistency(self) -> Dict[str, Any]:
        """Validate internal consistency of snapshot data for integrity checking.
        
        Returns:
            dict: Validation results with consistency analysis and issues found
        """
        validation_issues = []
        
        # Validate agent and episode state consistency
        if self.episode_state.agent_state.position != self.agent_state.position:
            validation_issues.append("Agent position mismatch between agent_state and episode_state")
        
        if self.episode_state.agent_state.step_count != self.agent_state.step_count:
            validation_issues.append("Step count mismatch between agent_state and episode_state")
        
        # Check timestamp reasonableness
        current_time = time.time()
        if self.timestamp > current_time:
            validation_issues.append("Snapshot timestamp is in the future")
        
        # Validate snapshot ID format (UUID)
        try:
            uuid.UUID(self.snapshot_id)
        except ValueError:
            validation_issues.append("Invalid snapshot ID format")
        
        # Check goal state consistency
        if self.agent_state.goal_reached and not self.episode_state.terminated:
            validation_issues.append("Goal reached but episode not terminated")
        
        validation_report = {
            'validation_timestamp': current_time,
            'consistency_valid': len(validation_issues) == 0,
            'issues_found': validation_issues,
            'snapshot_id': self.snapshot_id,
            'agent_state_valid': isinstance(self.agent_state, AgentState),
            'episode_state_valid': isinstance(self.episode_state, EpisodeState)
        }
        
        # Store validation results
        object.__setattr__(self, 'validation_results', validation_report)
        
        return validation_report
    
    def extract_trajectory_segment(self, include_history: bool = True) -> List[Coordinates]:
        """Extract trajectory information from snapshot for movement analysis.
        
        Args:
            include_history: Whether to include movement history
            
        Returns:
            list[Coordinates]: Trajectory segment from snapshot for analysis
        """
        if include_history:
            return self.agent_state.get_trajectory(include_current_position=True)
        else:
            return [self.agent_state.position]
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Generate concise state summary for logging and display.
        
        Returns:
            dict: Concise state summary with key information and metrics
        """
        return {
            'snapshot_id': self.snapshot_id,
            'timestamp': self.timestamp,
            'agent_position': self.agent_state.position.to_tuple(),
            'step_count': self.agent_state.step_count,
            'total_reward': self.agent_state.total_reward,
            'goal_reached': self.agent_state.goal_reached,
            'episode_terminated': self.episode_state.terminated,
            'episode_truncated': self.episode_state.truncated,
            'episode_duration': self.episode_state.get_episode_duration()
        }
    
    def to_dict(self, include_performance_data: bool = False, 
                deep_serialization: bool = False) -> Dict[str, Any]:
        """Convert complete snapshot to dictionary for serialization and external analysis.
        
        Args:
            include_performance_data: Whether to include performance metrics
            deep_serialization: Whether to include full state details
            
        Returns:
            dict: Complete snapshot data in dictionary format for serialization
        """
        snapshot_dict = {
            'snapshot_id': self.snapshot_id,
            'timestamp': self.timestamp,
            'agent_position': self.agent_state.position.to_tuple(),
            'step_count': self.agent_state.step_count,
            'total_reward': self.agent_state.total_reward,
            'goal_reached': self.agent_state.goal_reached,
            'episode_terminated': self.episode_state.terminated,
            'episode_truncated': self.episode_state.truncated
        }
        
        # Include performance data if requested
        if include_performance_data:
            snapshot_dict['performance_metrics'] = self.performance_metrics.copy()
        
        # Include deep serialization if requested
        if deep_serialization:
            snapshot_dict['agent_state'] = self.agent_state.to_dict(
                include_history=True, include_performance_metrics=True
            )
            snapshot_dict['episode_state'] = self.episode_state.get_episode_summary(
                include_trajectory_analysis=True, include_performance_metrics=True
            )
        
        # Include validation results if available
        if self.validation_results:
            snapshot_dict['validation_results'] = self.validation_results.copy()
        
        return snapshot_dict


@dataclass
class PerformanceMetrics:
    """Data class for collecting and analyzing performance metrics including timing data, resource usage, 
    operation counts, and system performance statistics with trend analysis and optimization recommendations."""
    
    component_name: str
    start_time: float
    end_time: Optional[float] = None
    timing_data: Dict[str, List[float]] = field(default_factory=dict)
    operation_counts: Dict[str, int] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize performance metrics collection with component identification and timing setup."""
        if not self.component_name:
            raise ValidationError(
                "Component name must be non-empty string",
                parameter_name="component_name",
                invalid_value=self.component_name,
                expected_format="non-empty string"
            )
        
        # Initialize timing data structure
        self.timing_data.setdefault('operation_times', [])
        self.timing_data.setdefault('step_latencies', [])
        
        # Initialize operation counters
        self.operation_counts.setdefault('total_operations', 0)
        self.operation_counts.setdefault('successful_operations', 0)
        self.operation_counts.setdefault('failed_operations', 0)
    
    def record_timing(self, operation_name: str, duration_ms: float, 
                     operation_context: Optional[Dict[str, Any]] = None) -> None:
        """Record timing data for specific operations with performance analysis.
        
        Args:
            operation_name: Name of operation being timed
            duration_ms: Duration in milliseconds
            operation_context: Optional context information
        """
        if not isinstance(operation_name, str) or not operation_name:
            raise ValidationError(
                "Operation name must be non-empty string",
                parameter_name="operation_name",
                invalid_value=operation_name,
                expected_format="non-empty string"
            )
        
        if not isinstance(duration_ms, (int, float)) or duration_ms < 0:
            raise ValidationError(
                f"Duration must be non-negative number, got {duration_ms}",
                parameter_name="duration_ms",
                invalid_value=duration_ms,
                expected_format="non-negative float"
            )
        
        # Store timing data
        if operation_name not in self.timing_data:
            self.timing_data[operation_name] = []
        self.timing_data[operation_name].append(duration_ms)
        
        # Update operation counts
        self.operation_counts['total_operations'] += 1
        self.operation_counts.setdefault(f'{operation_name}_count', 0)
        self.operation_counts[f'{operation_name}_count'] += 1
        
        # Include operation context if provided
        if operation_context:
            context_key = f'{operation_name}_contexts'
            if context_key not in self.performance_summary:
                self.performance_summary[context_key] = []
            self.performance_summary[context_key].append(operation_context)
    
    def record_resource_usage(self, resource_type: str, usage_value: float, usage_unit: str) -> None:
        """Record resource usage data including memory, CPU, and system resources.
        
        Args:
            resource_type: Type of resource (memory, cpu, disk, etc.)
            usage_value: Resource usage value
            usage_unit: Unit of measurement
        """
        if not isinstance(resource_type, str) or not resource_type:
            raise ValidationError(
                "Resource type must be non-empty string",
                parameter_name="resource_type",
                invalid_value=resource_type,
                expected_format="non-empty string"
            )
        
        resource_entry = {
            'value': usage_value,
            'unit': usage_unit,
            'timestamp': time.time()
        }
        
        if resource_type not in self.resource_usage:
            self.resource_usage[resource_type] = []
        self.resource_usage[resource_type].append(resource_entry)
    
    def finalize_metrics(self, end_time: Optional[float] = None) -> Dict[str, Any]:
        """Finalize metrics collection and generate performance summary with analysis.
        
        Args:
            end_time: Optional end time, uses current time if None
            
        Returns:
            dict: Complete performance analysis with summary and recommendations
        """
        # Set end time
        if end_time is None:
            end_time = time.time()
        self.end_time = end_time
        
        # Calculate total duration
        total_duration = self.end_time - self.start_time
        
        # Generate timing statistics
        timing_stats = {}
        for operation, times in self.timing_data.items():
            if times:
                timing_stats[operation] = {
                    'count': len(times),
                    'total_ms': sum(times),
                    'average_ms': sum(times) / len(times),
                    'min_ms': min(times),
                    'max_ms': max(times)
                }
        
        # Generate resource usage summary
        resource_summary = {}
        for resource_type, usage_list in self.resource_usage.items():
            if usage_list:
                values = [entry['value'] for entry in usage_list]
                resource_summary[resource_type] = {
                    'current': usage_list[-1]['value'],
                    'unit': usage_list[-1]['unit'],
                    'peak': max(values),
                    'average': sum(values) / len(values)
                }
        
        # Compile complete analysis
        analysis = {
            'component_name': self.component_name,
            'total_duration_seconds': total_duration,
            'timing_statistics': timing_stats,
            'resource_summary': resource_summary,
            'operation_counts': self.operation_counts.copy(),
            'performance_recommendations': self._generate_recommendations(timing_stats, resource_summary)
        }
        
        # Store in performance summary
        self.performance_summary.update(analysis)
        
        return analysis
    
    def _generate_recommendations(self, timing_stats: Dict[str, Any], 
                                 resource_summary: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations based on metrics."""
        recommendations = []
        
        # Analyze timing performance
        for operation, stats in timing_stats.items():
            if stats['average_ms'] > 10.0:  # Operations taking >10ms
                recommendations.append(f"Optimize {operation} - average {stats['average_ms']:.1f}ms")
            
            if stats['max_ms'] > stats['average_ms'] * 3:  # High variance
                recommendations.append(f"Investigate {operation} performance variance")
        
        # Analyze resource usage
        for resource_type, summary in resource_summary.items():
            if 'memory' in resource_type.lower() and summary['peak'] > 100:  # >100MB
                recommendations.append(f"Monitor {resource_type} usage - peak {summary['peak']:.1f}{summary['unit']}")
        
        return recommendations
    
    def get_performance_summary(self, include_trends: bool = False, 
                               include_recommendations: bool = False) -> Dict[str, Any]:
        """Get current performance summary without finalizing metrics collection.
        
        Args:
            include_trends: Whether to include performance trend analysis
            include_recommendations: Whether to include optimization recommendations
            
        Returns:
            dict: Current performance summary with optional trends and recommendations
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        summary = {
            'component_name': self.component_name,
            'elapsed_time_seconds': elapsed_time,
            'total_operations': self.operation_counts.get('total_operations', 0),
            'current_timestamp': current_time
        }
        
        # Include basic timing info
        if self.timing_data:
            recent_operations = {}
            for operation, times in self.timing_data.items():
                if times:
                    recent_operations[operation] = {
                        'recent_average_ms': sum(times[-10:]) / min(len(times), 10),
                        'total_count': len(times)
                    }
            summary['recent_operations'] = recent_operations
        
        # Include trends if requested
        if include_trends:
            summary['trends'] = self._analyze_trends()
        
        # Include recommendations if requested
        if include_recommendations:
            summary['recommendations'] = self._generate_recommendations(
                {}, {}  # Simplified for current summary
            )
        
        return summary
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends from timing data."""
        trends = {}
        
        for operation, times in self.timing_data.items():
            if len(times) >= 10:  # Need sufficient data
                recent_avg = sum(times[-5:]) / 5
                older_avg = sum(times[-10:-5]) / 5
                
                trend_direction = 'improving' if recent_avg < older_avg else 'degrading'
                trend_magnitude = abs(recent_avg - older_avg) / older_avg
                
                trends[operation] = {
                    'direction': trend_direction,
                    'magnitude_percent': trend_magnitude * 100,
                    'recent_average': recent_avg,
                    'previous_average': older_avg
                }
        
        return trends
    
    def compare_with_baseline(self, baseline_metrics: 'PerformanceMetrics') -> Dict[str, Any]:
        """Compare current performance metrics with baseline for regression analysis.
        
        Args:
            baseline_metrics: Baseline performance metrics for comparison
            
        Returns:
            dict: Performance comparison with baseline including regression analysis
        """
        if not isinstance(baseline_metrics, PerformanceMetrics):
            raise ValidationError(
                f"Baseline must be PerformanceMetrics instance, got {type(baseline_metrics).__name__}",
                parameter_name="baseline_metrics",
                invalid_value=baseline_metrics,
                expected_format="PerformanceMetrics instance"
            )
        
        comparison = {
            'comparison_timestamp': time.time(),
            'baseline_component': baseline_metrics.component_name,
            'current_component': self.component_name,
            'timing_comparisons': {},
            'resource_comparisons': {},
            'performance_regression': False,
            'improvements': [],
            'regressions': []
        }
        
        # Compare timing data
        for operation in set(self.timing_data.keys()) | set(baseline_metrics.timing_data.keys()):
            current_times = self.timing_data.get(operation, [])
            baseline_times = baseline_metrics.timing_data.get(operation, [])
            
            if current_times and baseline_times:
                current_avg = sum(current_times) / len(current_times)
                baseline_avg = sum(baseline_times) / len(baseline_times)
                
                change_percent = ((current_avg - baseline_avg) / baseline_avg) * 100
                
                comparison['timing_comparisons'][operation] = {
                    'current_average_ms': current_avg,
                    'baseline_average_ms': baseline_avg,
                    'change_percent': change_percent,
                    'regression': change_percent > 20  # >20% slower is regression
                }
                
                if change_percent > 20:
                    comparison['regressions'].append(f"{operation}: {change_percent:.1f}% slower")
                    comparison['performance_regression'] = True
                elif change_percent < -10:  # >10% faster is improvement
                    comparison['improvements'].append(f"{operation}: {abs(change_percent):.1f}% faster")
        
        return comparison


# Type aliases for external interfaces and API compliance
ActionType = Union[Action, int]
ObservationType = np.ndarray
RewardType = float
InfoType = Dict[str, Any]
CoordinateTuple = Tuple[int, int]
GridSizeTuple = Tuple[int, int]
MovementVector = Tuple[int, int]
ConcentrationValue = float
DistanceValue = float
ValidationResult = Dict[str, Any]


# Factory functions for creating validated type instances

def create_coordinates(x_or_coords: Union[Tuple[int, int], List[int], Coordinates, int], 
                      y: Optional[int] = None, 
                      grid_bounds: Optional[GridSize] = None, 
                      validate_bounds: bool = True) -> Coordinates:
    """Factory function to create validated Coordinates from various input formats including tuples, 
    lists, existing Coordinates, and individual x,y values with comprehensive type validation and bounds checking.
    
    Args:
        x_or_coords: X coordinate or coordinate pair (tuple/list) or existing Coordinates
        y: Y coordinate if x_or_coords is single value
        grid_bounds: Optional grid bounds for validation
        validate_bounds: Whether to perform bounds checking
        
    Returns:
        Coordinates: Validated Coordinates object with proper bounds checking and type consistency
        
    Raises:
        ValidationError: If coordinate validation fails
    """
    # Handle different input formats
    if y is None:
        # Single input that should be tuple, list, or Coordinates
        if isinstance(x_or_coords, Coordinates):
            coords = x_or_coords
        elif isinstance(x_or_coords, (tuple, list)) and len(x_or_coords) == 2:
            coords = Coordinates(int(x_or_coords[0]), int(x_or_coords[1]))
        else:
            raise ValidationError(
                f"Invalid coordinate format: {x_or_coords}",
                parameter_name="x_or_coords",
                invalid_value=x_or_coords,
                expected_format="Coordinates, tuple[int, int], or list[int, int]"
            )
    else:
        # Separate x and y values
        coords = Coordinates(int(x_or_coords), int(y))
    
    # Perform bounds checking if requested
    if validate_bounds and grid_bounds is not None:
        if not coords.is_within_bounds(grid_bounds):
            raise ValidationError(
                f"Coordinates {coords.to_tuple()} outside grid bounds {grid_bounds.to_tuple()}",
                parameter_name="coordinates",
                invalid_value=coords.to_tuple(),
                expected_format=f"coordinates within bounds (0,0) to ({grid_bounds.width-1},{grid_bounds.height-1})"
            )
    
    return coords


def create_grid_size(width_or_size: Union[Tuple[int, int], List[int], GridSize, int], 
                    height: Optional[int] = None, 
                    validate_memory_limits: bool = True,
                    validate_performance_feasibility: bool = False) -> GridSize:
    """Factory function to create validated GridSize from various input formats including tuples, 
    lists, existing GridSize, and individual width,height values with memory estimation and feasibility validation.
    
    Args:
        width_or_size: Width or size pair (tuple/list) or existing GridSize
        height: Height if width_or_size is single value
        validate_memory_limits: Whether to validate memory usage
        validate_performance_feasibility: Whether to check performance feasibility
        
    Returns:
        GridSize: Validated GridSize object with memory estimation and performance feasibility validation
        
    Raises:
        ValidationError: If grid size validation fails
    """
    # Handle different input formats
    if height is None:
        # Single input that should be tuple, list, or GridSize
        if isinstance(width_or_size, GridSize):
            grid_size = width_or_size
        elif isinstance(width_or_size, (tuple, list)) and len(width_or_size) == 2:
            grid_size = GridSize(int(width_or_size[0]), int(width_or_size[1]))
        else:
            raise ValidationError(
                f"Invalid grid size format: {width_or_size}",
                parameter_name="width_or_size",
                invalid_value=width_or_size,
                expected_format="GridSize, tuple[int, int], or list[int, int]"
            )
    else:
        # Separate width and height values
        grid_size = GridSize(int(width_or_size), int(height))
    
    # Validate memory limits if requested
    if validate_memory_limits:
        memory_estimate = grid_size.estimate_memory_mb()
        if memory_estimate > MEMORY_LIMIT_PLUME_FIELD_MB:
            raise ValidationError(
                f"Grid size memory estimate {memory_estimate:.1f}MB exceeds limit {MEMORY_LIMIT_PLUME_FIELD_MB}MB",
                parameter_name="grid_size",
                invalid_value=grid_size.to_tuple(),
                expected_format=f"grid size with memory usage <= {MEMORY_LIMIT_PLUME_FIELD_MB}MB"
            )
    
    # Validate performance feasibility if requested
    if validate_performance_feasibility:
        if not grid_size.is_performance_feasible():
            raise ValidationError(
                f"Grid size {grid_size.to_tuple()} may not meet performance targets",
                parameter_name="grid_size",
                invalid_value=grid_size.to_tuple(),
                expected_format="grid size meeting performance requirements"
            )
    
    return grid_size


def create_agent_state(initial_position: Coordinates, 
                      grid_bounds: Optional[GridSize] = None, 
                      validate_position: bool = True,
                      enable_performance_tracking: bool = True) -> AgentState:
    """Factory function to create initialized AgentState with validated position, reset counters, 
    and proper initialization for episode start with performance tracking setup.
    
    Args:
        initial_position: Starting position coordinates
        grid_bounds: Optional grid bounds for position validation
        validate_position: Whether to validate position against bounds
        enable_performance_tracking: Whether to enable performance tracking
        
    Returns:
        AgentState: Initialized AgentState ready for episode execution with tracking and validation
        
    Raises:
        ValidationError: If position validation fails
    """
    # Validate initial position
    if not isinstance(initial_position, Coordinates):
        raise ValidationError(
            f"Initial position must be Coordinates instance, got {type(initial_position).__name__}",
            parameter_name="initial_position",
            invalid_value=initial_position,
            expected_format="Coordinates instance"
        )
    
    # Check position bounds if validation enabled
    if validate_position and grid_bounds is not None:
        if not initial_position.is_within_bounds(grid_bounds):
            raise ValidationError(
                f"Initial position {initial_position.to_tuple()} outside grid bounds",
                parameter_name="initial_position",
                invalid_value=initial_position.to_tuple(),
                expected_format=f"position within grid bounds {grid_bounds.to_tuple()}"
            )
    
    # Create agent state with validated position
    agent_state = AgentState(position=initial_position)
    
    # Configure performance tracking
    if enable_performance_tracking and PERFORMANCE_TRACKING_ENABLED:
        agent_state.performance_metrics['creation_timestamp'] = time.time()
        agent_state.performance_metrics['tracking_enabled'] = True
    
    return agent_state


def create_episode_state(agent_state: AgentState, 
                        terminated: bool = False, 
                        truncated: bool = False,
                        episode_id: Optional[str] = None,
                        enable_history_tracking: bool = False) -> EpisodeState:
    """Factory function to create initialized EpisodeState with AgentState, termination flags, 
    and episode tracking setup for environment episode management.
    
    Args:
        agent_state: Valid agent state for episode
        terminated: Initial termination status
        truncated: Initial truncation status  
        episode_id: Optional custom episode ID
        enable_history_tracking: Whether to enable state history tracking
        
    Returns:
        EpisodeState: Initialized EpisodeState ready for episode execution with tracking and state management
        
    Raises:
        ValidationError: If agent state or termination flags are invalid
    """
    # Validate agent state
    if not isinstance(agent_state, AgentState):
        raise ValidationError(
            f"Agent state must be AgentState instance, got {type(agent_state).__name__}",
            parameter_name="agent_state",
            invalid_value=agent_state,
            expected_format="AgentState instance"
        )
    
    # Create episode state with optional custom ID
    if episode_id is None:
        episode_id = str(uuid.uuid4())
    
    episode_state = EpisodeState(
        agent_state=agent_state,
        terminated=terminated,
        truncated=truncated,
        episode_id=episode_id
    )
    
    # Configure history tracking
    if enable_history_tracking:
        episode_state.episode_summary['history_tracking_enabled'] = True
        # Record initial state
        episode_state.record_state({'context': 'episode_initialization'})
    
    return episode_state


def create_plume_parameters(source_location: Union[Coordinates, Tuple[int, int]], 
                           sigma: float, 
                           grid_size: Optional[GridSize] = None,
                           validate_mathematical_consistency: bool = True) -> PlumeParameters:
    """Factory function to create validated PlumeParameters with source location, sigma value, 
    and mathematical consistency checking for Gaussian plume model configuration.
    
    Args:
        source_location: Source location as Coordinates or tuple
        sigma: Gaussian dispersion parameter
        grid_size: Optional grid size for compatibility validation
        validate_mathematical_consistency: Whether to validate mathematical consistency
        
    Returns:
        PlumeParameters: Validated PlumeParameters ready for Gaussian plume model initialization with mathematical consistency
        
    Raises:
        ValidationError: If plume parameters fail validation
    """
    # Convert source location to Coordinates if needed
    if isinstance(source_location, (tuple, list)):
        source_coords = create_coordinates(source_location, validate_bounds=False)
    elif isinstance(source_location, Coordinates):
        source_coords = source_location
    else:
        raise ValidationError(
            f"Source location must be Coordinates or tuple, got {type(source_location).__name__}",
            parameter_name="source_location",
            invalid_value=source_location,
            expected_format="Coordinates instance or tuple[int, int]"
        )
    
    # Create plume parameters
    plume_params = PlumeParameters(source_location=source_coords, sigma=sigma)
    
    # Validate with grid compatibility if provided
    if grid_size is not None:
        plume_params.validate(grid_size=grid_size, strict_validation=validate_mathematical_consistency)
    
    return plume_params


def create_environment_config(grid_size: Union[GridSize, Tuple[int, int]], 
                             source_location: Union[Coordinates, Tuple[int, int]], 
                             plume_params: Union[PlumeParameters, Dict[str, Any]], 
                             max_steps: int = DEFAULT_MAX_STEPS, 
                             goal_radius: float = DEFAULT_GOAL_RADIUS,
                             validate_consistency: bool = True) -> EnvironmentConfig:
    """Factory function to create comprehensive EnvironmentConfig with all environment parameters, 
    cross-parameter validation, and resource constraint checking for complete environment setup.
    
    Args:
        grid_size: Grid dimensions as GridSize or tuple
        source_location: Source location as Coordinates or tuple
        plume_params: Plume parameters as PlumeParameters or dict
        max_steps: Maximum episode steps
        goal_radius: Goal detection radius
        validate_consistency: Whether to perform consistency validation
        
    Returns:
        EnvironmentConfig: Complete validated environment configuration ready for environment initialization with consistency validation
        
    Raises:
        ConfigurationError: If configuration validation fails
    """
    # Convert grid size if needed
    if isinstance(grid_size, (tuple, list)):
        grid_size_obj = create_grid_size(grid_size)
    elif isinstance(grid_size, GridSize):
        grid_size_obj = grid_size
    else:
        raise ConfigurationError(
            f"Grid size must be GridSize or tuple, got {type(grid_size).__name__}",
            config_parameter="grid_size",
            invalid_value=grid_size
        )
    
    # Convert plume parameters if needed
    if isinstance(plume_params, dict):
        plume_params_obj = create_plume_parameters(
            plume_params.get('source_location', source_location),
            plume_params.get('sigma', DEFAULT_PLUME_SIGMA),
            grid_size=grid_size_obj
        )
    elif isinstance(plume_params, PlumeParameters):
        plume_params_obj = plume_params
    else:
        # Create from provided source_location
        plume_params_obj = create_plume_parameters(
            source_location, 
            DEFAULT_PLUME_SIGMA, 
            grid_size=grid_size_obj
        )
    
    # Create environment configuration
    config = EnvironmentConfig(
        grid_size=grid_size_obj,
        plume_params=plume_params_obj,
        max_steps=max_steps,
        goal_radius=goal_radius
    )
    
    # Perform consistency validation if requested
    if validate_consistency:
        config.validate(strict_mode=False, check_resource_constraints=True)
    
    return config


def create_step_info(agent_state: AgentState, 
                    source_location: Coordinates,
                    include_performance_metrics: bool = False,
                    additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Factory function to create standardized step information dictionary with agent position, 
    distance metrics, episode statistics, and optional performance data for Gymnasium step return.
    
    Args:
        agent_state: Current agent state
        source_location: Plume source location for distance calculation
        include_performance_metrics: Whether to include performance data
        additional_info: Optional additional info to include
        
    Returns:
        dict: Standardized step info dictionary with agent state, distance metrics, and optional performance data
    """
    # Calculate distance to source
    distance_to_source = calculate_euclidean_distance(agent_state.position, source_location)
    
    # Create base info dictionary
    info = {
        'agent_xy': agent_state.position.to_tuple(),
        'distance_to_source': distance_to_source,
        'step_count': agent_state.step_count,
        'total_reward': agent_state.total_reward,
        'goal_reached': agent_state.goal_reached
    }
    
    # Include performance metrics if requested
    if include_performance_metrics and agent_state.performance_metrics:
        info['performance_metrics'] = agent_state.performance_metrics.copy()
    
    # Include additional info if provided
    if additional_info:
        info.update(additional_info)
    
    return info


# Validation and utility functions

def validate_action(action: Union[Action, int], 
                   convert_to_enum: bool = False,
                   strict_type_checking: bool = False) -> Union[Action, int]:
    """Type-safe action validation ensuring action is valid Action enum or integer in discrete action space range 
    with comprehensive error reporting.
    
    Args:
        action: Action to validate (Action enum or integer)
        convert_to_enum: Whether to convert valid integer to Action enum
        strict_type_checking: Whether to enable strict type checking
        
    Returns:
        Union[Action, int]: Validated action in requested format (enum or integer) ready for environment processing
        
    Raises:
        ValidationError: If action validation fails
    """
    # Check if action is already Action enum
    if isinstance(action, Action):
        return action if not convert_to_enum else action
    
    # Validate integer action
    if isinstance(action, int):
        if not (0 <= action <= 3):
            raise ValidationError(
                f"Action {action} outside valid range [0, 3]",
                parameter_name="action",
                invalid_value=action,
                expected_format="integer in range [0, 3] for cardinal directions"
            )
        
        # Convert to enum if requested
        if convert_to_enum:
            return Action(action)
        else:
            return action
    
    # Handle strict type checking
    if strict_type_checking:
        raise ValidationError(
            f"Action must be Action enum or int in strict mode, got {type(action).__name__}",
            parameter_name="action",
            invalid_value=action,
            expected_format="Action enum or integer"
        )
    
    # Try to convert other types
    try:
        action_int = int(action)
        if 0 <= action_int <= 3:
            return Action(action_int) if convert_to_enum else action_int
        else:
            raise ValueError("Out of range")
    except (ValueError, TypeError):
        raise ValidationError(
            f"Cannot convert {action} to valid action",
            parameter_name="action",
            invalid_value=action,
            expected_format="Action enum or convertible to integer [0, 3]"
        )


def validate_coordinates(coordinates: Union[Coordinates, Tuple[int, int], List[int]], 
                        grid_bounds: Optional[GridSize] = None,
                        allow_negative: bool = False,
                        strict_bounds_checking: bool = False) -> bool:
    """Comprehensive coordinate validation with type checking, bounds validation, and grid compatibility 
    ensuring coordinate consistency across system components.
    
    Args:
        coordinates: Coordinates to validate
        grid_bounds: Optional grid bounds for validation
        allow_negative: Whether to allow negative coordinates
        strict_bounds_checking: Whether to enable strict bounds checking
        
    Returns:
        bool: True if coordinates are valid, raises ValidationError if invalid with detailed error context
        
    Raises:
        ValidationError: If coordinate validation fails
    """
    # Convert to Coordinates if needed
    if isinstance(coordinates, (tuple, list)):
        coords = create_coordinates(coordinates, validate_bounds=False)
    elif isinstance(coordinates, Coordinates):
        coords = coordinates
    else:
        raise ValidationError(
            f"Coordinates must be Coordinates, tuple, or list, got {type(coordinates).__name__}",
            parameter_name="coordinates",
            invalid_value=coordinates,
            expected_format="Coordinates instance, tuple[int, int], or list[int, int]"
        )
    
    # Check non-negative constraint
    if not allow_negative:
        if coords.x < 0 or coords.y < 0:
            raise ValidationError(
                f"Coordinates {coords.to_tuple()} contain negative values",
                parameter_name="coordinates",
                invalid_value=coords.to_tuple(),
                expected_format="non-negative integer coordinates"
            )
    
    # Check grid bounds if provided
    if grid_bounds is not None:
        if not coords.is_within_bounds(grid_bounds):
            raise ValidationError(
                f"Coordinates {coords.to_tuple()} outside grid bounds {grid_bounds.to_tuple()}",
                parameter_name="coordinates",
                invalid_value=coords.to_tuple(),
                expected_format=f"coordinates within bounds (0,0) to ({grid_bounds.width-1},{grid_bounds.height-1})"
            )
    
    return True


def validate_grid_size(grid_size: Union[GridSize, Tuple[int, int], List[int]], 
                      check_memory_limits: bool = True,
                      validate_performance_targets: bool = False,
                      system_constraints: Optional[Dict[str, Any]] = None) -> bool:
    """Grid size validation with dimension checking, memory estimation, and performance feasibility analysis 
    ensuring grid configuration is viable for system operation.
    
    Args:
        grid_size: Grid size to validate
        check_memory_limits: Whether to check memory constraints
        validate_performance_targets: Whether to validate performance feasibility
        system_constraints: Optional system-specific constraints
        
    Returns:
        bool: True if grid size is valid, raises ValidationError if invalid with resource constraint details
        
    Raises:
        ValidationError: If grid size validation fails
    """
    # Convert to GridSize if needed
    if isinstance(grid_size, (tuple, list)):
        grid = create_grid_size(grid_size, validate_memory_limits=False)
    elif isinstance(grid_size, GridSize):
        grid = grid_size
    else:
        raise ValidationError(
            f"Grid size must be GridSize, tuple, or list, got {type(grid_size).__name__}",
            parameter_name="grid_size",
            invalid_value=grid_size,
            expected_format="GridSize instance, tuple[int, int], or list[int, int]"
        )
    
    # Check memory limits if enabled
    if check_memory_limits:
        memory_estimate = grid.estimate_memory_mb()
        if memory_estimate > MEMORY_LIMIT_PLUME_FIELD_MB:
            raise ValidationError(
                f"Grid size memory estimate {memory_estimate:.1f}MB exceeds limit {MEMORY_LIMIT_PLUME_FIELD_MB}MB",
                parameter_name="grid_size",
                invalid_value=grid.to_tuple(),
                expected_format=f"grid size with memory usage <= {MEMORY_LIMIT_PLUME_FIELD_MB}MB"
            )
    
    # Check performance targets if enabled
    if validate_performance_targets:
        if not grid.is_performance_feasible():
            raise ValidationError(
                f"Grid size {grid.to_tuple()} may not meet performance targets",
                parameter_name="grid_size",
                invalid_value=grid.to_tuple(),
                expected_format="grid size meeting performance requirements"
            )
    
    # Check system constraints if provided
    if system_constraints:
        if 'max_total_cells' in system_constraints:
            if grid.total_cells() > system_constraints['max_total_cells']:
                raise ValidationError(
                    f"Grid total cells {grid.total_cells()} exceeds system constraint",
                    parameter_name="grid_size",
                    invalid_value=grid.to_tuple(),
                    expected_format=f"total cells <= {system_constraints['max_total_cells']}"
                )
    
    return True


def is_valid_action(action: Union[Action, int, Any]) -> bool:
    """Quick boolean check for action validity without raising exceptions, useful for conditional logic 
    and performance-critical validation paths.
    
    Args:
        action: Action to check for validity
        
    Returns:
        bool: True if action is valid Action enum or integer in range [0,3], False otherwise
    """
    try:
        # Check Action enum
        if isinstance(action, Action):
            return True
        
        # Check integer in valid range
        if isinstance(action, int):
            return 0 <= action <= 3
        
        # Try to convert and validate
        action_int = int(action)
        return 0 <= action_int <= 3
    
    except (ValueError, TypeError, AttributeError):
        return False


def convert_to_coordinates(input_value: Union[Coordinates, Tuple[int, int], List[int], Dict[str, int], Any], 
                          strict_conversion: bool = False,
                          validation_bounds: Optional[GridSize] = None) -> Optional[Coordinates]:
    """Flexible conversion utility to transform various input formats to Coordinates with comprehensive 
    type handling and validation.
    
    Args:
        input_value: Value to convert to Coordinates
        strict_conversion: Whether to raise errors on conversion failure
        validation_bounds: Optional bounds for coordinate validation
        
    Returns:
        Optional[Coordinates]: Coordinates object if conversion successful, None if conversion fails in non-strict mode
        
    Raises:
        ValidationError: If conversion fails and strict_conversion is True
    """
    try:
        # Already Coordinates
        if isinstance(input_value, Coordinates):
            if validation_bounds and not input_value.is_within_bounds(validation_bounds):
                if strict_conversion:
                    raise ValidationError(
                        f"Coordinates {input_value.to_tuple()} outside validation bounds",
                        parameter_name="input_value",
                        invalid_value=input_value.to_tuple(),
                        expected_format=f"coordinates within bounds {validation_bounds.to_tuple()}"
                    )
                return None
            return input_value
        
        # Tuple or list
        elif isinstance(input_value, (tuple, list)) and len(input_value) == 2:
            coords = Coordinates(int(input_value[0]), int(input_value[1]))
            if validation_bounds and not coords.is_within_bounds(validation_bounds):
                if strict_conversion:
                    raise ValidationError(
                        f"Converted coordinates {coords.to_tuple()} outside validation bounds",
                        parameter_name="input_value",
                        invalid_value=input_value,
                        expected_format=f"coordinates within bounds {validation_bounds.to_tuple()}"
                    )
                return None
            return coords
        
        # Dictionary with x, y keys
        elif isinstance(input_value, dict) and 'x' in input_value and 'y' in input_value:
            coords = Coordinates(int(input_value['x']), int(input_value['y']))
            if validation_bounds and not coords.is_within_bounds(validation_bounds):
                if strict_conversion:
                    raise ValidationError(
                        f"Converted coordinates {coords.to_tuple()} outside validation bounds",
                        parameter_name="input_value",
                        invalid_value=input_value,
                        expected_format=f"coordinates within bounds {validation_bounds.to_tuple()}"
                    )
                return None
            return coords
        
        # String parsing (basic format like "(x,y)")
        elif isinstance(input_value, str):
            # Remove parentheses and split
            cleaned = input_value.strip('()[]').replace(' ', '')
            parts = cleaned.split(',')
            if len(parts) == 2:
                coords = Coordinates(int(parts[0]), int(parts[1]))
                if validation_bounds and not coords.is_within_bounds(validation_bounds):
                    if strict_conversion:
                        raise ValidationError(
                            f"Parsed coordinates {coords.to_tuple()} outside validation bounds",
                            parameter_name="input_value",
                            invalid_value=input_value,
                            expected_format=f"coordinates within bounds {validation_bounds.to_tuple()}"
                        )
                    return None
                return coords
        
        # Failed conversion
        if strict_conversion:
            raise ValidationError(
                f"Cannot convert {input_value} to Coordinates",
                parameter_name="input_value",
                invalid_value=input_value,
                expected_format="Coordinates, tuple[int, int], list[int, int], dict with x/y keys, or string '(x,y)'"
            )
        return None
        
    except (ValueError, TypeError, KeyError, IndexError) as e:
        if strict_conversion:
            raise ValidationError(
                f"Conversion error for {input_value}: {e}",
                parameter_name="input_value",
                invalid_value=input_value,
                expected_format="Valid coordinate format"
            )
        return None


def convert_to_grid_size(input_value: Union[GridSize, Tuple[int, int], List[int], Dict[str, int], Any], 
                        strict_conversion: bool = False,
                        validate_resources: bool = False) -> Optional[GridSize]:
    """Flexible conversion utility to transform various input formats to GridSize with comprehensive 
    type handling and resource validation.
    
    Args:
        input_value: Value to convert to GridSize
        strict_conversion: Whether to raise errors on conversion failure
        validate_resources: Whether to validate resource constraints
        
    Returns:
        Optional[GridSize]: GridSize object if conversion successful, None if conversion fails in non-strict mode
        
    Raises:
        ValidationError: If conversion fails and strict_conversion is True
    """
    try:
        # Already GridSize
        if isinstance(input_value, GridSize):
            if validate_resources and not input_value.is_performance_feasible():
                if strict_conversion:
                    raise ValidationError(
                        f"GridSize {input_value.to_tuple()} fails resource validation",
                        parameter_name="input_value",
                        invalid_value=input_value.to_tuple(),
                        expected_format="grid size meeting resource constraints"
                    )
                return None
            return input_value
        
        # Tuple or list
        elif isinstance(input_value, (tuple, list)) and len(input_value) == 2:
            grid_size = GridSize(int(input_value[0]), int(input_value[1]))
            if validate_resources and not grid_size.is_performance_feasible():
                if strict_conversion:
                    raise ValidationError(
                        f"Converted GridSize {grid_size.to_tuple()} fails resource validation",
                        parameter_name="input_value",
                        invalid_value=input_value,
                        expected_format="grid size meeting resource constraints"
                    )
                return None
            return grid_size
        
        # Dictionary with width, height keys
        elif isinstance(input_value, dict) and 'width' in input_value and 'height' in input_value:
            grid_size = GridSize(int(input_value['width']), int(input_value['height']))
            if validate_resources and not grid_size.is_performance_feasible():
                if strict_conversion:
                    raise ValidationError(
                        f"Converted GridSize {grid_size.to_tuple()} fails resource validation",
                        parameter_name="input_value",
                        invalid_value=input_value,
                        expected_format="grid size meeting resource constraints"
                    )
                return None
            return grid_size
        
        # String parsing (format like "WxH")
        elif isinstance(input_value, str):
            if 'x' in input_value:
                parts = input_value.lower().split('x')
                if len(parts) == 2:
                    grid_size = GridSize(int(parts[0]), int(parts[1]))
                    if validate_resources and not grid_size.is_performance_feasible():
                        if strict_conversion:
                            raise ValidationError(
                                f"Parsed GridSize {grid_size.to_tuple()} fails resource validation",
                                parameter_name="input_value",
                                invalid_value=input_value,
                                expected_format="grid size meeting resource constraints"
                            )
                        return None
                    return grid_size
        
        # Failed conversion
        if strict_conversion:
            raise ValidationError(
                f"Cannot convert {input_value} to GridSize",
                parameter_name="input_value",
                invalid_value=input_value,
                expected_format="GridSize, tuple[int, int], list[int, int], dict with width/height keys, or string 'WxH'"
            )
        return None
        
    except (ValueError, TypeError, KeyError, IndexError) as e:
        if strict_conversion:
            raise ValidationError(
                f"Conversion error for {input_value}: {e}",
                parameter_name="input_value",
                invalid_value=input_value,
                expected_format="Valid grid size format"
            )
        return None


def calculate_euclidean_distance(point1: Coordinates, point2: Coordinates, high_precision: bool = False) -> float:
    """Optimized Euclidean distance calculation between two coordinate points with numerical precision handling 
    and performance optimization for frequent distance calculations.
    
    Args:
        point1: First coordinate point
        point2: Second coordinate point  
        high_precision: Whether to use high precision calculation
        
    Returns:
        float: Euclidean distance between the two points with appropriate precision
        
    Raises:
        ValidationError: If coordinates are invalid
    """
    # Validate coordinate inputs
    if not isinstance(point1, Coordinates) or not isinstance(point2, Coordinates):
        raise ValidationError(
            f"Distance calculation requires Coordinates instances, got {type(point1).__name__}, {type(point2).__name__}",
            parameter_name="coordinates",
            invalid_value=(point1, point2),
            expected_format="(Coordinates, Coordinates)"
        )
    
    # Calculate coordinate differences
    dx = point2.x - point1.x
    dy = point2.y - point1.y
    
    # Handle identical points (distance = 0)
    if dx == 0 and dy == 0:
        return 0.0
    
    # Calculate distance squared
    distance_squared = dx * dx + dy * dy
    
    # Apply high precision calculation if requested
    if high_precision:
        # Use higher precision for small distances
        if distance_squared < DISTANCE_CALCULATION_EPSILON:
            return 0.0
        # Use math.sqrt with additional precision handling
        distance = math.sqrt(float(distance_squared))
    else:
        # Standard precision calculation
        distance = math.sqrt(distance_squared)
    
    return distance


def get_movement_vector(action: Union[Action, int]) -> Tuple[int, int]:
    """Retrieve movement vector (dx, dy) for given action using MOVEMENT_VECTORS lookup with validation 
    and coordinate system consistency.
    
    Args:
        action: Action enum or integer to get movement vector for
        
    Returns:
        tuple[int, int]: Movement vector (dx, dy) for coordinate arithmetic and position updates
        
    Raises:
        ValidationError: If action is invalid or not found in MOVEMENT_VECTORS
    """
    # Convert Action enum to integer if needed
    if isinstance(action, Action):
        action_int = action.value
    elif isinstance(action, int):
        action_int = action
    else:
        raise ValidationError(
            f"Action must be Action enum or int, got {type(action).__name__}",
            parameter_name="action",
            invalid_value=action,
            expected_format="Action enum or integer"
        )
    
    # Validate action is in valid range
    if not (0 <= action_int <= 3):
        raise ValidationError(
            f"Action {action_int} outside valid range [0, 3]",
            parameter_name="action",
            invalid_value=action_int,
            expected_format="integer in range [0, 3]"
        )
    
    # Lookup movement vector
    try:
        return MOVEMENT_VECTORS[action_int]
    except KeyError:
        raise ValidationError(
            f"Action {action_int} not found in movement vectors",
            parameter_name="action",
            invalid_value=action_int,
            expected_format="valid action with corresponding movement vector"
        )


def get_action_from_vector(movement_vector: Tuple[int, int], return_enum: bool = True) -> Union[Action, int, None]:
    """Reverse lookup to find Action from movement vector (dx, dy) for movement analysis and trajectory processing 
    with comprehensive vector matching.
    
    Args:
        movement_vector: Movement delta vector (dx, dy)
        return_enum: Whether to return Action enum or integer
        
    Returns:
        Union[Action, int, None]: Action enum or integer corresponding to movement vector, None if vector not found
    """
    # Validate movement vector format
    if not isinstance(movement_vector, tuple) or len(movement_vector) != 2:
        return None
    
    try:
        dx, dy = movement_vector
        # Ensure components are integers
        dx, dy = int(dx), int(dy)
        vector_tuple = (dx, dy)
    except (ValueError, TypeError):
        return None
    
    # Search through MOVEMENT_VECTORS for matching vector
    for action_int, vector in MOVEMENT_VECTORS.items():
        if vector == vector_tuple:
            # Return Action enum or integer based on return_enum parameter
            if return_enum:
                try:
                    return Action(action_int)
                except ValueError:
                    return None
            else:
                return action_int
    
    # Vector not found in movement vectors
    return None


# Export comprehensive public interface
__all__ = [
    # Enumerations
    'Action', 'RenderMode',
    
    # Type aliases for external interfaces
    'ActionType', 'ObservationType', 'RewardType', 'InfoType', 
    'CoordinateTuple', 'GridSizeTuple', 'MovementVector', 
    'ConcentrationValue', 'DistanceValue', 'ValidationResult',
    
    # Core data classes
    'Coordinates', 'GridSize', 'AgentState', 'EpisodeState', 
    'PlumeParameters', 'EnvironmentConfig', 'StateSnapshot', 'PerformanceMetrics',
    
    # Factory functions for type creation
    'create_coordinates', 'create_grid_size', 'create_agent_state', 'create_episode_state',
    'create_plume_parameters', 'create_environment_config', 'create_step_info',
    
    # Validation and utility functions
    'validate_action', 'validate_coordinates', 'validate_grid_size', 'is_valid_action',
    'convert_to_coordinates', 'convert_to_grid_size', 
    'calculate_euclidean_distance', 'get_movement_vector', 'get_action_from_vector'
]