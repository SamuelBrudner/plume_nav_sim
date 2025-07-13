# Action Interface API Reference

This document provides comprehensive API reference for the standardized action processing interfaces in plume_nav_sim v1.0, supporting unified action handling across different RL frameworks and navigation strategies.

## Table of Contents

1. [Action System Overview](#action-system-overview)
2. [ActionInterfaceProtocol](#actioninterface-protocol)
3. [Continuous2DAction Reference](#continuous2d-action-reference)
4. [CardinalDiscreteAction Reference](#cardinal-discrete-action-reference)
5. [Configuration Examples](#configuration-examples)
6. [Integration Patterns](#integration-patterns)
7. [Performance Specifications](#performance-specifications)

## Action System Overview

The action interface layer provides standardized action processing for unified RL framework integration, enabling seamless switching between continuous and discrete control paradigms through configuration-driven component instantiation.

### Protocol-Based Architecture

The v1.0 action system implements a protocol-based architecture that ensures consistent action handling while maintaining type safety and framework compatibility:

```python
from plume_nav_sim.core.actions import create_action_interface
from plume_nav_sim.core.protocols import ActionInterfaceProtocol

# Configuration-driven instantiation
config = {
    'type': 'Continuous2D',
    'max_velocity': 2.0,
    'max_angular_velocity': 45.0
}
action_interface = create_action_interface(config)

# Uniform API across implementations
action = np.array([1.5, 15.0])  # RL framework action
nav_command = action_interface.translate_action(action)
action_space = action_interface.get_action_space()
```

### Unified Action Handling

All action interfaces implement the `ActionInterfaceProtocol`, providing consistent methods for:

- **Action Translation**: Convert RL framework actions to navigation commands
- **Action Validation**: Enforce bounds and constraints with configurable error handling
- **Action Space Definition**: Generate Gymnasium-compatible action spaces
- **Performance Optimization**: Achieve <0.05ms translation overhead per step

### Continuous and Discrete Support

The system supports both continuous and discrete control paradigms:

- **Continuous2DAction**: Smooth velocity-based control for continuous RL algorithms (PPO, SAC, TD3)
- **CardinalDiscreteAction**: Grid-based directional movement for discrete RL algorithms (DQN, A2C)

### Gymnasium Integration

Full compatibility with Gymnasium 0.29.x action spaces:

```python
import gymnasium as gym
from gymnasium import spaces

# Automatic action space generation
action_space = action_interface.get_action_space()
assert isinstance(action_space, spaces.Space)

# Framework integration
env_action_space = action_space
sample_action = env_action_space.sample()
```

### Performance Specifications

All action interfaces meet strict performance requirements:

- **Translation Overhead**: <0.05ms per agent per step
- **Validation Time**: <0.02ms per action for constraint checking  
- **Memory Usage**: <200 bytes per instance for lightweight operation
- **Vectorized Support**: Batch processing for multi-agent scenarios

## ActionInterface Protocol

The `ActionInterfaceProtocol` defines the contract for all action interface implementations, ensuring consistent API across different control modalities.

### Protocol Specification

```python
from typing import Protocol, Union, Optional, Dict, Any
import numpy as np
from gymnasium import spaces

@runtime_checkable
class ActionInterfaceProtocol(Protocol):
    """Protocol defining standardized action processing interface."""
    
    def translate_action(
        self, 
        action: Union[np.ndarray, int, float, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Translate RL framework action to navigation controller commands."""
        ...
    
    def validate_action(
        self, 
        action: Union[np.ndarray, int, float, Dict[str, Any]]
    ) -> bool:
        """Validate action compliance with interface constraints."""
        ...
    
    def get_action_space(self) -> Optional[spaces.Space]:
        """Construct Gymnasium action space definition."""
        ...
```

### Method Signatures

#### translate_action()

Converts RL framework actions to navigation commands with standardized output format:

**Parameters:**
- `action`: RL action in framework-specific format
  - `np.ndarray`: Continuous control vectors
  - `int`: Discrete action indices  
  - `float`: Scalar actions for 1D control
  - `Dict[str, Any]`: Structured actions for complex control schemes

**Returns:**
- `Dict[str, Any]`: Navigation command dictionary with keys:
  - `'linear_velocity'`: Target linear velocity (float)
  - `'angular_velocity'`: Target angular velocity (float) 
  - `'action_type'`: Control scheme identifier (str)

**Performance**: <0.1ms per agent execution time

#### validate_action()

Validates and constrains actions to valid ranges with configurable error handling:

**Parameters:**
- `action`: Action to validate in same format as translate_action()

**Returns:**
- `bool`: True if action is valid and safe to execute

**Features:**
- Bounds checking for continuous actions
- Index validation for discrete actions
- Safety constraints (maximum velocities, acceleration limits)
- Type checking and format validation

**Performance**: <0.05ms per action execution time

#### get_action_space()

Constructs Gymnasium action space definitions for RL framework integration:

**Returns:**
- `Optional[spaces.Space]`: Gymnasium action space or None if unavailable
  - `spaces.Box`: Continuous control with bounded ranges
  - `spaces.Discrete`: Discrete action indices
  - `spaces.Dict`: Structured action dictionaries

**Features:**
- Automatic space reflection of interface configuration
- Consistency with translate_action() and validate_action() methods
- Type safety for RL framework integration

### Implementation Requirements

Custom action interfaces must:

1. **Implement all protocol methods** with correct signatures and behavior
2. **Maintain performance requirements** for real-time operation
3. **Provide consistent output formats** for navigation command integration
4. **Support vectorized operations** for multi-agent scenarios
5. **Handle edge cases gracefully** with appropriate error handling

### Compliance Guidelines

- Use `isinstance(obj, ActionInterfaceProtocol)` for runtime type checking
- Implement protocol methods with exact signatures and return types
- Follow naming conventions for navigation command keys
- Provide comprehensive docstrings for all public methods
- Include usage examples in class documentation

### Type Annotations

Full type safety with static type checking support:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type checker validates protocol compliance
    action_interface: ActionInterfaceProtocol = create_action_interface(config)
    
# Runtime protocol checking
assert isinstance(action_interface, ActionInterfaceProtocol)
```

## Continuous2D Action Reference

The `Continuous2DAction` class provides continuous 2D navigation control with velocity commands for smooth movement and real-time performance optimization.

### Class API Reference

```python
class Continuous2DAction(ActionInterfaceProtocol):
    """Continuous 2D action interface for velocity-based navigation control."""
    
    def __init__(
        self,
        max_velocity: float = 2.0,
        max_angular_velocity: float = 45.0,
        min_velocity: float = -2.0,
        min_angular_velocity: float = -45.0
    ):
        """Initialize continuous 2D action interface."""
```

### Constructor Parameters

- **max_velocity** (`float`, default: 2.0): Maximum linear velocity in units per time step
- **max_angular_velocity** (`float`, default: 45.0): Maximum angular velocity in degrees per second
- **min_velocity** (`float`, default: -2.0): Minimum linear velocity (negative for backward movement)
- **min_angular_velocity** (`float`, default: -45.0): Minimum angular velocity (negative for counter-rotation)

### Core Methods

#### translate_action(action: np.ndarray) → Dict[str, Any]

Translates RL action arrays to navigation commands with automatic validation and bounds checking.

**Parameters:**
- `action` (`np.ndarray`): Action array with shapes:
  - `(2,)`: Standard 2D action `[linear_velocity, angular_velocity]`
  - `(1,)`: Linear velocity only, angular velocity = 0.0
  - `()`: Scalar linear velocity, angular velocity = 0.0

**Returns:**
```python
{
    'linear_velocity': float,      # Validated linear velocity
    'angular_velocity': float,     # Validated angular velocity
    'action_type': 'continuous_2d' # Interface identifier
}
```

**Example:**
```python
action_interface = Continuous2DAction(max_velocity=2.0, max_angular_velocity=45.0)

# Standard 2D control
action = np.array([1.5, 20.0])
command = action_interface.translate_action(action)
# Returns: {'linear_velocity': 1.5, 'angular_velocity': 20.0, 'action_type': 'continuous_2d'}

# Linear-only control
action = np.array([1.0])
command = action_interface.translate_action(action)
# Returns: {'linear_velocity': 1.0, 'angular_velocity': 0.0, 'action_type': 'continuous_2d'}
```

#### validate_action(action: np.ndarray) → np.ndarray

Validates and constrains action values within configured bounds with NaN/Inf handling.

**Parameters:**
- `action` (`np.ndarray`): Action array to validate

**Returns:**
- `np.ndarray`: Validated action with bounds applied and invalid values replaced

**Features:**
- **Bounds Clipping**: Uses `numpy.clip` for efficient constraint enforcement
- **Invalid Value Handling**: Replaces NaN/Inf with zero
- **Shape Normalization**: Handles scalar, 1D, and 2D action inputs
- **Type Safety**: Returns float32 arrays for memory efficiency

**Example:**
```python
# Clip action to bounds
action = np.array([3.0, 60.0])  # Exceeds max bounds
valid_action = action_interface.validate_action(action)
# Returns: [2.0, 45.0] (clipped to configured maximums)

# Handle invalid values
action = np.array([np.nan, np.inf])
valid_action = action_interface.validate_action(action)
# Returns: [0.0, 0.0] (replaced with safe values)
```

#### get_action_space() → Optional[spaces.Box]

Constructs Gymnasium Box action space with configured velocity bounds for RL framework integration.

**Returns:**
- `Optional[spaces.Box]`: Action space with shape `(2,)` and velocity bounds, or None if Gymnasium unavailable

**Features:**
- **Dynamic Bounds**: Reflects current min/max velocity configuration
- **Memory Efficiency**: Uses float32 dtype for RL compatibility
- **Framework Integration**: Compatible with all major RL libraries

**Example:**
```python
action_space = action_interface.get_action_space()
assert action_space.shape == (2,)
assert action_space.low[0] == -2.0   # min_velocity
assert action_space.high[0] == 2.0   # max_velocity
assert action_space.low[1] == -45.0  # min_angular_velocity
assert action_space.high[1] == 45.0  # max_angular_velocity
```

### Configuration Methods

#### set_bounds(max_velocity=None, max_angular_velocity=None, min_velocity=None, min_angular_velocity=None)

Updates velocity bounds dynamically with validation.

**Parameters:**
- All parameters are optional floats for selective bound updates
- Bounds are validated to ensure min < max relationships

**Example:**
```python
# Update maximum velocities only
action_interface.set_bounds(max_velocity=3.0, max_angular_velocity=60.0)

# Update all bounds
action_interface.set_bounds(
    min_velocity=-1.0, max_velocity=3.0,
    min_angular_velocity=-30.0, max_angular_velocity=60.0
)
```

#### get_max_velocity() → float
#### get_max_angular_velocity() → float

Access current bound configuration for dynamic parameter adjustment.

### Performance Optimization

#### Vectorized Operations

Built-in support for efficient multi-agent scenarios:

```python
# Single agent
action = np.array([1.5, 20.0])
command = action_interface.translate_action(action)

# Multi-agent (implemented by environment wrapper)
actions = np.array([[1.5, 20.0], [1.0, -10.0], [0.5, 5.0]])
commands = [action_interface.translate_action(a) for a in actions]
```

#### Memory Efficiency

- **Float32 Operations**: Consistent with RL framework requirements
- **Minimal Allocations**: Reuses arrays where possible
- **Bounds Caching**: Pre-computed bounds arrays for validation

#### Real-Time Performance

Optimized for <0.05ms translation overhead:

```python
import time

# Performance validation
start = time.perf_counter()
command = action_interface.translate_action(action)
duration = time.perf_counter() - start
assert duration < 0.00005  # <0.05ms requirement
```

### Usage Patterns

#### RL Framework Integration

```python
import stable_baselines3 as sb3
from stable_baselines3 import PPO

# Create environment with continuous action interface
env = PlumeNavigationEnv(action_interface=Continuous2DAction())

# Train RL agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Test trained agent
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
```

#### Custom Control Logic

```python
class CustomContinuousAction(Continuous2DAction):
    """Custom action interface with PID control."""
    
    def translate_action(self, action: np.ndarray) -> Dict[str, Any]:
        # Apply PID control logic
        pid_output = self.pid_controller.update(action)
        
        # Call parent translation with PID-adjusted action
        return super().translate_action(pid_output)
```

#### Configuration-Driven Setup

```python
# From configuration file
config = {
    'type': 'Continuous2D',
    'max_velocity': 2.5,
    'max_angular_velocity': 60.0,
    'min_velocity': -1.0,
    'min_angular_velocity': -30.0
}

action_interface = create_action_interface(config)
```

### Bounds Validation

The interface enforces strict bounds validation to ensure safe operation:

```python
# Configuration validation
try:
    action_interface = Continuous2DAction(
        min_velocity=2.0,    # Invalid: min >= max
        max_velocity=1.0
    )
except ValueError as e:
    print(f"Configuration error: {e}")

# Runtime action validation
action = np.array([5.0, 100.0])  # Exceeds bounds
safe_action = action_interface.validate_action(action)
# Automatically clipped to [2.0, 45.0]
```

## Cardinal Discrete Action Reference

The `CardinalDiscreteAction` class provides discrete directional movement with cardinal and intercardinal directions for grid-based navigation and interpretable action commands.

### Class API Reference

```python
class CardinalDiscreteAction(ActionInterfaceProtocol):
    """Cardinal discrete action interface for directional movement control."""
    
    def __init__(
        self,
        speed: float = 1.0,
        use_8_directions: bool = True,
        include_stay_action: bool = True
    ):
        """Initialize cardinal discrete action interface."""
```

### Constructor Parameters

- **speed** (`float`, default: 1.0): Movement speed for all directions in units per time step
- **use_8_directions** (`bool`, default: True): Enable 8-direction mode (N,S,E,W,NE,NW,SE,SW) vs 4-direction mode (N,S,E,W)
- **include_stay_action** (`bool`, default: True): Include stay-in-place action (action index 0)

### Action Space Configuration

The action space size depends on configuration parameters:

| Configuration | Actions | Total | Action Indices |
|---------------|---------|-------|----------------|
| 4-direction + stay | N, S, E, W, STAY | 5 | 0-4 |
| 4-direction no stay | N, S, E, W | 4 | 0-3 |
| 8-direction + stay | N, S, E, W, NE, NW, SE, SW, STAY | 9 | 0-8 |
| 8-direction no stay | N, S, E, W, NE, NW, SE, SW | 8 | 0-7 |

### Direction Mapping

#### Standard Action Mapping (8-direction + stay)

```python
action_mapping = {
    0: "STAY",       # No movement
    1: "NORTH",      # Negative Y (up)
    2: "SOUTH",      # Positive Y (down)  
    3: "EAST",       # Positive X (right)
    4: "WEST",       # Negative X (left)
    5: "NORTHEAST",  # Diagonal: +X, -Y
    6: "NORTHWEST",  # Diagonal: -X, -Y
    7: "SOUTHEAST",  # Diagonal: +X, +Y
    8: "SOUTHWEST"   # Diagonal: -X, +Y
}
```

### Core Methods

#### translate_action(action: Union[int, np.ndarray]) → Dict[str, Any]

Translates discrete action indices to navigation commands with direction vectors and metadata.

**Parameters:**
- `action` (`Union[int, np.ndarray]`): Discrete action index or array containing single index

**Returns:**
```python
{
    'linear_velocity': float,        # Computed linear velocity magnitude
    'angular_velocity': float,       # Angular velocity (always 0.0 for discrete)
    'velocity_x': float,            # X-component of velocity vector
    'velocity_y': float,            # Y-component of velocity vector  
    'direction': str,               # Direction name (e.g., 'NORTH', 'NORTHEAST')
    'action_type': 'cardinal_discrete'  # Interface identifier
}
```

**Features:**
- **Diagonal Normalization**: Diagonal movements scaled by 1/√2 for consistent step distance
- **Coordinate System**: Standard navigation coordinates (positive X = east, negative Y = north)
- **Zero Angular Velocity**: Discrete actions provide pure translation

**Example:**
```python
action_interface = CardinalDiscreteAction(speed=1.0, use_8_directions=True)

# Cardinal direction
action = 1  # North
command = action_interface.translate_action(action)
# Returns: {
#     'linear_velocity': 1.0, 'angular_velocity': 0.0,
#     'velocity_x': 0.0, 'velocity_y': -1.0, 
#     'direction': 'NORTH', 'action_type': 'cardinal_discrete'
# }

# Diagonal direction (8-direction mode)
action = 5  # Northeast
command = action_interface.translate_action(action)
# Returns: {
#     'linear_velocity': 0.707, 'angular_velocity': 0.0,
#     'velocity_x': 0.707, 'velocity_y': -0.707,
#     'direction': 'NORTHEAST', 'action_type': 'cardinal_discrete'
# }
```

#### validate_action(action: Union[int, np.ndarray]) → Union[int, np.ndarray]

Validates and constrains discrete actions to valid index ranges with clipping.

**Parameters:**
- `action` (`Union[int, np.ndarray]`): Discrete action to validate

**Returns:**
- `Union[int, np.ndarray]`: Validated action clipped to valid range [0, num_actions-1]

**Features:**
- **Index Clipping**: Invalid actions clipped to nearest valid index
- **Array Support**: Handles both scalar and array inputs
- **Type Safety**: Returns int32 for discrete compatibility

**Example:**
```python
# Valid action
action = 3  # East
valid_action = action_interface.validate_action(action)
# Returns: 3 (no change)

# Invalid action (too high)
action = 15  # Invalid index
valid_action = action_interface.validate_action(action)
# Returns: 8 (clipped to max action index)

# Invalid action (negative)
action = -1  # Invalid index
valid_action = action_interface.validate_action(action)
# Returns: 0 (clipped to minimum)
```

#### get_action_space() → Optional[spaces.Discrete]

Constructs Gymnasium Discrete action space with appropriate action count for RL framework integration.

**Returns:**
- `Optional[spaces.Discrete]`: Discrete action space with `n` actions, or None if Gymnasium unavailable

**Example:**
```python
# 8-direction + stay configuration
action_interface = CardinalDiscreteAction(use_8_directions=True, include_stay_action=True)
action_space = action_interface.get_action_space()
assert isinstance(action_space, spaces.Discrete)
assert action_space.n == 9  # 8 directions + stay

# 4-direction only configuration  
action_interface = CardinalDiscreteAction(use_8_directions=False, include_stay_action=False)
action_space = action_interface.get_action_space()
assert action_space.n == 4  # 4 cardinal directions only
```

### Configuration Methods

#### set_speed(new_speed: float)

Updates movement speed and rebuilds internal action mapping with validation.

**Parameters:**
- `new_speed` (`float`): New movement speed for all directions (must be positive)

**Features:**
- **Dynamic Speed Adjustment**: Updates all direction vectors while preserving relationships
- **Diagonal Preservation**: Maintains diagonal normalization with new speed
- **Validation**: Ensures positive speed values

**Example:**
```python
# Initial speed
action_interface = CardinalDiscreteAction(speed=1.0)

# Increase movement speed
action_interface.set_speed(2.0)

# Test new speed
action = 1  # North
command = action_interface.translate_action(action)
assert command['linear_velocity'] == 2.0  # Updated speed
```

#### get_action_mapping() → Dict[int, str]

Returns human-readable mapping from action indices to direction names for debugging and visualization.

**Returns:**
- `Dict[int, str]`: Mapping from action index to direction name

**Example:**
```python
mapping = action_interface.get_action_mapping()
print(f"Action 0: {mapping[0]}")  # "STAY"
print(f"Action 1: {mapping[1]}")  # "NORTH"
print(f"Action 5: {mapping[5]}")  # "NORTHEAST" (if 8-direction enabled)
```

### Query Methods

#### get_available_actions() → List[int]
#### get_speed() → float  
#### get_num_actions() → int
#### get_direction_for_action(action: int) → str

Utility methods for action space introspection and configuration access.

**Example:**
```python
# Available action indices
actions = action_interface.get_available_actions()
print(f"Available actions: {actions}")  # [0, 1, 2, 3, 4, 5, 6, 7, 8]

# Current configuration
speed = action_interface.get_speed()
num_actions = action_interface.get_num_actions()
print(f"Speed: {speed}, Actions: {num_actions}")

# Direction lookup
direction = action_interface.get_direction_for_action(1)
print(f"Action 1: {direction}")  # "NORTH"
```

### Grid Navigation Patterns

#### Basic Grid Movement

```python
# Setup for grid navigation
action_interface = CardinalDiscreteAction(
    speed=1.0,                # One grid cell per action
    use_8_directions=False,   # Only cardinal directions
    include_stay_action=True  # Allow staying in place
)

# Grid-based agent policy
def grid_policy(agent_pos, target_pos):
    dx = target_pos[0] - agent_pos[0]
    dy = target_pos[1] - agent_pos[1]
    
    if abs(dx) > abs(dy):
        return 3 if dx > 0 else 4  # East or West
    elif dy != 0:
        return 2 if dy > 0 else 1  # South or North
    else:
        return 0  # Stay
```

#### Diagonal Movement Optimization

```python
# Enable diagonal movement for efficiency
action_interface = CardinalDiscreteAction(
    speed=1.0,
    use_8_directions=True,   # Enable diagonals
    include_stay_action=True
)

# Optimal pathfinding policy
def diagonal_policy(agent_pos, target_pos):
    dx = target_pos[0] - agent_pos[0]
    dy = target_pos[1] - agent_pos[1]
    
    # Diagonal movement when possible
    if dx > 0 and dy < 0:
        return 5  # Northeast
    elif dx < 0 and dy < 0:
        return 6  # Northwest
    elif dx > 0 and dy > 0:
        return 7  # Southeast
    elif dx < 0 and dy > 0:
        return 8  # Southwest
    # Cardinal movement for remaining cases
    elif dx > 0:
        return 3  # East
    elif dx < 0:
        return 4  # West
    elif dy < 0:
        return 1  # North
    elif dy > 0:
        return 2  # South
    else:
        return 0  # Stay
```

### RL Framework Integration

#### DQN Training Example

```python
import stable_baselines3 as sb3
from stable_baselines3 import DQN

# Create environment with discrete action interface
action_interface = CardinalDiscreteAction(speed=0.1, use_8_directions=True)
env = PlumeNavigationEnv(action_interface=action_interface)

# Train DQN agent
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# Test trained agent
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    
    # Log action interpretation
    direction = action_interface.get_direction_for_action(action)
    print(f"Action: {action} ({direction})")
```

#### A2C Multi-Environment Training

```python
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env():
    action_interface = CardinalDiscreteAction(speed=0.1)
    return PlumeNavigationEnv(action_interface=action_interface)

# Parallel training environments
env = SubprocVecEnv([make_env for _ in range(4)])

# Train A2C agent
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

### Performance Optimization

#### Lookup Table Optimization

Internal action mapping uses pre-computed lookup tables for O(1) translation:

```python
# Pre-computed direction vectors (internal implementation)
action_mapping = {
    1: {  # North
        'direction': 'NORTH',
        'velocity_x': 0.0,
        'velocity_y': -speed,  # Pre-computed
        'linear_velocity': speed,
        'angular_velocity': 0.0
    },
    # ... other directions
}

# O(1) action translation
def translate_action(self, action):
    return self._action_mapping[action].copy()
```

#### Vectorized Validation

Efficient multi-agent action validation:

```python
# Single agent validation
action = 5
valid_action = action_interface.validate_action(action)

# Multi-agent batch validation (array input)
actions = np.array([5, 10, -1, 3])  # Mixed valid/invalid
valid_actions = action_interface.validate_action(actions)
# Returns: [5, 8, 0, 3] (clipped invalid actions)
```

### Edge Case Handling

The interface handles various edge cases gracefully:

```python
# Array input handling
action = np.array([3])  # Array with single element
command = action_interface.translate_action(action)

# Scalar numpy input
action = np.int32(2)
command = action_interface.translate_action(action)

# Invalid action indices
action = 999  # Far out of range
valid_action = action_interface.validate_action(action)
# Returns: max_valid_action (clipped)

# Negative action indices  
action = -5
valid_action = action_interface.validate_action(action)
# Returns: 0 (clipped to minimum)
```

## Configuration Examples

This section provides comprehensive configuration examples for both continuous and discrete action interfaces, demonstrating Hydra integration patterns and environment-specific overrides.

### Continuous2D Configuration

#### Basic Continuous Configuration

```yaml
# conf/base/action/continuous2d.yaml
action:
  _target_: plume_nav_sim.core.actions.Continuous2DAction
  
  # Action space bounds
  max_velocity: 2.0
  max_angular_velocity: 45.0
  min_velocity: -2.0
  min_angular_velocity: -45.0
  
  # Validation settings
  validation:
    enable_bounds_checking: true
    clipping:
      mode: "clip"
      clip_to_bounds: true
    input_validation:
      reject_nan: true
      reject_inf: true
      type_checking: true
```

#### Advanced Continuous Configuration

```yaml
# High-performance continuous action configuration
action:
  _target_: plume_nav_sim.core.actions.Continuous2DAction
  
  # Dynamic velocity bounds with environment overrides
  max_velocity: ${oc.env:ACTION_SPEED_MAX,3.0}
  max_angular_velocity: ${oc.env:ACTION_ANGULAR_VELOCITY_MAX,60.0}
  min_velocity: ${oc.env:ACTION_SPEED_MIN,-1.0}
  min_angular_velocity: ${oc.env:ACTION_ANGULAR_VELOCITY_MIN,-30.0}
  
  # Performance optimization
  performance:
    vectorized_operations: true
    memory_optimization:
      preallocate_arrays: true
      use_float32: true
      minimize_copies: true
    caching:
      enable_bounds_cache: true
      cache_validation_results: false
  
  # Action translation settings
  translation:
    coordinate_system: "local"
    normalization:
      input_range: [-1.0, 1.0]
      auto_normalize: true
    smoothing:
      enable: false
      alpha: 0.1
  
  # Gymnasium integration
  gymnasium_config:
    space_type: "Box"
    space_kwargs:
      low: [-1.0, -1.0]
      high: [1.0, 1.0]
      shape: [2]
      dtype: "float32"
```

#### Environment-Specific Overrides

```yaml
# Environment-specific configuration overrides
defaults:
  - base_config: continuous2d

# Production environment
production:
  action:
    validation:
      error_handling:
        mode: "raise"
    performance:
      profiling:
        enable_timing: false
    debug:
      enable_debug_mode: false

# Development environment  
development:
  action:
    validation:
      error_handling:
        mode: "warn"
    performance:
      profiling:
        enable_timing: true
    debug:
      enable_debug_mode: true
      action_logging:
        log_all_actions: true

# Testing environment
testing:
  action:
    validation:
      error_handling:
        mode: "raise"
    performance:
      caching:
        enable_bounds_cache: false
    debug:
      enable_debug_mode: true
```

### Cardinal Discrete Configuration

#### Basic Discrete Configuration

```yaml
# conf/base/action/cardinal_discrete.yaml
action:
  _target_: plume_nav_sim.core.actions.CardinalDiscreteAction
  
  # Movement configuration
  speed: 1.0
  use_8_directions: true
  include_stay_action: true
  
  # Action space definition
  action_space:
    n_actions: 9  # 8 directions + stay
  
  # Movement parameters
  movement:
    step_size: 0.1
    angular_step: 0.314159  # π/10 radians
    enable_diagonal: true
    diagonal_scale: 0.7071  # sqrt(2)/2
```

#### Grid Navigation Configuration

```yaml
# Optimized for grid-based navigation
action:
  _target_: plume_nav_sim.core.actions.CardinalDiscreteAction
  
  # Grid movement settings
  speed: 1.0  # One grid cell per action
  use_8_directions: false  # Cardinal directions only
  include_stay_action: true
  
  # Validation for grid constraints
  validation:
    check_bounds: true
    bounds_handling: "clip"
    validate_output: true
    max_step_distance: 1.0
  
  # Performance optimization for discrete actions
  optimization:
    use_lookup_table: true
    enable_action_cache: true
    vectorized_operations: true
    vector_pool_size: 100
  
  # Gymnasium discrete space
  gymnasium_config:
    space_type: "Discrete"
    space_kwargs:
      n: 5  # N, S, E, W, STAY
    action_meanings:
      0: "STOP"
      1: "NORTH"
      2: "SOUTH"
      3: "EAST"  
      4: "WEST"
```

#### 8-Direction Configuration

```yaml
# Full 8-direction movement with diagonal optimization
action:
  _target_: plume_nav_sim.core.actions.CardinalDiscreteAction
  
  # 8-direction movement
  speed: ${oc.env:DISCRETE_SPEED,1.5}
  use_8_directions: ${oc.env:ENABLE_8_DIRECTIONS,true}
  include_stay_action: true
  
  # Movement configuration
  movement:
    step_size: ${oc.env:DISCRETE_STEP_SIZE,0.15}
    angular_step: 0.785398  # π/4 radians for 8-direction
    enable_diagonal: true
    diagonal_scale: 0.7071  # Normalized diagonal movement
  
  # Extended action space
  action_space:
    n_actions: 9
    n_actions_with_diagonal: 9
  
  # Action meanings for RL frameworks
  gymnasium_config:
    space_type: "Discrete"
    space_kwargs:
      n: 9
    action_meanings:
      0: "STAY"
      1: "NORTH"
      2: "SOUTH"
      3: "EAST"
      4: "WEST"
      5: "NORTHEAST"
      6: "NORTHWEST"
      7: "SOUTHEAST"
      8: "SOUTHWEST"
```

### Hydra Composition Patterns

#### Multi-Configuration Composition

```yaml
# Main configuration file with action interface selection
defaults:
  - _self_
  - action: continuous2d  # or cardinal_discrete

# Override action interface via command line
# python train.py action=cardinal_discrete
# python train.py action=continuous2d action.max_velocity=3.0
```

#### Experimental Configuration Groups

```yaml
# conf/config.yaml - Experimental setup
defaults:
  - base_config
  - action: ???  # Must be specified
  - environment: plume_navigation
  - algorithm: ppo

# Experiment-specific overrides
experiment:
  name: "continuous_navigation_v1"
  action_interface: "continuous2d"
  
# Group-based configuration
action_config:
  type: "continuous"
  performance_mode: "high_speed"
  bounds_mode: "strict"

# Configuration validation
hydra:
  help:
    template: |
      Available action interfaces:
      - continuous2d: Smooth velocity control
      - cardinal_discrete: Grid-based movement
```

#### Runtime Switching Patterns

```yaml
# Dynamic action interface switching
action_interface:
  _target_: plume_nav_sim.core.actions.create_action_interface
  config:
    type: ${action_type}  # Specified at runtime
    # Type-specific parameters loaded based on action_type

# Conditional configuration loading
defaults:
  - action: ${select_action_config:${action_type}}

# Function to select configuration based on type
# select_action_config function maps:
# - "continuous" -> "continuous2d"
# - "discrete" -> "cardinal_discrete"
```

### Environment Variable Overrides

#### Production Deployment

```bash
# Environment variables for production tuning
export ACTION_SPEED_MAX=2.5
export ACTION_ANGULAR_VELOCITY_MAX=50.0
export DISCRETE_STEP_SIZE=0.08
export ENABLE_8_DIRECTIONS=true

# Run with environment overrides
python train.py action=continuous2d
```

#### Development Settings

```bash
# Development environment with debugging
export ACTION_SPEED_MAX=1.0  # Slower for debugging
export ACTION_DEBUG_MODE=true
export ACTION_LOG_TRANSLATIONS=true

python train.py action=continuous2d debug.enable_debug_mode=true
```

#### A/B Testing Configuration

```bash
# A/B testing with different action configurations
export EXPERIMENT_GROUP=A
export ACTION_CONFIG=${EXPERIMENT_GROUP}_action_config

# Configuration file references
python train.py action=${ACTION_CONFIG}
```

### Integration Examples

#### RL Framework Integration

```python
# Programmatic configuration with Hydra
from hydra import compose, initialize
from plume_nav_sim.core.actions import create_action_interface

# Initialize Hydra configuration
with initialize(config_path="../conf"):
    cfg = compose(config_name="config", overrides=["action=continuous2d"])
    
    # Create action interface from configuration
    action_interface = create_action_interface(cfg.action)
    
    # Use in environment
    env = PlumeNavigationEnv(action_interface=action_interface)
```

#### Configuration Validation

```python
# Validate configuration before creating interface
from plume_nav_sim.core.actions import validate_action_config

config = {
    'type': 'Continuous2D',
    'max_velocity': 2.0,
    'max_angular_velocity': 45.0
}

if validate_action_config(config):
    action_interface = create_action_interface(config)
else:
    raise ValueError("Invalid action configuration")
```

#### Dynamic Configuration Updates

```python
# Runtime configuration updates
action_interface = create_action_interface(initial_config)

# Update bounds dynamically
if isinstance(action_interface, Continuous2DAction):
    action_interface.set_bounds(max_velocity=3.0)
elif isinstance(action_interface, CardinalDiscreteAction):
    action_interface.set_speed(1.5)

# Generate new action space
updated_action_space = action_interface.get_action_space()
```

## Integration Patterns

This section covers comprehensive integration patterns for using action interfaces with RL frameworks, custom environments, and multi-agent systems.

### RL Framework Integration

#### Stable-Baselines3 Integration

```python
import gymnasium as gym
from stable_baselines3 import PPO, DQN, SAC
from plume_nav_sim.core.actions import create_action_interface
from plume_nav_sim.envs import PlumeNavigationEnv

# Continuous control with PPO
continuous_config = {
    'type': 'Continuous2D',
    'max_velocity': 2.0,
    'max_angular_velocity': 45.0
}

action_interface = create_action_interface(continuous_config)
env = PlumeNavigationEnv(action_interface=action_interface)

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Discrete control with DQN
discrete_config = {
    'type': 'CardinalDiscrete',
    'speed': 1.0,
    'use_8_directions': True
}

action_interface = create_action_interface(discrete_config)
env = PlumeNavigationEnv(action_interface=action_interface)

# Train DQN agent
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
```

#### Ray RLlib Integration

```python
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

# Action interface factory for RLlib
def create_env(config):
    action_config = config.get('action_config', {
        'type': 'Continuous2D',
        'max_velocity': 2.0
    })
    action_interface = create_action_interface(action_config)
    return PlumeNavigationEnv(action_interface=action_interface)

# RLlib training configuration
config = (
    PPOConfig()
    .environment(create_env, env_config={
        'action_config': {
            'type': 'Continuous2D',
            'max_velocity': 2.5,
            'max_angular_velocity': 60.0
        }
    })
    .framework('torch')
    .training(
        lr=3e-4,
        num_sgd_iter=10,
        sgd_minibatch_size=64
    )
)

# Run training
ray.init()
tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=ray.air.RunConfig(stop={"training_iteration": 100})
)
results = tuner.fit()
```

#### Tianshou Integration

```python
import torch
import numpy as np
from tianshou.policy import PPOPolicy, DQNPolicy
from tianshou.trainer import onpolicy_trainer, offpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer

# Custom Tianshou environment wrapper
class TianshouActionWrapper:
    def __init__(self, action_interface):
        self.action_interface = action_interface
        self.action_space = action_interface.get_action_space()
    
    def step(self, action):
        # Translate action using interface
        nav_command = self.action_interface.translate_action(action)
        # Pass to underlying environment
        return self.env.step(nav_command)

# PPO with continuous actions
continuous_interface = create_action_interface({
    'type': 'Continuous2D',
    'max_velocity': 2.0
})

env = TianshouActionWrapper(continuous_interface)
policy = PPOPolicy(actor, critic, optim, torch.distributions.Normal)

# Training loop
collector = Collector(policy, env)
result = onpolicy_trainer(
    policy, collector, max_epoch=100, step_per_epoch=1000
)
```

### Gymnasium Compatibility

#### Environment Registration

```python
import gymnasium as gym
from gymnasium.envs.registration import register

# Register environment with action interface
def make_plume_env(action_config=None):
    if action_config is None:
        action_config = {'type': 'Continuous2D'}
    
    action_interface = create_action_interface(action_config)
    return PlumeNavigationEnv(action_interface=action_interface)

# Register multiple variants
register(
    id='PlumeNavigation-Continuous-v1',
    entry_point=lambda: make_plume_env({
        'type': 'Continuous2D',
        'max_velocity': 2.0
    }),
    max_episode_steps=500
)

register(
    id='PlumeNavigation-Discrete-v1', 
    entry_point=lambda: make_plume_env({
        'type': 'CardinalDiscrete',
        'speed': 1.0,
        'use_8_directions': True
    }),
    max_episode_steps=500
)

# Use registered environments
env = gym.make('PlumeNavigation-Continuous-v1')
```

#### Action Space Validation

```python
# Automatic action space validation
def validate_env_action_space(env, action_interface):
    """Validate environment action space matches interface."""
    expected_space = action_interface.get_action_space()
    actual_space = env.action_space
    
    if type(expected_space) != type(actual_space):
        raise ValueError(f"Action space type mismatch: {type(expected_space)} vs {type(actual_space)}")
    
    if hasattr(expected_space, 'shape') and expected_space.shape != actual_space.shape:
        raise ValueError(f"Action space shape mismatch: {expected_space.shape} vs {actual_space.shape}")
    
    if hasattr(expected_space, 'n') and expected_space.n != actual_space.n:
        raise ValueError(f"Action space size mismatch: {expected_space.n} vs {actual_space.n}")
    
    return True

# Usage
action_interface = create_action_interface(config)
env = PlumeNavigationEnv(action_interface=action_interface)
validate_env_action_space(env, action_interface)
```

### Environment Step Integration

#### Standard Environment Integration

```python
class PlumeNavigationEnv(gym.Env):
    """Plume navigation environment with action interface integration."""
    
    def __init__(self, action_interface: ActionInterfaceProtocol, **kwargs):
        super().__init__()
        
        self.action_interface = action_interface
        self.action_space = action_interface.get_action_space()
        
        # Initialize navigator and other components
        self.navigator = NavigatorFactory.from_config(kwargs.get('navigator_config'))
        self.plume_model = PlumeModelFactory.from_config(kwargs.get('plume_config'))
    
    def step(self, action):
        """Execute one environment step with action interface translation."""
        # Validate action
        if not self.action_interface.validate_action(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Translate to navigation command
        nav_command = self.action_interface.translate_action(action)
        
        # Apply navigation command
        self.navigator.apply_command(nav_command)
        
        # Execute simulation step
        self.navigator.step(self.plume_model.get_current_frame())
        
        # Compute observation and reward
        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_termination()
        info = self._get_info()
        
        return obs, reward, done, info
    
    def _get_observation(self):
        """Construct observation including action interface state."""
        base_obs = {
            'position': self.navigator.positions[0],
            'orientation': self.navigator.orientations[0],
            'concentration': self.navigator.sample_odor(self.plume_model.get_current_frame())
        }
        
        # Add action interface specific observations
        if hasattr(self.action_interface, 'get_last_action'):
            base_obs['last_action'] = self.action_interface.get_last_action()
        
        return base_obs
```

#### Multi-Agent Environment Integration

```python
class MultiAgentPlumeNavigationEnv(gym.Env):
    """Multi-agent environment with shared action interface."""
    
    def __init__(self, action_interface: ActionInterfaceProtocol, num_agents: int = 4):
        super().__init__()
        
        self.action_interface = action_interface
        self.num_agents = num_agents
        
        # Multi-agent action space
        self.action_space = gym.spaces.Tuple([
            action_interface.get_action_space() for _ in range(num_agents)
        ])
    
    def step(self, actions):
        """Execute multi-agent step with vectorized action processing."""
        # Validate all actions
        validated_actions = []
        for i, action in enumerate(actions):
            if not self.action_interface.validate_action(action):
                raise ValueError(f"Invalid action for agent {i}: {action}")
            validated_actions.append(action)
        
        # Translate all actions
        nav_commands = []
        for action in validated_actions:
            nav_command = self.action_interface.translate_action(action)
            nav_commands.append(nav_command)
        
        # Apply commands to all agents
        for i, nav_command in enumerate(nav_commands):
            self.navigators[i].apply_command(nav_command)
        
        # Execute simulation step
        current_frame = self.plume_model.get_current_frame()
        for navigator in self.navigators:
            navigator.step(current_frame)
        
        # Compute multi-agent observations and rewards
        observations = [self._get_agent_observation(i) for i in range(self.num_agents)]
        rewards = [self._compute_agent_reward(i) for i in range(self.num_agents)]
        dones = [self._check_agent_termination(i) for i in range(self.num_agents)]
        infos = [self._get_agent_info(i) for i in range(self.num_agents)]
        
        return observations, rewards, dones, infos
```

### Action Space Validation

#### Pre-Training Validation

```python
def validate_training_setup(env, model, action_interface):
    """Comprehensive validation before training."""
    
    # Validate action space compatibility
    model_action_space = model.action_space
    env_action_space = env.action_space
    interface_action_space = action_interface.get_action_space()
    
    assert model_action_space == env_action_space, "Model-environment action space mismatch"
    assert env_action_space == interface_action_space, "Environment-interface action space mismatch"
    
    # Test action sampling and validation
    for _ in range(100):
        sample_action = env_action_space.sample()
        assert action_interface.validate_action(sample_action), f"Interface rejected valid action: {sample_action}"
        
        nav_command = action_interface.translate_action(sample_action)
        assert isinstance(nav_command, dict), "Invalid navigation command format"
        assert 'linear_velocity' in nav_command, "Missing linear_velocity in command"
        assert 'angular_velocity' in nav_command, "Missing angular_velocity in command"
    
    # Test environment step with interface
    obs = env.reset()
    for _ in range(10):
        action = env_action_space.sample()
        obs, reward, done, info = env.step(action)
        assert isinstance(reward, (int, float)), "Invalid reward type"
        if done:
            obs = env.reset()
    
    print("✓ Training setup validation passed")
```

#### Runtime Action Monitoring

```python
class ActionMonitorWrapper(gym.Wrapper):
    """Wrapper to monitor action interface performance and validity."""
    
    def __init__(self, env, action_interface):
        super().__init__(env)
        self.action_interface = action_interface
        self.action_stats = {
            'total_actions': 0,
            'invalid_actions': 0,
            'translation_times': [],
            'action_distribution': {}
        }
    
    def step(self, action):
        """Monitor action processing performance."""
        import time
        
        self.action_stats['total_actions'] += 1
        
        # Monitor validation
        is_valid = self.action_interface.validate_action(action)
        if not is_valid:
            self.action_stats['invalid_actions'] += 1
        
        # Monitor translation performance
        start_time = time.perf_counter()
        nav_command = self.action_interface.translate_action(action)
        translation_time = time.perf_counter() - start_time
        self.action_stats['translation_times'].append(translation_time)
        
        # Monitor action distribution
        if isinstance(action, (int, np.integer)):
            action_key = int(action)
        else:
            action_key = tuple(action.round(2)) if hasattr(action, 'round') else str(action)
        
        self.action_stats['action_distribution'][action_key] = (
            self.action_stats['action_distribution'].get(action_key, 0) + 1
        )
        
        return self.env.step(action)
    
    def get_action_stats(self):
        """Get comprehensive action statistics."""
        stats = self.action_stats.copy()
        
        if stats['translation_times']:
            stats['avg_translation_time'] = np.mean(stats['translation_times'])
            stats['max_translation_time'] = np.max(stats['translation_times'])
            stats['translation_overhead_ok'] = stats['max_translation_time'] < 0.00005  # <0.05ms
        
        stats['invalid_action_rate'] = (
            stats['invalid_actions'] / max(stats['total_actions'], 1)
        )
        
        return stats
```

### Custom Action Interfaces

#### Custom Continuous Action Interface

```python
class PIDContinuousAction(Continuous2DAction):
    """Continuous action interface with PID control integration."""
    
    def __init__(self, pid_params=None, **kwargs):
        super().__init__(**kwargs)
        
        # PID controller parameters
        self.pid_params = pid_params or {
            'kp': 1.0, 'ki': 0.1, 'kd': 0.05
        }
        
        # PID state
        self.error_integral = np.zeros(2)
        self.previous_error = np.zeros(2)
        self.target_position = None
    
    def set_target(self, target_position):
        """Set target position for PID control."""
        self.target_position = np.array(target_position)
        # Reset PID state
        self.error_integral = np.zeros(2)
        self.previous_error = np.zeros(2)
    
    def translate_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Translate action with PID control augmentation."""
        if self.target_position is None:
            # No target set, use standard translation
            return super().translate_action(action)
        
        # Current position (would be provided by environment)
        current_position = self.get_current_position()  # Implementation specific
        
        # Compute PID control
        error = self.target_position - current_position
        self.error_integral += error
        error_derivative = error - self.previous_error
        
        pid_output = (
            self.pid_params['kp'] * error +
            self.pid_params['ki'] * self.error_integral +
            self.pid_params['kd'] * error_derivative
        )
        
        self.previous_error = error
        
        # Combine RL action with PID output
        combined_action = action + pid_output
        
        # Use parent translation with combined action
        return super().translate_action(combined_action)
```

#### Custom Discrete Action Interface

```python
class AdaptiveDiscreteAction(CardinalDiscreteAction):
    """Discrete action interface with adaptive step size."""
    
    def __init__(self, speed_schedule=None, **kwargs):
        super().__init__(**kwargs)
        
        # Speed adaptation schedule
        self.speed_schedule = speed_schedule or {
            'initial_speed': 1.0,
            'min_speed': 0.1,
            'max_speed': 2.0,
            'adaptation_rate': 0.95
        }
        
        self.current_speed = self.speed_schedule['initial_speed']
        self.step_count = 0
        self.success_rate_window = []
    
    def adapt_speed(self, success: bool):
        """Adapt movement speed based on success feedback."""
        self.step_count += 1
        self.success_rate_window.append(success)
        
        # Keep window size manageable
        if len(self.success_rate_window) > 100:
            self.success_rate_window.pop(0)
        
        # Adapt speed every 10 steps
        if self.step_count % 10 == 0 and len(self.success_rate_window) >= 10:
            recent_success_rate = np.mean(self.success_rate_window[-10:])
            
            if recent_success_rate > 0.8:
                # High success rate, increase speed
                self.current_speed = min(
                    self.current_speed * 1.05,
                    self.speed_schedule['max_speed']
                )
            elif recent_success_rate < 0.3:
                # Low success rate, decrease speed
                self.current_speed = max(
                    self.current_speed * self.speed_schedule['adaptation_rate'],
                    self.speed_schedule['min_speed']
                )
            
            # Update action interface speed
            self.set_speed(self.current_speed)
    
    def translate_action(self, action: Union[int, np.ndarray]) -> Dict[str, Any]:
        """Translate action with current adaptive speed."""
        command = super().translate_action(action)
        
        # Add speed information to command
        command['adaptive_speed'] = self.current_speed
        command['step_count'] = self.step_count
        
        return command
```

## Performance Specifications

This section details the performance requirements, optimization strategies, and benchmarking guidelines for action interface implementations.

### Performance Requirements

#### Translation Overhead Targets

All action interface implementations must meet strict performance requirements for real-time operation:

| Operation | Target Time | Maximum Time | Measurement Method |
|-----------|-------------|--------------|-------------------|
| `translate_action()` | <0.05ms | 0.1ms | Per-agent per-step |
| `validate_action()` | <0.02ms | 0.05ms | Per-action validation |
| `get_action_space()` | <1ms | 5ms | One-time initialization |
| Memory allocation | <100 bytes | 500 bytes | Per-interface instance |

#### Multi-Agent Scaling

Performance must scale linearly with agent count:

```python
# Performance scaling validation
import time
import numpy as np

def benchmark_action_interface(action_interface, num_agents_list):
    """Benchmark action interface scaling performance."""
    results = {}
    
    for num_agents in num_agents_list:
        # Generate test actions
        if isinstance(action_interface, Continuous2DAction):
            actions = [np.random.uniform(-1, 1, 2) for _ in range(num_agents)]
        else:
            actions = [np.random.randint(0, action_interface.get_num_actions()) 
                      for _ in range(num_agents)]
        
        # Benchmark translation
        start_time = time.perf_counter()
        for action in actions:
            action_interface.translate_action(action)
        total_time = time.perf_counter() - start_time
        
        # Calculate per-agent timing
        per_agent_time = total_time / num_agents
        results[num_agents] = {
            'total_time': total_time,
            'per_agent_time': per_agent_time,
            'meets_requirement': per_agent_time < 0.00005  # <0.05ms
        }
    
    return results

# Usage
continuous_interface = Continuous2DAction()
results = benchmark_action_interface(continuous_interface, [1, 10, 50, 100])

for num_agents, timing in results.items():
    print(f"{num_agents} agents: {timing['per_agent_time']*1000:.3f}ms per agent")
    print(f"  Meets requirement: {timing['meets_requirement']}")
```

#### Memory Efficiency

Action interfaces must maintain minimal memory footprint:

```python
import sys
import psutil
import gc

def measure_memory_usage(action_interface_class, num_instances=100):
    """Measure memory usage of action interface instances."""
    
    # Baseline memory
    gc.collect()
    baseline_memory = psutil.Process().memory_info().rss
    
    # Create instances
    instances = []
    for _ in range(num_instances):
        if action_interface_class == Continuous2DAction:
            instance = action_interface_class(max_velocity=2.0)
        else:
            instance = action_interface_class(speed=1.0)
        instances.append(instance)
    
    # Measure memory after creation
    gc.collect()
    final_memory = psutil.Process().memory_info().rss
    
    # Calculate per-instance memory
    total_memory_used = final_memory - baseline_memory
    per_instance_memory = total_memory_used / num_instances
    
    # Object size introspection
    sample_instance = instances[0]
    object_size = sys.getsizeof(sample_instance)
    
    return {
        'total_memory_kb': total_memory_used / 1024,
        'per_instance_bytes': per_instance_memory,
        'object_size_bytes': object_size,
        'meets_requirement': per_instance_memory < 500  # <500 bytes
    }

# Benchmark both interface types
continuous_memory = measure_memory_usage(Continuous2DAction)
discrete_memory = measure_memory_usage(CardinalDiscreteAction)

print(f"Continuous2DAction: {continuous_memory['per_instance_bytes']:.1f} bytes per instance")
print(f"CardinalDiscreteAction: {discrete_memory['per_instance_bytes']:.1f} bytes per instance")
```

### Optimization Strategies

#### Vectorized Operations

Implement vectorized operations for multi-agent scenarios:

```python
class OptimizedContinuous2DAction(Continuous2DAction):
    """Optimized continuous action interface with vectorization."""
    
    def translate_actions_batch(self, actions: np.ndarray) -> List[Dict[str, Any]]:
        """Vectorized action translation for multiple agents."""
        # Input shape: (num_agents, 2)
        assert actions.ndim == 2 and actions.shape[1] == 2
        
        # Vectorized validation
        validated_actions = np.clip(
            actions,
            [self._min_velocity, self._min_angular_velocity],
            [self._max_velocity, self._max_angular_velocity]
        )
        
        # Handle NaN/Inf vectorized
        validated_actions = np.where(np.isfinite(validated_actions), validated_actions, 0.0)
        
        # Batch create command dictionaries
        commands = []
        for i in range(validated_actions.shape[0]):
            commands.append({
                'linear_velocity': float(validated_actions[i, 0]),
                'angular_velocity': float(validated_actions[i, 1]),
                'action_type': 'continuous_2d'
            })
        
        return commands
    
    def validate_actions_batch(self, actions: np.ndarray) -> np.ndarray:
        """Vectorized action validation."""
        # Ensure 2D array
        if actions.ndim == 1:
            actions = actions.reshape(-1, 2)
        
        # Vectorized bounds checking and clipping
        return np.clip(
            actions,
            [self._min_velocity, self._min_angular_velocity],
            [self._max_velocity, self._max_angular_velocity]
        ).astype(np.float32)
```

#### Lookup Table Optimization

Pre-compute action mappings for discrete interfaces:

```python
class OptimizedCardinalDiscreteAction(CardinalDiscreteAction):
    """Optimized discrete action interface with lookup tables."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Pre-compute all possible command dictionaries
        self._command_cache = {}
        for action_idx in self._action_mapping.keys():
            self._command_cache[action_idx] = self._create_command_dict(action_idx)
    
    def _create_command_dict(self, action_idx: int) -> Dict[str, Any]:
        """Create command dictionary for action index."""
        movement_data = self._action_mapping[action_idx]
        return {
            'linear_velocity': movement_data['linear_velocity'],
            'angular_velocity': movement_data['angular_velocity'],
            'velocity_x': movement_data['velocity_x'],
            'velocity_y': movement_data['velocity_y'],
            'direction': movement_data['direction'],
            'action_type': 'cardinal_discrete'
        }
    
    def translate_action(self, action: Union[int, np.ndarray]) -> Dict[str, Any]:
        """O(1) action translation using pre-computed lookup table."""
        # Handle array input
        if isinstance(action, np.ndarray):
            if action.shape == ():
                action = int(action.item())
            elif action.shape == (1,):
                action = int(action[0])
            else:
                raise ValueError(f"Invalid action array shape: {action.shape}")
        
        action = int(action)
        
        # Validate action bounds
        if action not in self._command_cache:
            raise ValueError(f"Invalid action {action}. Valid actions: 0-{self._num_actions-1}")
        
        # O(1) lookup - return copy to prevent modification
        return self._command_cache[action].copy()
```

#### Memory Pool Optimization

Use memory pools for high-frequency allocations:

```python
class MemoryPoolActionInterface:
    """Action interface with memory pool optimization."""
    
    def __init__(self, pool_size: int = 1000):
        self.pool_size = pool_size
        
        # Pre-allocate command dictionaries
        self._command_pool = []
        for _ in range(pool_size):
            self._command_pool.append({
                'linear_velocity': 0.0,
                'angular_velocity': 0.0,
                'action_type': 'pooled'
            })
        
        self._pool_index = 0
    
    def get_pooled_command(self) -> Dict[str, Any]:
        """Get command dictionary from memory pool."""
        command = self._command_pool[self._pool_index]
        self._pool_index = (self._pool_index + 1) % self.pool_size
        return command
    
    def translate_action_pooled(self, action: np.ndarray) -> Dict[str, Any]:
        """Memory-pool optimized action translation."""
        # Get pre-allocated command dict
        command = self.get_pooled_command()
        
        # Validate action
        validated_action = self.validate_action(action)
        
        # Update command in-place
        command['linear_velocity'] = float(validated_action[0])
        command['angular_velocity'] = float(validated_action[1])
        
        return command
```

### Benchmarking Guidelines

#### Performance Test Suite

```python
class ActionInterfacePerformanceSuite:
    """Comprehensive performance testing for action interfaces."""
    
    def __init__(self, action_interface: ActionInterfaceProtocol):
        self.action_interface = action_interface
        self.test_results = {}
    
    def run_translation_benchmark(self, num_iterations: int = 10000) -> Dict[str, float]:
        """Benchmark action translation performance."""
        import time
        
        # Generate test actions
        if isinstance(self.action_interface, Continuous2DAction):
            test_actions = [np.random.uniform(-1, 1, 2) for _ in range(num_iterations)]
        else:
            test_actions = [np.random.randint(0, self.action_interface.get_num_actions()) 
                           for _ in range(num_iterations)]
        
        # Warm-up
        for _ in range(100):
            self.action_interface.translate_action(test_actions[0])
        
        # Benchmark translation
        start_time = time.perf_counter()
        for action in test_actions:
            self.action_interface.translate_action(action)
        total_time = time.perf_counter() - start_time
        
        avg_time = total_time / num_iterations
        max_allowed_time = 0.00005  # 0.05ms
        
        return {
            'total_time_ms': total_time * 1000,
            'avg_time_ms': avg_time * 1000,
            'avg_time_us': avg_time * 1000000,
            'operations_per_second': num_iterations / total_time,
            'meets_requirement': avg_time < max_allowed_time,
            'performance_margin': (max_allowed_time - avg_time) / max_allowed_time * 100
        }
    
    def run_validation_benchmark(self, num_iterations: int = 10000) -> Dict[str, float]:
        """Benchmark action validation performance."""
        import time
        
        # Generate test actions (mix of valid and invalid)
        if isinstance(self.action_interface, Continuous2DAction):
            test_actions = [np.random.uniform(-5, 5, 2) for _ in range(num_iterations)]
        else:
            test_actions = [np.random.randint(-10, 20) for _ in range(num_iterations)]
        
        # Benchmark validation
        start_time = time.perf_counter()
        for action in test_actions:
            self.action_interface.validate_action(action)
        total_time = time.perf_counter() - start_time
        
        avg_time = total_time / num_iterations
        max_allowed_time = 0.00002  # 0.02ms
        
        return {
            'total_time_ms': total_time * 1000,
            'avg_time_ms': avg_time * 1000,
            'avg_time_us': avg_time * 1000000,
            'meets_requirement': avg_time < max_allowed_time,
            'performance_margin': (max_allowed_time - avg_time) / max_allowed_time * 100
        }
    
    def run_memory_benchmark(self) -> Dict[str, Any]:
        """Benchmark memory usage."""
        import sys
        import gc
        
        # Measure object size
        object_size = sys.getsizeof(self.action_interface)
        
        # Measure deep memory usage
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss
        
        # Create multiple instances
        instances = []
        for _ in range(100):
            if isinstance(self.action_interface, Continuous2DAction):
                instance = Continuous2DAction()
            else:
                instance = CardinalDiscreteAction()
            instances.append(instance)
        
        gc.collect()
        final_memory = psutil.Process().memory_info().rss
        per_instance_memory = (final_memory - baseline_memory) / 100
        
        max_allowed_memory = 500  # 500 bytes
        
        return {
            'object_size_bytes': object_size,
            'per_instance_memory_bytes': per_instance_memory,
            'meets_requirement': per_instance_memory < max_allowed_memory,
            'memory_efficiency': (max_allowed_memory - per_instance_memory) / max_allowed_memory * 100
        }
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all benchmark tests."""
        self.test_results = {
            'translation_performance': self.run_translation_benchmark(),
            'validation_performance': self.run_validation_benchmark(),
            'memory_usage': self.run_memory_benchmark()
        }
        
        # Overall performance assessment
        all_requirements_met = all([
            self.test_results['translation_performance']['meets_requirement'],
            self.test_results['validation_performance']['meets_requirement'],
            self.test_results['memory_usage']['meets_requirement']
        ])
        
        self.test_results['overall_assessment'] = {
            'all_requirements_met': all_requirements_met,
            'performance_score': self._calculate_performance_score()
        }
        
        return self.test_results
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        translation_score = min(100, self.test_results['translation_performance']['performance_margin'])
        validation_score = min(100, self.test_results['validation_performance']['performance_margin'])
        memory_score = min(100, self.test_results['memory_usage']['memory_efficiency'])
        
        return (translation_score + validation_score + memory_score) / 3

# Usage
continuous_interface = Continuous2DAction()
suite = ActionInterfacePerformanceSuite(continuous_interface)
results = suite.run_comprehensive_benchmark()

print(f"Translation Performance: {results['translation_performance']['avg_time_us']:.2f}μs")
print(f"Validation Performance: {results['validation_performance']['avg_time_us']:.2f}μs")
print(f"Memory Usage: {results['memory_usage']['per_instance_memory_bytes']:.1f} bytes")
print(f"Overall Score: {results['overall_assessment']['performance_score']:.1f}/100")
```

#### Continuous Integration Benchmarks

```python
def ci_performance_test():
    """Performance test for continuous integration pipeline."""
    
    # Test both interface types
    interfaces = [
        ('Continuous2D', Continuous2DAction()),
        ('CardinalDiscrete', CardinalDiscreteAction())
    ]
    
    all_passed = True
    
    for name, interface in interfaces:
        suite = ActionInterfacePerformanceSuite(interface)
        results = suite.run_comprehensive_benchmark()
        
        # Check requirements
        translation_ok = results['translation_performance']['meets_requirement']
        validation_ok = results['validation_performance']['meets_requirement']
        memory_ok = results['memory_usage']['meets_requirement']
        
        print(f"\n{name} Performance Test:")
        print(f"  Translation: {'✓' if translation_ok else '✗'} "
              f"({results['translation_performance']['avg_time_us']:.2f}μs)")
        print(f"  Validation: {'✓' if validation_ok else '✗'} "
              f"({results['validation_performance']['avg_time_us']:.2f}μs)")
        print(f"  Memory: {'✓' if memory_ok else '✗'} "
              f"({results['memory_usage']['per_instance_memory_bytes']:.1f} bytes)")
        
        if not (translation_ok and validation_ok and memory_ok):
            all_passed = False
    
    if not all_passed:
        raise AssertionError("Performance requirements not met")
    
    print("\n✓ All performance tests passed")

# Run in CI
if __name__ == "__main__":
    ci_performance_test()
```

This comprehensive API reference provides complete documentation for the standardized action processing interfaces, covering protocol specifications, implementation details, configuration patterns, integration strategies, and performance requirements. The documentation enables developers to effectively use and extend the action interface system for diverse RL applications and navigation scenarios.