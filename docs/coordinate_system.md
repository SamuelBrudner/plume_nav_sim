# Coordinate System and Spatial Conventions

## Overview

The `plume_nav_sim` library implements a standardized 2D coordinate system designed for odor plume navigation research. This document details the spatial conventions, coordinate transformations, and API changes introduced with the Gymnasium 0.29.x migration.

## Table of Contents

- [Core Coordinate System](#core-coordinate-system)
- [Spatial Transformations](#spatial-transformations)
- [Gymnasium API Changes](#gymnasium-api-changes)
- [Space Definitions](#space-definitions)
- [Coordinate Frame Consistency](#coordinate-frame-consistency)
- [Usage Examples](#usage-examples)
- [Performance Considerations](#performance-considerations)
- [Migration Guide](#migration-guide)

## Core Coordinate System

### Reference Frame

The plume_nav_sim environment uses a **2D Cartesian coordinate system** with the following conventions:

```
Video/Environment Frame:
┌─────────────────────► X (width)
│ (0,0)
│    ┌─────┬─────┬─────┐
│    │     │     │     │
│    ├─────┼─────┼─────┤  
│    │     │ Ag  │     │  ← Agent at position (x, y)
│    ├─────┼─────┼─────┤
│    │     │     │     │
│    └─────┴─────┴─────┘
▼
Y (height)
```

### Key Properties

- **Origin**: Top-left corner at (0, 0)
- **X-axis**: Horizontal, increasing rightward (positive direction)
- **Y-axis**: Vertical, increasing downward (positive direction)
- **Units**: Typically pixels for video-based environments
- **Bounds**: [0, width] × [0, height] for environment dimensions

### Orientation Convention

Agent orientations follow standard navigation conventions:

```
        90° (Up)
         ↑
         │
180° ←───●───→ 0° (Right)
         │
         ↓
       270° (Down)
```

- **0°**: Right (positive X direction)
- **90°**: Up (negative Y direction) 
- **180°**: Left (negative X direction)
- **270°**: Down (positive Y direction)
- **Range**: [0°, 360°) with automatic normalization

## Spatial Transformations

### Position Updates

Agent positions are updated using standard kinematic equations:

```python
# Position update formula
new_position = position + speed * dt * [cos(θ), sin(θ)]

# Where:
# position: Current [x, y] coordinates
# speed: Linear velocity magnitude (units/time)
# dt: Time step (typically 1.0 for discrete steps)
# θ: Orientation in radians (converted from degrees)
```

### Orientation Updates

Agent orientations are updated based on angular velocity:

```python
# Orientation update formula  
new_orientation = (orientation + angular_velocity * dt) % 360.0

# Where:
# orientation: Current heading in degrees
# angular_velocity: Rotational velocity (degrees/second)
# dt: Time step duration
# % 360.0: Normalization to [0, 360) range
```

### Coordinate Transformations

#### Environment to Screen Coordinates

For visualization and rendering:

```python
def env_to_screen(env_pos, screen_height):
    """Convert environment coordinates to screen coordinates."""
    screen_x = env_pos[0]  # X remains the same
    screen_y = screen_height - env_pos[1]  # Flip Y-axis
    return (screen_x, screen_y)
```

#### Sensor Position Calculation

Multi-sensor positions relative to agent:

```python
def calculate_sensor_positions(agent_pos, agent_orientation, sensor_distance, sensor_angles):
    """Calculate sensor positions relative to agent."""
    sensor_positions = []
    
    for sensor_angle in sensor_angles:
        # Total angle = agent orientation + sensor offset
        total_angle = (agent_orientation + sensor_angle) % 360.0
        angle_rad = np.radians(total_angle)
        
        # Sensor position relative to agent
        sensor_x = agent_pos[0] + sensor_distance * np.cos(angle_rad)
        sensor_y = agent_pos[1] + sensor_distance * np.sin(angle_rad)
        
        sensor_positions.append((sensor_x, sensor_y))
    
    return np.array(sensor_positions)
```

## Gymnasium API Changes

### New Environment IDs

The Gymnasium migration introduces new standardized environment identifiers:

```python
# Modern Gymnasium registration (Recommended)
"PlumeNavSim-v0"  # Primary environment ID for Gymnasium 0.29.x

# Legacy compatibility (Deprecated)  
"OdorPlumeNavigation-v1"  # Maintained for backward compatibility
```

### Step Return Format Changes

#### Legacy Gym API (4-tuple)
```python
observation, reward, done, info = env.step(action)
```

#### Modern Gymnasium API (5-tuple)
```python
observation, reward, terminated, truncated, info = env.step(action)
```

### Automatic API Detection

The environment automatically detects the calling context and returns the appropriate format:

```python
# The environment automatically detects legacy vs modern callers
from plume_nav_sim.shims import gym_make  # Legacy pathway - returns 4-tuple
import gymnasium as gym; env = gym.make("PlumeNavSim-v0")  # Modern - returns 5-tuple
```

### Reset Method Changes

#### Legacy Reset
```python
observation = env.reset()
```

#### Modern Reset (Gymnasium 0.29.x)
```python
observation, info = env.reset(seed=42, options={"position": (100, 200)})
```

## Space Definitions

### Action Space

The action space represents continuous control commands for agent navigation:

```python
from plume_nav_sim.envs.spaces import ActionSpace

# Create action space for continuous control
action_space = ActionSpace.create(
    max_speed=2.0,              # Maximum linear velocity
    max_angular_velocity=90.0,  # Maximum angular velocity (degrees/sec)
    min_speed=0.0,              # Minimum speed (non-negative)
    dtype=np.float32            # Numerical precision
)

# Action format: [speed, angular_velocity]
# Shape: (2,)
# Bounds: [min_speed, max_speed] × [-max_angular_velocity, +max_angular_velocity]
```

### Observation Space

The observation space provides multi-modal environmental perception:

```python
from plume_nav_sim.envs.spaces import ObservationSpace

# Create observation space for environmental sensing
observation_space = ObservationSpace.create(
    env_width=640.0,           # Environment width (pixels)
    env_height=480.0,          # Environment height (pixels)
    num_sensors=2,             # Number of additional sensors
    include_multi_sensor=True, # Enable multi-sensor observations
    dtype=np.float32           # Numerical precision
)

# Observation format (dictionary):
{
    "odor_concentration": float,    # Scalar [0.0, 1.0] 
    "agent_position": [float, float],  # [x, y] coordinates
    "agent_orientation": float,     # Degrees [0.0, 360.0)
    "multi_sensor_readings": [float, ...]  # Optional sensor array
}
```

### Space Factory Utilities

Standardized space creation for common configurations:

```python
from plume_nav_sim.envs.spaces import SpaceFactory

# Standard single-sensor configuration
action_space, obs_space = SpaceFactory.create_standard_spaces(
    env_width=640, env_height=480, max_speed=2.0
)

# Research configurations with predefined sensor layouts
action_space, obs_space = SpaceFactory.create_research_spaces(
    env_width=800, env_height=600,
    sensor_config='bilateral',  # 'single', 'bilateral', 'triangular'
    speed_config='fast'         # 'slow', 'standard', 'fast'
)
```

## Coordinate Frame Consistency

### Validation Patterns

The system implements comprehensive validation to ensure coordinate frame consistency:

#### Position Bounds Validation
```python
def validate_position_bounds(position, env_width, env_height):
    """Validate position is within environment bounds."""
    x, y = position
    if not (0 <= x <= env_width and 0 <= y <= env_height):
        raise ValueError(f"Position {position} outside bounds ({env_width}x{env_height})")
    return True
```

#### Observation Consistency Checks
```python
from plume_nav_sim.envs.spaces import ObservationSpace

# Validate observation against space constraints
is_valid = ObservationSpace.validate_observation(
    observation=obs_dict, 
    observation_space=obs_space,
    step_info=step_info  # Optional Gymnasium 0.29.x step info
)
```

### Coordinate Frame Transformations

#### Boundary Enforcement
```python
def enforce_boundaries(position, env_width, env_height):
    """Clamp position to environment boundaries."""
    x = np.clip(position[0], 0, env_width)
    y = np.clip(position[1], 0, env_height)
    return np.array([x, y])
```

#### Multi-Agent Coordinate Management
```python
# Vectorized position updates for multiple agents
def update_multi_agent_positions(positions, speeds, orientations, dt):
    """Update positions for multiple agents simultaneously."""
    orientations_rad = np.radians(orientations)
    
    # Vectorized velocity calculation
    velocities = speeds[:, np.newaxis] * np.column_stack([
        np.cos(orientations_rad),
        np.sin(orientations_rad)
    ])
    
    # Apply position updates
    new_positions = positions + velocities * dt
    return new_positions
```

## Usage Examples

### Basic Navigation Setup

```python
import numpy as np
from plume_nav_sim.envs import GymnasiumEnv
from plume_nav_sim.envs.spaces import ActionSpace, ObservationSpace

# Create environment with coordinate system configuration
env = GymnasiumEnv(
    video_path="data/plume_movie.mp4",
    initial_position=(320, 240),    # Center of 640x480 environment
    initial_orientation=0.0,         # Facing right (0°)
    max_speed=2.0,                  # 2 pixels per time step
    max_angular_velocity=90.0       # 90 degrees per second
)

# Reset with custom initial conditions
obs, info = env.reset(
    seed=42,
    options={
        "position": (100, 200),     # Custom starting position
        "orientation": 45.0         # Custom starting orientation
    }
)

print(f"Initial position: {obs['agent_position']}")
print(f"Initial orientation: {obs['agent_orientation']}")
print(f"Environment dimensions: {info['video_metadata']['width']}x{info['video_metadata']['height']}")
```

### Multi-Sensor Navigation

```python
# Configure environment with triangular sensor layout
env = GymnasiumEnv(
    video_path="data/plume_movie.mp4",
    include_multi_sensor=True,
    num_sensors=3,
    sensor_distance=10.0,
    sensor_layout="triangular"
)

obs, info = env.reset()

# Multi-sensor observation format
print(f"Agent position: {obs['agent_position']}")           # [x, y]
print(f"Agent orientation: {obs['agent_orientation']}")     # degrees
print(f"Odor concentration: {obs['odor_concentration']}")   # [0.0, 1.0]
print(f"Sensor readings: {obs['multi_sensor_readings']}")   # [s1, s2, s3]

# Action format: [speed, angular_velocity]
action = np.array([1.5, 30.0])  # Move forward at 1.5 units/step, turn 30°/sec
obs, reward, terminated, truncated, info = env.step(action)
```

### Coordinate Transformation Example

```python
import numpy as np

def transform_to_world_coordinates(agent_pos, agent_orientation, relative_pos):
    """Transform relative coordinates to world coordinates."""
    # Convert orientation to radians
    angle_rad = np.radians(agent_orientation)
    
    # Rotation matrix for 2D transformation
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ])
    
    # Apply rotation and translation
    world_pos = agent_pos + rotation_matrix @ relative_pos
    return world_pos

# Example: Sensor 10 pixels to the right of agent
agent_pos = np.array([100, 200])
agent_orientation = 45.0  # 45 degrees
relative_sensor_pos = np.array([10, 0])  # 10 pixels right

world_sensor_pos = transform_to_world_coordinates(
    agent_pos, agent_orientation, relative_sensor_pos
)
print(f"Sensor world position: {world_sensor_pos}")
```

### Space Validation Example

```python
from plume_nav_sim.envs.spaces import SpaceFactory

# Create and validate spaces
action_space, obs_space = SpaceFactory.create_standard_spaces(640, 480)

# Validate space configuration
validation_results = SpaceFactory.validate_gymnasium_compatibility(
    action_space, obs_space, check_terminated_truncated=True
)

if validation_results["overall_valid"]:
    print("✓ Spaces are Gymnasium 0.29.x compatible")
else:
    print("✗ Space validation failed:")
    for error in validation_results["errors"]:
        print(f"  - {error}")
```

## Performance Considerations

### Coordinate System Performance

The coordinate system implementation is optimized for real-time performance:

#### Vectorized Operations
```python
# Efficient multi-agent position updates using NumPy
def vectorized_position_update(positions, speeds, orientations, dt):
    """High-performance position update for multiple agents."""
    # Convert to radians once
    angles_rad = np.radians(orientations)
    
    # Vectorized trigonometry
    dx = speeds * np.cos(angles_rad) * dt
    dy = speeds * np.sin(angles_rad) * dt
    
    # Single array operation
    positions += np.column_stack([dx, dy])
    return positions
```

#### Memory-Efficient Bounds Checking
```python
def efficient_bounds_check(positions, width, height):
    """Memory-efficient boundary validation."""
    # Use in-place operations to minimize memory allocation
    np.clip(positions[:, 0], 0, width, out=positions[:, 0])
    np.clip(positions[:, 1], 0, height, out=positions[:, 1])
    return positions
```

### Frame Cache Integration

The coordinate system integrates with the enhanced frame caching system:

```python
# Cache-aware coordinate transformations
def cache_optimized_sampling(cache, frame_index, positions):
    """Coordinate-aware frame sampling with caching."""
    frame = cache.get(frame_index)  # Sub-10ms retrieval target
    
    # Batch coordinate sampling for efficiency
    concentrations = []
    for pos in positions:
        if 0 <= pos[0] < frame.shape[1] and 0 <= pos[1] < frame.shape[0]:
            # Bilinear interpolation for sub-pixel accuracy
            concentration = bilinear_sample(frame, pos[0], pos[1])
            concentrations.append(concentration)
        else:
            concentrations.append(0.0)  # Outside bounds
    
    return np.array(concentrations)
```

## Migration Guide

### From Legacy Gym to Gymnasium 0.29.x

#### Step 1: Update Environment Creation

**Before (Legacy Gym):**
```python
import gym
from plume_nav_sim.shims import gym_make

env = gym_make("OdorPlumeNavigation-v1")  # Deprecated
```

**After (Modern Gymnasium):**
```python
import gymnasium as gym

env = gym.make("PlumeNavSim-v0")  # Recommended
```

#### Step 2: Update Step Return Handling

**Before (4-tuple):**
```python
obs, reward, done, info = env.step(action)
if done:
    obs = env.reset()
```

**After (5-tuple):**
```python
obs, reward, terminated, truncated, info = env.step(action)
if terminated or truncated:
    obs, info = env.reset()
```

#### Step 3: Update Reset Method Calls

**Before:**
```python
obs = env.reset()
```

**After:**
```python
obs, info = env.reset(seed=42, options={"position": (100, 200)})
```

#### Step 4: Space Definition Updates

**Before (Manual Configuration):**
```python
# Manual space creation
action_space = gym.spaces.Box(low=0, high=2, shape=(2,))
```

**After (Standardized Factory):**
```python
from plume_nav_sim.envs.spaces import ActionSpace

action_space = ActionSpace.create(max_speed=2.0, max_angular_velocity=90.0)
```

### Backward Compatibility

The system maintains full backward compatibility through the shim layer:

```python
# Legacy code continues to work unchanged
from plume_nav_sim.shims import gym_make

env = gym_make("PlumeNavSim-v0")  # Automatically returns 4-tuple format
obs, reward, done, info = env.step(action)  # Works as expected

# Deprecation warnings guide migration:
# DeprecationWarning: Using gym_make is deprecated and will be removed in v1.0.
# Please update to: gymnasium.make('PlumeNavSim-v0')
```

### Configuration Updates

Update Hydra configurations for the new coordinate system features:

```yaml
# conf/base/env/coordinate_system.yaml
coordinate_system:
  origin: "top_left"           # Coordinate system origin
  orientation_convention: "standard"  # 0° = right, 90° = up
  bounds_enforcement: true     # Automatic boundary enforcement
  precision: "float32"         # Coordinate precision

# conf/base/env/spaces.yaml  
spaces:
  action_space:
    max_speed: 2.0
    max_angular_velocity: 90.0
    min_speed: 0.0
    
  observation_space:
    include_multi_sensor: false
    num_sensors: 2
    sensor_distance: 5.0
    sensor_layout: "bilateral"
```

---

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [NavigatorProtocol Interface](../src/plume_nav_sim/core/protocols.py)
- [Space Definitions](../src/plume_nav_sim/envs/spaces.py)
- [Frame Cache Performance](../src/plume_nav_sim/utils/frame_cache.py)

## Version History

- **v0.3.0**: Gymnasium 0.29.x compatibility with coordinate system enhancements
- **v0.2.x**: Legacy Gym 0.26 coordinate system
- **v0.1.x**: Initial coordinate system implementation

For questions or issues related to the coordinate system, please refer to the technical specification or open an issue in the project repository.