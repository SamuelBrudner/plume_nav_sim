# Configuration System API Reference

## Overview

The plume_nav_sim v1.0 configuration system provides a modular, hierarchical approach to system configuration through Hydra's powerful composition framework. This system enables zero-code extensibility, component-based architecture, and runtime parameter management for scientific reproducibility and deployment flexibility.

## Table of Contents

- [Configuration Architecture](#configuration-architecture)
- [Hydra Configuration Groups](#hydra-configuration-groups)
- [Component Configuration Reference](#component-configuration-reference)
- [Composition Patterns](#composition-patterns)
- [Environment Variable Integration](#environment-variable-integration)
- [Migration Guide (v0.3.0 → v1.0)](#migration-guide-v030--v10)
- [Validation and Type Safety](#validation-and-type-safety)
- [Performance Considerations](#performance-considerations)

## Configuration Architecture

### Hierarchical Structure

The v1.0 configuration system follows a hierarchical structure that enables modular component composition:

```
conf/
├── config.yaml                 # Main user configuration
├── base.yaml                   # System foundation defaults
├── base/
│   ├── source/                 # Source component configurations
│   │   ├── point_source.yaml
│   │   ├── multi_source.yaml
│   │   └── dynamic_source.yaml
│   ├── agent_init/             # Agent initialization strategies
│   │   ├── uniform_random.yaml
│   │   ├── grid.yaml
│   │   ├── fixed_list.yaml
│   │   └── from_dataset.yaml
│   ├── boundary/               # Boundary handling policies
│   │   ├── terminate.yaml
│   │   ├── bounce.yaml
│   │   ├── wrap.yaml
│   │   └── clip.yaml
│   ├── action/                 # Action interface implementations
│   │   ├── continuous2d.yaml
│   │   └── cardinal_discrete.yaml
│   ├── record/                 # Data recording backends
│   │   ├── parquet.yaml
│   │   ├── hdf5.yaml
│   │   ├── sqlite.yaml
│   │   └── none.yaml
│   └── hooks/                  # Extension hook configurations
│       ├── none.yaml
│       ├── research.yaml
│       └── custom.yaml
└── local/                      # Environment-specific overrides
```

### Configuration Composition

The system uses Hydra's `defaults` composition to enable flexible component selection:

```yaml
# config.yaml
defaults:
  - base                        # Foundation parameters
  - source: point_source        # Odor source type
  - agent_init: uniform_random  # Initialization strategy
  - boundary: terminate         # Boundary policy
  - action: continuous2d        # Action interface
  - record: parquet            # Recording backend
  - hooks: none                # Extension hooks
  - _self_                     # User overrides
```

## Hydra Configuration Groups

### Component Selection Syntax

Components can be selected using Hydra's configuration group syntax:

```bash
# Basic component selection
python run_simulation.py source=point_source boundary=terminate

# Multiple component configuration
python run_simulation.py source=multi_source agent_init=grid action=continuous2d

# Adding optional components
python run_simulation.py +hooks=research +record=hdf5

# Removing components
python run_simulation.py -record

# Multi-run experiments
python run_simulation.py --multirun source=point_source,multi_source,dynamic_source
```

### Configuration Override Patterns

```bash
# Parameter overrides
python run_simulation.py source.position=[25.0,75.0]
python run_simulation.py agent_init.grid.spacing=8.0

# Nested parameter modification
python run_simulation.py boundary.bounce.restitution_coefficient=0.9
python run_simulation.py record.parquet.compression=gzip

# Environment variable integration
BOUNDARY_TYPE=bounce python run_simulation.py
SOURCE_POSITION="[50.0,50.0]" python run_simulation.py
```

## Component Configuration Reference

### Source Components (`conf/base/source/`)

#### Point Source Configuration

```yaml
# point_source.yaml
_target_: plume_nav_sim.core.sources.PointSource

# Basic configuration
position: ${oc.env:SOURCE_POSITION,[10.0, 10.0]}
emission_rate: ${oc.env:SOURCE_STRENGTH,1.0}
detection_threshold: 0.01
max_detection_distance: 50.0
active: true

# Emission profile
emission_profile:
  type: constant  # constant, pulsed, variable
  pulse_duration: 1.0
  pulse_interval: 5.0
  variation_amplitude: 0.1
  variation_frequency: 0.1

# Geometry and detection
geometry:
  radius: 0.0
  height: 0.0

detection:
  noise_level: 0.05
  response_time: 0.1
  binary_detection: false

# Performance optimization
performance:
  update_frequency: 100.0
  vectorized: true
  cache_emissions: true
```

**Environment Variables:**
- `SOURCE_POSITION`: Source coordinates as `"[x,y]"`
- `SOURCE_STRENGTH`: Emission rate (float)

**Usage Example:**
```bash
# Basic point source
python run_simulation.py source=point_source

# Custom position and strength
SOURCE_POSITION="[25.0,75.0]" SOURCE_STRENGTH=1500.0 python run_simulation.py source=point_source

# Parameter override
python run_simulation.py source=point_source source.emission_rate=2000.0
```

### Agent Initialization Components (`conf/base/agent_init/`)

#### Uniform Random Initialization

```yaml
# uniform_random.yaml
_target_: plume_nav_sim.core.initialization.UniformRandomInitializer

# Core parameters
bounds: [100.0, 100.0]  # Domain dimensions [width, height]
seed: null               # Random seed (null = non-deterministic)
margin: 0.0             # Safety margin from edges

# Advanced features
min_distance: 0.0       # Minimum inter-agent distance
max_attempts: 100       # Rejection sampling limit

# Collision avoidance
collision_avoidance:
  enabled: false
  agent_radius: 0.5
  resolution_method: "push"  # push, redistribute, cluster

# Performance settings
performance:
  vectorized_ops: true
  dtype: "float32"
```

**Environment Variables:**
- `AGENT_INIT_SEED`: Random seed for deterministic initialization
- `AGENT_INIT_DOMAIN_WIDTH`: Domain width override
- `AGENT_INIT_DOMAIN_HEIGHT`: Domain height override

#### Grid Initialization

```yaml
# grid.yaml
_target_: plume_nav_sim.core.initialization.GridInitializer

# Grid configuration
grid_shape: ${oc.env:GRID_SHAPE,"[3, 3]"}  # [rows, cols]
spacing: ${oc.env:GRID_SPACING,"10.0"}     # Distance between points
center_position: [50.0, 50.0]             # Grid center

# Orientation settings
grid_orientation: 0.0                      # Grid rotation
agent_orientations: "toward_center"        # random, uniform, toward_center

# Jitter for variation
position_jitter: 0.0      # Random position offset
orientation_jitter: 0.0   # Random orientation offset
```

**Environment Variables:**
- `GRID_SHAPE`: Grid dimensions as `"[rows,cols]"`
- `GRID_SPACING`: Inter-agent spacing (float)

### Boundary Policy Components (`conf/base/boundary/`)

#### Terminate Policy

```yaml
# terminate.yaml
_target_: plume_nav_sim.core.boundaries.TerminateBoundary

# Domain configuration
domain_bounds: [100.0, 100.0]  # [width, height]
allow_negative_coords: false

# Termination settings
status_on_violation: "oob"      # Termination status string
```

**Environment Variables:**
- `BOUNDARY_TERMINATE_STATUS`: Custom termination status
- `BOUNDARY_DOMAIN_BOUNDS`: Domain bounds as `"[width,height]"`

#### Bounce Policy

```yaml
# bounce.yaml
_target_: plume_nav_sim.core.boundaries.BounceBoundary

# Physics parameters
restitution_coefficient: ${oc.env:BOUNCE_RESTITUTION,"0.8"}
velocity_damping: 0.1
angular_damping: 0.05

# Bounce behavior
perfect_reflection: false
noise_on_bounce: 0.02
minimum_bounce_velocity: 0.1
```

**Environment Variables:**
- `BOUNCE_RESTITUTION`: Energy retention coefficient (0.0-1.0)

### Action Interface Components (`conf/base/action/`)

#### Continuous 2D Actions

```yaml
# continuous2d.yaml
_target_: plume_nav_sim.core.actions.Continuous2DAction

# Action space configuration
action_bounds: [[-1.0, -1.0], [1.0, 1.0]]  # [[min_x, min_y], [max_x, max_y]]
action_scaling: "linear"                     # linear, quadratic, sigmoid
coordinate_system: "local"                   # local, global
velocity_control: true

# Action processing
temporal_smoothing: false
smoothing_window: 3
noise_injection: 0.0

# Constraints
max_acceleration: ${oc.env:MAX_ACCELERATION,"5.0"}
min_action_threshold: 0.01
```

#### Cardinal Discrete Actions

```yaml
# cardinal_discrete.yaml
_target_: plume_nav_sim.core.actions.CardinalDiscreteAction

# Discrete action configuration
num_actions: ${oc.env:NUM_ACTIONS,"8"}        # 4, 8, or 9
include_stop_action: true
fixed_action_magnitude: ${oc.env:ACTION_MAGNITUDE,"1.0"}

# Direction mapping
action_directions: null  # Custom directions (null = default)
angle_offset: 0.0       # Angular offset in degrees
```

**Environment Variables:**
- `NUM_ACTIONS`: Number of discrete actions (4, 8, or 9)
- `ACTION_MAGNITUDE`: Fixed action strength

### Recording Components (`conf/base/record/`)

#### Parquet Backend

```yaml
# parquet.yaml
_target_: plume_nav_sim.recording.backends.ParquetRecorder

# Recording configuration
recording:
  full: ${oc.env:PLUME_RECORD_FULL,true}
  step_interval: 1
  include_observations: true
  include_actions: true
  include_rewards: true

# Storage settings
storage:
  base_directory: ${oc.env:PLUME_OUTPUT_DIR,"./simulation_data"}
  compression:
    algorithm: ${oc.env:PLUME_PARQUET_COMPRESSION,"snappy"}

# Performance optimization
performance:
  buffer:
    max_records: ${oc.env:PLUME_BUFFER_SIZE,1000}
    flush_interval_seconds: 30.0
  async_io:
    enabled: true
    worker_threads: 2
```

**Environment Variables:**
- `PLUME_RECORD_FULL`: Enable full trajectory recording
- `PLUME_BUFFER_SIZE`: Buffer size for batch writes
- `PLUME_PARQUET_COMPRESSION`: Compression algorithm

### Extension Hook Components (`conf/base/hooks/`)

#### No Hooks (Default)

```yaml
# none.yaml
_target_: plume_nav_sim.core.hooks.NullHookSystem

hooks:
  enabled: false
  pre_step:
    enabled: false
    hooks: []
  post_step:
    enabled: false
    hooks: []
  episode_end:
    enabled: false
    hooks: []
  extensions:
    extra_obs_fn: null
    extra_reward_fn: null
    extra_done_fn: null
```

#### Research Hooks

```yaml
# research.yaml
recorder:
  _target_: plume_nav_sim.recording.backends.ParquetRecorder
  full_trajectory: true
  performance_metrics: true

stats_aggregator:
  _target_: plume_nav_sim.analysis.stats.StatsAggregator
  metrics_definitions:
    trajectory: [mean, std, total_distance, displacement_efficiency]
    concentration: [mean, std, max, detection_rate]
    speed: [mean_speed, max_speed, movement_efficiency]

observation_hooks:
  enable_extra_observations: true
  extra_observations:
    wind_direction: true
    distance_to_boundaries: true
    concentration_history: true

reward_hooks:
  enable_extra_rewards: true
  extra_rewards:
    exploration_bonus:
      enabled: true
      weight: 0.1
```

## Composition Patterns

### Single Agent Basic Setup

```yaml
# Basic single-agent configuration
defaults:
  - base
  - source: point_source
  - agent_init: uniform_random
  - boundary: terminate
  - action: continuous2d
  - record: none
  - hooks: none

source:
  position: [50.0, 50.0]
  emission_rate: 1000.0

agent_init:
  bounds: [100.0, 100.0]
  seed: 42
```

### Multi-Agent Research Configuration

```yaml
# Comprehensive multi-agent research setup
defaults:
  - base
  - source: multi_source
  - agent_init: grid
  - boundary: bounce
  - action: continuous2d
  - record: parquet
  - hooks: research

source:
  positions: [[25.0, 25.0], [75.0, 25.0], [50.0, 75.0]]
  strengths: [800.0, 1200.0, 1000.0]

agent_init:
  grid_shape: [4, 4]
  spacing: 8.0
  center_position: [50.0, 30.0]

boundary:
  restitution_coefficient: 0.9

record:
  full_trajectory: true
  compression: gzip
```

### Performance Optimization Configuration

```yaml
# Maximum performance configuration
defaults:
  - base
  - source: point_source
  - agent_init: uniform_random
  - boundary: terminate
  - action: continuous2d
  - record: none
  - hooks: none

# Disable all non-essential features
source:
  performance:
    vectorized: true
    cache_emissions: true

record:
  backend: none  # Disable recording for maximum speed

hooks:
  enabled: false  # Disable all hooks
```

## Environment Variable Integration

### Comprehensive Variable Reference

The configuration system supports extensive environment variable integration for deployment flexibility:

#### Component Selection Variables

```bash
# V1.0 Component Selection
export SOURCE_TYPE="point_source"          # point_source, multi_source, dynamic_source
export AGENT_INIT_TYPE="uniform_random"    # uniform_random, grid, fixed_list, from_dataset
export BOUNDARY_TYPE="terminate"           # terminate, bounce, wrap, clip
export ACTION_TYPE="continuous2d"          # continuous2d, cardinal_discrete
export RECORD_BACKEND="parquet"            # parquet, hdf5, sqlite, none
export HOOKS_CONFIG="basic"                # basic, research, custom

# Legacy Component Selection (v0.3.0 compatibility)
export PLUME_MODEL_TYPE="gaussian"         # gaussian, turbulent, video_adapter
export WIND_FIELD_TYPE="constant"          # constant, turbulent, time_varying
export SENSOR_TYPE="concentration"         # binary, concentration, gradient
```

#### Component Parameter Variables

```bash
# Source Configuration
export SOURCE_POSITION="[50.0,50.0]"       # Point source coordinates
export SOURCE_STRENGTH="1000.0"            # Emission rate
export SOURCE_MOVEMENT="circular"          # Dynamic source movement type

# Agent Initialization
export GRID_SHAPE="[5,5]"                  # Grid dimensions
export GRID_SPACING="10.0"                 # Inter-agent spacing
export AGENT_INIT_SEED="42"                # Deterministic seed

# Boundary Policy
export BOUNDARY_REWARD="-10.0"             # Boundary violation penalty
export BOUNCE_RESTITUTION="0.8"            # Bounce energy retention

# Action Interface
export MAX_ACCELERATION="5.0"              # Maximum action magnitude
export NUM_ACTIONS="8"                     # Discrete action count

# Recording Configuration
export RECORD_ENABLED="true"               # Enable/disable recording
export RECORD_FULL="true"                  # Full trajectory recording
export PLUME_BUFFER_SIZE="1000"           # Recording buffer size
```

#### System Configuration Variables

```bash
# Performance and System
export NUMPY_THREADS="8"                   # NumPy optimization
export FRAME_CACHE_MODE="lru"             # Cache strategy
export LOG_LEVEL="INFO"                    # Logging verbosity

# Paths and Storage
export DATA_DIR="./data"                   # Base data directory
export OUTPUT_DIR="./outputs"              # Simulation output directory
export PARQUET_DIR="./data/parquet"       # Parquet-specific output
```

### Environment Variable Usage Patterns

#### Development Environment

```bash
# .env file for development
ENVIRONMENT_TYPE=development
DEBUG=true
SOURCE_TYPE=point_source
AGENT_INIT_TYPE=uniform_random
BOUNDARY_TYPE=terminate
RECORD_BACKEND=none
FRAME_CACHE_MODE=lru
LOG_LEVEL=DEBUG
```

#### Production Environment

```bash
# .env file for production
ENVIRONMENT_TYPE=production
DEBUG=false
SOURCE_TYPE=multi_source
AGENT_INIT_TYPE=grid
BOUNDARY_TYPE=bounce
RECORD_BACKEND=parquet
RECORD_FULL=true
FRAME_CACHE_MODE=all
LOG_LEVEL=WARNING
```

#### Research Configuration

```bash
# .env file for research
ENVIRONMENT_TYPE=research
SOURCE_TYPE=dynamic_source
AGENT_INIT_TYPE=grid
BOUNDARY_TYPE=bounce
ACTION_TYPE=continuous2d
RECORD_BACKEND=hdf5
HOOKS_CONFIG=research
PLUME_MODEL_TYPE=turbulent
SENSOR_TYPE=gradient
```

## Migration Guide (v0.3.0 → v1.0)

### Configuration Structure Changes

#### v0.3.0 Configuration

```yaml
# Old v0.3.0 style - monolithic configuration
navigator:
  type: single_agent
  max_speed: 1.5
  initial_position: [10.0, 10.0]

plume_model:
  type: gaussian
  source_position: [50.0, 50.0]
  source_strength: 1000.0

boundary_handling: "terminate"
action_processing: "continuous"
```

#### v1.0 Configuration

```yaml
# New v1.0 style - modular composition
defaults:
  - base
  - source: point_source
  - agent_init: uniform_random
  - boundary: terminate
  - action: continuous2d

# Component-specific configuration
source:
  position: [50.0, 50.0]
  emission_rate: 1000.0

agent_init:
  bounds: [100.0, 100.0]

navigator:
  max_speed: 1.5
```

### Migration Steps

1. **Split Monolithic Configuration**: Break large configuration blocks into component-specific sections
2. **Update Parameter Names**: Map old parameter names to new component structure
3. **Add Component Selection**: Use Hydra defaults to specify component implementations
4. **Environment Variable Migration**: Update environment variable names to match new conventions

### Backward Compatibility

The v1.0 system maintains full backward compatibility:

```yaml
# Legacy parameters are automatically mapped
# Old style still works
navigator:
  type: single_agent  # Mapped to appropriate agent_init strategy
  
# New style preferred
defaults:
  - agent_init: uniform_random
```

### Migration Utilities

```bash
# Migration script (planned)
python -m plume_nav_sim.migration.upgrade_config --input old_config.yaml --output new_config.yaml

# Validation script
python -m plume_nav_sim.migration.validate_config --config config.yaml --version v1.0
```

## Validation and Type Safety

### Pydantic Integration

The configuration system uses Pydantic for comprehensive validation:

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union

class PointSourceConfig(BaseModel):
    """Point source configuration schema with validation"""
    
    position: List[float] = Field(..., min_items=2, max_items=2)
    emission_rate: float = Field(gt=0.0, le=10000.0)
    detection_threshold: float = Field(ge=0.0, le=1.0)
    active: bool = True
    
    @validator('position')
    def validate_position(cls, v):
        if len(v) != 2:
            raise ValueError('Position must be [x, y] coordinates')
        if any(coord < 0 for coord in v):
            raise ValueError('Coordinates must be non-negative')
        return v

class BoundaryPolicyConfig(BaseModel):
    """Boundary policy configuration with constraints"""
    
    domain_bounds: List[float] = Field(..., min_items=2, max_items=2)
    status_on_violation: str = Field(min_length=1, max_length=50)
    
    @validator('domain_bounds')
    def validate_bounds(cls, v):
        if any(bound <= 0 for bound in v):
            raise ValueError('Domain bounds must be positive')
        return v
```

### Runtime Validation

```python
# Configuration validation at startup
def validate_configuration(config: DictConfig) -> None:
    """Validate complete configuration structure"""
    
    # Component compatibility checks
    validate_component_compatibility(config)
    
    # Parameter range validation
    validate_parameter_ranges(config)
    
    # Performance requirement validation
    validate_performance_targets(config)
    
    # Dependency availability checks
    validate_dependencies(config)

# Performance validation
def validate_performance_targets(config: DictConfig) -> None:
    """Ensure configuration meets performance requirements"""
    
    # Check buffer sizes don't exceed memory limits
    if config.record.buffer.max_memory_mb > 2048:
        warnings.warn("Buffer size may exceed performance targets")
    
    # Validate hook overhead targets
    if config.hooks.enabled and not config.hooks.performance.monitor_overhead:
        warnings.warn("Hook performance monitoring recommended")
```

### Schema Export

```bash
# Export configuration schemas for external validation
python -m plume_nav_sim.config.export_schemas --output schemas/

# Generate configuration templates
python -m plume_nav_sim.config.generate_templates --output templates/
```

## Performance Considerations

### Configuration Loading Performance

The modular configuration system is optimized for fast loading:

```python
# Lazy loading for large configurations
@lru_cache(maxsize=128)
def load_component_config(component_type: str, component_name: str):
    """Cache component configurations for reuse"""
    return load_config(f"base/{component_type}/{component_name}.yaml")

# Parallel configuration validation
async def validate_components_parallel(config: DictConfig):
    """Validate multiple components in parallel"""
    tasks = [
        validate_component(config.source),
        validate_component(config.agent_init),
        validate_component(config.boundary),
        validate_component(config.action),
        validate_component(config.record),
        validate_component(config.hooks)
    ]
    await asyncio.gather(*tasks)
```

### Runtime Configuration Updates

```python
# Hot configuration reloading (where supported)
def reload_component_config(component_name: str, new_config: Dict):
    """Reload component configuration without system restart"""
    
    # Validate new configuration
    validate_component_config(new_config)
    
    # Apply configuration updates
    component = get_component(component_name)
    component.update_config(new_config)
    
    # Log configuration change
    logger.info(f"Reloaded configuration for {component_name}")
```

### Memory Optimization

```yaml
# Memory-efficient configuration for large-scale simulations
performance:
  buffer:
    # Optimize buffer sizes for memory constraints
    max_records: 500           # Smaller buffer for memory efficiency
    flush_interval_seconds: 15  # More frequent flushes
  
  memory:
    max_usage_mb: 256          # Strict memory limit
    monitor_interval_seconds: 5 # Frequent monitoring
    auto_cleanup: true         # Aggressive cleanup
```

### Performance Monitoring

```python
# Configuration performance metrics
class ConfigPerformanceMonitor:
    def __init__(self):
        self.load_times = {}
        self.validation_times = {}
        self.memory_usage = {}
    
    def record_load_time(self, component: str, time: float):
        self.load_times[component] = time
    
    def record_validation_time(self, component: str, time: float):
        self.validation_times[component] = time
    
    def get_performance_report(self) -> Dict:
        return {
            "total_load_time": sum(self.load_times.values()),
            "total_validation_time": sum(self.validation_times.values()),
            "peak_memory_usage": max(self.memory_usage.values()),
            "component_breakdown": {
                "load_times": self.load_times,
                "validation_times": self.validation_times
            }
        }
```

---

This configuration system enables flexible, maintainable, and performant simulation setups while maintaining scientific reproducibility and supporting complex research workflows. The modular design allows researchers to focus on their specific requirements while leveraging the full power of the plume navigation simulation toolkit.