# Source System API Reference

This document provides comprehensive API reference for the pluggable odor source abstraction system in plume_nav_sim v1.0. The source system enables flexible odor source modeling through protocol-based architecture, supporting various source types and emission patterns for research into source configurations and dynamics.

## Table of Contents

- [System Overview](#system-overview)
- [SourceProtocol Interface](#sourceprotocol-interface)
- [Source Implementations](#source-implementations)
  - [PointSource](#pointsource)
  - [MultiSource](#multisource)
  - [DynamicSource](#dynamicsource)
- [Configuration System](#configuration-system)
- [Integration Patterns](#integration-patterns)
- [Performance Specifications](#performance-specifications)
- [Usage Examples](#usage-examples)

## System Overview

The source system implements Feature F-013 (Source Abstraction) through a protocol-based architecture that enables runtime switching between different source types without code changes. The system supports:

### Protocol-Based Architecture

- **SourceProtocol Interface**: Defines the contract for all source implementations
- **Flexible Source Modeling**: Support for point sources, multi-source scenarios, and dynamic temporal patterns
- **Vectorized Multi-Agent Support**: Efficient operations for scenarios with up to 100 concurrent agents
- **Configuration-Driven Selection**: Zero-code source type switching through Hydra configuration
- **Performance Specifications**: ≤33ms/step simulation target with comprehensive source modeling

### Key Features

- **Real-time Performance**: Source operations <1ms per query for minimal simulation overhead
- **Memory Efficiency**: <10MB for typical source configurations
- **Deterministic Behavior**: Reproducible source patterns with configurable seeding
- **Integration Ready**: Seamless integration with plume models via dependency injection

## SourceProtocol Interface

The `SourceProtocol` defines the fundamental interface that all source implementations must provide:

```python
from typing import Protocol, Tuple

@runtime_checkable
class SourceProtocol(Protocol):
    """Protocol interface for pluggable odor source modeling."""
    
    def get_emission_rate(self) -> float:
        """Get current odor emission rate from this source."""
        ...
    
    def get_position(self) -> Tuple[float, float]:
        """Get current source position coordinates."""
        ...
    
    def update_state(self, dt: float = 1.0) -> None:
        """Advance source state by specified time delta."""
        ...
```

### Method Specifications

#### `get_emission_rate() -> float`

Returns the current emission strength in source-specific units (typically molecules/second or concentration units/second).

**Performance Requirements:**
- Execution time: <0.1ms for real-time simulation compatibility
- Return value: Non-negative float representing current source strength
- Thread safety: Must support concurrent access from multiple agents

**Behavior:**
- For static sources: Returns constant emission rate
- For dynamic sources: Returns time-varying emission based on internal patterns
- For multi-source implementations: Returns aggregate emission across all sub-sources

#### `get_position() -> Tuple[float, float]`

Returns the current source location as (x, y) coordinates in the environment coordinate system.

**Performance Requirements:**
- Execution time: <0.1ms for minimal spatial query overhead
- Return format: Tuple[float, float] representing (x, y) coordinates
- Coordinate system: Environment units (typically pixels for video-based environments)

**Behavior:**
- Static sources return fixed position
- Dynamic sources return current position after temporal evolution
- Multi-source implementations return centroid or primary source position

#### `update_state(dt: float = 1.0) -> None`

Advances the source state by the specified time step, updating internal dynamics.

**Performance Requirements:**
- Execution time: <1ms per step for real-time simulation compatibility
- Thread safety: Must handle concurrent updates appropriately
- State consistency: Maintain internal state coherency across updates

**Parameters:**
- `dt`: Time step size in seconds (controls temporal resolution)

**Behavior:**
- Updates emission rate variations based on temporal patterns
- Advances position for mobile sources
- Evolves internal parameters for complex source dynamics

## Source Implementations

### PointSource

Single-point odor source implementation with configurable emission rates and optional temporal variation.

```python
from plume_nav_sim.core.sources import PointSource
import numpy as np

class PointSource:
    """Single-point odor source with configurable emission rates."""
    
    def __init__(
        self,
        position: Union[Tuple[float, float], List[float], np.ndarray] = (0.0, 0.0),
        emission_rate: Union[float, int] = 1.0,
        seed: Optional[int] = None,
        enable_temporal_variation: bool = False
    ):
        """Initialize point source with specified parameters."""
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `position` | `PositionType` | `(0.0, 0.0)` | Source location as (x, y) coordinates |
| `emission_rate` | `EmissionRateType` | `1.0` | Base emission strength (arbitrary units) |
| `seed` | `Optional[int]` | `None` | Random seed for deterministic behavior |
| `enable_temporal_variation` | `bool` | `False` | Enable time-varying emission patterns |

#### Key Methods

##### `get_emission_rate(agent_positions: Optional[np.ndarray] = None) -> Union[float, np.ndarray]`

Supports both scalar and vectorized queries for multi-agent scenarios.

**Input Formats:**
- `None`: Returns scalar emission rate
- `(2,)` shape: Single agent position, returns scalar
- `(n_agents, 2)` shape: Multi-agent positions, returns array of shape `(n_agents,)`

**Performance:**
- Single query: <0.1ms
- 100 agents: <1ms through efficient numpy broadcasting

**Example:**
```python
# Scalar query
source = PointSource(position=(50.0, 50.0), emission_rate=1000.0)
rate = source.get_emission_rate()  # Returns 1000.0

# Multi-agent vectorized query
positions = np.array([[45, 48], [52, 47], [60, 55]])
rates = source.get_emission_rate(positions)  # Returns [1000.0, 1000.0, 1000.0]
```

##### `configure(**kwargs: Any) -> None`

Runtime configuration updates without requiring new instantiation.

**Supported Parameters:**
- `position`: New source position (x, y)
- `emission_rate`: New emission rate  
- `enable_temporal_variation`: Enable/disable temporal variation

**Example:**
```python
source.configure(position=(75.0, 80.0), emission_rate=2000.0)
```

#### Performance Characteristics

- **get_emission_rate()**: <0.1ms for single query, <1ms for 100 agents
- **get_position()**: O(1) property access with no computation
- **update_state()**: <0.05ms for simple emission rate updates
- **Memory usage**: <1KB per source instance

### MultiSource

Container for multiple simultaneous odor sources with vectorized operations.

```python
from plume_nav_sim.core.sources import MultiSource

class MultiSource:
    """Multiple simultaneous odor sources with vectorized operations."""
    
    def __init__(
        self, 
        sources: Optional[List[Any]] = None, 
        seed: Optional[int] = None
    ):
        """Initialize multi-source container."""
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sources` | `Optional[List[Any]]` | `None` | Optional list of source instances to initialize with |
| `seed` | `Optional[int]` | `None` | Random seed for deterministic behavior across all sources |

#### Key Methods

##### `get_emission_rate(agent_positions: Optional[np.ndarray] = None) -> Union[float, np.ndarray]`

Computes the sum of emission rates from all contained sources.

**Aggregation Behavior:**
- Sums emission rates across all active sources for each agent position
- Uses vectorized operations for efficient computation
- Handles empty source collections (returns zero emission)

**Performance:**
- 10 sources with 100 agents: <2ms through vectorized summation

**Example:**
```python
multi_source = MultiSource()
multi_source.add_source(PointSource((30, 30), emission_rate=500))
multi_source.add_source(PointSource((70, 70), emission_rate=800))

positions = np.array([[30, 30], [50, 50], [70, 70]])
total_rates = multi_source.get_emission_rate(positions)  # Sums across sources
```

##### `add_source(source: Any) -> None`

Adds a new source to the collection with interface validation.

**Validation:**
- Checks for required methods: `get_emission_rate()`, `get_position()`, `update_state()`
- Ensures methods are callable
- Raises `TypeError` for non-compliant sources

**Example:**
```python
# Add point source
point_source = PointSource((60, 40), emission_rate=750)
multi_source.add_source(point_source)

# Add dynamic source
dynamic_source = DynamicSource((80, 20), pattern_type="linear")
multi_source.add_source(dynamic_source)
```

##### `remove_source(index: int) -> None`

Removes source from collection by index.

**Example:**
```python
multi_source.remove_source(0)  # Remove first source
multi_source.remove_source(-1)  # Remove last source
```

#### Management Methods

- `get_sources() -> List[Any]`: Get copy of all sources in collection
- `get_source_count() -> int`: Get number of active sources
- `clear_sources() -> None`: Remove all sources from collection

#### Performance Characteristics

- **get_emission_rate()**: <2ms for 10 sources with 100 agents
- **add_source()/remove_source()**: <0.1ms for source management
- **update_state()**: <1ms for temporal updates across all sources
- **Memory usage**: <1KB base + linear scaling with source count

### DynamicSource

Time-varying source with configurable movement patterns and emission dynamics.

```python
from plume_nav_sim.core.sources import DynamicSource

class DynamicSource:
    """Time-varying source with configurable movement patterns."""
    
    def __init__(
        self,
        initial_position: PositionType = (0.0, 0.0),
        emission_rate: EmissionRateType = 1.0,
        pattern_type: str = "stationary",
        amplitude: float = 0.0,
        frequency: float = 0.0,
        velocity: Tuple[float, float] = (0.0, 0.0),
        noise_std: float = 0.0,
        seed: Optional[int] = None
    ):
        """Initialize dynamic source with movement and emission patterns."""
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_position` | `PositionType` | `(0.0, 0.0)` | Starting position as (x, y) coordinates |
| `emission_rate` | `EmissionRateType` | `1.0` | Base emission strength |
| `pattern_type` | `str` | `"stationary"` | Movement pattern type |
| `amplitude` | `float` | `0.0` | Movement amplitude for oscillatory patterns |
| `frequency` | `float` | `0.0` | Movement frequency in Hz for periodic patterns |
| `velocity` | `Tuple[float, float]` | `(0.0, 0.0)` | Velocity vector (vx, vy) for linear motion |
| `noise_std` | `float` | `0.0` | Standard deviation for random walk noise |
| `seed` | `Optional[int]` | `None` | Random seed for deterministic behavior |

#### Movement Patterns

##### "stationary"
- No movement - position remains at initial value
- Use for static sources with potential emission variation

##### "linear"  
- Constant velocity linear motion
- Position updated as: `position += velocity * dt`

**Example:**
```python
linear_source = DynamicSource(
    initial_position=(10, 50),
    pattern_type="linear", 
    velocity=(2.0, 0.0)  # 2 units/second eastward
)
```

##### "circular"
- Circular orbit around initial position
- Position follows: `position = initial_pos + amplitude * [cos(2π*freq*t), sin(2π*freq*t)]`

**Example:**
```python
circular_source = DynamicSource(
    initial_position=(50, 50),
    pattern_type="circular",
    amplitude=20.0,  # 20-unit radius
    frequency=0.05   # Complete orbit every 20 seconds
)
```

##### "sinusoidal"
- Sinusoidal oscillation along x-axis
- Position follows: `x = initial_x + amplitude * sin(2π*freq*t)`

**Example:**
```python
oscillating_source = DynamicSource(
    initial_position=(50, 50),
    pattern_type="sinusoidal",
    amplitude=15.0,   # ±15 unit oscillation
    frequency=0.1     # 10-second period
)
```

##### "random_walk"
- Brownian motion with Gaussian noise
- Position updated as: `position += normal(0, noise_std) * sqrt(dt)`

**Example:**
```python
random_source = DynamicSource(
    initial_position=(50, 50),
    pattern_type="random_walk",
    noise_std=1.0,  # Standard deviation of steps
    seed=42         # For reproducible random walks
)
```

#### Advanced Features

##### `set_trajectory(trajectory_points: List[Tuple[float, float]], timestamps: Optional[List[float]] = None) -> None`

Enables predefined trajectory patterns defined by waypoints.

**Example:**
```python
# Waypoint trajectory
waypoints = [(0, 0), (50, 25), (100, 50), (50, 75)]
dynamic_source.set_trajectory(waypoints)

# Timed trajectory with custom timestamps
waypoints = [(0, 0), (50, 50), (100, 0)]
times = [0.0, 5.0, 10.0]
dynamic_source.set_trajectory(waypoints, times)
```

##### `set_emission_pattern(pattern_func: Callable[[float], float]) -> None`

Enables custom temporal emission patterns.

**Example:**
```python
# Pulsed emission pattern
def pulse_pattern(t):
    return 1.0 if int(t) % 2 == 0 else 0.5

dynamic_source.set_emission_pattern(pulse_pattern)

# Exponential decay
import math
def decay_pattern(t):
    return math.exp(-0.1 * t)

dynamic_source.set_emission_pattern(decay_pattern)
```

#### Performance Characteristics

- **get_emission_rate()**: <0.2ms including pattern calculations
- **get_position()**: <0.1ms for position updates
- **update_state()**: <0.5ms for full temporal evolution
- **Memory usage**: <2KB per source with pattern state

## Configuration System

The source system integrates with Hydra for configuration-driven source selection and parameterization.

### Configuration Structure

Source configurations are organized under `conf/base/source/` with YAML files for each source type:

```yaml
# conf/base/source/point_source.yaml
_target_: plume_nav_sim.core.sources.PointSource
position: [50.0, 50.0]
emission_rate: 1000.0
enable_temporal_variation: false
```

### Point Source Configuration

The `point_source.yaml` configuration provides comprehensive parameterization:

```yaml
# Core PointSource implementation
_target_: plume_nav_sim.core.sources.PointSource

# Source position with environment variable override
position: ${oc.env:SOURCE_POSITION,[10.0, 10.0]}

# Emission characteristics
emission_rate: ${oc.env:SOURCE_STRENGTH,1.0}
detection_threshold: 0.01
max_detection_distance: 50.0
active: true

# Emission profile parameters
emission_profile:
  type: constant                    # constant, pulsed, or variable
  pulse_duration: 1.0              # For pulsed emission
  pulse_interval: 5.0
  variation_amplitude: 0.1         # For variable emission
  variation_frequency: 0.1

# Source geometry
geometry:
  radius: 0.0                      # 0.0 for true point source
  height: 0.0                      # For 3D visualization

# Detection and sensing parameters
detection:
  noise_level: 0.05               # Measurement noise (0.0 = perfect)
  response_time: 0.1              # Sensor response time constant
  binary_detection: false         # Binary vs continuous detection

# Performance optimization
performance:
  update_frequency: 100.0         # State update frequency (Hz)
  vectorized: true                # Enable vectorized computation
  cache_emissions: true           # Cache calculations for performance

# Validation bounds
validation:
  enforce_domain_bounds: true
  min_emission_rate: 0.0
  max_emission_rate: 10.0
  position_bounds: [0.0, 0.0, 100.0, 100.0]
```

### Multi-Source Configuration

```yaml
# conf/base/source/multi_source.yaml
_target_: plume_nav_sim.core.sources.MultiSource
sources:
  - _target_: plume_nav_sim.core.sources.PointSource
    position: [30.0, 30.0]
    emission_rate: 500.0
  - _target_: plume_nav_sim.core.sources.PointSource  
    position: [70.0, 70.0]
    emission_rate: 800.0
seed: 42
```

### Dynamic Source Configuration

```yaml
# conf/base/source/dynamic_source.yaml
_target_: plume_nav_sim.core.sources.DynamicSource
initial_position: [25.0, 75.0]
emission_rate: 1200.0
pattern_type: circular
amplitude: 15.0
frequency: 0.1
seed: 123
```

### Environment Variable Overrides

Configurations support environment variable overrides for deployment flexibility:

```bash
# Override source position
export SOURCE_POSITION="[75.0, 25.0]"

# Override emission strength  
export SOURCE_STRENGTH=2000.0

# Run with overrides
python main.py
```

### Hydra Composition Patterns

#### Override Source Type

```bash
# Use different source type
python main.py source=multi_source

# Use dynamic source
python main.py source=dynamic_source
```

#### Runtime Parameter Modification

```bash
# Modify parameters via CLI
python main.py source.emission_rate=1500.0 source.position=[60,40]

# Multiple parameter overrides
python main.py source=dynamic_source source.pattern_type=linear source.velocity=[1.5,0.5]
```

## Integration Patterns

### Plume Model Integration

Sources integrate with plume models via dependency injection:

```python
from hydra import initialize, compose
from plume_nav_sim.core.sources import create_source

# Configuration-driven source creation
with initialize(config_path="../conf"):
    cfg = compose(config_name="config")
    source = create_source(cfg.source)

# Plume model integration
plume_model = GaussianPlumeModel(source=source)
plume_model.step(dt=1.0)  # Source automatically updates
```

### Environment Initialization

Sources participate in environment reset and initialization:

```python
from plume_nav_sim.envs import PlumeNavigationEnv

class PlumeNavigationEnv:
    def __init__(self, source_config):
        self.source = create_source(source_config)
        
    def reset(self):
        # Reset source state for new episode
        if hasattr(self.source, 'reset_time'):
            self.source.reset_time()
        return self._get_observation()
        
    def step(self, action):
        # Update source state each step
        self.source.update_state(dt=self.dt)
        # Use source data for plume modeling
        emission_rate = self.source.get_emission_rate()
        source_position = self.source.get_position()
```

### Factory Function Usage

The `create_source()` factory function enables runtime source selection:

```python
from plume_nav_sim.core.sources import create_source

# Point source from config
point_config = {
    'type': 'PointSource',
    'position': (50.0, 50.0), 
    'emission_rate': 1000.0
}
point_source = create_source(point_config)

# Dynamic source from config
dynamic_config = {
    'type': 'DynamicSource',
    'initial_position': (25, 75),
    'pattern_type': 'circular',
    'amplitude': 15.0,
    'frequency': 0.1
}
dynamic_source = create_source(dynamic_config)

# Multi-source from config with nested sources
multi_config = {
    'type': 'MultiSource',
    'sources': [
        {'type': 'PointSource', 'position': (30, 30), 'emission_rate': 500},
        {'type': 'PointSource', 'position': (70, 70), 'emission_rate': 800}
    ]
}
multi_source = create_source(multi_config)
```

### Protocol Compliance Validation

```python
from plume_nav_sim.core.protocols import SourceProtocol

# Validate source implements protocol
def validate_source(source):
    if not isinstance(source, SourceProtocol):
        raise TypeError("Source must implement SourceProtocol")
    
    # Test required methods
    emission_rate = source.get_emission_rate()
    position = source.get_position()
    source.update_state(dt=1.0)
    
    return True

# Usage
source = create_source(config)
validate_source(source)  # Ensures protocol compliance
```

## Performance Specifications

The source system is designed to meet strict performance requirements for real-time simulation:

### Timing Requirements

| Operation | Target | Multi-Agent (100 agents) |
|-----------|--------|--------------------------|
| `get_emission_rate()` | <0.1ms | <1ms |
| `get_position()` | <0.1ms | N/A (scalar) |
| `update_state()` | <1ms | <1ms per source |
| Source creation | <5ms | <5ms |
| Configuration loading | <10ms | <10ms |

### Memory Requirements

| Component | Memory Usage |
|-----------|-------------|
| PointSource instance | <1KB |
| MultiSource base | <1KB + linear scaling |
| DynamicSource instance | <2KB |
| Configuration cache | <100KB |
| Total system overhead | <10MB |

### Scalability Characteristics

#### Multi-Agent Performance
- Linear scaling with agent count for emission queries
- Vectorized operations prevent performance degradation
- Efficient numpy broadcasting for position-based calculations

#### Multi-Source Performance  
- Linear scaling with source count in MultiSource
- Independent source updates allow parallelization
- Aggregate operations optimized for minimal overhead

### Performance Monitoring

Source implementations include built-in performance tracking:

```python
# Get performance statistics
stats = source.get_performance_stats()
print(f"Average query time: {stats['avg_query_time']:.6f}s")
print(f"Total queries: {stats['query_count']}")

# Multi-source statistics
multi_stats = multi_source.get_performance_stats()
print(f"Source count: {multi_stats['source_count']}")
print(f"Average multi-source query: {multi_stats['avg_query_time']:.6f}s")
```

## Usage Examples

### Basic Source Creation

```python
from plume_nav_sim.core.sources import PointSource, MultiSource, DynamicSource
import numpy as np

# Create simple point source
point_source = PointSource(
    position=(50.0, 50.0),
    emission_rate=1000.0
)

# Query emission rate
rate = point_source.get_emission_rate()
position = point_source.get_position()

# Multi-agent query
agent_positions = np.array([[45, 48], [52, 47], [60, 55]])
rates = point_source.get_emission_rate(agent_positions)
```

### Multi-Source Scenarios

```python
# Create multi-source environment
multi_source = MultiSource()

# Add various source types
multi_source.add_source(PointSource((30, 30), emission_rate=500))
multi_source.add_source(PointSource((70, 70), emission_rate=800))

# Add dynamic source
dynamic_source = DynamicSource(
    initial_position=(50, 10),
    pattern_type="linear",
    velocity=(0.0, 1.0)  # Moving north
)
multi_source.add_source(dynamic_source)

# Query total emission across all sources
total_emission = multi_source.get_emission_rate(agent_positions)
```

### Dynamic Source Patterns

```python
# Circular orbit source
circular_source = DynamicSource(
    initial_position=(50, 50),
    pattern_type="circular",
    amplitude=20.0,
    frequency=0.05  # 20-second orbit
)

# Random walk source
random_source = DynamicSource(
    initial_position=(25, 25),
    pattern_type="random_walk", 
    noise_std=2.0,
    seed=42  # Reproducible randomness
)

# Simulate temporal evolution
for t in range(100):
    circular_source.update_state(dt=1.0)
    random_source.update_state(dt=1.0)
    
    # Get current positions
    circular_pos = circular_source.get_position()
    random_pos = random_source.get_position()
    
    print(f"t={t}: Circular={circular_pos}, Random={random_pos}")
```

### Custom Emission Patterns

```python
import math

# Create dynamic source with custom emission pattern
dynamic_source = DynamicSource(
    initial_position=(50, 50),
    emission_rate=1000.0
)

# Define pulsed emission pattern
def pulsed_emission(t):
    """Pulsed emission: on for 2s, off for 3s, repeat."""
    cycle_time = t % 5.0  # 5-second cycle
    return 1.0 if cycle_time < 2.0 else 0.0

# Apply custom pattern
dynamic_source.set_emission_pattern(pulsed_emission)

# Simulate pulsed behavior
for t in range(20):
    dynamic_source.update_state(dt=1.0)
    emission = dynamic_source.get_emission_rate()
    print(f"t={t}s: emission_rate={emission}")
```

### Configuration-Driven Source Creation

```python
from plume_nav_sim.core.sources import create_source

# Point source configuration
point_config = {
    'type': 'PointSource',
    'position': (60.0, 40.0),
    'emission_rate': 1500.0,
    'enable_temporal_variation': True
}

# Create from configuration
source = create_source(point_config)

# Hydra integration example
from hydra import initialize, compose

with initialize(config_path="../conf"):
    cfg = compose(config_name="config")
    # Override source type
    cfg.source = point_config
    source = create_source(cfg.source)
```

### Integration with Simulation Loop

```python
# Complete simulation integration example
class SourceIntegratedSimulation:
    def __init__(self, source_config):
        self.source = create_source(source_config)
        self.time = 0.0
        
    def step(self, dt=1.0):
        # Update source state
        self.source.update_state(dt)
        self.time += dt
        
        # Sample source for agent positions
        agent_positions = self.get_agent_positions()
        emission_rates = self.source.get_emission_rate(agent_positions)
        
        # Use emission data for plume modeling
        self.update_plume_model(emission_rates)
        
        return {
            'source_position': self.source.get_position(),
            'emission_rates': emission_rates,
            'simulation_time': self.time
        }

# Usage
config = {
    'type': 'DynamicSource',
    'initial_position': (25, 75),
    'pattern_type': 'circular',
    'amplitude': 10.0,
    'frequency': 0.1
}

sim = SourceIntegratedSimulation(config)
for step in range(1000):
    results = sim.step(dt=0.1)  # 10Hz simulation
```

This comprehensive API reference provides complete documentation for the pluggable source system, enabling researchers to effectively utilize the flexible source modeling capabilities for odor plume navigation research.