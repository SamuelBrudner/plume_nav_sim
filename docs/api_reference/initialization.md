# Agent Initialization API Reference

## Table of Contents

1. [System Overview](#system-overview)
2. [Protocol Reference](#protocol-reference)
3. [Initialization Strategies](#initialization-strategies)
4. [Configuration Examples](#configuration-examples)
5. [Integration Patterns](#integration-patterns)
6. [Advanced Usage](#advanced-usage)

---

## System Overview

The Agent Initialization system provides configurable strategies for generating diverse experimental setups through the `AgentInitializerProtocol` interface. This protocol-based architecture enables zero-code extensibility and deterministic seeding for reproducible experiments, supporting the v1.0 transformation from project-specific implementations to a general-purpose simulation toolkit.

### Protocol-Based Architecture

The initialization framework follows strict protocol interfaces for:
- **Strategy abstraction**: Uniform interface across all initialization patterns
- **Deterministic seeding**: Reproducible position generation for scientific rigor
- **Multi-agent support**: Efficient vectorized operations for large-scale scenarios
- **Domain validation**: Automatic constraint checking and compliance verification
- **Hydra integration**: Configuration-driven strategy selection without code changes

```python
from plume_nav_sim.core.initialization import create_agent_initializer
from plume_nav_sim.core.protocols import AgentInitializerProtocol

# Factory-based creation with automatic protocol compliance
config = {'type': 'uniform_random', 'bounds': (100, 100), 'seed': 42}
initializer: AgentInitializerProtocol = create_agent_initializer(config)

# Generate deterministic positions
positions = initializer.initialize_positions(num_agents=50)
print(f"Generated {len(positions)} positions with shape {positions.shape}")
```

### Configurable Strategies

Four core strategies enable diverse experimental configurations:

| Strategy | Purpose | Use Cases | Performance |
|----------|---------|-----------|-------------|
| **uniform_random** | Random spatial distribution | Unbiased experimental setups, baseline conditions | <1ms for 100 agents |
| **grid** | Systematic spatial coverage | Controlled studies, parameter sweeps | <1ms for 100 agents |
| **fixed_list** | Precise position control | Reproducible tests, specific scenarios | <0.5ms for predefined lists |
| **from_dataset** | Data-driven initialization | Comparative studies, real experiment replication | <1ms after dataset loading |

### Deterministic Seeding

All strategies support deterministic seeding for scientific reproducibility:

```python
# Identical results across runs with same seed
initializer.reset(seed=12345)
positions_1 = initializer.initialize_positions(num_agents=25)

initializer.reset(seed=12345) 
positions_2 = initializer.initialize_positions(num_agents=25)

assert np.array_equal(positions_1, positions_2)  # Guaranteed identical
```

### Multi-Agent Support

Vectorized operations enable efficient large-scale initialization:

```python
# Efficient batch generation for research scenarios
large_scale_positions = initializer.initialize_positions(num_agents=1000)
print(f"Generated {len(large_scale_positions)} positions in <5ms")

# Automatic domain validation for all positions
is_valid = initializer.validate_domain(large_scale_positions)
assert is_valid, "All positions must be within domain bounds"
```

### Experimental Reproducibility

The initialization system provides complete experimental reproducibility through:
- **Immutable configuration snapshots**: Configuration parameters preserved with generated positions
- **Deterministic random number generation**: Identical results with same seed across platforms
- **Version-controlled strategy implementations**: Consistent behavior across different software versions
- **Domain constraint validation**: Automatic verification of spatial requirements and boundaries

---

## Protocol Reference

### AgentInitializerProtocol Interface

The `AgentInitializerProtocol` defines the standard interface that all initialization strategies must implement, ensuring consistent behavior and type safety across the framework.

#### Method Signatures

```python
from typing import Protocol, Optional, Any
import numpy as np

@runtime_checkable
class AgentInitializerProtocol(Protocol):
    """Protocol for configurable agent initialization strategies."""
    
    def initialize_positions(
        self, 
        num_agents: int,
        **kwargs: Any
    ) -> np.ndarray:
        """Generate initial agent positions."""
        ...
    
    def validate_domain(
        self, 
        positions: np.ndarray,
        domain_bounds: Tuple[float, float]
    ) -> bool:
        """Validate position compliance with domain constraints."""
        ...
    
    def reset(self, seed: Optional[int] = None, **kwargs: Any) -> None:
        """Reset initializer state for deterministic behavior."""
        ...
    
    def get_strategy_name(self) -> str:
        """Get human-readable strategy identifier."""
        ...
```

#### initialize_positions()

Generates initial agent positions based on the configured strategy.

**Parameters:**
- `num_agents` (int): Number of agent positions to generate (must be positive)
- `**kwargs` (Any): Additional strategy-specific parameters

**Returns:**
- `np.ndarray`: Agent positions with shape `(num_agents, 2)` containing `[x, y]` coordinates

**Performance Requirements:**
- Execution time: <5ms for 100 agents
- Memory usage: O(n) scaling with agent count
- Deterministic behavior when seeded

**Example Usage:**
```python
# Basic position generation
positions = initializer.initialize_positions(num_agents=25)
assert positions.shape == (25, 2)
assert positions.dtype == np.float32

# Strategy-specific parameters via kwargs
grid_positions = grid_initializer.initialize_positions(
    num_agents=16,
    grid_shape=(4, 4),
    jitter_enabled=True
)
```

#### validate_domain()

Validates that generated positions comply with domain constraints and strategy requirements.

**Parameters:**
- `positions` (np.ndarray): Agent positions to validate with shape `(n_agents, 2)`
- `domain_bounds` (Tuple[float, float]): Spatial domain limits as `(width, height)`

**Returns:**
- `bool`: True if all positions are valid and compliant, False otherwise

**Validation Criteria:**
- Boundary checking for domain compliance
- Strategy-specific constraint verification
- Spatial requirement satisfaction

**Example Usage:**
```python
# Validate generated positions
domain_bounds = (100.0, 100.0)
positions = initializer.initialize_positions(num_agents=50)
is_valid = initializer.validate_domain(positions, domain_bounds)

if not is_valid:
    print("Warning: Some positions violate domain constraints")
    
# Batch validation for multiple position sets
all_valid = all(
    initializer.validate_domain(pos_set, domain_bounds)
    for pos_set in position_sets
)
```

#### reset()

Resets initializer state with optional deterministic seeding for reproducible experiments.

**Parameters:**
- `seed` (Optional[int]): Random seed for deterministic behavior
- `**kwargs` (Any): Additional reset parameters for strategy-specific state

**Performance Requirements:**
- Execution time: <1ms for state reset
- Deterministic reproduction with identical seeds

**Example Usage:**
```python
# Deterministic reset for reproducible experiments
initializer.reset(seed=42)
positions_1 = initializer.initialize_positions(num_agents=10)

initializer.reset(seed=42)
positions_2 = initializer.initialize_positions(num_agents=10)
assert np.allclose(positions_1, positions_2)

# Strategy-specific reset parameters
grid_initializer.reset(
    seed=123,
    grid_origin=(10, 10),
    grid_spacing=(15.0, 15.0)
)
```

#### get_strategy_name()

Returns a human-readable strategy identifier for experimental tracking and logging.

**Returns:**
- `str`: Strategy name for documentation and analysis

**Common Strategy Names:**
- `"uniform_random"`: Uniform spatial distribution
- `"grid"`: Regular grid pattern
- `"fixed_list"`: Predetermined position list
- `"from_dataset"`: External dataset loading

**Example Usage:**
```python
# Strategy identification for experimental logging
strategy_name = initializer.get_strategy_name()
experiment_log = {
    'initialization_strategy': strategy_name,
    'timestamp': datetime.now(),
    'agent_count': num_agents
}

# Runtime strategy switching validation
if initializer.get_strategy_name() == "uniform_random":
    # Apply uniform random specific validation
    assert hasattr(initializer, 'bounds')
    assert hasattr(initializer, 'seed')
```

### Implementation Requirements

All implementations must satisfy:

**Type Safety:**
- Implement all protocol methods with exact signatures
- Return appropriate numpy array types (`np.float32` recommended)
- Handle edge cases gracefully with proper error messages

**Performance Compliance:**
- Initialize 100 agents in <5ms (vectorized operations required)
- Domain validation in <1ms (efficient boundary checking)
- State reset in <1ms (minimal computational overhead)

**Reproducibility Guarantees:**
- Identical results with same seed across multiple runs
- Platform-independent deterministic behavior
- Immutable configuration snapshot preservation

---

## Initialization Strategies

### UniformRandomInitializer Reference

Provides uniform random agent placement within rectangular domain boundaries with deterministic seeding for reproducible experimental setups.

#### Constructor Parameters

```python
UniformRandomInitializer(
    bounds: Tuple[float, float] = (100.0, 100.0),
    seed: Optional[int] = None,
    margin: float = 0.0
)
```

**Parameters:**
- `bounds`: Domain dimensions as `(width, height)` tuple in simulation units
- `seed`: Random seed for deterministic behavior (None for non-deterministic)
- `margin`: Safety margin from domain edges in simulation units

#### Key Features

- **Uniform Distribution**: Even spatial probability across entire domain
- **Margin Support**: Configurable buffer zones from domain boundaries
- **Vectorized Generation**: Efficient NumPy operations for large agent populations
- **Boundary Validation**: Automatic position constraint checking

#### Usage Examples

```python
from plume_nav_sim.core.initialization import UniformRandomInitializer

# Basic uniform random initialization
initializer = UniformRandomInitializer(bounds=(100, 100))
positions = initializer.initialize_positions(num_agents=50)

# Deterministic initialization with seed
seeded_initializer = UniformRandomInitializer(
    bounds=(200, 150), 
    seed=42, 
    margin=5.0
)
deterministic_positions = seeded_initializer.initialize_positions(num_agents=25)

# Verify margin compliance
effective_bounds = (200 - 2*5.0, 150 - 2*5.0)  # Account for margin
assert np.all(deterministic_positions >= 5.0)  # Minimum margin
assert np.all(deterministic_positions[:, 0] <= 195.0)  # Width margin
assert np.all(deterministic_positions[:, 1] <= 145.0)  # Height margin
```

#### Performance Characteristics

- **Time Complexity**: O(n) where n is number of agents
- **Memory Complexity**: O(n) for position storage
- **Target Latency**: <1ms for 100 agents
- **Scaling**: Linear performance up to 1000+ agents

### GridInitializer Reference

Implements systematic grid-based agent positioning with configurable spacing, orientation, jitter, and boundary handling for controlled experimental conditions.

#### Constructor Parameters

```python
GridInitializer(
    domain_bounds: Tuple[float, float] = (100.0, 100.0),
    grid_spacing: Tuple[float, float] = (10.0, 10.0),
    grid_shape: Tuple[int, int] = (5, 5),
    orientation: float = 0.0,
    jitter_enabled: bool = False,
    jitter_std: float = 0.5,
    boundary_handling: Dict[str, Any] = None,
    seed: Optional[int] = None
)
```

**Parameters:**
- `domain_bounds`: Domain dimensions as `(width, height)` tuple
- `grid_spacing`: Grid spacing as `(dx, dy)` tuple in simulation units
- `grid_shape`: Grid dimensions as `(cols, rows)` tuple
- `orientation`: Grid rotation angle in radians (0 = axis-aligned)
- `jitter_enabled`: Enable small random perturbations from grid positions
- `jitter_std`: Standard deviation of jitter noise in domain units
- `boundary_handling`: Dictionary with boundary strategy configuration
- `seed`: Random seed for jitter generation

#### Grid Generation Patterns

**Square Grid (Default):**
```python
# 5x5 square grid with automatic spacing
grid_initializer = GridInitializer(
    domain_bounds=(100, 100),
    grid_shape=(5, 5)
)
positions = grid_initializer.initialize_positions(num_agents=25)
```

**Rectangular Grid:**
```python
# 8x4 rectangular grid with custom spacing
rect_initializer = GridInitializer(
    domain_bounds=(160, 80),
    grid_shape=(8, 4),
    grid_spacing=(20.0, 20.0)
)
positions = rect_initializer.initialize_positions(num_agents=32)
```

**Rotated Grid:**
```python
# Grid rotated 45 degrees with jitter
rotated_initializer = GridInitializer(
    domain_bounds=(100, 100),
    orientation=np.pi/4,  # 45 degrees
    jitter_enabled=True,
    jitter_std=1.0,
    seed=42
)
positions = rotated_initializer.initialize_positions(num_agents=16)
```

#### Boundary Handling Strategies

The grid initializer supports multiple boundary handling strategies when grid positions exceed domain limits:

**Clip Strategy (Default):**
```python
boundary_handling = {
    'strategy': 'clip',
    'preserve_shape': True,
    'margin': 1.0
}
```
- Clips positions to domain bounds with margin
- Preserves relative grid structure
- Safe for all domain/grid combinations

**Scale Strategy:**
```python
boundary_handling = {
    'strategy': 'scale',
    'preserve_shape': True,
    'margin': 2.0
}
```
- Scales entire grid to fit within domain bounds
- Maintains grid proportions when `preserve_shape=True`
- Optimal spacing for complex grid configurations

**Wrap Strategy:**
```python
boundary_handling = {
    'strategy': 'wrap',
    'margin': 0.0
}
```
- Wraps positions around domain boundaries (periodic)
- Useful for toroidal domain topologies
- Maintains exact grid spacing

**Error Strategy:**
```python
boundary_handling = {
    'strategy': 'error',
    'margin': 5.0
}
```
- Raises exception for boundary violations
- Strict constraint enforcement
- Development and validation scenarios

#### Natural Variation with Jitter

```python
# Grid with natural positioning variation
natural_grid = GridInitializer(
    domain_bounds=(100, 100),
    grid_shape=(6, 6),
    jitter_enabled=True,
    jitter_std=2.0,  # 2-unit standard deviation
    seed=123
)

# Generate positions with controlled randomness
positions = natural_grid.initialize_positions(num_agents=36)

# Verify jitter bounds (positions within ~3 standard deviations)
grid_centers = natural_grid.calculate_grid_centers()
deviations = positions - grid_centers
max_deviation = np.max(np.abs(deviations))
assert max_deviation <= 3 * 2.0  # Within reasonable jitter bounds
```

### FixedListInitializer Reference

Enables precise agent placement at predetermined coordinates specified via lists, arrays, or coordinate data for exact experimental control.

#### Constructor Parameters

```python
FixedListInitializer(
    positions: Union[List[List[float]], np.ndarray],
    domain_bounds: Tuple[float, float] = (100.0, 100.0),
    cycling_enabled: bool = True
)
```

**Parameters:**
- `positions`: Predefined agent positions as list or numpy array
- `domain_bounds`: Domain dimensions for validation purposes
- `cycling_enabled`: Whether to cycle through positions for excess agents

#### Position Specification Formats

**List of Coordinates:**
```python
# Direct coordinate specification
position_list = [
    [10.0, 20.0],
    [30.0, 40.0], 
    [50.0, 60.0],
    [70.0, 80.0]
]
fixed_initializer = FixedListInitializer(positions=position_list)
```

**NumPy Array:**
```python
# NumPy array specification (more efficient)
position_array = np.array([
    [25.0, 25.0],
    [75.0, 25.0],
    [25.0, 75.0],
    [75.0, 75.0]
], dtype=np.float32)
array_initializer = FixedListInitializer(positions=position_array)
```

**Coordinate Validation:**
```python
# Automatic validation and type conversion
mixed_positions = [
    (15, 25),      # Tuple format
    [35.5, 45.2],  # List format
    np.array([55, 65])  # Array format
]
validated_initializer = FixedListInitializer(positions=mixed_positions)

# Verify all positions converted to consistent format
generated = validated_initializer.initialize_positions(num_agents=3)
assert generated.dtype == np.float32
assert generated.shape == (3, 2)
```

#### Cycling Behavior

```python
# Handle more agents than available positions
limited_positions = [[10, 10], [20, 20], [30, 30]]
cycling_initializer = FixedListInitializer(
    positions=limited_positions,
    cycling_enabled=True
)

# Request more agents than positions available
positions = cycling_initializer.initialize_positions(num_agents=7)
# Result: [10,10], [20,20], [30,30], [10,10], [20,20], [30,30], [10,10]

# Disable cycling for strict position control
strict_initializer = FixedListInitializer(
    positions=limited_positions,
    cycling_enabled=False
)

try:
    strict_positions = strict_initializer.initialize_positions(num_agents=5)
    assert False, "Should raise ValueError"
except ValueError as e:
    assert "cycling is disabled" in str(e)
```

#### Dynamic Position Updates

```python
# Runtime position modification
dynamic_initializer = FixedListInitializer(positions=[[0, 0]])

# Update positions during experiment
new_positions = [[10, 15], [25, 35], [40, 55]]
dynamic_initializer.set_positions(new_positions)

# Verify position count
assert dynamic_initializer.get_position_count() == 3

# Generate with updated positions
updated_positions = dynamic_initializer.initialize_positions(num_agents=3)
assert np.allclose(updated_positions, new_positions)
```

### FromDatasetInitializer Reference

Loads agent starting positions from external datasets, experimental recordings, or trajectory files with flexible sampling strategies and format support.

#### Constructor Parameters

```python
FromDatasetInitializer(
    dataset_path: Union[str, Path],
    x_column: str = "x",
    y_column: str = "y",
    domain_bounds: Tuple[float, float] = (100.0, 100.0),
    sampling_mode: str = "sequential",
    seed: Optional[int] = None,
    filter_conditions: Optional[Dict[str, Any]] = None
)
```

**Parameters:**
- `dataset_path`: Path to dataset file (CSV, JSON, etc.)
- `x_column`: Column name for x-coordinates
- `y_column`: Column name for y-coordinates
- `domain_bounds`: Domain dimensions for validation
- `sampling_mode`: Position sampling strategy ("sequential", "random", "stratified")
- `seed`: Random seed for sampling reproducibility
- `filter_conditions`: Optional data filtering conditions

#### Supported File Formats

**CSV Files:**
```python
# Load from CSV with default column names
csv_initializer = FromDatasetInitializer(
    dataset_path="experiments/initial_positions.csv",
    x_column="x_pos",
    y_column="y_pos"
)

# CSV file format example:
# x_pos,y_pos,experiment_id
# 10.5,20.3,exp_001
# 35.7,41.2,exp_001
# 52.1,63.8,exp_002
```

**JSON Files:**
```python
# Load from JSON with structured data
json_initializer = FromDatasetInitializer(
    dataset_path="data/positions.json",
    x_column="coordinates.x",  # Nested access
    y_column="coordinates.y"
)

# JSON file format example:
# [
#   {"coordinates": {"x": 15.0, "y": 25.0}, "metadata": {...}},
#   {"coordinates": {"x": 45.0, "y": 55.0}, "metadata": {...}}
# ]
```

#### Sampling Strategies

**Sequential Sampling (Default):**
```python
# Load positions in dataset order
sequential_initializer = FromDatasetInitializer(
    dataset_path="trajectory_data.csv",
    sampling_mode="sequential",
    seed=42
)

# Predictable position sequence
positions = sequential_initializer.initialize_positions(num_agents=20)
# Returns first 20 positions from dataset in original order
```

**Random Sampling:**
```python
# Random position selection with replacement
random_initializer = FromDatasetInitializer(
    dataset_path="large_dataset.csv",
    sampling_mode="random",
    seed=123
)

# Each call produces different random subset
positions_1 = random_initializer.initialize_positions(num_agents=50)
positions_2 = random_initializer.initialize_positions(num_agents=50)
# positions_1 != positions_2 (different random samples)

# Reset for reproducible random sampling
random_initializer.reset(seed=123)
positions_3 = random_initializer.initialize_positions(num_agents=50)
assert np.array_equal(positions_1, positions_3)  # Identical due to seed reset
```

**Stratified Sampling:**
```python
# Spatially representative sampling
stratified_initializer = FromDatasetInitializer(
    dataset_path="experiment_positions.csv",
    sampling_mode="stratified",
    seed=456
)

# Ensures spatial coverage across dataset extent
positions = stratified_initializer.initialize_positions(num_agents=100)
# Positions distributed across spatial bins for representative coverage
```

#### Data Filtering

```python
# Apply filtering conditions to dataset
filtered_initializer = FromDatasetInitializer(
    dataset_path="comprehensive_data.csv",
    filter_conditions={
        'experiment_type': 'control',  # Exact match filter
        'quality_score': {'min': 0.8, 'max': 1.0},  # Range filter
        'timestamp': {'min': '2024-01-01'}  # Date filter
    },
    sampling_mode="random",
    seed=789
)

# Only positions meeting filter criteria are available for sampling
filtered_positions = filtered_initializer.initialize_positions(num_agents=30)
```

#### Dynamic Dataset Operations

```python
# Runtime dataset management
dataset_initializer = FromDatasetInitializer(
    dataset_path="initial_dataset.csv"
)

# Force dataset reload (useful for dynamic files)
dataset_initializer.load_dataset(force_reload=True)

# Change sampling strategy at runtime
dataset_initializer.set_sampling_mode("stratified")

# Verify sampling mode update
assert dataset_initializer.sampling_mode == "stratified"

# Sample with new strategy
updated_positions = dataset_initializer.initialize_positions(num_agents=40)
```

#### Performance Optimization

```python
# Efficient large dataset handling
optimized_initializer = FromDatasetInitializer(
    dataset_path="massive_trajectory_data.h5",  # HDF5 for large datasets
    sampling_mode="random",
    seed=101112
)

# Dataset loaded once, cached for subsequent calls
start_time = time.time()
positions_1 = optimized_initializer.initialize_positions(num_agents=100)
first_call_time = time.time() - start_time

start_time = time.time()
positions_2 = optimized_initializer.initialize_positions(num_agents=100)
cached_call_time = time.time() - start_time

# Cached calls should be significantly faster
assert cached_call_time < first_call_time / 10  # At least 10x speedup
```

---

## Configuration Examples

### Uniform Random Configuration

#### Basic Configuration

```yaml
# conf/base/agent_init/uniform_random.yaml
_target_: plume_nav_sim.core.initialization.UniformRandomInitializer
bounds: [100.0, 100.0]
seed: 42
margin: 0.0
```

#### Hydra Composition Patterns

```yaml
# conf/experiment/baseline_random.yaml
defaults:
  - base_config
  - agent_init: uniform_random
  - override hydra/launcher: joblib

# Override specific parameters
agent_init:
  bounds: [200.0, 150.0]
  seed: ${random_seed}
  margin: 5.0

# Environment variable substitution
random_seed: ${oc.env:EXPERIMENT_SEED,42}
```

#### Advanced Uniform Random with Environment Overrides

```yaml
# conf/base/agent_init/uniform_random_advanced.yaml
_target_: plume_nav_sim.core.initialization.UniformRandomInitializer

# Core parameters with environment variable support
bounds: ${oc.env:DOMAIN_BOUNDS,[100.0,100.0]}
seed: ${oc.env:INIT_SEED,null}
margin: ${oc.env:BOUNDARY_MARGIN,2.0}

# Performance optimization settings
performance:
  vectorized_ops: true
  dtype: "float32"
  enable_profiling: ${oc.env:ENABLE_PROFILING,false}

# Validation configuration
validation:
  strict_validation: true
  verbose_logging: ${oc.env:VERBOSE_INIT,false}
```

### Grid Configuration

#### Basic Grid Layout

```yaml
# conf/base/agent_init/grid.yaml
_target_: plume_nav_sim.core.initialization.GridInitializer
domain_bounds: [100.0, 100.0]
grid_spacing: [10.0, 10.0]
grid_shape: [5, 5]
orientation: 0.0
jitter_enabled: false
seed: 42
```

#### Grid with Natural Variation

```yaml
# conf/base/agent_init/grid_jitter.yaml
_target_: plume_nav_sim.core.initialization.GridInitializer
domain_bounds: [200.0, 150.0]
grid_spacing: [15.0, 12.0]
grid_shape: [8, 6]
orientation: 0.087  # ~5 degrees rotation

# Natural variation parameters
jitter_enabled: true
jitter_std: 2.0
seed: ${experiment_seed}

# Boundary handling configuration
boundary_handling:
  strategy: "scale"
  preserve_shape: true
  margin: 5.0
```

#### Dynamic Grid Configuration

```yaml
# conf/base/agent_init/adaptive_grid.yaml
_target_: plume_nav_sim.core.initialization.GridInitializer

# Environment-driven parameters
domain_bounds: ${oc.env:GRID_DOMAIN,[100.0,100.0]}
grid_spacing: ${oc.env:GRID_SPACING,[10.0,10.0]}
grid_shape: ${oc.env:GRID_SHAPE,[5,5]}

# Runtime overrides
environment_overrides:
  grid_spacing_override: ${oc.env:RUNTIME_GRID_SPACING,null}
  orientation_override: ${oc.env:GRID_ROTATION,null}
  jitter_std_override: ${oc.env:JITTER_LEVEL,null}

# Deterministic seeding with override
seed: ${oc.env:GRID_SEED,123}
```

### Fixed List Configuration

#### Direct Position Specification

```yaml
# conf/base/agent_init/fixed_list.yaml
_target_: plume_nav_sim.core.initialization.FixedListInitializer

# Direct coordinate specification
positions:
  - [10.0, 20.0]
  - [30.0, 40.0]
  - [50.0, 60.0]
  - [70.0, 80.0]

domain_bounds: [100.0, 100.0]
cycling_enabled: true
```

#### File-Based Position Loading

```yaml
# conf/base/agent_init/fixed_from_file.yaml
_target_: plume_nav_sim.core.initialization.FixedListInitializer

# File-based position loading
positions: []  # Empty - positions loaded from file
position_file: ${oc.env:POSITION_FILE,"data/initial_positions.csv"}

# File format configuration
file_format:
  type: "csv"
  csv:
    delimiter: ","
    has_header: true
    skip_rows: 0

# Column specification
coordinate_columns:
  x: "x_coordinate"
  y: "y_coordinate"
  agent_id: "id"

# Validation and fallback
validation:
  strict_validation: true
  fallback_strategy: "uniform_random"
  
domain_bounds: [200.0, 150.0]
cycling_enabled: false
```

### From Dataset Configuration

#### CSV Dataset Loading

```yaml
# conf/base/agent_init/from_dataset.yaml
_target_: plume_nav_sim.core.initialization.FromDatasetInitializer

# Dataset source configuration
dataset_path: ${oc.env:DATASET_PATH,"data/experimental_positions.csv"}
x_column: "position_x"
y_column: "position_y"
domain_bounds: [100.0, 100.0]

# Sampling strategy
sampling_mode: "random"
seed: 42

# Data filtering
filter_conditions:
  experiment_type: "baseline"
  data_quality: {"min": 0.8}
  valid_position: true
```

#### Multi-Format Dataset Support

```yaml
# conf/base/agent_init/dataset_multi_format.yaml
_target_: plume_nav_sim.core.initialization.FromDatasetInitializer

# Auto-detect format from extension
dataset_path: ${dataset_file}
format:
  auto_detect: true
  format_options:
    csv:
      delimiter: ","
      header: 0
      encoding: "utf-8"
    json:
      orient: "records"
      lines: false
    hdf5:
      dataset_key: "/trajectories/initial_positions"
      chunk_size: 10000

# Column specification with fallbacks
position_columns:
  x: ["x", "pos_x", "position_x", "coordinate_x"]
  y: ["y", "pos_y", "position_y", "coordinate_y"]
  metadata: ["experiment_id", "timestamp", "quality_score"]

# Temporal sampling configuration
temporal_sampling:
  enabled: ${oc.env:TEMPORAL_SAMPLING,false}
  time_column: "timestamp" 
  time_point: ${oc.env:SAMPLE_TIME_POINT,0.0}
  time_window: 1.0  # ±1.0 time units

# Statistical resampling
statistical_resampling:
  mode: "stratified"
  strata_columns: ["experiment_id", "condition"]
  min_samples_per_stratum: 5
  seed: ${oc.env:DATASET_SEED,456}
```

### Hydra Composition Examples

#### Multi-Experiment Configuration

```yaml
# conf/experiment/multi_init_comparison.yaml
defaults:
  - base_config
  - _self_

# Hydra multirun with different initialization strategies
hydra:
  mode: MULTIRUN
  sweeper:
    _target_: hydra._internal.BasicSweeper
    max_batch_size: 4
    
  sweep:
    dir: outputs/init_comparison/${now:%Y-%m-%d_%H-%M-%S}
    subdir: ${hydra.job.override_dirname}

# Sweep over initialization strategies
agent_init:
  - uniform_random
  - grid  
  - fixed_list
  - from_dataset

# Sweep over agent counts
num_agents: 
  - 25
  - 50
  - 100

# Fixed experimental parameters
domain_bounds: [100.0, 100.0]
experiment_seed: 42
```

#### Environment Variable Overrides

```yaml
# conf/deployment/production.yaml
defaults:
  - base_config
  - agent_init: ${oc.env:INIT_STRATEGY,uniform_random}

# Production deployment with full environment control
agent_init:
  # Core parameters from environment
  domain_bounds: ${oc.env:SIMULATION_DOMAIN,[100.0,100.0]}
  seed: ${oc.env:RANDOM_SEED,null}
  
  # Strategy-specific overrides
  bounds: ${oc.env:INIT_BOUNDS,${agent_init.domain_bounds}}
  grid_spacing: ${oc.env:GRID_SPACING,[10.0,10.0]}
  dataset_path: ${oc.env:DATASET_PATH,"data/default_positions.csv"}
  
  # Performance tuning
  performance:
    vectorized_ops: ${oc.env:VECTORIZED_INIT,true}
    enable_profiling: ${oc.env:PROFILE_INIT,false}

# Runtime validation
experiment:
  validate_initialization: ${oc.env:VALIDATE_INIT,true}
  max_init_time_ms: ${oc.env:MAX_INIT_TIME,5}
```

---

## Integration Patterns

### Environment Reset Integration

The agent initialization system integrates seamlessly with `PlumeNavigationEnv.reset()` method through dependency injection and configuration-driven strategy selection.

#### Basic Environment Integration

```python
from plume_nav_sim.envs import PlumeNavigationEnv
from plume_nav_sim.core.initialization import create_agent_initializer

# Environment with custom initialization strategy
config = {
    'type': 'grid',
    'domain_bounds': (100, 100),
    'grid_spacing': (10, 10),
    'seed': 42
}

initializer = create_agent_initializer(config)
env = PlumeNavigationEnv(agent_initializer=initializer)

# Reset automatically uses configured initialization
obs = env.reset()
print(f"Agents initialized at: {env.navigator.positions}")
```

#### Multi-Agent Initialization

```python
# Multi-agent environment with deterministic initialization
multi_agent_config = {
    'type': 'uniform_random',
    'bounds': (200, 150),
    'seed': 123,
    'margin': 5.0
}

multi_initializer = create_agent_initializer(multi_agent_config)
multi_env = PlumeNavigationEnv(
    num_agents=50,
    agent_initializer=multi_initializer
)

# All 50 agents initialized simultaneously
obs = multi_env.reset()
assert len(env.navigator.positions) == 50

# Verify positions within bounds accounting for margin
positions = env.navigator.positions
assert np.all(positions >= 5.0)  # Margin compliance
assert np.all(positions[:, 0] <= 195.0)  # Width bound
assert np.all(positions[:, 1] <= 145.0)  # Height bound
```

#### Deterministic Seeding

```python
# Reproducible episode initialization across runs
seeded_config = {
    'type': 'grid',
    'domain_bounds': (100, 100),
    'grid_shape': (5, 4),
    'jitter_enabled': True,
    'jitter_std': 1.0,
    'seed': 456
}

seeded_initializer = create_agent_initializer(seeded_config)
env = PlumeNavigationEnv(agent_initializer=seeded_initializer)

# Identical initialization across multiple resets
positions_1 = env.reset()['agent_positions']
positions_2 = env.reset()['agent_positions']
assert np.allclose(positions_1, positions_2)  # Deterministic reproduction

# Different seed produces different positions
seeded_initializer.reset(seed=789)
positions_3 = env.reset()['agent_positions'] 
assert not np.allclose(positions_1, positions_3)  # Different seed, different positions
```

### Factory Function Usage

The `create_agent_initializer()` factory function provides centralized creation with automatic strategy selection and error handling.

#### Configuration-Driven Creation

```python
from plume_nav_sim.core.initialization import create_agent_initializer

# Dictionary-based configuration
uniform_config = {
    'type': 'uniform_random',
    'bounds': (100, 100),
    'seed': 42,
    'margin': 2.0
}

grid_config = {
    'type': 'grid', 
    'domain_bounds': (150, 100),
    'grid_spacing': (15, 10),
    'jitter_enabled': True,
    'seed': 123
}

# Factory automatically selects appropriate class
uniform_initializer = create_agent_initializer(uniform_config)
grid_initializer = create_agent_initializer(grid_config)

assert uniform_initializer.get_strategy_name() == "uniform_random"
assert grid_initializer.get_strategy_name() == "grid"
```

#### Hydra Target Integration

```python
# Hydra-style configuration with _target_ specification
hydra_config = {
    '_target_': 'plume_nav_sim.core.initialization.FromDatasetInitializer',
    'dataset_path': 'data/experiment_positions.csv',
    'sampling_mode': 'stratified',
    'seed': 789,
    'filter_conditions': {'quality': {'min': 0.9}}
}

# Factory handles Hydra instantiation automatically
dataset_initializer = create_agent_initializer(hydra_config)
assert dataset_initializer.get_strategy_name() == "from_dataset"

# Verify configuration parameters applied correctly
positions = dataset_initializer.initialize_positions(num_agents=30)
assert len(positions) == 30
```

#### Error Handling and Validation

```python
# Comprehensive error handling for invalid configurations
invalid_configs = [
    {},  # Missing type/target
    {'type': 'invalid_strategy'},  # Unknown strategy
    {'type': 'uniform_random', 'bounds': (-10, 50)},  # Invalid bounds
    {'_target_': 'NonExistentClass'},  # Invalid target class
]

for config in invalid_configs:
    try:
        initializer = create_agent_initializer(config)
        assert False, f"Should have raised error for config: {config}"
    except (ValueError, TypeError, ImportError) as e:
        print(f"Expected error for {config}: {e}")

# Valid configuration with parameter validation
try:
    valid_initializer = create_agent_initializer({
        'type': 'grid',
        'domain_bounds': (100, 100),
        'grid_spacing': (10, 10),
        'seed': 42
    })
    print("Valid configuration created successfully")
except Exception as e:
    assert False, f"Valid configuration should not raise error: {e}"
```

### Validation Patterns

#### Domain Compliance Validation

```python
# Automatic domain validation for all initialization strategies
def validate_initialization_setup(initializer, domain_bounds, num_agents):
    """Comprehensive initialization validation."""
    
    # Generate test positions
    positions = initializer.initialize_positions(num_agents=num_agents)
    
    # Domain boundary validation
    is_valid = initializer.validate_domain(positions, domain_bounds)
    if not is_valid:
        raise ValueError(f"Generated positions violate domain bounds {domain_bounds}")
    
    # Shape validation
    expected_shape = (num_agents, 2)
    if positions.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {positions.shape}")
    
    # Data type validation
    if positions.dtype not in [np.float32, np.float64]:
        raise ValueError(f"Expected float type, got {positions.dtype}")
    
    # Position validity (no NaN/Inf values)
    if not np.all(np.isfinite(positions)):
        raise ValueError("Positions contain NaN or infinite values")
    
    return True

# Test validation with different strategies
strategies = [
    {'type': 'uniform_random', 'bounds': (100, 100), 'seed': 42},
    {'type': 'grid', 'domain_bounds': (100, 100), 'seed': 42},
    {'type': 'fixed_list', 'positions': [[10, 10], [20, 20]], 'cycling_enabled': True}
]

for config in strategies:
    initializer = create_agent_initializer(config)
    validate_initialization_setup(initializer, (100, 100), 25)
    print(f"Strategy '{initializer.get_strategy_name()}' validation passed")
```

#### Performance Validation

```python
import time

def validate_performance_requirements(initializer, num_agents, max_time_ms=5):
    """Validate initialization performance requirements."""
    
    # Measure initialization time
    start_time = time.time()
    positions = initializer.initialize_positions(num_agents=num_agents)
    duration_ms = (time.time() - start_time) * 1000
    
    if duration_ms > max_time_ms:
        raise ValueError(
            f"Initialization took {duration_ms:.2f}ms, "
            f"exceeds limit of {max_time_ms}ms for {num_agents} agents"
        )
    
    # Measure validation time
    start_time = time.time()
    is_valid = initializer.validate_domain(positions, (100, 100))
    validation_ms = (time.time() - start_time) * 1000
    
    if validation_ms > 1.0:  # 1ms validation limit
        raise ValueError(
            f"Validation took {validation_ms:.2f}ms, exceeds 1ms limit"
        )
    
    # Measure reset time
    start_time = time.time()
    initializer.reset(seed=42)
    reset_ms = (time.time() - start_time) * 1000
    
    if reset_ms > 1.0:  # 1ms reset limit
        raise ValueError(
            f"Reset took {reset_ms:.2f}ms, exceeds 1ms limit"
        )
    
    return duration_ms, validation_ms, reset_ms

# Performance validation for all strategies
performance_results = {}
for strategy_name in ['uniform_random', 'grid', 'fixed_list']:
    config = {
        'type': strategy_name,
        'domain_bounds': (100, 100) if strategy_name != 'uniform_random' else None,
        'bounds': (100, 100) if strategy_name == 'uniform_random' else None,
        'positions': [[10, 10], [20, 20]] if strategy_name == 'fixed_list' else None,
        'cycling_enabled': True if strategy_name == 'fixed_list' else None,
        'seed': 42
    }
    config = {k: v for k, v in config.items() if v is not None}
    
    initializer = create_agent_initializer(config)
    times = validate_performance_requirements(initializer, num_agents=100)
    performance_results[strategy_name] = times
    
    print(f"{strategy_name}: init={times[0]:.2f}ms, "
          f"validation={times[1]:.2f}ms, reset={times[2]:.2f}ms")
```

---

## Advanced Usage

### Custom Initialization Strategies

Developers can create custom initialization strategies by implementing the `AgentInitializerProtocol` interface for specialized research requirements.

#### Protocol Implementation Template

```python
from plume_nav_sim.core.protocols import AgentInitializerProtocol
import numpy as np
from typing import Optional, Any, Tuple

class CustomRadialInitializer:
    """Custom radial initialization strategy with concentric rings."""
    
    def __init__(
        self,
        center: Tuple[float, float] = (50.0, 50.0),
        max_radius: float = 40.0,
        ring_count: int = 3,
        seed: Optional[int] = None
    ):
        self.center = center
        self.max_radius = max_radius
        self.ring_count = ring_count
        self.seed = seed
        self._rng = np.random.RandomState(seed)
    
    def initialize_positions(self, num_agents: int, **kwargs: Any) -> np.ndarray:
        """Generate positions in concentric rings around center point."""
        if num_agents <= 0:
            raise ValueError(f"Number of agents must be positive: {num_agents}")
        
        positions = []
        agents_per_ring = num_agents // self.ring_count
        remaining_agents = num_agents % self.ring_count
        
        for ring_idx in range(self.ring_count):
            # Calculate ring radius
            ring_radius = (ring_idx + 1) * (self.max_radius / self.ring_count)
            
            # Determine agents for this ring
            ring_agents = agents_per_ring
            if ring_idx < remaining_agents:
                ring_agents += 1
            
            # Generate angular positions
            angles = self._rng.uniform(0, 2 * np.pi, ring_agents)
            
            # Convert to Cartesian coordinates
            ring_x = self.center[0] + ring_radius * np.cos(angles)
            ring_y = self.center[1] + ring_radius * np.sin(angles)
            
            # Add to positions list
            for x, y in zip(ring_x, ring_y):
                positions.append([x, y])
        
        return np.array(positions[:num_agents], dtype=np.float32)
    
    def validate_domain(self, positions: np.ndarray, domain_bounds: Tuple[float, float]) -> bool:
        """Validate positions within domain bounds."""
        if positions.ndim != 2 or positions.shape[1] != 2:
            return False
        
        # Check domain bounds
        x_valid = np.all((positions[:, 0] >= 0) & (positions[:, 0] <= domain_bounds[0]))
        y_valid = np.all((positions[:, 1] >= 0) & (positions[:, 1] <= domain_bounds[1]))
        
        return bool(x_valid and y_valid)
    
    def reset(self, seed: Optional[int] = None, **kwargs: Any) -> None:
        """Reset random state for deterministic behavior."""
        reset_seed = seed if seed is not None else self.seed
        self._rng = np.random.RandomState(reset_seed)
    
    def get_strategy_name(self) -> str:
        """Return strategy identifier."""
        return "custom_radial"

# Usage example
custom_initializer = CustomRadialInitializer(
    center=(50, 50),
    max_radius=30,
    ring_count=4,
    seed=42
)

# Verify protocol compliance
from plume_nav_sim.core.protocols import AgentInitializerProtocol
assert isinstance(custom_initializer, AgentInitializerProtocol)

# Test custom strategy
positions = custom_initializer.initialize_positions(num_agents=20)
print(f"Generated {len(positions)} positions in radial pattern")

# Visualize pattern (optional)
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.scatter(positions[:, 0], positions[:, 1], alpha=0.7)
plt.scatter(*custom_initializer.center, color='red', s=100, marker='x')
plt.axis('equal')
plt.title('Custom Radial Initialization Pattern')
plt.show()
```

#### Integration with Factory Function

```python
# Extend factory function to support custom strategies
def create_extended_agent_initializer(config):
    """Extended factory with custom strategy support."""
    
    if config.get('type') == 'custom_radial':
        params = {k: v for k, v in config.items() if k != 'type'}
        return CustomRadialInitializer(**params)
    
    # Fallback to standard factory
    return create_agent_initializer(config)

# Test extended factory
custom_config = {
    'type': 'custom_radial',
    'center': (75, 75),
    'max_radius': 25,
    'ring_count': 3,
    'seed': 123
}

custom_init = create_extended_agent_initializer(custom_config)
custom_positions = custom_init.initialize_positions(num_agents=15)
```

### Dataset Integration Patterns

Advanced dataset integration enables sophisticated initialization from experimental data, trajectory recordings, and multi-format sources.

#### Multi-Format Dataset Loader

```python
from pathlib import Path
import pandas as pd
import h5py
import json

class AdvancedDatasetInitializer:
    """Advanced dataset initializer with multi-format support."""
    
    def __init__(
        self,
        dataset_sources: List[str],
        format_priority: List[str] = ['hdf5', 'parquet', 'csv', 'json'],
        coordinate_mapping: Dict[str, str] = None,
        temporal_filter: Dict[str, Any] = None,
        seed: Optional[int] = None
    ):
        self.dataset_sources = dataset_sources
        self.format_priority = format_priority
        self.coordinate_mapping = coordinate_mapping or {'x': 'x', 'y': 'y'}
        self.temporal_filter = temporal_filter
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._cached_positions = None
    
    def _load_dataset(self, source_path: str) -> pd.DataFrame:
        """Load dataset from file with format auto-detection."""
        path = Path(source_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {source_path}")
        
        # Auto-detect format from extension
        extension = path.suffix.lower()
        
        if extension in ['.h5', '.hdf5']:
            with h5py.File(path, 'r') as f:
                # Find position datasets
                position_data = f['positions'][:]  # Assumes HDF5 structure
                df = pd.DataFrame(position_data, columns=['x', 'y'])
                
        elif extension == '.parquet':
            df = pd.read_parquet(path)
            
        elif extension == '.csv':
            df = pd.read_csv(path)
            
        elif extension == '.json':
            df = pd.read_json(path)
            
        else:
            raise ValueError(f"Unsupported file format: {extension}")
        
        return df
    
    def _apply_temporal_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply temporal filtering to dataset."""
        if not self.temporal_filter:
            return df
        
        filtered_df = df.copy()
        
        # Time range filtering
        if 'time_range' in self.temporal_filter:
            time_col = self.temporal_filter.get('time_column', 'timestamp')
            if time_col in df.columns:
                time_min, time_max = self.temporal_filter['time_range']
                filtered_df = filtered_df[
                    (filtered_df[time_col] >= time_min) & 
                    (filtered_df[time_col] <= time_max)
                ]
        
        # Sampling rate filtering
        if 'sample_rate' in self.temporal_filter:
            sample_rate = self.temporal_filter['sample_rate']
            filtered_df = filtered_df.iloc[::sample_rate]
        
        return filtered_df
    
    def _merge_datasets(self, dataframes: List[pd.DataFrame]) -> np.ndarray:
        """Merge multiple datasets into position array."""
        all_positions = []
        
        for df in dataframes:
            # Apply coordinate mapping
            x_col = self.coordinate_mapping['x']
            y_col = self.coordinate_mapping['y']
            
            if x_col not in df.columns or y_col not in df.columns:
                continue
            
            # Extract positions
            positions = df[[x_col, y_col]].values
            valid_mask = np.isfinite(positions).all(axis=1)
            all_positions.append(positions[valid_mask])
        
        if not all_positions:
            raise ValueError("No valid positions found in datasets")
        
        # Combine all positions
        combined_positions = np.vstack(all_positions)
        return combined_positions.astype(np.float32)
    
    def initialize_positions(self, num_agents: int, **kwargs: Any) -> np.ndarray:
        """Initialize positions from merged datasets."""
        if self._cached_positions is None:
            # Load all datasets
            dataframes = []
            for source in self.dataset_sources:
                try:
                    df = self._load_dataset(source)
                    df = self._apply_temporal_filter(df)
                    dataframes.append(df)
                except Exception as e:
                    print(f"Warning: Failed to load {source}: {e}")
            
            if not dataframes:
                raise ValueError("No datasets loaded successfully")
            
            # Merge and cache positions
            self._cached_positions = self._merge_datasets(dataframes)
        
        # Sample requested number of positions
        if len(self._cached_positions) == 0:
            raise ValueError("No positions available after filtering")
        
        if num_agents > len(self._cached_positions):
            # Sample with replacement if not enough positions
            indices = self._rng.choice(
                len(self._cached_positions), 
                size=num_agents, 
                replace=True
            )
        else:
            # Sample without replacement
            indices = self._rng.choice(
                len(self._cached_positions), 
                size=num_agents, 
                replace=False
            )
        
        return self._cached_positions[indices].copy()
    
    def validate_domain(self, positions: np.ndarray, domain_bounds: Tuple[float, float]) -> bool:
        """Validate positions within domain bounds."""
        if positions.ndim != 2 or positions.shape[1] != 2:
            return False
        
        x_valid = np.all((positions[:, 0] >= 0) & (positions[:, 0] <= domain_bounds[0]))
        y_valid = np.all((positions[:, 1] >= 0) & (positions[:, 1] <= domain_bounds[1]))
        
        return bool(x_valid and y_valid)
    
    def reset(self, seed: Optional[int] = None, **kwargs: Any) -> None:
        """Reset random state."""
        reset_seed = seed if seed is not None else self.seed
        self._rng = np.random.RandomState(reset_seed)
    
    def get_strategy_name(self) -> str:
        """Return strategy identifier."""
        return "advanced_dataset"

# Usage example
advanced_dataset_init = AdvancedDatasetInitializer(
    dataset_sources=[
        'data/experiment_1_positions.csv',
        'data/experiment_2_positions.h5',
        'data/experiment_3_positions.parquet'
    ],
    coordinate_mapping={'x': 'pos_x', 'y': 'pos_y'},
    temporal_filter={
        'time_column': 'timestamp',
        'time_range': (0.0, 100.0),
        'sample_rate': 2  # Every 2nd sample
    },
    seed=42
)

# Generate positions from merged datasets
merged_positions = advanced_dataset_init.initialize_positions(num_agents=100)
print(f"Loaded {len(merged_positions)} positions from multiple datasets")
```

### Reproducible Research Workflows

#### Experimental Configuration Management

```python
from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import json
import hashlib
from datetime import datetime

@dataclass
class ExperimentConfig:
    """Complete experimental configuration with version tracking."""
    
    # Initialization strategy configuration
    initialization_strategy: str
    initialization_params: Dict[str, Any]
    
    # Environmental parameters
    domain_bounds: Tuple[float, float]
    num_agents: int
    
    # Reproducibility parameters
    random_seed: int
    software_version: str
    config_version: str
    
    # Metadata
    experiment_id: str
    timestamp: str
    researcher: str
    description: str
    
    def __post_init__(self):
        """Generate configuration hash for reproducibility."""
        # Create deterministic hash of configuration
        config_dict = asdict(self)
        config_str = json.dumps(config_dict, sort_keys=True)
        self.config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def save_config(self, output_path: str) -> str:
        """Save configuration to file."""
        full_path = f"{output_path}/experiment_{self.experiment_id}_config.json"
        with open(full_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        return full_path
    
    @classmethod
    def load_config(cls, config_path: str) -> 'ExperimentConfig':
        """Load configuration from file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

def create_reproducible_experiment(
    strategy_name: str,
    strategy_params: Dict[str, Any],
    experiment_metadata: Dict[str, Any]
) -> Tuple[AgentInitializerProtocol, ExperimentConfig]:
    """Create reproducible experimental setup."""
    
    # Generate unique experiment ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{strategy_name}_{timestamp}"
    
    # Create complete configuration
    config = ExperimentConfig(
        initialization_strategy=strategy_name,
        initialization_params=strategy_params,
        domain_bounds=experiment_metadata.get('domain_bounds', (100, 100)),
        num_agents=experiment_metadata.get('num_agents', 50),
        random_seed=experiment_metadata.get('seed', 42),
        software_version="plume_nav_sim_v1.0",
        config_version="1.0.0",
        experiment_id=experiment_id,
        timestamp=timestamp,
        researcher=experiment_metadata.get('researcher', 'unknown'),
        description=experiment_metadata.get('description', '')
    )
    
    # Create initializer with configuration
    initializer_config = {
        'type': strategy_name,
        **strategy_params,
        'seed': config.random_seed
    }
    
    initializer = create_agent_initializer(initializer_config)
    
    return initializer, config

# Example reproducible experiment setup
strategy_params = {
    'domain_bounds': (150, 100),
    'grid_spacing': (12, 10),
    'jitter_enabled': True,
    'jitter_std': 1.5
}

experiment_metadata = {
    'domain_bounds': (150, 100),
    'num_agents': 60,
    'seed': 12345,
    'researcher': 'Dr. Smith',
    'description': 'Grid initialization with natural variation study'
}

# Create reproducible experiment
initializer, exp_config = create_reproducible_experiment(
    strategy_name='grid',
    strategy_params=strategy_params,
    experiment_metadata=experiment_metadata
)

# Save configuration for reproducibility
config_path = exp_config.save_config('./experiment_configs')
print(f"Experiment configuration saved: {config_path}")
print(f"Configuration hash: {exp_config.config_hash}")

# Initialize positions with full traceability
positions = initializer.initialize_positions(num_agents=exp_config.num_agents)
print(f"Generated {len(positions)} positions for experiment {exp_config.experiment_id}")
```

### Performance Optimization

#### Vectorized Large-Scale Initialization

```python
import time
from typing import List
import psutil
import os

class PerformanceOptimizedInitializer:
    """High-performance initializer for large-scale scenarios."""
    
    def __init__(self, base_initializer: AgentInitializerProtocol):
        self.base_initializer = base_initializer
        self._position_cache = {}
        self._performance_stats = []
    
    def initialize_positions_batched(
        self, 
        num_agents: int,
        batch_size: int = 1000,
        use_cache: bool = True,
        parallel: bool = False
    ) -> np.ndarray:
        """Initialize positions in batches for memory efficiency."""
        
        start_time = time.time()
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        # Check cache first
        cache_key = (num_agents, id(self.base_initializer))
        if use_cache and cache_key in self._position_cache:
            cached_positions = self._position_cache[cache_key]
            if len(cached_positions) >= num_agents:
                return cached_positions[:num_agents].copy()
        
        # Generate positions in batches
        all_positions = []
        remaining_agents = num_agents
        
        while remaining_agents > 0:
            current_batch = min(batch_size, remaining_agents)
            
            # Generate batch
            batch_positions = self.base_initializer.initialize_positions(
                num_agents=current_batch
            )
            all_positions.append(batch_positions)
            remaining_agents -= current_batch
        
        # Combine batches
        combined_positions = np.vstack(all_positions)
        
        # Update cache
        if use_cache:
            self._position_cache[cache_key] = combined_positions
        
        # Record performance statistics
        end_time = time.time()
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        perf_stat = {
            'num_agents': num_agents,
            'batch_size': batch_size,
            'duration_ms': (end_time - start_time) * 1000,
            'memory_used_mb': final_memory - initial_memory,
            'positions_per_second': num_agents / (end_time - start_time),
            'strategy': self.base_initializer.get_strategy_name()
        }
        self._performance_stats.append(perf_stat)
        
        return combined_positions
    
    def benchmark_performance(
        self, 
        agent_counts: List[int],
        trials: int = 3
    ) -> Dict[str, Any]:
        """Comprehensive performance benchmarking."""
        
        benchmark_results = {
            'strategy': self.base_initializer.get_strategy_name(),
            'trials': trials,
            'results': []
        }
        
        for num_agents in agent_counts:
            trial_results = []
            
            for trial in range(trials):
                # Clear cache for fair measurement
                self._position_cache.clear()
                
                # Reset initializer state
                self.base_initializer.reset(seed=42 + trial)
                
                # Measure initialization time
                start_time = time.time()
                positions = self.initialize_positions_batched(
                    num_agents=num_agents,
                    use_cache=False
                )
                duration_ms = (time.time() - start_time) * 1000
                
                # Validate results
                is_valid = self.base_initializer.validate_domain(
                    positions, (100, 100)
                )
                
                trial_results.append({
                    'duration_ms': duration_ms,
                    'positions_per_second': num_agents / (duration_ms / 1000),
                    'valid': is_valid,
                    'memory_efficient': duration_ms < (num_agents * 0.05)  # <0.05ms per agent
                })
            
            # Calculate statistics
            durations = [r['duration_ms'] for r in trial_results]
            rates = [r['positions_per_second'] for r in trial_results]
            
            benchmark_results['results'].append({
                'num_agents': num_agents,
                'mean_duration_ms': np.mean(durations),
                'std_duration_ms': np.std(durations),
                'mean_rate_pos_per_sec': np.mean(rates),
                'max_rate_pos_per_sec': np.max(rates),
                'all_valid': all(r['valid'] for r in trial_results),
                'meets_performance_target': np.mean(durations) < 5.0  # <5ms target
            })
        
        return benchmark_results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of recorded performance statistics."""
        if not self._performance_stats:
            return {'message': 'No performance data recorded'}
        
        durations = [stat['duration_ms'] for stat in self._performance_stats]
        rates = [stat['positions_per_second'] for stat in self._performance_stats]
        memory_usage = [stat['memory_used_mb'] for stat in self._performance_stats]
        
        return {
            'total_calls': len(self._performance_stats),
            'mean_duration_ms': np.mean(durations),
            'max_duration_ms': np.max(durations),
            'mean_rate_pos_per_sec': np.mean(rates),
            'peak_rate_pos_per_sec': np.max(rates),
            'mean_memory_mb': np.mean(memory_usage),
            'peak_memory_mb': np.max(memory_usage),
            'performance_target_met': np.mean(durations) < 5.0
        }

# Performance optimization example
base_grid_init = create_agent_initializer({
    'type': 'grid',
    'domain_bounds': (200, 200),
    'grid_spacing': (5, 5),
    'seed': 42
})

optimized_init = PerformanceOptimizedInitializer(base_grid_init)

# Benchmark different agent counts
agent_counts = [100, 500, 1000, 2000, 5000]
benchmark_results = optimized_init.benchmark_performance(
    agent_counts=agent_counts,
    trials=5
)

# Display benchmark results
print("Performance Benchmark Results:")
print(f"Strategy: {benchmark_results['strategy']}")
print("-" * 60)

for result in benchmark_results['results']:
    print(f"Agents: {result['num_agents']:5d} | "
          f"Time: {result['mean_duration_ms']:6.2f}ms | "
          f"Rate: {result['mean_rate_pos_per_sec']:8.0f} pos/sec | "
          f"Target Met: {result['meets_performance_target']}")
```

### Troubleshooting Guide

#### Common Issues and Solutions

**Issue 1: Initialization Time Exceeds Performance Requirements**

```python
def diagnose_performance_issues(initializer, num_agents=100):
    """Diagnose and suggest solutions for performance issues."""
    
    issues = []
    suggestions = []
    
    # Measure initialization time
    start_time = time.time()
    positions = initializer.initialize_positions(num_agents=num_agents)
    duration_ms = (time.time() - start_time) * 1000
    
    if duration_ms > 5.0:  # Exceeds 5ms target
        issues.append(f"Initialization time {duration_ms:.2f}ms exceeds 5ms target")
        
        strategy_name = initializer.get_strategy_name()
        
        if strategy_name == "from_dataset":
            suggestions.extend([
                "Enable dataset caching to avoid repeated file I/O",
                "Use more efficient file formats (HDF5, Parquet vs CSV)",
                "Implement lazy loading with position pre-sampling",
                "Reduce dataset size or add data filtering"
            ])
        elif strategy_name == "grid":
            suggestions.extend([
                "Disable jitter for performance-critical scenarios",
                "Use simpler boundary handling strategies",
                "Pre-calculate grid positions and cache results"
            ])
        elif strategy_name == "uniform_random":
            suggestions.extend([
                "Use vectorized NumPy operations",
                "Reduce margin validation complexity",
                "Enable float32 instead of float64 precision"
            ])
    
    # Check memory usage
    if hasattr(initializer, '_positions_cache'):
        cache_size = len(initializer._positions_cache)
        if cache_size > 10000:
            issues.append(f"Large position cache ({cache_size} entries)")
            suggestions.append("Implement LRU cache with size limits")
    
    return issues, suggestions

# Example diagnosis
grid_init = create_agent_initializer({
    'type': 'grid',
    'domain_bounds': (100, 100),
    'jitter_enabled': True,
    'jitter_std': 2.0,
    'seed': 42
})

issues, suggestions = diagnose_performance_issues(grid_init, num_agents=500)
if issues:
    print("Performance Issues Detected:")
    for issue in issues:
        print(f"  - {issue}")
    print("\nSuggested Solutions:")
    for suggestion in suggestions:
        print(f"  - {suggestion}")
```

**Issue 2: Domain Validation Failures**

```python
def diagnose_domain_issues(initializer, domain_bounds, num_agents=50):
    """Diagnose domain validation problems."""
    
    positions = initializer.initialize_positions(num_agents=num_agents)
    is_valid = initializer.validate_domain(positions, domain_bounds)
    
    if not is_valid:
        # Identify specific violations
        x_violations = np.sum(
            (positions[:, 0] < 0) | (positions[:, 0] > domain_bounds[0])
        )
        y_violations = np.sum(
            (positions[:, 1] < 0) | (positions[:, 1] > domain_bounds[1])
        )
        
        print(f"Domain Validation Failed:")
        print(f"  Domain bounds: {domain_bounds}")
        print(f"  X violations: {x_violations}/{num_agents}")
        print(f"  Y violations: {y_violations}/{num_agents}")
        print(f"  Position range: X=[{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}], "
              f"Y=[{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
        
        # Strategy-specific suggestions
        strategy_name = initializer.get_strategy_name()
        
        if strategy_name == "grid":
            print("\nGrid-specific suggestions:")
            print("  - Reduce grid spacing or grid size")
            print("  - Enable boundary handling with 'clip' or 'scale' strategy")
            print("  - Increase domain bounds to accommodate grid")
            
        elif strategy_name == "uniform_random":
            print("\nUniform Random suggestions:")
            print("  - Check margin parameter - may be too large")
            print("  - Verify bounds parameter matches domain_bounds")
            
        elif strategy_name == "fixed_list":
            print("\nFixed List suggestions:")
            print("  - Verify all predefined positions are within domain")
            print("  - Enable position scaling or transformation")
    
    return is_valid

# Example domain diagnosis
problematic_grid = create_agent_initializer({
    'type': 'grid',
    'domain_bounds': (50, 50),  # Small domain
    'grid_spacing': (20, 20),   # Large spacing
    'grid_shape': (5, 5),       # Too many grid points
    'seed': 42
})

is_valid = diagnose_domain_issues(problematic_grid, (50, 50), num_agents=25)
```

**Issue 3: Deterministic Seeding Problems**

```python
def test_deterministic_behavior(initializer, seed=42, trials=5):
    """Test and diagnose deterministic seeding issues."""
    
    reference_positions = None
    seed_issues = []
    
    for trial in range(trials):
        initializer.reset(seed=seed)
        positions = initializer.initialize_positions(num_agents=10)
        
        if reference_positions is None:
            reference_positions = positions.copy()
        else:
            if not np.allclose(positions, reference_positions, rtol=1e-6):
                seed_issues.append(f"Trial {trial}: Positions differ from reference")
    
    if seed_issues:
        print("Deterministic Seeding Issues:")
        for issue in seed_issues:
            print(f"  - {issue}")
        
        print("\nDebugging suggestions:")
        print("  - Verify reset() method properly reinitializes random state")
        print("  - Check for external random number generators")
        print("  - Ensure thread safety in multi-threaded environments")
        print("  - Validate NumPy random seed propagation")
        
        return False
    else:
        print("Deterministic behavior verified: All trials produce identical results")
        return True

# Test deterministic behavior
test_initializer = create_agent_initializer({
    'type': 'uniform_random',
    'bounds': (100, 100),
    'seed': 42
})

is_deterministic = test_deterministic_behavior(test_initializer, seed=42, trials=10)
```

This comprehensive API reference provides complete documentation for the agent initialization system, covering all aspects from basic usage to advanced customization and troubleshooting. The documentation supports the v1.0 transformation goals by enabling researchers to leverage the full extensibility and reproducibility features of the initialization framework.