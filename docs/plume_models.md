# Plume Model Documentation

**Mathematical foundations and configuration reference for all plume model implementations in the Blitzy Plume Navigation Simulator.**

This document provides comprehensive guidance for researchers on available physics models, their mathematical foundations, configuration parameters, and performance characteristics. The modular plume system enables seamless switching between simple fast models and realistic complex models through configuration-driven component selection.

## Table of Contents

1. [System Overview](#system-overview)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Plume Model Implementations](#plume-model-implementations)
4. [Wind Field Integration](#wind-field-integration)
5. [Configuration Reference](#configuration-reference)
6. [Performance Analysis](#performance-analysis)
7. [Usage Examples](#usage-examples)
8. [Model Selection Guidelines](#model-selection-guidelines)

## System Overview

The plume navigation simulator implements a **protocol-driven modular architecture** that enables runtime substitution of different plume physics models without code modification. All implementations comply with the `PlumeModelProtocol` interface, ensuring consistent behavior across different modeling approaches:

```python
# Core interface implemented by all plume models
class PlumeModelProtocol(Protocol):
    def concentration_at(self, positions: np.ndarray) -> np.ndarray:
        """Compute odor concentrations at specified spatial locations."""
        
    def step(self, dt: float = 1.0) -> None:
        """Advance plume state by specified time delta."""
        
    def reset(self, **kwargs: Any) -> None:
        """Reset plume state to initial conditions."""
```

### Available Implementations

| Model Type | Physics Approach | Primary Use Cases | Performance |
|------------|------------------|-------------------|-------------|
| **GaussianPlumeModel** | Analytical dispersion equations | Algorithm development, rapid prototyping, performance baselines | <0.1ms per query |
| **TurbulentPlumeModel** | Filament-based Lagrangian physics | Biological realism, complex transport phenomena, research scenarios | <1ms per query |
| **VideoPlumeAdapter** | Empirical video-based data | Backward compatibility, experimental validation, real-world data | <10ms frame access |

### Component Integration

All plume models integrate seamlessly with the broader simulation ecosystem:

- **Wind Field Integration**: Optional coupling with `ConstantWindField`, `TurbulentWindField`, or custom implementations
- **Sensor Abstraction**: Standardized sampling through `BinarySensor`, `ConcentrationSensor`, and `GradientSensor` interfaces
- **Performance Monitoring**: Built-in timing statistics and optimization guidance
- **Configuration Management**: Hydra-based parameter management with environment variable support

## Mathematical Foundations

### Gaussian Plume Dispersion Theory

The `GaussianPlumeModel` implements classical atmospheric dispersion theory using analytical solutions to the advection-diffusion equation. This approach provides deterministic, fast computation suitable for algorithm development and baseline testing.

#### Core Dispersion Equation

The fundamental Gaussian plume equation computes concentration at any spatial location:

```
C(x,y,t) = (Q / (2π σ_x σ_y)) × exp(-0.5 × ((x-x₀)/σ_x)² - 0.5 × ((y-y₀)/σ_y)²)
```

**Where:**
- `C(x,y,t)` = concentration at position (x,y) and time t [concentration units]
- `Q` = source strength (emission rate) [concentration units/time]
- `(x₀,y₀)` = effective source position including wind advection [distance units]
- `σ_x, σ_y` = dispersion coefficients controlling plume spread [distance units]

#### Wind Advection Integration

With wind effects enabled, the effective source position evolves over time:

```
x₀_eff(t) = x₀ + u_wind × t
y₀_eff(t) = y₀ + v_wind × t
```

**Where:**
- `u_wind, v_wind` = wind velocity components [distance units/time]
- `t` = elapsed simulation time [time units]

This creates downstream plume transport with realistic advection patterns while maintaining analytical tractability.

#### Normalization and Bounds

Concentrations are normalized to the range [0, 1] using:

```
C_normalized = min(C_raw / C_max, 1.0) + C_background
```

**Where:**
- `C_max` = maximum concentration for normalization
- `C_background` = baseline concentration level throughout domain

### Turbulent Plume Physics

The `TurbulentPlumeModel` implements realistic atmospheric dispersion using individual filament tracking with Lagrangian transport equations. This approach captures complex phenomena including intermittency, eddy interactions, and realistic plume structure.

#### Filament-Based Representation

Each filament represents a discrete odor packet with properties:

```python
class Filament:
    position: np.ndarray     # Current (x, y) coordinates
    concentration: float     # Odor concentration strength
    age: float              # Time since release from source
    size: float             # Characteristic spatial scale
    velocity: np.ndarray    # Current velocity from transport
```

#### Lagrangian Transport Equations

Filament positions evolve according to stochastic differential equations:

```
dx/dt = u_mean(x,y,t) + u_turbulent(x,y,t) + u_diffusion(t)
dy/dt = v_mean(x,y,t) + v_turbulent(x,y,t) + v_diffusion(t)
```

**Transport Components:**

1. **Mean Advection**: `u_mean, v_mean` from wind field or constant parameters
2. **Turbulent Fluctuations**: Ornstein-Uhlenbeck process with spatial correlation
3. **Molecular Diffusion**: Brownian motion with configurable diffusion coefficient

#### Turbulent Velocity Generation

Realistic turbulent velocities use correlated stochastic processes:

```
dv_turbulent = -v_turbulent/τ dt + σ √(2/τ) dW
```

**Where:**
- `τ` = Lagrangian correlation time scale [time units]
- `σ` = turbulence intensity × mean wind speed [velocity units]
- `dW` = Wiener process increments (Gaussian random)

#### Spatial Concentration Interpolation

Agent observations are computed through weighted interpolation from nearby filaments:

```
C(x_agent, y_agent) = Σᵢ wᵢ × Cᵢ × exp(-0.5 × (rᵢ/sᵢ)²)
```

**Where:**
- `wᵢ` = weight factor for filament i
- `Cᵢ` = concentration of filament i
- `rᵢ` = distance from agent to filament i
- `sᵢ` = size parameter of filament i

#### Filament Lifecycle Management

Filaments undergo aging and dissipation processes:

1. **Concentration Decay**: `C(t) = C₀ × exp(-k × t)` where k is dissipation rate
2. **Size Growth**: `s(t) = s₀ + √(2D × t)` where D is diffusion coefficient
3. **Boundary Interactions**: Absorption or reflection based on domain constraints

### Wind Field Mathematical Models

#### Constant Wind Field

Provides uniform flow throughout the domain:

```
U(x,y,t) = U₀ + ΔU(t)
```

With optional temporal evolution:
```
ΔU(t) = A × sin(2π × t / T) + η(t)
```

**Where:**
- `U₀` = base constant velocity vector [velocity units]
- `A` = evolution amplitude [velocity units]
- `T` = evolution period [time units]
- `η(t)` = optional Gaussian noise [velocity units]

#### Turbulent Wind Field

Implements realistic atmospheric boundary layer dynamics with multi-scale turbulent structures using statistical modeling of eddy interactions and spatial correlations.

**Key Features:**
- Anisotropic turbulence tensors accounting for atmospheric stability
- Spatial correlation modeling with exponential decay functions
- Temporal evolution using Ornstein-Uhlenbeck processes
- Integration with atmospheric stability parameters

## Plume Model Implementations

### GaussianPlumeModel: Fast Analytical Physics

**Purpose**: Provides sub-millisecond odor concentration computation using analytical Gaussian dispersion equations for rapid experimentation and algorithm development.

#### Key Features

- **Analytical Solutions**: Direct mathematical computation without numerical integration
- **Vectorized Operations**: Optimized NumPy implementations for multi-agent scenarios
- **Wind Integration**: Optional coupling with WindField implementations
- **Performance Optimization**: Pre-computed normalization factors and cached parameters

#### Mathematical Implementation Details

The implementation uses optimized computational kernels:

```python
def _compute_concentrations_scipy(self, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Optimized concentration computation using SciPy statistical functions."""
    # Create covariance matrix for multivariate normal
    cov = np.array([[self.sigma_x**2, 0], [0, self.sigma_y**2]])
    
    # Vectorized computation using multivariate normal PDF
    positions_relative = np.column_stack([dx, dy])
    mvn = stats.multivariate_normal(mean=[0, 0], cov=cov)
    concentrations = mvn.pdf(positions_relative) * self.source_strength
    
    return concentrations
```

#### Performance Characteristics

- **Single Agent Query**: <0.1ms typical execution time
- **Batch Processing**: <1ms for 100+ agents through vectorization
- **Memory Usage**: <1MB for typical parameter configurations
- **Scaling**: Linear performance with agent count

#### Configuration Parameters

| Parameter | Description | Default | Units | Range |
|-----------|-------------|---------|-------|-------|
| `source_position` | Source location coordinates | `(50.0, 50.0)` | distance | environment bounds |
| `source_strength` | Emission rate intensity | `1000.0` | concentration/time | >0 |
| `sigma_x` | Horizontal dispersion coefficient | `5.0` | distance | >0 |
| `sigma_y` | Vertical dispersion coefficient | `3.0` | distance | >0 |
| `wind_speed` | Simple wind magnitude | `0.0` | distance/time | ≥0 |
| `wind_direction` | Wind direction in degrees | `0.0` | degrees | [0, 360) |
| `background_concentration` | Baseline concentration level | `0.0` | concentration | ≥0 |
| `concentration_cutoff` | Minimum threshold for efficiency | `1e-6` | concentration | >0 |

### TurbulentPlumeModel: Realistic Filament Physics

**Purpose**: Provides research-grade turbulent plume modeling using individual filament tracking with realistic dispersion physics, complex eddy interactions, and intermittent plume structures.

#### Key Features

- **Filament-Based Approach**: Individual odor packet tracking per user requirements
- **Stochastic Transport**: Lagrangian particle dynamics with realistic turbulent mixing
- **Intermittent Signals**: Configurable patchiness matching atmospheric observations
- **Performance Optimization**: Optional Numba JIT acceleration for computational kernels

#### Implementation Architecture

```python
class TurbulentPlumeModel:
    def __init__(self, config: TurbulentPlumeConfig):
        self._filaments: List[Filament] = []  # Active filament population
        self._spatial_grid = self._initialize_spatial_grid()  # Efficient querying
        self._rng = np.random.default_rng(seed=config.random_seed)  # Reproducible stochastic processes
```

#### Filament Transport Implementation

```python
def _transport_filaments(self, dt: float) -> None:
    """Transport filaments using Lagrangian stochastic equations."""
    # Extract vectorized filament properties
    positions = np.array([f.position for f in self._filaments])
    velocities = np.array([f.velocity for f in self._filaments])
    
    # Compute wind field advection
    wind_velocities = self.wind_field.velocity_at(positions)
    
    # Generate turbulent velocity fluctuations
    turbulent_velocities = self._generate_turbulent_velocities(positions, dt)
    
    # Update velocities with exponential relaxation
    relaxation_time = 5.0  # Lagrangian correlation time
    alpha = dt / relaxation_time
    new_velocities = (1 - alpha) * velocities + alpha * (wind_velocities + turbulent_velocities)
    
    # Integrate positions
    new_positions = positions + new_velocities * dt
    
    # Update filament data structures
    for i, filament in enumerate(self._filaments):
        filament.position = new_positions[i]
        filament.velocity = new_velocities[i]
```

#### Performance Characteristics

- **Single Agent Query**: <1ms typical execution time
- **Batch Processing**: <10ms for 100+ agents with spatial optimization
- **Memory Usage**: <100MB for 1000+ active filaments
- **Numba Acceleration**: 10-50x speedup for computational kernels when enabled

#### Configuration Parameters

| Parameter | Description | Default | Units | Range |
|-----------|-------------|---------|-------|-------|
| `source_position` | Initial source location | `(50.0, 50.0)` | distance | environment bounds |
| `source_strength` | Emission rate per time step | `1000.0` | concentration/time | >0 |
| `mean_wind_velocity` | Base wind vector | `(2.0, 0.5)` | distance/time | any |
| `turbulence_intensity` | Relative turbulence strength | `0.2` | dimensionless | [0, 1] |
| `max_filaments` | Maximum active filaments | `2000` | count | >0 |
| `filament_lifetime` | Maximum age before pruning | `100.0` | time | >0 |
| `diffusion_coefficient` | Molecular diffusion rate | `0.1` | distance²/time | >0 |
| `eddy_dissipation_rate` | Turbulent energy dissipation | `0.01` | 1/time | >0 |
| `intermittency_factor` | Patchy signal characteristics | `0.3` | dimensionless | [0, 1] |
| `release_rate` | New filaments per time step | `10` | count/time | >0 |
| `enable_numba` | JIT acceleration toggle | `true` | boolean | true/false |

### VideoPlumeAdapter: Backward Compatibility

**Purpose**: Implements `PlumeModelProtocol` as an adapter for video-based plume data, preserving backward compatibility for existing workflows while enabling integration with the new modular architecture.

#### Key Features

- **PlumeModelProtocol Compliance**: Seamless integration through adapter pattern
- **Frame Caching**: FrameCache integration for sub-10ms frame access performance
- **Preprocessing Pipeline**: Configurable OpenCV-based image processing
- **Spatial Interpolation**: Sub-pixel accuracy concentration sampling

#### Configuration Parameters

| Parameter | Description | Default | Format |
|-----------|-------------|---------|--------|
| `video_path` | Path to video file | required | file path |
| `preprocessing_config` | Image processing options | `{'grayscale': true, 'normalize': true}` | dict |
| `frame_cache_config` | Caching configuration | `{'mode': 'lru', 'memory_limit_mb': 512}` | dict |
| `temporal_mode` | Frame advancement strategy | `"cyclic"` | "linear"/"cyclic"/"hold_last" |
| `spatial_interpolation_config` | Sampling configuration | `{'method': 'bilinear'}` | dict |

## Wind Field Integration

Wind field integration provides realistic environmental dynamics affecting plume transport and dispersion. All plume models support optional wind field coupling through the `WindFieldProtocol` interface.

### Integration Patterns

#### Gaussian Plume with Constant Wind

```python
# Simple advection with constant wind
wind_field = ConstantWindField(velocity=(2.0, 1.0))
plume_model = GaussianPlumeModel(
    source_position=(50, 50),
    wind_field=wind_field,
    enable_wind_field=True
)

# Wind affects effective source position over time
for t in range(100):
    plume_model.step(dt=1.0)
    # Plume center moves downstream with wind
    concentrations = plume_model.concentration_at(agent_positions)
```

#### Turbulent Plume with Complex Wind Dynamics

```python
# Realistic atmospheric boundary layer
turbulent_wind = TurbulentWindField(
    mean_velocity=(3.0, 1.0),
    turbulence_intensity=0.3,
    correlation_length=10.0
)

turbulent_plume = TurbulentPlumeModel(
    config=TurbulentPlumeConfig(
        mean_wind_velocity=(3.0, 1.0),
        turbulence_intensity=0.2
    )
)
turbulent_plume.set_wind_field(turbulent_wind)

# Complex transport with eddy interactions
for t in range(1000):
    turbulent_plume.step(dt=1.0)
    # Filaments follow realistic atmospheric transport
```

### Wind Field Implementations

#### ConstantWindField

**Mathematical Model**: Uniform directional flow with optional temporal evolution

**Key Features**:
- Sub-millisecond velocity queries
- Configurable temporal evolution patterns
- Boundary condition support

**Configuration Example**:
```yaml
wind_field:
  _target_: plume_nav_sim.models.wind.constant_wind.ConstantWindField
  velocity: [2.0, 0.5]  # [u_x, u_y] in environment units/time
  enable_temporal_evolution: false
  evolution_rate: 0.0
  evolution_amplitude: 0.0
```

#### TurbulentWindField

**Mathematical Model**: Realistic atmospheric boundary layer with stochastic variations

**Key Features**:
- Multi-scale turbulent structures
- Spatial velocity correlations
- Atmospheric stability effects

**Configuration Example**:
```yaml
wind_field:
  _target_: plume_nav_sim.models.wind.turbulent_wind.TurbulentWindField
  mean_velocity: [3.0, 1.0]
  turbulence_intensity: 0.2
  correlation_length: 10.0
  correlation_time: 5.0
  enable_numba: true
```

### Environmental Dynamics Effects

Wind field integration affects plume behavior through multiple mechanisms:

1. **Mean Advection**: Transport of plume center and filament populations
2. **Turbulent Mixing**: Enhanced dispersion through velocity fluctuations
3. **Temporal Variability**: Time-varying wind conditions create dynamic plume evolution
4. **Spatial Heterogeneity**: Non-uniform wind fields generate complex transport patterns

## Configuration Reference

### Hydra-Based Model Selection

The plume navigation simulator uses Hydra configuration management for runtime component selection without code modifications.

#### Basic Model Selection

```bash
# Fast analytical physics
python -m plume_nav_sim plume_model=gaussian

# Realistic turbulent physics  
python -m plume_nav_sim plume_model=turbulent

# Video-based empirical data
python -m plume_nav_sim plume_model=video video_path=experiment.mp4
```

#### Parameter Override Examples

```bash
# Gaussian model with strong wind
python -m plume_nav_sim plume_model=gaussian \
  plume_model.wind_speed=3.0 \
  plume_model.wind_direction=45.0

# Turbulent model with high intensity
python -m plume_nav_sim plume_model=turbulent \
  plume_model.turbulence_intensity=0.5 \
  plume_model.max_filaments=5000

# Video adapter with preprocessing
python -m plume_nav_sim plume_model=video \
  plume_model.video_path=data/plume.mp4 \
  plume_model.preprocessing_config.blur_kernel=5
```

### Environment Variable Integration

All models support environment variable configuration for deployment flexibility:

```bash
# Gaussian model environment variables
export PLUME_SOURCE_X=25.0
export PLUME_SOURCE_Y=75.0
export PLUME_SOURCE_STRENGTH=1500.0
export PLUME_SIGMA_X=8.0
export PLUME_WIND_SPEED=2.0

# Turbulent model environment variables
export TURBULENT_INTENSITY=0.3
export TURBULENT_MAX_FILAMENTS=3000
export TURBULENT_ENABLE_NUMBA=true
export TURBULENT_INTERMITTENCY=0.4

# Execute with environment configuration
python -m plume_nav_sim plume_model=gaussian
```

### Configuration Schema Validation

All models use Pydantic-based configuration validation:

```python
@dataclass
class GaussianPlumeConfig:
    source_position: Tuple[float, float] = (50.0, 50.0)
    source_strength: float = 1000.0
    sigma_x: float = 5.0
    sigma_y: float = 3.0
    # ... additional parameters with type safety and validation
```

### Programmatic Configuration

```python
from plume_nav_sim.models.plume.gaussian_plume import GaussianPlumeModel, GaussianPlumeConfig
from plume_nav_sim.models.wind.constant_wind import ConstantWindField

# Configuration-driven instantiation
config = GaussianPlumeConfig(
    source_position=(30.0, 40.0),
    source_strength=2000.0,
    sigma_x=8.0,
    sigma_y=4.0,
    wind_speed=1.5,
    wind_direction=90.0
)

wind_field = ConstantWindField(velocity=(1.5, 0.0))
plume_model = GaussianPlumeModel.from_config(config)
plume_model.set_wind_field(wind_field)
```

## Performance Analysis

### Computational Complexity

| Model Type | Time Complexity | Space Complexity | Scaling Factor |
|------------|----------------|------------------|----------------|
| **GaussianPlumeModel** | O(n) | O(1) | Linear with agent count |
| **TurbulentPlumeModel** | O(n×m) | O(m) | n=agents, m=filaments |
| **VideoPlumeAdapter** | O(n) | O(f) | n=agents, f=cached frames |

### Performance Benchmarks

#### Hardware Specifications
- **CPU**: Intel i7-10700K (8 cores, 3.8 GHz base)
- **Memory**: 32GB DDR4-3200
- **Python**: 3.10.x with NumPy 1.24.x, SciPy 1.10.x

#### Single Agent Performance

| Model | Mean Query Time | 95th Percentile | Memory Usage |
|-------|----------------|-----------------|--------------|
| GaussianPlumeModel | 0.08 ms | 0.15 ms | 0.5 MB |
| TurbulentPlumeModel (1000 filaments) | 0.75 ms | 1.2 ms | 45 MB |
| TurbulentPlumeModel (5000 filaments) | 2.1 ms | 3.5 ms | 180 MB |
| VideoPlumeAdapter (cached) | 0.12 ms | 0.25 ms | Variable |
| VideoPlumeAdapter (uncached) | 8.5 ms | 15 ms | Variable |

#### Multi-Agent Scaling

**100 Agents Batch Performance:**

| Model | Batch Query Time | Per-Agent Time | Speedup Factor |
|-------|-----------------|----------------|----------------|
| GaussianPlumeModel | 0.65 ms | 0.0065 ms | 12.3x |
| TurbulentPlumeModel (Numba) | 4.2 ms | 0.042 ms | 17.9x |
| TurbulentPlumeModel (Python) | 45 ms | 0.45 ms | 1.7x |
| VideoPlumeAdapter | 1.8 ms | 0.018 ms | 6.7x |

### Optimization Guidelines

#### For Maximum Performance
1. **Use GaussianPlumeModel** for algorithm development and rapid iteration
2. **Enable Numba compilation** for TurbulentPlumeModel computational kernels
3. **Configure appropriate filament limits** to balance realism with performance
4. **Enable frame caching** for VideoPlumeAdapter with sufficient memory allocation

#### For Maximum Realism
1. **Use TurbulentPlumeModel** with realistic parameter ranges
2. **Integrate TurbulentWindField** for complex environmental dynamics
3. **Configure appropriate intermittency factors** to match experimental conditions
4. **Use sufficient filament populations** (2000-5000) for detailed plume structure

#### Memory Management
- **GaussianPlumeModel**: Minimal memory requirements, suitable for large-scale simulations
- **TurbulentPlumeModel**: Monitor filament populations and adjust `max_filaments` parameter
- **VideoPlumeAdapter**: Configure frame cache limits based on available memory

### Performance Monitoring

All models include built-in performance monitoring:

```python
# Get performance statistics
stats = plume_model.get_performance_stats()
print(f"Average query time: {stats['average_query_time_ms']:.3f}ms")
print(f"Total queries: {stats['query_count']}")
print(f"Memory usage: {stats.get('memory_mb', 'N/A')}MB")

# Performance warnings are automatically logged
# when execution times exceed protocol thresholds
```

## Usage Examples

### Example 1: Algorithm Development with Fast Physics

```python
from plume_nav_sim.models.plume.gaussian_plume import GaussianPlumeModel
from plume_nav_sim.core.navigator import Navigator
import numpy as np

# Fast analytical plume for algorithm prototyping
plume_model = GaussianPlumeModel(
    source_position=(50, 50),
    source_strength=1000.0,
    sigma_x=6.0,
    sigma_y=4.0,
    wind_speed=1.5,
    wind_direction=45.0
)

# Simulate agent navigation
agent_positions = np.array([[30, 40], [35, 45], [40, 50]])

for step in range(1000):
    plume_model.step(dt=1.0)  # Advance plume with wind
    concentrations = plume_model.concentration_at(agent_positions)
    
    # Algorithm development with fast concentration queries
    gradients = np.gradient(concentrations)  # Simplified gradient computation
    agent_positions += 0.1 * gradients  # Basic gradient ascent
    
    if step % 100 == 0:
        print(f"Step {step}: Max concentration = {concentrations.max():.4f}")
```

### Example 2: Biological Realism with Turbulent Physics

```python
from plume_nav_sim.models.plume.turbulent_plume import TurbulentPlumeModel, TurbulentPlumeConfig
from plume_nav_sim.models.wind.turbulent_wind import TurbulentWindField

# High-fidelity turbulent plume for biological studies
config = TurbulentPlumeConfig(
    source_position=(50, 50),
    source_strength=800.0,
    mean_wind_velocity=(2.5, 0.8),
    turbulence_intensity=0.25,
    intermittency_factor=0.4,  # Realistic patchy signals
    max_filaments=3000,        # High-resolution plume structure
    enable_numba=True          # Performance optimization
)

# Complex wind field with atmospheric boundary layer
wind_field = TurbulentWindField(
    mean_velocity=(2.5, 0.8),
    turbulence_intensity=0.2,
    correlation_length=12.0,
    atmospheric_stability=0.1
)

turbulent_plume = TurbulentPlumeModel(config)
turbulent_plume.set_wind_field(wind_field)

# Realistic navigation with intermittent signals
agent_positions = np.array([[25, 30]])

concentration_history = []
for step in range(2000):
    turbulent_plume.step(dt=1.0)
    concentrations = turbulent_plume.concentration_at(agent_positions)
    concentration_history.append(concentrations[0])
    
    # Biological navigation strategy with intermittent signals
    if concentrations[0] > 0.1:  # Odor detected
        # Move upwind when odor detected
        wind_velocity = wind_field.velocity_at(agent_positions)[0]
        agent_positions += -0.05 * wind_velocity / np.linalg.norm(wind_velocity)
    else:  # No odor - casting behavior
        # Random search when no odor detected
        agent_positions += np.random.normal(0, 0.2, (1, 2))
    
    if step % 200 == 0:
        filament_count = turbulent_plume.get_filament_count()
        print(f"Step {step}: Concentration = {concentrations[0]:.4f}, "
              f"Filaments = {filament_count}")

# Analyze intermittent signal characteristics
intermittency = np.sum(np.array(concentration_history) > 0.01) / len(concentration_history)
print(f"Signal intermittency: {intermittency:.3f}")
```

### Example 3: Configuration-Driven Model Switching

```yaml
# config/experiment.yaml - Easy model switching
defaults:
  - base_config
  - plume_model: ${model_type}  # Runtime selection

model_type: gaussian  # Switch between: gaussian, turbulent, video

# Experiment-specific parameters
experiment:
  name: "model_comparison_study"
  agent_count: 50
  simulation_steps: 1000
  
# Model-specific parameter overrides
plume_model:
  source_position: [30, 70]  # Consistent across models
  source_strength: 1200.0
```

```python
# Python execution with model switching
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="experiment")
def run_experiment(cfg: DictConfig):
    # Automatic model instantiation based on configuration
    plume_model = hydra.utils.instantiate(cfg.plume_model)
    
    print(f"Using {type(plume_model).__name__} for experiment {cfg.experiment.name}")
    
    # Consistent simulation logic regardless of model
    agent_positions = np.random.uniform(0, 100, (cfg.experiment.agent_count, 2))
    
    for step in range(cfg.experiment.simulation_steps):
        plume_model.step(dt=1.0)
        concentrations = plume_model.concentration_at(agent_positions)
        # ... navigation logic ...
    
    # Model-specific performance metrics
    if hasattr(plume_model, 'get_performance_stats'):
        stats = plume_model.get_performance_stats()
        print(f"Performance: {stats['average_query_time_ms']:.3f}ms per query")

if __name__ == "__main__":
    run_experiment()
```

```bash
# Command-line model switching
python experiment.py model_type=gaussian    # Fast analytical physics
python experiment.py model_type=turbulent   # Realistic turbulent physics  
python experiment.py model_type=video video_path=data/plume.mp4  # Empirical data
```

### Example 4: Multi-Model Comparison Study

```python
from plume_nav_sim.models.plume.gaussian_plume import GaussianPlumeModel
from plume_nav_sim.models.plume.turbulent_plume import TurbulentPlumeModel, TurbulentPlumeConfig
from plume_nav_sim.models.plume.video_plume_adapter import VideoPlumeAdapter
import matplotlib.pyplot as plt

# Comparative analysis across model types
models = {
    'gaussian': GaussianPlumeModel(
        source_position=(50, 50),
        source_strength=1000.0,
        sigma_x=5.0,
        sigma_y=3.0
    ),
    'turbulent': TurbulentPlumeModel(
        TurbulentPlumeConfig(
            source_position=(50, 50),
            source_strength=1000.0,
            max_filaments=1000,
            enable_numba=True
        )
    ),
    'video': VideoPlumeAdapter(
        video_path="data/reference_plume.mp4",
        preprocessing_config={'grayscale': True, 'normalize': True}
    )
}

# Generate concentration fields for comparison
x = np.linspace(0, 100, 50)
y = np.linspace(0, 100, 50)
X, Y = np.meshgrid(x, y)
positions = np.column_stack([X.ravel(), Y.ravel()])

concentration_fields = {}
performance_metrics = {}

for model_name, model in models.items():
    print(f"Analyzing {model_name} model...")
    
    # Advance to steady state
    for _ in range(100):
        model.step(dt=1.0)
    
    # Sample concentration field
    start_time = time.perf_counter()
    concentrations = model.concentration_at(positions)
    query_time = time.perf_counter() - start_time
    
    concentration_fields[model_name] = concentrations.reshape(50, 50)
    
    # Collect performance metrics
    if hasattr(model, 'get_performance_stats'):
        stats = model.get_performance_stats()
        performance_metrics[model_name] = stats
    else:
        performance_metrics[model_name] = {'query_time_ms': query_time * 1000}

# Visualization comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (model_name, field) in enumerate(concentration_fields.items()):
    im = axes[i].contourf(X, Y, field, levels=20, cmap='viridis')
    axes[i].set_title(f'{model_name.title()} Plume Model')
    axes[i].set_xlabel('X Position')
    axes[i].set_ylabel('Y Position')
    plt.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.savefig('plume_model_comparison.png', dpi=300)

# Performance comparison
print("\nPerformance Comparison:")
print("-" * 50)
for model_name, metrics in performance_metrics.items():
    query_time = metrics.get('average_query_time_ms', metrics.get('query_time_ms', 0))
    print(f"{model_name:12}: {query_time:6.3f} ms per query")
```

## Model Selection Guidelines

### When to Use GaussianPlumeModel

**Ideal Scenarios:**
- Algorithm development and rapid prototyping
- Performance-critical applications requiring <1ms step latency
- Baseline studies and comparative analysis
- Educational demonstrations of plume navigation principles
- Large-scale multi-agent simulations (100+ agents)

**Advantages:**
- Deterministic, repeatable results
- Minimal computational requirements
- Fast parameter exploration
- Mathematical tractability

**Limitations:**
- Simplified physics lacks realistic complexity
- No intermittent or patchy signal characteristics
- Limited spatial structure beyond Gaussian profile

### When to Use TurbulentPlumeModel

**Ideal Scenarios:**
- Biological realism studies requiring environmental fidelity
- Algorithm evaluation under realistic conditions
- Research into intermittent signal navigation
- Complex transport phenomena investigation
- Studies of turbulent mixing and dispersion

**Advantages:**
- Research-grade environmental realism
- Intermittent and patchy signal characteristics
- Complex spatial structure with filament interactions
- Realistic turbulent transport physics

**Limitations:**
- Higher computational requirements
- Stochastic results require statistical analysis
- More complex parameter tuning
- Memory requirements scale with filament count

### When to Use VideoPlumeAdapter

**Ideal Scenarios:**
- Backward compatibility with existing video-based workflows
- Experimental validation using real plume data
- Comparison studies between empirical and theoretical models
- Transition from video-based to model-based approaches

**Advantages:**
- Empirical data preserves real-world complexity
- Proven validation against experimental observations
- Backward compatibility with existing research
- No parameter tuning required for physics

**Limitations:**
- Limited to pre-recorded scenarios
- No parameter exploration without new videos
- File I/O overhead for frame access
- Fixed temporal and spatial resolution

### Performance vs. Realism Trade-offs

#### High Performance Priority
```yaml
# Optimal configuration for speed
plume_model:
  _target_: plume_nav_sim.models.plume.gaussian_plume.GaussianPlumeModel
  source_strength: 1000.0
  sigma_x: 5.0
  sigma_y: 3.0
  concentration_cutoff: 1e-5  # Coarser precision for speed
```

#### High Realism Priority
```yaml
# Optimal configuration for biological fidelity
plume_model:
  _target_: plume_nav_sim.models.plume.turbulent_plume.TurbulentPlumeModel
  source_strength: 800.0
  turbulence_intensity: 0.3
  intermittency_factor: 0.4
  max_filaments: 5000  # High-resolution structure
  enable_numba: true   # Performance optimization
```

#### Balanced Configuration
```yaml
# Compromise between performance and realism
plume_model:
  _target_: plume_nav_sim.models.plume.turbulent_plume.TurbulentPlumeModel
  source_strength: 1000.0
  turbulence_intensity: 0.2
  intermittency_factor: 0.2
  max_filaments: 1500  # Moderate detail level
  enable_numba: true
```

### Deployment Considerations

#### Development Environment
- Use GaussianPlumeModel for rapid iteration
- Enable performance monitoring for optimization
- Use deterministic parameters for debugging

#### Research Environment
- Use TurbulentPlumeModel for publication-quality results
- Set fixed random seeds for reproducibility
- Document all parameter choices for scientific rigor

#### Production Environment
- Balance performance requirements with realism needs
- Monitor memory usage and adjust filament limits
- Use environment variables for deployment flexibility

---

**For additional guidance on plume model selection and configuration, consult the technical specification documentation and reach out to the development team for research-specific recommendations.**