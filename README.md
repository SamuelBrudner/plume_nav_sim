# Plume Navigation Simulation Library

A reusable Python library for simulating agent navigation through odor plumes with sophisticated Hydra-based configuration management, designed for integration with Kedro pipelines, reinforcement learning frameworks, and machine learning/neural network analyses.

## Overview

The Plume Navigation Simulation library (v1.0.0) provides a comprehensive toolkit for research-grade simulation of how agents navigate through odor plumes. Designed as an importable library, it offers clean APIs, protocol-based component architecture, and enterprise-grade configuration management for seamless integration into research workflows.

### Key Features

#### Core Architecture (v1.0 - Protocol-Based Component System)
- **Protocol-Based Component Architecture**: Revolutionary v1.0 architecture enabling pluggable sources, boundary policies, action interfaces, recorders, and statistics aggregators through Python protocols with runtime component swapping
- **Zero-Code Extensibility**: Add new odor sources, boundary behaviors, data recorders, and analysis metrics through protocol implementation without core system modification
- **Agent-Agnostic Design**: Memory-based and memory-less navigation strategies supported through configuration without code changes
- **Configuration-Driven Component Selection**: Complete simulation customization through Hydra configuration files with comprehensive recording and analysis pipeline integration
- **Comprehensive Recording System**: Multi-backend data persistence (parquet, HDF5, SQLite) with performance-aware buffering achieving <1ms disabled-mode overhead
- **Automated Statistics Collection**: Built-in metrics calculation and summary.json export for research reproducibility and cross-project comparison

#### Plume Modeling Capabilities
- **GaussianPlumeModel**: Fast analytical plume modeling with mathematical precision for rapid prototyping and training
- **TurbulentPlumeModel**: Realistic filament-based turbulent physics simulation with optional Numba acceleration
- **FilamentBasedPlumeModel**: Biologically-accurate intermittent odor signals matching real-world plume dynamics  
- **VideoPlumeAdapter**: Backward-compatible video-based plume processing with high-performance frame caching

#### Environmental Dynamics
- **Wind Field System**: Configurable wind dynamics (constant, turbulent, time-varying) integrated with plume transport
- **Sensor Abstraction**: Binary, concentration, and gradient sensors with optional temporal history
- **Multi-Modal Sensing**: Flexible perception systems supporting diverse research scenarios

#### v1.0 Component Architecture System
- **SourceProtocol Implementation**: Pluggable odor source modeling with PointSource, MultiSource, and DynamicSource implementations for flexible experiment design
- **BoundaryPolicyProtocol Support**: Configurable domain boundary handling with terminate, bounce, wrap, and clip policies for diverse experimental scenarios
- **ActionInterfaceProtocol Abstraction**: Unified action processing supporting Continuous2D and CardinalDiscrete implementations with runtime selection
- **RecorderProtocol Framework**: Comprehensive data persistence with parquet, HDF5, SQLite, and none backends achieving ≤33ms step latency with 100 concurrent agents
- **StatsAggregatorProtocol Integration**: Automated research metrics calculation with episode-level and run-level aggregation and standardized summary.json export
- **AgentInitializerProtocol Flexibility**: Configurable agent starting positions with uniform_random, grid, fixed_list, and from_dataset strategies
- **Performance Optimization**: All v1.0 components maintain ≤33ms step latency target with 100 concurrent agents through optimized algorithms and buffering

#### Framework Integration
- **Reusable Library Architecture**: Import and use in any Python project
- **Modern Gymnasium 0.29.x Compliance**: Full compatibility with modern RL frameworks and APIs, replacing legacy OpenAI Gym
- **Backward Compatibility**: Legacy Gym support through compatibility shims with VideoPlumeAdapter preservation
- **High-Performance Frame Caching**: Configurable LRU frame caching system with ≤2 GiB RAM constraints and sub-10ms step times
- **Extensible Hook System**: Overridable hooks for custom observations, rewards, and episode handling
- **Hydra Configuration Management**: Sophisticated hierarchical configuration with environment variable integration
- **Multi-Framework Integration**: Compatible with Kedro, RL frameworks, and ML/neural network analyses
- **CLI Interface**: Command-line tools for automation, batch processing, and RL training workflows
- **Docker-Ready**: Containerized development and deployment environments

- **Research-Grade Quality**: Type-safe, well-documented, and thoroughly tested

## Modular Architecture Overview

### Protocol-Driven Plugin System

The Plume Navigation Simulation library features a revolutionary **protocol-driven plugin architecture** that transforms odor plume navigation research from rigid, single-implementation systems into a flexible, extensible framework supporting diverse research scenarios through configuration-driven component swapping.

#### Core Architectural Principles

**Plugin-Oriented Design**: Every major simulation component implements Python protocols, enabling seamless substitution without code modifications:

```python
# Switch between plume models purely through configuration
from plume_nav_sim.api.navigation import create_gymnasium_environment

# Mathematical modeling for fast experimentation
env_gaussian = create_gymnasium_environment(plume_model="gaussian")

# Realistic physics for publication-quality research
env_turbulent = create_gymnasium_environment(plume_model="turbulent")

# Video-based modeling for existing datasets
env_video = create_gymnasium_environment(plume_model="video")
```

**Memory-Agnostic Navigation**: The enhanced `NavigatorProtocol` supports both memory-based and memory-less strategies through optional extensibility hooks:

```python
# Memory-less reactive agent (pure stimulus-response)
navigator_config = {
    "type": "reactive",
    "memory_enabled": False,
    "strategy": "gradient_following"
}

# Memory-based cognitive agent (with internal state)
navigator_config = {
    "type": "cognitive", 
    "memory_enabled": True,
    "strategy": "infotaxis",
    "memory_size": 1000
}
```

### Component Architecture

#### Plume Model Ecosystem

**GaussianPlumeModel** - Mathematical Precision
- Analytical solutions for rapid prototyping and algorithm development
- Sub-millisecond computation time for real-time applications  
- Configurable source strength, diffusion coefficients, and wind integration
- Perfect for initial research and high-speed training scenarios

**TurbulentPlumeModel** - Realistic Physics  
- Filament-based turbulent dispersion modeling
- Intermittent, patchy odor signals matching biological observations
- Optional Numba JIT acceleration for computationally intensive scenarios
- Ideal for publication-quality research requiring environmental fidelity

**FilamentBasedPlumeModel** - Biological Accuracy
- Individual odor packets with age, concentration, and transport dynamics
- Stochastic wind field integration for realistic temporal variations
- Sparse, intermittent detection patterns observed in natural environments
- Essential for biologically-inspired navigation algorithm validation

**VideoPlumeAdapter** - Legacy Compatibility
- Backward-compatible processing of existing video datasets
- High-performance frame caching with LRU and preload strategies
- Seamless integration with existing research workflows
- Maintains all performance optimizations from previous versions

#### Environmental Dynamics System

**Wind Field Integration**: 
```yaml
# Configuration-driven wind field selection
wind_field:
  type: "turbulent"  # Options: constant, turbulent, time_varying
  mean_velocity: [2.0, 0.5]  # Base wind vector
  turbulence_intensity: 0.3  # Gust strength
  correlation_time: 5.0      # Temporal coherence
```

**Sensor Abstraction Layer**:
```python
# Multiple sensing modalities through protocol implementation
sensor_config = {
    "primary": {"type": "concentration", "range": 10.0},
    "secondary": {"type": "binary", "threshold": 0.1},
    "gradient": {"type": "gradient", "spatial_resolution": 0.5}
}
```

### Configuration-Driven Research Workflows

#### Experiment Configuration Examples

**Comparative Memory Studies**:
```yaml
# conf/experiments/memory_comparison.yaml
defaults:
  - base_experiment
  - plume_models: gaussian  # Fast, consistent baseline
  - _self_

experiment:
  name: "memory_vs_memoryless_navigation"
  
  agents:
    memoryless:
      memory_enabled: false
      strategy: "gradient_following"
      
    memory_based:
      memory_enabled: true
      strategy: "infotaxis"
      memory_size: 500

  evaluation:
    episodes: 1000
    metrics: ["success_rate", "path_efficiency", "search_time"]
```

**Plume Model Validation Study**:
```yaml
# conf/experiments/plume_fidelity.yaml
plume_models:
  - type: "gaussian"
    config: {source_strength: 1.0, diffusion_rate: 0.1}
  - type: "turbulent" 
    config: {filament_count: 1000, dispersion_model: "lagrangian"}
  - type: "video"
    config: {video_path: "data/experimental_plume.mp4"}

comparison:
  metrics: ["spatial_distribution", "temporal_correlation", "concentration_statistics"]
  validation_data: "data/field_measurements.csv"
```

#### Command-Line Component Switching

```bash
# Run identical experiment with different plume models
plume-nav-sim run --config-name memory_study plume_model=gaussian
plume-nav-sim run --config-name memory_study plume_model=turbulent  
plume-nav-sim run --config-name memory_study plume_model=video

# Compare performance across environmental complexity
plume-nav-sim train --algorithm PPO \
    plume_model=gaussian \
    wind_field=constant \
    sensor_config=simple

plume-nav-sim train --algorithm PPO \
    plume_model=turbulent \
    wind_field=turbulent \
    sensor_config=multi_modal
```

#### v1.0 Component Configuration Examples

**Boundary Policy Configuration**
```yaml
# conf/boundary/terminate.yaml - End episode when agent exits domain
boundary:
  _target_: plume_nav_sim.core.boundaries.TerminateBoundary
  status_on_violation: "oob"  # Status reported when boundary violated
  domain_bounds: [100, 100]   # Domain size [width, height]

# conf/boundary/bounce.yaml - Reflect agent back into domain
boundary:
  _target_: plume_nav_sim.core.boundaries.BounceBoundary
  reflection_damping: 0.8     # Energy loss on bounce (0-1)
  domain_bounds: [100, 100]

# conf/boundary/wrap.yaml - Wrap agent to opposite side
boundary:
  _target_: plume_nav_sim.core.boundaries.WrapBoundary
  domain_bounds: [100, 100]
```

**Recorder Backend Configuration**
```yaml
# conf/record/parquet.yaml - High-performance columnar storage
record:
  _target_: plume_nav_sim.recording.backends.ParquetRecorder
  backend: parquet
  output_dir: ./data
  buffer_size: 1000
  compression: snappy         # Fast compression for real-time recording
  async_io: true             # Non-blocking writes
  full: true                 # Save per-step trajectories

# conf/record/hdf5.yaml - Scientific data format for complex structures
record:
  _target_: plume_nav_sim.recording.backends.HDF5Recorder
  backend: hdf5
  output_dir: ./data
  compression: gzip          # Better compression ratio
  chunking: true            # Optimized for partial reads

# conf/record/sqlite.yaml - Embedded database for transactional storage
record:
  _target_: plume_nav_sim.recording.backends.SQLiteRecorder
  backend: sqlite
  output_dir: ./data
  vacuum_on_close: true     # Optimize database on completion

# conf/record/none.yaml - Disabled recording for maximum performance
record:
  _target_: plume_nav_sim.recording.backends.NoneRecorder
  backend: none             # <1ms overhead when disabled
```

**Debug and Analysis Configuration**
```yaml
# conf/debug/interactive.yaml - Real-time debugging with GUI
debug:
  _target_: plume_nav_sim.debug.gui.DebugGUI
  backend: qt              # Options: qt, streamlit, auto
  config:
    refresh_rate: 30       # ≥30 FPS for real-time visualization
    enable_profiling: true # Performance monitoring
    performance_target_ms: 33.0  # ≤33ms step latency target
    show_inspector: true   # Detailed state inspection
    export_format: png

# conf/analysis/comprehensive.yaml - Automated statistics collection
analysis:
  _target_: plume_nav_sim.analysis.StatsAggregator
  metrics_definitions:
    trajectory: [mean, std, efficiency, tortuosity]
    concentration: [detection_rate, mean, percentiles]
    performance: [step_time_ms, memory_usage]
  aggregation_levels: [episode, run, batch]
  performance_tracking: true
  export_summary: true     # Automatic summary.json generation
```

### Performance Guarantees and Backward Compatibility

#### v1.0 Performance Standards
- **Step Latency**: ≤33ms for all v1.0 components with 100 concurrent agents (enhanced from v0.3.0's <10ms single-agent target)
- **Recording Overhead**: <1ms when disabled across all recorder backends
- **Memory Efficiency**: Linear scaling with agent count across all implementations and v1.0 components
- **Cache Hit Rates**: >90% frame cache efficiency for video-based models
- **Deterministic Behavior**: Identical results across runs with fixed random seeds
- **Component Performance**: All protocol-based components (sources, boundaries, actions, recorders, stats) maintain performance targets
- **Analysis Speed**: Statistics aggregation and summary generation within performance envelope

#### Backward Compatibility
- **VideoPlume Integration**: Existing video-based workflows preserved through `VideoPlumeAdapter`
- **API Compatibility**: All existing `VideoPlume` methods maintained with deprecation warnings
- **Configuration Migration**: Automatic conversion from legacy configuration formats
- **Performance Preservation**: Frame caching optimizations maintained for video models

#### Migration Path from v0.3.0 to v1.0.0

**Legacy v0.3.0 Approach (still supported with deprecation warnings):**
```python
# Legacy approach (still supported)
from plume_nav_sim import VideoPlume, Navigator
video_plume = VideoPlume("data/plume.mp4")
navigator = Navigator(position=(0, 0))

# Modern v0.3.0 modular approach  
from plume_nav_sim.api.navigation import create_gymnasium_environment
env = create_gymnasium_environment(
    plume_model="video",
    plume_config={"video_path": "data/plume.mp4"}
)
```

**New v1.0.0 Protocol-Based Approach (recommended):**
```python
# v1.0 comprehensive modular approach with full protocol integration
from plume_nav_sim.core.protocols import NavigatorFactory
from plume_nav_sim.recording import RecorderFactory
from plume_nav_sim.analysis import create_stats_aggregator
from plume_nav_sim.debug.gui import launch_viewer

# Create complete v1.0 environment with all components
env = NavigatorFactory.create_modular_environment(
    navigator_config={'position': (0, 0), 'max_speed': 2.0},
    plume_model_config={'type': 'GaussianPlumeModel'},
    source_config={'type': 'PointSource', 'position': (50, 50), 'emission_rate': 1000.0},
    boundary_policy_config={'type': 'TerminatePolicy', 'domain_bounds': (100, 100)},
    action_interface_config={'type': 'Continuous2D', 'max_linear_velocity': 2.0},
    recorder_config={'type': 'ParquetRecorder', 'output_dir': './data'},
    stats_aggregator_config={'type': 'StandardStatsAggregator'},
    agent_initializer_config={'type': 'UniformRandomInitializer', 'domain_bounds': (100, 100)}
)

# Launch interactive debugging
debug_gui = launch_viewer(env=env, backend='auto')
debug_gui.start_session()
```

**v1.0 Migration Benefits:**
- **Zero-Code Extensibility**: Add new components without modifying core library
- **Comprehensive Recording**: Automatic data persistence with multiple backend options
- **Built-in Analysis**: Automated statistics calculation and summary generation
- **Interactive Debugging**: Real-time visualization and step-through debugging capabilities
- **Enhanced Performance**: Optimized for ≤33ms step latency with 100 concurrent agents
- **Research Reproducibility**: Standardized output formats and correlation tracking

### Extensibility and Custom Development

#### Creating Custom v1.0 Components

**Custom Source Implementation**
```python
from plume_nav_sim.core.protocols import SourceProtocol
import numpy as np

class CustomOdorSource:
    """Custom odor source implementation following v1.0 SourceProtocol."""
    
    def __init__(self, position: tuple, emission_rate: float):
        self.position = position
        self.emission_rate = emission_rate
    
    def get_emission_rate(self, timestamp: float) -> float:
        """Get time-varying emission rate."""
        return self.emission_rate * (1 + 0.1 * np.sin(timestamp))
    
    def get_position(self) -> tuple:
        """Get source position."""
        return self.position
```

**Custom Boundary Policy Implementation**
```python
from plume_nav_sim.core.protocols import BoundaryPolicyProtocol
import numpy as np

class CustomBoundaryPolicy:
    """Custom boundary handling following v1.0 BoundaryPolicyProtocol."""
    
    def __init__(self, domain_bounds: tuple):
        self.domain_bounds = domain_bounds
    
    def check_violations(self, positions: np.ndarray) -> np.ndarray:
        """Check boundary violations for agent positions."""
        violations = np.zeros(len(positions), dtype=bool)
        violations |= (positions[:, 0] < 0) | (positions[:, 0] > self.domain_bounds[0])
        violations |= (positions[:, 1] < 0) | (positions[:, 1] > self.domain_bounds[1])
        return violations
    
    def apply_policy(self, positions: np.ndarray, violations: np.ndarray) -> np.ndarray:
        """Apply custom boundary policy to violating agents."""
        corrected_positions = positions.copy()
        # Custom boundary behavior implementation
        return corrected_positions
```

**Custom Recorder Implementation**
```python
from plume_nav_sim.core.protocols import RecorderProtocol
from typing import Dict, Any

class CustomDataRecorder:
    """Custom data recording following v1.0 RecorderProtocol."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.enabled = False
    
    def record_step(self, step_data: Dict[str, Any], step_number: int, 
                   episode_id: int = None, **metadata: Any) -> None:
        """Record step-level data with custom processing."""
        if not self.enabled:
            return  # <1ms overhead when disabled
        
        # Custom data processing and storage
        processed_data = self._process_step_data(step_data)
        self._write_to_custom_backend(processed_data)
    
    def record_episode(self, episode_data: Dict[str, Any], episode_id: int,
                      **metadata: Any) -> None:
        """Record episode-level summary data."""
        # Custom episode data handling
        pass
```

#### v1.0 Research Integration Patterns

```python
# Jupyter notebook integration with v1.0 protocol-based components
from plume_nav_sim.core.protocols import NavigatorFactory
from plume_nav_sim.recording import RecorderFactory
from plume_nav_sim.analysis import create_stats_aggregator, generate_summary
from plume_nav_sim.debug.gui import launch_viewer, plot_initial_state

# Create comprehensive v1.0 research environment
research_env = NavigatorFactory.create_modular_environment(
    navigator_config={'position': (0, 0), 'max_speed': 2.0},
    plume_model_config={'type': 'GaussianPlumeModel', 'source_strength': 1.0},
    source_config={'type': 'PointSource', 'position': (50, 50)},
    boundary_policy_config={'type': 'TerminatePolicy'},
    recorder_config={'backend': 'parquet', 'output_dir': './research_data'},
    stats_aggregator_config={'metrics_definitions': {'trajectory': ['efficiency', 'tortuosity']}}
)

# Interactive research with built-in analysis
debug_gui = launch_viewer(env=research_env, backend='auto')
plot_initial_state(research_env)  # Visualize experimental setup

# Automated data collection and analysis
obs, info = research_env.reset()
for step in range(1000):
    action = research_policy(obs)  # Your research algorithm
    obs, reward, terminated, truncated, info = research_env.step(action)
    
    # Automatic recording and performance monitoring
    if info.get('perf_stats', {}).get('step_time_ms', 0) > 33.0:
        print(f"⚠ Performance target exceeded at step {step}")
    
    if terminated or truncated:
        break

# Generate comprehensive research summary
episodes_data = [{'episode_id': 1, 'total_steps': step, 'final_reward': reward}]
summary = generate_summary(
    research_env._stats_aggregator,  # Integrated stats aggregator
    episodes_data,
    output_path='./research_summary.json'
)

print(f"Research completed: {summary['episode_count']} episodes analyzed")
print(f"Processing efficiency: {summary['summary_statistics']['episodes_per_second']:.1f} eps/s")
```

This v1.0 protocol-based architecture enables researchers to focus on algorithm development while automatically handling data collection, performance monitoring, and comprehensive analysis - all without requiring infrastructure modifications.

## Installation

### Prerequisites

- Python 3.10 or higher (minimum requirement upgraded for Gymnasium 0.29.x compatibility)
- pip for dependency management (setuptools>=61 used as build backend)
- Docker and docker-compose (optional, for containerized development)

### Installation Methods

#### Standard Installation

```bash
# Standard installation
pip install plume_nav_sim

# Development installation with optional dependencies
pip install "plume_nav_sim[dev,docs,viz]"

# Installation with reinforcement learning dependencies
pip install "plume_nav_sim[rl]"

# v1.0 Optional dependency groups for enhanced capabilities
pip install "plume_nav_sim[recording]"  # Multi-backend data persistence (pandas, pyarrow, h5py)
pip install "plume_nav_sim[debug]"      # Interactive debugging GUI (PySide6, streamlit)
pip install "plume_nav_sim[analysis]"   # Automated statistics collection (scipy, psutil)

# Full installation with all v1.0 components
pip install "plume_nav_sim[dev,docs,viz,rl,recording,debug,analysis]"
```

#### Development Installation

```bash
# Clone repository
git clone https://github.com/organization/plume_nav_sim.git
cd plume_nav_sim

# Development setup with editable install
pip install -e ".[dev,docs,viz,rl]"

# Or with virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev,docs,viz,rl]"
```

#### Docker-Based Development Environment

```bash
# Full development environment with database and pgAdmin
docker-compose up --build

# Library container only
docker build -t plume_nav_sim .
docker run -it plume_nav_sim
```

## Library Usage Patterns

### For Kedro Projects

#### Modern Modular Approach

```python
from plume_nav_sim.api.navigation import create_gymnasium_environment, create_plume_model
from plume_nav_sim.core.protocols import NavigatorFactory
from hydra import compose, initialize

# Kedro pipeline with modular architecture
def create_navigation_pipeline():
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config")
        
        # Create modular components with protocol-based architecture
        plume_model = create_plume_model(cfg.plume_model)
        navigator = NavigatorFactory.from_config(cfg.navigator)
        
        # Or create complete environment
        env = create_gymnasium_environment(cfg)
        
        return navigator, plume_model, env

# Kedro node supporting multiple plume models
def navigation_node(config: dict) -> dict:
    """Kedro node for modular plume navigation simulation."""
    
    # Support different research scenarios through configuration
    if config["experiment_type"] == "memory_comparison":
        results = run_memory_comparison_study(config)
    elif config["experiment_type"] == "plume_fidelity":
        results = run_plume_model_validation(config)
    else:
        results = run_standard_navigation(config)
    
    return {
        "trajectory": results.trajectory, 
        "sensor_data": results.sensor_data,
        "plume_model_type": config["plume_model"]["type"],
        "memory_enabled": config["navigator"]["memory_enabled"]
    }

def run_memory_comparison_study(config: dict) -> dict:
    """Compare memory-based vs memory-less navigation strategies."""
    
    results = {}
    
    # Memory-less agent
    memoryless_config = config.copy()
    memoryless_config["navigator"]["memory_enabled"] = False
    env_memoryless = create_gymnasium_environment(memoryless_config)
    results["memoryless"] = run_agent_evaluation(env_memoryless)
    
    # Memory-based agent
    memory_config = config.copy()
    memory_config["navigator"]["memory_enabled"] = True
    memory_config["navigator"]["memory_size"] = 1000
    env_memory = create_gymnasium_environment(memory_config)
    results["memory_based"] = run_agent_evaluation(env_memory)
    
    return results
```

#### Legacy Compatibility (Deprecated but Supported)

```python
from plume_nav_sim import Navigator, VideoPlume  # Legacy imports
from plume_nav_sim.config import NavigatorConfig
from hydra import compose, initialize
import warnings

# Legacy approach still supported with deprecation warnings
def create_legacy_navigation_pipeline():
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config")
        
        # Legacy components with deprecation warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            navigator = Navigator.from_config(cfg.navigator)
            video_plume = VideoPlume.from_config(cfg.video_plume)
        
        return navigator, video_plume
```

### For Reinforcement Learning Projects

#### Using the Modern Modular Gymnasium Environment

```python
from plume_nav_sim.api.navigation import create_gymnasium_environment
from plume_nav_sim.utils import set_global_seed
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

# Create modular environment with configurable components
def create_rl_environment(
    plume_model: str = "gaussian",
    memory_enabled: bool = False,
    config_path: str = "conf/config.yaml"
):
    """Create a Gymnasium environment with modular component selection."""
    
    # Set deterministic behavior for RL training
    set_global_seed(42)
    
    # Method 1: Direct component configuration
    env = create_gymnasium_environment(
        config_path=config_path,
        plume_model=plume_model,
        navigator_config={
            "memory_enabled": memory_enabled,
            "strategy": "gradient_following" if not memory_enabled else "infotaxis"
        }
    )
    
    # Method 2: Use registered environment with overrides
    env = gymnasium.make(
        'PlumeNavSim-v0',
        plume_model=plume_model,
        memory_enabled=memory_enabled
    )
    
    return env

# Research-oriented environment creation for different scenarios
def create_research_environments():
    """Create environments for systematic research comparison."""
    
    environments = {}
    
    # Fast training environment with mathematical plume model
    environments["gaussian_memoryless"] = create_gymnasium_environment(
        plume_model="gaussian",
        navigator_config={"memory_enabled": False, "strategy": "reactive"},
        wind_field="constant"
    )
    
    # Realistic environment with turbulent physics
    environments["turbulent_memory"] = create_gymnasium_environment(
        plume_model="turbulent", 
        navigator_config={"memory_enabled": True, "strategy": "infotaxis"},
        wind_field="turbulent",
        sensor_config={"type": "multi_modal", "noise_level": 0.1}
    )
    
    # Legacy video-based environment for comparison
    environments["video_baseline"] = create_gymnasium_environment(
        plume_model="video",
        plume_config={"video_path": "data/experimental_plume.mp4"},
        navigator_config={"memory_enabled": False}
    )
    
    return environments

# Train with stable-baselines3 using modular architecture
def train_rl_agent(
    plume_model: str = "gaussian",
    memory_enabled: bool = False,
    algorithm: str = "PPO"
):
    """Train an RL agent with configurable plume model and memory settings."""
    
    # Create modular environment
    env = create_rl_environment(
        plume_model=plume_model,
        memory_enabled=memory_enabled
    )
    
    # Verify environment compatibility
    from gymnasium.utils.env_checker import check_env
    check_env(env)
    
    # Create vectorized environment for training
    vec_env = DummyVecEnv([lambda: env])
    
    # Algorithm selection based on research needs
    if algorithm == "PPO":
        model = PPO(
            "MultiInputPolicy",  # Handle dict observations from modular system
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            verbose=1
        )
    elif algorithm == "SAC":
        model = SAC(
            "MultiInputPolicy",
            vec_env,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
            verbose=1
        )
    
    # Train the model with experiment tracking
    model.learn(total_timesteps=100000)
    
    # Save with descriptive naming
    model_name = f"{algorithm.lower()}_{plume_model}_memory{memory_enabled}"
    model.save(f"models/{model_name}")
    
    return model

# Comparative training across different configurations
def train_comparative_study():
    """Train agents across different modular configurations for research comparison."""
    
    configurations = [
        {"plume_model": "gaussian", "memory_enabled": False, "algorithm": "PPO"},
        {"plume_model": "gaussian", "memory_enabled": True, "algorithm": "PPO"},
        {"plume_model": "turbulent", "memory_enabled": False, "algorithm": "SAC"},
        {"plume_model": "turbulent", "memory_enabled": True, "algorithm": "SAC"},
        {"plume_model": "video", "memory_enabled": False, "algorithm": "PPO"}  # Baseline
    ]
    
    results = {}
    
    for config in configurations:
        print(f"Training {config['algorithm']} on {config['plume_model']} model "
              f"with memory={'enabled' if config['memory_enabled'] else 'disabled'}")
        
        model = train_rl_agent(**config)
        
        # Evaluate trained model
        evaluation_results = evaluate_agent(model, config)
        
        config_key = f"{config['plume_model']}_{config['memory_enabled']}_{config['algorithm']}"
        results[config_key] = {
            "model": model,
            "evaluation": evaluation_results,
            "config": config
        }
    
    return results

# Evaluate trained agent with Gymnasium 5-tuple API
def evaluate_agent(model_path: str = "ppo_plume_navigation"):
    """Evaluate a trained RL agent using modern Gymnasium API."""
    
    env = create_rl_environment()
    model = PPO.load(model_path)
    
    # Modern Gymnasium reset returns (observation, info)
    obs, info = env.reset(seed=42)
    total_reward = 0
    
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        # Modern Gymnasium step returns (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Total reward: {total_reward}")
    return total_reward
```

#### Advanced RL Integration

```python
from plume_nav_sim.environments.gymnasium_env import PlumeNavigationEnv
from plume_nav_sim.environments.wrappers import NormalizeObservation, RewardShaping
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import gymnasium as gym

# Custom environment configuration using modern Gymnasium API
def create_advanced_rl_setup():
    """Create advanced RL training setup with custom wrappers."""
    
    # Create base environment using the new PlumeNavSim-v0 environment
    base_env = gym.make('PlumeNavSim-v0', 
                       max_episode_steps=1000,
                       render_mode="rgb_array")
    
    # Alternative: Direct environment instantiation with custom parameters
    # base_env = PlumeNavigationEnv(
    #     video_path="data/complex_plume.mp4",
    #     max_episode_steps=1000,
    #     reward_shaping={'odor_weight': 1.0, 'distance_weight': 0.5, 'control_penalty': 0.1}
    # )
    
    # Apply preprocessing wrappers
    env = NormalizeObservation(base_env)
    env = RewardShaping(env, dense_rewards=True)
    
    return env

# Training with advanced callbacks
def train_advanced_agent():
    """Train agent with evaluation callbacks and checkpointing."""
    
    # Training and evaluation environments
    train_env = DummyVecEnv([create_advanced_rl_setup for _ in range(4)])
    eval_env = create_advanced_rl_setup()
    
    # Create SAC agent for continuous control
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=1000000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./checkpoints/",
        name_prefix="sac_plume_nav"
    )
    
    # Train with callbacks
    model.learn(
        total_timesteps=500000,
        callback=[eval_callback, checkpoint_callback]
    )
    
    return model
```

## Gymnasium API Compliance and Migration Guide

### Modern Gymnasium 0.29.x Integration

The Plume Navigation Simulation library provides full compliance with modern Gymnasium 0.29.x API standards while maintaining backward compatibility with legacy OpenAI Gym implementations.

#### New Environment Registration

**Primary Environment ID**: `PlumeNavSim-v0` (Gymnasium-native)
- Full Gymnasium 0.29.x API compliance with 5-tuple step returns
- Enhanced observation space with structured dictionary observations
- Modern reset signature: `reset(seed=None, options=None) -> (obs, info)`
- Optimized performance with sub-10ms step times

```python
import gymnasium as gym
from plume_nav_sim.api.navigation import create_gymnasium_environment

# Method 1: Direct environment registration
env = gym.make('PlumeNavSim-v0')

# Method 2: Factory function with configuration
env = create_gymnasium_environment("conf/config.yaml")

# Verify Gymnasium compliance
from gymnasium.utils.env_checker import check_env
check_env(env)  # Validates full API compliance
```

#### Enhanced Observation Space

The new environment provides structured dictionary observations for improved RL training:

```python
# Modern Gymnasium observation space
observation_space = gymnasium.spaces.Dict({
    'odor_concentration': gymnasium.spaces.Box(
        shape=(1,), low=0.0, high=1.0, dtype=np.float32
    ),
    'agent_position': gymnasium.spaces.Box(
        shape=(2,), low=0.0, high=100.0, dtype=np.float32
    ),
    'agent_orientation': gymnasium.spaces.Box(
        shape=(1,), low=-np.pi, high=np.pi, dtype=np.float32
    ),
    'plume_gradient': gymnasium.spaces.Box(
        shape=(2,), low=-1.0, high=1.0, dtype=np.float32
    )
})
```

### Backward Compatibility and Legacy Support

#### Legacy Gym Compatibility Shim

For projects still using legacy OpenAI Gym patterns, the library provides a compatibility layer:

```python
# Legacy support through compatibility shim
from plume_nav_sim.shims import gym_make
import warnings

# This will emit a deprecation warning but maintain functionality
env = gym_make("PlumeNavSim-v0")  # Proxies to gymnasium internally

# Legacy environment ID for existing code
env = gym_make("OdorPlumeNavigation-v1")  # Maintains 4-tuple returns
```

#### Automatic API Translation

The library automatically detects and converts between API formats:

```python
# Legacy 4-tuple returns (for backward compatibility)
obs, reward, done, info = env.step(action)  # Legacy format

# Modern 5-tuple returns (recommended)
obs, reward, terminated, truncated, info = env.step(action)  # Gymnasium format
done = terminated or truncated  # Convert to legacy if needed
```

### Migration Timeline and Deprecation Policy

**Current Version (v0.3.0)**: Full dual API support
- Both legacy Gym and modern Gymnasium APIs fully supported
- Deprecation warnings guide migration paths
- Zero breaking changes for existing code

**Future Version (v1.0)**: Legacy Gym removal
- Legacy Gym support will be removed in v1.0
- Modern Gymnasium API will be the only supported interface
- Migration period provides ample time for updates

### Performance Optimization Features

#### High-Performance Frame Caching

The new version includes sophisticated frame caching for optimal training performance:

```python
# Configure frame caching mode for optimal performance
env = create_gymnasium_environment(
    config_path="conf/config.yaml",
    frame_cache="lru"  # Options: "none", "lru", "all"
)

# Access performance metrics
obs, reward, terminated, truncated, info = env.step(action)
perf_stats = info["perf_stats"]
print(f"Cache hit rate: {perf_stats['cache_hit_rate']:.2%}")
print(f"Step time: {perf_stats['step_time_ms']:.2f}ms")
```

#### Extensible Hook System

New overridable hooks enable custom behavior without modifying core code:

```python
class CustomPlumeEnvironment(PlumeNavigationEnv):
    def compute_additional_obs(self, base_obs: dict) -> dict:
        """Add custom observations."""
        return {"custom_feature": self.compute_custom_feature()}
    
    def compute_extra_reward(self, base_reward: float, info: dict) -> float:
        """Add reward shaping."""
        return 0.1 * self.efficiency_bonus()
    
    def on_episode_end(self, final_info: dict) -> None:
        """Custom episode termination logic."""
        self.log_episode_statistics(final_info)
```

## Backward Compatibility and Migration Support

### Legacy Gym Support Through Compatibility Shims

For projects still using OpenAI Gym patterns, the library provides seamless compatibility through the shims module:

```python
# Legacy support with automatic deprecation warnings
from plume_nav_sim.shims import gym_make

# This will work exactly like the old gym.make() but emit helpful warnings
env = gym_make("PlumeNavSim-v0")  # Logs DeprecationWarning, proxies to gymnasium

# Legacy environment ID for existing code (maintains 4-tuple returns)
env = gym_make("OdorPlumeNavigation-v1")
```

### Migration Timeline and Deprecation Policy

**Current Version (v0.3.0)**: Full backward compatibility
- Both legacy Gym and modern Gymnasium APIs fully supported
- Zero breaking changes for existing codebases
- Deprecation warnings provide clear migration guidance
- Automatic API format detection and conversion

**Future Version (v1.0)**: Legacy Gym removal planned
- Legacy OpenAI Gym support will be removed in v1.0
- Modern Gymnasium 0.29.x will be the only supported interface
- Extended transition period provides ample time for migration
- Comprehensive migration tools and documentation available

### Migration Examples

**Minimal Changes Required for Most Code**:

```python
# Old approach (still works with deprecation warnings)
import gym
env = gym.make('OdorPlumeNavigation-v1')
obs = env.reset()
obs, reward, done, info = env.step(action)

# New approach (recommended)
import gymnasium as gym
env = gym.make('PlumeNavSim-v0')
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated  # Convert for legacy compatibility
```

### v1.0 Interactive Debugging and Analysis

#### Debug Utilities with Real-Time Visualization

```python
from plume_nav_sim.debug.gui import launch_viewer, DebugGUI, DebugConfig
from plume_nav_sim.debug.gui import plot_initial_state
from plume_nav_sim.api.navigation import create_gymnasium_environment

# Launch interactive Qt-based desktop debugging
debug_config = DebugConfig(
    backend='qt',
    refresh_rate=30,           # ≥30 FPS for real-time visualization
    enable_profiling=True,     # Performance monitoring
    performance_target_ms=33.0 # ≤33ms step latency target
)

env = create_gymnasium_environment(config_path="conf/config.yaml")
debug_gui = launch_viewer(env=env, config=debug_config)

# Interactive step-through debugging
debug_gui.add_breakpoint("odor_reading > 0.8")  # Conditional breakpoint
debug_gui.step_through()                        # Single-step execution
debug_gui.export_screenshots('./debug_exports') # Save visualization

# Plot initial experimental setup
fig = plot_initial_state(
    env=env,
    source=my_source,
    agent_positions=start_positions,
    domain_bounds=(0, 100, 0, 100),
    title="Experimental Setup - v1.0"
)

# Launch web-based collaborative debugging
web_debug_gui = DebugGUI(backend='streamlit')
web_debug_gui.configure_backend(port=8501, host='localhost')
web_debug_gui.show()  # Browser-based debugging interface
```

#### Comprehensive Recording and Analysis Pipeline

```python
from plume_nav_sim.recording import RecorderFactory, RecorderManager
from plume_nav_sim.analysis import create_stats_aggregator, generate_summary_report

# Create high-performance parquet recorder
recorder_config = {
    'backend': 'parquet',
    'output_dir': './experiment_data',
    'buffer_size': 1000,
    'compression': 'snappy',
    'async_io': True          # Non-blocking writes for real-time performance
}
recorder = RecorderFactory.create_recorder(recorder_config)

# Setup automated statistics collection
stats_config = {
    'metrics_definitions': {
        'trajectory': ['mean', 'std', 'efficiency', 'tortuosity'],
        'concentration': ['detection_rate', 'mean', 'percentiles'],
        'performance': ['step_time_ms', 'cache_hit_rate']
    },
    'aggregation_levels': ['episode', 'run'],
    'performance_tracking': True,
    'export_summary': True
}
stats_aggregator = create_stats_aggregator(stats_config)

# Integrated recording and analysis workflow
with RecorderManager(recorder) as rec_manager:
    rec_manager.start_recording(episode_id=1)
    
    # Run simulation with automatic data collection
    obs, info = env.reset()
    for step in range(1000):
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Automatic recording with performance monitoring
        step_data = {
            'observation': obs,
            'action': action,
            'reward': reward,
            'performance_stats': info.get('perf_stats', {})
        }
        recorder.record_step(step_data, step_number=step)
        
        # Real-time performance validation
        perf_metrics = rec_manager.get_performance_metrics()
        if not perf_metrics['performance_analysis']['meets_target']:
            print(f"⚠ Performance warning: {perf_metrics['performance_analysis']['avg_write_time_ms']:.1f}ms")
        
        if terminated or truncated:
            break
    
    # Automatic episode-level analysis
    episode_data = {
        'episode_id': 1,
        'total_steps': step,
        'final_reward': reward,
        'success': reward > 0.8
    }
    recorder.record_episode(episode_data, episode_id=1)

# Generate comprehensive summary report
episodes_data = [episode_data]  # Would contain multiple episodes
success = generate_summary_report(
    episodes_data,
    output_path='./results/experiment_summary.json',
    config=stats_config
)

if success:
    print("✓ Comprehensive analysis report generated with v1.0 statistics")
```

### For ML/Neural Network Analyses

```python
from plume_nav_sim.utils import set_global_seed
from plume_nav_sim.data import VideoPlume
from plume_nav_sim.api.navigation import run_plume_simulation
import torch
import numpy as np

# Neural network training data generation
def generate_training_data(num_episodes: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Generate training data for neural navigation models."""
    
    # Set reproducible seeds for ML workflows
    set_global_seed(42)
    
    # Load configuration with ML-optimized parameters
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config", overrides=[
            "simulation.recording.export_format=numpy",
            "performance.numpy.precision=float32"
        ])
    
    # Generate diverse navigation scenarios
    trajectories = []
    sensor_readings = []
    
    for episode in range(num_episodes):
        # Create randomized navigator for data diversity
        navigator = Navigator.from_config(cfg.navigator)
        navigator.position = np.random.uniform(0, 100, 2)
        
        # Run simulation
        results = run_plume_simulation(navigator, video_plume, cfg)
        
        trajectories.append(results.trajectory)
        sensor_readings.append(results.sensor_data)
    
    return np.array(trajectories), np.array(sensor_readings)

# PyTorch dataset integration
class NavigationDataset(torch.utils.data.Dataset):
    def __init__(self, config_path: str = "conf/config.yaml"):
        self.trajectories, self.sensor_data = generate_training_data()
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return {
            'trajectory': torch.tensor(self.trajectories[idx], dtype=torch.float32),
            'sensor_reading': torch.tensor(self.sensor_data[idx], dtype=torch.float32)
        }
```

## Reinforcement Learning

### Overview

The Plume Navigation Simulation library provides comprehensive reinforcement learning integration through the modern Gymnasium 0.29.x API, enabling seamless compatibility with state-of-the-art RL frameworks and algorithms. The RL integration transforms the existing simulation framework into a standardized RL environment while preserving all core navigation, sensing, and visualization capabilities with enhanced performance through intelligent frame caching.

### Gymnasium Environment

#### Environment Interface

The `PlumeNavigationEnv` class provides a fully Gymnasium-compliant environment implementing the standard `reset()`, `step()`, `render()`, and `close()` methods with performance optimizations:

```python
from plume_nav_sim.environments.gymnasium_env import PlumeNavigationEnv
import gymnasium as gym

# Method 1: Use registered environment ID (recommended)
env = gym.make('PlumeNavSim-v0', render_mode="human")

# Method 2: Direct instantiation with custom parameters
env = PlumeNavigationEnv(
    video_path="data/plume_video.mp4",
    max_episode_steps=1000,
    render_mode="human",  # or "rgb_array" for headless
    frame_cache="lru"     # Enable high-performance frame caching
)

# Standard Gymnasium workflow with performance monitoring
observation, info = env.reset(seed=42)
for step in range(1000):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Access performance statistics
    perf_stats = info.get("perf_stats", {})
    if step % 100 == 0:
        print(f"Step {step}: {perf_stats.get('step_time_ms', 0):.2f}ms, "
              f"Cache hit rate: {perf_stats.get('cache_hit_rate', 0):.2%}")
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

#### Action and Observation Spaces

**Action Space**: `gymnasium.spaces.Box`
- **Shape**: `(2,)` - Continuous control vector
- **Range**: `[-1.0, 1.0]` for both dimensions
- **Components**:
  - `action[0]`: Linear speed control (normalized)
  - `action[1]`: Angular velocity control (normalized)

**Observation Space**: `gymnasium.spaces.Dict`
- **odor_concentration**: `Box(shape=(1,), dtype=float32)` - Current sensor reading
- **agent_position**: `Box(shape=(2,), dtype=float32)` - Agent [x, y] coordinates
- **agent_orientation**: `Box(shape=(1,), dtype=float32)` - Agent heading in radians
- **plume_gradient**: `Box(shape=(2,), dtype=float32, low=-1.0, high=1.0)` - Estimated gradient direction (optional)

```python
# Access observation components
obs, info = env.reset()
odor_value = obs['odor_concentration'][0]
agent_x, agent_y = obs['agent_position']
agent_heading = obs['agent_orientation'][0]
```

#### Environment Factory Function

```python
from plume_nav_sim.api.navigation import create_gymnasium_environment
from hydra import compose, initialize

# Create environment from Hydra configuration
with initialize(config_path="../conf", version_base=None):
    cfg = compose(config_name="config")
    env = create_gymnasium_environment(cfg)

# Create environment with parameter overrides and frame caching
env = create_gymnasium_environment(
    cfg,
    max_episode_steps=2000,
    reward_shaping={'odor_weight': 2.0, 'efficiency_bonus': 0.5},
    frame_cache="lru"  # Enable LRU frame caching for performance
)

# Environment validation
from gymnasium.utils.env_checker import check_env
check_env(env)  # Verify Gymnasium API compliance
```

### RL Training Workflows

#### Command-Line Training Interface

The library provides comprehensive CLI commands for RL training workflows:

```bash
# Basic PPO training
plume-nav-sim train --algorithm PPO

# SAC training with custom parameters
plume-nav-sim train --algorithm SAC \
    --total-timesteps 500000 \
    --learning-rate 3e-4 \
    --batch-size 256

# Multi-environment parallel training
plume-nav-sim train --algorithm PPO \
    --n-envs 8 \
    --env-parallel \
    --total-timesteps 1000000

# Training with checkpointing and evaluation
plume-nav-sim train --algorithm SAC \
    --checkpoint-dir ./checkpoints \
    --save-freq 50000 \
    --eval-freq 10000 \
    --eval-episodes 20

# Custom configuration with parameter overrides
plume-nav-sim train --algorithm PPO \
    navigator.max_speed=15.0 \
    simulation.max_episode_steps=2000 \
    rl.learning_rate=1e-4 \
    --config-name rl_config
```

#### Algorithm Support

**Proximal Policy Optimization (PPO)**
- **Use Case**: Robust policy learning with good sample efficiency
- **Best For**: Continuous control tasks with stable learning
- **Default Hyperparameters**: Optimized for odor plume navigation

```bash
# PPO with custom hyperparameters
plume-nav-sim train --algorithm PPO \
    --learning-rate 3e-4 \
    --n-steps 2048 \
    --batch-size 64 \
    --n-epochs 10 \
    --clip-range 0.2
```

**Soft Actor-Critic (SAC)**
- **Use Case**: Sample-efficient learning for continuous control
- **Best For**: Exploration-heavy environments with sparse rewards
- **Features**: Off-policy learning with entropy regularization

```bash
# SAC with custom configuration
plume-nav-sim train --algorithm SAC \
    --learning-rate 3e-4 \
    --buffer-size 1000000 \
    --batch-size 256 \
    --learning-starts 10000
```

**Twin Delayed DDPG (TD3)**
- **Use Case**: Deterministic policy learning with reduced overestimation
- **Best For**: High-dimensional continuous control tasks

```bash
# TD3 training
plume-nav-sim train --algorithm TD3 \
    --learning-rate 1e-3 \
    --batch-size 256 \
    --policy-delay 2
```

#### Training Progress Monitoring

```bash
# Training with comprehensive logging
plume-nav-sim train --algorithm PPO \
    --tensorboard-log ./tensorboard_logs \
    --verbose 1 \
    --progress-bar

# Training with custom evaluation metrics
plume-nav-sim train --algorithm SAC \
    --eval-freq 10000 \
    --eval-episodes 20 \
    --eval-log-path ./evaluation_logs
```

### Environment Wrappers

#### Available Wrappers

```python
from odor_plume_nav.environments.wrappers import (
    NormalizeObservation,
    RewardShaping,
    FrameStack,
    ActionClipping
)

# Apply normalization wrapper
env = NormalizeObservation(base_env, epsilon=1e-8)

# Add reward shaping
env = RewardShaping(
    env,
    odor_weight=1.0,
    distance_weight=0.5,
    control_penalty=0.1,
    efficiency_bonus=0.2
)

# Frame stacking for temporal awareness
env = FrameStack(env, num_stack=4)

# Action space clipping for safety
env = ActionClipping(env, min_action=-0.8, max_action=0.8)
```

#### Custom Wrapper Development

```python
import gymnasium as gym
from gymnasium.wrappers import Wrapper

class CustomPlumeWrapper(Wrapper):
    """Custom wrapper for specialized preprocessing."""
    
    def __init__(self, env, custom_parameter=1.0):
        super().__init__(env)
        self.custom_parameter = custom_parameter
    
    def step(self, action):
        # Custom action preprocessing
        modified_action = self.preprocess_action(action)
        obs, reward, terminated, truncated, info = self.env.step(modified_action)
        
        # Custom observation/reward postprocessing
        modified_obs = self.postprocess_observation(obs)
        modified_reward = self.shape_reward(reward, obs)
        
        return modified_obs, modified_reward, terminated, truncated, info
    
    def preprocess_action(self, action):
        """Apply custom action transformations."""
        return action * self.custom_parameter
    
    def postprocess_observation(self, obs):
        """Apply custom observation transformations."""
        return obs
    
    def shape_reward(self, reward, obs):
        """Apply custom reward shaping."""
        return reward + self.compute_bonus(obs)
```

## Migration Guide: Gym to Gymnasium API

### Overview

The Odor Plume Navigation library has been refactored to support the modern Gymnasium API while maintaining full backward compatibility with existing gym-based code. This migration guide helps you understand the changes and provides clear paths for updating your code.

### API Compatibility Matrix

| API Version | Step Return Signature | Environment ID | Import Statement |
|-------------|----------------------|----------------|-----------------|
| **Legacy Gym** | `(obs, reward, done, info)` | `OdorPlumeNavigation-v1` | `import gym` |
| **New Gymnasium** | `(obs, reward, terminated, truncated, info)` | `PlumeNavSim-v0` | `import gymnasium` |

### Dual API Support

The library automatically detects which API you're using and returns the appropriate tuple format:

#### Legacy Gym API (4-tuple) - Maintained for Backward Compatibility

```python
import gym
from odor_plume_nav.environments import register_environments

# Register environments for gym usage
register_environments()

# Legacy gym usage continues to work unchanged
env = gym.make('OdorPlumeNavigation-v1')
obs = env.reset()

for step in range(1000):
    action = env.action_space.sample()
    # Returns 4-tuple: (obs, reward, done, info)
    obs, reward, done, info = env.step(action)
    
    if done:
        obs = env.reset()
        
env.close()
```

#### New Gymnasium API (5-tuple) - Recommended for New Projects

```python
import gymnasium as gym
from odor_plume_nav.environments import register_environments

# Register environments for gymnasium usage
register_environments()

# New gymnasium environment with enhanced API
env = gym.make('PlumeNavSim-v0')
obs, info = env.reset(seed=42)

for step in range(1000):
    action = env.action_space.sample()
    # Returns 5-tuple: (obs, reward, terminated, truncated, info)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
        
env.close()
```

### Environment Registration Details

#### New PlumeNavSim-v0 Environment

The new `PlumeNavSim-v0` environment provides enhanced features and full Gymnasium compliance:

**Key Features:**
- **Gymnasium 0.29.x API compliance** with 5-tuple step returns
- **Enhanced observation space** with structured dictionary observations
- **Improved reward shaping** with separated termination conditions
- **Seed support** for deterministic episode initialization
- **Performance optimizations** maintaining ≤10ms step() execution time

**Environment Specification:**
```python
from odor_plume_nav.api.navigation import create_gymnasium_environment
import gymnasium as gym

# Create environment with factory function
env = create_gymnasium_environment(config_path="conf/config.yaml")

# Or use gymnasium.make() with registration
env = gym.make('PlumeNavSim-v0', 
               max_episode_steps=1000,
               render_mode="human")

# Environment information
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
```

**Action Space:**
```python
# Continuous control: Box(2,) with range [-1.0, 1.0]
action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
# action[0]: Linear velocity control
# action[1]: Angular velocity control
```

**Observation Space:**
```python
# Dictionary observation space with multiple components
observation_space = gymnasium.spaces.Dict({
    'odor_concentration': gymnasium.spaces.Box(
        shape=(1,), low=0.0, high=1.0, dtype=np.float32
    ),
    'agent_position': gymnasium.spaces.Box(
        shape=(2,), low=0.0, high=100.0, dtype=np.float32
    ),
    'agent_orientation': gymnasium.spaces.Box(
        shape=(1,), low=-np.pi, high=np.pi, dtype=np.float32
    ),
    'plume_gradient': gymnasium.spaces.Box(
        shape=(2,), low=-1.0, high=1.0, dtype=np.float32
    )
})
```

### Migration Strategies

#### Strategy 1: Gradual Migration (Recommended)

Maintain both APIs during transition period:

```python
# Phase 1: Test new API alongside existing code
def create_environment(use_gymnasium=False):
    if use_gymnasium:
        import gymnasium as gym
        env = gym.make('PlumeNavSim-v0')
        return env, "gymnasium"
    else:
        import gym
        env = gym.make('OdorPlumeNavigation-v1')
        return env, "gym"

# Phase 2: Validate consistency between APIs
def validate_api_consistency():
    gym_env, _ = create_environment(use_gymnasium=False)
    gymnasium_env, _ = create_environment(use_gymnasium=True)
    
    # Compare observation and action spaces
    assert gym_env.action_space == gymnasium_env.action_space
    # Note: observation spaces may differ due to enhanced features
    
    gym_env.close()
    gymnasium_env.close()

# Phase 3: Switch to Gymnasium for new development
env, api_type = create_environment(use_gymnasium=True)
obs, info = env.reset(seed=42)

for step in range(1000):
    action = env.action_space.sample()
    if api_type == "gymnasium":
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    else:
        obs, reward, done, info = env.step(action)
    
    if done:
        if api_type == "gymnasium":
            obs, info = env.reset()
        else:
            obs = env.reset()
```

#### Strategy 2: Direct Migration

For new projects or major refactoring:

```python
# Before: Legacy gym implementation
"""
import gym
env = gym.make('OdorPlumeNavigation-v1')
obs = env.reset()
obs, reward, done, info = env.step(action)
"""

# After: Modern gymnasium implementation
import gymnasium as gym
env = gym.make('PlumeNavSim-v0')
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated  # Convert to legacy done flag if needed
```

### Stable-Baselines3 Integration Updates

#### Legacy Integration
```python
from stable_baselines3 import PPO
import gym

env = gym.make('OdorPlumeNavigation-v1')
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=100000)
```

#### Updated Gymnasium Integration
```python
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make('PlumeNavSim-v0')
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=100000)
```

### Troubleshooting Common Migration Issues

#### Issue 1: Import Errors
```python
# Problem: ModuleNotFoundError for gymnasium
# Solution: Install RL dependencies
pip install "plume_nav_sim[rl]"
```

#### Issue 2: Step Return Tuple Length
```python
# Problem: Unpacking wrong number of values
# Solution: Use detection wrapper
def step_with_compatibility(env, action):
    result = env.step(action)
    if len(result) == 4:
        # Legacy gym API
        obs, reward, done, info = result
        return obs, reward, done, False, info  # Convert to 5-tuple
    else:
        # Modern gymnasium API
        return result

# Usage
obs, reward, terminated, truncated, info = step_with_compatibility(env, action)
done = terminated or truncated
```

#### Issue 3: Reset Method Signature
```python
# Problem: Different reset signatures
# Solution: Use compatibility wrapper
def reset_with_compatibility(env, seed=None):
    if hasattr(env, 'seed') and seed is not None:
        env.seed(seed)
        return env.reset()
    else:
        # Modern gymnasium reset with seed parameter
        if seed is not None:
            return env.reset(seed=seed)
        else:
            return env.reset()

# Usage works with both APIs
result = reset_with_compatibility(env, seed=42)
if isinstance(result, tuple):
    obs, info = result  # Gymnasium
else:
    obs = result  # Legacy gym
    info = {}
```

### Deprecation Warnings

When using legacy gym imports, you'll see helpful deprecation warnings:

```python
import gym
# UserWarning: You are using the legacy gym API. Consider migrating to gymnasium
# for enhanced features and future compatibility. See migration guide in README.md

env = gym.make('OdorPlumeNavigation-v1')
# UserWarning: Environment 'OdorPlumeNavigation-v1' is deprecated. 
# Use 'PlumeNavSim-v0' with gymnasium for new features and improvements.
```

### Migration from Legacy Simulation API

#### Migration Guide: Simulation to Gymnasium

**Legacy Simulation Approach:**
```python
# Old approach - direct simulation execution
from odor_plume_nav.api.navigation import (
    create_navigator, create_video_plume, run_plume_simulation
)

navigator = create_navigator(cfg.navigator)
video_plume = create_video_plume(cfg.video_plume)
positions, orientations, readings = run_plume_simulation(
    navigator, video_plume, cfg.simulation
)
```

**New Gymnasium Approach:**
```python
# New approach - Gymnasium environment
from odor_plume_nav.api.navigation import create_gymnasium_environment

env = create_gymnasium_environment(cfg)
obs, info = env.reset(seed=42)

positions, orientations, readings = [], [], []
for step in range(1000):
    action = your_policy(obs)  # Your control policy
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Extract data for analysis
    positions.append(obs['agent_position'])
    orientations.append(obs['agent_orientation'])
    readings.append(obs['odor_concentration'])
    
    if terminated or truncated:
        break
```

#### Backward Compatibility

The library maintains full backward compatibility with existing simulation APIs:

```python
# Legacy factory functions still available
from odor_plume_nav.api.navigation import (
    create_navigator_from_config,  # Legacy alias
    create_video_plume_from_config,  # Legacy alias
    run_simulation  # Legacy alias
)

# Original API continues to work unchanged
navigator = create_navigator_from_config(cfg.navigator)
video_plume = create_video_plume_from_config(cfg.video_plume)
results = run_simulation(navigator, video_plume, cfg.simulation)
```

#### Migration Strategies

**Gradual Migration:**
```python
# Phase 1: Use both APIs in parallel
legacy_results = run_plume_simulation(navigator, video_plume, cfg)
gymnasium_env = create_gymnasium_environment(cfg)

# Phase 2: Validate consistency
assert verify_results_consistency(legacy_results, gymnasium_env)

# Phase 3: Switch to Gymnasium for new features
model = PPO("MlpPolicy", gymnasium_env)
model.learn(total_timesteps=100000)
```

**Configuration Migration:**
```yaml
# Legacy configuration (still supported)
navigator:
  position: [50.0, 50.0]
  orientation: 45.0
  max_speed: 10.0

# Enhanced RL configuration
rl:
  environment:
    max_episode_steps: 1000
    reward_shaping:
      odor_weight: 1.0
      distance_weight: 0.5
  training:
    algorithm: "PPO"
    total_timesteps: 500000
    learning_rate: 3e-4
```

### Best Practices and Examples

#### Training Configuration Templates

**PPO Configuration for Quick Prototyping:**
```python
from stable_baselines3 import PPO

def quick_ppo_training():
    env = create_gymnasium_environment()
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        verbose=1
    )
    model.learn(total_timesteps=100000)
    return model
```

**SAC Configuration for Sample Efficiency:**
```python
from stable_baselines3 import SAC

def efficient_sac_training():
    env = create_gymnasium_environment()
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=256,
        learning_starts=1000,
        train_freq=1,
        verbose=1
    )
    model.learn(total_timesteps=200000)
    return model
```

#### Hyperparameter Optimization

```python
import optuna
from stable_baselines3 import PPO

def optimize_hyperparameters(trial):
    """Optuna hyperparameter optimization for RL training."""
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # Create environment and model
    env = create_gymnasium_environment()
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        verbose=0
    )
    
    # Train and evaluate
    model.learn(total_timesteps=50000)
    
    # Evaluation metric
    mean_reward = evaluate_policy(model, env, n_eval_episodes=10)[0]
    return mean_reward

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(optimize_hyperparameters, n_trials=50)
```

#### Multi-Environment Training

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

def create_vectorized_env(n_envs=4):
    """Create vectorized environment for parallel training."""
    
    def make_env():
        env = create_gymnasium_environment()
        # Add environment-specific wrappers here
        return env
    
    return SubprocVecEnv([make_env for _ in range(n_envs)])

# Training with vectorized environments
def parallel_training():
    vec_env = create_vectorized_env(n_envs=8)
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=1000000)
    return model
```

## Enhanced Structured Logging with Performance Analytics

### Overview

The library features a comprehensive structured logging system built on Loguru with deep integration into the frame caching and performance monitoring systems. This provides machine-parseable JSON logs, correlation IDs for distributed debugging, and automatic performance metrics collection embedded directly in simulation workflows.

### Key Features

- **JSON-Structured Logs**: Machine-readable logging for automated analysis and monitoring
- **Performance Metrics Integration**: Automatic embedding of cache statistics and step timings in `info["perf_stats"]`
- **Correlation ID Tracking**: Complete experiment traceability across distributed systems
- **Multi-Sink Configuration**: Flexible output to console, files, and external systems
- **Environment-Specific Profiles**: Development vs production logging configurations

### Performance Analytics Integration

#### Automatic Performance Logging

The logging system automatically captures and structures performance metrics from the frame caching system:

```python
from odor_plume_nav.api.navigation import create_gymnasium_environment
from loguru import logger

# Environment automatically logs performance metrics
env = create_gymnasium_environment(
    config_path="conf/config.yaml",
    frame_cache="lru"
)

obs, info = env.reset()
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Performance stats automatically logged as structured JSON
    perf_stats = info["perf_stats"]
    # Logs include: step_time_ms, frame_retrieval_ms, cache_hit_rate, 
    # cache_hits, cache_misses, cache_evictions, memory_usage_mb, fps_estimate
    
    logger.info("Simulation step completed", 
                step=step, 
                **perf_stats)
```

#### Accessing Video Frame Data

```python
# Enable video frame logging for analysis
env = create_gymnasium_environment(
    config_path="conf/config.yaml",
    frame_cache="lru",
    include_video_frame=True  # Adds frame data to info
)

obs, info = env.reset()
for step in range(100):
    action = policy.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Access current video frame for analysis (when enabled)
    if "video_frame" in info:
        current_frame = info["video_frame"]  # NumPy array (H, W) or (H, W, C)
        
        # Log frame statistics
        frame_stats = {
            "frame_shape": current_frame.shape,
            "frame_mean": float(current_frame.mean()),
            "frame_std": float(current_frame.std()),
            "frame_min": float(current_frame.min()),
            "frame_max": float(current_frame.max())
        }
        
        logger.debug("Frame analysis", **frame_stats)
```

### Logging Configuration

#### Production Configuration (JSON Output)

```yaml
# logging.yaml - Production configuration
version: 1
formatters:
  json:
    format: "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}"
    serialize: true
  structured:
    format: "{time} | {level} | {extra[correlation_id]} | {message}"

handlers:
  console:
    sink: "sys.stderr"
    format: "{time} | {level} | {message}"
    level: "INFO"
    colorize: false
    
  file_json:
    sink: "logs/odor_plume_nav_{time:YYYY-MM-DD}.json"
    format: "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {extra} | {message}"
    level: "DEBUG"
    rotation: "100 MB"
    retention: "30 days"
    compression: "gzip"
    serialize: true
    
  performance:
    sink: "logs/performance_{time:YYYY-MM-DD}.jsonl"
    format: "{time:YYYY-MM-DD HH:mm:ss.SSS} | {extra[perf_stats]} | {message}"
    level: "INFO"
    filter: "performance"
    rotation: "50 MB"
    retention: "90 days"

loggers:
  odor_plume_nav:
    level: "INFO"
    handlers: ["console", "file_json", "performance"]
    propagate: false
```

#### Development Configuration (Human-Readable)

```yaml
# logging.yaml - Development configuration  
handlers:
  console:
    sink: "sys.stderr"
    format: "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    level: "DEBUG"
    colorize: true
    
  file_human:
    sink: "logs/development.log"
    format: "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"
    level: "DEBUG"
    rotation: "10 MB"
    retention: "7 days"

loggers:
  odor_plume_nav:
    level: "DEBUG"
    handlers: ["console", "file_human"]
    propagate: false
```

### Structured Logging with Loguru

### Overview

The library features a modern structured logging system built on Loguru, replacing traditional print statements and basic logging with comprehensive JSON-formatted logs, correlation IDs, and performance monitoring integration.

### Basic Logging Configuration

#### Default Setup

The logging system is automatically configured when importing the library:

```python
from odor_plume_nav.utils.logging_setup import setup_logging
from loguru import logger

# Automatic setup with default configuration
setup_logging()

# Use structured logging throughout your code
logger.info("Starting odor plume navigation simulation")
logger.debug("Navigator initialized", position=[10.0, 15.0], orientation=45.0)
```

#### Custom Logging Configuration

```python
from odor_plume_nav.utils.logging_setup import setup_logging
from loguru import logger

# Custom logging configuration
setup_logging(
    level="DEBUG",
    format_type="json",
    file_rotation="50 MB",
    retention="2 weeks",
    compression="gzip",
    correlation_id=True
)

# Structured logging with context
logger.bind(
    component="navigator",
    experiment_id="exp_001",
    agent_count=5
).info("Multi-agent simulation started")
```

### Configuration Examples

#### Development Environment Configuration

```python
# conf/config.yaml - Development logging setup
logging:
  # Core logging configuration
  level: "DEBUG"
  console_enabled: true
  file_enabled: true
  
  # Loguru-specific configuration
  loguru:
    format: "json"  # Options: text, json
    colorize: true  # Colorize console output
    diagnose: true  # Include detailed exception information
    
    # File logging configuration
    file:
      path: "logs/odor_plume_nav_{time:YYYY-MM-DD}.log"
      rotation: "10 MB"
      retention: "1 week"
      compression: "gzip"
      level: "DEBUG"
    
    # Console logging configuration  
    console:
      level: "INFO"
      colorize: true
      format: "text"  # Human-readable for development
    
    # Performance monitoring integration
    performance:
      enabled: true
      slow_threshold: 0.033  # 33ms (30 FPS target)
      memory_tracking: true
      step_timing: true
    
    # Correlation ID tracking
    correlation:
      enabled: true
      auto_generate: true
      propagate_context: true
```

#### Production Environment Configuration

```python
# conf/local/production.yaml - Production logging setup
logging:
  level: "INFO"
  console_enabled: false  # Reduce noise in production
  file_enabled: true
  
  loguru:
    format: "json"  # Structured logs for analysis
    colorize: false
    diagnose: false  # Security: don't expose stack traces
    
    # Centralized logging configuration
    file:
      path: "/var/log/odor_plume_nav/app_{time:YYYY-MM-DD}.log"
      rotation: "100 MB"
      retention: "30 days"
      compression: "gzip"
      level: "INFO"
    
    # Syslog integration for centralized logging
    syslog:
      enabled: true
      address: "localhost"
      port: 514
      facility: "local0"
      format: "json"
    
    # Performance monitoring for production
    performance:
      enabled: true
      slow_threshold: 0.010  # Stricter production threshold
      memory_tracking: false  # Reduce overhead
      alert_on_degradation: true
    
    # Request correlation tracking
    correlation:
      enabled: true
      header_name: "X-Correlation-ID"
      propagate_downstream: true
```

### Usage Patterns

#### Real-Time Performance Logging

```python
from loguru import logger
from odor_plume_nav.utils.logging_setup import setup_logger
import contextvars

# Setup enhanced logging with performance tracking
setup_logger(
    level="INFO",
    format_type="json",
    correlation_id=True,
    performance_tracking=True
)

# Use structured logging in RL training loops
correlation_id = contextvars.ContextVar('correlation_id')

def train_agent_with_logging():
    """RL training with comprehensive performance logging."""
    
    # Set correlation ID for experiment tracking
    corr_id = f"train_{time.strftime('%Y%m%d_%H%M%S')}"
    correlation_id.set(corr_id)
    
    with logger.contextualize(correlation_id=corr_id, experiment="ppo_training"):
        logger.info("Starting RL training", algorithm="PPO", timesteps=100000)
        
        env = create_gymnasium_environment(
            config_path="conf/config.yaml",
            frame_cache="lru"
        )
        
        obs, info = env.reset()
        for step in range(100000):
            action = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Performance metrics automatically included in logs
            perf_stats = info["perf_stats"]
            
            # Log step completion with performance context
            logger.debug("Training step completed",
                        step=step,
                        reward=reward,
                        **perf_stats)
            
            # Alert on performance degradation
            if perf_stats["cache_hit_rate"] < 0.85:
                logger.warning("Cache performance degraded",
                              hit_rate=perf_stats["cache_hit_rate"],
                              memory_usage_mb=perf_stats["cache_memory_mb"])
            
            if terminated or truncated:
                logger.info("Episode completed",
                           episode_steps=step,
                           final_reward=reward,
                           cache_final_hit_rate=perf_stats["cache_hit_rate"])
                obs, info = env.reset()
```

#### Analysis and Debugging Workflows

```python
# Notebook analysis with video frame access
def analyze_plume_behavior():
    """Analyze odor plume navigation with frame-level insights."""
    
    env = create_gymnasium_environment(
        config_path="conf/config.yaml",
        frame_cache="lru",
        include_video_frame=True  # Enable frame data in info
    )
    
    trajectory_data = []
    obs, info = env.reset()
    
    with logger.contextualize(analysis_session="plume_behavior_study"):
        logger.info("Starting plume behavior analysis")
        
        for step in range(1000):
            action = your_policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Extract comprehensive step data
            step_data = {
                "step": step,
                "observation": obs,
                "action": action,
                "reward": reward,
                "performance": info["perf_stats"],
                "frame_data": info.get("video_frame")
            }
            trajectory_data.append(step_data)
            
            # Log interesting events
            if reward > 0.8:
                logger.info("High reward achieved",
                           step=step,
                           reward=reward,
                           odor_concentration=obs.get("odor_concentration"),
                           agent_position=obs.get("agent_position"))
            
            # Log frame analysis
            if "video_frame" in info:
                frame = info["video_frame"]
                logger.debug("Frame analysis",
                            step=step,
                            frame_mean=float(frame.mean()),
                            frame_std=float(frame.std()),
                            unique_values=len(np.unique(frame)))
            
            if terminated or truncated:
                logger.info("Analysis episode completed", total_steps=step)
                break
    
    return trajectory_data
```

#### Basic Structured Logging

```python
from loguru import logger

# Simple structured logging with performance context
logger.info("Simulation started", cache_mode="lru", cache_size_mb=2048)
logger.debug("Configuration loaded", config_path="conf/config.yaml")
logger.warning("Performance degradation detected", 
               fps=25.5, 
               threshold=30.0,
               cache_hit_rate=0.72)
logger.error("Navigation failed", 
             error="obstacle_collision", 
             position=[15.2, 22.1],
             cache_memory_usage=1856)

# Exception logging with context
try:
    navigator.update_position(invalid_position)
except ValueError as e:
    logger.exception("Invalid position update", 
                    position=invalid_position, 
                    component="navigator",
                    cache_stats=cache.get_statistics())
```

#### Component-Specific Logging

```python
from loguru import logger

class Navigator:
    def __init__(self, config):
        # Component-specific logger with context binding
        self.logger = logger.bind(component="navigator", id=id(self))
        self.logger.info("Navigator initialized", config=config.dict())
    
    def update_position(self, position):
        self.logger.debug("Position update", 
                         old_position=self.position,
                         new_position=position,
                         timestamp=time.time())
        self.position = position

class VideoPlume:
    def __init__(self, video_path):
        self.logger = logger.bind(component="video_plume", video=video_path)
        self.logger.info("Video plume loaded", 
                        path=video_path,
                        frame_count=self.frame_count,
                        resolution=self.resolution)
    
    def get_frame(self, frame_idx):
        start_time = time.perf_counter()
        frame = self._load_frame(frame_idx)
        duration = time.perf_counter() - start_time
        
        self.logger.debug("Frame retrieved",
                         frame_idx=frame_idx,
                         duration_ms=duration * 1000,
                         frame_shape=frame.shape)
        return frame
```

#### Performance Monitoring Integration

```python
from loguru import logger
from odor_plume_nav.utils.performance import performance_monitor
import time

class SimulationRunner:
    def __init__(self):
        self.logger = logger.bind(component="simulation")
    
    @performance_monitor
    def step(self, action):
        """Simulation step with automatic performance logging."""
        start_time = time.perf_counter()
        
        # Execute simulation step
        obs, reward, terminated, truncated, info = self._internal_step(action)
        
        # Automatic performance logging via decorator
        duration = time.perf_counter() - start_time
        
        # Structured step logging
        self.logger.debug("Simulation step completed",
                         step=self.step_count,
                         duration_ms=duration * 1000,
                         fps=1.0 / duration if duration > 0 else float('inf'),
                         agent_position=obs['agent_position'],
                         reward=reward,
                         terminated=terminated)
        
        return obs, reward, terminated, truncated, info
```

#### Correlation ID Tracking

```python
from loguru import logger
from odor_plume_nav.utils.logging_setup import generate_correlation_id
import contextvars

# Correlation context for request tracking
correlation_id = contextvars.ContextVar('correlation_id')

def run_experiment(experiment_config):
    # Generate unique correlation ID for experiment
    corr_id = generate_correlation_id()
    correlation_id.set(corr_id)
    
    # All logging within this context includes correlation ID
    with logger.contextualize(correlation_id=corr_id, experiment=experiment_config.name):
        logger.info("Experiment started", config=experiment_config.dict())
        
        try:
            # Run simulation components
            navigator = create_navigator(experiment_config.navigator)
            results = run_simulation(navigator, experiment_config.simulation)
            
            logger.info("Experiment completed successfully", 
                       total_steps=results.step_count,
                       duration=results.duration,
                       final_reward=results.total_reward)
            
        except Exception as e:
            logger.exception("Experiment failed", 
                           error=str(e),
                           error_type=type(e).__name__)
            raise
```

### Advanced Logging Features

#### Custom Log Formatters

```python
from loguru import logger
import json

def custom_json_formatter(record):
    """Custom JSON formatter with additional metadata."""
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
        # Add custom fields
        "service": "odor_plume_nav",
        "version": "0.2.0",
        "environment": os.getenv("ENVIRONMENT_TYPE", "development")
    }
    
    # Include extra fields from structured logging
    if record["extra"]:
        log_entry["extra"] = record["extra"]
    
    # Include exception information if present
    if record["exception"]:
        log_entry["exception"] = {
            "type": record["exception"].type.__name__,
            "value": str(record["exception"].value),
            "traceback": record["exception"].traceback.format()
        }
    
    return json.dumps(log_entry)

# Configure custom formatter
logger.add("logs/structured.log", 
          format=custom_json_formatter,
          rotation="50 MB",
          retention="2 weeks")
```

#### Log Filtering and Sampling

```python
from loguru import logger

def filter_performance_logs(record):
    """Filter out high-frequency performance logs in production."""
    if record["extra"].get("component") == "performance":
        # Only log every 100th performance record
        return record["extra"].get("step", 0) % 100 == 0
    return True

def filter_debug_in_production(record):
    """Suppress debug logs in production environment."""
    if os.getenv("ENVIRONMENT_TYPE") == "production":
        return record["level"].no >= logger.level("INFO").no
    return True

# Apply filters
logger.add("logs/filtered.log", 
          filter=lambda record: filter_performance_logs(record) and 
                               filter_debug_in_production(record))
```

#### JSON Output Format for Analysis

The structured logging system produces machine-parseable JSON logs optimized for performance analysis:

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "logger": "odor_plume_nav.environments.gymnasium_env",
  "function": "step",
  "line": 245,
  "correlation_id": "exp_ppo_20240115_103045",
  "message": "Environment step completed",
  "extra": {
    "step": 1000,
    "reward": 0.85,
    "terminated": false,
    "perf_stats": {
      "step_time_ms": 8.5,
      "frame_retrieval_ms": 2.1,
      "cache_hit_rate": 0.92,
      "cache_hits": 920,
      "cache_misses": 80,
      "cache_evictions": 12,
      "cache_memory_mb": 1856,
      "fps_estimate": 117.6
    },
    "observation": {
      "odor_concentration": [0.67],
      "agent_position": [45.2, 67.8],
      "agent_orientation": [1.23]
    }
  }
}
```

#### Automated Log Analysis

```python
import json
from pathlib import Path

def analyze_performance_logs(log_file: Path):
    """Analyze performance from structured JSON logs."""
    
    step_times = []
    cache_hit_rates = []
    memory_usage = []
    
    with open(log_file) as f:
        for line in f:
            log_entry = json.loads(line)
            
            if "perf_stats" in log_entry.get("extra", {}):
                perf = log_entry["extra"]["perf_stats"]
                step_times.append(perf["step_time_ms"])
                cache_hit_rates.append(perf["cache_hit_rate"])
                memory_usage.append(perf["cache_memory_mb"])
    
    # Performance analysis
    avg_step_time = sum(step_times) / len(step_times)
    avg_hit_rate = sum(cache_hit_rates) / len(cache_hit_rates)
    max_memory = max(memory_usage)
    
    return {
        "average_step_time_ms": avg_step_time,
        "average_cache_hit_rate": avg_hit_rate,
        "peak_memory_usage_mb": max_memory,
        "performance_target_met": avg_step_time < 10.0,
        "cache_efficiency_good": avg_hit_rate > 0.90
    }

# Usage
results = analyze_performance_logs(Path("logs/performance_2024-01-15.jsonl"))
print(f"Performance analysis: {results}")
```

#### Integration with External Logging Systems

```python
from loguru import logger
import logging

# Integration with standard Python logging
class InterceptHandler(logging.Handler):
    """Intercept standard logging and redirect to Loguru."""
    
    def emit(self, record):
        # Get corresponding Loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # Find caller from where logging was called
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

# Configure integration
logging.basicConfig(handlers=[InterceptHandler()], level=0)

# Configure third-party library logging
for library in ["matplotlib", "stable_baselines3", "gymnasium"]:
    logging.getLogger(library).handlers = [InterceptHandler()]

# Integration with monitoring systems
def setup_monitoring_integration():
    """Setup integration with external monitoring systems."""
    
    # Custom sink for Prometheus metrics
    def prometheus_sink(message):
        if "perf_stats" in message.record["extra"]:
            stats = message.record["extra"]["perf_stats"]
            # Export metrics to Prometheus
            step_time_metric.set(stats["step_time_ms"])
            cache_hit_rate_metric.set(stats["cache_hit_rate"])
    
    # Custom sink for alerting
    def alert_sink(message):
        if message.record["level"].name == "WARNING":
            if "cache_hit_rate" in message.record["extra"]:
                hit_rate = message.record["extra"]["cache_hit_rate"]
                if hit_rate < 0.85:
                    send_alert(f"Cache performance degraded: {hit_rate:.2%}")
    
    logger.add(prometheus_sink, level="INFO", filter="performance")
    logger.add(alert_sink, level="WARNING")
```

### Environment Variable Configuration

Set up logging through environment variables for deployment flexibility:

```bash
# .env file for development
LOG_LEVEL=DEBUG
LOG_FORMAT=text
LOG_FILE_ENABLED=true
LOG_CONSOLE_COLORIZE=true
LOG_CORRELATION_ENABLED=true
LOG_PERFORMANCE_TRACKING=true

# Frame caching configuration
FRAME_CACHE_MODE=lru
FRAME_CACHE_SIZE_MB=2048
FRAME_CACHE_PRESSURE_THRESHOLD=0.90
LOG_JSON_SINK=true

# Production environment variables
export LOG_LEVEL=INFO
export LOG_FORMAT=json
export LOG_FILE_PATH=/var/log/odor_plume_nav/app.log
export LOG_ROTATION=100MB
export LOG_RETENTION=30days
export LOG_COMPRESSION=gzip
export LOG_SYSLOG_ENABLED=true

# Production frame caching
export FRAME_CACHE_MODE=all
export FRAME_CACHE_SIZE_MB=8192
export FRAME_CACHE_PRELOAD_ENABLED=true
```

Use in configuration:

```python
# Automatic environment variable integration
logging:
  level: ${oc.env:LOG_LEVEL,"INFO"}
  console_enabled: ${oc.env:LOG_CONSOLE_ENABLED,"true"}
  file_enabled: ${oc.env:LOG_FILE_ENABLED,"true"}
  
  loguru:
    format: ${oc.env:LOG_FORMAT,"json"}
    colorize: ${oc.env:LOG_CONSOLE_COLORIZE,"false"}
    
    file:
      path: ${oc.env:LOG_FILE_PATH,"logs/odor_plume_nav.log"}
      rotation: ${oc.env:LOG_ROTATION,"10 MB"}
      retention: ${oc.env:LOG_RETENTION,"1 week"}
      compression: ${oc.env:LOG_COMPRESSION,"gzip"}
```

## High-Performance Frame Caching

### Overview

The library features an advanced frame caching system designed to dramatically accelerate reinforcement learning training by eliminating video decoding bottlenecks. The caching system provides dual operational modes (LRU and full-preload) with configurable memory limits, achieving sub-10ms environment step times for optimal training performance.

### Cache Operating Modes

#### LRU (Least Recently Used) Cache Mode

Intelligent caching with automatic memory management:

```python
from odor_plume_nav.api.navigation import create_gymnasium_environment

# Create environment with LRU caching (recommended for most scenarios)
env = create_gymnasium_environment(
    config_path="conf/config.yaml",
    frame_cache="lru"
)

# Monitor cache performance through info dictionary
obs, info = env.reset()
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Access cache performance metrics
    cache_stats = info["perf_stats"]
    print(f"Cache hit rate: {cache_stats['cache_hit_rate']:.2%}")
    print(f"Frame retrieval time: {cache_stats['frame_retrieval_ms']:.2f}ms")
    
    if terminated or truncated:
        obs, info = env.reset()
```

#### Full Preload Cache Mode

Maximum performance for memory-rich environments:

```python
# Create environment with full preload (optimal for short videos)
env = create_gymnasium_environment(
    config_path="conf/config.yaml", 
    frame_cache="all"
)

# All frames loaded into memory during initialization
# Achieves consistent <5ms frame retrieval times
obs, info = env.reset()
```

#### Direct I/O Mode

Bypass caching for debugging or memory-constrained scenarios:

```python
# Disable caching for debugging or minimal memory usage
env = create_gymnasium_environment(
    config_path="conf/config.yaml",
    frame_cache="none"
)
```

### Performance Characteristics

| Cache Mode | Memory Usage | Frame Retrieval Time | Best Use Case |
|------------|--------------|---------------------|---------------|
| **LRU** | 2 GiB default (configurable) | <10ms avg, >90% hit rate | General RL training |
| **Full Preload** | Entire video in memory | <5ms guaranteed | Short videos, maximum speed |
| **None** | Minimal | Variable, 20-100ms | Debugging, memory constraints |

### Configuration Examples

#### Environment Variables

```bash
# Set cache mode via environment variable
export FRAME_CACHE_MODE=lru
export FRAME_CACHE_SIZE_MB=4096  # 4 GiB cache limit

# Run training with enhanced caching
plume-nav-sim train --algorithm PPO
```

#### Hydra Configuration

```yaml
# conf/config.yaml
environment:
  frame_cache:
    mode: lru  # Options: none, lru, all
    cache_size_mb: 2048  # 2 GiB default
    memory_pressure_threshold: 0.90  # Trigger cleanup at 90%
    
# Override via CLI
plume-nav-sim run environment.frame_cache.mode=all environment.frame_cache.cache_size_mb=8192
```

#### Programmatic Configuration

```python
from odor_plume_nav.cache import FrameCache
from odor_plume_nav.environments.gymnasium_env import OdorPlumeNavigationEnv

# Custom cache configuration
cache = FrameCache(
    mode="lru",
    max_size_bytes=2 * 1024**3,  # 2 GiB
    memory_pressure_callback=lambda: print("Memory pressure detected")
)

# Pass to environment
env = OdorPlumeNavigationEnv(
    video_path="data/complex_plume.mp4",
    frame_cache=cache,
    max_episode_steps=1000
)
```

### Performance Monitoring

#### Accessing Cache Statistics

```python
# During RL training loops
obs, info = env.reset()
for step in range(10000):
    action = policy.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Access performance statistics
    perf_stats = info["perf_stats"]
    cache_metrics = {
        "hit_rate": perf_stats["cache_hit_rate"],
        "hits": perf_stats["cache_hits"],
        "misses": perf_stats["cache_misses"],
        "evictions": perf_stats["cache_evictions"],
        "memory_usage_mb": perf_stats["cache_memory_mb"],
        "step_time_ms": perf_stats["step_time_ms"],
        "frame_retrieval_ms": perf_stats["frame_retrieval_ms"],
        "fps_estimate": perf_stats["fps_estimate"]
    }
    
    # Optional: Access current video frame for analysis
    if "video_frame" in info:
        current_frame = info["video_frame"]  # NumPy array
        # Perform frame analysis or visualization
```

#### Real-Time Performance Analysis

```python
# Monitor cache performance during training
class CachePerformanceCallback:
    def __init__(self):
        self.hit_rates = []
        self.step_times = []
    
    def on_step(self, info):
        perf_stats = info.get("perf_stats", {})
        self.hit_rates.append(perf_stats.get("cache_hit_rate", 0))
        self.step_times.append(perf_stats.get("step_time_ms", 0))
        
        # Alert on performance degradation
        if len(self.hit_rates) > 100:
            recent_hit_rate = sum(self.hit_rates[-100:]) / 100
            if recent_hit_rate < 0.85:
                print(f"WARNING: Cache hit rate degraded to {recent_hit_rate:.2%}")

# Use in training
callback = CachePerformanceCallback()
obs, info = env.reset()
callback.on_step(info)
```

## Command-Line Interface

The library provides comprehensive CLI commands for automation and batch processing with integrated frame caching support.

### Available Commands

```bash
# Run simulation with modular component selection
plume-nav-sim run

# Run with different plume models
plume-nav-sim run plume_model=gaussian
plume-nav-sim run plume_model=turbulent
plume-nav-sim run plume_model=video plume_config.video_path=data/my_plume.mp4

# Run with memory configurations
plume-nav-sim run navigator.memory_enabled=true navigator.strategy=infotaxis
plume-nav-sim run navigator.memory_enabled=false navigator.strategy=reactive

# Run with environmental dynamics
plume-nav-sim run plume_model=turbulent wind_field=turbulent sensor_config=multi_modal

# Run with frame caching (video models only)
plume-nav-sim run plume_model=video --frame-cache lru

# Comparative studies with parameter sweeps
plume-nav-sim run --multirun \
    plume_model=gaussian,turbulent,video \
    navigator.memory_enabled=true,false \
    navigator.max_speed=1.0,2.0,3.0

# Memory comparison study
plume-nav-sim run --multirun \
    +experiment=memory_comparison \
    plume_model=gaussian,turbulent \
    navigator.memory_enabled=true,false

# Reinforcement learning training with modular component selection
plume-nav-sim train --algorithm PPO plume_model=gaussian navigator.memory_enabled=false
plume-nav-sim train --algorithm SAC plume_model=turbulent navigator.memory_enabled=true
plume-nav-sim train --algorithm TD3 plume_model=video --frame-cache lru

# Visualization commands
plume-nav-sim visualize --input-path outputs/experiment_results.npz
plume-nav-sim visualize --animation --save-video output.mp4

# Configuration validation
plume-nav-sim config validate
plume-nav-sim config show

# Environment setup
plume-nav-sim setup --create-dirs --init-config
```

### Frame Cache CLI Options

```bash
# Basic cache modes
plume-nav-sim run --frame-cache none    # Disable caching
plume-nav-sim run --frame-cache lru     # LRU caching (default)
plume-nav-sim run --frame-cache all     # Full preload caching

# Advanced cache configuration
plume-nav-sim run --frame-cache lru \
    environment.frame_cache.cache_size_mb=4096 \
    environment.frame_cache.memory_pressure_threshold=0.85

# RL training with optimized caching
plume-nav-sim train --algorithm PPO \
    --frame-cache all \
    --total-timesteps 1000000 \
    --n-envs 8 \
    environment.frame_cache.cache_size_mb=8192
```

### RL Training Commands

```bash
# Basic algorithm training
plume-nav-sim train --algorithm PPO
plume-nav-sim train --algorithm SAC
plume-nav-sim train --algorithm TD3

# Training with custom parameters
plume-nav-sim train --algorithm PPO \
    --total-timesteps 1000000 \
    --learning-rate 3e-4 \
    --n-steps 2048 \
    --batch-size 64

# Parallel training with multiple environments
plume-nav-sim train --algorithm SAC \
    --n-envs 8 \
    --env-parallel \
    --total-timesteps 2000000

# Training with checkpointing and evaluation
plume-nav-sim train --algorithm PPO \
    --checkpoint-dir ./checkpoints \
    --save-freq 50000 \
    --eval-freq 10000 \
    --eval-episodes 20 \
    --tensorboard-log ./logs

# Training with modular environment configuration
plume-nav-sim train --algorithm SAC \
    plume_model=turbulent \
    wind_field=turbulent \
    navigator.memory_enabled=true \
    navigator.max_speed=15.0 \
    simulation.max_episode_steps=2000 \
    rl.reward_shaping.odor_weight=2.0 \
    --config-name rl_training_config

# Memory comparison training study
plume-nav-sim train --algorithm PPO \
    +experiment=memory_comparison \
    plume_model=gaussian \
    navigator.memory_enabled=true,false \
    --total-timesteps 500000

# Plume model validation training
plume-nav-sim train --algorithm SAC \
    +experiment=plume_fidelity \
    plume_model=turbulent \
    wind_field=turbulent \
    --eval-freq 10000

# High-performance training with full frame preloading
plume-nav-sim train --algorithm PPO \
    --frame-cache all \
    --n-envs 8 \
    --total-timesteps 2000000 \
    environment.frame_cache.cache_size_mb=8192 \
    --tensorboard-log ./logs

# Advanced training options with performance monitoring
plume-nav-sim train --algorithm PPO \
    --frame-cache lru \
    --learning-rate 1e-4 \
    --clip-range 0.1 \
    --entropy-coef 0.01 \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --gae-lambda 0.95 \
    --verbose 1 \
    --progress-bar \
    environment.frame_cache.memory_pressure_threshold=0.85
```

### CLI Integration Examples

```bash
# Research workflow automation
#!/bin/bash
# Multi-condition RL experiment execution
for algorithm in PPO SAC TD3; do
    for lr in 1e-4 3e-4 1e-3; do
        plume-nav-sim train \
            --algorithm $algorithm \
            --learning-rate $lr \
            --total-timesteps 500000 \
            --checkpoint-dir "./experiments/${algorithm}_lr_${lr}" \
            hydra.job.name="${algorithm}_lr_${lr}"
    done
done

# Batch evaluation of trained models
plume-nav-sim evaluate \
    --model-dir experiments/ \
    --eval-episodes 100 \
    --output-format csv \
    --metrics reward,success_rate,episode_length

# Automated hyperparameter search
plume-nav-sim train --algorithm PPO \
    --optuna-trials 50 \
    --optuna-study-name ppo_optimization \
    --optuna-db sqlite:///optimization.db
```

## Configuration System

The library uses a sophisticated Hydra-based configuration hierarchy with dataclass-based structured configs, supporting environment variable integration, parameter sweeps, and multi-environment deployment with full type safety and validation.

### Modern Structured Configuration Architecture

The refactored configuration system replaces unstructured YAML with Pydantic-validated dataclasses, providing:

- **Type Safety**: Automatic validation of all configuration parameters
- **IDE Support**: Full autocomplete and type hints in development
- **Runtime Validation**: Configuration errors caught at startup, not runtime
- **Schema Evolution**: Backward-compatible configuration upgrades
- **Documentation**: Self-documenting configuration through type annotations

### Configuration Structure

```
conf/
├── base.yaml          # Foundation defaults with dataclass annotations
├── config.yaml        # User customizations with structured config composition
├── logging.yaml       # Loguru logging configuration with JSON sinks
├── frame_cache/       # Frame caching configuration group
│   ├── lru.yaml       # LRU cache configuration
│   ├── preload.yaml   # Full preload cache configuration
│   └── disabled.yaml  # Disabled cache configuration
├── rl/                # RL-specific structured configurations
│   ├── algorithms/    # Algorithm-specific hyperparameters with validation
│   │   ├── ppo.yaml   # PPO algorithm dataclass configuration
│   │   ├── sac.yaml   # SAC algorithm dataclass configuration
│   │   └── td3.yaml   # TD3 algorithm dataclass configuration
│   ├── environments/  # Environment configurations with type enforcement
│   │   ├── basic.yaml      # Basic environment dataclass config
│   │   ├── advanced.yaml   # Advanced environment dataclass config
│   │   └── multi_agent.yaml # Multi-agent dataclass config
│   └── training.yaml  # Training pipeline structured configuration
└── local/             # Local development with secret management
    ├── credentials.yaml.template
    ├── development.yaml
    ├── production.yaml
    └── paths.yaml.template
```

### Dataclass-Based Configuration Examples

#### Structured Configuration Models

The library defines comprehensive dataclass models with automatic validation:

```python
from dataclasses import dataclass, field
from typing import Optional, List, Union
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
import numpy as np

@dataclass
class NavigatorConfig:
    """Structured configuration for navigator parameters with validation."""
    
    # Core navigation parameters with type enforcement
    position: Optional[List[float]] = None
    orientation: float = 0.0
    speed: float = 0.0
    max_speed: float = 1.0
    angular_velocity: float = 0.0
    
    # Multi-agent configuration with structured defaults
    positions: Optional[List[List[float]]] = None
    orientations: Optional[List[float]] = None
    speeds: Optional[List[float]] = None
    max_speeds: Optional[List[float]] = None
    angular_velocities: Optional[List[float]] = None
    num_agents: Optional[int] = None
    
    # Control parameters with validation constraints
    control: ControlConfig = field(default_factory=lambda: ControlConfig())
    formation: FormationConfig = field(default_factory=lambda: FormationConfig())
    
    def __post_init__(self):
        """Post-initialization validation."""
        # Validate speed constraints
        if self.speed > self.max_speed:
            raise ValueError(f"Speed {self.speed} cannot exceed max_speed {self.max_speed}")
        
        # Validate multi-agent consistency
        if self.positions is not None:
            if self.num_agents is None:
                self.num_agents = len(self.positions)
            elif len(self.positions) != self.num_agents:
                raise ValueError("Number of positions must match num_agents")

@dataclass
class ControlConfig:
    """Control system configuration with performance constraints."""
    acceleration: float = 0.1
    turning_rate: float = 30.0
    sensor_range: float = 10.0
    sensor_noise: float = 0.0
    sensor_resolution: float = 1.0
    
    def __post_init__(self):
        """Validate control parameters."""
        if self.acceleration <= 0:
            raise ValueError("Acceleration must be positive")
        if self.sensor_range <= 0:
            raise ValueError("Sensor range must be positive")

@dataclass 
class FormationConfig:
    """Formation control configuration for multi-agent scenarios."""
    type: str = "grid"  # Options: grid, line, circle, custom
    spacing: float = 5.0
    maintain_formation: bool = False
    communication_range: float = 15.0
    
    def __post_init__(self):
        """Validate formation parameters."""
        valid_types = {"grid", "line", "circle", "custom"}
        if self.type not in valid_types:
            raise ValueError(f"Formation type must be one of {valid_types}")

@dataclass
class VideoPlumeConfig:
    """Video plume processing configuration with OpenCV integration."""
    video_path: str = MISSING  # Required field
    flip: bool = False
    grayscale: bool = True
    kernel_size: int = 0
    kernel_sigma: float = 1.0
    threshold: Optional[float] = None
    normalize: bool = True
    
    # Preprocessing configuration with structured validation
    preprocessing: PreprocessingConfig = field(default_factory=lambda: PreprocessingConfig())
    sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig())
    
    def __post_init__(self):
        """Validate video processing parameters."""
        if self.kernel_size < 0:
            raise ValueError("Kernel size must be non-negative")
        if self.kernel_size > 0 and self.kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd when > 0")
        if self.threshold is not None and not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

@dataclass
class GymnasiumEnvironmentConfig:
    """Structured configuration for Gymnasium environment with type safety."""
    max_episode_steps: int = 1000
    render_mode: Optional[str] = None
    
    # Action space configuration with validation
    action_space: ActionSpaceConfig = field(default_factory=lambda: ActionSpaceConfig())
    
    # Observation space with structured components
    observation_space: ObservationSpaceConfig = field(default_factory=lambda: ObservationSpaceConfig())
    
    # Reward shaping with performance optimization
    reward_shaping: RewardShapingConfig = field(default_factory=lambda: RewardShapingConfig())
    
    # Termination conditions with clear criteria
    termination: TerminationConfig = field(default_factory=lambda: TerminationConfig())

@dataclass
class RewardShapingConfig:
    """Reward shaping configuration with algorithm optimization."""
    odor_weight: float = 1.0
    distance_weight: float = 0.5
    control_penalty: float = 0.1
    efficiency_bonus: float = 0.2
    success_reward: float = 10.0
    
    def __post_init__(self):
        """Validate reward parameters."""
        if self.odor_weight < 0:
            raise ValueError("Odor weight must be non-negative")

@dataclass
class LoguruLoggingConfig:
    """Structured logging configuration with Loguru integration."""
    level: str = "INFO"
    format: str = "json"  # Options: text, json
    colorize: bool = True
    diagnose: bool = True
    
    # File logging configuration
    file: FileLoggingConfig = field(default_factory=lambda: FileLoggingConfig())
    
    # Console logging configuration  
    console: ConsoleLoggingConfig = field(default_factory=lambda: ConsoleLoggingConfig())
    
    # Performance monitoring integration
    performance: PerformanceLoggingConfig = field(default_factory=lambda: PerformanceLoggingConfig())
    
    # Correlation ID tracking
    correlation: CorrelationConfig = field(default_factory=lambda: CorrelationConfig())

# Register structured configurations with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(name="navigator_config", node=NavigatorConfig)
cs.store(name="video_plume_config", node=VideoPlumeConfig)
cs.store(name="gymnasium_env_config", node=GymnasiumEnvironmentConfig)
cs.store(name="logging_config", node=LoguruLoggingConfig)
```

#### Structured Configuration Usage

##### Loading and Validation

```python
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from odor_plume_nav.config.models import NavigatorConfig, VideoPlumeConfig

# Initialize Hydra with structured config support
with initialize(config_path="../conf", version_base=None):
    # Compose configuration with automatic validation
    cfg = compose(config_name="config")
    
    # Access structured configurations with full type safety
    navigator_config: NavigatorConfig = cfg.navigator
    video_plume_config: VideoPlumeConfig = cfg.video_plume
    
    # Configuration automatically validated at load time
    print(f"Navigator max speed: {navigator_config.max_speed}")
    print(f"Video path: {video_plume_config.video_path}")
    
    # Type errors caught by IDE and runtime validation
    # navigator_config.max_speed = "invalid"  # TypeError caught immediately
```

##### Configuration Composition and Overrides

```python
# Configuration composition with structured validation
with initialize(config_path="../conf"):
    # Base configuration with overrides
    cfg = compose(
        config_name="config",
        overrides=[
            "navigator.max_speed=2.0",  # Validated against NavigatorConfig
            "video_plume.grayscale=false",  # Type-checked boolean
            "logging.level=DEBUG",  # Validated against allowed levels
            "+gymnasium=advanced_env"  # Add structured environment config
        ]
    )
    
    # All overrides automatically validated against dataclass schemas
    assert isinstance(cfg.navigator.max_speed, float)
    assert isinstance(cfg.video_plume.grayscale, bool)
```

##### Factory Pattern with Structured Configs

```python
from odor_plume_nav.config.models import NavigatorConfig
from odor_plume_nav.core import Navigator

def create_navigator_from_structured_config(config: NavigatorConfig) -> Navigator:
    """Create navigator with full type safety and validation."""
    
    # Configuration already validated by dataclass post_init
    navigator = Navigator(
        position=config.position,
        orientation=config.orientation,
        max_speed=config.max_speed,
        control_config=config.control
    )
    
    # Multi-agent configuration handling
    if config.positions is not None:
        navigator.configure_multi_agent(
            positions=config.positions,
            orientations=config.orientations,
            formation=config.formation
        )
    
    return navigator

# Usage with automatic validation
navigator_config = NavigatorConfig(
    position=[10.0, 15.0],
    max_speed=2.5,
    control=ControlConfig(acceleration=0.2, turning_rate=45.0)
)

navigator = create_navigator_from_structured_config(navigator_config)
```

#### Development vs. Production Configuration

```python
# Development configuration with enhanced debugging
@dataclass
class DevelopmentConfig:
    """Development environment structured configuration."""
    environment: str = "development"
    debug_mode: bool = True
    verbose_output: bool = True
    
    navigator: NavigatorConfig = field(default_factory=lambda: NavigatorConfig(
        max_speed=1.0,  # Conservative speed for debugging
        control=ControlConfig(sensor_noise=0.0)  # No noise for reproducibility
    ))
    
    logging: LoguruLoggingConfig = field(default_factory=lambda: LoguruLoggingConfig(
        level="DEBUG",
        format="text",  # Human-readable for development
        colorize=True,
        performance=PerformanceLoggingConfig(enabled=True, step_timing=True)
    ))

# Production configuration with performance optimization
@dataclass  
class ProductionConfig:
    """Production environment structured configuration."""
    environment: str = "production"
    debug_mode: bool = False
    verbose_output: bool = False
    
    navigator: NavigatorConfig = field(default_factory=lambda: NavigatorConfig(
        max_speed=2.0,  # Higher performance in production
        control=ControlConfig(sensor_noise=0.1)  # Realistic noise
    ))
    
    logging: LoguruLoggingConfig = field(default_factory=lambda: LoguruLoggingConfig(
        level="INFO",
        format="json",  # Structured logs for analysis
        colorize=False,
        performance=PerformanceLoggingConfig(
            enabled=True,
            slow_threshold=0.010,  # Stricter production threshold
            alert_on_degradation=True
        )
    ))

# Register environment-specific configurations
cs.store(name="development_config", node=DevelopmentConfig)
cs.store(name="production_config", node=ProductionConfig)
```

#### Configuration Validation and Error Handling

```python
from pydantic import ValidationError
from hydra.errors import ConfigCompositionException

def load_validated_config(config_name: str = "config"):
    """Load configuration with comprehensive validation and error handling."""
    
    try:
        with initialize(config_path="../conf", version_base=None):
            cfg = compose(config_name=config_name)
            
            # Additional business logic validation
            validate_configuration_constraints(cfg)
            
            return cfg
            
    except ValidationError as e:
        print(f"Configuration validation error: {e}")
        print("Please check your configuration files for type errors.")
        raise
        
    except ConfigCompositionException as e:
        print(f"Configuration composition error: {e}")
        print("Please check your configuration file structure.")
        raise

def validate_configuration_constraints(config):
    """Additional validation beyond dataclass constraints."""
    
    # Cross-component validation
    if config.navigator.max_speed > 5.0 and config.environment.debug_mode:
        logger.warning(
            "High speed in debug mode may affect debugging",
            max_speed=config.navigator.max_speed
        )
    
    # Performance constraint validation
    if config.video_plume.kernel_size > 7:
        estimated_fps = estimate_processing_fps(config.video_plume)
        if estimated_fps < 30:
            raise ValueError(
                f"Configuration may not meet 30 FPS requirement. "
                f"Estimated FPS: {estimated_fps}"
            )
    
    # Resource availability validation
    if config.navigator.num_agents and config.navigator.num_agents > 100:
        available_memory = get_available_memory()
        required_memory = estimate_memory_usage(config.navigator.num_agents)
        if required_memory > available_memory:
            raise ValueError(
                f"Insufficient memory for {config.navigator.num_agents} agents. "
                f"Required: {required_memory}MB, Available: {available_memory}MB"
            )
```

### Modular Configuration Examples

#### Plume Model Configuration

The new modular architecture supports multiple plume modeling approaches through configuration:

```yaml
# conf/plume_models/gaussian.yaml - Fast mathematical modeling
_target_: plume_nav_sim.models.plume.GaussianPlumeModel

# Core parameters for analytical plume modeling
source_strength: 1.0
diffusion_rate: 0.1
source_position: [50.0, 50.0]

# Environmental integration
wind_integration: true
decay_rate: 0.01

# Performance optimizations
use_analytical_solution: true
cache_gradients: true
```

```yaml
# conf/plume_models/turbulent.yaml - Realistic physics simulation
_target_: plume_nav_sim.models.plume.TurbulentPlumeModel

# Filament-based modeling parameters
filament_count: 1000
initial_concentration: 1.0
source_position: [50.0, 50.0]

# Physics parameters
dispersion_model: "lagrangian"
turbulence_intensity: 0.3
correlation_length: 5.0

# Performance options
use_numba: true  # Optional JIT acceleration
update_frequency: 1.0  # Time steps between physics updates
```

```yaml
# conf/plume_models/video.yaml - Legacy video-based modeling
_target_: plume_nav_sim.models.plume.VideoPlumeAdapter

# Video source configuration
video_path: "data/experimental_plume.mp4"
grayscale: true
normalize: true

# Frame caching for performance
frame_cache:
  mode: "lru"
  cache_size_mb: 2048
  memory_pressure_threshold: 0.90
```

#### Navigator Memory Configuration

```yaml
# conf/navigator/memory_enabled.yaml - Cognitive navigation with memory
_target_: plume_nav_sim.core.controllers.MemoryBasedController

# Core navigation parameters
position: [10.0, 10.0]
orientation: 0.0
max_speed: 2.0

# Memory system configuration
memory_enabled: true
memory_size: 1000
strategy: "infotaxis"

# Cognitive modeling parameters
exploration_noise: 0.1
information_integration: "bayesian"
decision_horizon: 10
```

```yaml
# conf/navigator/memoryless.yaml - Reactive navigation without memory
_target_: plume_nav_sim.core.controllers.ReactiveController

# Core navigation parameters  
position: [10.0, 10.0]
orientation: 0.0
max_speed: 2.0

# Reactive behavior configuration
memory_enabled: false
strategy: "gradient_following"

# Reactive response parameters
response_gain: 1.0
noise_level: 0.05
```

#### Environmental Dynamics Configuration

```yaml
# conf/wind_fields/turbulent.yaml - Realistic wind dynamics
_target_: plume_nav_sim.models.wind.TurbulentWindField

# Base wind parameters
mean_velocity: [2.0, 0.5]  # [u, v] components
turbulence_intensity: 0.3

# Temporal correlation
correlation_time: 5.0  # seconds
update_frequency: 0.1  # 10 Hz updates

# Spatial variation
correlation_length: 10.0  # meters
boundary_conditions: "periodic"
```

#### Sensor Configuration

```yaml
# conf/sensors/multi_modal.yaml - Multiple sensor types
_target_: plume_nav_sim.core.sensors.MultiModalSensor

sensors:
  primary:
    _target_: plume_nav_sim.core.sensors.ConcentrationSensor
    range: 10.0
    noise_level: 0.05
    
  binary:
    _target_: plume_nav_sim.core.sensors.BinarySensor  
    threshold: 0.1
    response_time: 0.1
    
  gradient:
    _target_: plume_nav_sim.core.sensors.GradientSensor
    spatial_resolution: 0.5
    temporal_smoothing: 0.2

# Optional memory for temporal integration
enable_history: true
history_length: 10
```

#### Complete Research Configuration Example

```yaml
# conf/experiments/memory_comparison.yaml - Complete experiment setup
defaults:
  - base
  - plume_models: gaussian  # Fast, consistent baseline
  - wind_fields: constant   # Minimize confounding variables
  - sensors: simple        # Single concentration sensor
  - _self_

experiment:
  name: "memory_vs_memoryless_comparison" 
  description: "Compare navigation strategies with and without memory"

# Experimental conditions
conditions:
  memoryless:
    navigator:
      _target_: plume_nav_sim.core.controllers.ReactiveController
      memory_enabled: false
      strategy: "gradient_following"
      
  memory_based:
    navigator:
      _target_: plume_nav_sim.core.controllers.MemoryBasedController  
      memory_enabled: true
      strategy: "infotaxis"
      memory_size: 500

# Evaluation parameters
evaluation:
  episodes_per_condition: 1000
  success_criteria:
    odor_threshold: 0.8
    max_episode_steps: 2000
    
  metrics:
    - "success_rate"
    - "path_efficiency" 
    - "search_time"
    - "exploration_coverage"

# Analysis configuration
analysis:
  statistical_tests: ["t_test", "mann_whitney"]
  significance_level: 0.05
  effect_size_metrics: ["cohen_d", "cliff_delta"]
```

### RL Configuration Examples

#### Structured Algorithm Configuration

Using dataclass-based RL algorithm configuration with automatic validation:

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from hydra.core.config_store import ConfigStore

@dataclass
class PPOAlgorithmConfig:
    """Structured PPO algorithm configuration with hyperparameter validation."""
    algorithm: str = "PPO"
    
    # Core PPO hyperparameters with validation
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    entropy_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training configuration with type safety
    training: PPOTrainingConfig = field(default_factory=lambda: PPOTrainingConfig())
    
    def __post_init__(self):
        """Validate PPO hyperparameters."""
        if not 0 < self.learning_rate < 1:
            raise ValueError("Learning rate must be between 0 and 1")
        if self.n_steps <= 0:
            raise ValueError("Number of steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("Gamma must be between 0 and 1")

@dataclass
class PPOTrainingConfig:
    """PPO training pipeline configuration."""
    total_timesteps: int = 1000000
    eval_freq: int = 10000
    eval_episodes: int = 20
    save_freq: int = 50000
    tensorboard_log: str = "./tensorboard_logs"
    verbose: int = 1
    
    def __post_init__(self):
        """Validate training parameters."""
        if self.total_timesteps <= 0:
            raise ValueError("Total timesteps must be positive")
        if self.eval_freq <= 0:
            raise ValueError("Evaluation frequency must be positive")

@dataclass
class SACAlgorithmConfig:
    """Structured SAC algorithm configuration with parameter constraints."""
    algorithm: str = "SAC"
    
    # SAC-specific hyperparameters
    learning_rate: float = 3e-4
    buffer_size: int = 1000000
    learning_starts: int = 10000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"  # or float value
    target_update_interval: int = 1
    target_entropy: str = "auto"  # or float value
    
    training: SACTrainingConfig = field(default_factory=lambda: SACTrainingConfig())

@dataclass
class SACTrainingConfig:
    """SAC training configuration with off-policy optimization."""
    total_timesteps: int = 500000
    eval_freq: int = 5000
    eval_episodes: int = 10
    save_freq: int = 25000
    verbose: int = 1

# Register algorithm configurations
cs.store(group="rl/algorithms", name="ppo", node=PPOAlgorithmConfig)
cs.store(group="rl/algorithms", name="sac", node=SACAlgorithmConfig)
```

#### YAML Configuration with Structured Config Integration

```yaml
# conf/rl/algorithms/ppo.yaml - now with structured config annotations
# @package rl.algorithm
_target_: odor_plume_nav.config.models.PPOAlgorithmConfig

# All parameters automatically validated against dataclass
algorithm: PPO
learning_rate: 3e-4
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
clip_range_vf: null
entropy_coef: 0.0
vf_coef: 0.5
max_grad_norm: 0.5

training:
  total_timesteps: 1000000
  eval_freq: 10000
  eval_episodes: 20
  save_freq: 50000
  tensorboard_log: ./tensorboard_logs
  verbose: 1
```

```yaml
# conf/rl/algorithms/sac.yaml
algorithm: SAC
hyperparameters:
  learning_rate: 3e-4
  buffer_size: 1000000
  learning_starts: 10000
  batch_size: 256
  tau: 0.005
  gamma: 0.99
  train_freq: 1
  gradient_steps: 1
  ent_coef: auto
  target_update_interval: 1
  target_entropy: auto
  
training:
  total_timesteps: 500000
  eval_freq: 5000
  eval_episodes: 10
  save_freq: 25000
  verbose: 1
```

#### Structured Environment Configuration

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np

@dataclass
class ActionSpaceConfig:
    """Action space configuration with automatic Gymnasium space creation."""
    type: str = "Box"
    low: List[float] = field(default_factory=lambda: [-1.0, -1.0])
    high: List[float] = field(default_factory=lambda: [1.0, 1.0])
    dtype: str = "float32"
    
    def create_space(self):
        """Create Gymnasium action space from configuration."""
        import gymnasium as gym
        return gym.spaces.Box(
            low=np.array(self.low, dtype=self.dtype),
            high=np.array(self.high, dtype=self.dtype)
        )

@dataclass
class ObservationComponentConfig:
    """Individual observation component configuration."""
    type: str = "Box"
    shape: List[int] = field(default_factory=list)
    low: float = 0.0
    high: float = 1.0
    dtype: str = "float32"

@dataclass
class ObservationSpaceConfig:
    """Structured observation space with component validation."""
    odor_concentration: ObservationComponentConfig = field(
        default_factory=lambda: ObservationComponentConfig(
            shape=[1], low=0.0, high=1.0
        )
    )
    agent_position: ObservationComponentConfig = field(
        default_factory=lambda: ObservationComponentConfig(
            shape=[2], low=0.0, high=100.0
        )
    )
    agent_orientation: ObservationComponentConfig = field(
        default_factory=lambda: ObservationComponentConfig(
            shape=[1], low=-np.pi, high=np.pi
        )
    )
    plume_gradient: ObservationComponentConfig = field(
        default_factory=lambda: ObservationComponentConfig(
            shape=[2], low=-1.0, high=1.0
        )
    )
    
    def create_space(self):
        """Create Gymnasium Dict observation space."""
        import gymnasium as gym
        
        components = {}
        for name, component in self.__dict__.items():
            components[name] = gym.spaces.Box(
                low=component.low,
                high=component.high,
                shape=component.shape,
                dtype=getattr(np, component.dtype)
            )
        
        return gym.spaces.Dict(components)

@dataclass
class AdvancedEnvironmentConfig:
    """Advanced environment configuration with full validation."""
    max_episode_steps: int = 2000
    render_mode: Optional[str] = None  # 'human', 'rgb_array', or None
    
    # Structured space configurations
    action_space: ActionSpaceConfig = field(default_factory=ActionSpaceConfig)
    observation_space: ObservationSpaceConfig = field(default_factory=ObservationSpaceConfig)
    
    # Reward and termination with validation
    reward_shaping: RewardShapingConfig = field(default_factory=lambda: RewardShapingConfig(
        odor_weight=1.0,
        distance_weight=0.5,
        control_penalty=0.1,
        efficiency_bonus=0.2,
        success_reward=10.0
    ))
    
    termination: TerminationConfig = field(default_factory=lambda: TerminationConfig(
        max_distance_from_start=150.0,
        min_odor_threshold=0.01,
        success_odor_threshold=0.8
    ))
    
    def __post_init__(self):
        """Validate environment configuration."""
        if self.max_episode_steps <= 0:
            raise ValueError("Max episode steps must be positive")
        
        valid_render_modes = {None, 'human', 'rgb_array'}
        if self.render_mode not in valid_render_modes:
            raise ValueError(f"Render mode must be one of {valid_render_modes}")

@dataclass
class TerminationConfig:
    """Termination condition configuration with clear criteria."""
    max_distance_from_start: float = 150.0
    min_odor_threshold: float = 0.01
    success_odor_threshold: float = 0.8
    max_steps_without_progress: Optional[int] = None
    
    def __post_init__(self):
        """Validate termination parameters."""
        if not 0 <= self.min_odor_threshold <= 1:
            raise ValueError("Min odor threshold must be between 0 and 1")
        if not 0 <= self.success_odor_threshold <= 1:
            raise ValueError("Success odor threshold must be between 0 and 1")
        if self.min_odor_threshold >= self.success_odor_threshold:
            raise ValueError("Min threshold must be less than success threshold")

# Register environment configurations
cs.store(group="rl/environments", name="advanced", node=AdvancedEnvironmentConfig)
```

#### YAML Configuration with Structured Config Integration

```yaml
# conf/rl/environments/advanced.yaml - with dataclass validation
# @package rl.environment
_target_: odor_plume_nav.config.models.AdvancedEnvironmentConfig

environment:
  max_episode_steps: 2000
  render_mode: null  # Automatically validated against allowed values
  
action_space:
  type: Box
  low: [-1.0, -1.0]
  high: [1.0, 1.0]
  dtype: float32

observation_space:
  odor_concentration:
    type: Box
    shape: [1]
    low: 0.0
    high: 1.0
    dtype: float32
  agent_position:
    type: Box
    shape: [2]
    low: 0.0  # Broadcast to [0.0, 0.0] automatically
    high: 100.0  # Broadcast to [100.0, 100.0] automatically
    dtype: float32
  agent_orientation:
    type: Box
    shape: [1]
    low: -3.14159
    high: 3.14159
    dtype: float32

reward_shaping:
  odor_weight: 1.0
  distance_weight: 0.5
  control_penalty: 0.1
  efficiency_bonus: 0.2
  success_reward: 10.0
  
termination:
  max_distance_from_start: 150.0
  min_odor_threshold: 0.01
  success_odor_threshold: 0.8
```

#### Training Pipeline Configuration

```yaml
# conf/rl/training.yaml
defaults:
  - algorithms: ppo
  - environments: advanced
  - _self_

training_pipeline:
  parallel_envs: 4
  evaluation:
    enabled: true
    frequency: 10000
    episodes: 20
    deterministic: true
    
  checkpointing:
    enabled: true
    frequency: 50000
    keep_best: true
    max_checkpoints: 5
    
  logging:
    tensorboard: true
    wandb: false
    csv: true
    console_level: INFO
    
  callbacks:
    early_stopping:
      enabled: false
      patience: 100000
      min_delta: 0.01
    learning_rate_scheduler:
      enabled: false
      schedule: linear
      final_lr: 1e-6
```

### Basic Configuration Usage

```python
from hydra import compose, initialize
from odor_plume_nav.api import create_gymnasium_environment

# Basic RL configuration loading
with initialize(config_path="../conf", version_base=None):
    cfg = compose(config_name="config", config_path="../conf/rl")
    env = create_gymnasium_environment(cfg.environment)

# Dynamic parameter overrides for RL training
with initialize(config_path="../conf"):
    cfg = compose(
        config_name="config",
        overrides=[
            "rl/algorithms=sac",
            "rl/environments=advanced",
            "rl.training.total_timesteps=1000000",
            "rl.environment.reward_shaping.odor_weight=2.0"
        ]
    )
```

### Environment Variable Integration

The configuration system supports secure credential management through environment variables:

```yaml
# conf/config.yaml
database:
  url: ${oc.env:DATABASE_URL,sqlite:///local.db}
  username: ${oc.env:DB_USER,admin}
  password: ${oc.env:DB_PASSWORD}

video_plume:
  video_path: ${oc.env:VIDEO_PATH,data/videos/example_plume.mp4}
  
navigator:
  max_speed: ${oc.env:NAVIGATOR_MAX_SPEED,1.5}

# Frame caching configuration with environment variable integration
environment:
  frame_cache:
    mode: ${oc.env:FRAME_CACHE_MODE,lru}
    cache_size_mb: ${oc.env:FRAME_CACHE_SIZE_MB,2048}
    memory_pressure_threshold: ${oc.env:FRAME_CACHE_PRESSURE_THRESHOLD,0.90}
    preload_enabled: ${oc.env:FRAME_CACHE_PRELOAD_ENABLED,false}
  include_video_frame: ${oc.env:INCLUDE_VIDEO_FRAME,false}

# Logging configuration with environment variables
logging:
  level: ${oc.env:LOG_LEVEL,INFO}
  json_sink_enabled: ${oc.env:LOG_JSON_SINK,true}
  performance_tracking: ${oc.env:LOG_PERFORMANCE_TRACKING,true}

rl:
  tensorboard_log: ${oc.env:TENSORBOARD_LOG_DIR,./tensorboard_logs}
  checkpoint_dir: ${oc.env:CHECKPOINT_DIR,./checkpoints}
  model_save_path: ${oc.env:MODEL_SAVE_PATH,./models}
```

#### Environment Variable Setup

Create a `.env` file in your project root:

```bash
# .env file
ENVIRONMENT_TYPE=development
DATABASE_URL=postgresql://user:password@localhost:5432/plume_nav
VIDEO_PATH=/data/experiments/high_resolution_plume.mp4
NAVIGATOR_MAX_SPEED=2.0
DEBUG=true
LOG_LEVEL=INFO

# RL-specific environment variables
TENSORBOARD_LOG_DIR=/data/tensorboard_logs
CHECKPOINT_DIR=/data/checkpoints
MODEL_SAVE_PATH=/data/trained_models
WANDB_PROJECT=odor_plume_navigation
WANDB_API_KEY=your_wandb_api_key
```

### Migration from Legacy Configuration

#### Migration Guide: Unstructured YAML to Dataclass Configuration

The library has evolved from unstructured YAML configuration to Pydantic-validated dataclass configuration, providing type safety, IDE support, and runtime validation.

#### Legacy vs. Modern Configuration Comparison

**Legacy Unstructured Configuration (v0.1.x):**
```yaml
# configs/default.yaml - Old approach
navigator:
  position: [10.0, 15.0]  # No type validation
  max_speed: "2.0"        # String accepted, potential runtime errors
  invalid_field: "value" # Unknown fields silently ignored

video_plume:
  video_path: null        # Missing required field not caught
  kernel_size: 2          # Even numbers allowed, causes OpenCV errors
  threshold: 1.5          # Invalid range not validated
```

**Modern Structured Configuration (v0.2.x):**
```yaml
# conf/config.yaml - New approach with validation
# @package _global_
_target_: odor_plume_nav.config.models.NavigatorConfig

navigator:
  position: [10.0, 15.0]  # Type-validated as List[float]
  max_speed: 2.0          # Must be float, validated at load time
  # invalid_field: "value" # Rejected with clear error message

video_plume:
  video_path: "data/example.mp4"  # Required field enforced
  kernel_size: 3          # Must be odd, validated in __post_init__
  threshold: 0.8          # Range [0.0, 1.0] enforced automatically
```

#### Step-by-Step Migration Process

##### Step 1: Install Updated Dependencies

```bash
# Ensure you have the latest version with structured config support
pip install "plume_nav_sim[rl]>=0.2.0"
```

##### Step 2: Update Configuration Directory Structure

```bash
# Old structure
configs/
├── default.yaml
├── example_user_config.yaml
└── README.md

# New structure
conf/
├── base.yaml          # Replaces default.yaml
├── config.yaml        # Replaces example_user_config.yaml  
├── rl/                # New: RL-specific configurations
│   ├── algorithms/
│   ├── environments/
│   └── training.yaml
└── local/             # New: environment-specific overrides
    ├── development.yaml
    └── production.yaml
```

##### Step 3: Convert Configuration Loading Code

**Legacy Loading Approach:**
```python
# Old approach - error-prone manual loading
from odor_plume_nav.services.config_loader import load_config
import yaml

config = load_config("configs/default.yaml")
# No validation, runtime errors possible
navigator = create_navigator(
    position=config["navigator"]["position"],  # Dictionary access
    max_speed=float(config["navigator"]["max_speed"])  # Manual type conversion
)
```

**Modern Structured Loading:**
```python
# New approach - type-safe with automatic validation
from hydra import compose, initialize
from odor_plume_nav.config.models import NavigatorConfig
from odor_plume_nav.core import Navigator

with initialize(config_path="../conf", version_base=None):
    cfg = compose(config_name="config")
    
    # Fully typed configuration object
    navigator_config: NavigatorConfig = cfg.navigator
    
    # Type safety and IDE autocomplete
    navigator = Navigator.from_config(navigator_config)
    # All validation happened at cfg composition time
```

##### Step 4: Update Parameter Override Patterns

**Legacy Override Pattern:**
```python
# Old approach - manual dictionary manipulation
config = load_config("configs/default.yaml")
config["navigator"]["max_speed"] = 2.5  # No validation
config["video_plume"]["nonexistent"] = "value"  # Silently accepted
```

**Modern Override Pattern:**
```python
# New approach - validated overrides
with initialize(config_path="../conf"):
    cfg = compose(
        config_name="config",
        overrides=[
            "navigator.max_speed=2.5",  # Validated against NavigatorConfig
            # "navigator.nonexistent=value"  # Rejected with clear error
        ]
    )
```

##### Step 5: Update CLI Integration

**Legacy CLI Usage:**
```bash
# Old approach - manual parameter passing
python scripts/run_simulation.py \
    --config configs/my_config.yaml \
    --max-speed 2.0 \
    --video-path data/video.mp4
```

**Modern CLI Usage:**
```bash
# New approach - Hydra integration with validation
plume-nav-sim run \
    navigator.max_speed=2.0 \
    video_plume.video_path=data/video.mp4 \
    --config-name my_config
```

#### Configuration Validation Benefits

##### Immediate Error Detection

```python
# Example validation errors caught at startup:

# Type validation
navigator:
  max_speed: "invalid"  # ValidationError: Input should be a valid number

# Range validation  
video_plume:
  threshold: 1.5        # ValidationError: Threshold must be between 0.0 and 1.0

# Required field validation
video_plume:
  # video_path missing   # ValidationError: Field required

# Custom business logic validation
navigator:
  speed: 3.0
  max_speed: 2.0        # ValidationError: Speed cannot exceed max_speed

# Cross-component validation
gymnasium:
  max_episode_steps: -100  # ValidationError: Max episode steps must be positive
```

##### IDE Integration Benefits

```python
from odor_plume_nav.config.models import NavigatorConfig

# Full IDE autocomplete and type hints
config = NavigatorConfig(
    position=[10.0, 15.0],  # IDE knows this is List[float]
    max_speed=2.0,          # IDE knows this is float
    # IDE will suggest valid field names and types
)

# Type checking with mypy
def create_navigator(config: NavigatorConfig) -> Navigator:
    # mypy verifies all attribute access
    return Navigator(
        position=config.position,      # Validated List[float]
        max_speed=config.max_speed,    # Validated float
    )
```

#### Backward Compatibility Strategy

The library maintains backward compatibility during the transition period:

```python
# Compatibility loading function for legacy configurations
from odor_plume_nav.config.legacy import load_legacy_config
from odor_plume_nav.config.models import migrate_legacy_config

def load_config_with_migration(config_path: str):
    """Load configuration with automatic migration from legacy format."""
    
    if config_path.endswith('configs/'):  # Legacy path pattern
        # Load legacy configuration
        legacy_config = load_legacy_config(config_path)
        
        # Migrate to structured format with validation
        structured_config = migrate_legacy_config(legacy_config)
        
        # Warn user about deprecated usage
        logger.warning(
            "Using legacy configuration format. "
            "Consider migrating to structured configuration in conf/ directory.",
            legacy_path=config_path
        )
        
        return structured_config
    
    else:
        # Load modern structured configuration
        with initialize(config_path=config_path):
            return compose(config_name="config")

# Usage supports both formats during migration
config = load_config_with_migration("configs/")  # Legacy support
# config = load_config_with_migration("conf/")   # Modern approach
```

#### Migration Checklist

- [ ] **Install updated dependencies** with dataclass support
- [ ] **Create new `conf/` directory** structure  
- [ ] **Migrate configuration files** from `configs/` to `conf/`
- [ ] **Add structured config annotations** to YAML files
- [ ] **Update configuration loading code** to use Hydra compose()
- [ ] **Replace manual parameter access** with typed configuration objects
- [ ] **Update CLI usage** to use new Hydra integration
- [ ] **Test configuration validation** with intentional errors
- [ ] **Update documentation** and examples to use structured configs
- [ ] **Remove legacy configuration files** after validation

If migrating from the old `configs/` structure to the new Hydra-based `conf/` system:

#### Legacy Structure (Old)
```
configs/
├── default.yaml
├── example_user_config.yaml
└── README.md
```

#### New Hydra Structure
```
conf/
├── base.yaml          # Replaces default.yaml
├── config.yaml        # Replaces example_user_config.yaml
├── rl/                # New: RL-specific configurations
│   ├── algorithms/
│   ├── environments/
│   └── training.yaml
└── local/             # New: environment-specific overrides
    ├── development.yaml
    └── production.yaml
```

#### Migration Steps

1. **Copy base parameters**: Move `configs/default.yaml` content to `conf/base.yaml`
2. **User customizations**: Move `configs/example_user_config.yaml` to `conf/config.yaml`
3. **RL configuration**: Create new `conf/rl/` directory with algorithm and environment configs
4. **Environment setup**: Create environment-specific files in `conf/local/`
5. **Update imports**: Change from:
   ```python
   # Old approach
   from odor_plume_nav.services.config_loader import load_config
   config = load_config("configs/default.yaml")
   ```
   
   To:
   ```python
   # New Hydra approach
   from hydra import compose, initialize
   with initialize(config_path="../conf"):
       cfg = compose(config_name="config")
   ```

6. **CLI migration**: Replace manual script execution with new CLI commands:
   ```bash
   # Old approach
   python scripts/run_simulation.py --config configs/my_config.yaml
   
   # New approach (simulation)
   plume-nav-sim run --config-name my_config
   
   # New approach (RL training)
   plume-nav-sim train --algorithm PPO --config-name my_rl_config
   ```

## Development Workflow

### Setup Development Environment

```bash
# Clone and setup
git clone https://github.com/organization/odor_plume_nav.git
cd odor_plume_nav

# Standard development setup
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev,docs,viz,rl]"

# Install pre-commit hooks
pre-commit install

# Alternative: Make-based setup
make setup-dev
```

### Makefile Commands

The project includes comprehensive Makefile automation:

```bash
# Development commands
make install-dev       # pip install with dev dependencies
make setup-dev         # Complete development environment setup
make install          # Standard pip install

# Code quality
make format           # Run black and isort formatting
make lint            # Run flake8 linting
make type-check      # Run mypy type checking
make test            # Run pytest with coverage
make test-all        # Run all quality checks

# Build and distribution
make build           # Build wheel and sdist using python -m build
make clean           # Clean build artifacts

# Documentation
make docs            # Build Sphinx documentation
make docs-serve      # Serve documentation locally

# Docker commands
make docker-build    # Build Docker image
make docker-run      # Run container
make docker-dev      # Development environment with docker-compose
```

### Pre-commit Hooks

The project includes automated code quality checks:

```yaml
# .pre-commit-config.yaml (example hooks)
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/isort
    rev: 5.13.0
    hooks:
      - id: isort
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

### Testing Strategy

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=odor_plume_nav --cov-report=html

# Run specific test categories
pytest -m "not slow"          # Skip slow tests
pytest -m integration         # Integration tests only
pytest tests/unit/            # Unit tests only
pytest tests/rl/              # RL-specific tests only

# Run with parallel execution
pytest -n auto               # Parallel execution

# Test RL environments specifically
pytest tests/rl/test_gymnasium_env.py -v
pytest tests/rl/test_training_workflows.py -v
```

## Advanced Features

### Multi-Run Experiment Management

```bash
# Systematic parameter exploration
plume-nav-sim run --multirun \
  navigator.max_speed=1.0,1.5,2.0 \
  navigator.angular_velocity=0.1,0.2,0.3 \
  video_plume.gaussian_blur.sigma=1.0,2.0,3.0

# RL hyperparameter sweeps
plume-nav-sim train --multirun \
  --algorithm PPO \
  rl.hyperparameters.learning_rate=1e-4,3e-4,1e-3 \
  rl.hyperparameters.n_steps=1024,2048,4096 \
  rl.training.total_timesteps=500000

# Results organized automatically in:
# outputs/multirun/2024-01-15_10-30-00/
# ├── run_0_navigator.max_speed=1.0,navigator.angular_velocity=0.1,video_plume.gaussian_blur.sigma=1.0/
# ├── run_1_navigator.max_speed=1.0,navigator.angular_velocity=0.1,video_plume.gaussian_blur.sigma=2.0/
# └── ...
```

### Docker-Compose Development Environment

The library includes a complete development infrastructure:

```yaml
# docker-compose.yml
version: '3.8'
services:
  odor_plume_nav:
    build: .
    volumes:
      - ./src:/app/src
      - ./conf:/app/conf
      - ./data:/app/data
      - ./outputs:/app/outputs
    depends_on:
      - postgres
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/odor_nav
      - ENVIRONMENT_TYPE=development

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: odor_nav
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  pgadmin:
    image: dpage/pgadmin4:latest
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@example.com
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "8080:80"
    depends_on:
      - postgres

volumes:
  postgres_data:
```

### Performance Optimization

```python
# Configure for high-performance computation
from odor_plume_nav.utils import configure_performance

# Optimize NumPy and OpenCV for multi-core systems
configure_performance(
    numpy_threads=8,
    opencv_threads=6,
    use_gpu=True
)

# Environment variable configuration
export NUMPY_THREADS=8
export OPENCV_OPENCL=true
export MATPLOTLIB_BACKEND=Agg  # Headless mode for batch processing
```

## Project Structure

```
plume_nav_sim/
├── src/plume_nav_sim/            # Main library package
│   ├── __init__.py                   # Public API exports
│   ├── api/                          # Public interfaces
│   │   ├── __init__.py
│   │   └── navigation.py             # Main API functions
│   ├── cli/                          # Command-line interface
│   │   ├── __init__.py
│   │   └── main.py                   # CLI entry point
│   ├── config/                       # Configuration management
│   │   ├── __init__.py
│   │   └── schemas.py                # Pydantic validation schemas
│   ├── core/                         # Core business logic
│   │   ├── __init__.py
│   │   ├── navigator.py              # Navigation protocols
│   │   ├── controllers.py            # Agent controllers
│   │   └── sensors.py                # Sensor strategies
│   ├── data/                         # Data processing
│   │   ├── __init__.py
│   │   └── video_plume.py            # Video plume processing
│   ├── db/                           # Database integration (future)
│   │   └── session.py                # Session management
│   ├── environments/                 # RL environments
│   │   ├── __init__.py
│   │   ├── gymnasium_env.py          # Main Gymnasium environment
│   │   ├── spaces.py                 # Action/observation space definitions
│   │   ├── wrappers.py               # Environment preprocessing wrappers
│   │   └── video_plume.py            # Video plume environment
│   ├── shims/                        # Backward compatibility layer (new)
│   │   ├── __init__.py
│   │   └── gym_make.py               # Legacy Gym compatibility shim
│   ├── rl/                           # RL utilities
│   │   ├── __init__.py
│   │   ├── training.py               # Training utilities and workflows
│   │   └── policies.py               # Custom policy implementations
│   ├── utils/                        # Utilities
│   │   ├── __init__.py
│   │   ├── seed_manager.py           # Reproducibility
│   │   ├── visualization.py          # Plotting and animation
│   │   ├── logging.py                # Structured logging configuration
│   │   └── frame_cache.py            # High-performance frame caching (new)
├── conf/                             # Hydra configuration
│   ├── base.yaml                     # Foundation defaults
│   ├── config.yaml                   # User customizations
│   ├── rl/                           # RL-specific configurations
│   │   ├── algorithms/               # Algorithm hyperparameters
│   │   ├── environments/             # Environment configurations
│   │   └── training.yaml             # Training pipeline settings
│   └── local/                        # Environment-specific
│       ├── credentials.yaml.template
│       └── paths.yaml.template
├── tests/                            # Test suite
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   └── rl/                           # RL-specific tests
├── notebooks/                        # Example notebooks
│   ├── demos/                        # Demonstration notebooks
│   ├── rl_examples/                  # RL training examples
│   └── exploratory/                  # Research notebooks
├── workflow/                         # Workflow definitions (future)
│   ├── dvc/                          # DVC pipelines
│   └── snakemake/                    # Snakemake workflows
├── docker-compose.yml               # Development environment
├── Dockerfile                       # Container image
├── Makefile                         # Development automation
├── pyproject.toml                   # Package configuration
└── README.md                        # This file
```

## High-Performance Frame Caching

### Overview

The library features an advanced frame caching system designed to dramatically accelerate reinforcement learning training by eliminating video decoding bottlenecks. The caching system provides multiple operational modes with configurable memory limits, achieving sub-10ms environment step times for optimal training performance.

### Cache Operating Modes

**LRU (Least Recently Used) Cache Mode** - Recommended for most scenarios
- Intelligent caching with automatic memory management
- Default 2 GiB memory limit with configurable size
- Automatic eviction based on LRU policy when memory pressure detected
- Typically achieves >90% cache hit rates for training workloads

**Full Preload Mode** - Maximum performance for memory-rich environments
- Complete video preloading into memory during initialization
- Achieves consistent <5ms frame retrieval times
- Best for shorter videos and high-memory systems

**Direct I/O Mode** - Bypass caching for debugging or memory-constrained scenarios
- Minimal memory usage with direct file access
- Variable retrieval times (20-100ms typical)

### Configuration Examples

```python
from plume_nav_sim.api.navigation import create_gymnasium_environment

# LRU caching (recommended for most scenarios)
env = create_gymnasium_environment(
    config_path="conf/config.yaml",
    frame_cache="lru"
)

# Full preload caching (maximum performance)
env = create_gymnasium_environment(
    config_path="conf/config.yaml", 
    frame_cache="all"
)

# Disable caching (debugging or memory constraints)
env = create_gymnasium_environment(
    config_path="conf/config.yaml",
    frame_cache="none"
)
```

### Performance Monitoring

Access real-time performance metrics through the environment info dictionary:

```python
obs, reward, terminated, truncated, info = env.step(action)

# Access comprehensive performance statistics
perf_stats = info["perf_stats"]
print(f"Step execution time: {perf_stats['step_time_ms']:.2f}ms")
print(f"Frame retrieval time: {perf_stats['frame_retrieval_ms']:.2f}ms")
print(f"Cache hit rate: {perf_stats['cache_hit_rate']:.2%}")
print(f"Cache memory usage: {perf_stats['cache_memory_mb']:.1f}MB")
print(f"Estimated FPS: {perf_stats['fps_estimate']:.1f}")
```

### Environment Variable Configuration

Configure caching behavior via environment variables for deployment flexibility:

```bash
# Set cache mode and size
export FRAME_CACHE_MODE=lru
export FRAME_CACHE_SIZE_MB=4096  # 4 GiB cache limit

# Run training with enhanced caching
plume-nav-sim train --algorithm PPO
```

## Integration Examples

### Research Workflow Integration

#### place_mem_rl Training Loop Integration

```python
# place_mem_rl training loop with frame caching and performance monitoring
from plume_nav_sim.api.navigation import create_gymnasium_environment
from stable_baselines3 import PPO
import time

def train_place_mem_rl_agent():
    """Enhanced training loop with frame caching and performance analytics."""
    
    # Create environment with optimized frame caching
    env = create_gymnasium_environment(
        config_path="conf/rl_config.yaml",
        frame_cache="lru",  # Enable LRU caching for training efficiency
        include_video_frame=False  # Disable frame data for training (memory optimization)
    )
    
    # Initialize PPO agent
    model = PPO("MultiInputPolicy", env, verbose=1)
    
    # Training loop with performance monitoring
    start_time = time.time()
    performance_history = []
    
    obs, info = env.reset()
    for step in range(100000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Access performance metrics for optimization
        perf_stats = info["perf_stats"]
        performance_history.append({
            "step": step,
            "step_time_ms": perf_stats["step_time_ms"],
            "cache_hit_rate": perf_stats["cache_hit_rate"],
            "fps_estimate": perf_stats["fps_estimate"]
        })
        
        # Log performance every 1000 steps
        if step % 1000 == 0:
            avg_step_time = sum(p["step_time_ms"] for p in performance_history[-1000:]) / 1000
            avg_hit_rate = sum(p["cache_hit_rate"] for p in performance_history[-1000:]) / 1000
            
            print(f"Step {step}: Avg step time: {avg_step_time:.2f}ms, "
                  f"Cache hit rate: {avg_hit_rate:.2%}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    # Training performance analysis
    total_time = time.time() - start_time
    avg_performance = {
        "total_training_time": total_time,
        "steps_per_second": 100000 / total_time,
        "avg_step_time_ms": sum(p["step_time_ms"] for p in performance_history) / len(performance_history),
        "avg_cache_hit_rate": sum(p["cache_hit_rate"] for p in performance_history) / len(performance_history)
    }
    
    print(f"Training completed: {avg_performance}")
    return model, avg_performance
```

#### Demo/Analysis Notebook Integration

```python
# demo/analysis notebooks with video frame access and structured logging
import numpy as np
import matplotlib.pyplot as plt
from plume_nav_sim.api.navigation import create_gymnasium_environment
from loguru import logger

def analyze_plume_navigation_behavior():
    """Comprehensive plume navigation analysis with video frame access."""
    
    # Setup structured logging for analysis
    logger.add("analysis_session.json", 
               format="{time} | {level} | {extra} | {message}",
               serialize=True)
    
    # Create environment with video frame access enabled
    env = create_gymnasium_environment(
        config_path="conf/analysis_config.yaml",
        frame_cache="lru",
        include_video_frame=True  # Enable video frame access for analysis
    )
    
    trajectory_data = []
    frame_analysis = []
    
    with logger.contextualize(analysis_session="plume_behavior_study"):
        logger.info("Starting comprehensive plume navigation analysis")
        
        obs, info = env.reset()
        for step in range(1000):
            # Use your analysis policy or random actions
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Extract comprehensive step data including video frame
            step_data = {
                "step": step,
                "agent_position": obs["agent_position"].tolist(),
                "odor_concentration": obs["odor_concentration"].tolist(),
                "reward": reward,
                "performance_stats": info["perf_stats"]
            }
            trajectory_data.append(step_data)
            
            # Access and analyze video frame when available
            if "video_frame" in info:
                video_frame = info["video_frame"]  # NumPy array shape (H, W) or (H, W, C)
                
                frame_stats = {
                    "step": step,
                    "frame_shape": video_frame.shape,
                    "mean_intensity": float(video_frame.mean()),
                    "std_intensity": float(video_frame.std()),
                    "max_intensity": float(video_frame.max()),
                    "min_intensity": float(video_frame.min()),
                    "unique_values": len(np.unique(video_frame))
                }
                frame_analysis.append(frame_stats)
                
                # Log interesting frame events
                if frame_stats["mean_intensity"] > 0.7:
                    logger.info("High-intensity frame detected",
                               step=step,
                               mean_intensity=frame_stats["mean_intensity"],
                               agent_position=obs["agent_position"].tolist())
            
            # Log performance metrics
            perf_stats = info["perf_stats"]
            logger.debug("Analysis step completed",
                        step=step,
                        **perf_stats)
            
            if terminated or truncated:
                logger.info("Episode completed", episode_steps=step)
                obs, info = env.reset()
    
    # Generate comprehensive analysis report
    analysis_results = {
        "trajectory_stats": {
            "total_steps": len(trajectory_data),
            "avg_reward": sum(d["reward"] for d in trajectory_data) / len(trajectory_data),
            "max_reward": max(d["reward"] for d in trajectory_data),
            "final_position": trajectory_data[-1]["agent_position"] if trajectory_data else None
        },
        "frame_analysis": {
            "frames_analyzed": len(frame_analysis),
            "avg_frame_intensity": sum(f["mean_intensity"] for f in frame_analysis) / len(frame_analysis) if frame_analysis else 0,
            "intensity_std": np.std([f["mean_intensity"] for f in frame_analysis]) if frame_analysis else 0
        },
        "performance_metrics": {
            "avg_step_time_ms": sum(d["performance_stats"]["step_time_ms"] for d in trajectory_data) / len(trajectory_data),
            "avg_cache_hit_rate": sum(d["performance_stats"]["cache_hit_rate"] for d in trajectory_data) / len(trajectory_data),
            "avg_fps": sum(d["performance_stats"]["fps_estimate"] for d in trajectory_data) / len(trajectory_data)
        }
    }
    
    logger.info("Analysis completed", **analysis_results)
    
    # Visualization with frame data correlation
    if frame_analysis:
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Trajectory with reward coloring
        plt.subplot(2, 3, 1)
        positions = np.array([d["agent_position"] for d in trajectory_data])
        rewards = [d["reward"] for d in trajectory_data]
        plt.scatter(positions[:, 0], positions[:, 1], c=rewards, cmap='viridis')
        plt.colorbar(label='Reward')
        plt.title('Agent Trajectory (colored by reward)')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
        # Subplot 2: Frame intensity over time
        plt.subplot(2, 3, 2)
        frame_steps = [f["step"] for f in frame_analysis]
        frame_intensities = [f["mean_intensity"] for f in frame_analysis]
        plt.plot(frame_steps, frame_intensities)
        plt.title('Frame Intensity Over Time')
        plt.xlabel('Step')
        plt.ylabel('Mean Frame Intensity')
        
        # Subplot 3: Performance metrics
        plt.subplot(2, 3, 3)
        step_times = [d["performance_stats"]["step_time_ms"] for d in trajectory_data]
        plt.plot(step_times)
        plt.axhline(y=10, color='r', linestyle='--', label='10ms target')
        plt.title('Step Execution Time')
        plt.xlabel('Step')
        plt.ylabel('Step Time (ms)')
        plt.legend()
        
        # Subplot 4: Cache performance
        plt.subplot(2, 3, 4)
        cache_hit_rates = [d["performance_stats"]["cache_hit_rate"] for d in trajectory_data]
        plt.plot(cache_hit_rates)
        plt.axhline(y=0.9, color='r', linestyle='--', label='90% target')
        plt.title('Cache Hit Rate')
        plt.xlabel('Step')
        plt.ylabel('Hit Rate')
        plt.legend()
        
        # Subplot 5: Frame intensity vs reward correlation
        plt.subplot(2, 3, 5)
        # Align frame data with trajectory data
        aligned_intensities = []
        aligned_rewards = []
        for frame in frame_analysis:
            if frame["step"] < len(trajectory_data):
                aligned_intensities.append(frame["mean_intensity"])
                aligned_rewards.append(trajectory_data[frame["step"]]["reward"])
        
        if aligned_intensities:
            plt.scatter(aligned_intensities, aligned_rewards, alpha=0.6)
            plt.title('Frame Intensity vs Reward Correlation')
            plt.xlabel('Frame Mean Intensity')
            plt.ylabel('Reward')
        
        # Subplot 6: Performance distribution
        plt.subplot(2, 3, 6)
        plt.hist(step_times, bins=50, alpha=0.7, label='Step Times')
        plt.axvline(x=10, color='r', linestyle='--', label='10ms target')
        plt.title('Step Time Distribution')
        plt.xlabel('Step Time (ms)')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return analysis_results, trajectory_data, frame_analysis

# Run comprehensive analysis
results, trajectory, frames = analyze_plume_navigation_behavior()
```

### Jupyter Notebook Integration

```python
# notebook_example.ipynb
from hydra import compose, initialize
from plume_nav_sim import Navigator, VideoPlume
from plume_nav_sim.utils import set_global_seed

# Setup reproducible environment
set_global_seed(42)

# Load configuration in notebook
with initialize(config_path="../conf", version_base=None):
    cfg = compose(config_name="config", overrides=[
        "visualization.animation.enabled=true",
        "visualization.plotting.figure_size=[14,10]",
        "environment.frame_cache.mode=lru",
        "environment.include_video_frame=true"
    ])

# Create and run simulation with frame caching
navigator = Navigator.from_config(cfg.navigator)
video_plume = VideoPlume.from_config(cfg.video_plume)

# Interactive visualization with performance monitoring
results = navigator.simulate(video_plume, duration=60)
results.plot_trajectory(interactive=True)

# Access performance statistics
print(f"Average step time: {results.avg_step_time_ms:.2f}ms")
print(f"Cache hit rate: {results.cache_hit_rate:.2%}")
```

### Kedro Pipeline Integration

```python
# kedro_pipeline_example.py
from kedro.pipeline import Pipeline, node
from plume_nav_sim.api import create_navigator, run_plume_simulation

def create_navigation_pipeline(**kwargs) -> Pipeline:
    """Create Kedro pipeline for plume navigation simulation."""
    
    return Pipeline([
        node(
            func=create_navigator,
            inputs=["params:navigator_config"],
            outputs="navigator",
            name="create_navigator_node"
        ),
        node(
            func=run_plume_simulation,
            inputs=["navigator", "video_plume", "params:simulation_config"],
            outputs="simulation_results",
            name="run_simulation_node"
        ),
        node(
            func=analyze_trajectory,
            inputs=["simulation_results"],
            outputs="trajectory_analysis",
            name="analyze_trajectory_node"
        )
    ])
```

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository** and create a feature branch
2. **Setup development environment**: `make setup-dev`
3. **Make changes** with appropriate tests
4. **Run quality checks**: `make test-all`
5. **Submit pull request** with clear description

### Development Standards

- **Code Style**: Black formatting, isort imports, flake8 compliance
- **Type Safety**: MyPy static type checking required
- **Test Coverage**: Minimum 80% coverage for new code
- **Documentation**: Docstrings for all public APIs
- **Commit Messages**: Conventional commits format

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{plume_navigation_simulation_library,
  title={Plume Navigation Simulation Library},
  author={Samuel Brudner},
  year={2024},
  url={https://github.com/organization/plume_nav_sim},
  version={1.0.0},
  note={Protocol-based component architecture with zero-code extensibility for comprehensive plume navigation research}
}
```

## Support and Documentation

- **Documentation**: [https://plume-nav-sim.readthedocs.io](https://plume-nav-sim.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/organization/plume_nav_sim/issues)
- **Discussions**: [GitHub Discussions](https://github.com/organization/plume_nav_sim/discussions)
- **API Reference**: Generated automatically from docstrings

## Changelog

### Version 1.0.0 (Protocol-Based Component Architecture and Zero-Code Extensibility Release)

#### Revolutionary v1.0 Architecture Transformation

**Complete Protocol-Based Component System**
- **SourceProtocol Implementation**: Pluggable odor source modeling enabling PointSource, MultiSource, and DynamicSource configurations without code modification
- **BoundaryPolicyProtocol Framework**: Configurable domain boundary handling with terminate, bounce, wrap, and clip policies for diverse experimental scenarios  
- **ActionInterfaceProtocol Abstraction**: Unified action processing supporting Continuous2D and CardinalDiscrete implementations with runtime selection
- **RecorderProtocol Infrastructure**: Comprehensive data persistence with parquet, HDF5, SQLite, and none backends achieving <1ms disabled-mode overhead
- **StatsAggregatorProtocol Integration**: Automated research metrics calculation with episode-level and run-level aggregation plus standardized summary.json export
- **AgentInitializerProtocol Flexibility**: Configurable agent starting positions with uniform_random, grid, fixed_list, and from_dataset strategies

**Zero-Code Extensibility Achievement**
- **Component Plugin System**: Add new sources, boundary policies, action interfaces, recorders, and analyzers through protocol implementation without core library modification
- **Configuration-Driven Selection**: Runtime component swapping through Hydra configuration with full type safety and validation
- **Research Workflow Integration**: Seamless integration with existing research pipelines while enabling advanced capabilities through optional dependency groups

**Comprehensive Recording and Analysis Framework**
- **Multi-Backend Recording**: Performance-optimized data persistence supporting parquet (columnar), HDF5 (scientific), SQLite (transactional), and none (disabled) backends
- **Automated Statistics Collection**: Built-in research metrics calculation with configurable aggregation levels and export capabilities
- **Performance Monitoring**: Real-time performance tracking ensuring ≤33ms step latency with 100 concurrent agents across all v1.0 components
- **Research Reproducibility**: Standardized output formats, correlation tracking, and comprehensive metadata collection for cross-project comparison

**Interactive Debugging and Visualization**
- **Dual-Backend Debug GUI**: PySide6 desktop and Streamlit web interfaces with real-time visualization, step-through debugging, and performance profiling
- **Collaborative Debugging**: Shared debugging sessions with correlation tracking and state synchronization across multiple researchers
- **Advanced Debugging Features**: Conditional breakpoints, state inspection, performance monitoring, screenshot export, and session management
- **Zero-Overhead Design**: Debug utilities achieve zero performance impact when disabled while maintaining full functionality when enabled

#### v1.0 Performance Enhancements

**Enhanced Performance Standards**
- **Multi-Agent Performance**: ≤33ms step latency with 100 concurrent agents (enhanced from v0.3.0's <10ms single-agent target)
- **Recording Efficiency**: <1ms overhead when disabled across all recorder backends with configurable buffering and compression
- **Component Integration**: All protocol-based components maintain performance targets while adding comprehensive functionality
- **Memory Optimization**: Linear scaling with agent count plus configurable memory limits and efficient buffering strategies

**Production-Ready Capabilities**
- **Enterprise Integration**: Full Hydra configuration management with environment variable support and structured validation
- **Docker Deployment**: Containerized environments with optional dependency management and multi-stage builds
- **CLI Enhancement**: Command-line tools supporting all v1.0 components with parameter overrides and batch processing
- **Installation Flexibility**: Optional dependency groups (recording, debug, analysis) enabling lightweight or full-featured installations

### Version 0.4.0 (Modular Architecture and Agent-Agnostic Design Release)

#### Revolutionary Architecture Transformation

**Protocol-Driven Plugin System**
- **Modular Component Architecture**: Complete transformation from rigid VideoPlume-only system to flexible protocol-based plugin architecture
- **Zero-Code Extensibility**: Add new plume models, wind fields, and sensors through protocol implementation without core system modification
- **Runtime Component Swapping**: Switch between GaussianPlumeModel, TurbulentPlumeModel, and FilamentBasedPlumeModel purely through Hydra configuration
- **Research Flexibility**: Enable diverse experimental scenarios through component composition rather than code modification

**Agent-Agnostic Navigation System**
- **Memory Configuration Toggle**: Support both memory-based and memory-less navigation strategies through `navigator.memory_enabled` flag
- **Strategy Abstraction**: Configurable navigation strategies (reactive, gradient_following, infotaxis, casting) through protocol implementation
- **Cognitive Modeling Support**: Optional memory hooks enable complex cognitive modeling without affecting baseline performance
- **Fair Comparison Framework**: Identical environmental conditions for memory vs memory-less agent comparison studies

**Environmental Dynamics Integration**
- **Wind Field System**: Configurable wind dynamics with ConstantWindField, TurbulentWindField, and TimeVaryingWindField implementations
- **Multi-Modal Sensing**: Flexible sensor abstraction with BinarySensor, ConcentrationSensor, and GradientSensor implementations
- **Physics Integration**: Realistic environmental coupling between wind fields and plume transport dynamics
- **Temporal Dynamics**: Support for time-varying environmental conditions and correlation modeling

#### New Plume Modeling Capabilities

**GaussianPlumeModel - Mathematical Precision**
- **Analytical Solutions**: Sub-millisecond computation time with mathematical precision for rapid prototyping
- **Wind Integration**: Automatic wind field coupling for realistic transport modeling
- **Configurable Parameters**: Source strength, diffusion coefficients, and decay rates through configuration
- **Performance Optimization**: Gradient caching and analytical solutions for real-time applications

**TurbulentPlumeModel - Realistic Physics**
- **Filament-Based Simulation**: Individual odor packets with age, concentration, and transport dynamics
- **Stochastic Dispersion**: Lagrangian transport modeling with turbulent wind field integration
- **Biological Accuracy**: Intermittent, patchy odor signals matching real-world plume observations
- **Numba Acceleration**: Optional JIT compilation for computationally intensive scenarios

**FilamentBasedPlumeModel - Research Fidelity**
- **Intermittent Signals**: Sparse, realistic odor detection patterns observed in natural environments
- **Temporal Correlation**: Realistic plume evolution with proper temporal coherence modeling
- **Biological Validation**: Designed for biologically-inspired navigation algorithm validation
- **Scientific Accuracy**: Publication-quality environmental modeling for research applications

**VideoPlumeAdapter - Backward Compatibility**
- **Legacy Preservation**: Complete backward compatibility with existing video-based workflows
- **Performance Maintained**: All frame caching optimizations preserved through adapter pattern
- **Seamless Migration**: Existing VideoPlume code continues working with deprecation warnings
- **Configuration Integration**: Video models integrated into modular configuration system

#### Configuration-Driven Research Workflows

**Experimental Configuration Framework**
- **Memory Comparison Studies**: Pre-configured experiment templates for memory vs memory-less navigation comparison
- **Plume Model Validation**: Systematic comparison frameworks for different plume modeling approaches
- **Environmental Complexity Studies**: Configurable complexity levels from simple mathematical models to realistic physics
- **Multi-Condition Experiments**: Support for complex experimental designs through configuration composition

**Component Selection Examples**
```yaml
# Mathematical modeling for fast experimentation
plume_model: gaussian
wind_field: constant
navigator: {memory_enabled: false, strategy: gradient_following}

# Realistic physics for publication research  
plume_model: turbulent
wind_field: turbulent
navigator: {memory_enabled: true, strategy: infotaxis}
```

**Research Integration Patterns**
- **Jupyter Notebook Support**: Direct access to modular components for interactive research
- **Command-Line Flexibility**: Complete experiment configuration through CLI parameter overrides
- **Kedro Pipeline Integration**: Modular components designed for complex research pipeline integration
- **Version Control Friendly**: Configuration-based experiments enable reproducible research workflows

#### Performance and Compatibility Guarantees

**Performance Standards Maintained**
- **Step Latency**: <10ms execution time preserved across all plume model implementations
- **Memory Efficiency**: Linear scaling with agent count for all modular components
- **Deterministic Behavior**: Identical results across runs with fixed random seeds for all models
- **Cache Optimization**: Frame caching benefits preserved for video-based models through adapter pattern

**Backward Compatibility Preservation**
- **API Compatibility**: All existing NavigatorProtocol methods maintained with enhanced extensibility hooks
- **Configuration Migration**: Automatic conversion from legacy configuration formats to modular system
- **Video Workflow Support**: Existing video-based research workflows preserved through VideoPlumeAdapter
- **Performance Preservation**: No performance regression for existing use cases

**Migration Support**
- **Gradual Adoption**: Modular components can be adopted incrementally without breaking existing code
- **Legacy Deprecation**: Clear deprecation timeline with comprehensive migration documentation
- **Tool Support**: Configuration conversion utilities for seamless migration to modular architecture
- **Documentation Coverage**: Complete migration guides for all common usage patterns

#### Developer Experience Enhancements

**Enhanced Documentation**
- **Modular Architecture Guide**: Comprehensive documentation of plugin system and component protocols
- **Configuration Examples**: Extensive examples for all component combinations and research scenarios
- **Migration Documentation**: Step-by-step guides for adopting modular architecture
- **Protocol Reference**: Complete API documentation for extending the system with custom components

**Research Workflow Integration**
- **Example Implementations**: Reference implementations for common research scenarios and algorithm types
- **Comparative Studies**: Pre-configured experimental frameworks for systematic research comparison
- **Performance Benchmarking**: Standardized benchmarks for comparing different plume model approaches
- **Extensibility Examples**: Clear examples of implementing custom plume models, sensors, and navigation strategies

### Version 0.3.0 (Gymnasium Migration and Performance Enhancement Release)

#### Major Features

**High-Performance Frame Caching System**
- **Dual-Mode Caching**: LRU and full-preload modes for optimal memory/performance trade-offs
- **Sub-10ms Performance**: Achieve <10ms environment step times with intelligent frame caching
- **Memory Management**: Configurable 2 GiB default limit with automatic pressure handling
- **Cache Analytics**: Comprehensive hit/miss statistics and memory usage monitoring
- **Thread-Safe Operations**: Concurrent access support for multi-agent simulation scenarios

**Enhanced Structured Logging with Performance Integration**
- **Loguru-Based Architecture**: JSON-structured logs with correlation ID tracking for distributed debugging
- **Performance Metrics Embedding**: Automatic cache statistics and step timing data in `info["perf_stats"]`
- **Multi-Sink Configuration**: Flexible output to console, JSON files, and external monitoring systems
- **Machine-Parseable Output**: Standardized JSON format for automated performance analysis and alerting

**CLI Frame Cache Integration**
- **Cache Mode Selection**: `--frame-cache {none,lru,all}` parameter for runtime cache configuration
- **Memory Configuration**: Runtime cache size adjustment via `environment.frame_cache.cache_size_mb`
- **Performance Monitoring**: Built-in cache performance reporting in CLI output
- **Environment Variable Support**: `FRAME_CACHE_MODE`, `FRAME_CACHE_SIZE_MB` for deployment flexibility

#### Performance Enhancements

**Training Speed Optimization**
- **RL Training Acceleration**: 3-5x faster training loops through frame cache hit rates >90%
- **Memory Efficiency**: Linear memory scaling with configurable limits preventing OOM conditions
- **Zero-Copy Frame Access**: NumPy array views for optimal memory usage in video processing
- **Batch Preloading**: Intelligent frame warming strategies for sequential access patterns

**Analytics and Monitoring**
- **Real-Time Metrics**: Live cache performance and step timing embedded in simulation info
- **Structured Diagnostics**: JSON logs with correlation IDs for experiment traceability
- **Performance Regression Detection**: Automated threshold monitoring with configurable alerts
- **Resource Usage Tracking**: Memory pressure monitoring with graceful degradation strategies

#### Developer Experience Improvements

**Enhanced Documentation**
- **Frame Caching Guide**: Comprehensive usage examples for all cache modes and configurations
- **Performance Tuning**: Memory sizing guidelines and optimization strategies for different scenarios
- **Analytics Integration**: Examples for accessing `info["perf_stats"]` and `info["video_frame"]` in research workflows
- **Logging Configuration**: Production-ready logging setups with rotation, retention, and sink management

**Configuration Management**
- **Cache Configuration Groups**: Hydra-based cache profiles (LRU, preload, disabled) for easy switching
- **Environment Variable Integration**: Runtime configuration via `FRAME_CACHE_*` environment variables
- **Validation and Error Handling**: Comprehensive cache parameter validation with clear error messages
- **Migration Support**: Seamless integration with existing configurations without breaking changes

#### Integration and Compatibility

**RL Framework Integration**
- **Stable-Baselines3 Optimization**: Enhanced performance metrics for all supported algorithms (PPO, SAC, TD3, A2C, DDPG)
- **Vectorized Environment Support**: Cache sharing optimization for parallel training environments
- **Training Pipeline Integration**: Automatic cache statistics logging during training with TensorBoard integration
- **Checkpoint Correlation**: Cache performance tracking correlated with model checkpoints for analysis

**Research Workflow Support**
- **Notebook Integration**: Enhanced `info["video_frame"]` access for Jupyter-based analysis workflows
- **Experiment Tracking**: Correlation ID integration with MLflow, Weights & Biases, and custom tracking systems
- **Reproducibility**: Deterministic cache warming and statistics for consistent experimental results
- **Performance Benchmarking**: Standardized performance metrics for algorithm and configuration comparison

### Version 0.2.0 (API Consistency and Integration Hardening Release)

#### Core Refactoring and Modernization

- **Gymnasium 0.29.x API Compliance**: Full upgrade from legacy gym to modern Gymnasium with pinned 0.29.x dependency for stability
- **Dual API Support**: Automatic detection and compatibility layer supporting both 4-tuple legacy gym and 5-tuple Gymnasium APIs without breaking changes
- **New Environment ID**: Introduced `PlumeNavSim-v0` environment alongside existing `OdorPlumeNavigation-v1` for enhanced features
- **Backward Compatibility**: Zero breaking changes - all existing gym-based code continues to work unchanged

#### Structured Configuration Revolution

- **Dataclass-Based Configuration**: Complete migration from unstructured YAML to Pydantic-validated dataclass configuration
- **Type Safety**: Full type validation with IDE autocomplete and mypy support throughout configuration system
- **Runtime Validation**: Configuration errors caught at startup rather than runtime, improving reliability
- **Hydra 1.3+ Integration**: Enhanced structured config support with ConfigStore registration and automatic schema validation

#### Centralized Logging Architecture

- **Loguru Integration**: Replaced ad-hoc print statements and basic logging with structured JSON logging system
- **Correlation ID Tracking**: Cross-component request correlation for distributed debugging and analysis
- **Performance Monitoring**: Integrated logging with performance threshold monitoring and automatic alerting
- **Structured Output**: JSON-formatted logs with configurable sinks for development and production environments

#### Enhanced Developer Experience

- **Migration Guides**: Comprehensive documentation for gym→gymnasium and unstructured→structured config transitions
- **Configuration Examples**: Extensive examples of dataclass configuration patterns for all components
- **Logging Configuration**: Detailed examples of Loguru setup for development and production environments
- **API Documentation**: Updated examples showing dual API support and best practices

#### Performance and Quality Improvements

- **Test Coverage Enhancement**: Expanded test coverage targeting ≥70% overall, ≥80% for new code
- **Cross-Repository Integration**: CI verification against `place_mem_rl` main branch and v0.2.0 tag
- **Performance Preservation**: Maintained ≤10ms average step() time on Intel i7-9700K single thread
- **Memory Efficiency**: Optimized memory usage patterns for large-scale multi-agent scenarios

#### Integration and Compatibility

- **stable-baselines3 Compatibility**: Seamless integration with latest stable-baselines3 versions
- **Gymnasium Ecosystem**: Full compatibility with gymnasium wrappers and utilities
- **Legacy Support**: Comprehensive backward compatibility layer with deprecation warnings
- **Environment Registration**: Enhanced registration system supporting both legacy and modern environment IDs

#### Documentation and Examples

- **Migration Documentation**: Step-by-step guides for API and configuration transitions
- **Structured Config Examples**: Comprehensive examples of dataclass-based configuration patterns
- **Logging Examples**: Detailed Loguru configuration and usage patterns
- **Best Practices**: Updated recommendations for modern development workflows

### Version 0.1.0 (Initial Release)

- **Library Architecture**: Transformed from standalone application to importable library
- **Hydra Configuration**: Sophisticated hierarchical configuration management
- **CLI Interface**: Comprehensive command-line tools with Click framework
- **Multi-Framework Support**: Integration patterns for Kedro, RL, and ML workflows
- **Docker Support**: Containerized development and deployment environments
- **Setuptools Build System**: Modern PEP 621-compliant packaging with setuptools>=61
- **Enhanced Documentation**: Comprehensive usage examples and migration guides

For detailed changes, see [CHANGELOG.md](CHANGELOG.md).