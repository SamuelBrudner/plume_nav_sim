# plume-nav-sim

🧪 Proof-of-Life Gymnasium Environment for Plume Navigation Research

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![Gymnasium](https://img.shields.io/badge/gymnasium-0.29%2B-green.svg)](https://gymnasium.farama.org/) [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A minimal Gymnasium-compatible reinforcement learning environment for plume navigation research, providing a standard API to communicate between learning algorithms and chemical plume environments. Designed specifically for researchers, educators, and students developing autonomous agents that navigate chemical plumes to locate their sources.

## 🚀 Key Features

- **🎯 Gymnasium API Compliance**: Full compatibility with standard reinforcement learning frameworks
- **📊 Static Gaussian Plume Model**: Mathematically defined concentration field for reproducible research
- **🖥️ Dual Rendering Modes**: Both programmatic (`rgb_array`) and interactive (`human`) visualization
- **🔢 Deterministic Reproducibility**: Comprehensive seeding system for scientific validity
- **🧪 Educational Focus**: Designed for learning plume navigation concepts and RL development

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Features](#features)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License & Citation](#license--citation)

## 🛠️ Installation

### Requirements

- **Python 3.10+** (Required for Gymnasium compatibility)
- **Gymnasium ≥0.29.0** (Core RL environment framework)
- **NumPy ≥2.1.0** (Mathematical computing foundation)
- **Matplotlib ≥3.9.0** (Visualization and rendering)
- Optional for notebooks: **ipympl ≥0.9** (Interactive Matplotlib widgets in Jupyter)
  - Also install: **ipywidgets ≥8.0.0** (widget support)

### Installation Steps

1. **Create Virtual Environment** (Recommended):

   ```bash
   python -m venv plume-nav-env
   source plume-nav-env/bin/activate  # Linux/macOS
   plume-nav-env\Scripts\activate     # Windows
   ```

2. **Install Package**:

   ```bash
   # Development installation (recommended for research)
   pip install -e .
   
   # With development dependencies
   pip install -e .[dev]

   # For Jupyter notebooks with interactive plots (includes ipympl)
   pip install -e .[notebooks]

   # If '%matplotlib widget' is not recognized, install runtime deps to the kernel
   # pip
   %pip install -U ipympl ipywidgets matplotlib ipykernel
   # conda
   # conda install -c conda-forge ipympl ipywidgets matplotlib ipykernel
   # Then restart the kernel
   ```

3. **Verify Installation**:

   ```bash
   python -c "import plume_nav_sim; print(f'Version: {plume_nav_sim.get_version()}')"
   ```

## ⚡ Quick Start

### Basic Usage

```python
import gymnasium as gym
import plume_nav_sim

# Register environment with Gymnasium
plume_nav_sim.register_env()

# Create environment instance
env = gym.make('PlumeNav-StaticGaussian-v0')

# Basic episode execution
obs, info = env.reset(seed=42)
for step in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print(f"Episode completed in {step + 1} steps")
        break

env.close()
```

### Quick Start with Convenience Function

```python
import plume_nav_sim

# Streamlined environment setup
env = plume_nav_sim.quick_start()
obs, info = env.reset()

# Run single step
action = 1  # Move right
obs, reward, terminated, truncated, info = env.step(action)
print(f"Reward: {reward}, Position: {info['agent_position']}")

env.close()
```

## 📖 API Reference

### Environment Class: `PlumeSearchEnv`

The core Gymnasium environment for plume navigation simulation.

#### Action Space

- **Type**: `Discrete(4)`
- **Actions**:
  - `0`: Move Up
  - `1`: Move Right  
  - `2`: Move Down
  - `3`: Move Left

#### Observation Space

- **Type**: `Box(low=0.0, high=1.0, shape=(1,), dtype=float32)`
- **Description**: Concentration value at agent's current position

#### Key Methods

```python
# Episode initialization
observation, info = env.reset(seed=None)

# Environment step
observation, reward, terminated, truncated, info = env.step(action)

# Rendering
rgb_array = env.render(mode="rgb_array")  # Returns np.ndarray
env.render(mode="human")                  # Interactive display

# Resource cleanup
env.close()
```

#### Configuration Parameters

```python
env_config = {
    'grid_size': (128, 128),          # Grid dimensions (width, height)
    'source_location': (64, 64),      # Source position (x, y)
    'max_steps': 1000,                # Maximum episode length
    'goal_radius': 0.0,               # Success radius around source
    'sigma': 12.0                     # Plume dispersion parameter
}

### Component-Based Environment (Dependency Injection)

For advanced use, you can assemble an environment from swappable components (actions, observations, rewards, plume model). See `plume_nav_sim/envs/component_env.py`.

Two ways to use components:

- Direct factory assembly:
  ```python
  from plume_nav_sim.envs.factory import create_component_environment

  env = create_component_environment(
      grid_size=(128, 128),
      goal_location=(64, 64),
      action_type='oriented',           # 'discrete' or 'oriented'
      observation_type='antennae',      # 'concentration' or 'antennae'
      reward_type='step_penalty',       # 'sparse' or 'step_penalty'
      goal_radius=2.0,
  )
  obs, info = env.reset(seed=42)
  ```

- Register the component-based environment id (first‑class DI env):

  ```python
  import gymnasium as gym
  from plume_nav_sim.registration import register_env, COMPONENT_ENV_ID

  # Register DI env id (factory-backed)
  env_id = register_env(env_id=COMPONENT_ENV_ID, force_reregister=True)
  env = gym.make(env_id)
  ```

  Or opt-in globally without changing code by setting an environment variable before registering:

  ```bash
  # Use DI behind the default env id for this process
  export PLUMENAV_DEFAULT=components
  ```

  ```python
  from plume_nav_sim.registration import register_env, ENV_ID
  env_id = register_env()  # Will use the DI entry point when PLUMENAV_DEFAULT is set
  ```

Components derive spaces:

- `action_space` comes from the ActionProcessor
- `observation_space` comes from the ObservationModel

See `plume_nav_sim/config/factories.py` for config-driven creation (Hydra/YAML).

```

### Registration System

```python
import plume_nav_sim

# Register with default parameters
plume_nav_sim.register_env()

# Register with custom configuration
plume_nav_sim.register_env(kwargs=custom_config)

# Check registration status
if plume_nav_sim.is_registered('PlumeNav-StaticGaussian-v0'):
    print("Environment registered successfully")
```

## 💡 Examples

### Basic Navigation

```python
import gymnasium as gym
import plume_nav_sim

# Create tutorial environment
env, config_info = plume_nav_sim.create_example_environment('tutorial')
print(f"Environment: {config_info['description']}")

obs, info = env.reset(seed=42)
total_reward = 0

for step in range(500):
    # Simple gradient-following strategy
    current_concentration = obs[0]
    
    # Try all actions and pick the one with highest concentration
    best_action = 0
    best_concentration = current_concentration
    
    for test_action in range(4):
        # This is a simplified example - in practice, you'd need
        # to implement proper exploration strategies
        test_obs, _, _, _, _ = env.step(test_action)
        if test_obs[0] > best_concentration:
            best_concentration = test_obs[0]
            best_action = test_action
        # Reset to previous state (simplified)
        env.reset(seed=42)
        for _ in range(step):
            env.step(env.action_space.sample())
    
    obs, reward, terminated, truncated, info = env.step(best_action)
    total_reward += reward
    
    if terminated:
        print(f"🎉 Goal reached in {step + 1} steps!")
        break
    elif truncated:
        print(f"Episode truncated after {step + 1} steps")
        break

print(f"Total reward: {total_reward}")
env.close()
```

### Visualization Demo

```python
import plume_nav_sim
import matplotlib.pyplot as plt

# Create environment with visualization enabled
env = plume_nav_sim.quick_start(
    env_config={'grid_size': (64, 64)},
    auto_register=True
)

# Run episode with human rendering
obs, info = env.reset(seed=123)
env.render(mode="human")

for step in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Render every 5 steps for visualization
    if step % 5 == 0:
        env.render(mode="human")
        plt.pause(0.1)  # Brief pause for visualization
    
    if terminated or truncated:
        break

env.close()
```

### Reproducibility Demo

```python
import plume_nav_sim

def run_deterministic_episode(seed=42):
    env = plume_nav_sim.quick_start()
    obs, info = env.reset(seed=seed)
    
    trajectory = [info['agent_position']]
    for _ in range(10):
        action = 1  # Always move right
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append(info['agent_position'])
        
        if terminated or truncated:
            break
    
    env.close()
    return trajectory

# Demonstrate reproducibility
traj1 = run_deterministic_episode(seed=42)
traj2 = run_deterministic_episode(seed=42)

print(f"Trajectory 1: {traj1}")
print(f"Trajectory 2: {traj2}")
print(f"Identical trajectories: {traj1 == traj2}")
```

## 🔧 Features

### Gymnasium API Compliance

Full compatibility with the Gymnasium reinforcement learning framework:

- **Standard Interface**: Implements `reset()`, `step()`, `render()`, and `close()` methods
- **5-Tuple Returns**: Proper handling of observation, reward, terminated, truncated, and info
- **Space Definitions**: Correctly defined action and observation spaces
- **Seeding Support**: Deterministic episode generation with seed parameter

### Static Gaussian Plume Model

Mathematical implementation of chemical plume distribution:

- **Gaussian Formula**: `C(x,y) = exp(-((x-sx)² + (y-sy)²) / (2*σ²))`
- **Configurable Parameters**: Source location and dispersion settings
- **Normalized Values**: Concentration values in range [0,1] with peak at source
- **Efficient Sampling**: O(1) concentration lookup for agent positions

### Dual-Mode Rendering

Comprehensive visualization support for different use cases:

- **RGB Array Mode**: Returns NumPy arrays for programmatic analysis
  - Shape: `(height, width, 3)` with uint8 values
  - Agent visualization: Red 3×3 square
  - Source visualization: White 5×5 cross
  - Grayscale concentration heatmap

- **Human Mode**: Interactive matplotlib visualization
  - Real-time agent position updates
  - Concentration colormap display
  - Headless compatibility with Agg backend
  - Graceful fallback to RGB array mode

### Reproducible Research Support

Deterministic episode generation for scientific validity:

- **Comprehensive Seeding**: Uses `gymnasium.utils.seeding.np_random`
- **Identical Episodes**: Same seed produces identical behavior
- **Statistical Independence**: Different seeds ensure proper randomization
- **Research Workflows**: Compatible with experimental design standards

## ⚙️ Configuration

### Environment Parameters

```python
# Complete configuration options
config = {
    # Grid Configuration
    'grid_size': (128, 128),          # Environment dimensions (width, height)
    
    # Plume Configuration  
    'source_location': (64, 64),      # Source position (x, y)
    'sigma': 12.0,                    # Gaussian dispersion parameter
    
    # Episode Configuration
    'max_steps': 1000,                # Maximum steps before truncation
    'goal_radius': 0.0,               # Success distance from source
    
    # Agent Configuration
    'agent_start_position': None,     # None for random start, or (x, y)
}
```

### Performance Tuning

```python
# Optimized for different use cases
performance_configs = {
    'fast_training': {
        'grid_size': (64, 64),        # Smaller grid for speed
        'max_steps': 500,             # Shorter episodes
    },
    'detailed_analysis': {
        'grid_size': (256, 256),      # Larger grid for precision
        'max_steps': 5000,            # Longer episodes
        'sigma': 20.0,                # Wider plume spread
    },
    'memory_efficient': {
        'grid_size': (32, 32),        # Minimal memory footprint
        'max_steps': 200,
    }
}

# Apply configuration
env = plume_nav_sim.quick_start(env_config=performance_configs['fast_training'])
```

### Custom Environment Creation

```python
import plume_nav_sim

# Factory function for custom environments
def create_custom_environment(difficulty='medium'):
    difficulty_configs = {
        'easy': {
            'grid_size': (32, 32),
            'source_location': (16, 16),
            'sigma': 6.0,
            'goal_radius': 2.0,
        },
        'medium': {
            'grid_size': (128, 128),
            'source_location': (64, 64), 
            'sigma': 12.0,
            'goal_radius': 1.0,
        },
        'hard': {
            'grid_size': (256, 256),
            'source_location': (200, 200),
            'sigma': 8.0,
            'goal_radius': 0.0,
        }
    }
    
    config = difficulty_configs.get(difficulty, difficulty_configs['medium'])
    return plume_nav_sim.create_plume_search_env(**config)

# Usage
env = create_custom_environment('hard')
```

## 🔍 Troubleshooting

### Common Installation Issues

**Problem: ImportError for gymnasium**

```bash
# Solution: Install compatible version
pip install gymnasium>=0.29.0
```

**Problem: NumPy version conflicts**

```bash
# Solution: Upgrade to compatible version
pip install numpy>=2.1.0
```

**Problem: Python version compatibility**

```bash
# Check Python version
python --version

# Upgrade if needed (Python 3.10+ required)
# Use pyenv or conda to manage Python versions
```

### Matplotlib Backend Issues

**Problem: No display available (headless systems)**

```python
import matplotlib
matplotlib.use('Agg')  # Set backend before importing plume_nav_sim

import plume_nav_sim
env = plume_nav_sim.quick_start()
```

**Problem: Interactive rendering not working**

```python
# Check available backends
import matplotlib
print(matplotlib.get_backend())

# Set interactive backend
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

**Notebook interactivity (Jupyter)**

```bash
# Ensure notebook extras (installs ipympl)
pip install -e .[notebooks]
```

In a Jupyter notebook cell, enable the widget backend before plotting:

```python
%matplotlib widget
```

If you see “'widget' is not a recognised backend name”:
- Install ipympl and ipywidgets in the active kernel and restart it:
  - `%pip install -U ipympl ipywidgets`
- Clear forced backends: `import os; os.environ.pop('MPLBACKEND', None)`
- Classic Notebook: `jupyter nbextension enable --py widgetsnbextension`

### Performance Optimization

**Problem: Slow step execution**

```python
# Use smaller grid sizes for faster performance
config = {
    'grid_size': (64, 64),    # Instead of (128, 128)
    'max_steps': 500,         # Shorter episodes
}
env = plume_nav_sim.quick_start(env_config=config)
```

**Problem: Memory usage too high**

```python
# Monitor memory usage
import psutil
import os

process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"Memory usage: {memory_mb:.1f} MB")

# Use memory-efficient configuration
config = {
    'grid_size': (32, 32),    # Minimal grid
}
```

### Common Runtime Errors

**Problem: Environment not registered**

```python
# Ensure registration before gym.make()
import plume_nav_sim
plume_nav_sim.register_env()

import gymnasium as gym
env = gym.make('PlumeNav-StaticGaussian-v0')
```

**Problem: Inconsistent reproducibility**

```python
# Proper seeding approach
env = plume_nav_sim.quick_start()

# Always pass seed to reset()
obs, info = env.reset(seed=42)

# Don't call env.seed() - use reset(seed=) instead
```

## 🛠️ Development

### Development Setup

1. **Clone Repository**:

   ```bash
   git clone <repository-url>
   cd plume-nav-sim
   ```

2. **Create Development Environment**:

   ```bash
   python -m venv dev-env
   source dev-env/bin/activate
   ```

3. **Install in Development Mode**:

   ```bash
   pip install -e .[dev]
   ```

4. **Verify Development Setup**:

   ```bash
   python -c "import plume_nav_sim; print('Development setup successful')"
   ```

### Running Tests

```bash
# Run complete test suite
pytest

# Run with coverage report
pytest --cov=plume_nav_sim --cov-report=html

# Run specific test categories
pytest tests/test_environment.py -v
pytest tests/test_rendering.py -v
pytest tests/test_reproducibility.py -v
```

### Code Quality

```bash
# Format code (if using black)
black src/

# Type checking (if using mypy)  
mypy src/plume_nav_sim/

# Linting (if using flake8)
flake8 src/plume_nav_sim/
```

### Testing Your Changes

```bash
# Quick functionality test
python examples/basic_usage.py

# Rendering test
python examples/visualization_demo.py

# Performance test
python examples/performance_benchmark.py
```

### Contribution Guidelines

1. **Code Standards**: Follow PEP 8 style guidelines
2. **Testing**: Add tests for new functionality
3. **Documentation**: Update docstrings and README
4. **Backwards Compatibility**: Maintain API compatibility
5. **Performance**: Ensure changes don't degrade performance

### Development Workflow

1. **Feature Branch**: Create branch from main
2. **Development**: Implement changes with tests
3. **Validation**: Run full test suite
4. **Documentation**: Update relevant documentation
5. **Review**: Submit pull request for review

## 📄 License & Citation

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 plume_nav_sim Development Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

### Citation

If you use plume-nav-sim in your research, please cite:

```bibtex
@software{plume_nav_sim2024,
  title={plume-nav-sim: Gymnasium Environment for Plume Navigation Research},
  author={plume_nav_sim Development Team},
  year={2024},
  url={https://github.com/your-org/plume-nav-sim},
  version={0.0.1}
}
```

### Acknowledgments

This project builds upon the excellent work of the scientific Python ecosystem:

- **Gymnasium**: [Farama Foundation](https://gymnasium.farama.org/) for the RL environment framework
- **NumPy**: [NumPy Community](https://numpy.org/) for mathematical computing foundations  
- **Matplotlib**: [Matplotlib Development Team](https://matplotlib.org/) for visualization capabilities
- **Python**: [Python Software Foundation](https://python.org/) for the core language

### Contact & Support

- **Documentation**: [Link to full documentation]
- **Issues**: [GitHub Issues](https://github.com/your-org/plume-nav-sim/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/plume-nav-sim/discussions)
- **Email**: <plume-nav-sim@example.com>

---

## 📑 Operational Logging vs Data Capture

- Use loguru for operational, human-readable logs (console/file). Configure it independently of data capture.

```python
from plume_nav_sim.logging.loguru_bootstrap import setup_logging
setup_logging(level="INFO", console=True, file_path="run.log", rotation="10 MB", retention="7 days")
```

- Use the data capture pipeline for analysis-ready data (validated JSONL.gz, optional Parquet export).

Quick start (data capture):

```python
from plume_nav_sim.data_capture import RunRecorder
from plume_nav_sim.data_capture.wrapper import DataCaptureWrapper
from plume_nav_sim.core.types import EnvironmentConfig

env = plume_nav_sim.make_env(action_type="oriented", observation_type="concentration")
rec = RunRecorder("results", experiment="demo")
cfg = EnvironmentConfig()
env = DataCaptureWrapper(env, rec, cfg)

obs, info = env.reset(seed=123)
for _ in range(100):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc:
        break
rec.finalize(export_parquet=False)
```

### Data Capture Dependencies

- Install optional extras for analysis workflows:

```bash
pip install -e .[data]
```

Includes:
- orjson: fast JSON serialization for JSONL.gz
- pandas + pandera: DataFrame operations and batch validation
- pyarrow: Parquet export (optional)

Operational logging extras (separate):

```bash
pip install -e .[ops]
```

Parquet export and Pandera validation require the `[data]` extra; JSONL.gz capture works without it.

### Validation and Parquet Examples

- Validate a run’s artifacts:

```python
from pathlib import Path
from plume_nav_sim.data_capture.validate import validate_run_artifacts

run_dir = Path("results/demo/<run_id>")
report = validate_run_artifacts(run_dir)
print(report)
```

- Load JSONL.gz and export Parquet:

```python
import pandas as pd

steps_df = pd.read_json(run_dir / "steps.jsonl.gz", lines=True, compression="gzip")
episodes_df = pd.read_json(run_dir / "episodes.jsonl.gz", lines=True, compression="gzip")

steps_df.to_parquet(run_dir / "steps.parquet", index=False)
episodes_df.to_parquet(run_dir / "episodes.parquet", index=False)
```

- Or export Parquet automatically using the CLI:

```bash
plume-nav-capture --output results --experiment demo --episodes 2 --grid 8x8 --parquet
```

### Schema Reference

See the detailed field definitions and evolution policy:

- `src/backend/docs/data_capture_schemas.md`

**Ready to start your plume navigation research?** 🧪

```bash
pip install -e .
python examples/basic_usage.py
```

Happy researching! 🚀
