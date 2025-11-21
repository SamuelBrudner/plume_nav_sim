# plume-nav-sim

🧪 Proof-of-Life Gymnasium Environment for Plume Navigation Research

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![Gymnasium](https://img.shields.io/badge/gymnasium-0.29%2B-green.svg)](https://gymnasium.farama.org/) [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)

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
from plume_nav_sim.registration import ensure_registered, ENV_ID
ensure_registered()  # make ENV_ID available to gym.make()

import gymnasium as gym
env = gym.make(ENV_ID)

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

## 🧭 Public API and Repository Layout

For most users and external researchers, the supported way to interact with the package is:

- **Top-level API**: `import plume_nav_sim as pns`
  - Use `pns.make_env(...)` as the recommended way to create environments.
  - Core types, constants, and metadata are exported from `plume_nav_sim.__init__` (e.g., `GridSize`, `EnvironmentConfig`, `DEFAULT_*`, `ENVIRONMENT_ID`).
- **Configuration and composition**: `plume_nav_sim.config`
  - Typed specs and composition helpers live under `plume_nav_sim.config` and `plume_nav_sim.config.composition` (e.g., `SimulationSpec`, `PolicySpec`, `prepare`).
  - Legacy imports from `plume_nav_sim.compose.*` are still supported as shims but new code should prefer `plume_nav_sim.config`.

If you are browsing the source code in this repository:

- Installable package code lives under `src/backend/plume_nav_sim/`.
- Hydra/YAML configuration files live under `src/backend/conf/`.
- Built-in scenarios and benchmarks live under `src/backend/scenarios/`.
- Documentation and design notes live under `src/backend/docs/`.
- Examples and tutorials live under `src/backend/examples/` and `notebooks/`.
- Tests live under `src/backend/tests/`.

Contributors extending the library (e.g., new policies, plume models, or data-capture features) should generally add code under the corresponding subpackages:

- `plume_nav_sim.envs` – environment implementations and factories.
- `plume_nav_sim.policies` – policies and policy helpers.
- `plume_nav_sim.plume` – plume models and concentration field logic.
- `plume_nav_sim.render` – rendering utilities, colormaps, and templates.
- `plume_nav_sim.data_capture`, `plume_nav_sim.media`, `plume_nav_sim.video` – capture pipeline, dataset manifests, and video plume schemas.

### Quick Start with Factory Function

```python
import plume_nav_sim as pns

# Streamlined environment setup using the factory
env = pns.make_env()
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

See also (external-style DI example):

- The plug‑and‑play demo shows DI assembly from an external project and applies the core `ConcentrationNBackWrapper(n=2)` via `SimulationSpec`.
  - Quick run from repo root: `python plug-and-play-demo/main.py`
  - Full walkthrough and options: `plug-and-play-demo/README.md`

See `plume_nav_sim/config/factories.py` for config-driven creation (Hydra/YAML).

```

### Registration System

```python
from plume_nav_sim.registration import ensure_registered, is_registered, register_env, ENV_ID

# Register with default parameters (idempotent)
ensure_registered()

# Register with custom configuration
register_env(kwargs=custom_config)

# Check registration status
if is_registered():
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
import plume_nav_sim as pns
import matplotlib.pyplot as plt

# Create environment with visualization in mind
env = pns.make_env(grid_size=(64, 64))

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

### Compose: Applying observation wrappers via SimulationSpec

Use SimulationSpec to declare observation adapters (wrappers) so the full runtime
behavior is defined in one place. Each wrapper is specified with a dotted path
and kwargs and applied in order by `compose.prepare()`.

```python
from plume_nav_sim.compose import SimulationSpec, PolicySpec, WrapperSpec, prepare

sim = SimulationSpec(
    grid_size=(64, 64),
    max_steps=200,
    action_type="run_tumble",           # Discrete(2): 0=RUN, 1=TUMBLE
    observation_type="concentration",    # Box(1,): odor at agent position
    reward_type="step_penalty",
    render=False,
    seed=123,
    policy=PolicySpec(spec="my_project.policies:MyRunTumblePolicy"),
    observation_wrappers=[
        # Core 1‑back odor history: transforms Box(1,) → Box(2,) [c_prev, c_now]
        WrapperSpec(
            spec="plume_nav_sim.observations.history_wrappers:ConcentrationNBackWrapper",
            kwargs={"n": 2},
        ),
    ],
)

env, policy = prepare(sim)
obs, info = env.reset(seed=sim.seed)
print(env.observation_space.shape)  # (2,)
```

Notes:
- Wrapper targets accept `(env, **kwargs)` and return a Gymnasium `Env`.
- Dotted path can be `"module:Attr"` or `"module.sub.Attr"`.
- Wrappers are applied before policy subset validation so the policy can target
  the adapted observation space.

### Reproducibility Demo

```python
import plume_nav_sim as pns

def run_deterministic_episode(seed=42):
    env = pns.make_env()
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

### Movie Plume Field (Zarr-backed)

Video-derived concentration field that advances one frame per step:

- **Dataset schema**: `concentration (t, y, x)` with float32 values
- **Step policy**: `wrap` (loop) or `clamp` (hold last frame)
- **Metadata**: fps, pixel_to_grid, origin, extent validated via `VideoPlumeAttrs`
- **Usage**: `plume="movie"` with either a curated registry id (`movie_dataset_id="colorado_jet_v1"`) that resolves via the data zoo cache, or a direct path (`movie_path`) to a Zarr dataset/raw movie + sidecar
- **Cache**: registry downloads default to `~/.cache/plume_nav_sim/data_zoo`; override with `movie_cache_root` (e.g., HPC scratch) and set `movie_auto_download=True` to fetch when missing
- Details and examples: `src/backend/docs/plume_types.md`

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
import plume_nav_sim as pns
env = pns.make_env(**performance_configs['fast_training'])
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

import plume_nav_sim as pns
env = pns.make_env()
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
import plume_nav_sim as pns
config = {
    'grid_size': (64, 64),    # Instead of (128, 128)
    'max_steps': 500,         # Shorter episodes
}
env = pns.make_env(**config)
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
from plume_nav_sim.registration import ensure_registered, ENV_ID
ensure_registered()

import gymnasium as gym
env = gym.make(ENV_ID)
```

**Problem: Inconsistent reproducibility**

```python
# Proper seeding approach
import plume_nav_sim as pns
env = pns.make_env()

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

# Linting (flake8)
# Recommended: run the Makefile target from repo root, which mirrors CI exactly
make lint ENV_NAME=plume-nav-sim

# Or run flake8 directly with the same flags as CI
flake8 src/plume_nav_sim/ \
  --max-line-length=88 \
  --extend-ignore=E203,W503,E501 \
  --select=E,W,F,C,N \
  --max-complexity=10 \
  --per-file-ignores="src/backend/plume_nav_sim/__init__.py:F401,F403,F405,src/backend/plume_nav_sim/envs/base_env.py:C901,src/backend/plume_nav_sim/core/episode_manager.py:C901"
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

### Render Module Structure

- All render-specific Python code and utilities live under `src/backend/plume_nav_sim/render/`.
- Use `plume_nav_sim.render.*` imports exclusively for rendering utilities.
- Colormap utilities moved to `plume_nav_sim.render.colormaps` (formerly `assets.default_colormap`).
- Rendering templates moved to `plume_nav_sim.render.templates` (formerly `assets.render_templates`).
- The legacy `src/backend/assets/` package has been removed.

### Contribution Guidelines

1. **Code Standards**: Follow PEP 8 style guidelines
2. **Testing**: Add tests for new functionality
3. **Documentation**: Update docstrings and README
4. **Backwards Compatibility**: Maintain API compatibility
5. **Performance**: Ensure changes don't degrade performance

#### Pre-commit Hooks

- Install hooks (from repo root using backend config):
  - `pre-commit install -c src/backend/.pre-commit-config.yaml`
- Run all hooks (from repo root):
  - `pre-commit run -c src/backend/.pre-commit-config.yaml --all-files`
- Alternative (from backend directory):
  - `(cd src/backend && pre-commit install && pre-commit run --all-files)`

### Development Workflow

1. **Feature Branch**: Create branch from main
2. **Development**: Implement changes with tests
3. **Validation**: Run full test suite
4. **Documentation**: Update relevant documentation
5. **Review**: Submit pull request for review

## 📄 License & Citation

### License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

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
@software{plume_nav_sim_2025,
  title={plume-nav-sim: Gymnasium Environment for Plume Navigation Research},
  author={plume_nav_sim Development Team},
  year={2025},
  url={https://github.com/SamuelBrudner/plume_nav_sim},
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
- **Issues**: [GitHub Issues](https://github.com/SamuelBrudner/plume_nav_sim/issues)
- **Discussions**: [GitHub Discussions](https://github.com/SamuelBrudner/plume_nav_sim/discussions)
- **Email**: <plume-nav-sim@example.com>

---

## 📑 Operational Logging vs Data Capture

- Use loguru for operational, human-readable logs (console/file). Configure it independently of data capture.

```python
from plume_nav_sim.logging.loguru_bootstrap import setup_logging
setup_logging(level="INFO", console=True, file_path="run.log", rotation="10 MB", retention="7 days")
```

- By default, the `file_path` above is interpreted **relative to your current working directory**. When running from the
  project root, this will create `run.log` (or any other log file you choose) in the repo root, which is already
  ignored by `.gitignore`. There is no requirement for a committed `logs/` directory under `src/backend`; any
  `logs/` directory you see there is a runtime artifact and can be safely deleted.

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

Note: For manifest usage and an end-to-end reference, see bead plume_nav_sim-152.

### Data Directories

See `src/backend/docs/data_directories_overview.md` for full details.

- `plume_nav_sim/data_capture/` → runtime capture pipeline (JSONL.gz, validation, CLI)
- `plume_nav_sim/media/` → dataset metadata/manifests and xarray‑like dataset validation
- `plume_nav_sim/video/` → canonical video plume dataset schema and attrs validation

Contracts:
- Video plume dataset: `src/backend/docs/contracts/video_plume_dataset.md`

### Schema Reference

See the detailed field definitions and evolution policy:

- `src/backend/docs/data_capture_schemas.md`

### Data Catalog

For a consolidated overview of artifacts, DVC workflow, and consumer loading/validation examples, see:

- `src/backend/docs/data_catalog_capture.md`

**Ready to start your plume navigation research?** 🧪

```bash
pip install -e .
python examples/basic_usage.py
```

Happy researching! 🚀
- Exploration notebook (capture end‑to‑end): notebooks/stable/capture_end_to_end.ipynb
  - Render to HTML for docs with nbconvert:
    - `make nb-render` (outputs to `src/backend/docs/notebooks/capture_end_to_end.html`)
