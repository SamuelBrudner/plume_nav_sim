# Odor Plume Navigation Library

A reusable Python library for simulating agent navigation through odor plumes with sophisticated Hydra-based configuration management, designed for integration with Kedro pipelines, reinforcement learning frameworks, and machine learning/neural network analyses.

## Overview

The Odor Plume Navigation library provides a comprehensive toolkit for research-grade simulation of how agents navigate through odor plumes. Designed as an importable library, it offers clean APIs, modular architecture, and enterprise-grade configuration management for seamless integration into research workflows.

### Key Features

- **Reusable Library Architecture**: Import and use in any Python project
- **Hydra Configuration Management**: Sophisticated hierarchical configuration with environment variable integration
- **Multi-Framework Integration**: Compatible with Kedro, RL frameworks, and ML/neural network analyses
- **CLI Interface**: Command-line tools for automation and batch processing
- **Docker-Ready**: Containerized development and deployment environments
- **Dual Workflow Support**: Poetry and pip installation methods
- **Research-Grade Quality**: Type-safe, well-documented, and thoroughly tested

## Installation

### Prerequisites

- Python 3.9 or higher
- Poetry (recommended) or pip for dependency management
- Docker and docker-compose (optional, for containerized development)

### Installation Methods

#### Poetry Installation (Recommended)

```bash
# Install from PyPI
poetry add {{cookiecutter.project_slug}}

# For development with all optional dependencies
poetry add {{cookiecutter.project_slug}} --group dev,docs,viz
```

#### Pip Installation

```bash
# Standard installation
pip install {{cookiecutter.project_slug}}

# Development installation with optional dependencies
pip install "{{cookiecutter.project_slug}}[dev,docs,viz]"
```

#### Development Installation

```bash
# Clone repository
git clone https://github.com/organization/{{cookiecutter.project_slug}}.git
cd {{cookiecutter.project_slug}}

# Poetry development setup (recommended)
poetry install --with dev,docs,viz
poetry shell

# Alternative: pip development setup
pip install -e ".[dev,docs,viz]"
```

#### Docker-Based Development Environment

```bash
# Full development environment with database and pgAdmin
docker-compose up --build

# Library container only
docker build -t {{cookiecutter.project_slug}} .
docker run -it {{cookiecutter.project_slug}}
```

## Library Usage Patterns

### For Kedro Projects

```python
from {{cookiecutter.project_slug}} import Navigator, VideoPlume
from {{cookiecutter.project_slug}}.config import NavigatorConfig
from hydra import compose, initialize

# Kedro pipeline integration
def create_navigation_pipeline():
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config")
        
        # Create components using Hydra configuration
        navigator = Navigator.from_config(cfg.navigator)
        video_plume = VideoPlume.from_config(cfg.video_plume)
        
        return navigator, video_plume

# Use in Kedro nodes
def navigation_node(navigator: Navigator, video_plume: VideoPlume) -> dict:
    """Kedro node for odor plume navigation simulation."""
    results = navigator.simulate(video_plume, duration=cfg.simulation.max_duration)
    return {"trajectory": results.trajectory, "sensor_data": results.sensor_data}
```

### For Reinforcement Learning Projects

```python
from {{cookiecutter.project_slug}}.core import NavigatorProtocol
from {{cookiecutter.project_slug}}.api import create_navigator
from {{cookiecutter.project_slug}}.utils import set_global_seed

# RL environment integration
class OdorPlumeRLEnv(gym.Env):
    def __init__(self, config_path: str = "conf/config.yaml"):
        super().__init__()
        
        # Set deterministic behavior for RL training
        set_global_seed(42)
        
        # Create navigator from configuration
        self.navigator = create_navigator(config_path)
        self.video_plume = VideoPlume.from_config(config_path)
        
        # Define RL action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(64, 64, 1), dtype=np.uint8
        )
    
    def step(self, action):
        # Execute action and get next state
        self.navigator.update(action)
        observation = self.video_plume.get_sensor_reading(self.navigator.position)
        reward = self._calculate_reward()
        done = self._check_termination()
        return observation, reward, done, {}
```

### For ML/Neural Network Analyses

```python
from {{cookiecutter.project_slug}}.utils import set_global_seed
from {{cookiecutter.project_slug}}.data import VideoPlume
from {{cookiecutter.project_slug}}.api.navigation import run_plume_simulation
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

## Command-Line Interface

The library provides comprehensive CLI commands for automation and batch processing.

### Available Commands

```bash
# Run a simulation with default configuration
plume-nav-sim run

# Run with parameter overrides
plume-nav-sim run navigator.max_speed=2.0 simulation.fps=60

# Parameter sweep execution
plume-nav-sim run --multirun navigator.max_speed=1.0,2.0,3.0 video_plume.kernel_size=3,5,7

# Visualization commands
plume-nav-sim visualize --input-path outputs/experiment_results.npz
plume-nav-sim visualize --animation --save-video output.mp4

# Configuration validation
plume-nav-sim config validate
plume-nav-sim config show

# Environment setup
plume-nav-sim setup --create-dirs --init-config
```

### CLI Integration Examples

```bash
# Research workflow automation
#!/bin/bash
# Multi-condition experiment execution
for speed in 1.0 1.5 2.0; do
    for kernel in 3 5 7; do
        plume-nav-sim run \
            navigator.max_speed=$speed \
            video_plume.kernel_size=$kernel \
            hydra.job.name="speed_${speed}_kernel_${kernel}"
    done
done

# Batch visualization generation
plume-nav-sim visualize \
    --input-dir outputs/multirun/2024-01-15_10-30-00 \
    --output-format mp4 \
    --quality high
```

## Configuration System

The library uses a sophisticated Hydra-based configuration hierarchy that supports environment variable integration, parameter sweeps, and multi-environment deployment.

### Configuration Structure

```
conf/
├── base.yaml          # Foundation defaults and core parameters
├── config.yaml        # User customizations and environment-specific overrides
└── local/             # Local development and deployment-specific settings
    ├── credentials.yaml.template
    ├── development.yaml
    ├── production.yaml
    └── paths.yaml.template
```

### Basic Configuration Usage

```python
from hydra import compose, initialize
from {{cookiecutter.project_slug}}.api import create_navigator

# Basic configuration loading
with initialize(config_path="../conf", version_base=None):
    cfg = compose(config_name="config")
    navigator = create_navigator(cfg.navigator)

# Dynamic parameter overrides
with initialize(config_path="../conf"):
    cfg = compose(config_name="config", overrides=[
        "navigator.max_speed=2.5",
        "video_plume.flip=true",
        "simulation.fps=60"
    ])
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
```

### Migration from Legacy Configuration

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
└── local/             # New: environment-specific overrides
    ├── development.yaml
    └── production.yaml
```

#### Migration Steps

1. **Copy base parameters**: Move `configs/default.yaml` content to `conf/base.yaml`
2. **User customizations**: Move `configs/example_user_config.yaml` to `conf/config.yaml`
3. **Environment setup**: Create environment-specific files in `conf/local/`
4. **Update imports**: Change from:
   ```python
   # Old approach
   from {{cookiecutter.project_slug}}.services.config_loader import load_config
   config = load_config("configs/default.yaml")
   ```
   
   To:
   ```python
   # New Hydra approach
   from hydra import compose, initialize
   with initialize(config_path="../conf"):
       cfg = compose(config_name="config")
   ```

5. **CLI migration**: Replace manual script execution with new CLI commands:
   ```bash
   # Old approach
   python scripts/run_simulation.py --config configs/my_config.yaml
   
   # New approach
   plume-nav-sim run --config-name my_config
   ```

## Development Workflow

### Setup Development Environment

```bash
# Clone and setup
git clone https://github.com/organization/{{cookiecutter.project_slug}}.git
cd {{cookiecutter.project_slug}}

# Poetry setup (recommended)
poetry install --with dev,docs,viz
poetry shell

# Install pre-commit hooks
pre-commit install

# Alternative: Make-based setup
make setup-dev
```

### Makefile Commands

The project includes comprehensive Makefile automation:

```bash
# Development commands
make install-dev       # Poetry install with dev dependencies
make setup-dev         # Complete development environment setup
make install          # Traditional pip install (fallback)

# Code quality
make format           # Run black and isort formatting
make lint            # Run flake8 linting
make type-check      # Run mypy type checking
make test            # Run pytest with coverage
make test-all        # Run all quality checks

# Build and distribution
make build           # Build wheel and sdist
make poetry-build    # Build using Poetry
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
pytest --cov={{cookiecutter.project_slug}} --cov-report=html

# Run specific test categories
pytest -m "not slow"          # Skip slow tests
pytest -m integration         # Integration tests only
pytest tests/unit/            # Unit tests only

# Run with parallel execution
pytest -n auto               # Parallel execution
```

## Advanced Features

### Multi-Run Experiment Management

```bash
# Systematic parameter exploration
plume-nav-sim run --multirun \
  navigator.max_speed=1.0,1.5,2.0 \
  navigator.angular_velocity=0.1,0.2,0.3 \
  video_plume.gaussian_blur.sigma=1.0,2.0,3.0

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
  {{cookiecutter.project_slug}}:
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
from {{cookiecutter.project_slug}}.utils import configure_performance

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
{{cookiecutter.project_slug}}/
├── src/{{cookiecutter.project_slug}}/           # Main library package
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
│   └── utils/                        # Utilities
│       ├── __init__.py
│       ├── seed_manager.py           # Reproducibility
│       ├── visualization.py          # Plotting and animation
│       └── logging.py                # Logging configuration
├── conf/                             # Hydra configuration
│   ├── base.yaml                     # Foundation defaults
│   ├── config.yaml                   # User customizations
│   └── local/                        # Environment-specific
│       ├── credentials.yaml.template
│       └── paths.yaml.template
├── tests/                            # Test suite
├── notebooks/                        # Example notebooks
│   ├── demos/                        # Demonstration notebooks
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

## Integration Examples

### Jupyter Notebook Integration

```python
# notebook_example.ipynb
from hydra import compose, initialize
from {{cookiecutter.project_slug}} import Navigator, VideoPlume
from {{cookiecutter.project_slug}}.utils import set_global_seed

# Setup reproducible environment
set_global_seed(42)

# Load configuration in notebook
with initialize(config_path="../conf", version_base=None):
    cfg = compose(config_name="config", overrides=[
        "visualization.animation.enabled=true",
        "visualization.plotting.figure_size=[14,10]"
    ])

# Create and run simulation
navigator = Navigator.from_config(cfg.navigator)
video_plume = VideoPlume.from_config(cfg.video_plume)

# Interactive visualization
results = navigator.simulate(video_plume, duration=60)
results.plot_trajectory(interactive=True)
```

### Kedro Pipeline Integration

```python
# kedro_pipeline_example.py
from kedro.pipeline import Pipeline, node
from {{cookiecutter.project_slug}}.api import create_navigator, run_plume_simulation

def create_navigation_pipeline(**kwargs) -> Pipeline:
    """Create Kedro pipeline for odor plume navigation."""
    
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
@software{odor_plume_navigation_library,
  title={Odor Plume Navigation Library},
  author={Samuel Brudner},
  year={2024},
  url={https://github.com/organization/{{cookiecutter.project_slug}}},
  version={0.1.0}
}
```

## Support and Documentation

- **Documentation**: [https://{{cookiecutter.project_slug}}.readthedocs.io](https://{{cookiecutter.project_slug}}.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/organization/{{cookiecutter.project_slug}}/issues)
- **Discussions**: [GitHub Discussions](https://github.com/organization/{{cookiecutter.project_slug}}/discussions)
- **API Reference**: Generated automatically from docstrings

## Changelog

### Version 0.1.0 (Initial Release)

- **Library Architecture**: Transformed from standalone application to importable library
- **Hydra Configuration**: Sophisticated hierarchical configuration management
- **CLI Interface**: Comprehensive command-line tools with Click framework
- **Multi-Framework Support**: Integration patterns for Kedro, RL, and ML workflows
- **Docker Support**: Containerized development and deployment environments
- **Dual Workflows**: Poetry and pip installation support
- **Enhanced Documentation**: Comprehensive usage examples and migration guides

For detailed changes, see [CHANGELOG.md](CHANGELOG.md).