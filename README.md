# Odor Plume Navigation: A Reusable Simulation Library

[![PyPI version](https://badge.fury.io/py/{{cookiecutter.project_slug}}.svg)](https://badge.fury.io/py/{{cookiecutter.project_slug}})
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A reusable Python library for simulating navigation of odor plumes with Hydra configuration management, designed for seamless integration with Kedro pipelines, reinforcement learning frameworks, and machine learning analysis workflows.

## 🚀 Quick Start

### Library Import Patterns

```python
# For Kedro projects
from {{cookiecutter.project_slug}} import Navigator, VideoPlume
from {{cookiecutter.project_slug}}.config import NavigatorConfig

# For RL projects
from {{cookiecutter.project_slug}}.core import NavigatorProtocol
from {{cookiecutter.project_slug}}.api import create_navigator

# For ML/neural network analyses
from {{cookiecutter.project_slug}}.utils import set_global_seed
from {{cookiecutter.project_slug}}.data import VideoPlume
```

### Basic Usage with Hydra Configuration

```python
from hydra import compose, initialize
from {{cookiecutter.project_slug}}.api.navigation import create_navigator
from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume

# Initialize with Hydra configuration
with initialize(config_path="conf", version_base=None):
    cfg = compose(config_name="config")
    
    # Create navigator from configuration
    navigator = create_navigator(cfg.navigator)
    
    # Create video plume environment
    video_plume = VideoPlume.from_config(cfg.video_plume)
    
    # Run simulation
    results = run_plume_simulation(navigator, video_plume, cfg.simulation)
```

## 📦 Installation

### Option 1: Poetry (Recommended)

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Clone and install
git clone <repository-url>
cd {{cookiecutter.project_slug}}
poetry install

# Activate environment
poetry shell
```

### Option 2: pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Option 3: Conda

```bash
# Create conda environment
conda create -n {{cookiecutter.project_slug}} python=3.9
conda activate {{cookiecutter.project_slug}}

# Install package
pip install -e .
```

### Option 4: Docker Development Environment

```bash
# Clone repository
git clone <repository-url>
cd {{cookiecutter.project_slug}}

# Start development environment with docker-compose
docker-compose up -d

# Access development container
docker-compose exec app bash

# Or use integrated development with volume mounting
docker-compose -f docker-compose.dev.yml up
```

## 🖥️ Command Line Interface

The library provides comprehensive CLI commands using Click-based interface:

### Basic Commands

```bash
# Run simulation with default configuration
plume-nav-sim run

# Run with parameter overrides
plume-nav-sim run navigator.max_speed=2.0 video_plume.flip=true

# Generate visualization from results
plume-nav-sim visualize outputs/latest/trajectories.npy

# Multi-run parameter sweep
plume-nav-sim run --multirun navigator.max_speed=1.0,2.0,3.0 navigator.orientation=0,45,90
```

### Advanced CLI Usage

```bash
# Run with custom configuration
plume-nav-sim run --config-path=./custom_configs --config-name=experiment_1

# Environment variable integration
DEBUG=true LOG_LEVEL=INFO plume-nav-sim run

# Batch processing with specific output directory
plume-nav-sim run hydra.run.dir=./outputs/experiment_batch_001

# Export configuration template
plume-nav-sim config --template > my_config.yaml
```

## ⚙️ Configuration System

### New Hydra-Based Configuration Architecture

The library uses a sophisticated hierarchical configuration system:

```
conf/
├── base.yaml          # Foundation defaults and core parameters
├── config.yaml        # Environment-specific overrides
└── local/             # Local development configurations
    ├── credentials.yaml.template
    └── paths.yaml.template
```

### Configuration Hierarchy

1. **conf/base.yaml** - Immutable foundation defaults
2. **conf/config.yaml** - Environment-specific overrides  
3. **conf/local/*.yaml** - Runtime customizations

### Example Configuration Usage

```yaml
# conf/config.yaml
defaults:
  - base
  - _self_

# Override specific parameters
navigator:
  orientation: 90.0  # Start facing up
  speed: 0.5         # Initial movement speed
  max_speed: 2.0     # Enhanced speed limit

video_plume:
  flip: true          # Horizontal flip for testing
  kernel_size: 3      # Gaussian smoothing
  
# Environment variable integration
database:
  url: ${oc.env:DATABASE_URL,sqlite:///local.db}
  username: ${oc.env:DB_USER,dev_user}
  password: ${oc.env:DB_PASSWORD}
```

### Environment Variable Integration

```bash
# .env file for development
export VIDEO_PATH="/path/to/video.mp4"
export DATABASE_URL="postgresql://user:pass@localhost/db"
export DEBUG="true"
export LOG_LEVEL="DEBUG"
export MATPLOTLIB_BACKEND="Qt5Agg"

# Load with python-dotenv
from dotenv import load_dotenv
load_dotenv()
```

## 🏗️ Project Structure

```
{{cookiecutter.project_slug}}/
├── src/{{cookiecutter.project_slug}}/
│   ├── __init__.py                     # Public API exports
│   ├── api/                           # Public interfaces
│   │   ├── __init__.py
│   │   └── navigation.py              # Main API functions
│   ├── cli/                           # Command-line interface
│   │   ├── __init__.py
│   │   └── main.py                    # Click-based CLI commands
│   ├── config/                        # Configuration schemas
│   │   ├── __init__.py
│   │   └── schemas.py                 # Pydantic validation models
│   ├── core/                          # Navigation algorithms
│   │   ├── __init__.py
│   │   ├── navigator.py               # NavigatorProtocol definition
│   │   ├── controllers.py             # Agent controller implementations
│   │   └── sensors.py                 # Sensor configuration models
│   ├── data/                          # Data processing
│   │   ├── __init__.py
│   │   └── video_plume.py             # OpenCV video processing
│   ├── db/                            # Database support (future)
│   │   └── session.py                 # SQLAlchemy session management
│   └── utils/                         # Utilities and visualization
│       ├── __init__.py
│       ├── seed_manager.py            # Random seed management
│       ├── visualization.py           # Matplotlib visualization
│       └── logging.py                 # Loguru configuration
├── conf/                              # Hydra configuration
│   ├── base.yaml                      # Foundation defaults
│   ├── config.yaml                    # Environment-specific settings
│   └── local/                         # Local development configs
├── notebooks/                         # Jupyter notebooks
│   ├── demos/                         # Example demonstrations
│   └── exploratory/                   # Research and analysis
├── tests/                             # Test suite
├── workflow/                          # DVC/Snakemake integration
│   ├── dvc/                          # DVC pipeline definitions
│   └── snakemake/                    # Snakemake workflow rules
├── docker-compose.yml                # Development environment
├── Makefile                          # Development automation
├── pyproject.toml                    # Project metadata and dependencies
└── README.md                         # This file
```

## 💻 Development Workflow

### Development Environment Setup

```bash
# Install pre-commit hooks
pre-commit install

# Install development dependencies
make install-dev

# Set up development environment
make setup-dev
```

### Available Make Commands

```bash
# Core development tasks
make install              # Install package in development mode
make install-dev          # Install with development dependencies
make test                 # Run test suite with coverage
make lint                 # Run code linting (black, isort, flake8)
make type-check           # Run mypy type checking
make format               # Format code with black and isort

# Advanced development tasks
make test-fast            # Run tests without coverage
make test-integration     # Run integration tests only
make docs                 # Build documentation
make docs-serve           # Serve documentation locally
make clean                # Clean build artifacts
make setup-dev           # Complete development environment setup

# Docker development
make docker-build        # Build development Docker image
make docker-dev          # Start development environment
make docker-test         # Run tests in Docker container
```

### Code Quality and Testing

```bash
# Run all quality checks
make qa

# Run tests with coverage report
pytest --cov={{cookiecutter.project_slug}} --cov-report=html

# Type checking with mypy
mypy src tests

# Code formatting
black src tests
isort src tests

# Linting
flake8 src tests
```

## 🔧 Advanced Usage

### Multi-Agent Simulations

```python
from {{cookiecutter.project_slug}}.core.controllers import MultiAgentController
import numpy as np

# Configure multi-agent simulation
cfg.navigator.num_agents = 10
cfg.navigator.formation = "grid"
cfg.navigator.communication_range = 15.0

# Create multi-agent navigator
navigator = MultiAgentController.from_config(cfg.navigator)

# Run swarm simulation
results = run_plume_simulation(navigator, video_plume, cfg.simulation)
```

### Custom Visualization

```python
from {{cookiecutter.project_slug}}.utils.visualization import visualize_simulation_results
import matplotlib.pyplot as plt

# Generate publication-quality figures
fig = visualize_simulation_results(
    results, 
    cfg.visualization,
    show_trails=True,
    trail_length=100,
    export_format="pdf"
)

# Save high-resolution plot
plt.savefig("trajectory_analysis.pdf", dpi=300, bbox_inches='tight')
```

### Reproducible Research

```python
from {{cookiecutter.project_slug}}.utils.seed_manager import set_global_seed

# Set reproducible random seeds
set_global_seed(42)

# Enable strict deterministic mode
cfg.reproducibility.strict_mode = True
cfg.reproducibility.validate_reproducibility = True

# Run reproducible experiment
results = run_plume_simulation(navigator, video_plume, cfg.simulation)
```

## 📊 Integration Examples

### Kedro Pipeline Integration

```python
# kedro_project/src/nodes.py
from kedro.pipeline import node, Pipeline
from {{cookiecutter.project_slug}}.api.navigation import create_navigator
from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume

def simulate_navigation(video_path: str, nav_config: dict) -> dict:
    """Kedro node for odor plume navigation simulation."""
    video_plume = VideoPlume(video_path=video_path)
    navigator = create_navigator(nav_config)
    return run_plume_simulation(navigator, video_plume)

# kedro_project/src/pipeline.py
def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=simulate_navigation,
            inputs=["video_data", "params:navigator"],
            outputs="simulation_results",
            name="navigation_simulation"
        )
    ])
```

### Reinforcement Learning Integration

```python
import gym
from {{cookiecutter.project_slug}}.core import NavigatorProtocol
from {{cookiecutter.project_slug}}.data import VideoPlume

class OdorNavigationEnv(gym.Env):
    """RL environment using odor plume navigation library."""
    
    def __init__(self, config):
        self.video_plume = VideoPlume.from_config(config.video_plume)
        self.navigator = create_navigator(config.navigator)
        
    def step(self, action):
        # Use library components for environment dynamics
        return self.navigator.step(action, self.video_plume.current_frame)
```

### Jupyter Notebook Analysis

```python
# Research notebook integration
%load_ext autoreload
%autoreload 2

from {{cookiecutter.project_slug}} import Navigator, VideoPlume
from {{cookiecutter.project_slug}}.utils import set_global_seed

# Set up reproducible experiment
set_global_seed(123)

# Interactive parameter exploration
cfg.navigator.max_speed = 3.0  # Modify configuration dynamically
navigator = Navigator.from_config(cfg.navigator)

# Visualize results inline
%matplotlib inline
visualize_simulation_results(results, cfg.visualization)
```

## 🔄 Migration Guide

### From Legacy configs/ to New conf/ System

The refactored library replaces the original PyYAML-based configuration with Hydra:

#### Legacy Configuration (Old)
```python
# Old approach - manual YAML loading
import yaml
with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

navigator = Navigator.from_config(config['navigator'])
```

#### New Hydra Configuration (Current)
```python
# New approach - Hydra composition
from hydra import compose, initialize

with initialize(config_path="conf", version_base=None):
    cfg = compose(config_name="config")
    navigator = create_navigator(cfg.navigator)
```

#### Migration Steps

1. **Move configuration files:**
   ```bash
   mkdir conf/
   mv configs/default.yaml conf/base.yaml
   mv configs/example_user_config.yaml conf/config.yaml
   ```

2. **Update configuration format:**
   ```bash
   # Add Hydra defaults to conf/config.yaml
   echo "defaults:\n  - base\n  - _self_" > conf/config.yaml.new
   cat conf/config.yaml >> conf/config.yaml.new
   mv conf/config.yaml.new conf/config.yaml
   ```

3. **Update import statements:**
   ```python
   # Replace old imports
   from odor_plume_nav.api import Navigator
   
   # With new imports
   from {{cookiecutter.project_slug}}.api.navigation import create_navigator
   ```

4. **Update CLI usage:**
   ```bash
   # Old CLI (if existed)
   python -m odor_plume_nav --config configs/default.yaml
   
   # New CLI
   plume-nav-sim run navigator.max_speed=2.0
   ```

## 📚 Examples and Demos

### Basic Single-Agent Navigation

```python
# examples/basic_navigation.py
from hydra import compose, initialize
from {{cookiecutter.project_slug}}.api.navigation import create_navigator, run_plume_simulation
from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume

def basic_navigation_demo():
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config", overrides=[
            "navigator.orientation=45",
            "navigator.max_speed=1.5",
            "video_plume.kernel_size=3"
        ])
        
        navigator = create_navigator(cfg.navigator)
        video_plume = VideoPlume.from_config(cfg.video_plume)
        
        results = run_plume_simulation(navigator, video_plume, cfg.simulation)
        print(f"Simulation completed: {len(results.trajectory)} steps")

if __name__ == "__main__":
    basic_navigation_demo()
```

### Batch Parameter Study

```python
# examples/parameter_study.py
import itertools
from {{cookiecutter.project_slug}}.utils.seed_manager import set_global_seed

def parameter_sweep():
    speeds = [0.5, 1.0, 1.5, 2.0]
    orientations = [0, 45, 90, 135]
    
    results = {}
    for speed, orientation in itertools.product(speeds, orientations):
        set_global_seed(42)  # Reproducible results
        
        with initialize(config_path="../conf", version_base=None):
            cfg = compose(config_name="config", overrides=[
                f"navigator.max_speed={speed}",
                f"navigator.orientation={orientation}"
            ])
            
            navigator = create_navigator(cfg.navigator)
            video_plume = VideoPlume.from_config(cfg.video_plume)
            
            sim_results = run_plume_simulation(navigator, video_plume, cfg.simulation)
            results[(speed, orientation)] = sim_results
    
    return results
```

## 🔗 Dependencies

### Core Dependencies
- **numpy** ≥1.24.0 - Numerical computing and array operations
- **matplotlib** ≥3.7.0 - Scientific visualization and plotting
- **opencv-python** ≥4.8.0 - Video processing and computer vision
- **scipy** ≥1.10.0 - Scientific computing and algorithms
- **hydra-core** ≥1.3.2 - Configuration management and composition
- **pydantic** ≥2.5.0 - Data validation and configuration schemas
- **loguru** ≥0.7.0 - Structured logging and debugging

### Infrastructure Dependencies
- **click** ≥8.2.1 - Command-line interface framework
- **python-dotenv** ≥1.1.0 - Environment variable management
- **sqlalchemy** ≥2.0.41 - Database ORM for future persistence features
- **typing-extensions** ≥4.13.2 - Enhanced type hints and annotations

### Development Dependencies
- **pytest** ≥7.4.0 - Test framework and execution
- **pytest-cov** ≥4.1.0 - Code coverage reporting
- **pre-commit** ≥3.6.0 - Git hooks and code quality automation
- **black** ≥23.12.0 - Code formatting and style consistency
- **isort** ≥5.13.0 - Import sorting and organization
- **flake8** ≥6.0.0 - Code linting and style checking
- **mypy** ≥1.5.0 - Static type checking and validation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Setting up the development environment
- Code style and testing requirements
- Submitting pull requests
- Reporting issues and feature requests

### Development Setup for Contributors

```bash
# Fork and clone the repository
git clone https://github.com/your-username/{{cookiecutter.project_slug}}.git
cd {{cookiecutter.project_slug}}

# Set up development environment
make setup-dev

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
make test
```

## 📈 Roadmap

- [ ] **Enhanced RL Integration** - Gym environment wrappers and OpenAI Baselines compatibility
- [ ] **Neural Navigation Models** - Built-in neural network-based navigation algorithms
- [ ] **Cloud Storage Integration** - S3/GCS support for large-scale dataset management
- [ ] **Real-time Streaming** - WebSocket-based real-time visualization and monitoring
- [ ] **Performance Optimization** - GPU acceleration for multi-agent simulations
- [ ] **Extended Sensor Models** - Additional sensor types and noise models

## 📞 Support

- **Documentation**: [https://{{cookiecutter.project_slug}}.readthedocs.io](https://{{cookiecutter.project_slug}}.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/organization/{{cookiecutter.project_slug}}/issues)
- **Discussions**: [GitHub Discussions](https://github.com/organization/{{cookiecutter.project_slug}}/discussions)
- **Email**: support@{{cookiecutter.project_slug}}.org

---

**🧭 Navigate the future of odor-guided simulation with confidence.**