# Agent Simulation: Odor Plume Navigation

A Python package for simulating agent navigation through odor plumes, with support for both single and multi-agent simulations.

## Overview

This package provides tools for simulating how agents navigate through odor plumes. The simulation framework supports:

- Video-based odor plume environments
- Configuration-based agent initialization
- Single and multi-agent (vectorized) simulations
- Visualizations and analysis tools

## Installation

### Prerequisites

- Python 3.8 or higher
- Conda environment manager (recommended)

### Setting up the environment

```bash
# Create and activate the conda environment
conda create -n agent-simulation python=3.8
conda activate agent-simulation

# Clone the repository
git clone <repository-url>
cd agent-simulation

# Install the package in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Usage

### Basic usage

```python
import numpy as np
from odor_plume_nav.core.navigator import Navigator
from odor_plume_nav.video_plume import VideoPlume

# Create a navigator with a single agent
navigator = Navigator(position=(10, 20), orientation=45, speed=0.5, max_speed=2.0)

# Or create multiple agents
positions = np.array([[10, 20], [30, 40], [50, 60]])
orientations = np.array([45, 90, 135])
speeds = np.array([0.5, 0.7, 0.9])
multi_navigator = Navigator(positions=positions, orientations=orientations, speeds=speeds)

# Create a video plume environment
video_plume = VideoPlume(video_path="path/to/video.mp4")

# Access plume data
frame = video_plume.get_frame(frame_idx=0)
```

### Using configuration files

The package supports configuration-based initialization:

```python
# Load from configuration
navigator = Navigator.from_config({
    "positions": [[10, 20], [30, 40]],
    "orientations": [45, 90],
    "speeds": [0.5, 0.7],
    "max_speeds": [2.0, 2.5]
})

# Load video plume from configuration
video_plume = VideoPlume.from_config(
    video_path="path/to/video.mp4",
    config_dict={"flip": True, "kernel_size": 5}
)
```

## Project Structure

```
agent_simulation/
├── configs/          # Configuration templates and examples
├── examples/         # Example scripts and notebooks
├── src/              # Source code
│   └── odor_plume_nav/
│       ├── core/                   # Core components
│       │   ├── navigator.py        # Unified navigation system
│       │   └── ...                 # Other core modules
│       ├── video_plume.py          # Video-based plume environment
│       ├── config/                 # Configuration-related modules
│       │   ├── config_models.py    # Configuration validation models
│       │   ├── utils.py            # Configuration utility functions
│       │   └── ...                 # Other config modules
│       ├── visualization.py        # Data visualization tools
│       └── ...
├── tests/            # Test suite
├── pyproject.toml    # Project metadata and dependencies
└── README.md         # This file
```

## Development

### Setting up for development

```bash
# Activate the conda environment
conda activate agent-simulation

# Install development dependencies
pip install -e ".[dev]"
```

### Running tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=odor_plume_nav
```

### Code style

The project uses:
- Flake8 for code style
- MyPy for type checking

```bash
# Run linting
flake8 src tests

# Run type checking
mypy src
```

## Key Features

### Unified Navigator

The `Navigator` class provides a unified interface for both single and multi-agent navigation:

- Configuration-based initialization with validation
- Vectorized operations for efficient multi-agent simulations
- Support for both single and multi-agent simulations with a consistent API
- Structured in a modular architecture for better maintainability

### Video Plume Environment

The `VideoPlume` class provides a frame-based odor plume environment:

- Video file-based plume representation
- Frame retrieval and manipulation options
- Configuration validation

## License

This project is licensed under the MIT License - see the LICENSE file for details.