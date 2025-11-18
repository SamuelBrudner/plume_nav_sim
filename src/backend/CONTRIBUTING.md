# Contributing to plume-nav-sim

Welcome to the **plume-nav-sim** project! We're excited that you're interested in contributing to this proof-of-life Gymnasium environment for plume navigation research. This document provides comprehensive guidelines for contributing to the project, ensuring high-quality, consistent, and scientifically rigorous contributions.

## Table of Contents

1. [Welcome and Project Overview](#welcome-and-project-overview)
2. [Development Environment Setup](#development-environment-setup)
3. [Repository Layout and Public API](#repository-layout-and-public-api)
4. [Contribution Workflow](#contribution-workflow)
5. [Code Quality Standards](#code-quality-standards)
6. [Testing Requirements](#testing-requirements)
7. [Documentation Standards](#documentation-standards)
8. [Scientific Reproducibility Standards](#scientific-reproducibility-standards)
9. [Issue Reporting Guidelines](#issue-reporting-guidelines)
10. [Community Guidelines](#community-guidelines)
11. [Release Procedures](#release-procedures)
12. [Troubleshooting](#troubleshooting)

## Welcome and Project Overview

**plume-nav-sim** is a proof-of-life Gymnasium environment for plume navigation research, designed to provide researchers with a standardized platform for developing and evaluating reinforcement learning algorithms in chemical plume navigation scenarios. Built on the Gymnasium framework with scientific Python standards, this project emphasizes:

- **Scientific Rigor**: Reproducible research with deterministic seeding and cross-session consistency
- **RL Ecosystem Integration**: Full Gymnasium API compliance for seamless integration with training frameworks
- **Performance Optimization**: Sub-millisecond step latency and efficient memory usage for real-time research
- **Community Standards**: Comprehensive testing, documentation, and code quality enforcement

Your contributions help advance the field of plume navigation research by providing reliable, well-tested tools for the scientific community.

## Development Environment Setup

### Automated Setup (Recommended)

We provide an automated setup script that handles the complete development environment configuration:

```bash
# Clone the repository
git clone https://github.com/SamuelBrudner/plume_nav_sim.git
cd plume_nav_sim/src/backend

# Run automated setup script
python scripts/setup_dev_env.py --verbose

# Activate the created virtual environment
source plume-nav-env/bin/activate  # Linux/macOS
plume-nav-env\Scripts\activate     # Windows
```

The setup script automatically:
- Creates a Python 3.10+ virtual environment
- Installs all development dependencies from `pyproject.toml`
- Configures pre-commit hooks for automated quality checking
- Validates the installation with comprehensive tests
- Sets up development tools (pytest, black, flake8, mypy)

### Manual Setup Procedure

If you prefer manual setup or need to customize the environment:

```bash
# Create virtual environment
python -m venv plume-nav-env
source plume-nav-env/bin/activate  # Linux/macOS
plume-nav-env\Scripts\activate     # Windows

# Upgrade pip and install build tools
pip install --upgrade pip
pip install build wheel

# Install project in development mode
pip install -e .
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install --install-hooks
pre-commit install --hook-type pre-push
pre-commit install --hook-type commit-msg

# Validate installation
python scripts/validate_installation.py
```

### System Requirements

- **Python**: 3.10, 3.11, 3.12, or 3.13
- **Operating System**: 
  - **Primary Support**: Linux, macOS
  - **Community Support**: Windows (PRs accepted, not officially supported)
- **Memory**: Minimum 4GB RAM (8GB+ recommended for performance testing)
- **Dependencies**: 
  - `gymnasium>=0.29.0` - RL environment framework
  - `numpy>=2.1.0` - Mathematical computing
  - `matplotlib>=3.9.0` - Visualization (optional for rgb_array mode)

### Installation Validation

After setup, verify your installation:

```bash
# Run comprehensive validation
python scripts/validate_installation.py

# Run quick test suite
python -m pytest tests/ -x -q --tb=short -m "not slow"

# Verify environment creation
python -c "import gymnasium as gym; import plume_nav_sim; env = gym.make('PlumeNav-StaticGaussian-v0'); print('✓ Environment creation successful')"
```

## Repository Layout and Public API

Understanding the repository layout and public API surface will help you place new code in the right location and design contributions that are easy for other researchers to use.

### High-level layout

- **Installable package**: `plume_nav_sim` (source under `src/backend/plume_nav_sim`).
- **Configs**: `src/backend/conf/` for Hydra/YAML configuration files; `plume_nav_sim.config` for typed configuration and composition helpers.
- **Scenarios and benchmarks**: `src/backend/scenarios/` for built-in scenario and benchmark definitions.
- **Documentation**: `src/backend/docs/` for user/developer documentation, contracts, and data guides.
- **Examples and notebooks**: `src/backend/examples/` and `notebooks/` for usage examples and exploratory analysis.
- **Tests**: `src/backend/tests/` for unit, integration, contract, and performance tests.
- **Vendored dependencies**: `src/backend/vendor/` for vendored shims (e.g., `gymnasium_vendored`, `psutil`).

### Public API surface

For most users and external researchers, the supported public surface is:

- `plume_nav_sim` package-level API via `plume_nav_sim.__init__`:
  - `make_env` – recommended way to create environments.
  - Core types, constants, and metadata exported via `__all__` (e.g., `GridSize`, `EnvironmentConfig`, `DEFAULT_*`, `ENVIRONMENT_ID`).
  - `get_package_info` and `initialize_package` for metadata and legacy bootstrap.
- `plume_nav_sim.config` and `plume_nav_sim.config.composition`:
  - Typed configuration and composition helpers such as `SimulationSpec`, `PolicySpec`, and composition utilities (e.g., `prepare`).
- `plume_nav_sim.compose.*`:
  - Backwards-compatibility shim re-exporting the modern configuration/composition API. New code should prefer imports from `plume_nav_sim.config` and `plume_nav_sim.config.composition`.

When you design a new feature that should be usable by downstream researchers, prefer to expose it via these modules or through clearly documented extension points below.

### Extension points for new contributions

Use these namespaces when extending plume-nav-sim:

- **Environment implementations and registration**:
  - `plume_nav_sim.envs` – environment classes and factories.
  - `plume_nav_sim.registration` – Gymnasium registration, `ENV_ID`, and helpers like `ensure_registered`.
- **Policies and control logic**:
  - `plume_nav_sim.policies` – built-in policies and policy helpers.
- **Plume models and concentration fields**:
  - `plume_nav_sim.plume` – plume model implementations (e.g., static Gaussian) and related utilities.
- **Rendering and visualization**:
  - `plume_nav_sim.render` – rendering utilities, colormaps, and templates.
- **Data capture and datasets**:
  - `plume_nav_sim.data_capture` – runtime capture pipeline, recorders, and validation.
  - `plume_nav_sim.media` – dataset manifests, metadata, and validation utilities.
  - `plume_nav_sim.video` – video plume dataset schema and I/O helpers.

Modules outside these areas (`utils`, `io`, `storage`, `data_formats`, `vendor`, etc.) are primarily internal infrastructure. If you are unsure where a new contribution belongs, open an issue or draft PR describing the proposed change and we can help place it appropriately.

## Contribution Workflow

### Git Workflow and Branching Strategy

We follow a **feature branch workflow** with the following conventions:

1. **Fork the Repository**: Create your own fork for contributions
2. **Create Feature Branches**: Use descriptive branch names following the pattern:
   ```bash
   # Feature development
   git checkout -b feature/your-feature-name
   
   # Bug fixes
   git checkout -b fix/issue-description
   
   # Documentation updates
   git checkout -b docs/update-description
   
   # Performance improvements
   git checkout -b perf/optimization-description
   ```

3. **Commit Standards**: Follow conventional commit format:
   ```bash
   # Format: type(scope): description
   git commit -m "feat(env): add goal radius configuration parameter"
   git commit -m "fix(render): resolve matplotlib backend fallback issue"
   git commit -m "docs(api): update environment initialization examples"
   git commit -m "test(integration): add cross-platform reproducibility tests"
   ```

4. **Pre-commit Quality Checks**: All commits automatically run:
   - **Black**: Code formatting (line length 88)
   - **isort**: Import sorting and organization
   - **flake8**: Linting and style checking
   - **mypy**: Static type checking
   - **bandit**: Security vulnerability scanning
   - **pytest**: Quick test subset execution

### Pull Request Guidelines

#### Before Submitting a Pull Request

1. **Sync with Main Branch**:
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-feature-branch
   git rebase main
   ```

2. **Run Comprehensive Tests**:
   ```bash
   # Full test suite with performance benchmarks
   python scripts/run_tests.py --category all --coverage --performance
   
   # Ensure >95% test coverage
   python -m pytest --cov=plume_nav_sim --cov-report=term-missing tests/
   ```

3. **Validate Code Quality**:
   ```bash
   # Run all pre-commit hooks
   pre-commit run --all-files
   
   # Type checking
   mypy src/plume_nav_sim/
   
   # Security scanning
   bandit -r src/plume_nav_sim/
   ```

### Internal Task Tracking (bd beads)

Maintainers and internal contributors track work using `bd` (beads):

- Check ready work: `bd ready --json`
- Claim/update: `bd update <id> --status in_progress --json`
- Create: `bd create "Title" -t bug|feature|task -p 0-4 --json`
- Link discovered work: `bd create "Found bug" -p 1 --deps discovered-from:<parent-id> --json`
- Close: `bd close <id> --reason "Completed" --json`

Beads auto-sync to `.beads/issues.jsonl` alongside the code. External users should continue using GitHub Issues/Discussions; maintainers will mirror/import into beads when appropriate.

#### Pull Request Template

When creating a pull request, please include:

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] All existing tests pass
- [ ] Added tests for new functionality
- [ ] Performance benchmarks pass
- [ ] Reproducibility tests pass

## Checklist
- [ ] Code follows style guidelines (pre-commit hooks pass)
- [ ] Self-review of code completed
- [ ] Code is commented, particularly in hard-to-understand areas
- [ ] Corresponding documentation updates made
- [ ] Changes generate no new warnings

## Scientific Impact
Describe any impact on research reproducibility, performance, or API compatibility.
```

#### Review Process

1. **Automated Checks**: CI/CD pipeline runs comprehensive testing
2. **Code Review**: Maintainers review for:
   - Code quality and consistency
   - Test coverage and completeness
   - Documentation accuracy
   - Performance impact
   - Scientific rigor and reproducibility
3. **Integration Testing**: Cross-platform compatibility validation
4. **Merge Requirements**: 
   - All tests passing
   - Approved review from maintainer
   - Conflicts resolved
   - Documentation updated

## Code Quality Standards

### Automated Quality Enforcement

All code must pass automated quality checks configured in `.pre-commit-config.yaml`:

#### Code Formatting
- **Black**: Python code formatter (line length: 88 characters)
  ```bash
  black --line-length=88 --target-version=py310 src/
  ```

- **isort**: Import sorting with Black compatibility
  ```bash
  isort --profile=black --line-length=88 --multi-line=3 src/
  ```

#### Linting and Style
- **flake8**: Comprehensive linting with plugins
  ```bash
  flake8 --max-line-length=88 --extend-ignore=E203,W503,E501 --max-complexity=10 src/
  ```
  - Enforced rules: PEP 8 compliance, complexity limits, docstring validation
  - Excluded errors: Black-incompatible formatting rules

#### Type Checking
- **mypy**: Static type analysis with strict configuration
  ```bash
  mypy --config-file=mypy.ini --show-error-codes src/plume_nav_sim/
  ```
  - Required for all public APIs
  - Type hints for function signatures and class definitions
  - Generic types for NumPy arrays and Gymnasium spaces

#### Security Scanning
- **bandit**: Security vulnerability detection
  ```bash
  bandit -r src/plume_nav_sim/ -f json -o bandit-report.json -ll
  ```
  - Scans for common security issues
  - Excludes test assertions and development code

### Coding Style Requirements

#### Python Style Guidelines

1. **PEP 8 Compliance**: Follow Python Enhancement Proposal 8
2. **Function and Variable Naming**: Use `snake_case` for functions and variables
3. **Class Naming**: Use `PascalCase` for class names
4. **Constants**: Use `UPPER_SNAKE_CASE` for module-level constants
5. **Private Methods**: Prefix with single underscore `_private_method`

#### Example Code Style

```python
"""Module docstring following NumPy style."""

import numpy as np  # >=2.1.0 - Mathematical computing
from typing import Tuple, Optional, Dict, Any

from plume_nav_sim.core.geometry import Coordinates
from plume_nav_sim.core.enums import Action


class PlumeSearchEnvironment:
    """Plume navigation environment following Gymnasium API.
    
    This class implements a reinforcement learning environment for chemical
    plume navigation research with static Gaussian plume distribution.
    
    Attributes:
        grid_size: Grid dimensions as (width, height) tuple
        source_location: Plume source coordinates
        _agent_position: Private agent position state
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (128, 128),
        source_location: Optional[Coordinates] = None,
        random_seed: Optional[int] = None
    ) -> None:
        """Initialize plume navigation environment.
        
        Args:
            grid_size: Environment grid dimensions
            source_location: Plume source position, defaults to grid center
            random_seed: Random seed for reproducible episodes
            
        Raises:
            ValueError: If grid_size dimensions are invalid
            TypeError: If source_location is not Coordinates type
        """
        # Validate input parameters
        if not all(dim > 0 for dim in grid_size):
            raise ValueError(f"Invalid grid dimensions: {grid_size}")
        
        self.grid_size = grid_size
        self.source_location = source_location or self._get_default_source()
        self._agent_position: Optional[Coordinates] = None
        
        # Initialize seeding utilities
        self._setup_random_generator(random_seed)
    
    def _get_default_source(self) -> Coordinates:
        """Calculate default source location at grid center."""
        center_x = self.grid_size[0] // 2
        center_y = self.grid_size[1] // 2
        return Coordinates(x=center_x, y=center_y)
```

#### Performance-Critical Code Guidelines

For performance-sensitive operations:

```python
def sample_concentration(self, position: Coordinates) -> float:
    """Sample concentration at agent position with <0.1ms target latency.
    
    Args:
        position: Agent coordinates for concentration sampling
        
    Returns:
        Concentration value in range [0.0, 1.0]
    """
    # Use NumPy vectorized operations for performance
    dx = position.x - self.source_location.x
    dy = position.y - self.source_location.y
    distance_squared = dx * dx + dy * dy
    
    # Direct mathematical computation avoiding function calls
    concentration = np.exp(-distance_squared / (2.0 * self.sigma_squared))
    
    return min(1.0, max(0.0, concentration))  # Clamp to valid range
```

### Pre-commit Hook Setup

Pre-commit hooks automatically enforce code quality:

```yaml
# .pre-commit-config.yaml (excerpt)
repos:
  - repo: https://github.com/psf/black
    rev: '24.2.0'
    hooks:
      - id: black
        args: ['--line-length=88', '--target-version=py310']
  
  - repo: https://github.com/PyCQA/flake8
    rev: '7.0.0'
    hooks:
      - id: flake8
        args: ['--max-line-length=88', '--extend-ignore=E203,W503']
```

To run pre-commit hooks manually:
```bash
# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
pre-commit run mypy --all-files
```

## Testing Requirements

### Comprehensive Test Coverage Standards

We maintain **>95% test coverage** across all components with comprehensive validation:

#### Test Categories and Requirements

1. **Unit Tests**: Component-level testing with isolated validation
   ```bash
   # Run unit tests only
   python scripts/run_tests.py --category unit --coverage
   
   # Individual component testing
   python -m pytest tests/unit/test_plume_model.py -v
   ```

2. **Integration Tests**: Cross-component and API compliance testing
   ```bash
   # Full integration test suite
   python scripts/run_tests.py --category integration --verbose
   
   # Gymnasium API compliance
   python -m pytest tests/integration/test_gymnasium_api.py -v
   ```

3. **Performance Tests**: Latency and resource usage benchmarks
   ```bash
   # Performance benchmarks
   python scripts/run_tests.py --category performance --strict-timing
   
   # Individual performance targets
   python -m pytest tests/performance/ -v --benchmark-only
   ```

4. **Reproducibility Tests**: Deterministic behavior validation
   ```bash
   # Cross-session reproducibility
   python scripts/run_tests.py --category reproducibility --seeds-validation
   
   # Specific reproducibility validation
   python -m pytest tests/reproducibility/test_seeding.py -v
   ```

#### Test Configuration and Fixtures

Our comprehensive test infrastructure provides:

```python
# Example test using provided fixtures
import pytest
from plume_nav_sim.envs.plume_search_env import PlumeSearchEnv


def test_environment_step_performance(performance_test_env, performance_tracker):
    """Test environment step latency meets <1ms target."""
    env = performance_test_env
    
    # Use performance tracker for comprehensive monitoring
    measurement_id = performance_tracker.start_measurement("environment_step")
    
    # Execute performance-critical operation
    obs, info = env.reset()
    action = env.action_space.sample()
    result = env.step(action)
    
    # Complete measurement and validate targets
    metrics = performance_tracker.end_measurement(measurement_id)
    assert metrics['duration_ms'] < 1.0, f"Step latency {metrics['duration_ms']:.3f}ms exceeds target"


def test_reproducibility_with_seeding(reproducibility_test_env, test_seeds):
    """Validate deterministic behavior across episodes."""
    env = reproducibility_test_env
    seed = test_seeds['reproducibility_seeds'][0]
    
    # Run first episode
    obs1, _ = env.reset(seed=seed)
    observations1 = [obs1]
    
    for _ in range(10):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        observations1.append(obs)
    
    # Run second episode with same seed
    obs2, _ = env.reset(seed=seed)
    observations2 = [obs2]
    
    for _ in range(10):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        observations2.append(obs)
    
    # Validate identical sequences
    for obs1, obs2 in zip(observations1, observations2):
        np.testing.assert_array_equal(obs1, obs2, 
                                     "Reproducibility failure: observations differ")
```

#### Performance Testing Standards

All performance tests must validate against these targets:

| Operation | Target Latency | Memory Limit | Test Method |
|-----------|----------------|--------------|-------------|
| Environment Step | <1ms | N/A | Direct timing with `time.perf_counter()` |
| Episode Reset | <10ms | N/A | Initialization timing including RNG setup |
| RGB Rendering | <5ms | <1MB | Frame generation with memory monitoring |
| Human Rendering | <50ms | <10MB | Interactive display with backend fallback |
| Plume Generation | <10ms | <40MB | Field computation for 128×128 grid |

### Test Writing Guidelines

#### Unit Test Structure

```python
"""Test module following our standardized structure."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from plume_nav_sim.core.plume_model import StaticGaussianPlume
from plume_nav_sim.core.geometry import Coordinates


class TestStaticGaussianPlume:
    """Test suite for StaticGaussianPlume component."""
    
    def test_initialization_with_valid_parameters(self):
        """Test plume initialization with valid configuration."""
        # Arrange
        source = Coordinates(x=64, y=64)
        sigma = 12.0
        grid_size = (128, 128)
        
        # Act
        plume = StaticGaussianPlume(
            source_location=source,
            sigma=sigma,
            grid_size=grid_size
        )
        
        # Assert
        assert plume.source_location == source
        assert plume.sigma == sigma
        assert plume.grid_size == grid_size
        assert plume.concentration_field is not None
        assert plume.concentration_field.shape == grid_size
    
    def test_concentration_sampling_accuracy(self):
        """Test concentration value accuracy at known positions."""
        # Arrange
        source = Coordinates(x=10, y=10)
        sigma = 5.0
        plume = StaticGaussianPlume(source, sigma, (20, 20))
        
        # Act & Assert - Source position should have maximum concentration
        source_concentration = plume.sample_concentration(source)
        assert abs(source_concentration - 1.0) < 1e-10
        
        # Position at distance sigma should have concentration ≈ 0.6065
        test_position = Coordinates(x=15, y=10)  # Distance = sigma
        concentration = plume.sample_concentration(test_position)
        expected = np.exp(-0.5)  # e^(-0.5) ≈ 0.6065
        assert abs(concentration - expected) < 1e-6
    
    @pytest.mark.parametrize("invalid_position", [
        Coordinates(x=-1, y=0),
        Coordinates(x=0, y=-1),
        Coordinates(x=128, y=64),
        Coordinates(x=64, y=128)
    ])
    def test_boundary_condition_handling(self, invalid_position):
        """Test plume behavior at grid boundaries."""
        plume = StaticGaussianPlume(
            Coordinates(x=64, y=64), 
            sigma=12.0, 
            grid_size=(128, 128)
        )
        
        # Should handle boundary conditions gracefully
        with pytest.raises(IndexError):
            plume.sample_concentration(invalid_position)
```

#### Integration Test Structure

```python
"""Integration tests for Gymnasium API compliance."""

import gymnasium as gym
import pytest
import numpy as np

import plume_nav_sim  # Registers environments


class TestGymnasiumAPICompliance:
    """Test suite for Gymnasium interface compliance."""
    
    @pytest.fixture
    def environment(self):
        """Create test environment instance."""
        env = gym.make('PlumeNav-StaticGaussian-v0')
        yield env
        env.close()
    
    def test_reset_method_compliance(self, environment):
        """Test reset method returns proper 2-tuple."""
        result = environment.reset()
        
        # Should return (observation, info) tuple
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        observation, info = result
        assert observation is not None
        assert isinstance(info, dict)
        
        # Observation should match observation space
        assert environment.observation_space.contains(observation)
    
    def test_step_method_compliance(self, environment):
        """Test step method returns proper 5-tuple."""
        environment.reset()
        action = environment.action_space.sample()
        
        result = environment.step(action)
        
        # Should return (obs, reward, terminated, truncated, info) tuple
        assert isinstance(result, tuple)
        assert len(result) == 5
        
        obs, reward, terminated, truncated, info = result
        
        # Validate return types
        assert environment.observation_space.contains(obs)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_action_space_validation(self, environment):
        """Test action space validation and error handling."""
        environment.reset()
        
        # Valid actions should work
        for valid_action in range(4):
            result = environment.step(valid_action)
            assert len(result) == 5
        
        # Invalid actions should raise ValueError
        invalid_actions = [-1, 4, 10, "up", None]
        for invalid_action in invalid_actions:
            with pytest.raises((ValueError, TypeError)):
                environment.step(invalid_action)
```

### Coverage Requirements

Maintain >95% test coverage with these commands:

```bash
# Generate coverage report
python -m pytest --cov=plume_nav_sim --cov-report=html --cov-report=term-missing tests/

# Coverage by component
python -m pytest --cov=plume_nav_sim.envs --cov-report=term tests/unit/envs/
python -m pytest --cov=plume_nav_sim.core --cov-report=term tests/unit/core/

# Fail if coverage below threshold
python -m pytest --cov=plume_nav_sim --cov-fail-under=95 tests/
```

## Documentation Standards

### API Documentation Requirements

All public APIs must include comprehensive docstrings following **NumPy style**:

```python
def reset(
    self,
    *,
    seed: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Reset environment to initial state for new episode.
    
    Initializes agent position, resets episode counters, and samples initial
    concentration observation. Supports deterministic seeding for reproducible
    research workflows.
    
    Parameters
    ----------
    seed : int, optional
        Random seed for reproducible episode initialization. If None, uses
        existing random number generator state. Must be non-negative integer.
    options : dict, optional
        Additional options for environment reset. Currently unused but
        maintained for Gymnasium API compatibility.
    
    Returns
    -------
    observation : np.ndarray
        Initial concentration observation at agent starting position.
        Shape: (1,) with dtype float32, values in range [0.0, 1.0].
    info : dict
        Episode information dictionary containing:
        - 'agent_position': tuple of (x, y) coordinates
        - 'step_count': int, always 0 for reset
        - 'distance_to_source': float, Euclidean distance to plume source
    
    Raises
    ------
    ValueError
        If seed is negative integer or grid configuration is invalid.
    TypeError
        If seed is not integer or None.
    
    Notes
    -----
    This method implements the Gymnasium environment reset specification,
    returning a 2-tuple of (observation, info). The agent is positioned
    randomly within grid boundaries, excluding the source location.
    
    Examples
    --------
    >>> env = gym.make('PlumeNav-StaticGaussian-v0')
    >>> obs, info = env.reset(seed=42)
    >>> print(f"Initial concentration: {obs[0]:.3f}")
    Initial concentration: 0.234
    >>> print(f"Agent position: {info['agent_position']}")
    Agent position: (23, 45)
    
    Scientific reproducibility requires consistent seeding:
    
    >>> # Reproducible episodes
    >>> obs1, _ = env.reset(seed=123)
    >>> obs2, _ = env.reset(seed=123)
    >>> np.array_equal(obs1, obs2)  # True
    """
```

#### Type Hints and Annotations

All functions must include comprehensive type hints:

```python
from typing import Tuple, Optional, Dict, Any, Union, List
import numpy as np
from numpy.typing import NDArray

# Type aliases for clarity
Observation = NDArray[np.float32]
Action = int
Info = Dict[str, Any]
StepResult = Tuple[Observation, float, bool, bool, Info]

def step(self, action: Action) -> StepResult:
    """Environment step with full type annotations."""
    pass

def render(self, mode: str = "rgb_array") -> Optional[NDArray[np.uint8]]:
    """Rendering with proper return type annotation."""
    pass
```

### Code Comment Standards

#### Inline Comments for Complex Logic

```python
def calculate_reward(self, agent_position: Coordinates) -> float:
    """Calculate distance-based reward for current agent position."""
    # Calculate Euclidean distance to source using vectorized operations
    # for sub-millisecond performance in tight training loops
    dx = agent_position.x - self.source_location.x
    dy = agent_position.y - self.source_location.y
    distance = np.sqrt(dx * dx + dy * dy)
    
    # Binary reward system: +1.0 for goal achievement, 0.0 otherwise
    # Goal radius of 0 requires exact source position for termination
    if distance <= self.goal_radius:
        return 1.0
    
    return 0.0
```

#### Module-Level Documentation

```python
"""Plume navigation environment implementation.

This module provides the core PlumeSearchEnv class implementing the Gymnasium
environment interface for chemical plume navigation research. The environment
features a static Gaussian plume model with discrete agent movement and
binary reward structure optimized for reinforcement learning research.

Key Components:
    PlumeSearchEnv: Main environment class with Gymnasium API compliance
    create_plume_search_env: Factory function for environment creation
    
Performance Characteristics:
    Step Latency: <1ms per environment step
    Memory Usage: <50MB for default 128x128 grid
    Rendering: <5ms RGB array generation, <50ms interactive display
    
Scientific Features:
    Deterministic Seeding: Reproducible episodes for research validation
    Cross-Platform Consistency: Identical behavior across operating systems
    Performance Benchmarking: Built-in timing and resource monitoring
    
Example Usage:
    >>> import gymnasium as gym
    >>> import plume_nav_sim
    >>> env = gym.make('PlumeNav-StaticGaussian-v0')
    >>> obs, info = env.reset(seed=42)
    >>> obs, reward, terminated, truncated, info = env.step(0)

Author: plume-nav-sim development team
License: MIT
Version: 0.1.0 (Proof-of-Life)
"""
```

### Example Script Guidelines

Example scripts must demonstrate:

1. **Basic Usage**: Simple environment interaction
2. **Research Workflow**: Complete training loop example
3. **Visualization**: Both rendering modes demonstration
4. **Performance Analysis**: Benchmarking and optimization

```python
"""Basic usage example for plume-nav-sim environment.

This script demonstrates the essential workflow for using plume-nav-sim
in reinforcement learning research, including environment creation,
episode execution, and performance monitoring.
"""

import time
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Register plume-nav-sim environments
import plume_nav_sim

def basic_usage_example():
    """Demonstrate basic environment usage with performance monitoring."""
    print("=== Plume Navigation Environment - Basic Usage ===\n")
    
    # Create environment with performance tracking
    env = gym.make('PlumeNav-StaticGaussian-v0')
    print(f"Environment created: {env.spec.id}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Episode with reproducible seeding
    start_time = time.perf_counter()
    obs, info = env.reset(seed=42)
    reset_time = time.perf_counter() - start_time
    
    print(f"\nEpisode initialized in {reset_time*1000:.2f}ms")
    print(f"Initial concentration: {obs[0]:.4f}")
    print(f"Agent position: {info['agent_position']}")
    print(f"Distance to source: {info['distance_to_source']:.2f}")
    
    # Execute episode with step timing
    episode_rewards = []
    step_times = []
    
    for step in range(100):
        action = env.action_space.sample()  # Random action
        
        step_start = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.perf_counter() - step_start
        
        step_times.append(step_time * 1000)  # Convert to milliseconds
        episode_rewards.append(reward)
        
        if terminated or truncated:
            print(f"\nEpisode completed at step {step + 1}")
            print(f"Final reward: {reward}")
            print(f"Termination reason: {'Goal reached' if terminated else 'Step limit'}")
            break
    
    # Performance analysis
    avg_step_time = np.mean(step_times)
    max_step_time = np.max(step_times)
    total_reward = sum(episode_rewards)
    
    print(f"\n=== Performance Metrics ===")
    print(f"Average step time: {avg_step_time:.3f}ms")
    print(f"Maximum step time: {max_step_time:.3f}ms")
    print(f"Performance target (<1ms): {'✓ PASS' if avg_step_time < 1.0 else '✗ FAIL'}")
    print(f"Total episode reward: {total_reward}")
    
    # Visualization demonstration
    print(f"\n=== Visualization Demonstration ===")
    
    # RGB array rendering
    rgb_start = time.perf_counter()
    rgb_array = env.render(mode="rgb_array")
    rgb_time = time.perf_counter() - rgb_start
    
    print(f"RGB array generation: {rgb_time*1000:.2f}ms")
    print(f"RGB array shape: {rgb_array.shape}")
    print(f"Render target (<5ms): {'✓ PASS' if rgb_time*1000 < 5.0 else '✗ FAIL'}")
    
    # Human mode rendering (if matplotlib available)
    try:
        human_start = time.perf_counter()
        env.render(mode="human")
        human_time = time.perf_counter() - human_start
        print(f"Human mode rendering: {human_time*1000:.2f}ms")
        plt.pause(0.1)  # Brief display
        plt.close('all')  # Clean up
    except Exception as e:
        print(f"Human rendering unavailable: {e}")
    
    env.close()
    print(f"\n=== Example completed successfully ===")

if __name__ == "__main__":
    basic_usage_example()
```

## Scientific Reproducibility Standards

### Deterministic Seeding Requirements

All research using plume-nav-sim must maintain reproducibility through proper seeding:

#### Reproducible Episode Generation

```python
"""Reproducibility validation example."""

import numpy as np
import gymnasium as gym
import plume_nav_sim

def validate_reproducibility():
    """Demonstrate and validate reproducible behavior."""
    env = gym.make('PlumeNav-StaticGaussian-v0')
    
    # Reproducibility test with fixed seed
    SEED = 12345
    
    # First run
    obs1, info1 = env.reset(seed=SEED)
    trajectory1 = [obs1]
    
    for _ in range(10):
        action = 0  # Fixed action sequence
        obs, _, _, _, _ = env.step(action)
        trajectory1.append(obs)
    
    # Second run with identical seed
    obs2, info2 = env.reset(seed=SEED)
    trajectory2 = [obs2]
    
    for _ in range(10):
        action = 0  # Identical action sequence
        obs, _, _, _, _ = env.step(action)
        trajectory2.append(obs)
    
    # Validate identical results
    for i, (obs1, obs2) in enumerate(zip(trajectory1, trajectory2)):
        np.testing.assert_array_equal(
            obs1, obs2, 
            err_msg=f"Reproducibility failure at step {i}"
        )
    
    print("✓ Reproducibility validation passed")
    
    # Cross-session consistency validation
    env.close()
    
    # Create new environment instance
    env2 = gym.make('PlumeNav-StaticGaussian-v0')
    obs3, info3 = env2.reset(seed=SEED)
    
    # Should match first observation from previous sessions
    np.testing.assert_array_equal(
        obs1, obs3,
        err_msg="Cross-session reproducibility failure"
    )
    
    print("✓ Cross-session consistency validated")
    env2.close()
```

#### Performance Benchmarking Standards

Research contributions must include performance validation:

```python
"""Performance benchmarking for scientific validation."""

import time
import statistics
import numpy as np
import gymnasium as gym
import plume_nav_sim

def benchmark_environment_performance():
    """Benchmark environment performance against scientific targets."""
    env = gym.make('PlumeNav-StaticGaussian-v0')
    
    # Warm-up phase
    env.reset()
    for _ in range(10):
        env.step(env.action_space.sample())
    
    # Step latency benchmark
    print("=== Step Latency Benchmark ===")
    step_times = []
    
    obs, info = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()
        
        start_time = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.perf_counter() - start_time
        
        step_times.append(step_time * 1000)  # Convert to milliseconds
        
        if terminated or truncated:
            obs, info = env.reset()
    
    # Statistical analysis
    mean_latency = statistics.mean(step_times)
    median_latency = statistics.median(step_times)
    p95_latency = np.percentile(step_times, 95)
    p99_latency = np.percentile(step_times, 99)
    
    print(f"Mean step latency: {mean_latency:.3f}ms")
    print(f"Median step latency: {median_latency:.3f}ms")
    print(f"95th percentile: {p95_latency:.3f}ms")
    print(f"99th percentile: {p99_latency:.3f}ms")
    
    # Validate against targets
    TARGET_LATENCY = 1.0  # 1ms target
    
    print(f"\n=== Performance Validation ===")
    print(f"Target: <{TARGET_LATENCY}ms per step")
    print(f"Mean performance: {'✓ PASS' if mean_latency < TARGET_LATENCY else '✗ FAIL'}")
    print(f"95th percentile: {'✓ PASS' if p95_latency < TARGET_LATENCY else '✗ FAIL'}")
    
    # Memory usage benchmark
    print(f"\n=== Memory Usage Benchmark ===")
    import psutil
    
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Create multiple environments to test memory scaling
    environments = []
    for i in range(5):
        env_instance = gym.make('PlumeNav-StaticGaussian-v0')
        env_instance.reset()
        environments.append(env_instance)
    
    peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    memory_per_env = (peak_memory - initial_memory) / 5
    
    print(f"Memory per environment: {memory_per_env:.1f}MB")
    print(f"Target: <50MB per environment")
    print(f"Memory efficiency: {'✓ PASS' if memory_per_env < 50 else '✗ FAIL'}")
    
    # Cleanup
    for env_instance in environments:
        env_instance.close()
    env.close()
```

#### Research Workflow Integration

```python
"""Research workflow integration example."""

def research_workflow_example():
    """Demonstrate integration with research methodology."""
    import json
    from datetime import datetime
    
    # Research configuration
    research_config = {
        'experiment_name': 'plume_navigation_baseline',
        'environment': 'PlumeNav-StaticGaussian-v0',
        'random_seed': 42,
        'episodes': 100,
        'max_steps_per_episode': 1000,
        'performance_targets': {
            'step_latency_ms': 1.0,
            'memory_usage_mb': 50.0
        },
        'reproducibility_validation': True
    }
    
    # Environment setup
    env = gym.make(research_config['environment'])
    
    # Data collection
    experiment_data = {
        'config': research_config,
        'timestamp': datetime.now().isoformat(),
        'episodes': [],
        'performance_metrics': {
            'step_times': [],
            'memory_usage': [],
            'reproducibility_checks': []
        }
    }
    
    print(f"Starting research experiment: {research_config['experiment_name']}")
    
    for episode in range(research_config['episodes']):
        episode_data = {
            'episode': episode,
            'seed': research_config['random_seed'] + episode,
            'steps': 0,
            'total_reward': 0,
            'trajectory': []
        }
        
        # Reproducibility check every 10 episodes
        if episode % 10 == 0:
            obs1, _ = env.reset(seed=episode_data['seed'])
            obs2, _ = env.reset(seed=episode_data['seed'])
            
            reproducible = np.array_equal(obs1, obs2)
            experiment_data['performance_metrics']['reproducibility_checks'].append(reproducible)
            
            if not reproducible:
                print(f"⚠ Reproducibility failure at episode {episode}")
        
        obs, info = env.reset(seed=episode_data['seed'])
        episode_data['trajectory'].append({
            'step': 0,
            'observation': obs.tolist(),
            'agent_position': info['agent_position'],
            'distance_to_source': info['distance_to_source']
        })
        
        for step in range(research_config['max_steps_per_episode']):
            action = env.action_space.sample()  # Replace with your algorithm
            
            # Performance monitoring
            start_time = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            step_time = time.perf_counter() - start_time
            
            experiment_data['performance_metrics']['step_times'].append(step_time * 1000)
            
            episode_data['steps'] = step + 1
            episode_data['total_reward'] += reward
            episode_data['trajectory'].append({
                'step': step + 1,
                'action': action,
                'observation': obs.tolist(),
                'reward': reward,
                'terminated': terminated,
                'truncated': truncated,
                'info': info
            })
            
            if terminated or truncated:
                break
        
        experiment_data['episodes'].append(episode_data)
        
        if episode % 10 == 0:
            print(f"Episode {episode}: {episode_data['steps']} steps, "
                  f"reward {episode_data['total_reward']}")
    
    # Performance analysis
    step_times = experiment_data['performance_metrics']['step_times']
    mean_step_time = statistics.mean(step_times)
    reproducibility_rate = statistics.mean(experiment_data['performance_metrics']['reproducibility_checks'])
    
    print(f"\n=== Research Results ===")
    print(f"Average step time: {mean_step_time:.3f}ms")
    print(f"Reproducibility rate: {reproducibility_rate:.1%}")
    print(f"Total episodes: {len(experiment_data['episodes'])}")
    
    # Save research data
    with open(f"{research_config['experiment_name']}_data.json", 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    print(f"Research data saved to {research_config['experiment_name']}_data.json")
    
    env.close()
```

## Issue Reporting Guidelines

### Bug Report Template

When reporting bugs, please use this template to provide comprehensive information:

```markdown
# Bug Report

## Environment Information
- **OS**: [e.g., Ubuntu 22.04, macOS 13.0, Windows 11]
- **Python Version**: [e.g., 3.10.12]
- **plume-nav-sim Version**: [e.g., 0.1.0]
- **Gymnasium Version**: [e.g., 0.29.1]
- **NumPy Version**: [e.g., 2.1.0]
- **Matplotlib Version**: [e.g., 3.9.0] (if rendering issue)

## Bug Description
**Clear and concise description of the bug:**

**Expected Behavior:**
What should happen?

**Actual Behavior:**
What actually happens?

## Reproduction Steps
```python
# Minimal code to reproduce the issue
import gymnasium as gym
import plume_nav_sim

env = gym.make('PlumeNav-StaticGaussian-v0')
# ... reproduction steps
```

## Error Output
```
# Full error traceback and output
```

## Performance Impact
- [ ] Causes performance degradation
- [ ] Breaks reproducibility
- [ ] Affects rendering
- [ ] Memory leak suspected
- [ ] Cross-platform compatibility issue

## Additional Context
Any additional information, screenshots, or context about the issue.

## Reproducibility Validation
- [ ] Bug reproduces consistently
- [ ] Bug reproduces across different seeds
- [ ] Bug reproduces on different systems
- [ ] Minimal reproduction case provided
```

### Feature Request Procedures

For new features, please provide:

```markdown
# Feature Request

## Feature Summary
Brief description of the proposed feature.

## Scientific Motivation
How does this feature advance plume navigation research?

## Proposed Implementation
High-level description of how this could be implemented.

## API Design
```python
# Proposed API usage
env = gym.make('PlumeNav-StaticGaussian-v0', new_feature_param=value)
result = env.new_method()
```

## Performance Considerations
Expected impact on:
- Step latency
- Memory usage
- Reproducibility
- Cross-platform compatibility

## Breaking Changes
- [ ] This is a breaking change
- [ ] Backward compatibility maintained
- [ ] New optional parameters only

## Research Impact
How will this feature enable new research capabilities?

## Alternative Solutions
Other approaches considered and why this is preferred.
```

### Support and Discussion Channels

For questions and discussion:

1. **GitHub Issues**: Bug reports and feature requests
2. **GitHub Discussions**: General questions and research discussion
3. **Scientific Community**: Research methodology and algorithmic questions

**Before Creating Issues:**
1. Search existing issues for duplicates
2. Check the documentation and examples
3. Test with the latest version
4. Provide minimal reproduction cases

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Our community standards include:

#### Expected Behavior

- **Professional Communication**: Use clear, respectful language in all interactions
- **Constructive Feedback**: Focus on code and ideas, not individuals
- **Scientific Rigor**: Support claims with evidence and reproducible examples
- **Collaborative Spirit**: Help others learn and contribute effectively
- **Inclusive Language**: Use terminology accessible to international researchers

#### Unacceptable Behavior

- Harassment, discrimination, or personal attacks
- Unprofessional language or inappropriate content
- Spam, advertising, or off-topic discussions
- Sharing others' private information without permission
- Any behavior that would be inappropriate in a professional academic setting

### Communication Standards

#### Issue and PR Comments

```markdown
# Good Examples

## Constructive Code Review
"The implementation looks solid! I noticed the step latency might be improved 
by caching the concentration field. Here's a suggested optimization..."

## Helpful Bug Report
"I can reproduce this issue on Ubuntu 22.04 with Python 3.11. The error 
occurs specifically when the agent reaches grid boundaries. Minimal reproduction 
code attached."

## Scientific Discussion
"This approach aligns well with the reinforcement learning literature. Have you 
considered how this affects sample efficiency? Reference: [paper citation]"
```

```markdown
# Avoid

## Non-constructive Criticism
"This code is terrible and makes no sense."

## Vague Reports
"It doesn't work on my machine."

## Demanding Language
"Fix this immediately!" or "This should be obvious!"
```

#### Research-Focused Communication

Since plume-nav-sim supports scientific research, maintain academic standards:

- **Cite relevant literature** when discussing algorithmic approaches
- **Provide quantitative evidence** for performance claims
- **Share reproducible examples** with your contributions
- **Acknowledge limitations** and potential improvements
- **Respect intellectual property** and give appropriate credit

### Recognition and Attribution

#### Contributor Recognition

We recognize contributions through:

- **Contributors list** in README.md and documentation
- **Release notes** highlighting significant contributions
- **Academic citations** for research-impacting contributions
- **Community highlights** for exceptional help and support

#### Attribution Standards

When contributing:

1. **Original Work**: Ensure your contributions are your own or properly attributed
2. **Algorithm Citations**: Reference academic papers for implemented algorithms
3. **Code Attribution**: Credit third-party code snippets or inspirations
4. **Data Sources**: Acknowledge any external data or benchmarks used

#### Academic Citation

If plume-nav-sim contributes to your research, please cite:

```bibtex
@software{plume_nav_sim_2025,
  title = {plume-nav-sim: A Gymnasium Environment for Plume Navigation Research},
  version = {0.0.1},
  url = {https://github.com/SamuelBrudner/plume_nav_sim},
  year = {2025},
  note = {Proof-of-Life Implementation}
}
```

## Release Procedures

### Version Management

We follow **Semantic Versioning (SemVer)** with scientific software considerations:

- **MAJOR.MINOR.PATCH** format (e.g., 0.1.0)
- **Major**: Breaking changes affecting reproducibility or API compatibility
- **Minor**: New features maintaining backward compatibility
- **Patch**: Bug fixes and performance improvements

#### Version Compatibility Matrix

| Version | Python | Gymnasium | NumPy | Breaking Changes |
|---------|---------|-----------|-------|------------------|
| 0.1.0 | 3.10-3.13 | >=0.29.0 | >=2.1.0 | Initial release |
| 0.2.x | 3.10-3.13 | >=0.29.0 | >=2.1.0 | New features only |
| 1.0.x | TBD | TBD | TBD | Production release |

### Changelog Maintenance

All releases require comprehensive changelog entries:

```markdown
# Changelog

## [0.1.1] - 2024-XX-XX

### Added
- New configuration parameters for plume dispersion
- Cross-platform reproducibility validation
- Performance benchmarking utilities

### Changed
- Improved step latency by 15% through NumPy optimization
- Enhanced error messages for invalid actions
- Updated documentation with research workflow examples

### Fixed
- Matplotlib backend fallback on headless systems
- Memory leak in rendering cleanup
- Reproducibility issue with random seed initialization

### Performance
- Step latency: 0.8ms → 0.7ms (12% improvement)
- Memory usage: 45MB → 42MB (7% reduction)
- Test suite execution: 25s → 20s (20% faster)

### Scientific Impact
- Maintains backward compatibility for all research workflows
- No changes to default environment behavior
- All reproducibility guarantees preserved

### Migration Guide
No breaking changes in this release. Optional new parameters available.
```

### Release Validation Procedures

Before each release:

#### 1. Automated Testing
```bash
# Full test suite with all categories
python scripts/run_tests.py --category all --coverage --performance --strict

# Cross-platform validation (CI/CD)
# - Linux: Ubuntu 20.04, 22.04
# - macOS: 11, 12, 13
# - Windows: 10, 11 (community support)

# Python version matrix
# - 3.10, 3.11, 3.12, 3.13
```

#### 2. Performance Benchmarking
```bash
# Performance regression testing
python scripts/benchmark_performance.py --baseline-version=0.1.0

# Memory usage validation
python scripts/memory_profiling.py --episodes=1000

# Reproducibility verification
python scripts/validate_reproducibility.py --seeds=10 --cross-platform
```

#### 3. Documentation Review
```bash
# API documentation completeness
python scripts/check_documentation_coverage.py

# Example script validation
python scripts/validate_examples.py --all

# Changelog completeness check
python scripts/check_changelog_completeness.py
```

#### 4. Scientific Validation
```bash
# Research workflow compatibility
python scripts/validate_research_workflows.py

# Reproducibility across versions
python scripts/cross_version_reproducibility.py

# Performance comparison
python scripts/compare_version_performance.py
```

### Pre-release Process

1. **Feature Freeze**: Complete all planned features
2. **Testing Phase**: 1 week of comprehensive testing
3. **Documentation Review**: Update all documentation
4. **Community Testing**: Beta release for community validation
5. **Performance Validation**: Benchmark against previous versions
6. **Release Candidate**: Final testing with release artifacts
7. **Release**: Tagged release with complete documentation

### Release Artifacts

Each release includes:

- **Source Distribution**: `plume-nav-sim-X.Y.Z.tar.gz`
- **Wheel Package**: `plume_nav_sim-X.Y.Z-py3-none-any.whl`
- **Documentation**: Complete API and user documentation
- **Examples**: Updated example scripts and tutorials
- **Benchmarks**: Performance and reproducibility validation results
- **Migration Guide**: Breaking changes and upgrade instructions

## Troubleshooting

### Common Development Issues

#### Environment Setup Problems

**Issue**: `python scripts/setup_dev_env.py` fails with dependency conflicts

**Solution**:
```bash
# Clean environment setup
rm -rf plume-nav-env/
python -m venv plume-nav-env
source plume-nav-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e . --no-cache-dir
```

**Issue**: Pre-commit hooks fail after installation

**Solution**:
```bash
# Reinstall pre-commit hooks
pre-commit uninstall
pre-commit clean
pre-commit install --install-hooks
pre-commit run --all-files
```

#### Testing Issues

**Issue**: Tests fail with "Environment not found" error

**Solution**:
```bash
# Ensure environment is registered
python -c "import plume_nav_sim; print('Registration successful')"

# Check Gymnasium registration
python -c "import gymnasium as gym; print(gym.envs.registration.registry.env_specs.keys())"
```

**Issue**: Performance tests fail on slower systems

**Solution**:
```bash
# Run with relaxed timing
python scripts/run_tests.py --category performance --timing-tolerance=2.0

# Skip performance tests
python -m pytest tests/ -m "not performance"
```

#### Import and Module Issues

**Issue**: `ImportError: No module named 'plume_nav_sim'`

**Solution**:
```bash
# Reinstall in development mode
pip install -e .

# Verify installation
pip list | grep plume-nav-sim
python -c "import plume_nav_sim; print(plume_nav_sim.__version__)"
```

#### Rendering Problems

**Issue**: Matplotlib rendering fails in headless environment

**Solution**:
```python
# Force non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Or use rgb_array mode only
env.render(mode='rgb_array')  # Instead of 'human'
```

**Issue**: "Qt platform plugin" error on Linux

**Solution**:
```bash
# Install system dependencies
sudo apt-get install python3-tk  # Ubuntu/Debian
sudo yum install tkinter         # CentOS/RHEL

# Or force Agg backend
export MPLBACKEND=Agg
```

### Performance Troubleshooting

#### Slow Environment Steps

**Diagnosis**:
```python
import time
import gymnasium as gym
import plume_nav_sim

env = gym.make('PlumeNav-StaticGaussian-v0')
env.reset()

# Profile step performance
times = []
for _ in range(1000):
    start = time.perf_counter()
    env.step(env.action_space.sample())
    times.append(time.perf_counter() - start)

print(f"Mean: {np.mean(times)*1000:.3f}ms")
print(f"95th percentile: {np.percentile(times, 95)*1000:.3f}ms")
```

**Solutions**:
- Check Python version (3.10+ recommended)
- Update NumPy to latest version
- Disable debug logging
- Use `rgb_array` mode instead of `human` for automated runs

#### Memory Issues

**Diagnosis**:
```python
import psutil
import gymnasium as gym
import plume_nav_sim

process = psutil.Process()
print(f"Initial memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")

env = gym.make('PlumeNav-StaticGaussian-v0')
env.reset()
print(f"After env creation: {process.memory_info().rss / 1024 / 1024:.1f} MB")

# Run episode
for _ in range(1000):
    env.step(env.action_space.sample())
    
print(f"After episode: {process.memory_info().rss / 1024 / 1024:.1f} MB")
env.close()
```

**Solutions**:
- Explicitly call `env.close()` after use
- Reduce grid size for memory-constrained systems
- Check for matplotlib memory leaks with `plt.close('all')`

### Getting Help

#### Self-Help Resources

1. **Documentation**: Check API documentation and examples
2. **Test Suite**: Run `python scripts/run_tests.py --help` for diagnostic options
3. **Validation Script**: Use `python scripts/validate_installation.py --verbose`

#### Community Support

1. **GitHub Issues**: For bugs and feature requests
2. **GitHub Discussions**: For general questions and research topics
3. **Code Review**: Submit draft PRs for early feedback

#### Reporting Unresolved Issues

When reporting issues that aren't covered here:

1. **System Information**: Include OS, Python version, dependency versions
2. **Reproduction Case**: Minimal code that demonstrates the issue
3. **Error Output**: Complete error messages and stack traces
4. **Attempted Solutions**: What troubleshooting steps you've tried
5. **Research Context**: How the issue affects your research workflow

---

## Thank You for Contributing!

Your contributions help advance scientific research in plume navigation and reinforcement learning. Whether you're fixing bugs, adding features, improving documentation, or supporting other researchers, every contribution makes a difference.

For questions about these guidelines or the contribution process, please open a discussion on GitHub or contact the maintainers.

**Happy coding and researching!** 🧪🤖

## Vendored Code Namespace

To keep first-party code clearly separated from copied third-party shims, all
vendored modules live under a dedicated `vendor/` namespace within the backend
source tree.

- Location: `src/backend/vendor/`
- Current entries: `vendor.gymnasium_vendored`, `vendor.psutil` (lightweight, test-only)
- Rationale: Avoid polluting top-level with third-party names and make policy explicit

Guidelines:

- Prefer real third-party dependencies in application code. Use `vendor.*` shims
  only when (a) tests require minimal functionality without the dependency, or
  (b) upstream packaging is unsuitable for our constrained CI environments.
- When adding/adjusting a shim, place it under `src/backend/vendor/<name>/` and
  document the subset implemented and intended scope (tests vs. runtime).
- Backward-compatibility shims may exist at the old import paths (e.g., a minimal
  `src/backend/psutil/__init__.py` re-exporting `vendor.psutil`) to avoid wide import
  churn. New code should import from `vendor.<name>` directly.
- If a shim mirrors a real dependency (e.g., `psutil`), ensure the real package can
  still be used when installed (typically via optional extras) and that tests behave
  correctly with either implementation.

This policy was adopted following the evaluation in plume_nav_sim-84 to group and
minimize shims while preserving test ergonomics.
