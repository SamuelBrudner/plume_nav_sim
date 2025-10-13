# Changelog

All notable changes to the `plume_nav_sim` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- DI environment id `PlumeNav-Components-v0` with registration-time kwargs mapping (source_location→goal_location) and factory-backed entry point.
- Env-var opt-in to make DI the default for the legacy env id: set `PLUMENAV_DEFAULT=components` (or `PLUMENAV_USE_COMPONENTS=1`).
- Helper `ensure_component_env_registered()` to register the DI environment id programmatically.
- Registration tests covering DI mappings, combinations, and gym.make() smoke checks.

### Changed
- Legacy `PlumeSearchEnv` remains the default, but registration now logs an INFO hint about upcoming deprecation and DI opt-in options.

### Todo
- PyPI distribution and release automation
- Performance optimization for larger grid sizes (>256×256)
- Multi-agent environment support
- Dynamic plume models with temporal evolution
- FAIR data persistence module
- Integration with specialized robotics simulation frameworks

## [0.0.1] - 2024-12-19

### Added

#### Core Environment Implementation
- **PlumeSearchEnv**: Complete Gymnasium-compatible RL environment for plume navigation
- **Static Gaussian Plume Model**: Mathematical implementation with configurable dispersion parameters
- **Discrete Action Space**: Cardinal direction navigation (UP=0, RIGHT=1, DOWN=2, LEFT=3)
- **Single-Agent Navigation**: Grid-based movement with boundary enforcement
- **Sparse Reward System**: Binary reward structure (+1.0 at source, 0.0 elsewhere)

#### Dual-Mode Rendering System
- **RGB Array Mode**: NumPy-based programmatic visualization for automated analysis
- **Human Mode**: Interactive matplotlib visualization for research and debugging
- **Visual Elements**: Red agent markers (3×3 pixels), white source markers (5×5 pixels)
- **Concentration Visualization**: Grayscale heatmaps with normalized [0,1] scaling
- **Backend Compatibility**: Automatic matplotlib backend selection with graceful fallback

#### Reproducibility Framework
- **Deterministic Seeding**: Complete episode reproducibility with identical seeds
- **Random State Management**: Centralized random number generation using gymnasium.utils.seeding
- **Agent Placement**: Seeded random start positions excluding source location
- **Validation Testing**: Comprehensive reproducibility test suite ensuring deterministic behavior

#### System Architecture
- **Modular Component Design**: Clear separation between environment, plume model, rendering, and utilities
- **Type Safety**: Comprehensive type definitions with Action, RenderMode, Coordinates, GridSize enums/dataclasses
- **Error Handling**: Hierarchical exception system with specific error types and recovery suggestions
- **Performance Optimization**: <1ms step latency, <5ms RGB rendering, <50ms human visualization

#### API Compliance & Registration
- **Gymnasium Integration**: Full compliance with 5-tuple step() and reset() API methods
- **Environment Registration**: Standard gym.make() support with "PlumeNav-StaticGaussian-v0" ID
- **Action/Observation Spaces**: Discrete(4) actions, Box observation space for concentration values
- **Render Modes**: Support for both 'rgb_array' and 'human' visualization modes

#### Testing & Validation Framework
- **Comprehensive Test Suite**: 100% API compliance testing with pytest framework
- **Performance Benchmarks**: Automated timing and memory usage validation
- **Integration Tests**: Cross-component validation and external dependency testing
- **Edge Case Coverage**: Boundary conditions, error scenarios, and fallback behavior testing

#### Modern Python Packaging
- **pyproject.toml Configuration**: Modern build system using hatchling backend
- **Dependency Management**: Minimal dependencies (gymnasium>=0.29.0, numpy>=2.1.0, matplotlib>=3.9.0)
- **Python Support**: Compatible with Python 3.10, 3.11, 3.12, and 3.13
- **Development Tools**: Black, flake8, mypy, pytest integration with comprehensive configuration

#### Example Scripts & Documentation
- **Basic Usage Examples**: Quick start guides and fundamental usage patterns
- **Visualization Demos**: Interactive plotting and programmatic analysis examples
- **Reproducibility Demonstrations**: Seeding examples and deterministic behavior validation
- **Performance Benchmarks**: Timing analysis and memory usage profiling scripts

#### Configuration & Customization
- **Environment Parameters**: Configurable grid size (default 128×128), source location, sigma values
- **Performance Tuning**: Memory limits (50MB total), timing targets, optimization thresholds
- **Validation Constants**: Parameter bounds, precision values, error message templates
- **Testing Constants**: Test-optimized parameters for faster test execution

---

## Technical Specifications

### Environment Defaults
- **DEFAULT_GRID_SIZE**: [128, 128]
- **MIN_GRID_SIZE**: [1, 1]
- **DEFAULT_SOURCE_LOCATION**: [64, 64]
- **DEFAULT_PLUME_SIGMA**: 12.0
- **DEFAULT_GOAL_RADIUS**: 1.1920929e-07 (float32 epsilon)
- **DEFAULT_MAX_STEPS**: 1000

### Performance Targets
- **Environment Step Latency**: <1ms
- **RGB Array Rendering**: <5ms
- **Human Mode Rendering**: <50ms
- **Memory Usage**: <50MB
- **Plume Generation**: <10ms (for 128×128 grid)

### Supported Platforms
- **Linux**: Full support
- **macOS**: Full support
- **Windows**: Limited support

## Dependencies

### Required
- `gymnasium>=0.29.0`
- `numpy>=2.1.0`
- `matplotlib>=3.9.0`

### Optional
#### Development
- `pytest>=8.0.0`
- `pytest-cov>=4.0.0`
- `black>=24.0.0`

#### Testing
- `pytest-benchmark>=4.0.0`
- `pytest-mock>=3.12.0`

## Development Infrastructure

- **Build System**: Modern hatchling build backend following PEP 517/518
- **Code Quality**: Black formatting, flake8 linting, mypy type checking
- **Testing**: pytest with coverage reporting and performance benchmarking
- **Documentation**: Comprehensive README with API reference and examples

## Known Limitations

- Static plume model only (no temporal dynamics)
- Single-agent environments only
- Local installation only (no PyPI distribution)
- Limited to discrete action spaces
- No continuous control or complex sensor models

## Research Applications

- Algorithm development and benchmarking for plume navigation
- Reinforcement learning research in olfactory robotics
- Educational demonstrations of RL concepts
- Scientific workflow integration with NumPy/matplotlib ecosystem
- Reproducible research with deterministic episode generation

---

## Development Notes

### Versioning Strategy
This project follows [Semantic Versioning](https://semver.org/): MAJOR for incompatible API changes, MINOR for new functionality, PATCH for bug fixes and performance improvements.

### Environment Versioning
Following Gymnasium conventions: environment IDs include version suffixes (e.g., '-v0'). Version numbers increase when changes might impact learning results.

### Future Release Planning
- **v0.1.x**: Enhanced features, performance optimizations, additional examples
- **v0.2.x**: Multi-agent support, dynamic plume models, advanced sensor models
- **v1.0.x**: Production-ready release with comprehensive features and PyPI distribution

### Contributing
See CONTRIBUTING.md for guidelines on development environment setup, testing, code style, and pull request workflow.

### License
MIT License (see LICENSE file for details)

### Citation
```bibtex
@software{plume_nav_sim_2024,
  title={plume-nav-sim: Gymnasium Environment for Plume Navigation Research},
  author={plume_nav_sim Development Team},
  year={2024},
  version={0.0.1},
  url={https://github.com/plume-nav-sim/plume_nav_sim}
}
```

### Contact
For questions, bug reports, or feature requests, please visit the GitHub Issues page: https://github.com/plume-nav-sim/plume_nav_sim/issues

---

*This changelog is automatically updated with each release and follows the principles of keeping a changelog for better project transparency and user communication.*
### Breaking Changes (planned v2.0.0)
- Observation structure (public wrapper): now a Dict with keys `agent_position`, `sensor_reading`, `source_location` (was Box(shape=(1,))).
- Info dictionary: exposes `seed` on reset; step info requires `step_count`, `total_reward`, `goal_reached`; optional `episode_count` may be present.
- Goal radius: `DEFAULT_GOAL_RADIUS` changed to float32 epsilon; `goal_radius=0` normalized to epsilon automatically.
- Grid minimum: `MIN_GRID_SIZE` relaxed to `[1, 1]` for testing flexibility.
- Seeding: `env.seed(seed)` is deprecated; wrapper provides a compatibility shim that calls `reset(seed=...)`.
