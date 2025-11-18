"""
Comprehensive developer guide documentation module for plume_nav_sim providing technical implementation
guidance, system architecture analysis, development workflows, contribution guidelines, and advanced
usage patterns for developers, contributors, and researchers extending the plume navigation reinforcement
learning environment. Serves as the central technical resource combining API reference, implementation
details, performance optimization, testing strategies, and extension patterns.

This module implements a complete technical documentation system including DeveloperGuideGenerator for
structured content creation, ArchitectureAnalyzer for system analysis, PerformanceBenchmarker for
optimization guidance, TestingFramework for quality assurance documentation, and ExtensionManager for
customization patterns. The implementation provides comprehensive coverage of development workflows,
architectural decisions, performance optimization, and extension strategies.

Key Features:
- Comprehensive developer guide generation with architecture analysis and technical workflows
- System architecture documentation with component relationships and design patterns
- Development environment setup with virtual environment and tool configuration
- Contribution guidelines with code standards and review processes
- Performance optimization guidance with benchmarking and profiling techniques
- Testing framework documentation with unit, integration, and performance testing
- Extension and customization patterns for research-specific modifications
- Advanced development patterns with research workflow integration

Technical Design:
- Modular documentation architecture with specialized generators and analyzers
- Integration with environment components for live analysis and examples
- Performance-focused documentation with <1ms step latency optimization guidance
- Scientific reproducibility emphasis with seeding and validation procedures
- Research workflow templates for academic and industrial applications
- Extensibility documentation for multi-agent and dynamic plume future development

Architecture Integration:
- Deep integration with PlumeSearchEnv for implementation examples and performance analysis
- Component analysis using BaseEnvironment, StaticGaussianPlume, and rendering systems
- Development workflow coordination with testing utilities and validation frameworks
- Extension pattern documentation for custom plume models and rendering backends
- Research integration templates for scientific Python ecosystem and RL frameworks
"""

import ast  # >=3.10 - Abstract syntax tree analysis for code quality tools and static analysis examples
import inspect  # >=3.10 - Code introspection for automated documentation generation and development tool integration
import json  # >=3.10 - Configuration management and serialization examples for development workflows
import logging  # >=3.10 - Development logging patterns, debugging strategies, and monitoring implementation guidance
import textwrap  # >=3.10 - Documentation formatting utilities for generating readable developer documentation
import time  # >=3.10 - Performance timing and profiling for development optimization examples and benchmarking
import unittest  # >=3.10 - Testing framework integration for development testing workflow examples and test writing guidance
from pathlib import (
    Path,  # >=3.10 - Path operations for development workflow documentation and file system examples
)
from typing import (  # >=3.10 - Advanced type hints for comprehensive type safety documentation and development guidance
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

# External imports with version requirements for development and documentation
import gymnasium as gym  # >=0.29.0 - Reinforcement learning framework for understanding API compliance and environment development patterns
import matplotlib.pyplot as plt  # >=3.9.0 - Visualization framework for rendering system development and debugging visualization examples
import numpy as np  # >=2.1.0 - Mathematical operations, array processing, and performance optimization for technical implementation examples
import pytest  # >=8.0 - Primary testing framework for development workflow documentation and testing strategy guidance

from ..examples.basic_usage import (  # Basic usage demonstration for developer onboarding and understanding user workflows
    demonstrate_basic_episode,
    demonstrate_reproducibility,
)
from ..plume_nav_sim.core.constants import (  # System constants for development examples and configuration understanding
    DEFAULT_GRID_SIZE,
    DEFAULT_SOURCE_LOCATION,
)
from ..plume_nav_sim.core.types import (  # Core type system for understanding data structures and type safety in development
    Action,
    Coordinates,
    EnvironmentConfig,
    GridSize,
)
from ..plume_nav_sim.envs.base_env import (
    BaseEnvironment,  # Base environment architecture for understanding extension patterns and custom environment development
)

# Internal imports - Core environment and components for technical implementation examples
from ..plume_nav_sim.envs.plume_search_env import (  # Main environment class for technical implementation examples and performance analysis
    PlumeSearchEnv,
    create_plume_search_env,
)
from ..plume_nav_sim.plume.static_gaussian import (
    StaticGaussianPlume,  # Plume model implementation for understanding mathematical components and custom plume development
)

# Internal imports - API documentation and user guide integration
from .api_reference import (  # API documentation generation for technical reference integration in developer workflows
    APIDocumentationGenerator,
    generate_api_documentation,
)
from .troubleshooting import (  # Troubleshooting utilities for development environment setup and debugging
    TroubleshootingGuide,
    run_full_diagnostic,
)
from .user_guide import (  # User guide generation for understanding user-facing features that developers must maintain
    create_quick_start_tutorial,
    generate_user_guide,
)

# Global constants for developer guide configuration
DEVELOPER_GUIDE_VERSION = '2.0.0'  # Version identifier for developer guide documentation and compatibility tracking
DEVELOPMENT_PYTHON_VERSIONS = ['3.10', '3.11', '3.12', '3.13']  # Supported Python versions for development environment setup
PERFORMANCE_TARGETS = {  # Performance targets for development optimization guidance and quality assurance
    'step_latency_ms': 1.0,     # Target step execution time for real-time training compatibility
    'memory_usage_mb': 50.0,    # Target memory usage for efficient resource utilization
    'render_time_ms': 5.0       # Target rendering time for interactive visualization
}
CODE_QUALITY_STANDARDS = {  # Code quality standards for contribution guidelines and automated validation
    'test_coverage_min': 95,     # Minimum test coverage percentage for quality assurance
    'type_hint_coverage': 100,   # Required type hint coverage for code maintainability
    'docstring_coverage': 90     # Required docstring coverage for documentation completeness
}
DEVELOPMENT_WORKFLOWS = {}  # Dictionary storing development workflow templates and procedures
EXTENSION_PATTERNS = {}  # Dictionary storing extension and customization patterns for advanced development
ARCHITECTURE_DIAGRAMS = {}  # Dictionary storing system architecture diagrams and component relationships
PERFORMANCE_BENCHMARKS = {}  # Dictionary storing performance benchmarking data and optimization guidance

# Module exports for comprehensive developer guide functionality
__all__ = [
    'generate_developer_guide',         # Main function for generating comprehensive developer guide with technical documentation
    'create_architecture_documentation',  # Creates detailed system architecture documentation with component analysis
    'create_development_setup_guide',   # Creates development environment setup guide with tools and workflows
    'create_contribution_guidelines',   # Creates contribution guidelines with standards and processes
    'create_performance_guide',         # Creates performance optimization guide with benchmarking and profiling
    'create_testing_guide',             # Creates comprehensive testing framework guide with procedures and standards
    'create_extension_guide',           # Creates extension and customization guide with patterns and examples
    'DeveloperGuideGenerator',          # Comprehensive developer guide generator for structured documentation
    'ArchitectureAnalyzer',             # System architecture analysis utility for component discovery and mapping
    'PerformanceBenchmarker',           # Performance benchmarking utility for optimization guidance and bottleneck identification
    'TestingFramework',                 # Testing framework documentation utility for quality assurance procedures
    'ExtensionManager'                  # Extension management utility for customization patterns and advanced development
]


def generate_developer_guide(
    output_format: Optional[str] = 'markdown',
    include_architecture_diagrams: bool = True,
    include_performance_analysis: bool = True,
    include_extension_examples: bool = True,
    output_directory: Optional[Path] = None,
    generation_options: dict = None
) -> str:
    """
    Main function for generating comprehensive developer guide documentation including system architecture
    analysis, development workflows, contribution guidelines, performance optimization, testing strategies,
    and extension patterns for technical contributors and advanced users.

    This function orchestrates the creation of a complete developer guide covering all aspects of the
    plume_nav_sim system from architectural understanding to advanced extension development. It integrates
    multiple specialized documentation components and provides comprehensive technical guidance.

    Args:
        output_format (Optional[str]): Output format for developer guide ('markdown', 'html', 'restructuredtext')
        include_architecture_diagrams (bool): Include component relationship diagrams and system architecture visualization
        include_performance_analysis (bool): Include performance benchmarking data and optimization recommendations
        include_extension_examples (bool): Include custom component implementations and integration patterns
        output_directory (Optional[Path]): Directory for output files and generated documentation assets
        generation_options (dict): Additional generation configuration options and customization settings

    Returns:
        str: Complete developer guide with technical documentation, workflows, and implementation guidance
        ready for distribution to technical contributors and system architects
    """
    # Initialize developer guide structure with version information and technical focus organization
    guide_sections = {
        'header': f"# Plume Navigation Simulation Developer Guide v{DEVELOPER_GUIDE_VERSION}",
        'table_of_contents': "",
        'introduction': "",
        'system_overview': "",
        'architecture_documentation': "",
        'development_setup': "",
        'contribution_guidelines': "",
        'performance_guide': "",
        'testing_guide': "",
        'extension_guide': "",
        'advanced_topics': "",
        'troubleshooting': "",
        'api_reference_integration': "",
        'appendices': ""
    }

    # Apply generation options and customization settings with developer-focused defaults
    if generation_options is None:
        generation_options = {}

    technical_depth = generation_options.get('technical_depth', 'advanced')
    include_code_examples = generation_options.get('include_code_examples', True)
    include_performance_tips = generation_options.get('include_performance_tips', True)
    target_audience = generation_options.get('target_audience', 'developers')

    try:
        # Generate comprehensive introduction covering system overview, architecture principles, and development philosophy
        guide_sections['introduction'] = f"""
## Introduction

Welcome to the comprehensive developer guide for plume_nav_sim! This guide is designed for technical
contributors, system architects, and researchers who want to understand, extend, or contribute to the
plume navigation simulation environment.

### Developer Guide Scope

This guide provides in-depth technical documentation for:

- **System Architects**: Understanding the modular layered architecture and component relationships
- **Core Contributors**: Contributing to the main codebase with quality standards and testing procedures
- **Extension Developers**: Creating custom plume models, rendering backends, and environment variants
- **Research Engineers**: Integrating with RL frameworks and building research-specific workflows
- **Performance Engineers**: Optimizing for <1ms step latency and memory efficiency requirements
- **Quality Engineers**: Implementing testing strategies and maintaining code quality standards

### Technical Philosophy

The plume_nav_sim system follows these core technical principles:

1. **Gymnasium API Compliance**: Full adherence to standard RL environment interfaces for ecosystem compatibility
2. **Scientific Reproducibility**: Deterministic behavior with comprehensive seeding and validation systems
3. **Performance First**: <1ms step latency targets with efficient NumPy-based mathematical operations
4. **Modular Architecture**: Component-based design enabling easy extension and customization
5. **Quality Assurance**: >95% test coverage with comprehensive validation and monitoring systems
6. **Research Focus**: Features and workflows optimized for scientific research and publication

### Development Context

**Current Status**: Proof-of-life implementation with static Gaussian plume model
**Target Applications**: Reinforcement learning research, educational applications, algorithm development
**Performance Targets**: {PERFORMANCE_TARGETS}
**Quality Standards**: {CODE_QUALITY_STANDARDS}

### Navigation Guide

This developer guide is organized progressively from system understanding to advanced development:

1. **System Overview**: High-level architecture and component relationships
2. **Architecture Deep-dive**: Detailed component analysis and design patterns
3. **Development Setup**: Environment configuration and tool installation
4. **Contribution Workflow**: Code standards, testing, and review processes
5. **Performance Optimization**: Benchmarking, profiling, and optimization techniques
6. **Testing Framework**: Comprehensive testing strategies and quality assurance
7. **Extension Development**: Custom components and research-specific modifications
8. **Advanced Topics**: Multi-agent considerations and future extensibility

Each section includes executable code examples, implementation details, and practical guidance for
productive development work.
        """

        # Generate system overview explaining the modular layered design and component-based patterns
        guide_sections['system_overview'] = f"""
## System Overview

### High-Level Architecture

plume_nav_sim employs a **modular layered architecture** designed around the Gymnasium reinforcement
learning framework. The system prioritizes simplicity, maintainability, and extensibility while
providing a foundation for research-grade plume navigation experiments.

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  RL Training Scripts │ Research Workflows │ User Examples   │
├─────────────────────────────────────────────────────────────┤
│                    Interface Layer                          │
│         Gymnasium API │ Registration │ Documentation        │
├─────────────────────────────────────────────────────────────┤
│                     Core Layer                              │
│    PlumeSearchEnv │ BaseEnvironment │ Action Processing     │
├─────────────────────────────────────────────────────────────┤
│                   Domain Layer                              │
│  Plume Models │ Reward Calculation │ State Management       │
├─────────────────────────────────────────────────────────────┤
│                Infrastructure Layer                         │
│   Rendering │ Seeding │ Validation │ Performance Monitoring │
└─────────────────────────────────────────────────────────────┘
```

### Module Map

The repository is organized to make it clear where public APIs live, where extension points are located, and which modules are internal infrastructure.

- **Top-level public API**: `plume_nav_sim.__init__`
  - Recommended entrypoint: `make_env(**kwargs)`.
  - Exposes core types and constants such as `GridSize`, `Coordinates`, `EnvironmentConfig`, `DEFAULT_*`, and `ENVIRONMENT_ID`.
  - Provides metadata helpers (`get_package_info`, `initialize_package`, `get_conf_dir`).

- **Configuration and composition**: `plume_nav_sim.config`
  - `plume_nav_sim.config.composition` defines typed specs and composition helpers (e.g., `SimulationSpec`, `PolicySpec`, and `prepare`).
  - Legacy imports from `plume_nav_sim.compose.*` are maintained as shims but new code should import from `plume_nav_sim.config`.

- **Environment implementations and registration**:
  - `plume_nav_sim.envs` contains environment classes (e.g., `PlumeSearchEnv`) and component-based envs.
  - `plume_nav_sim.registration` handles Gymnasium registration (`ENV_ID`, `ensure_registered`, `register_env`).

- **Domain components**:
  - `plume_nav_sim.plume` – plume models and concentration field logic (e.g., `StaticGaussianPlume`).
  - `plume_nav_sim.policies` – built-in policies and helpers.
  - `plume_nav_sim.actions`, `plume_nav_sim.observations`, `plume_nav_sim.rewards` – action/observation/reward components and contracts.

- **Rendering and visualization**: `plume_nav_sim.render`
  - Rendering pipeline, colormaps, templates, and utilities for `rgb_array` / `human` modes.

- **Data capture and datasets**:
  - `plume_nav_sim.data_capture` – runtime capture pipeline (JSONL.gz artifacts, validation, optional Parquet export).
  - `plume_nav_sim.media` – dataset manifests, provenance metadata, and dataset-level validation utilities.
  - `plume_nav_sim.video` – canonical video plume dataset schema and attribute validation.

- **Support and infrastructure modules** (primarily internal):
  - `plume_nav_sim.core` – fundamental types, constants, and helper functions shared across components.
  - `plume_nav_sim.utils`, `plume_nav_sim.io`, `plume_nav_sim.storage`, `plume_nav_sim.data_formats` – utilities, I/O helpers, storage policies, and format adapters.
  - `plume_nav_sim.logging` – logging configuration and loguru bootstrap.

When adding new functionality intended for reuse by other researchers, prefer to expose it either via the top-level `plume_nav_sim` package (for core functionality) or via the domain-specific subpackages above. Changes to support-only modules should keep their internal nature unless explicitly promoted and documented as part of the public API.

### Core Components

#### PlumeSearchEnv (Core Layer)
- **Purpose**: Main Gymnasium environment implementation and orchestration
- **Responsibilities**: Episode lifecycle, component coordination, API compliance
- **Key Methods**: `reset()`, `step()`, `render()`, `seed()`, `close()`
- **Performance**: <1ms step execution, efficient state management

#### StaticGaussianPlume (Domain Layer)
- **Purpose**: Mathematical plume model with Gaussian concentration distribution
- **Responsibilities**: Concentration field generation, observation sampling
- **Key Features**: Vectorized NumPy operations, O(1) concentration lookup
- **Extensibility**: Base for dynamic plume models and multi-source scenarios

#### Rendering Pipeline (Infrastructure Layer)
- **Purpose**: Dual-mode visualization supporting programmatic and interactive use
- **Modes**: `rgb_array` for ML workflows, `human` for interactive exploration
- **Performance**: <5ms RGB generation, matplotlib backend management
- **Customization**: Pluggable color schemes and visualization components

#### Seeding System (Infrastructure Layer)
- **Purpose**: Scientific reproducibility with deterministic episode generation
- **Features**: Comprehensive validation, episode-level seeding, state verification
- **Research Integration**: Experimental design support, statistical consistency
- **Quality Assurance**: Reproducibility testing and validation procedures

### Data Flow Architecture

```mermaid
graph TB
    A[RL Agent] --> B[PlumeSearchEnv.step]
    B --> C[Action Processor]
    C --> D[State Manager]
    D --> E[Boundary Enforcer]
    E --> F[Plume Model]
    F --> G[Reward Calculator]
    G --> H[Termination Checker]
    H --> I[Performance Monitor]
    I --> J[Return Observation]
    J --> A

    B --> K[Rendering Pipeline]
    K --> L[Visualization Output]

    style B fill:#e1f5fe
    style F fill:#f3e5f5
    style K fill:#e8f5e8
```

### Component Interaction Patterns

**Initialization Flow**:
1. Environment registration with Gymnasium
2. Component instantiation and validation
3. Plume field generation and caching
4. Rendering backend configuration
5. Seeding system initialization

**Episode Flow**:
1. `reset()` → State initialization and seed application
2. `step(action)` → Action validation and processing
3. Position update with boundary enforcement
4. Concentration sampling from plume model
5. Reward calculation and termination checking
6. Performance monitoring and logging
7. Optional rendering pipeline activation

**Extension Points**:
- **Custom Plume Models**: Extend `BasePlumeModel` for dynamic plumes
- **Rendering Backends**: Implement visualization interfaces for custom outputs
- **Action Spaces**: Modify action processing for multi-agent scenarios
- **Reward Functions**: Customize reward calculation for research objectives

### Performance Characteristics

The system is optimized for research-grade performance:

- **Step Latency**: <1ms average execution time for real-time training compatibility
- **Memory Efficiency**: <50MB for standard 128×128 grid configurations
- **Scalability**: Linear scaling with grid size, quadratic memory requirements
- **Rendering Performance**: <5ms RGB array generation, <50ms interactive updates

### Quality Assurance Framework

Comprehensive quality assurance with automated validation:

- **Unit Testing**: >95% code coverage with pytest framework integration
- **Integration Testing**: Component interaction validation and API compliance testing
- **Performance Testing**: Automated benchmarking and regression prevention
- **Reproducibility Testing**: Statistical validation of seeded episode consistency
- **Documentation Testing**: Code example execution and output validation
        """

        # Create architecture documentation using create_architecture_documentation with component analysis and interaction patterns
        guide_sections['architecture_documentation'] = create_architecture_documentation(
            include_component_diagrams=include_architecture_diagrams,
            include_performance_analysis=include_performance_analysis,
            include_design_rationale=True
        )

        # Generate development setup guide using create_development_setup_guide with environment configuration and tool setup
        guide_sections['development_setup'] = create_development_setup_guide(
            include_advanced_setup=True,
            include_ide_configuration=True,
            include_debugging_setup=True
        )

        # Create contribution guidelines using create_contribution_guidelines with workflow standards and code quality requirements
        guide_sections['contribution_guidelines'] = create_contribution_guidelines(
            include_workflow_diagrams=include_architecture_diagrams,
            include_code_examples=include_code_examples,
            include_review_criteria=True
        )

        # Generate performance guide using create_performance_guide with optimization strategies and benchmarking procedures
        guide_sections['performance_guide'] = create_performance_guide(
            include_benchmarking_tools=True,
            include_optimization_examples=include_extension_examples,
            include_profiling_analysis=include_performance_analysis
        )

        # Create testing guide using create_testing_guide with comprehensive testing frameworks and quality assurance procedures
        guide_sections['testing_guide'] = create_testing_guide(
            include_test_examples=include_code_examples,
            include_ci_configuration=True,
            include_coverage_analysis=include_performance_analysis
        )

        # Generate extension guide using create_extension_guide with customization patterns and advanced development techniques
        guide_sections['extension_guide'] = create_extension_guide(
            include_custom_examples=include_extension_examples,
            include_integration_patterns=True,
            include_research_extensions=True
        )

        # Include architecture diagrams if include_architecture_diagrams enabled with component relationships and data flow visualization
        if include_architecture_diagrams:
            guide_sections['architecture_diagrams'] = generate_architecture_diagrams()

        # Add performance analysis if include_performance_analysis enabled with benchmarking results and optimization recommendations
        if include_performance_analysis:
            guide_sections['performance_analysis'] = generate_performance_analysis()

        # Include extension examples if include_extension_examples enabled with custom component implementations and integration patterns
        if include_extension_examples:
            guide_sections['extension_examples'] = generate_extension_examples()

        # Generate troubleshooting section with development environment issues and debugging strategies
        guide_sections['troubleshooting'] = f"""
## Developer Troubleshooting Guide

### Common Development Issues

#### Environment Setup Problems

**Issue**: Import errors when importing plume_nav_sim components
```python
# Diagnostic script for import issues
import sys
import importlib.util

def diagnose_import_issues():
    \"\"\"Diagnose common import problems in development environment.\"\"\"

    print("Development Environment Diagnostics")
    print("=" * 40)

    # Check Python version
    print(f"Python version: {{sys.version}}")
    version_tuple = sys.version_info[:2]
    if version_tuple < (3, 10):
        print("❌ Python 3.10+ required for development")
        return False
    else:
        print("✅ Python version compatible")

    # Check development installation
    try:
        import plume_nav_sim
        print(f"✅ plume_nav_sim installed at: {{plume_nav_sim.__file__}}")
    except ImportError as e:
        print(f"❌ plume_nav_sim import failed: {{e}}")
        print("Solution: Run 'pip install -e .' from project root")
        return False

    # Check core dependencies
    dependencies = ['gymnasium', 'numpy', 'matplotlib']
    for dep in dependencies:
        try:
            spec = importlib.util.find_spec(dep)
            if spec is not None:
                print(f"✅ {{dep}} available")
            else:
                print(f"❌ {{dep}} missing")
        except Exception as e:
            print(f"❌ {{dep}} check failed: {{e}}")

    return True

# Run diagnostics
diagnose_import_issues()
```

**Solution**:
1. Ensure Python 3.10+ is installed
2. Install in development mode: `pip install -e .`
3. Verify virtual environment activation
4. Check PYTHONPATH includes project directory

#### Performance Testing Issues

**Issue**: Step latency exceeds 1ms target during development testing
```python
def debug_performance_issues():
    \"\"\"Debug and analyze performance bottlenecks.\"\"\"

    import time
    from ..plume_nav_sim.envs.plume_search_env import create_plume_search_env

    print("Performance Debugging Analysis")
    print("-" * 30)

    # Create test environment with profiling
    env = create_plume_search_env(grid_size=(128, 128))

    # Profile component initialization
    start_time = time.perf_counter()
    obs, info = env.reset(seed=42)
    init_time = (time.perf_counter() - start_time) * 1000

    print(f"Environment initialization: {{init_time:.2f}}ms")

    # Profile step execution with breakdown
    step_times = []
    for i in range(100):
        action = env.action_space.sample()

        step_start = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = (time.perf_counter() - step_start) * 1000
        step_times.append(step_time)

        if terminated or truncated:
            obs, info = env.reset(seed=42 + i)

    avg_step_time = np.mean(step_times)
    max_step_time = np.max(step_times)

    print(f"Average step time: {{avg_step_time:.3f}}ms")
    print(f"Maximum step time: {{max_step_time:.3f}}ms")
    print(f"Target compliance: {{'✅ PASS' if avg_step_time <= 1.0 else '❌ FAIL'}}")

    if avg_step_time > 1.0:
        print("\\nOptimization recommendations:")
        print("- Use smaller grid sizes during development")
        print("- Profile individual components with cProfile")
        print("- Check for debug logging overhead")
        print("- Verify NumPy is using optimized BLAS")

    env.close()

debug_performance_issues()
```

#### Testing Framework Issues

**Issue**: Tests failing due to non-deterministic behavior
```python
def debug_reproducibility_issues():
    \"\"\"Debug reproducibility and seeding problems.\"\"\"

    print("Reproducibility Debugging")
    print("-" * 25)

    from ..plume_nav_sim.utils.seeding import SeedManager

    # Test seed validation
    seed_manager = SeedManager()
    test_seeds = [42, -1, "invalid", 2**32]

    for seed in test_seeds:
        try:
            is_valid = seed_manager.validate_seed(seed)
            print(f"Seed {{seed}}: {{'Valid' if is_valid else 'Invalid'}}")
        except Exception as e:
            print(f"Seed {{seed}}: Error - {{e}}")

    # Test episode reproducibility
    env = create_plume_search_env()

    episodes = []
    for run in range(2):
        obs, info = env.reset(seed=42)
        episode_data = [obs[0]]

        for step in range(10):
            action = 0  # Deterministic action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_data.append(obs[0])

            if terminated or truncated:
                break

        episodes.append(episode_data)

    identical = episodes[0] == episodes[1]
    print(f"\\nEpisode reproducibility: {{'✅ PASS' if identical else '❌ FAIL'}}")

    if not identical:
        print("Reproducibility issues detected:")
        for i, (obs1, obs2) in enumerate(zip(episodes[0], episodes[1])):
            if obs1 != obs2:
                print(f"  Step {{i}}: {{obs1}} != {{obs2}}")

    env.close()

debug_reproducibility_issues()
```

### Development Best Practices

#### Code Quality Standards

1. **Type Hints**: 100% coverage for all public APIs
2. **Docstrings**: Google-style documentation for all functions and classes
3. **Testing**: >95% code coverage with comprehensive test cases
4. **Performance**: Profile critical paths and optimize for <1ms step latency
5. **Logging**: Use structured logging with appropriate levels

#### Testing Strategies

```python
# Example of comprehensive test structure
class TestEnvironmentDevelopment:
    \"\"\"Example test class showing development testing patterns.\"\"\"

    def test_environment_creation_performance(self):
        \"\"\"Test that environment creation meets performance targets.\"\"\"

        start_time = time.perf_counter()
        env = create_plume_search_env()
        creation_time = time.perf_counter() - start_time

        assert creation_time < 0.1, f"Environment creation too slow: {{creation_time:.3f}}s"
        env.close()

    def test_step_latency_compliance(self):
        \"\"\"Test that step execution meets <1ms target.\"\"\"

        env = create_plume_search_env()
        obs, info = env.reset(seed=42)

        step_times = []
        for i in range(1000):  # Statistical sample
            action = env.action_space.sample()

            start_time = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            step_time = time.perf_counter() - start_time
            step_times.append(step_time * 1000)  # Convert to ms

            if terminated or truncated:
                obs, info = env.reset(seed=42 + i)

        avg_step_time = np.mean(step_times)
        p99_step_time = np.percentile(step_times, 99)

        assert avg_step_time <= 1.0, f"Average step time exceeds target: {{avg_step_time:.3f}}ms"
        assert p99_step_time <= 5.0, f"P99 step time too high: {{p99_step_time:.3f}}ms"

        env.close()

    def test_reproducibility_validation(self):
        \"\"\"Test that seeded episodes are perfectly reproducible.\"\"\"

        env = create_plume_search_env()

        # Run identical episodes
        episodes = []
        for run in range(3):
            obs, info = env.reset(seed=12345)
            episode_data = []

            for step in range(50):
                action = (step % 4)  # Deterministic policy
                obs, reward, terminated, truncated, info = env.step(action)
                episode_data.append((obs[0], reward, terminated, truncated))

                if terminated or truncated:
                    break

            episodes.append(episode_data)

        # Verify all episodes are identical
        for i in range(1, len(episodes)):
            assert episodes[0] == episodes[i], f"Episode {{i}} differs from baseline"

        env.close()
```

#### Debugging Strategies

1. **Component Isolation**: Test individual components separately
2. **Performance Profiling**: Use cProfile for detailed analysis
3. **State Inspection**: Log intermediate states for debugging
4. **Reproducible Bugs**: Always include seed values in bug reports
5. **Integration Testing**: Test component interactions systematically

### Development Tool Integration

#### IDE Configuration

**VSCode settings.json**:
```json
{{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "editor.formatOnSave": true
}}
```

#### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.2.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
```
        """

        # Create advanced topics section covering research applications and production deployment considerations
        guide_sections['advanced_topics'] = f"""
## Advanced Development Topics

### Multi-Agent Architecture Considerations

While the current implementation focuses on single-agent scenarios, the architecture provides
foundation patterns for multi-agent extensions:

```python
# Conceptual multi-agent extension pattern
class MultiAgentPlumeSearchEnv(BaseEnvironment):
    \"\"\"
    Conceptual multi-agent environment extending current architecture.
    NOTE: This is architectural guidance, not current implementation.
    \"\"\"

    def __init__(self, num_agents: int = 2, agent_coordination: str = 'independent'):
        super().__init__()
        self.num_agents = num_agents
        self.agent_coordination = agent_coordination
        self.agents = {{}}

        # Multi-agent observation and action spaces
        self.observation_space = spaces.Dict({{
            f'agent_{{i}}': spaces.Box(0.0, 1.0, (3,), np.float32)  # [concentration, x_pos, y_pos]
            for i in range(num_agents)
        }})

        self.action_space = spaces.Dict({{
            f'agent_{{i}}': spaces.Discrete(4)
            for i in range(num_agents)
        }})

    def step(self, actions: Dict[str, int]):
        \"\"\"Process multi-agent actions with coordination handling.\"\"\"

        observations = {{}}
        rewards = {{}}
        terminated = {{}}
        truncated = {{}}
        infos = {{}}

        # Process each agent's action
        for agent_id, action in actions.items():
            obs, reward, term, trunc, info = self._step_single_agent(agent_id, action)

            observations[agent_id] = obs
            rewards[agent_id] = reward
            terminated[agent_id] = term
            truncated[agent_id] = trunc
            infos[agent_id] = info

        # Apply coordination effects
        if self.agent_coordination == 'cooperative':
            rewards = self._apply_cooperative_rewards(rewards, observations)
        elif self.agent_coordination == 'competitive':
            rewards = self._apply_competitive_rewards(rewards, observations)

        return observations, rewards, terminated, truncated, infos

    def _apply_cooperative_rewards(self, rewards, observations):
        \"\"\"Apply cooperative reward shaping based on agent proximity.\"\"\"
        # Implementation would consider agent coordination benefits
        return rewards

    def _apply_competitive_rewards(self, rewards, observations):
        \"\"\"Apply competitive reward structure with resource competition.\"\"\"
        # Implementation would handle competitive agent interactions
        return rewards
```

### Dynamic Plume Model Architecture

Extension patterns for dynamic and multi-source plume models:

```python
# Conceptual dynamic plume architecture
from abc import ABC, abstractmethod

class DynamicPlumeModel(ABC):
    \"\"\"
    Abstract base class for dynamic plume models.
    Extends current StaticGaussianPlume for temporal dynamics.
    \"\"\"

    def __init__(self, grid_size: GridSize, time_step: float = 0.1):
        self.grid_size = grid_size
        self.time_step = time_step
        self.current_time = 0.0
        self.concentration_field = None

    @abstractmethod
    def update_field(self, delta_time: float) -> np.ndarray:
        \"\"\"Update concentration field based on temporal dynamics.\"\"\"
        pass

    @abstractmethod
    def add_source(self, location: Coordinates, strength: float, start_time: float = 0.0):
        \"\"\"Add dynamic source with temporal activation.\"\"\"
        pass

    def step(self) -> np.ndarray:
        \"\"\"Advance plume model by one time step.\"\"\"
        self.current_time += self.time_step
        self.concentration_field = self.update_field(self.time_step)
        return self.concentration_field

class WindDrivenPlume(DynamicPlumeModel):
    \"\"\"Example dynamic plume with wind transport.\"\"\"

    def __init__(self, grid_size: GridSize, wind_velocity: Tuple[float, float] = (1.0, 0.0)):
        super().__init__(grid_size)
        self.wind_velocity = wind_velocity
        self.sources = []

    def update_field(self, delta_time: float) -> np.ndarray:
        \"\"\"Update field with wind-driven advection.\"\"\"

        # Initialize field
        field = np.zeros((self.grid_size.height, self.grid_size.width))

        # Add contribution from each active source
        for source in self.sources:
            if self.current_time >= source['start_time']:
                field += self._compute_wind_driven_concentration(source)

        return field.astype(np.float32)

    def _compute_wind_driven_concentration(self, source: dict) -> np.ndarray:
        \"\"\"Compute wind-driven Gaussian plume concentration.\"\"\"

        # Time since source activation
        active_time = self.current_time - source['start_time']

        # Wind-advected source location
        advected_x = source['location'].x + self.wind_velocity[0] * active_time
        advected_y = source['location'].y + self.wind_velocity[1] * active_time

        # Compute Gaussian plume with wind transport
        y, x = np.meshgrid(np.arange(self.grid_size.height), np.arange(self.grid_size.width), indexing='ij')

        # Distance calculations with wind direction consideration
        dx = x - advected_x
        dy = y - advected_y

        # Anisotropic dispersion (different spreading along/across wind)
        sigma_x = source['dispersion'] * (1 + 0.1 * active_time)  # Spreading over time
        sigma_y = source['dispersion'] * (1 + 0.05 * active_time)

        concentration = source['strength'] * np.exp(
            -(dx**2 / (2 * sigma_x**2) + dy**2 / (2 * sigma_y**2))
        )

        return concentration

    def add_source(self, location: Coordinates, strength: float, start_time: float = 0.0):
        \"\"\"Add wind-driven source to simulation.\"\"\"

        source = {{
            'location': location,
            'strength': strength,
            'start_time': start_time,
            'dispersion': 12.0  # Base dispersion parameter
        }}

        self.sources.append(source)
```

### Research Integration Patterns

#### Stable-Baselines3 Integration

```python
# Advanced SB3 integration with custom callbacks and monitoring
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

class PlumeNavigationCallback(BaseCallback):
    \"\"\"Custom callback for plume navigation training monitoring.\"\"\"

    def __init__(self, eval_env, eval_freq: int = 10000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.evaluation_results = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current policy
            success_rate = self._evaluate_policy()
            self.evaluation_results.append({{
                'timestep': self.n_calls,
                'success_rate': success_rate
            }})

            if self.verbose >= 1:
                print(f"Timestep {{self.n_calls}}: Success rate {{success_rate:.2%}}")

        return True

    def _evaluate_policy(self, num_episodes: int = 100) -> float:
        \"\"\"Evaluate current policy performance.\"\"\"

        successes = 0
        for episode in range(num_episodes):
            obs = self.eval_env.reset(seed=episode)
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated

                if terminated:
                    successes += 1

        return successes / num_episodes

# Research training workflow
def advanced_training_workflow():
    \"\"\"Comprehensive training workflow with monitoring and evaluation.\"\"\"

    # Create training and evaluation environments
    def make_env():
        return create_plume_search_env(
            grid_size=(64, 64),
            goal_radius=1.0,
            max_steps=200
        )

    train_env = DummyVecEnv([make_env])
    eval_env = make_env()

    # Create model with optimized hyperparameters
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./plume_nav_tensorboard/"
    )

    # Setup callback for evaluation and monitoring
    callback = PlumeNavigationCallback(eval_env, eval_freq=5000)

    # Train model with monitoring
    model.learn(
        total_timesteps=1000000,
        callback=callback,
        tb_log_name="plume_nav_ppo_experiment"
    )

    # Save trained model
    model.save("plume_nav_ppo_final")

    # Comprehensive evaluation
    final_success_rate = callback._evaluate_policy(num_episodes=1000)
    print(f"Final evaluation success rate: {{final_success_rate:.2%}}")

    return model, callback.evaluation_results
```

#### Ray RLlib Integration

```python
# Ray RLlib integration for distributed training
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.env_context import EnvContext

def create_rllib_env(env_config: EnvContext):
    \"\"\"Create environment for RLlib training.\"\"\"

    return create_plume_search_env(
        grid_size=env_config.get("grid_size", (64, 64)),
        goal_radius=env_config.get("goal_radius", 1.0),
        max_steps=env_config.get("max_steps", 200)
    )

def hyperparameter_optimization_study():
    \"\"\"Distributed hyperparameter optimization with Ray Tune.\"\"\"

    ray.init()

    # Define search space
    config = {{
        "env": create_rllib_env,
        "env_config": {{
            "grid_size": tune.choice([(32, 32), (64, 64), (128, 128)]),
            "goal_radius": tune.uniform(0.0, 2.0),
            "max_steps": tune.choice([100, 200, 500])
        }},
        "framework": "torch",
        "lr": tune.loguniform(1e-5, 1e-2),
        "gamma": tune.uniform(0.9, 0.999),
        "lambda": tune.uniform(0.9, 1.0),
        "clip_param": tune.uniform(0.1, 0.3),
        "num_workers": 4,
        "num_envs_per_worker": 8,
        "train_batch_size": tune.choice([4000, 8000, 16000]),
        "sgd_minibatch_size": tune.choice([128, 256, 512])
    }}

    # Run hyperparameter optimization
    analysis = tune.run(
        PPOTrainer,
        config=config,
        num_samples=50,
        metric="episode_reward_mean",
        mode="max",
        stop={{"training_iteration": 100}},
        resources_per_trial={{"cpu": 6, "gpu": 0.5}}
    )

    # Get best hyperparameters
    best_config = analysis.get_best_config(metric="episode_reward_mean", mode="max")
    print("Best hyperparameters:", best_config)

    ray.shutdown()
    return analysis
```

### Production Deployment Considerations

#### Docker Containerization

```dockerfile
# Dockerfile for production deployment
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY tests/ tests/

# Install in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/app/src
ENV MPLBACKEND=Agg

# Expose port for monitoring
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
  CMD python -c "import plume_nav_sim; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "src.examples.basic_usage"]
```

#### Monitoring and Observability

```python
# Production monitoring and observability patterns
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

# Metrics collection
ENVIRONMENT_STEPS = Counter('plume_nav_environment_steps_total', 'Total environment steps')
STEP_DURATION = Histogram('plume_nav_step_duration_seconds', 'Step execution time')
ACTIVE_ENVIRONMENTS = Gauge('plume_nav_active_environments', 'Number of active environments')
SUCCESS_RATE = Gauge('plume_nav_success_rate', 'Episode success rate')

class MonitoredPlumeSearchEnv:
    \"\"\"Production environment wrapper with monitoring.\"\"\"

    def __init__(self, **kwargs):
        self.env = create_plume_search_env(**kwargs)
        self.episode_count = 0
        self.success_count = 0
        ACTIVE_ENVIRONMENTS.inc()

    def step(self, action):
        with STEP_DURATION.time():
            result = self.env.step(action)
            ENVIRONMENT_STEPS.inc()

            # Track success rate
            obs, reward, terminated, truncated, info = result
            if terminated:
                self.success_count += 1
                SUCCESS_RATE.set(self.success_count / max(1, self.episode_count))

            return result

    def reset(self, **kwargs):
        self.episode_count += 1
        return self.env.reset(**kwargs)

    def close(self):
        ACTIVE_ENVIRONMENTS.dec()
        return self.env.close()

# Start metrics server
prometheus_client.start_http_server(8080)
```

### Future Architecture Evolution

#### Extensibility Roadmap

1. **Phase 1** (Current): Static single-agent environment with Gaussian plumes
2. **Phase 2**: Dynamic plume models with temporal evolution
3. **Phase 3**: Multi-agent environments with coordination mechanisms
4. **Phase 4**: 3D environments with altitude-dependent plume dynamics
5. **Phase 5**: Real-world sensor integration and hardware-in-the-loop testing

#### Extension Points Design

The current architecture provides foundation patterns for future extensions:

- **Pluggable Plume Models**: Abstract base classes for dynamic and multi-source plumes
- **Configurable Action Spaces**: Support for continuous actions and multi-agent coordination
- **Extensible Rendering**: Plugin architecture for custom visualization backends
- **Modular Reward Functions**: Composable reward systems for complex objectives
- **Scalable State Management**: Distributed state for large-scale multi-agent scenarios

This advanced topics section provides architectural guidance for extending plume_nav_sim beyond
its current proof-of-life implementation toward research-grade and production-ready applications.
        """

        # Integrate API reference documentation for comprehensive technical coverage
        guide_sections['api_reference_integration'] = f"""
## API Reference Integration

This developer guide integrates closely with the comprehensive API reference documentation.
Key integration points include:

### Core Environment API

For detailed API documentation of the main environment class, refer to:
- `PlumeSearchEnv` class methods and properties
- Environment configuration parameters and validation
- Performance characteristics and optimization guidance
- Extension points and customization patterns

### Component APIs

Detailed documentation for system components:
- `StaticGaussianPlume`: Mathematical plume model implementation
- Rendering pipeline: Dual-mode visualization system
- Seeding utilities: Scientific reproducibility framework
- Type system: Core data structures and validation

### Development Integration

The API reference provides:
- Code examples with expected outputs
- Performance benchmarking procedures
- Testing strategies and quality assurance
- Extension development patterns

To generate the complete API reference:

```python
from .api_reference import generate_api_documentation

api_docs = generate_api_documentation(
    include_internal_apis=True,
    include_performance_analysis=True,
    include_examples=True
)
print(api_docs)
```
        """

        # Generate comprehensive appendices with reference materials and development resources
        guide_sections['appendices'] = f"""
## Appendices

### Appendix A: Development Environment Reference

#### Required Python Packages

```
# Core dependencies
gymnasium>=0.29.0          # RL environment framework
numpy>=2.1.0               # Mathematical operations and arrays
matplotlib>=3.9.0          # Visualization and rendering

# Development dependencies
pytest>=8.0                # Testing framework
black>=22.0                # Code formatting
isort>=5.10                # Import sorting
mypy>=1.0                  # Static type checking
flake8>=5.0                # Code linting
pre-commit>=2.20           # Git hooks
```

#### Development Scripts

**setup_dev_env.sh**:
```bash
#!/bin/bash
# Development environment setup script

python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
pre-commit install
echo "Development environment ready!"
```

**run_tests.sh**:
```bash
#!/bin/bash
# Comprehensive testing script

echo "Running test suite..."
pytest tests/ --cov=src/plume_nav_sim --cov-report=html --cov-report=term
echo "Running type checking..."
mypy src/plume_nav_sim
echo "Running linting..."
flake8 src/plume_nav_sim
echo "All checks complete!"
```

### Appendix B: Performance Optimization Reference

#### Profiling Commands

```bash
# Profile environment step performance
python -m cProfile -s cumulative examples/performance_profile.py

# Memory profiling with memory_profiler
pip install memory_profiler
python -m memory_profiler examples/memory_analysis.py

# Line-by-line profiling
kernprof -l -v examples/detailed_profile.py
```

#### Optimization Checklist

- [ ] Environment step time <1ms average
- [ ] Memory usage <50MB for standard grid
- [ ] NumPy operations vectorized
- [ ] Minimal Python loops in hot paths
- [ ] Efficient data structures for state
- [ ] Optimized rendering pipeline
- [ ] Proper resource cleanup

### Appendix C: Testing Framework Reference

#### Test Categories

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Component interaction validation
3. **Performance Tests**: Latency and throughput validation
4. **Reproducibility Tests**: Seeding and determinism validation
5. **API Compliance Tests**: Gymnasium interface validation
6. **Regression Tests**: Version compatibility testing

#### Test Configuration

**pytest.ini**:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --verbose
    --tb=short
    --cov=src/plume_nav_sim
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=95
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    slow: Slow tests
```

### Appendix D: Extension Development Reference

#### Custom Plume Model Template

```python
from abc import ABC, abstractmethod
import numpy as np
from ..core.types import Coordinates, GridSize

class CustomPlumeModel(ABC):
    \"\"\"Template for custom plume model development.\"\"\"

    def __init__(self, grid_size: GridSize):
        self.grid_size = grid_size
        self.concentration_field = None

    @abstractmethod
    def initialize_plume(self, source_location: Coordinates) -> np.ndarray:
        \"\"\"Initialize plume concentration field.\"\"\"
        pass

    @abstractmethod
    def sample_concentration(self, position: Coordinates) -> float:
        \"\"\"Sample concentration at specific position.\"\"\"
        pass

    def validate_model(self) -> bool:
        \"\"\"Validate model configuration and state.\"\"\"
        return True
```

#### Custom Renderer Template

```python
class CustomRenderer(ABC):
    \"\"\"Template for custom rendering backend development.\"\"\"

    def __init__(self, grid_size: GridSize):
        self.grid_size = grid_size

    @abstractmethod
    def render_rgb_array(self, state: dict) -> np.ndarray:
        \"\"\"Generate RGB array representation.\"\"\"
        pass

    def cleanup_resources(self):
        \"\"\"Clean up rendering resources.\"\"\"
        pass
```

### Appendix E: Research Integration Templates

#### Experiment Configuration Template

```python
# experiment_config.py
EXPERIMENT_CONFIGS = {{
    'baseline': {{
        'env_params': {{
            'grid_size': (64, 64),
            'goal_radius': 1.0,
            'max_steps': 200
        }},
        'training_params': {{
            'total_timesteps': 1000000,
            'learning_rate': 3e-4,
            'batch_size': 64
        }},
        'evaluation': {{
            'num_episodes': 1000,
            'seeds': list(range(1000))
        }}
    }}
}}
```

#### Publication Results Template

```python
# results_analysis.py
def generate_publication_results():
    \"\"\"Generate results formatted for research publication.\"\"\"

    results = {{
        'environment': 'PlumeNav-StaticGaussian-v0',
        'grid_size': '64x64',
        'success_rates': {{
            'random_policy': 0.05,
            'gradient_following': 0.23,
            'ppo_trained': 0.87
        }},
        'performance_metrics': {{
            'avg_step_time_ms': 0.45,
            'memory_usage_mb': 12.3,
            'episodes_per_second': 2222
        }}
    }}

    return results
```

### Appendix F: Version History and Changelog

#### Version {DEVELOPER_GUIDE_VERSION}
- Comprehensive developer guide with architecture analysis
- Performance optimization guidance and benchmarking tools
- Testing framework documentation and quality standards
- Extension development patterns and customization examples
- Research integration templates and workflow automation
- Advanced topics covering multi-agent and dynamic plume considerations

#### Development Roadmap
- **v2.1**: Enhanced performance profiling and optimization tools
- **v2.2**: Extended testing framework with property-based testing
- **v2.3**: Advanced extension examples with multi-agent patterns
- **v3.0**: Dynamic plume model architecture and implementation guidance

This comprehensive developer guide provides technical contributors with all necessary information
for understanding, extending, and maintaining the plume_nav_sim system while ensuring quality,
performance, and scientific reproducibility standards.
        """

        # Generate table of contents based on included sections
        toc_entries = []
        section_order = [
            'introduction', 'system_overview', 'architecture_documentation',
            'development_setup', 'contribution_guidelines', 'performance_guide',
            'testing_guide', 'extension_guide', 'advanced_topics',
            'troubleshooting', 'api_reference_integration', 'appendices'
        ]

        for section_name in section_order:
            if section_name in guide_sections and guide_sections[section_name]:
                # Extract main headings from content
                content = guide_sections[section_name]
                lines = content.strip().split('\n')
                for line in lines:
                    if line.startswith('##') and not line.startswith('###'):
                        heading = line.replace('##', '').strip()
                        anchor = heading.lower().replace(' ', '-').replace('&', '').replace('(', '').replace(')', '')
                        toc_entries.append(f"- [{heading}](#{anchor})")

        guide_sections['table_of_contents'] = f"""
## Table of Contents

{chr(10).join(toc_entries)}
        """

        # Format complete developer guide with technical styling, code syntax highlighting, and comprehensive cross-references
        complete_guide = []

        # Add header and table of contents
        complete_guide.append(guide_sections['header'])
        complete_guide.append(guide_sections['table_of_contents'])

        # Add main content sections in logical order
        for section_name in section_order:
            if section_name in guide_sections and guide_sections[section_name]:
                complete_guide.append(guide_sections[section_name])

        # Join all sections with appropriate spacing
        formatted_guide = '\n\n'.join(complete_guide)

        # Apply output format-specific formatting
        if output_format == 'html':
            # Convert markdown to HTML (simplified transformation)
            formatted_guide = formatted_guide.replace('```python', '<pre><code class="language-python">')
            formatted_guide = formatted_guide.replace('```', '</code></pre>')
            formatted_guide = formatted_guide.replace('##', '<h2>').replace('#', '<h1>')
        elif output_format == 'restructuredtext':
            # Convert to reStructuredText format
            formatted_guide = formatted_guide.replace('#', '=')
            formatted_guide = formatted_guide.replace('```python', '.. code-block:: python')
            formatted_guide = formatted_guide.replace('```', '')

        # Return comprehensive developer guide optimized for technical contributors and system architects
        return formatted_guide

    except Exception as e:
        # Handle any errors in guide generation with technical fallback content
        error_guide = f"""
# Plume Navigation Simulation Developer Guide v{DEVELOPER_GUIDE_VERSION}

## Error in Guide Generation

An error occurred while generating the comprehensive developer guide: {str(e)}

### Technical Quick Start (Fallback)

```python
# Development environment setup
import gymnasium as gym
from plume_nav_sim.registration import register_env, ENV_ID
from plume_nav_sim.envs.plume_search_env import create_plume_search_env

# Register and create environment
register_env()
env = gym.make(ENV_ID)

# Performance testing
import time
obs, info = env.reset(seed=42)
start_time = time.perf_counter()
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset(seed=42 + step)
step_time_ms = (time.perf_counter() - start_time) / 1000 * 1000
print(f"Average step time: {{step_time_ms:.3f}}ms")

env.close()
```

### Development Resources

- **System Architecture**: Modular layered design with Gymnasium compliance
- **Performance Targets**: <1ms step latency, <50MB memory usage
- **Quality Standards**: >95% test coverage, 100% type hints
- **Extension Points**: Custom plume models, rendering backends, multi-agent patterns

Please check the system configuration and dependencies. For technical support,
refer to the troubleshooting section or contact the development team.
        """

        return error_guide


def create_architecture_documentation(
    include_component_diagrams: bool = True,
    include_performance_analysis: bool = True,
    include_design_rationale: bool = True
) -> str:
    """
    Creates comprehensive system architecture documentation including component analysis, interaction
    patterns, data flow diagrams, and design decision analysis for understanding and extending the
    plume_nav_sim system.

    This function generates detailed technical documentation of the system's modular layered
    architecture, component relationships, performance characteristics, and extensibility patterns
    for developers and system architects.

    Args:
        include_component_diagrams (bool): Include visual architecture representation and component relationship diagrams
        include_performance_analysis (bool): Include architectural performance implications and optimization guidance
        include_design_rationale (bool): Include design decisions and architectural trade-offs documentation

    Returns:
        str: Detailed architecture documentation with diagrams, analysis, and technical insights
        for system understanding and extension development
    """
    architecture_sections = []

    # Generate architecture overview explaining modular layered design and component-based patterns
    architecture_sections.append("""
## System Architecture Deep Dive

### Architectural Design Principles

The plume_nav_sim system implements a **modular layered architecture** with clear separation of concerns
and well-defined interfaces between components. This design supports maintainability, testability, and
extensibility while optimizing for research-grade performance.

#### Core Architectural Principles

1. **Single Responsibility Principle**: Each component handles one specific aspect of the simulation
2. **Dependency Inversion**: Core logic depends on abstractions rather than concrete implementations
3. **Interface Segregation**: Clean, minimal interfaces between components with clear contracts
4. **Open/Closed Principle**: Extensible design supporting future enhancements without core modifications
5. **Performance by Design**: Architecture optimized for <1ms step latency and efficient resource usage

#### Layered Architecture Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Application Layer                            │
│  Training Scripts │ Research Workflows │ Educational Examples       │
├─────────────────────────────────────────────────────────────────────┤
│                        Interface Layer                              │
│    Gymnasium API │ Environment Registration │ User Documentation    │
├─────────────────────────────────────────────────────────────────────┤
│                         Core Layer                                  │
│  PlumeSearchEnv │ BaseEnvironment │ Episode Management │ Validation │
├─────────────────────────────────────────────────────────────────────┤
│                        Domain Layer                                 │
│ Plume Models │ Reward Functions │ Action Processing │ State Updates │
├─────────────────────────────────────────────────────────────────────┤
│                     Infrastructure Layer                            │
│  Rendering │ Seeding │ Performance Monitoring │ Error Handling      │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Architecture Analysis

#### PlumeSearchEnv (Core Orchestrator)

The central environment class coordinates all system components and implements the Gymnasium interface:

```python
# PlumeSearchEnv architecture pattern
class PlumeSearchEnv:
    \"\"\"
    Central orchestrator implementing modular component coordination.
    Demonstrates architectural patterns for component integration.
    \"\"\"

    def __init__(self, grid_size, source_location, max_steps, goal_radius):
        # Component initialization with dependency injection pattern
        self.state_manager = StateManager(grid_size)
        self.plume_model = StaticGaussianPlume(grid_size, source_location)
        self.action_processor = ActionProcessor(grid_size)
        self.reward_calculator = RewardCalculator(source_location, goal_radius)
        self.boundary_enforcer = BoundaryEnforcer(grid_size)
        self.performance_monitor = PerformanceMonitor()

        # Rendering system with optional initialization
        self.rgb_renderer = self._initialize_renderer() if render_mode else None

        # Validation system for component integrity
        self._validate_component_consistency()

    def step(self, action):
        \"\"\"
        Orchestrated step execution demonstrating component interaction patterns.
        \"\"\"

        # Performance monitoring wrapper
        with self.performance_monitor.step_timer():
            # Component interaction sequence
            processed_action = self.action_processor.validate_and_process(action)
            new_position = self.state_manager.update_agent_position(processed_action)
            bounded_position = self.boundary_enforcer.enforce_bounds(new_position)
            concentration = self.plume_model.sample_concentration(bounded_position)
            reward = self.reward_calculator.calculate_reward(bounded_position)
            terminated = self.reward_calculator.check_termination(bounded_position)

            # State synchronization across components
            self.state_manager.update_episode_state(bounded_position, self.step_count)

            return self._construct_gymnasium_response(
                concentration, reward, terminated, self.step_count >= self.max_steps
            )
```

**Key Architectural Features**:
- **Component Composition**: Aggregates specialized components rather than inheritance
- **Dependency Injection**: Components receive dependencies through constructor injection
- **Interface Segregation**: Each component exposes minimal, focused interfaces
- **Performance Integration**: Built-in performance monitoring and optimization
- **Validation Framework**: Comprehensive component state validation and consistency checking

#### StaticGaussianPlume (Domain Component)

Mathematical plume model implementing domain-specific concentration calculations:

```python
class StaticGaussianPlume:
    \"\"\"
    Domain component demonstrating mathematical computation architecture.
    Optimized for performance with vectorized NumPy operations.
    \"\"\"

    def __init__(self, grid_size: GridSize, source_location: Coordinates):
        self.grid_size = grid_size
        self.source_location = source_location

        # Pre-computed concentration field for O(1) sampling
        self.concentration_field = self._generate_concentration_field()

        # Performance optimization with field caching
        self._field_cache_valid = True

    def _generate_concentration_field(self) -> np.ndarray:
        \"\"\"
        Vectorized field generation demonstrating performance-optimized mathematics.
        \"\"\"

        # Mesh generation for vectorized operations
        y_coords, x_coords = np.meshgrid(
            np.arange(self.grid_size.height),
            np.arange(self.grid_size.width),
            indexing='ij'
        )

        # Distance calculation with broadcasting
        dx = x_coords - self.source_location.x
        dy = y_coords - self.source_location.y
        distance_squared = dx**2 + dy**2

        # Gaussian concentration with optimized parameters
        dispersion_factor = 2 * (12.0 ** 2)  # Pre-computed for efficiency
        concentration = np.exp(-distance_squared / dispersion_factor)

        return concentration.astype(np.float32)  # Memory optimization

    def sample_concentration(self, position: Coordinates) -> float:
        \"\"\"
        O(1) concentration sampling with bounds checking.
        \"\"\"

        # Bounds validation with performance optimization
        if not (0 <= position.x < self.grid_size.width and
                0 <= position.y < self.grid_size.height):
            return 0.0

        # Direct array access for optimal performance
        return float(self.concentration_field[position.y, position.x])
```

**Architectural Highlights**:
- **Pre-computation Strategy**: Expensive calculations performed once during initialization
- **Memory Layout Optimization**: NumPy array indexing and data type optimization
- **Bounds Safety**: Integrated validation without performance penalty
- **Cache Management**: Field validity tracking for dynamic scenarios

#### Rendering Pipeline (Infrastructure Component)

Dual-mode visualization system with backend abstraction:

```python
class RenderingPipeline:
    \"\"\"
    Infrastructure component demonstrating plugin architecture patterns.
    Supports multiple rendering backends with consistent interfaces.
    \"\"\"

    def __init__(self, grid_size: GridSize, render_mode: str):
        self.grid_size = grid_size
        self.render_mode = render_mode

        # Backend selection with factory pattern
        self.renderer = self._create_renderer(render_mode)

        # Resource management for matplotlib backends
        self._matplotlib_resources = None

    def _create_renderer(self, mode: str):
        \"\"\"Factory method for renderer instantiation.\"\"\"

        if mode == 'rgb_array':
            return NumpyRGBRenderer(self.grid_size)
        elif mode == 'human':
            return MatplotlibRenderer(self.grid_size)
        else:
            raise ValueError(f"Unsupported render mode: {{mode}}")

    def render(self, environment_state: dict) -> Optional[np.ndarray]:
        \"\"\"
        Unified rendering interface supporting multiple backends.
        \"\"\"

        try:
            return self.renderer.render_frame(environment_state)
        except Exception as e:
            # Graceful degradation with error logging
            logging.warning(f"Rendering failed: {{e}}")
            return self._create_fallback_frame()

    def cleanup_resources(self):
        \"\"\"Resource cleanup with backend-specific handling.\"\"\"

        if hasattr(self.renderer, 'cleanup'):
            self.renderer.cleanup()

        # Matplotlib-specific cleanup
        if self.render_mode == 'human':
            plt.close('all')
```

**Design Patterns Demonstrated**:
- **Factory Pattern**: Dynamic renderer instantiation based on configuration
- **Strategy Pattern**: Interchangeable rendering algorithms through common interface
- **Resource Management**: Explicit cleanup and memory management
- **Error Recovery**: Graceful degradation with fallback mechanisms
    """)

    # Document core components including PlumeSearchEnv, StaticGaussianPlume, rendering pipeline, and state management
    architecture_sections.append("""
### Component Interaction Analysis

#### Data Flow Patterns

The system implements several key data flow patterns that optimize for both performance and maintainability:

```python
# Data flow analysis for step execution
def analyze_step_data_flow():
    \"\"\"
    Demonstrates the complete data flow through system components during step execution.
    Shows how data transforms and flows between architectural layers.
    \"\"\"

    # 1. Input Layer: Action validation and preprocessing
    raw_action = 2  # DOWN action from agent

    # 2. Core Layer: Action processing and validation
    validated_action = ActionProcessor.validate_action(
        action=raw_action,
        action_space_bounds=(0, 3),
        current_state="valid"
    )

    # 3. Domain Layer: State transformation
    current_position = Coordinates(x=32, y=32)
    new_position = ActionProcessor.apply_action_to_position(
        position=current_position,
        action=validated_action
    )

    # 4. Infrastructure Layer: Boundary enforcement
    bounded_position = BoundaryEnforcer.enforce_grid_bounds(
        position=new_position,
        grid_bounds=GridSize(width=128, height=128)
    )

    # 5. Domain Layer: Plume interaction
    concentration = StaticGaussianPlume.sample_concentration(
        position=bounded_position,
        concentration_field=np.ndarray  # Pre-computed field
    )

    # 6. Domain Layer: Reward calculation
    reward = RewardCalculator.calculate_sparse_reward(
        agent_position=bounded_position,
        source_location=Coordinates(x=64, y=64),
        goal_radius=1.0
    )

    # 7. Core Layer: Episode state management
    episode_status = StateManager.update_episode_state(
        step_count=150,
        max_steps=1000,
        terminated=reward > 0,
        agent_position=bounded_position
    )

    # 8. Interface Layer: Gymnasium response construction
    gym_response = {
        'observation': np.array([concentration], dtype=np.float32),
        'reward': reward,
        'terminated': episode_status['terminated'],
        'truncated': episode_status['truncated'],
        'info': episode_status['info_dict']
    }

    return gym_response
```

#### Component Communication Patterns

**Synchronous Method Calls**: Primary communication pattern for deterministic execution
- Direct method invocation between components
- Predictable execution order and timing
- Simplified debugging and testing
- Optimal for single-agent, single-threaded scenarios

**Event-driven Patterns**: Used for monitoring and logging
- Performance metric collection
- State change notifications
- Error handling and recovery
- Extensible for future multi-agent scenarios

**Data Sharing Patterns**: Efficient state representation
- Immutable data structures where possible
- Minimal data copying between components
- Shared references to large arrays (concentration fields)
- Clear ownership semantics for mutable state

#### Component Lifecycle Management

```python
class ComponentLifecycleManager:
    \"\"\"
    Demonstrates component lifecycle management patterns used throughout the system.
    Ensures proper initialization, operation, and cleanup of system components.
    \"\"\"

    def __init__(self):
        self.components = {}
        self.initialization_order = [
            'state_manager',
            'plume_model',
            'action_processor',
            'reward_calculator',
            'boundary_enforcer',
            'renderer',
            'performance_monitor'
        ]

    def initialize_components(self, config: EnvironmentConfig):
        \"\"\"
        Initialize components in dependency order with validation.
        \"\"\"

        for component_name in self.initialization_order:
            try:
                component = self._create_component(component_name, config)
                self._validate_component(component)
                self.components[component_name] = component

                logging.debug(f"Initialized {{component_name}} successfully")

            except Exception as e:
                logging.error(f"Failed to initialize {{component_name}}: {{e}}")
                self._cleanup_partial_initialization()
                raise

    def _create_component(self, name: str, config: EnvironmentConfig):
        \"\"\"Factory method for component creation with configuration injection.\"\"\"

        component_factories = {
            'state_manager': lambda: StateManager(config.grid_size),
            'plume_model': lambda: StaticGaussianPlume(
                config.grid_size,
                config.source_location
            ),
            'action_processor': lambda: ActionProcessor(config.grid_size),
            'reward_calculator': lambda: RewardCalculator(
                config.source_location,
                config.goal_radius
            ),
            'boundary_enforcer': lambda: BoundaryEnforcer(config.grid_size),
            'renderer': lambda: self._create_renderer(config.render_mode),
            'performance_monitor': lambda: PerformanceMonitor()
        }

        return component_factories[name]()

    def validate_component_integration(self):
        \"\"\"
        Validate that components are properly integrated and compatible.
        \"\"\"

        validation_results = {}

        # Check grid size consistency across components
        grid_size = self.components['state_manager'].grid_size
        for component_name, component in self.components.items():
            if hasattr(component, 'grid_size'):
                if component.grid_size != grid_size:
                    validation_results[component_name] = f"Grid size mismatch: {{component.grid_size}} != {{grid_size}}"

        # Check source location consistency
        source_location = self.components['plume_model'].source_location
        if hasattr(self.components['reward_calculator'], 'source_location'):
            if self.components['reward_calculator'].source_location != source_location:
                validation_results['reward_calculator'] = "Source location mismatch with plume model"

        # Validate component interfaces
        required_methods = {
            'plume_model': ['sample_concentration', 'get_concentration_field'],
            'state_manager': ['update_agent_position', 'get_current_state'],
            'action_processor': ['validate_action', 'apply_action'],
            'reward_calculator': ['calculate_reward', 'check_termination']
        }

        for component_name, methods in required_methods.items():
            component = self.components[component_name]
            for method_name in methods:
                if not hasattr(component, method_name):
                    validation_results[component_name] = f"Missing required method: {{method_name}}"

        if validation_results:
            raise ComponentValidationError(f"Component validation failed: {{validation_results}}")

        return True
```
    """)

    # Create component interaction analysis showing data flow, method calls, and dependency relationships
    architecture_sections.append("""
### Performance Architecture Analysis

#### Performance-Optimized Design Patterns

The system architecture incorporates several performance optimization patterns:

**Pre-computation Strategy**:
```python
class PerformanceOptimizedComponent:
    \"\"\"
    Demonstrates pre-computation patterns used throughout the architecture.
    Expensive operations are performed once and cached for repeated access.
    \"\"\"

    def __init__(self, grid_size: GridSize, source_location: Coordinates):
        # Pre-compute expensive mathematical operations
        self.distance_field = self._precompute_distance_field(grid_size, source_location)
        self.concentration_field = self._precompute_concentration_field()

        # Cache frequently accessed values
        self._cached_gradients = None
        self._cache_valid = True

    def _precompute_distance_field(self, grid_size: GridSize, source: Coordinates) -> np.ndarray:
        \"\"\"Pre-compute distance field for O(1) distance lookups.\"\"\"

        y_coords, x_coords = np.meshgrid(
            np.arange(grid_size.height),
            np.arange(grid_size.width),
            indexing='ij'
        )

        # Vectorized distance calculation
        dx = x_coords - source.x
        dy = y_coords - source.y
        distances = np.sqrt(dx**2 + dy**2)

        return distances.astype(np.float32)

    def _precompute_concentration_field(self) -> np.ndarray:
        \"\"\"Transform distances to concentrations with optimized parameters.\"\"\"

        # Avoid repeated division in hot path
        dispersion_inv = 1.0 / (2 * 12.0**2)

        # Vectorized exponential calculation
        concentration = np.exp(-self.distance_field**2 * dispersion_inv)

        return concentration.astype(np.float32)
```

**Memory Layout Optimization**:
- **Array Contiguity**: NumPy arrays stored in C-contiguous layout for cache efficiency
- **Data Type Selection**: float32 instead of float64 for 50% memory reduction
- **Memory Pooling**: Reuse of temporary arrays in computational hot paths
- **Garbage Collection Optimization**: Minimal object allocation in step execution

**Vectorization Patterns**:
```python
def demonstrate_vectorization_patterns():
    \"\"\"
    Shows how vectorization is used throughout the architecture for performance.
    \"\"\"

    # Instead of Python loops (slow):
    # concentrations = []
    # for x in range(width):
    #     for y in range(height):
    #         distance = sqrt((x - source_x)**2 + (y - source_y)**2)
    #         concentrations.append(exp(-distance**2 / (2 * sigma**2)))

    # Use vectorized operations (fast):
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    distances = np.sqrt((x_coords - source_x)**2 + (y_coords - source_y)**2)
    concentrations = np.exp(-distances**2 / (2 * sigma**2))

    # Performance benefit: ~100x faster for 128x128 grids
    return concentrations
```

#### Component Performance Characteristics

| Component | Initialization Time | Step Overhead | Memory Usage | Optimization Level |
|-----------|-------------------|---------------|--------------|-------------------|
| StateManager | <1ms | <0.01ms | <1MB | High |
| StaticGaussianPlume | <10ms | <0.1ms | ~20MB (128²) | Very High |
| ActionProcessor | <0.1ms | <0.01ms | <1MB | High |
| RewardCalculator | <0.1ms | <0.05ms | <1MB | Medium |
| BoundaryEnforcer | <0.1ms | <0.01ms | <1MB | High |
| RenderingPipeline | <50ms | <5ms | ~5MB | Medium |

**Performance Monitoring Integration**:
```python
class ArchitecturePerformanceMonitor:
    \"\"\"
    Integrated performance monitoring demonstrating observability patterns.
    \"\"\"

    def __init__(self):
        self.component_timings = {}
        self.memory_usage = {}
        self.call_counts = {}

    def profile_component_step(self, component_name: str):
        \"\"\"Context manager for component-level performance profiling.\"\"\"

        return ComponentProfiler(
            component_name=component_name,
            monitor=self
        )

    def get_performance_summary(self) -> dict:
        \"\"\"Generate comprehensive performance analysis.\"\"\"

        summary = {
            'total_components': len(self.component_timings),
            'average_step_time': np.mean([
                np.mean(times) for times in self.component_timings.values()
            ]),
            'slowest_component': max(
                self.component_timings.items(),
                key=lambda x: np.mean(x[1])
            )[0],
            'memory_footprint_mb': sum(
                usage[-1] for usage in self.memory_usage.values()
            ) / (1024 * 1024),
            'performance_bottlenecks': self._identify_bottlenecks()
        }

        return summary

    def _identify_bottlenecks(self) -> List[str]:
        \"\"\"Identify performance bottlenecks in component architecture.\"\"\"

        bottlenecks = []
        avg_times = {
            name: np.mean(times)
            for name, times in self.component_timings.items()
        }

        # Components taking >10% of step time are considered bottlenecks
        total_time = sum(avg_times.values())
        threshold = total_time * 0.1

        for component, avg_time in avg_times.items():
            if avg_time > threshold:
                bottlenecks.append(f"{{component}}: {{avg_time:.3f}}ms ({{avg_time/total_time:.1%}})")

        return bottlenecks
```
    """)

    # Include component diagrams if include_component_diagrams enabled with visual architecture representation
    if include_component_diagrams:
        architecture_sections.append("""
### Architecture Visualization

#### Component Relationship Diagram

```mermaid
graph TB
    subgraph "Application Layer"
        A[Training Scripts]
        B[Research Workflows]
        C[Educational Examples]
    end

    subgraph "Interface Layer"
        D[Gymnasium API]
        E[Environment Registration]
        F[Documentation System]
    end

    subgraph "Core Layer"
        G[PlumeSearchEnv]
        H[BaseEnvironment]
        I[Episode Management]
    end

    subgraph "Domain Layer"
        J[StaticGaussianPlume]
        K[RewardCalculator]
        L[ActionProcessor]
    end

    subgraph "Infrastructure Layer"
        M[RenderingPipeline]
        N[SeedingSystem]
        O[PerformanceMonitor]
        P[ValidationFramework]
    end

    A --> D
    B --> D
    C --> D

    D --> G
    E --> G
    F --> G

    G --> H
    G --> I
    H --> J
    H --> K
    H --> L

    G --> M
    G --> N
    G --> O
    G --> P

    style G fill:#e1f5fe
    style J fill:#f3e5f5
    style M fill:#e8f5e8
    style O fill:#fff3e0
```

#### Data Flow Architecture

```mermaid
sequenceDiagram
    participant Agent
    participant PlumeEnv as PlumeSearchEnv
    participant ActionProc as ActionProcessor
    participant StateMan as StateManager
    participant PlumeModel as StaticGaussianPlume
    participant RewardCalc as RewardCalculator
    participant PerfMon as PerformanceMonitor

    Agent->>PlumeEnv: step(action)

    PlumeEnv->>PerfMon: start_step_timer()
    PlumeEnv->>ActionProc: validate_action(action)
    ActionProc-->>PlumeEnv: validated_action

    PlumeEnv->>StateMan: update_position(action)
    StateMan-->>PlumeEnv: new_position

    PlumeEnv->>PlumeModel: sample_concentration(position)
    PlumeModel-->>PlumeEnv: concentration_value

    PlumeEnv->>RewardCalc: calculate_reward(position)
    RewardCalc-->>PlumeEnv: reward_value

    PlumeEnv->>StateMan: update_episode_state()
    StateMan-->>PlumeEnv: episode_status

    PlumeEnv->>PerfMon: end_step_timer()
    PerfMon-->>PlumeEnv: performance_metrics

    PlumeEnv-->>Agent: (obs, reward, terminated, truncated, info)
```

#### Memory Architecture Layout

```
Memory Layout Optimization:
┌─────────────────────────────────────────────────────────────┐
│                    Environment Instance                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐   │
│  │  State Manager  │ │ Action Processor│ │ Reward Calc   │   │
│  │    ~1MB         │ │     ~0.5MB      │ │    ~0.5MB     │   │
│  └─────────────────┘ └─────────────────┘ └───────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Plume Model (StaticGaussian)               │ │
│  │  Concentration Field: Width × Height × 4 bytes         │ │
│  │  Example 128×128: 65,536 cells × 4 = 262KB            │ │
│  │  Distance cache (optional): +262KB                     │ │
│  │  Total: ~525KB - 20MB (depends on grid size)          │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                Rendering Pipeline                       │ │
│  │  RGB Buffer: Width × Height × 3 × 1 byte              │ │
│  │  Example 128×128: 49,152 bytes = 48KB                 │ │
│  │  Matplotlib resources: ~2-5MB (when active)           │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            Performance Monitoring                       │ │
│  │  Timing arrays, metrics cache: ~100KB                 │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

Total Memory Footprint:
- Small grid (32×32):   ~5-8MB
- Standard grid (128×128): ~25-30MB
- Large grid (256×256): ~80-100MB
```
        """)

    # Generate integration patterns for external framework compatibility and workflow integration
    architecture_sections.append("""
### Integration Architecture Patterns

#### Framework Integration Design

The architecture provides clean integration points for external RL frameworks and scientific tools:

```python
class FrameworkIntegrationLayer:
    \"\"\"
    Demonstrates integration patterns for external RL frameworks.
    Provides adapter patterns and compatibility layers.
    \"\"\"

    def __init__(self, base_environment: PlumeSearchEnv):
        self.base_env = base_environment
        self.integration_adapters = {}

    def create_stable_baselines3_adapter(self):
        \"\"\"
        Adapter pattern for Stable-Baselines3 integration.
        Ensures compatibility with SB3's environment requirements.
        \"\"\"

        class SB3Adapter(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)

                # SB3-specific environment validation
                self._validate_sb3_compatibility()

            def _validate_sb3_compatibility(self):
                \"\"\"Validate environment meets SB3 requirements.\"\"\"

                # Check observation space compatibility
                assert isinstance(self.observation_space, gym.spaces.Box), \\
                    "SB3 requires Box observation space"

                # Check action space compatibility
                assert isinstance(self.action_space, gym.spaces.Discrete), \\
                    "SB3 requires Discrete action space for this environment"

                # Validate step function signature
                obs, info = self.env.reset(seed=42)
                obs, reward, terminated, truncated, info = self.env.step(0)

                assert isinstance(obs, np.ndarray), "Observation must be numpy array"
                assert isinstance(reward, (int, float)), "Reward must be numeric"
                assert isinstance(terminated, bool), "Terminated must be boolean"
                assert isinstance(truncated, bool), "Truncated must be boolean"
                assert isinstance(info, dict), "Info must be dictionary"

            def reset(self, **kwargs):
                \"\"\"SB3-compatible reset method.\"\"\"
                obs, info = self.env.reset(**kwargs)
                return obs, info

            def step(self, action):
                \"\"\"SB3-compatible step method with validation.\"\"\"

                # Validate action type and range
                action = int(action)  # Ensure integer action
                assert self.action_space.contains(action), f"Invalid action: {{action}}"

                return self.env.step(action)

        return SB3Adapter(self.base_env)

    def create_ray_rllib_config(self):
        \"\"\"
        Configuration generator for Ray RLlib integration.
        Provides optimized configuration for distributed training.
        \"\"\"

        config = {{
            "env": lambda config: create_plume_search_env(**config),
            "env_config": {{
                "grid_size": (64, 64),
                "goal_radius": 1.0,
                "max_steps": 200,
                "render_mode": None  # Disable rendering for distributed training
            }},

            # Performance optimizations for RLlib
            "num_workers": 8,
            "num_envs_per_worker": 4,
            "batch_mode": "complete_episodes",
            "rollout_fragment_length": 200,

            # Framework-specific settings
            "framework": "torch",
            "torch_compile_learner": True,
            "torch_compile_learner_dynamo_backend": "inductor",

            # Environment-specific optimizations
            "preprocessor_pref": None,  # Skip preprocessing for simple obs space
            "observation_filter": "NoFilter",
            "normalize_actions": False,  # Discrete actions don't need normalization

            # Resource management
            "placement_strategy": "SPREAD",  # Distribute workers across nodes
        }}

        return config

    def create_jupyter_integration(self):
        \"\"\"
        Jupyter notebook integration with enhanced visualization.
        Provides interactive exploration and analysis tools.
        \"\"\"

        class JupyterIntegration:
            def __init__(self, env: PlumeSearchEnv):
                self.env = env
                self.episode_data = []

            def create_interactive_widget(self):
                \"\"\"Create interactive widget for environment control.\"\"\"

                try:
                    from IPython.widgets import interact, IntSlider, Dropdown
                    from IPython.display import display

                    def step_environment(action_idx):
                        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
                        obs, reward, terminated, truncated, info = self.env.step(action_idx)

                        print(f"Action: {{action_names[action_idx]}}")
                        print(f"Observation: {{obs[0]:.4f}}")
                        print(f"Reward: {{reward}}")
                        print(f"Terminated: {{terminated}}")

                        # Store for analysis
                        self.episode_data.append({{
                            'action': action_idx,
                            'observation': obs[0],
                            'reward': reward,
                            'terminated': terminated
                        }})

                        # Render if possible
                        if hasattr(self.env, 'render'):
                            frame = self.env.render()
                            if frame is not None:
                                plt.figure(figsize=(6, 6))
                                plt.imshow(frame)
                                plt.axis('off')
                                plt.title(f'Step {{len(self.episode_data)}}')
                                plt.show()

                    # Create interactive controls
                    interact(
                        step_environment,
                        action_idx=Dropdown(
                            options=[(name, i) for i, name in enumerate(['UP', 'RIGHT', 'DOWN', 'LEFT'])],
                            description='Action:'
                        )
                    )

                except ImportError:
                    print("IPython widgets not available. Install with: pip install ipywidgets")

            def analyze_episode_data(self):
                \"\"\"Analyze collected episode data with visualizations.\"\"\"

                if not self.episode_data:
                    print("No episode data available.")
                    return

                import pandas as pd

                df = pd.DataFrame(self.episode_data)

                # Create analysis plots
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))

                # Observation trajectory
                axes[0, 0].plot(df['observation'])
                axes[0, 0].set_title('Concentration Over Time')
                axes[0, 0].set_xlabel('Step')
                axes[0, 0].set_ylabel('Concentration')
                axes[0, 0].grid(True)

                # Reward trajectory
                axes[0, 1].plot(df['reward'], 'ro-', markersize=3)
                axes[0, 1].set_title('Rewards Over Time')
                axes[0, 1].set_xlabel('Step')
                axes[0, 1].set_ylabel('Reward')
                axes[0, 1].grid(True)

                # Action distribution
                action_counts = df['action'].value_counts().sort_index()
                action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
                axes[1, 0].bar(range(len(action_counts)), action_counts.values)
                axes[1, 0].set_title('Action Distribution')
                axes[1, 0].set_xlabel('Action')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].set_xticks(range(len(action_names)))
                axes[1, 0].set_xticklabels(action_names)

                # Cumulative reward
                axes[1, 1].plot(df['reward'].cumsum())
                axes[1, 1].set_title('Cumulative Reward')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Cumulative Reward')
                axes[1, 1].grid(True)

                plt.tight_layout()
                plt.show()

                # Summary statistics
                print("Episode Summary:")
                print(f"Total steps: {{len(df)}}")
                print(f"Total reward: {{df['reward'].sum()}}")
                print(f"Average concentration: {{df['observation'].mean():.4f}}")
                print(f"Goal reached: {{df['terminated'].any()}}")

        return JupyterIntegration(self.base_env)
```

#### Extension Architecture Patterns

The system provides well-defined extension points for research-specific customizations:

```python
# Extension point documentation
class ExtensionArchitecture:
    \"\"\"
    Demonstrates extension patterns and customization points in the architecture.
    Shows how to extend the system while maintaining compatibility and performance.
    \"\"\"

    @staticmethod
    def create_custom_plume_model():
        \"\"\"
        Extension pattern for custom plume models.
        Demonstrates how to extend the domain layer with new mathematical models.
        \"\"\"

        class DynamicWindPlume(BasePlumeModel):
            \"\"\"Custom plume model with wind effects and temporal dynamics.\"\"\"

            def __init__(self, grid_size: GridSize, wind_velocity: Tuple[float, float]):
                super().__init__(grid_size)
                self.wind_velocity = wind_velocity
                self.time_step = 0.0

            def sample_concentration(self, position: Coordinates) -> float:
                \"\"\"Sample concentration with wind-driven dynamics.\"\"\"

                # Time-dependent source location due to wind advection
                effective_source_x = self.source_location.x + self.wind_velocity[0] * self.time_step
                effective_source_y = self.source_location.y + self.wind_velocity[1] * self.time_step

                # Distance to wind-advected source
                dx = position.x - effective_source_x
                dy = position.y - effective_source_y
                distance_squared = dx**2 + dy**2

                # Anisotropic dispersion (different along/across wind)
                sigma_along_wind = 12.0 * (1 + 0.1 * self.time_step)
                sigma_across_wind = 8.0 * (1 + 0.05 * self.time_step)

                # Wind-direction dependent concentration
                concentration = np.exp(
                    -(dx**2 / (2 * sigma_along_wind**2) +
                      dy**2 / (2 * sigma_across_wind**2))
                )

                return float(concentration)

            def update_dynamics(self, delta_time: float):
                \"\"\"Update temporal dynamics - called by environment.\"\"\"
                self.time_step += delta_time

        return DynamicWindPlume

    @staticmethod
    def create_custom_reward_function():
        \"\"\"
        Extension pattern for custom reward functions.
        Shows how to implement domain-specific reward shaping.
        \"\"\"

        class ShapedRewardCalculator(BaseRewardCalculator):
            \"\"\"Custom reward function with dense reward shaping.\"\"\"

            def __init__(self, source_location: Coordinates, goal_radius: float):
                super().__init__(source_location, goal_radius)
                self.previous_distance = None

            def calculate_reward(self, agent_position: Coordinates) -> float:
                \"\"\"Calculate shaped reward with progress incentive.\"\"\"

                # Sparse goal reward (unchanged)
                goal_reward = self._calculate_goal_reward(agent_position)
                if goal_reward > 0:
                    return goal_reward

                # Dense shaping reward based on progress
                current_distance = agent_position.distance_to(self.source_location)

                if self.previous_distance is not None:
                    # Reward for getting closer, penalty for getting farther
                    progress_reward = (self.previous_distance - current_distance) * 0.01
                else:
                    progress_reward = 0.0

                self.previous_distance = current_distance

                # Optional: concentration-based reward
                # concentration_reward = concentration_value * 0.1

                return progress_reward

            def reset_episode_state(self):
                \"\"\"Reset internal state for new episode.\"\"\"
                self.previous_distance = None

        return ShapedRewardCalculator

    @staticmethod
    def create_custom_renderer():
        \"\"\"
        Extension pattern for custom rendering backends.
        Demonstrates how to extend visualization capabilities.
        \"\"\"

        class HeatmapRenderer(BaseRenderer):
            \"\"\"Custom renderer with enhanced heatmap visualization.\"\"\"

            def __init__(self, grid_size: GridSize):
                super().__init__(grid_size)
                self.colormap = plt.cm.plasma  # Custom colormap
                self.trail_history = []  # Agent trail visualization

            def render_frame(self, environment_state: dict) -> np.ndarray:
                \"\"\"Generate enhanced visualization with heatmap and trail.\"\"\"

                # Extract state information
                agent_position = environment_state['agent_position']
                concentration_field = environment_state['concentration_field']
                source_location = environment_state['source_location']

                # Update trail history
                self.trail_history.append(agent_position)
                if len(self.trail_history) > 20:  # Limit trail length
                    self.trail_history.pop(0)

                # Create enhanced visualization
                fig, ax = plt.subplots(figsize=(8, 8))

                # Concentration heatmap with custom colormap
                im = ax.imshow(
                    concentration_field,
                    cmap=self.colormap,
                    alpha=0.8,
                    extent=[0, self.grid_size.width, 0, self.grid_size.height]
                )

                # Agent trail
                if len(self.trail_history) > 1:
                    trail_x = [pos.x for pos in self.trail_history]
                    trail_y = [pos.y for pos in self.trail_history]
                    ax.plot(trail_x, trail_y, 'w--', alpha=0.7, linewidth=2)

                # Current agent position (enhanced marker)
                ax.scatter(
                    agent_position.x, agent_position.y,
                    c='red', s=200, marker='o',
                    edgecolors='white', linewidth=2,
                    zorder=10
                )

                # Source location (enhanced marker)
                ax.scatter(
                    source_location.x, source_location.y,
                    c='gold', s=300, marker='*',
                    edgecolors='black', linewidth=2,
                    zorder=10
                )

                # Enhanced styling
                ax.set_title('Plume Navigation with Agent Trail', fontsize=14)
                ax.set_xlabel('X Position', fontsize=12)
                ax.set_ylabel('Y Position', fontsize=12)

                # Add colorbar
                plt.colorbar(im, ax=ax, label='Concentration')

                # Convert to RGB array
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                plt.close(fig)
                return frame

        return HeatmapRenderer
```

This comprehensive architecture documentation provides developers with deep understanding of the system's
design patterns, performance characteristics, and extension mechanisms for building upon the plume_nav_sim
foundation.
        """)

    # Combine all architecture sections into complete documentation
    complete_architecture_docs = '\n\n'.join(architecture_sections)

    return complete_architecture_docs


def create_development_setup_guide(
    include_advanced_setup: bool = True,
    include_ide_configuration: bool = True,
    include_debugging_setup: bool = True
) -> str:
    """
    Creates detailed development environment setup guide including Python version management, dependency
    installation, virtual environment configuration, development tools, and workflow setup for contributor
    onboarding and productive development work.

    This function generates comprehensive documentation for setting up a complete development environment
    optimized for contributing to plume_nav_sim with all necessary tools, configurations, and workflows.

    Args:
        include_advanced_setup (bool): Include performance profiling tools, debugging utilities, and advanced development configurations
        include_ide_configuration (bool): Include IDE-specific setup for VSCode, PyCharm, and other development environments
        include_debugging_setup (bool): Include debugger configuration, logging setup, and troubleshooting tools

    Returns:
        str: Complete development setup guide with step-by-step instructions, configuration examples,
        and tool integration for productive development workflows
    """
    setup_sections = []

    # Generate development setup overview with system requirements and compatibility matrix
    setup_sections.append(f"""
## Development Environment Setup Guide

### Overview

This guide provides comprehensive instructions for setting up a complete development environment for
plume_nav_sim. It covers everything from basic Python installation to advanced debugging and profiling
tools for productive contribution and development work.

### System Requirements

#### Python Version Requirements

plume_nav_sim requires Python 3.10 or higher for development:

```bash
# Check your current Python version
python --version
python3 --version

# Should output: Python 3.10.x or higher
```

**Supported Python Versions**: {', '.join(DEVELOPMENT_PYTHON_VERSIONS)}

**Operating System Support**:
- **Linux**: Full support (Ubuntu 20.04+, CentOS 8+, Arch Linux)
- **macOS**: Full support (10.15+ with Homebrew or MacPorts)
- **Windows**: Limited support (Windows 10+ with WSL recommended)

#### Hardware Requirements

**Minimum Requirements**:
- RAM: 4GB (8GB recommended for large grid development)
- CPU: Any modern processor (multi-core recommended for testing)
- Storage: 2GB free space for development environment
- Display: Any resolution (higher resolution helpful for visualization development)

**Recommended Development Setup**:
- RAM: 16GB+ for comfortable development with multiple environments
- CPU: 4+ cores for parallel testing and compilation
- SSD: Faster I/O for improved development experience
- GPU: Optional, not required for current implementation

### Environment Setup Methods

Choose one of the following methods based on your preference and system setup:
    """)

    # Create Python environment setup with version management using pyenv or similar tools
    setup_sections.append(f"""
### Method 1: Basic Development Setup

#### Step 1: Python Installation

**Linux (Ubuntu/Debian)**:
```bash
# Update package manager
sudo apt update

# Install Python 3.10+ and development tools
sudo apt install python3.10 python3.10-venv python3.10-dev python3-pip

# Install additional dependencies for matplotlib
sudo apt install python3-tk

# Verify installation
python3.10 --version
```

**macOS (Homebrew)**:
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.10+
brew install python@3.10

# Add to PATH (add to ~/.zshrc or ~/.bash_profile)
export PATH="/opt/homebrew/bin:$PATH"

# Verify installation
python3.10 --version
```

**Windows (Manual Installation)**:
1. Download Python 3.10+ from https://python.org/downloads/
2. Run installer with "Add Python to PATH" checked
3. Install Git for Windows: https://git-scm.com/download/win
4. Open PowerShell or Command Prompt and verify:
```powershell
python --version
git --version
```

#### Step 2: Virtual Environment Setup

```bash
# Navigate to your development directory
cd ~/dev  # or your preferred development location

# Clone the repository (replace with actual repository URL)
git clone https://github.com/SamuelBrudner/plume_nav_sim.git
cd plume_nav_sim

# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
# venv\\Scripts\\activate

# Verify virtual environment
which python  # Should show path within venv directory
python --version  # Should show Python 3.10+
```

#### Step 3: Development Dependencies Installation

```bash
# Ensure you're in the project directory with activated virtual environment
cd plume_nav_sim
source venv/bin/activate  # if not already activated

# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install project in development mode with dev dependencies
pip install -e .[dev]

# Alternative: Install dependencies manually
pip install -e .
pip install pytest>=8.0 black>=22.0 isort>=5.10 mypy>=1.0 flake8>=5.0

# Verify installation
python -c "import plume_nav_sim; print('Installation successful!')"
pytest --version
black --version
```

#### Step 4: Basic Configuration

```bash
# Create development configuration file
cat > dev_config.py << EOF
# Development configuration
DEBUG = True
LOG_LEVEL = 'DEBUG'
PERFORMANCE_MONITORING = True
RENDER_MODE = 'rgb_array'  # Default for development
TEST_SEED = 42
EOF

# Set up Git configuration (if not already done)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Initialize pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

### Method 2: Advanced Setup with pyenv

For contributors working with multiple Python versions:

```bash
# Install pyenv (Linux/macOS)
curl https://pyenv.run | bash

# Add to shell profile (~/.bashrc, ~/.zshrc)
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Restart shell or source profile
source ~/.bashrc  # or ~/.zshrc

# Install Python 3.10 through pyenv
pyenv install 3.10.12
pyenv install 3.11.7
pyenv install 3.12.1

# Set up project-specific Python version
cd plume_nav_sim
pyenv local 3.10.12

# Create virtual environment with pyenv
pyenv virtualenv 3.10.12 plume-nav-dev
pyenv local plume-nav-dev

# Install dependencies
pip install --upgrade pip
pip install -e .[dev]
```
    """)

    # Document development tools setup including pytest, type checking, and linting configuration
    setup_sections.append(f"""
### Development Tools Configuration

#### Code Quality Tools Setup

**Black (Code Formatting)**:
```bash
# Black is already installed with dev dependencies
# Configure in pyproject.toml:
cat >> pyproject.toml << EOF
[tool.black]
line-length = 100
target-version = ['py310']
include = '\\.pyi?$'
extend-exclude = '''
/(
  # directories
  \\.eggs
  | \\.git
  | \\.hg
  | \\.mypy_cache
  | \\.tox
  | \\.venv
  | build
  | dist
)/
'''
EOF

# Format code
black src/ tests/ examples/

# Check formatting without applying changes
black --check src/
```

**isort (Import Sorting)**:
```bash
# Configure isort to work with Black
cat >> pyproject.toml << EOF
[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
combine_as_imports = true
known_first_party = ["plume_nav_sim"]
EOF

# Sort imports
isort src/ tests/ examples/

# Check import sorting
isort --check-only src/
```

**mypy (Static Type Checking)**:
```bash
# Configure mypy for strict type checking
cat > mypy.ini << EOF
[mypy]
python_version = 3.10
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True
show_error_codes = True

[mypy-tests.*]
disallow_untyped_defs = False

[mypy-matplotlib.*]
ignore_missing_imports = True

[mypy-gym.*]
ignore_missing_imports = True
EOF

# Run type checking
mypy src/plume_nav_sim

# Check specific files
mypy src/plume_nav_sim/envs/plume_search_env.py
```

**flake8 (Linting)**:
```bash
# Configure flake8
cat > .flake8 << EOF
[flake8]
max-line-length = 100
max-complexity = 10
ignore =
    # Black compatibility
    E203,  # whitespace before ':'
    E501,  # line too long (handled by black)
    W503,  # line break before binary operator
exclude =
    .git,
    __pycache__,
    build,
    dist,
    *.egg-info,
    venv,
    .venv
per-file-ignores =
    __init__.py:F401
    tests/*:D
EOF

# Run linting
flake8 src/ tests/

# Check specific files
flake8 src/plume_nav_sim/envs/
```

#### Testing Framework Setup

**pytest Configuration**:
```bash
# Create pytest configuration
cat > pytest.ini << EOF
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --verbose
    --tb=short
    --cov=src/plume_nav_sim
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-report=xml
    --cov-fail-under=95
    --strict-markers
    --disable-warnings
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    slow: Slow running tests
    gpu: Tests requiring GPU
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
EOF

# Run tests
pytest tests/

# Run specific test categories
pytest -m unit tests/
pytest -m "not slow" tests/
pytest -k "test_environment" tests/

# Generate coverage report
pytest --cov=src/plume_nav_sim --cov-report=html
# View coverage report: open htmlcov/index.html
```

**Pre-commit Hooks Setup**:
```bash
# Create pre-commit configuration
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 22.12.0
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.991
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        exclude: ^(docs/|tests/)
EOF

# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files

# Update hooks to latest versions
pre-commit autoupdate
```
    """)

    # Include advanced setup if include_advanced_setup enabled with performance profiling and debugging tools
    if include_advanced_setup:
        setup_sections.append(f"""
### Advanced Development Setup

#### Performance Profiling Tools

**cProfile Integration**:
```bash
# Create profiling script
cat > tools/profile_environment.py << EOF
#!/usr/bin/env python3
\"\"\"
Performance profiling script for development optimization.
\"\"\"

import cProfile
import pstats
from pathlib import Path
import sys
sys.path.insert(0, 'src')

from plume_nav_sim.envs.plume_search_env import create_plume_search_env

def profile_environment_performance():
    \"\"\"Profile environment performance with detailed breakdown.\"\"\"

    env = create_plume_search_env(grid_size=(128, 128))

    # Profile environment operations
    def run_episode():
        obs, info = env.reset(seed=42)
        for step in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()

    return run_episode

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    # Run profiled code
    profile_func = profile_environment_performance()
    profile_func()

    profiler.disable()

    # Save profile results
    profiler.dump_stats('profile_results.prof')

    # Display results
    stats = pstats.Stats('profile_results.prof')
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
EOF

# Make executable and run
chmod +x tools/profile_environment.py
python tools/profile_environment.py

# Analyze with snakeviz (install: pip install snakeviz)
snakeviz profile_results.prof
```

**Memory Profiling**:
```bash
# Install memory profiling tools
pip install memory-profiler psutil

# Create memory profiling script
cat > tools/memory_profile.py << EOF
#!/usr/bin/env python3
\"\"\"Memory profiling for development optimization.\"\"\"

from memory_profiler import profile
import sys
sys.path.insert(0, 'src')

from plume_nav_sim.envs.plume_search_env import create_plume_search_env

@profile
def memory_test():
    \"\"\"Profile memory usage of environment operations.\"\"\"

    # Test different grid sizes
    for size in [(32, 32), (64, 64), (128, 128)]:
        print(f"Testing grid size: {{size}}")

        env = create_plume_search_env(grid_size=size)
        obs, info = env.reset(seed=42)

        # Run episode
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        env.close()
        print(f"Completed {{size}} test")

if __name__ == "__main__":
    memory_test()
EOF

# Run memory profiling
python tools/memory_profile.py

# Line-by-line memory profiling
kernprof -l -v tools/memory_profile.py
```

**Benchmarking Infrastructure**:
```bash
# Install benchmarking tools
pip install pytest-benchmark

# Create benchmark tests
mkdir -p tests/benchmarks
cat > tests/benchmarks/test_performance_benchmarks.py << EOF
\"\"\"
Performance benchmark tests for development optimization tracking.
\"\"\"

import pytest
from plume_nav_sim.envs.plume_search_env import create_plume_search_env

class TestEnvironmentBenchmarks:
    \"\"\"Benchmark tests for environment performance.\"\"\"

    @pytest.fixture
    def env(self):
        \"\"\"Create environment for benchmarking.\"\"\"
        return create_plume_search_env(grid_size=(64, 64))

    def test_environment_reset_benchmark(self, benchmark, env):
        \"\"\"Benchmark environment reset performance.\"\"\"

        def reset_env():
            return env.reset(seed=42)

        result = benchmark(reset_env)
        assert result[0] is not None  # obs should not be None

    def test_environment_step_benchmark(self, benchmark, env):
        \"\"\"Benchmark environment step performance.\"\"\"

        env.reset(seed=42)

        def step_env():
            return env.step(0)  # UP action

        result = benchmark(step_env)
        assert len(result) == 5  # obs, reward, terminated, truncated, info

    def test_rendering_benchmark(self, benchmark):
        \"\"\"Benchmark rendering performance.\"\"\"

        env = create_plume_search_env(
            grid_size=(64, 64),
            render_mode='rgb_array'
        )
        env.reset(seed=42)

        def render_env():
            return env.render()

        result = benchmark(render_env)
        assert result is not None
        assert result.shape == (64, 64, 3)  # RGB array

        env.close()
EOF

# Run benchmarks
pytest tests/benchmarks/ --benchmark-only

# Generate benchmark report
pytest tests/benchmarks/ --benchmark-only --benchmark-save=baseline

# Compare benchmarks
pytest tests/benchmarks/ --benchmark-only --benchmark-compare=baseline
```
        """)

    # Add IDE configuration if include_ide_configuration enabled with VSCode, PyCharm, and editor setup
    if include_ide_configuration:
        setup_sections.append(
            """
### IDE Configuration

#### Visual Studio Code Setup

**Installation**:
```bash
# Linux (Ubuntu/Debian)
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update
sudo apt install code

# macOS (Homebrew)
brew install --cask visual-studio-code

# Windows: Download from https://code.visualstudio.com/
```

**VSCode Extensions**:
```bash
# Install essential extensions
code --install-extension ms-python.python
code --install-extension ms-python.black-formatter
code --install-extension ms-python.isort
code --install-extension ms-python.mypy-type-checker
code --install-extension ms-python.flake8
code --install-extension ms-toolsai.jupyter
code --install-extension ms-vscode.test-adapter-converter
code --install-extension hbenl.vscode-test-explorer
code --install-extension ms-vscode.vscode-json
```

**VSCode Configuration**:

Refer to the editor-specific documentation for advanced workspace settings.
"""
        )
