"""
Test-specific configuration module providing specialized configuration factories for different testing categories
including unit tests, integration tests, performance benchmarks, reproducibility validation, and edge case testing.
Contains optimized configurations for fast test execution, deterministic behavior, and comprehensive system
validation with intelligent parameter selection based on test requirements and system capabilities.

This module serves as the centralized configuration hub for all testing scenarios in the plume_nav_sim system,
providing factory functions, metadata classes, and intelligent configuration selection for comprehensive test coverage.
"""

import copy  # >=3.10 - Deep copying of test configurations for parameter modification and isolation
import time  # >=3.10 - Time utilities for validation timestamps and test timing measurements

# External imports with version comments
from dataclasses import (  # >=3.10 - Data class decorators for test configuration structures and metadata classes
    dataclass,
    field,
)
from typing import (  # >=3.10 - Type hints for test configuration management and validation functions
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

# Internal imports from plume_nav_sim core modules
from plume_nav_sim.core.constants import (
    DEFAULT_PLUME_SIGMA,
    MEMORY_LIMIT_TOTAL_MB,
    PERFORMANCE_TARGET_STEP_LATENCY_MS,
)
from plume_nav_sim.utils.exceptions import (
    ConfigurationError,
    ResourceError,
    ValidationError,
)

from .default_config import CompleteConfig

# ===== TEST CONFIGURATION CONSTANTS =====

# Fixed seed values for deterministic reproducibility testing
REPRODUCIBILITY_SEEDS = [42, 123, 456, 789, 999]

# Grid sizes optimized for different test categories
UNIT_TEST_GRID_SIZE = (32, 32)  # Small grid size for fast unit test execution
INTEGRATION_TEST_GRID_SIZE = (
    64,
    64,
)  # Medium grid size for integration testing with realistic parameters
PERFORMANCE_TEST_GRID_SIZE = (
    128,
    128,
)  # Standard grid size for performance benchmarking and validation
STRESS_TEST_GRID_SIZE = (
    256,
    256,
)  # Large grid size for stress testing and scalability validation

# Episode step configurations for different test durations
TEST_EPISODE_STEPS_UNIT = 50  # Short episode length for fast unit test completion
TEST_EPISODE_STEPS_INTEGRATION = (
    200  # Medium episode length for integration testing scenarios
)
TEST_EPISODE_STEPS_PERFORMANCE = (
    1000  # Full episode length for performance benchmark testing
)

# Plume parameters optimized for testing scenarios
TEST_SIGMA_SIMPLE = 8.0  # Simple sigma value for predictable test plume behavior
TEST_SIGMA_COMPLEX = 15.0  # Complex sigma value for advanced testing scenarios

# Performance and timing constants for test execution
TEST_TIMEOUT_MULTIPLIER = (
    2.0  # Multiplier for test timeout calculations based on complexity
)
PERFORMANCE_TOLERANCE = (
    0.1  # Performance tolerance factor for benchmark validation (10%)
)

# Test scenario collections organized by category
UNIT_TEST_SCENARIOS = {
    "basic_functionality": {
        "grid_size": UNIT_TEST_GRID_SIZE,
        "max_steps": TEST_EPISODE_STEPS_UNIT,
        "description": "Basic functionality test with minimal parameters",
    },
    "parameter_validation": {
        "grid_size": UNIT_TEST_GRID_SIZE,
        "max_steps": 10,
        "description": "Parameter validation test with immediate termination",
    },
    "rendering_test": {
        "grid_size": UNIT_TEST_GRID_SIZE,
        "max_steps": 5,
        "render_mode": "rgb_array",
        "description": "Rendering functionality test with minimal steps",
    },
}

INTEGRATION_TEST_SCENARIOS = {
    "end_to_end_navigation": {
        "grid_size": INTEGRATION_TEST_GRID_SIZE,
        "max_steps": TEST_EPISODE_STEPS_INTEGRATION,
        "description": "Complete navigation scenario from start to goal",
    },
    "multi_episode_consistency": {
        "grid_size": INTEGRATION_TEST_GRID_SIZE,
        "max_steps": 100,
        "episode_count": 5,
        "description": "Multiple episode consistency validation",
    },
    "dual_mode_rendering": {
        "grid_size": INTEGRATION_TEST_GRID_SIZE,
        "max_steps": 50,
        "test_both_render_modes": True,
        "description": "Test both RGB array and human rendering modes",
    },
}

PERFORMANCE_TEST_SCENARIOS = {
    "step_latency_benchmark": {
        "grid_size": PERFORMANCE_TEST_GRID_SIZE,
        "max_steps": TEST_EPISODE_STEPS_PERFORMANCE,
        "target_latency_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS,
        "description": "Environment step latency benchmark validation",
    },
    "memory_usage_validation": {
        "grid_size": PERFORMANCE_TEST_GRID_SIZE,
        "max_steps": 100,
        "memory_limit_mb": MEMORY_LIMIT_TOTAL_MB,
        "description": "Memory usage validation under standard load",
    },
    "throughput_measurement": {
        "grid_size": PERFORMANCE_TEST_GRID_SIZE,
        "max_steps": 500,
        "measure_throughput": True,
        "description": "Environment throughput measurement and analysis",
    },
}

EDGE_CASE_TEST_SCENARIOS = {
    "minimal_grid": {
        "grid_size": (16, 16),
        "max_steps": 10,
        "description": "Minimal grid size edge case testing",
    },
    "large_grid_stress": {
        "grid_size": STRESS_TEST_GRID_SIZE,
        "max_steps": 100,
        "description": "Large grid size stress testing",
    },
    "extreme_sigma_values": {
        "grid_size": INTEGRATION_TEST_GRID_SIZE,
        "plume_sigma_range": (0.5, 50.0),
        "description": "Extreme plume sigma value testing",
    },
    "boundary_conditions": {
        "grid_size": UNIT_TEST_GRID_SIZE,
        "test_boundaries": True,
        "description": "Grid boundary condition testing",
    },
}


# ===== TEST CONFIGURATION FACTORY FUNCTIONS =====


def create_unit_test_config(
    grid_size: Optional[Tuple[int, int]] = None,
    max_steps: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
    validate: bool = True,
) -> CompleteConfig:
    """Creates configuration optimized for unit testing with minimal parameters, fast execution, and component
    isolation for rapid test feedback and development iteration.

    Args:
        grid_size: Optional grid size override, defaults to UNIT_TEST_GRID_SIZE
        max_steps: Optional max steps override, defaults to TEST_EPISODE_STEPS_UNIT
        overrides: Optional parameter overrides dictionary
        validate: Whether to validate configuration after creation

    Returns:
        CompleteConfig: Unit test configuration with minimal parameters optimized for fast execution and component isolation

    Raises:
        ConfigurationError: If configuration creation or validation fails
    """
    # Use UNIT_TEST_GRID_SIZE or provided grid_size for fast test execution
    test_grid_size = grid_size if grid_size is not None else UNIT_TEST_GRID_SIZE

    # Set max_steps to TEST_EPISODE_STEPS_UNIT for quick test completion
    test_max_steps = max_steps if max_steps is not None else TEST_EPISODE_STEPS_UNIT

    # Configure simple plume parameters with TEST_SIGMA_SIMPLE for predictable behavior
    test_source_location = (test_grid_size[0] // 2, test_grid_size[1] // 2)

    # Create base configuration optimized for unit testing
    config = CompleteConfig(
        grid_size=test_grid_size,
        source_location=test_source_location,
        plume_sigma=TEST_SIGMA_SIMPLE,
        max_steps=test_max_steps,
        render_mode="rgb_array",  # Fast headless testing without GUI dependencies
        enable_validation=True,
        enable_performance_monitoring=False,  # Disable for test isolation
        deterministic_mode=True,  # Enable for test consistency
        random_seed=REPRODUCIBILITY_SEEDS[0],  # Use first reproducibility seed
        memory_limit_mb=MEMORY_LIMIT_TOTAL_MB * 0.5,  # Reduced memory for unit tests
        step_latency_target_ms=PERFORMANCE_TARGET_STEP_LATENCY_MS
        * TEST_TIMEOUT_MULTIPLIER,
    )

    # Configure minimal performance targets with tolerance for unit testing flexibility
    config.advanced_options.update(
        {
            "test_category": "unit",
            "fast_mode": True,
            "disable_caching": True,  # Disable caching for test isolation and determinism
            "performance_tolerance": PERFORMANCE_TOLERANCE
            * 2,  # Relaxed tolerance for unit tests
        }
    )

    # Apply parameter overrides with validation for custom test scenarios
    if overrides:
        config = config.clone_with_overrides(overrides)

    # Validate complete configuration if validate flag is True
    if validate:
        try:
            config.validate_all(strict_mode=False)
        except Exception as e:
            raise ConfigurationError(
                f"Unit test configuration validation failed: {e}",
                config_parameter="unit_test_config",
                invalid_value=str(config),
            )

    # Update metadata for unit test tracking
    config.metadata.update(
        {
            "config_type": "unit_test",
            "optimization_target": "speed",
            "isolation_level": "high",
        }
    )

    # Return optimized unit test configuration ready for component testing
    return config


def create_integration_test_config(
    grid_size: Optional[Tuple[int, int]] = None,
    render_mode: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    enable_performance_monitoring: bool = False,
) -> CompleteConfig:
    """Creates configuration optimized for integration testing with realistic parameters, cross-component
    interaction, and comprehensive system validation for end-to-end testing scenarios.

    Args:
        grid_size: Optional grid size override, defaults to INTEGRATION_TEST_GRID_SIZE
        render_mode: Optional render mode override, defaults to 'rgb_array'
        overrides: Optional parameter overrides dictionary
        enable_performance_monitoring: Whether to enable performance monitoring for integration analysis

    Returns:
        CompleteConfig: Integration test configuration with realistic parameters for comprehensive cross-component validation

    Raises:
        ConfigurationError: If configuration creation or validation fails
    """
    # Use INTEGRATION_TEST_GRID_SIZE or provided grid_size for realistic system testing
    test_grid_size = grid_size if grid_size is not None else INTEGRATION_TEST_GRID_SIZE

    # Set max_steps to TEST_EPISODE_STEPS_INTEGRATION for thorough interaction testing
    test_max_steps = TEST_EPISODE_STEPS_INTEGRATION

    # Configure realistic plume parameters with balanced complexity for system validation
    test_source_location = (test_grid_size[0] // 2, test_grid_size[1] // 2)
    test_render_mode = render_mode if render_mode is not None else "rgb_array"

    # Create comprehensive integration configuration
    config = CompleteConfig(
        grid_size=test_grid_size,
        source_location=test_source_location,
        plume_sigma=DEFAULT_PLUME_SIGMA,  # Use default for realistic behavior
        max_steps=test_max_steps,
        render_mode=test_render_mode,
        enable_validation=True,
        enable_performance_monitoring=enable_performance_monitoring,
        deterministic_mode=False,  # Allow realistic variability
        random_seed=None,  # Use random initialization for integration realism
        memory_limit_mb=MEMORY_LIMIT_TOTAL_MB,
        step_latency_target_ms=PERFORMANCE_TARGET_STEP_LATENCY_MS,
    )

    # Configure comprehensive validation and consistency checking across components
    config.advanced_options.update(
        {
            "test_category": "integration",
            "realistic_mode": True,
            "enable_caching": True,  # Enable caching for realistic system behavior simulation
            "cross_component_validation": True,
            "performance_tolerance": PERFORMANCE_TOLERANCE,
            "fallback_testing": True,  # Test fallback mechanisms
        }
    )

    # Apply parameter overrides with cross-component validation
    if overrides:
        config = config.clone_with_overrides(overrides)

    # Validate complete integration configuration for system coherence
    try:
        config.validate_all(strict_mode=False)
    except Exception as e:
        raise ConfigurationError(
            f"Integration test configuration validation failed: {e}",
            config_parameter="integration_test_config",
            invalid_value=str(config),
        )

    # Update metadata for integration test tracking
    config.metadata.update(
        {
            "config_type": "integration_test",
            "optimization_target": "realism",
            "system_coverage": "comprehensive",
        }
    )

    # Return comprehensive integration test configuration
    return config


def create_performance_test_config(
    grid_size: Optional[Tuple[int, int]] = None,
    performance_targets: Optional[Dict[str, float]] = None,
    tolerance: Optional[float] = None,
    strict_timing: bool = False,
) -> CompleteConfig:
    """Creates configuration optimized for performance testing with benchmark parameters, timing validation,
    and resource monitoring for system performance verification and optimization validation.

    Args:
        grid_size: Optional grid size override, defaults to PERFORMANCE_TEST_GRID_SIZE
        performance_targets: Optional performance targets dictionary
        tolerance: Optional performance tolerance, defaults to PERFORMANCE_TOLERANCE
        strict_timing: Whether to enable strict timing validation for precise performance measurement

    Returns:
        CompleteConfig: Performance test configuration with benchmark parameters and strict timing validation for system optimization testing

    Raises:
        ConfigurationError: If configuration creation or validation fails
    """
    # Use PERFORMANCE_TEST_GRID_SIZE or provided grid_size for standard benchmark testing
    test_grid_size = grid_size if grid_size is not None else PERFORMANCE_TEST_GRID_SIZE

    # Set max_steps to TEST_EPISODE_STEPS_PERFORMANCE for full performance evaluation
    test_max_steps = TEST_EPISODE_STEPS_PERFORMANCE

    # Configure performance targets from constants or provided parameters with strict validation
    default_targets = {
        "step_latency_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS,
        "render_latency_ms": 5.0,
        "memory_limit_mb": MEMORY_LIMIT_TOTAL_MB,
        "plume_generation_ms": 10.0,
    }
    test_targets = (
        performance_targets if performance_targets is not None else default_targets
    )

    # Set tolerance to PERFORMANCE_TOLERANCE or provided value for benchmark flexibility
    test_tolerance = tolerance if tolerance is not None else PERFORMANCE_TOLERANCE

    # Create performance-optimized configuration
    config = CompleteConfig(
        grid_size=test_grid_size,
        source_location=(test_grid_size[0] // 2, test_grid_size[1] // 2),
        plume_sigma=DEFAULT_PLUME_SIGMA,
        max_steps=test_max_steps,
        render_mode="rgb_array",  # Performance-focused testing without GUI overhead
        enable_validation=True,
        enable_performance_monitoring=True,  # Enable comprehensive performance monitoring and resource tracking
        deterministic_mode=True,  # Consistent performance measurement
        random_seed=REPRODUCIBILITY_SEEDS[2],  # Fixed seed for consistent benchmarks
        memory_limit_mb=test_targets["memory_limit_mb"],
        step_latency_target_ms=test_targets["step_latency_ms"],
    )

    # Configure aggressive performance targets aligned with system specifications
    config.advanced_options.update(
        {
            "test_category": "performance",
            "benchmark_mode": True,
            "strict_timing": strict_timing,
            "performance_targets": test_targets,
            "performance_tolerance": test_tolerance,
            "enable_profiling": True,  # Enable detailed performance logging and resource tracking
            "resource_monitoring": True,
            "optimization_level": "maximum",  # Configure caching and optimization for maximum system performance
        }
    )

    # Set aggressive performance targets for stress testing validation
    if strict_timing:
        config.advanced_options["timing_precision"] = "microsecond"
        config.advanced_options["jitter_tolerance"] = test_tolerance * 0.5

    # Validate performance configuration meets benchmark requirements
    try:
        config.validate_all(strict_mode=True)

        # Additional performance-specific validation
        estimated_memory = (test_grid_size[0] * test_grid_size[1] * 4) / (1024 * 1024)
        if estimated_memory > test_targets["memory_limit_mb"]:
            raise ResourceError(
                f"Performance test memory estimate {estimated_memory:.1f}MB exceeds target {test_targets['memory_limit_mb']}MB",
                resource_type="memory",
                current_usage=estimated_memory,
                limit_exceeded=test_targets["memory_limit_mb"],
            )

    except Exception as e:
        raise ConfigurationError(
            f"Performance test configuration validation failed: {e}",
            config_parameter="performance_test_config",
            invalid_value=str(config),
        )

    # Update metadata for performance test tracking
    config.metadata.update(
        {
            "config_type": "performance_test",
            "optimization_target": "speed",
            "benchmark_category": "standard",
            "timing_requirements": "strict" if strict_timing else "standard",
        }
    )

    # Return performance-optimized benchmark configuration
    return config


def create_reproducibility_test_config(
    seed: Optional[int] = None,
    num_test_episodes: Optional[int] = None,
    deterministic_overrides: Optional[Dict[str, Any]] = None,
    strict_determinism: bool = False,
) -> CompleteConfig:
    """Creates configuration optimized for reproducibility testing with fixed seeding, deterministic parameters,
    and validation for identical episode generation across different execution environments.

    Args:
        seed: Optional seed value, defaults to first REPRODUCIBILITY_SEEDS value
        num_test_episodes: Optional number of test episodes, defaults to 10
        deterministic_overrides: Optional deterministic parameter overrides
        strict_determinism: Whether to enable strict determinism mode for maximum reproducibility control

    Returns:
        CompleteConfig: Reproducibility test configuration with fixed seeding and deterministic parameters for identical episode validation

    Raises:
        ConfigurationError: If configuration creation or validation fails
    """
    # Use provided seed or select from REPRODUCIBILITY_SEEDS for fixed deterministic behavior
    test_seed = seed if seed is not None else REPRODUCIBILITY_SEEDS[0]
    test_episodes = num_test_episodes if num_test_episodes is not None else 10

    # Configure deterministic grid size and source location for consistent initialization
    test_grid_size = INTEGRATION_TEST_GRID_SIZE  # Use stable size for reproducibility
    test_source_location = (test_grid_size[0] // 2, test_grid_size[1] // 2)

    # Create deterministic configuration
    config = CompleteConfig(
        grid_size=test_grid_size,
        source_location=test_source_location,
        plume_sigma=TEST_SIGMA_SIMPLE,  # Set fixed plume parameters with no randomization for exact reproducibility
        max_steps=TEST_EPISODE_STEPS_INTEGRATION,
        render_mode="rgb_array",
        enable_validation=True,
        enable_performance_monitoring=False,  # Disable to avoid timing variations
        deterministic_mode=True,
        random_seed=test_seed,
        memory_limit_mb=MEMORY_LIMIT_TOTAL_MB,
        step_latency_target_ms=PERFORMANCE_TARGET_STEP_LATENCY_MS,
    )

    # Disable all random elements and optimizations that could affect determinism
    config.advanced_options.update(
        {
            "test_category": "reproducibility",
            "deterministic_mode": True,
            "fixed_seed": test_seed,
            "num_test_episodes": test_episodes,
            "disable_randomization": True,
            "disable_optimizations": True,  # Disable optimizations that might affect determinism
            "strict_determinism": strict_determinism,
            "enable_episode_tracking": True,  # Configure comprehensive episode tracking and comparison validation
            "episode_comparison": True,
            "hash_validation": True,  # Enable hash-based episode validation
        }
    )

    # Apply deterministic overrides with validation for custom reproducibility scenarios
    if deterministic_overrides:
        config = config.clone_with_overrides(deterministic_overrides)

    # Enable strict determinism mode if requested for maximum reproducibility control
    if strict_determinism:
        config.advanced_options.update(
            {
                "floating_point_precision": "high",
                "disable_threading": True,
                "force_sequential": True,
                "numerical_stability": "maximum",
            }
        )

    # Validate reproducibility configuration for deterministic behavior guarantee
    try:
        config.validate_all(strict_mode=strict_determinism)

        # Additional reproducibility validation
        if config.random_seed is None:
            raise ValidationError(
                "Reproducibility test requires fixed random seed",
                parameter_name="random_seed",
                invalid_value=None,
                expected_format="integer seed value",
            )

    except Exception as e:
        raise ConfigurationError(
            f"Reproducibility test configuration validation failed: {e}",
            config_parameter="reproducibility_test_config",
            invalid_value=str(config),
        )

    # Update metadata for reproducibility test tracking
    config.metadata.update(
        {
            "config_type": "reproducibility_test",
            "optimization_target": "determinism",
            "reproducibility_level": "strict" if strict_determinism else "standard",
            "test_seed": test_seed,
            "expected_episodes": test_episodes,
        }
    )

    # Return reproducibility-optimized configuration with deterministic guarantees
    return config


def create_edge_case_test_config(
    edge_case_type: str,
    extreme_params: Optional[Dict[str, Any]] = None,
    boundary_conditions: Optional[Dict[str, Any]] = None,
    enable_error_monitoring: bool = False,
) -> CompleteConfig:
    """Creates configuration optimized for edge case testing with extreme parameters, boundary conditions,
    and stress scenarios for robustness validation and error handling testing.

    Args:
        edge_case_type: Type of edge case testing to configure
        extreme_params: Optional extreme parameter overrides
        boundary_conditions: Optional boundary condition specifications
        enable_error_monitoring: Whether to enable comprehensive error monitoring and exception tracking

    Returns:
        CompleteConfig: Edge case test configuration with extreme parameters and boundary conditions for robustness validation

    Raises:
        ValidationError: If edge_case_type is not supported
        ConfigurationError: If configuration creation fails
    """
    # Validate edge_case_type against supported edge case categories
    supported_edge_cases = [
        "minimal_grid",
        "large_grid_stress",
        "extreme_sigma_values",
        "boundary_conditions",
        "memory_limits",
        "timeout_stress",
    ]

    if edge_case_type not in supported_edge_cases:
        raise ValidationError(
            f"Unsupported edge case type: {edge_case_type}",
            parameter_name="edge_case_type",
            invalid_value=edge_case_type,
            expected_format=f"One of: {', '.join(supported_edge_cases)}",
        )

    # Configure extreme parameters based on edge case type (min/max values, boundary conditions)
    if edge_case_type == "minimal_grid":
        base_config = {
            "grid_size": (16, 16),
            "source_location": (8, 8),
            "max_steps": 50,
            "plume_sigma": 2.0,
        }
    elif edge_case_type == "large_grid_stress":
        base_config = {
            "grid_size": STRESS_TEST_GRID_SIZE,
            "source_location": (
                STRESS_TEST_GRID_SIZE[0] // 2,
                STRESS_TEST_GRID_SIZE[1] // 2,
            ),
            "max_steps": 500,
            "plume_sigma": 25.0,
        }
    elif edge_case_type == "extreme_sigma_values":
        base_config = {
            "grid_size": INTEGRATION_TEST_GRID_SIZE,
            "source_location": (
                INTEGRATION_TEST_GRID_SIZE[0] // 2,
                INTEGRATION_TEST_GRID_SIZE[1] // 2,
            ),
            "max_steps": 100,
            "plume_sigma": 0.5,  # Very small sigma for sharp concentration gradients
        }
    elif edge_case_type == "boundary_conditions":
        base_config = {
            "grid_size": UNIT_TEST_GRID_SIZE,
            "source_location": (0, 0),  # Source at corner boundary
            "max_steps": 100,
            "plume_sigma": 5.0,
        }
    elif edge_case_type == "memory_limits":
        base_config = {
            "grid_size": (200, 200),  # Large enough to approach memory limits
            "source_location": (100, 100),
            "max_steps": 50,
            "plume_sigma": 15.0,
        }
    else:  # timeout_stress
        base_config = {
            "grid_size": PERFORMANCE_TEST_GRID_SIZE,
            "source_location": (
                PERFORMANCE_TEST_GRID_SIZE[0] // 2,
                PERFORMANCE_TEST_GRID_SIZE[1] // 2,
            ),
            "max_steps": 5000,  # Very long episode for timeout testing
            "plume_sigma": 20.0,
        }

    # Apply extreme parameter overrides if provided
    if extreme_params:
        base_config.update(extreme_params)

    # Create edge case configuration with extreme parameters
    config = CompleteConfig(
        grid_size=base_config["grid_size"],
        source_location=base_config["source_location"],
        plume_sigma=base_config["plume_sigma"],
        max_steps=base_config["max_steps"],
        render_mode="rgb_array",
        enable_validation=True,
        enable_performance_monitoring=enable_error_monitoring,
        deterministic_mode=True,
        random_seed=REPRODUCIBILITY_SEEDS[
            3
        ],  # Fixed seed for edge case reproducibility
        memory_limit_mb=MEMORY_LIMIT_TOTAL_MB,
        step_latency_target_ms=PERFORMANCE_TARGET_STEP_LATENCY_MS
        * 5,  # Relaxed timing for edge cases
    )

    # Set up safety limits and recovery mechanisms for extreme parameter testing
    config.advanced_options.update(
        {
            "test_category": "edge_case",
            "edge_case_type": edge_case_type,
            "enable_error_monitoring": enable_error_monitoring,
            "safety_limits": True,
            "timeout_multiplier": TEST_TIMEOUT_MULTIPLIER
            * 3,  # Extended timeout for edge cases
            "memory_monitoring": True,
            "graceful_degradation": True,  # Enable graceful handling of extreme conditions
            "detailed_logging": True,  # Enable detailed logging and error reporting for edge case analysis
        }
    )

    # Set boundary conditions for grid edges, coordinate limits, and parameter bounds
    if boundary_conditions:
        config.advanced_options["boundary_conditions"] = boundary_conditions
    else:
        # Default boundary conditions based on edge case type
        if edge_case_type == "boundary_conditions":
            config.advanced_options["boundary_conditions"] = {
                "test_grid_edges": True,
                "test_corner_positions": True,
                "test_source_at_boundary": True,
            }

    # Configure timeout and resource limits for edge case test protection
    config.advanced_options.update(
        {
            "resource_limits": {
                "max_memory_mb": MEMORY_LIMIT_TOTAL_MB
                * 1.5,  # Allow some overrun for edge case testing
                "max_execution_time_s": 300,  # 5 minute timeout for edge cases
                "max_iterations": config.max_steps * 2,
            }
        }
    )

    # Validate edge case configuration maintains system safety while testing limits
    try:
        # Skip strict validation for edge cases as they test limits
        config.validate_all(strict_mode=False)

        # Additional safety checks for edge case testing
        if edge_case_type == "memory_limits":
            estimated_memory = (config.grid_size[0] * config.grid_size[1] * 4) / (
                1024 * 1024
            )
            if (
                estimated_memory > MEMORY_LIMIT_TOTAL_MB * 2
            ):  # 2x limit is too dangerous
                raise ResourceError(
                    f"Edge case memory requirement {estimated_memory:.1f}MB exceeds safety limit",
                    resource_type="memory",
                    current_usage=estimated_memory,
                    limit_exceeded=MEMORY_LIMIT_TOTAL_MB * 2,
                )

    except ResourceError:
        raise  # Re-raise resource errors
    except Exception as e:
        # For edge cases, log validation errors but don't fail configuration creation
        config.advanced_options["validation_warnings"] = [str(e)]

    # Update metadata for edge case test tracking
    config.metadata.update(
        {
            "config_type": "edge_case_test",
            "optimization_target": "robustness",
            "edge_case_category": edge_case_type,
            "safety_level": "monitored",
        }
    )

    # Return edge case configuration optimized for robustness testing
    return config


def get_test_config_for_category(
    test_category: str,
    category_overrides: Optional[Dict[str, Any]] = None,
    auto_optimize: bool = False,
    detect_system_capabilities: bool = False,
) -> CompleteConfig:
    """Intelligent factory function that creates appropriate test configuration based on test category with
    automatic parameter optimization and system capability detection for intelligent test setup.

    Args:
        test_category: Test category name for configuration selection
        category_overrides: Optional category-specific parameter overrides
        auto_optimize: Whether to enable automatic parameter optimization
        detect_system_capabilities: Whether to detect and optimize for system capabilities

    Returns:
        CompleteConfig: Test configuration automatically optimized for specified category with intelligent parameter selection

    Raises:
        ValidationError: If test_category is not supported
        ConfigurationError: If configuration creation fails
    """
    # Validate test_category against supported categories (unit, integration, performance, reproducibility, edge_case)
    supported_categories = [
        "unit",
        "integration",
        "performance",
        "reproducibility",
        "edge_case",
        "minimal",
        "stress",
    ]

    if test_category not in supported_categories:
        raise ValidationError(
            f"Unsupported test category: {test_category}",
            parameter_name="test_category",
            invalid_value=test_category,
            expected_format=f"One of: {', '.join(supported_categories)}",
        )

    # Detect system capabilities if requested (memory, CPU, rendering backends)
    system_capabilities = {}
    if detect_system_capabilities:
        try:
            import psutil

            system_capabilities = {
                "memory_gb": psutil.virtual_memory().total / (1024**3),
                "cpu_count": psutil.cpu_count(),
                "has_display": True,  # Simplified detection
            }
        except ImportError:
            # Fallback capability detection without psutil
            system_capabilities = {
                "memory_gb": 8.0,  # Conservative estimate
                "cpu_count": 4,  # Conservative estimate
                "has_display": True,
            }

    # Select appropriate configuration factory based on test category
    if test_category == "unit":
        base_config = create_unit_test_config()
    elif test_category == "integration":
        base_config = create_integration_test_config()
    elif test_category == "performance":
        base_config = create_performance_test_config()
    elif test_category == "reproducibility":
        base_config = create_reproducibility_test_config()
    elif test_category == "edge_case":
        # Default to boundary conditions edge case
        base_config = create_edge_case_test_config("boundary_conditions")
    elif test_category == "minimal":
        base_config = create_minimal_test_config()
    else:  # stress
        base_config = create_stress_test_config()

    # Apply auto-optimization for system-specific parameter tuning if enabled
    if auto_optimize and system_capabilities:
        optimization_overrides = {}

        # Optimize based on available memory
        if "memory_gb" in system_capabilities:
            memory_gb = system_capabilities["memory_gb"]
            if memory_gb < 4:  # Low memory system
                optimization_overrides["memory_limit_mb"] = 25
                if test_category in ["performance", "stress"]:
                    optimization_overrides["grid_size"] = INTEGRATION_TEST_GRID_SIZE
            elif memory_gb > 16:  # High memory system
                optimization_overrides["memory_limit_mb"] = 100
                if test_category == "stress":
                    optimization_overrides["grid_size"] = (512, 512)

        # Optimize based on CPU capabilities
        if "cpu_count" in system_capabilities and system_capabilities["cpu_count"] > 8:
            optimization_overrides["enable_performance_monitoring"] = True

        # Apply optimizations
        if optimization_overrides:
            base_config = base_config.clone_with_overrides(optimization_overrides)

    # Merge category-specific defaults with provided overrides
    if category_overrides:
        base_config = base_config.clone_with_overrides(category_overrides)

    # Validate final configuration meets category requirements and system constraints
    try:
        base_config.validate_all(strict_mode=(test_category == "performance"))
    except Exception as e:
        raise ConfigurationError(
            f"Category configuration validation failed for {test_category}: {e}",
            config_parameter="test_category",
            invalid_value=test_category,
        )

    # Update metadata with intelligent configuration information
    base_config.metadata.update(
        {
            "intelligent_config": True,
            "auto_optimized": auto_optimize,
            "system_optimized": detect_system_capabilities,
            "category": test_category,
        }
    )

    if system_capabilities:
        base_config.metadata["system_capabilities"] = system_capabilities

    # Return intelligently optimized configuration for specified test category
    return base_config


def validate_test_configuration(
    test_config: CompleteConfig,
    test_category: str,
    strict_validation: bool = False,
    system_constraints: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """Comprehensive validation function for test configurations ensuring parameter consistency, system
    compatibility, and test-specific requirements with detailed error reporting and recovery suggestions.

    Args:
        test_config: Test configuration to validate
        test_category: Test category for category-specific validation
        strict_validation: Whether to enable strict validation with additional checks
        system_constraints: Optional system-specific constraints

    Returns:
        tuple: Tuple of (is_valid: bool, validation_report: dict, recovery_suggestions: List[str])

    Raises:
        ValidationError: If test_config is not a CompleteConfig instance
    """
    # Validate test configuration using CompleteConfig.validate_all method
    if not isinstance(test_config, CompleteConfig):
        raise ValidationError(
            f"Test config must be CompleteConfig instance, got {type(test_config).__name__}",
            parameter_name="test_config",
            invalid_value=test_config,
            expected_format="CompleteConfig instance",
        )

    validation_report = {
        "validation_timestamp": time.time(),
        "test_category": test_category,
        "strict_mode": strict_validation,
        "issues_found": [],
        "warnings": [],
        "compatibility_check": {},
        "performance_feasibility": {},
        "category_compliance": {},
    }

    recovery_suggestions = []
    is_valid = True

    try:
        # Basic configuration validation
        test_config.validate_all(strict_mode=strict_validation)
        validation_report["basic_validation"] = "passed"

    except Exception as e:
        validation_report["basic_validation"] = "failed"
        validation_report["issues_found"].append(f"Basic validation failed: {str(e)}")
        recovery_suggestions.append("Fix basic configuration parameters")
        is_valid = False

    # Check configuration meets test category specific requirements and constraints
    category_requirements = {
        "unit": {
            "max_grid_size": UNIT_TEST_GRID_SIZE,
            "max_steps": TEST_EPISODE_STEPS_UNIT * 2,
            "required_mode": "rgb_array",
        },
        "integration": {
            "min_grid_size": UNIT_TEST_GRID_SIZE,
            "max_steps": TEST_EPISODE_STEPS_INTEGRATION * 2,
            "memory_limit": MEMORY_LIMIT_TOTAL_MB,
        },
        "performance": {
            "min_grid_size": PERFORMANCE_TEST_GRID_SIZE,
            "required_monitoring": True,
            "strict_timing": True,
        },
        "reproducibility": {
            "required_seed": True,
            "deterministic_mode": True,
            "no_randomization": True,
        },
    }

    if test_category in category_requirements:
        requirements = category_requirements[test_category]

        # Check category-specific requirements
        if "max_grid_size" in requirements:
            if (
                test_config.grid_size[0] > requirements["max_grid_size"][0]
                or test_config.grid_size[1] > requirements["max_grid_size"][1]
            ):
                validation_report["issues_found"].append(
                    f"Grid size exceeds {test_category} test limits"
                )
                recovery_suggestions.append(
                    f"Reduce grid size to {requirements['max_grid_size']} or smaller"
                )
                is_valid = False

        if (
            "required_seed" in requirements
            and requirements["required_seed"]
            and test_config.random_seed is None
        ):
            validation_report["issues_found"].append(
                "Reproducibility tests require fixed seed"
            )
            recovery_suggestions.append("Set random_seed to a fixed integer value")
            is_valid = False

        if (
            "required_monitoring" in requirements
            and requirements["required_monitoring"]
        ) and not test_config.enable_performance_monitoring:
            validation_report["warnings"].append(
                "Performance tests should enable monitoring"
            )
            recovery_suggestions.append(
                "Enable performance monitoring for accurate measurements"
            )

    # Validate system compatibility including memory limits and performance capabilities
    if system_constraints:
        constraint_issues = []

        if "max_memory_mb" in system_constraints:
            estimated_memory = (
                test_config.grid_size[0] * test_config.grid_size[1] * 4
            ) / (1024 * 1024)
            if estimated_memory > system_constraints["max_memory_mb"]:
                constraint_issues.append(
                    f"Memory estimate {estimated_memory:.1f}MB exceeds system limit"
                )
                recovery_suggestions.append("Reduce grid size or increase memory limit")
                is_valid = False

        if "max_execution_time_s" in system_constraints:
            estimated_time = test_config.max_steps * 0.001  # Rough estimate
            if estimated_time > system_constraints["max_execution_time_s"]:
                constraint_issues.append("Estimated execution time exceeds limit")
                recovery_suggestions.append("Reduce max_steps or increase timeout")

        validation_report["compatibility_check"] = {
            "system_constraints_met": len(constraint_issues) == 0,
            "constraint_violations": constraint_issues,
        }

    # Apply strict validation rules if strict_validation is enabled
    if strict_validation:
        strict_issues = []

        # Check parameter ranges are appropriate for test category
        if test_category == "performance":
            if (
                test_config.step_latency_target_ms
                > PERFORMANCE_TARGET_STEP_LATENCY_MS * 2
            ):
                strict_issues.append("Performance test latency target too relaxed")
                recovery_suggestions.append(
                    "Tighten step latency target for performance testing"
                )

        if test_category == "unit":
            if test_config.max_steps > TEST_EPISODE_STEPS_UNIT * 3:
                strict_issues.append("Unit test episode too long for fast execution")
                recovery_suggestions.append("Reduce max_steps for faster unit testing")

        validation_report["strict_validation"] = {
            "enabled": True,
            "issues_found": strict_issues,
        }

        if strict_issues:
            validation_report["issues_found"].extend(strict_issues)
            is_valid = False

    # Validate cross-component consistency and parameter dependencies
    consistency_issues = []

    # Check source location within grid bounds
    if (
        test_config.source_location[0] >= test_config.grid_size[0]
        or test_config.source_location[1] >= test_config.grid_size[1]
    ):
        consistency_issues.append("Source location outside grid boundaries")
        recovery_suggestions.append("Move source location within grid bounds")
        is_valid = False

    # Check plume sigma reasonable for grid size
    grid_diagonal = (
        test_config.grid_size[0] ** 2 + test_config.grid_size[1] ** 2
    ) ** 0.5
    if test_config.plume_sigma > grid_diagonal / 2:
        consistency_issues.append("Plume sigma too large for grid size")
        recovery_suggestions.append("Reduce plume sigma or increase grid size")
        validation_report["warnings"].append(
            "Large plume sigma may affect gradient visibility"
        )

    validation_report["consistency_check"] = {
        "cross_component_validation": len(consistency_issues) == 0,
        "consistency_issues": consistency_issues,
    }

    if consistency_issues:
        validation_report["issues_found"].extend(consistency_issues)
        is_valid = False

    # Generate detailed validation report with specific issues and constraint violations
    validation_report["overall_valid"] = is_valid
    validation_report["total_issues"] = len(validation_report["issues_found"])
    validation_report["total_warnings"] = len(validation_report["warnings"])

    # Create recovery suggestions for identified configuration problems
    if not recovery_suggestions and not is_valid:
        recovery_suggestions.append(
            "Review configuration parameters and test category requirements"
        )

    # Add general suggestions based on validation results
    if validation_report["total_warnings"] > 0:
        recovery_suggestions.append("Address warnings to improve test reliability")

    if strict_validation and is_valid:
        recovery_suggestions.append(
            "Configuration passed strict validation - ready for testing"
        )

    # Return comprehensive validation results with actionable feedback
    return is_valid, validation_report, recovery_suggestions


def create_minimal_test_config(
    seed: Optional[int] = None, headless_only: bool = False
) -> CompleteConfig:
    """Creates minimal test configuration with absolute minimum parameters for rapid smoke testing, basic API
    compliance, and quick system validation with maximum execution speed.

    Args:
        seed: Optional seed value for consistency, defaults to first REPRODUCIBILITY_SEEDS
        headless_only: Whether to force RGB_ARRAY render mode for headless testing

    Returns:
        CompleteConfig: Minimal test configuration optimized for maximum speed and basic functionality validation

    Raises:
        ConfigurationError: If minimal configuration creation fails
    """
    # Configure smallest possible grid size (16, 16) for maximum speed
    minimal_grid_size = (16, 16)
    minimal_source_location = (8, 8)  # Center of minimal grid

    # Set minimal episode steps (10) for immediate test completion
    minimal_max_steps = 10

    # Set provided seed or use first REPRODUCIBILITY_SEEDS value for consistency
    test_seed = seed if seed is not None else REPRODUCIBILITY_SEEDS[0]

    # Create ultra-minimal configuration
    config = CompleteConfig(
        grid_size=minimal_grid_size,
        source_location=minimal_source_location,
        plume_sigma=2.0,  # Configure simple plume parameters with minimal computational complexity
        max_steps=minimal_max_steps,
        render_mode="rgb_array",  # Force RGB_ARRAY render mode if headless_only is True
        enable_validation=False,  # Skip comprehensive validation for maximum setup speed
        enable_performance_monitoring=False,  # Disable all performance monitoring and complex optimizations
        deterministic_mode=True,
        random_seed=test_seed,
        memory_limit_mb=10,  # Configure minimal memory usage and resource requirements
        step_latency_target_ms=10.0,  # Relaxed timing for smoke testing
    )

    # Configure ultra-fast minimal settings
    config.advanced_options.update(
        {
            "test_category": "minimal",
            "smoke_test_mode": True,
            "skip_validation": True,
            "minimal_complexity": True,
            "disable_caching": True,
            "disable_optimizations": True,
            "fast_mode": True,
            "minimal_logging": True,
        }
    )

    # Force headless mode if requested
    if headless_only:
        config.render_mode = "rgb_array"
        config.advanced_options["headless_only"] = True
        config.advanced_options["no_display_required"] = True

    # Update metadata for minimal test tracking
    config.metadata.update(
        {
            "config_type": "minimal_test",
            "optimization_target": "speed",
            "complexity_level": "minimal",
            "purpose": "smoke_testing",
        }
    )

    # Return ultra-fast minimal configuration for smoke testing
    return config


def create_stress_test_config(
    max_grid_size: Optional[Tuple[int, int]] = None,
    max_episodes: Optional[int] = None,
    enable_resource_monitoring: bool = False,
    stress_overrides: Optional[Dict[str, Any]] = None,
) -> CompleteConfig:
    """Creates stress test configuration with maximum parameters for scalability testing, resource limit
    validation, and system capacity verification under heavy computational load.

    Args:
        max_grid_size: Optional maximum grid size, defaults to STRESS_TEST_GRID_SIZE
        max_episodes: Optional number of episodes for stress testing
        enable_resource_monitoring: Whether to enable comprehensive resource monitoring for stress analysis
        stress_overrides: Optional stress-specific parameter overrides

    Returns:
        CompleteConfig: Stress test configuration with maximum parameters for scalability and resource limit testing

    Raises:
        ResourceError: If stress configuration exceeds system safety limits
        ConfigurationError: If stress configuration creation fails
    """
    # Use STRESS_TEST_GRID_SIZE or provided max_grid_size for maximum computational load
    test_grid_size = (
        max_grid_size if max_grid_size is not None else STRESS_TEST_GRID_SIZE
    )
    test_source_location = (test_grid_size[0] // 2, test_grid_size[1] // 2)

    # Configure maximum episode steps for comprehensive stress testing duration
    stress_max_steps = 1000  # Long episodes for sustained stress testing
    test_episodes = max_episodes if max_episodes is not None else 5

    # Create maximum-load stress configuration
    config = CompleteConfig(
        grid_size=test_grid_size,
        source_location=test_source_location,
        plume_sigma=TEST_SIGMA_COMPLEX,  # Set complex plume parameters with high computational requirements
        max_steps=stress_max_steps,
        render_mode="rgb_array",
        enable_validation=True,
        enable_performance_monitoring=enable_resource_monitoring,  # Enable comprehensive resource monitoring if requested for stress analysis
        deterministic_mode=True,
        random_seed=REPRODUCIBILITY_SEEDS[4],  # Last seed for stress testing
        memory_limit_mb=MEMORY_LIMIT_TOTAL_MB
        * 2,  # Configure maximum memory usage approaching system limits
        step_latency_target_ms=PERFORMANCE_TARGET_STEP_LATENCY_MS
        * 5,  # Relaxed timing for heavy load
    )

    # Configure stress testing with maximum system load
    config.advanced_options.update(
        {
            "test_category": "stress",
            "stress_test_mode": True,
            "max_episodes": test_episodes,
            "heavy_computation": True,
            "resource_monitoring": enable_resource_monitoring,
            "memory_stress": True,
            "computational_stress": True,
            "sustained_load": True,
            "performance_tracking": True,  # Enable detailed performance logging and resource tracking
            "timeout_multiplier": TEST_TIMEOUT_MULTIPLIER
            * 4,  # Configure timeout and safety limits for stress test protection
            "safety_monitoring": True,
        }
    )

    # Set aggressive performance targets for stress testing validation
    config.advanced_options["stress_targets"] = {
        "max_memory_mb": MEMORY_LIMIT_TOTAL_MB * 1.8,
        "max_step_latency_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS * 10,
        "min_throughput_steps_per_second": 100,
    }

    # Apply stress-specific parameter overrides with safety validation
    if stress_overrides:
        config = config.clone_with_overrides(stress_overrides)

    # Validate stress configuration maintains system stability under load
    try:
        # Check memory estimate doesn't exceed safety limits
        estimated_memory = (test_grid_size[0] * test_grid_size[1] * 4) / (1024 * 1024)
        if estimated_memory > MEMORY_LIMIT_TOTAL_MB * 3:  # 3x limit is dangerous
            raise ResourceError(
                f"Stress test memory estimate {estimated_memory:.1f}MB exceeds safety limit",
                resource_type="memory",
                current_usage=estimated_memory,
                limit_exceeded=MEMORY_LIMIT_TOTAL_MB * 3,
            )

        # Validate with relaxed constraints for stress testing
        config.validate_all(strict_mode=False)

    except ResourceError:
        raise  # Re-raise resource errors for safety
    except Exception as e:
        raise ConfigurationError(
            f"Stress test configuration validation failed: {e}",
            config_parameter="stress_test_config",
            invalid_value=str(config),
        )

    # Update metadata for stress test tracking
    config.metadata.update(
        {
            "config_type": "stress_test",
            "optimization_target": "scalability",
            "load_level": "maximum",
            "safety_monitored": True,
            "estimated_memory_mb": estimated_memory,
        }
    )

    # Return stress-optimized configuration for scalability testing
    return config


# ===== TEST CONFIGURATION METADATA AND FACTORY CLASSES =====


@dataclass(frozen=True)
class TestConfigMetadata:
    """Data class containing metadata for test configurations including test category, execution requirements,
    performance expectations, and validation criteria for test configuration management and optimization.

    This immutable class provides comprehensive metadata for test configurations, enabling intelligent
    test selection, resource planning, and execution optimization.
    """

    __test__ = False

    # Required metadata fields
    category: str
    description: str
    requirements: Dict[str, Any]
    performance_expectations: Dict[str, Any]

    # Optional metadata fields with defaults
    system_constraints: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: float = 0.0
    dependencies: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize test configuration metadata with category validation and requirement processing."""
        # Store test category with validation against supported categories
        supported_categories = [
            "unit",
            "integration",
            "performance",
            "reproducibility",
            "edge_case",
            "minimal",
            "stress",
        ]
        if self.category not in supported_categories:
            raise ValidationError(
                f"Invalid test category: {self.category}",
                parameter_name="category",
                invalid_value=self.category,
                expected_format=f"One of: {', '.join(supported_categories)}",
            )

        # Initialize system constraints based on category and requirements
        default_constraints = {}
        if self.category == "performance":
            default_constraints = {
                "min_memory_mb": 100,
                "monitoring_required": True,
                "timing_precision": "millisecond",
            }
        elif self.category == "stress":
            default_constraints = {
                "min_memory_mb": 200,
                "safety_monitoring": True,
                "resource_tracking": True,
            }

        # Merge with provided system constraints
        if not self.system_constraints:
            object.__setattr__(self, "system_constraints", default_constraints)

        # Estimate test duration based on complexity and system capabilities
        if self.estimated_duration == 0.0:
            duration_estimate = self._estimate_duration()
            object.__setattr__(self, "estimated_duration", duration_estimate)

        # Initialize dependencies list for test execution requirements
        if not self.dependencies:
            category_dependencies = {
                "performance": ["psutil", "time"],
                "stress": ["psutil", "memory_profiler"],
                "integration": ["matplotlib"],
            }
            default_deps = category_dependencies.get(self.category, [])
            object.__setattr__(self, "dependencies", default_deps)

    def _estimate_duration(self) -> float:
        """Estimate test execution duration based on category and requirements."""
        base_durations = {
            "unit": 1.0,  # 1 second base for unit tests
            "integration": 5.0,  # 5 seconds base for integration
            "performance": 30.0,  # 30 seconds for performance benchmarks
            "reproducibility": 15.0,  # 15 seconds for reproducibility
            "edge_case": 10.0,  # 10 seconds for edge cases
            "minimal": 0.5,  # 0.5 seconds for minimal/smoke tests
            "stress": 120.0,  # 2 minutes for stress tests
        }

        base_duration = base_durations.get(self.category, 5.0)

        # Adjust based on requirements complexity
        if "grid_size" in self.requirements:
            grid_size = self.requirements["grid_size"]
            if isinstance(grid_size, (tuple, list)) and len(grid_size) == 2:
                complexity_factor = (grid_size[0] * grid_size[1]) / (
                    64 * 64
                )  # Relative to 64x64 base
                base_duration *= min(complexity_factor, 4.0)  # Cap at 4x multiplier

        return base_duration

    def estimate_execution_time(
        self, test_config: CompleteConfig, system_specs: Optional[Dict[str, Any]] = None
    ) -> float:
        """Estimates test execution time based on configuration complexity, system capabilities, and
        performance expectations.

        Args:
            test_config: Test configuration for execution time estimation
            system_specs: Optional system specifications for performance adjustment

        Returns:
            float: Estimated test execution time in seconds
        """
        # Calculate complexity factors from test configuration parameters
        base_time = self.estimated_duration

        # Grid size complexity factor
        grid_cells = test_config.grid_size[0] * test_config.grid_size[1]
        grid_factor = grid_cells / (64 * 64)  # Normalize to 64x64 baseline

        # Episode length factor
        step_factor = test_config.max_steps / 100  # Normalize to 100 steps baseline

        # Apply complexity adjustments
        complexity_time = base_time * min(grid_factor, 5.0) * min(step_factor, 10.0)

        # Factor in system capabilities if system_specs provided
        if system_specs:
            # Memory factor
            if "memory_gb" in system_specs:
                memory_factor = max(
                    0.5, min(2.0, system_specs["memory_gb"] / 8.0)
                )  # Normalize to 8GB
                complexity_time /= memory_factor

            # CPU factor
            if "cpu_count" in system_specs:
                cpu_factor = max(
                    0.5, min(2.0, system_specs["cpu_count"] / 4.0)
                )  # Normalize to 4 cores
                complexity_time /= cpu_factor

        # Apply category-specific timing multipliers and overhead
        category_multipliers = {
            "performance": 1.5,  # Additional overhead for timing measurements
            "reproducibility": 1.2,  # Multiple episodes for validation
            "stress": 2.0,  # Extended execution under load
            "edge_case": 1.3,  # Additional safety margins
        }

        final_time = complexity_time * category_multipliers.get(self.category, 1.0)

        # Include performance expectation adjustments and tolerances
        if "timeout_multiplier" in self.performance_expectations:
            final_time *= self.performance_expectations["timeout_multiplier"]

        # Return estimated execution time with confidence interval
        return max(0.1, final_time)  # Minimum 0.1 seconds

    def validate_system_compatibility(
        self, system_capabilities: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validates test configuration compatibility with system constraints and capability requirements.

        Args:
            system_capabilities: Dictionary of system capabilities and resources

        Returns:
            tuple: Tuple of (is_compatible: bool, compatibility_report: dict)
        """
        compatibility_report = {
            "overall_compatible": True,
            "capability_checks": {},
            "constraint_violations": [],
            "warnings": [],
            "recommendations": [],
        }

        # Check system memory meets test configuration requirements
        if (
            "memory_gb" in system_capabilities
            and "min_memory_mb" in self.system_constraints
        ):
            available_mb = system_capabilities["memory_gb"] * 1024
            required_mb = self.system_constraints["min_memory_mb"]

            if available_mb < required_mb:
                compatibility_report["overall_compatible"] = False
                compatibility_report["constraint_violations"].append(
                    f"Insufficient memory: {available_mb:.0f}MB available, {required_mb}MB required"
                )
                compatibility_report["recommendations"].append(
                    "Increase available memory or use smaller grid size"
                )
            else:
                compatibility_report["capability_checks"]["memory"] = "sufficient"

        # Validate CPU capabilities for performance expectations
        if "cpu_count" in system_capabilities and self.category == "performance":
            if system_capabilities["cpu_count"] < 2:
                compatibility_report["warnings"].append(
                    "Performance tests may be unreliable on single-core systems"
                )
                compatibility_report["recommendations"].append(
                    "Use multi-core system for performance benchmarks"
                )
            else:
                compatibility_report["capability_checks"]["cpu"] = "adequate"

        # Check rendering backend availability for visualization tests
        if "has_display" in system_capabilities:
            if (
                not system_capabilities["has_display"]
                and "rendering" in self.requirements
            ):
                compatibility_report["warnings"].append(
                    "No display detected - using RGB array mode only"
                )
                compatibility_report["recommendations"].append(
                    "Enable headless mode or use system with display"
                )

        # Validate dependency availability and version compatibility
        missing_deps = []
        for dep in self.dependencies:
            try:
                __import__(dep)
                compatibility_report["capability_checks"][
                    f"dependency_{dep}"
                ] = "available"
            except ImportError:
                missing_deps.append(dep)

        if missing_deps:
            compatibility_report["overall_compatible"] = False
            compatibility_report["constraint_violations"].append(
                f"Missing dependencies: {', '.join(missing_deps)}"
            )
            compatibility_report["recommendations"].append(
                f"Install missing dependencies: {', '.join(missing_deps)}"
            )

        # Generate detailed compatibility report with specific issues
        if not compatibility_report["overall_compatible"]:
            compatibility_report["summary"] = (
                "System incompatible with test requirements"
            )
        elif compatibility_report["warnings"]:
            compatibility_report["summary"] = "System compatible with warnings"
        else:
            compatibility_report["summary"] = "System fully compatible"

        return compatibility_report["overall_compatible"], compatibility_report

    def to_dict(self, include_estimates: bool = False) -> Dict[str, Any]:
        """Converts test configuration metadata to dictionary for serialization and external reporting.

        Args:
            include_estimates: Whether to include execution time estimates

        Returns:
            dict: Dictionary representation of test configuration metadata
        """
        # Create dictionary with category, description, and requirements
        metadata_dict = {
            "category": self.category,
            "description": self.description,
            "requirements": self.requirements.copy(),
            "performance_expectations": self.performance_expectations.copy(),
            "system_constraints": self.system_constraints.copy(),
            "dependencies": self.dependencies.copy(),
        }

        # Include estimated duration and dependencies information
        if include_estimates:
            metadata_dict["estimated_duration"] = self.estimated_duration
            metadata_dict["complexity_analysis"] = {
                "category_complexity": self.category
                in ["performance", "stress", "edge_case"],
                "resource_intensive": "memory" in self.system_constraints
                or "cpu" in self.system_constraints,
                "dependency_count": len(self.dependencies),
            }

        # Return complete metadata dictionary for external use
        return metadata_dict


class TestConfigFactory:
    """Factory class for intelligent test configuration creation with automatic parameter optimization, system
    capability detection, and test category specialization for comprehensive testing framework support.

    This class provides centralized test configuration management with intelligent optimization
    and caching for efficient test execution.
    """

    __test__ = False

    def __init__(
        self,
        default_overrides: Optional[Dict[str, Any]] = None,
        auto_optimize: bool = False,
    ):
        """Initialize test configuration factory with default overrides, auto-optimization settings, and system
        capability detection for intelligent test configuration generation.

        Args:
            default_overrides: Optional dictionary of default parameter overrides
            auto_optimize: Whether to enable automatic parameter optimization
        """
        # Store default overrides for consistent configuration modification
        self._default_overrides = default_overrides or {}

        # Set auto-optimization flag for intelligent parameter tuning
        self._auto_optimize = auto_optimize

        # Initialize system capabilities detection and caching
        self._system_capabilities = {}

        # Initialize configuration cache for performance optimization
        self._configuration_cache = {}

        # Initialize category metadata for test configuration specialization
        self._category_metadata = self._initialize_category_metadata()

        # Detect initial system capabilities for configuration optimization
        if auto_optimize:
            self.detect_system_capabilities(force_refresh=True)

    def _initialize_category_metadata(self) -> Dict[str, TestConfigMetadata]:
        """Initialize metadata for each test category."""
        metadata = {}

        # Unit test metadata
        metadata["unit"] = TestConfigMetadata(
            category="unit",
            description="Fast unit tests with minimal parameters for component isolation",
            requirements={
                "grid_size": UNIT_TEST_GRID_SIZE,
                "max_steps": TEST_EPISODE_STEPS_UNIT,
                "execution_speed": "fast",
            },
            performance_expectations={
                "max_duration_seconds": 2.0,
                "memory_limit_mb": 25,
                "deterministic": True,
            },
        )

        # Integration test metadata
        metadata["integration"] = TestConfigMetadata(
            category="integration",
            description="Integration tests with realistic parameters for system validation",
            requirements={
                "grid_size": INTEGRATION_TEST_GRID_SIZE,
                "max_steps": TEST_EPISODE_STEPS_INTEGRATION,
                "cross_component_testing": True,
            },
            performance_expectations={
                "max_duration_seconds": 10.0,
                "memory_limit_mb": 50,
                "realistic_behavior": True,
            },
        )

        # Performance test metadata
        metadata["performance"] = TestConfigMetadata(
            category="performance",
            description="Performance benchmarks with strict timing and resource monitoring",
            requirements={
                "grid_size": PERFORMANCE_TEST_GRID_SIZE,
                "max_steps": TEST_EPISODE_STEPS_PERFORMANCE,
                "benchmark_precision": True,
            },
            performance_expectations={
                "max_duration_seconds": 60.0,
                "step_latency_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS,
                "memory_monitoring": True,
            },
            system_constraints={"min_memory_mb": 100, "monitoring_required": True},
        )

        # Add other categories...
        metadata["reproducibility"] = TestConfigMetadata(
            category="reproducibility",
            description="Reproducibility tests with fixed seeding and deterministic behavior",
            requirements={
                "fixed_seed": True,
                "deterministic_mode": True,
                "episode_validation": True,
            },
            performance_expectations={
                "max_duration_seconds": 20.0,
                "consistency_validation": True,
            },
        )

        metadata["stress"] = TestConfigMetadata(
            category="stress",
            description="Stress tests with maximum parameters for scalability validation",
            requirements={
                "grid_size": STRESS_TEST_GRID_SIZE,
                "heavy_computation": True,
                "resource_monitoring": True,
            },
            performance_expectations={
                "max_duration_seconds": 300.0,
                "resource_limits": True,
                "safety_monitoring": True,
            },
            system_constraints={"min_memory_mb": 200, "safety_monitoring": True},
        )

        return metadata

    def create_for_test_type(
        self,
        test_type: str,
        type_overrides: Optional[Dict[str, Any]] = None,
        validate_result: bool = True,
    ) -> CompleteConfig:
        """Creates optimized test configuration for specific test type with automatic parameter tuning and
        system-specific optimization.

        Args:
            test_type: Test type/category for configuration creation
            type_overrides: Optional type-specific parameter overrides
            validate_result: Whether to validate resulting configuration

        Returns:
            CompleteConfig: Optimized test configuration for specified test type with intelligent parameter selection

        Raises:
            ValidationError: If test_type is not supported
            ConfigurationError: If configuration creation fails
        """
        # Validate test_type against supported test categories
        if test_type not in self._category_metadata:
            raise ValidationError(
                f"Unsupported test type: {test_type}",
                parameter_name="test_type",
                invalid_value=test_type,
                expected_format=f"One of: {', '.join(self._category_metadata.keys())}",
            )

        # Check configuration cache for performance optimization
        cache_key = f"{test_type}_{hash(str(type_overrides))}"
        if cache_key in self._configuration_cache and not type_overrides:
            cached_config = self._configuration_cache[cache_key]
            return copy.deepcopy(cached_config)

        # Select appropriate base configuration factory for test type
        config_factories = {
            "unit": create_unit_test_config,
            "integration": create_integration_test_config,
            "performance": create_performance_test_config,
            "reproducibility": create_reproducibility_test_config,
            "minimal": create_minimal_test_config,
            "stress": create_stress_test_config,
        }

        factory_func = config_factories.get(test_type)
        if not factory_func:
            raise ConfigurationError(
                f"No factory function available for test type: {test_type}",
                config_parameter="test_type",
                invalid_value=test_type,
            )

        # Create base configuration
        if test_type == "edge_case":
            # Edge case requires additional parameter
            base_config = create_edge_case_test_config("boundary_conditions")
        else:
            base_config = factory_func()

        # Apply auto-optimization based on detected system capabilities
        if self._auto_optimize and self._system_capabilities:
            optimization_overrides = self.optimize_for_system(base_config, "speed")
            if hasattr(optimization_overrides, "advanced_options"):
                base_config = optimization_overrides

        # Merge default overrides with type-specific overrides
        all_overrides = {}
        all_overrides.update(self._default_overrides)
        if type_overrides:
            all_overrides.update(type_overrides)

        # Apply intelligent parameter tuning for optimal test execution
        if all_overrides:
            base_config = base_config.clone_with_overrides(all_overrides)

        # Cache configuration for reuse if parameters are identical
        if not type_overrides:
            self._configuration_cache[cache_key] = copy.deepcopy(base_config)

        # Validate result configuration if validate_result is True
        if validate_result:
            try:
                is_valid, report, suggestions = validate_test_configuration(
                    base_config, test_type, strict_validation=False
                )
                if not is_valid:
                    raise ConfigurationError(
                        f"Generated configuration failed validation: {'; '.join(report['issues_found'])}",
                        config_parameter="generated_config",
                        invalid_value=test_type,
                    )
            except Exception as e:
                raise ConfigurationError(
                    f"Configuration validation failed for {test_type}: {e}",
                    config_parameter="test_type",
                    invalid_value=test_type,
                )

        # Update configuration metadata
        base_config.metadata.update(
            {
                "factory_created": True,
                "auto_optimized": self._auto_optimize,
                "cached": cache_key in self._configuration_cache,
            }
        )

        # Return optimized configuration ready for test execution
        return base_config

    def create_suite_configs(
        self,
        test_categories: List[str],
        suite_overrides: Optional[Dict[str, Any]] = None,
        optimize_for_speed: bool = False,
    ) -> Dict[str, CompleteConfig]:
        """Creates complete set of test configurations for test suite execution with category optimization and
        cross-test consistency.

        Args:
            test_categories: List of test categories to create configurations for
            suite_overrides: Optional suite-wide parameter overrides
            optimize_for_speed: Whether to optimize all configurations for execution speed

        Returns:
            dict: Dictionary mapping test categories to optimized configurations for comprehensive test suite execution

        Raises:
            ValidationError: If any test categories are invalid
        """
        # Validate all test categories in test_categories list
        invalid_categories = [
            cat for cat in test_categories if cat not in self._category_metadata
        ]
        if invalid_categories:
            raise ValidationError(
                f"Invalid test categories: {', '.join(invalid_categories)}",
                parameter_name="test_categories",
                invalid_value=invalid_categories,
                expected_format=f"Categories from: {', '.join(self._category_metadata.keys())}",
            )

        suite_configs = {}

        # Create base configurations for each test category
        for category in test_categories:
            base_config = self.create_for_test_type(category, validate_result=False)

            # Optimize for speed if optimize_for_speed flag is enabled
            if optimize_for_speed:
                speed_overrides = {
                    "max_steps": min(base_config.max_steps, 50),  # Limit episode length
                    "enable_performance_monitoring": False,  # Disable monitoring overhead
                    "grid_size": min(
                        base_config.grid_size, UNIT_TEST_GRID_SIZE
                    ),  # Use smaller grids
                    "deterministic_mode": True,  # Enable deterministic for consistency
                }
                base_config = base_config.clone_with_overrides(speed_overrides)

            suite_configs[category] = base_config

        # Apply suite-wide optimizations for consistency and performance
        if suite_overrides:
            for category, config in suite_configs.items():
                suite_configs[category] = config.clone_with_overrides(suite_overrides)

        # Ensure configuration compatibility across test categories
        # Standardize render modes for consistency
        for config in suite_configs.values():
            if not optimize_for_speed or config.render_mode != "rgb_array":
                config.render_mode = "rgb_array"  # Consistent rendering

        # Update metadata for suite execution
        for category, config in suite_configs.items():
            config.metadata.update(
                {
                    "suite_execution": True,
                    "suite_categories": test_categories,
                    "speed_optimized": optimize_for_speed,
                }
            )

        # Return complete suite configuration dictionary
        return suite_configs

    def detect_system_capabilities(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Detects current system capabilities including memory, CPU, rendering backends, and performance
        characteristics for configuration optimization.

        Args:
            force_refresh: Whether to force fresh detection ignoring cache

        Returns:
            dict: System capabilities dictionary with memory, CPU, and rendering information
        """
        # Check cache for existing capabilities if force_refresh is False
        if not force_refresh and self._system_capabilities:
            return self._system_capabilities.copy()

        capabilities = {}

        # Detect available system memory and memory limits
        try:
            import psutil

            memory_info = psutil.virtual_memory()
            capabilities["memory_gb"] = memory_info.total / (1024**3)
            capabilities["available_memory_gb"] = memory_info.available / (1024**3)
            capabilities["memory_usage_percent"] = memory_info.percent
        except ImportError:
            # Fallback memory detection
            capabilities["memory_gb"] = 8.0  # Conservative estimate
            capabilities["available_memory_gb"] = 6.0
            capabilities["memory_usage_percent"] = 25.0

        # Detect CPU capabilities and performance characteristics
        try:
            import psutil

            capabilities["cpu_count"] = psutil.cpu_count(logical=True)
            capabilities["cpu_count_physical"] = psutil.cpu_count(logical=False)
            capabilities["cpu_freq"] = (
                psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
            )
        except ImportError:
            capabilities["cpu_count"] = 4  # Conservative estimate
            capabilities["cpu_count_physical"] = 2
            capabilities["cpu_freq"] = {}

        # Test matplotlib backend availability and rendering capabilities
        try:
            import matplotlib
            import matplotlib.pyplot as plt

            # Test available backends
            available_backends = []
            test_backends = ["TkAgg", "Qt5Agg", "Agg"]

            for backend in test_backends:
                try:
                    matplotlib.use(backend)
                    available_backends.append(backend)
                except:
                    continue

            capabilities["matplotlib_backends"] = available_backends
            capabilities["has_display"] = (
                "TkAgg" in available_backends or "Qt5Agg" in available_backends
            )
            capabilities["headless_capable"] = "Agg" in available_backends

        except ImportError:
            capabilities["matplotlib_backends"] = []
            capabilities["has_display"] = False
            capabilities["headless_capable"] = False

        # Measure system performance baselines for timing optimization
        import time

        start_time = time.time()

        # Simple performance benchmark
        test_iterations = 10000
        for _ in range(test_iterations):
            pass  # Minimal loop for timing baseline

        loop_time = time.time() - start_time
        capabilities["performance_baseline"] = {
            "loop_time_microseconds": (loop_time * 1000000) / test_iterations,
            "estimated_performance_tier": (
                "high" if loop_time < 0.001 else "medium" if loop_time < 0.01 else "low"
            ),
        }

        # Cache detected capabilities for future configuration optimization
        self._system_capabilities = capabilities

        # Return comprehensive system capabilities dictionary
        return capabilities.copy()

    def optimize_for_system(
        self, base_config: CompleteConfig, optimization_target: str = "speed"
    ) -> CompleteConfig:
        """Optimizes test configuration parameters based on detected system capabilities for maximum performance
        and compatibility.

        Args:
            base_config: Base configuration to optimize
            optimization_target: Target for optimization ('speed', 'memory', 'compatibility')

        Returns:
            CompleteConfig: System-optimized configuration with parameters tuned for current system capabilities

        Raises:
            ValidationError: If optimization_target is invalid
        """
        valid_targets = ["speed", "memory", "compatibility"]
        if optimization_target not in valid_targets:
            raise ValidationError(
                f"Invalid optimization target: {optimization_target}",
                parameter_name="optimization_target",
                invalid_value=optimization_target,
                expected_format=f"One of: {', '.join(valid_targets)}",
            )

        # Analyze base configuration complexity and resource requirements
        optimizations = {}

        if not self._system_capabilities:
            self.detect_system_capabilities()

        caps = self._system_capabilities

        # Optimize for specified target (speed, memory, compatibility)
        if optimization_target == "speed":
            # Speed optimizations based on system performance
            if caps.get("cpu_count", 4) < 4:
                optimizations["max_steps"] = min(base_config.max_steps, 100)
                optimizations["grid_size"] = UNIT_TEST_GRID_SIZE

            if caps.get("memory_gb", 8) < 8:
                optimizations["memory_limit_mb"] = 25
                optimizations["enable_performance_monitoring"] = False

            # Disable expensive operations for speed
            optimizations["enable_validation"] = False
            optimizations["deterministic_mode"] = True

        elif optimization_target == "memory":
            # Memory optimizations based on available memory
            available_memory = caps.get("available_memory_gb", 4)

            if available_memory < 2:
                optimizations["grid_size"] = (32, 32)
                optimizations["memory_limit_mb"] = 15
            elif available_memory < 4:
                optimizations["grid_size"] = UNIT_TEST_GRID_SIZE
                optimizations["memory_limit_mb"] = 30

            # Memory-efficient settings
            optimizations["enable_performance_monitoring"] = False

        else:  # compatibility
            # Compatibility optimizations for maximum system compatibility
            if not caps.get("has_display", True):
                optimizations["render_mode"] = "rgb_array"

            if not caps.get("matplotlib_backends"):
                optimizations["render_mode"] = "rgb_array"
                optimizations["enable_rendering"] = False

            # Conservative settings for compatibility
            optimizations["grid_size"] = UNIT_TEST_GRID_SIZE
            optimizations["max_steps"] = min(base_config.max_steps, 200)

        # Adjust grid sizes and parameters based on system memory limits
        if "memory_gb" in caps:
            memory_gb = caps["memory_gb"]
            max_grid_cells = int(
                (memory_gb * 1024 * 1024 * 1024 * 0.1) / 4
            )  # 10% of memory, 4 bytes per cell
            max_dimension = int(max_grid_cells**0.5)

            if "grid_size" not in optimizations:
                current_cells = base_config.grid_size[0] * base_config.grid_size[1]
                if current_cells > max_grid_cells:
                    safe_dimension = min(max_dimension, max(32, max_dimension))
                    optimizations["grid_size"] = (safe_dimension, safe_dimension)

        # Optimize performance targets based on system performance baselines
        if "performance_baseline" in caps:
            perf_tier = caps["performance_baseline"].get(
                "estimated_performance_tier", "medium"
            )
            if perf_tier == "low":
                optimizations["step_latency_target_ms"] = (
                    PERFORMANCE_TARGET_STEP_LATENCY_MS * 3
                )
            elif perf_tier == "high":
                optimizations["step_latency_target_ms"] = (
                    PERFORMANCE_TARGET_STEP_LATENCY_MS * 0.7
                )

        # Apply system-specific optimizations
        optimized_config = base_config.clone_with_overrides(optimizations)

        # Validate optimized configuration maintains test validity
        try:
            optimized_config.validate_all(strict_mode=False)
        except Exception:
            # If optimization breaks configuration, return original with minimal optimizations
            minimal_optimizations = {"render_mode": "rgb_array"}
            optimized_config = base_config.clone_with_overrides(minimal_optimizations)

        # Update metadata with optimization information
        optimized_config.metadata.update(
            {
                "system_optimized": True,
                "optimization_target": optimization_target,
                "system_capabilities": caps,
                "applied_optimizations": list(optimizations.keys()),
            }
        )

        # Return system-optimized configuration ready for execution
        return optimized_config

    def clear_cache(self) -> None:
        """Clears configuration cache and forces fresh system capability detection for updated optimization.

        This method clears all internal caches and forces fresh system detection on next use.
        """
        # Clear configuration cache dictionary
        self._configuration_cache.clear()

        # Reset system capabilities cache
        self._system_capabilities.clear()

        # Clear category metadata cache (reinitialize if needed)
        # Keep metadata as it's static, just log cache clearing

        # Force fresh system capability detection on next optimization
        # This will happen automatically when detect_system_capabilities is called next

        # Log cache clearing for debugging and monitoring
        import logging

        logger = logging.getLogger("plume_nav_sim.test_configs")
        logger.debug("TestConfigFactory cache cleared - fresh detection on next use")


# ===== MODULE EXPORTS =====

__all__ = [
    # Mock implementations (temporary until default_config.py is implemented)
    "CompleteConfig",
    "EnvironmentConfig",
    "PerformanceConfig",
    "get_complete_default_config",
    # Test configuration factory functions
    "create_unit_test_config",
    "create_integration_test_config",
    "create_performance_test_config",
    "create_reproducibility_test_config",
    "create_edge_case_test_config",
    "get_test_config_for_category",
    "validate_test_configuration",
    "create_minimal_test_config",
    "create_stress_test_config",
    # Test configuration classes
    "TestConfigFactory",
    "TestConfigMetadata",
    # Test scenario constants
    "REPRODUCIBILITY_SEEDS",
    "UNIT_TEST_SCENARIOS",
    "INTEGRATION_TEST_SCENARIOS",
    "PERFORMANCE_TEST_SCENARIOS",
    "EDGE_CASE_TEST_SCENARIOS",
    # Test configuration constants
    "UNIT_TEST_GRID_SIZE",
    "INTEGRATION_TEST_GRID_SIZE",
    "PERFORMANCE_TEST_GRID_SIZE",
    "STRESS_TEST_GRID_SIZE",
    "TEST_EPISODE_STEPS_UNIT",
    "TEST_EPISODE_STEPS_INTEGRATION",
    "TEST_EPISODE_STEPS_PERFORMANCE",
    "TEST_SIGMA_SIMPLE",
    "TEST_SIGMA_COMPLEX",
    "TEST_TIMEOUT_MULTIPLIER",
    "PERFORMANCE_TOLERANCE",
]
