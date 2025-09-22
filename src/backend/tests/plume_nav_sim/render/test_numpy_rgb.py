"""
Comprehensive test suite for NumpyRGBRenderer and RGB array generation utilities providing
validation of high-performance programmatic visualization, color scheme integration, marker
accuracy, performance benchmarks, error handling, and API compliance testing for automated
analysis workflows and machine learning pipeline integration with pytest fixtures and
parameterized testing.

This test module ensures RGB array rendering meets <5ms performance targets, validates
visual design specifications, tests component integration, and provides comprehensive
error handling coverage for the plume navigation simulation system.
"""

import time  # >=3.10 - High-precision timing for performance benchmarks and latency validation testing
import warnings  # >=3.10 - Warning management for test execution and performance threshold validation
from contextlib import (  # >=3.10 - Context manager utilities for performance monitoring and resource cleanup testing
    contextmanager,
)
from typing import (  # >=3.10 - Type hints for test function parameters and return values
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)
from unittest.mock import (  # >=3.10 - Mocking utilities for testing error conditions, fallback scenarios, and edge cases
    MagicMock,
    Mock,
    call,
    patch,
)

import numpy as np  # >=2.1.0 - Array operations, mathematical testing, RGB array validation, and concentration field generation

# External imports with version comments
import pytest  # >=8.0.0 - Testing framework for fixtures, parameterized tests, markers, and comprehensive test execution

from plume_nav_sim.core.constants import (
    AGENT_MARKER_COLOR,
    AGENT_MARKER_SIZE,
    FIELD_DTYPE,
    PERFORMANCE_TARGET_RGB_RENDER_MS,
    PIXEL_VALUE_MAX,
    PIXEL_VALUE_MIN,
    RGB_DTYPE,
    SOURCE_MARKER_COLOR,
    SOURCE_MARKER_SIZE,
    get_testing_constants,
)
from plume_nav_sim.core.types import (
    Coordinates,
    GridSize,
    RenderMode,
    create_coordinates,
    create_grid_size,
)
from plume_nav_sim.render.base_renderer import (
    BaseRenderer,
    RenderContext,
    create_render_context,
)

# Internal imports from the module under test and supporting components
from plume_nav_sim.render.numpy_rgb import (
    NumpyRGBRenderer,
    clear_rgb_cache,
    create_rgb_renderer,
    generate_rgb_array_fast,
    get_rgb_performance_stats,
    validate_rgb_array_output,
)
from plume_nav_sim.utils.exceptions import (
    ComponentError,
    RenderingError,
    ValidationError,
)

# Global test configuration constants
TEST_GRID_SIZES = [(32, 32), (64, 64), (128, 128)]
TEST_PERFORMANCE_TOLERANCE_MS = 2.0  # Additional tolerance beyond target
BENCHMARK_ITERATIONS = 100  # Number of iterations for performance benchmarks
EXPECTED_RGB_SHAPE_CHANNELS = 3  # RGB array should have 3 color channels
TEST_CONCENTRATION_VALUES = [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
]  # Range of concentration values for testing
EDGE_CASE_POSITIONS = [
    (0, 0),
    (31, 31),
    (127, 127),
]  # Boundary and edge positions for testing
COLOR_VALIDATION_TOLERANCE = 5  # RGB color value tolerance for marker validation
MEMORY_USAGE_THRESHOLD_MB = 10  # Maximum acceptable memory usage threshold


# Factory functions for test data creation


def create_test_concentration_field(
    grid_size: GridSize,
    pattern_type: str = "gaussian",
    peak_location: Optional[Coordinates] = None,
    peak_value: float = 1.0,
) -> np.ndarray:
    """
    Factory function creating test concentration fields with various patterns including uniform
    fields, gradient fields, Gaussian distributions, and edge cases for comprehensive RGB
    rendering validation.

    Args:
        grid_size: Grid dimensions for field creation
        pattern_type: Type of concentration pattern to generate
        peak_location: Optional peak location for controlled test scenarios
        peak_value: Maximum concentration value for normalization

    Returns:
        Test concentration field with specified pattern and proper dtype for rendering tests
    """
    # Validate grid_size dimensions and create field array with FIELD_DTYPE
    field = np.zeros((grid_size.height, grid_size.width), dtype=FIELD_DTYPE)

    # Generate concentration pattern based on pattern_type
    if pattern_type == "uniform":
        field.fill(peak_value)
    elif pattern_type == "gradient":
        # Create horizontal gradient from left to right
        for x in range(grid_size.width):
            field[:, x] = (x / (grid_size.width - 1)) * peak_value
    elif pattern_type == "gaussian":
        # Create Gaussian distribution centered at peak_location or grid center
        center_x = peak_location.x if peak_location else grid_size.width // 2
        center_y = peak_location.y if peak_location else grid_size.height // 2

        # Generate 2D Gaussian using meshgrid
        y_indices, x_indices = np.mgrid[0 : grid_size.height, 0 : grid_size.width]
        distance_squared = (x_indices - center_x) ** 2 + (y_indices - center_y) ** 2
        sigma = (
            min(grid_size.width, grid_size.height) / 6
        )  # Reasonable sigma for visibility
        field = peak_value * np.exp(-distance_squared / (2 * sigma**2))
    elif pattern_type == "checkerboard":
        # Create checkerboard pattern for testing marker visibility
        for y in range(grid_size.height):
            for x in range(grid_size.width):
                if (x + y) % 2 == 0:
                    field[y, x] = peak_value
    elif pattern_type == "edge_peak":
        # Create concentration peak at edges for boundary testing
        field[0, :] = peak_value  # Top edge
        field[-1, :] = peak_value  # Bottom edge
        field[:, 0] = peak_value  # Left edge
        field[:, -1] = peak_value  # Right edge

    # Normalize field values to [0,1] range ensuring proper concentration bounds
    if np.max(field) > 0:
        field = field / np.max(field) * peak_value

    # Ensure field values are within valid concentration range [0, 1]
    field = np.clip(field, 0.0, 1.0)

    return field


def create_test_render_context(
    grid_size: Optional[GridSize] = None,
    agent_position: Optional[Coordinates] = None,
    source_position: Optional[Coordinates] = None,
    concentration_field: Optional[np.ndarray] = None,
    validate_immediately: bool = True,
) -> RenderContext:
    """
    Factory function creating validated test render contexts with concentration fields,
    agent/source positions, grid configurations, and validation setup for comprehensive
    renderer testing.

    Args:
        grid_size: Grid dimensions or default test grid size
        agent_position: Agent position or auto-generated safe position
        source_position: Source position or auto-generated position away from agent
        concentration_field: Concentration field or auto-generated Gaussian field
        validate_immediately: Whether to validate context after creation

    Returns:
        Validated test render context ready for RGB renderer testing with proper validation
    """
    # Use provided grid_size or default test grid size for performance optimization
    if grid_size is None:
        grid_size = create_grid_size(64, 64)  # Default test size for performance

    # Generate agent_position at grid center if not provided
    if agent_position is None:
        agent_position = create_coordinates(grid_size.width // 2, grid_size.height // 2)

    # Generate source_position at different location from agent if not provided
    if source_position is None:
        # Place source in opposite quadrant from agent for visibility testing
        source_x = min(grid_size.width - 5, max(5, grid_size.width - agent_position.x))
        source_y = min(
            grid_size.height - 5, max(5, grid_size.height - agent_position.y)
        )
        source_position = create_coordinates(source_x, source_y)

    # Create concentration_field using create_test_concentration_field if not provided
    if concentration_field is None:
        concentration_field = create_test_concentration_field(
            grid_size, "gaussian", source_position, 1.0
        )

    # Create RenderContext using create_render_context factory with validation
    context = create_render_context(
        concentration_field=concentration_field,
        agent_position=agent_position,
        source_position=source_position,
        grid_size=grid_size,
    )

    # Perform immediate validation if validate_immediately enabled
    if validate_immediately:
        context.validate()

    return context


def validate_rgb_array_format(
    rgb_array: np.ndarray,
    expected_grid_size: GridSize,
    strict_validation: bool = False,
    check_value_distribution: bool = False,
) -> Dict[str, Any]:
    """
    Comprehensive RGB array format validation ensuring shape, dtype, value ranges, and structure
    compliance for rendering output quality assurance and format consistency testing.

    Args:
        rgb_array: RGB array to validate for format compliance
        expected_grid_size: Expected grid dimensions for shape validation
        strict_validation: Whether to apply strict format validation rules
        check_value_distribution: Whether to analyze value distribution patterns

    Returns:
        Validation results with format compliance, quality metrics, and detailed analysis
    """
    validation_results = {
        "shape_valid": False,
        "dtype_valid": False,
        "value_range_valid": False,
        "structure_valid": False,
        "distribution_analysis": {},
        "quality_score": 0.0,
        "issues": [],
        "recommendations": [],
    }

    # Validate rgb_array shape is (height, width, 3) matching expected_grid_size
    expected_shape = (
        expected_grid_size.height,
        expected_grid_size.width,
        EXPECTED_RGB_SHAPE_CHANNELS,
    )
    if rgb_array.shape == expected_shape:
        validation_results["shape_valid"] = True
    else:
        validation_results["issues"].append(
            f"Shape mismatch: {rgb_array.shape} vs {expected_shape}"
        )

    # Check dtype is uint8 (RGB_DTYPE) for proper color representation
    if rgb_array.dtype == RGB_DTYPE:
        validation_results["dtype_valid"] = True
    else:
        validation_results["issues"].append(
            f"Dtype mismatch: {rgb_array.dtype} vs {RGB_DTYPE}"
        )

    # Validate pixel values are within [PIXEL_VALUE_MIN, PIXEL_VALUE_MAX] range
    if np.all(rgb_array >= PIXEL_VALUE_MIN) and np.all(rgb_array <= PIXEL_VALUE_MAX):
        validation_results["value_range_valid"] = True
    else:
        min_val, max_val = np.min(rgb_array), np.max(rgb_array)
        validation_results["issues"].append(
            f"Values outside range [{PIXEL_VALUE_MIN}, {PIXEL_VALUE_MAX}]: [{min_val}, {max_val}]"
        )

    # Check array structure and memory layout for efficiency
    if rgb_array.ndim == 3 and rgb_array.flags.c_contiguous:
        validation_results["structure_valid"] = True
    else:
        validation_results["issues"].append(
            "Array structure invalid or not C-contiguous"
        )

    # Analyze value distribution if check_value_distribution enabled
    if check_value_distribution:
        validation_results["distribution_analysis"] = {
            "mean_per_channel": [float(np.mean(rgb_array[:, :, i])) for i in range(3)],
            "std_per_channel": [float(np.std(rgb_array[:, :, i])) for i in range(3)],
            "unique_colors": len(np.unique(rgb_array.reshape(-1, 3), axis=0)),
            "dynamic_range": float(np.max(rgb_array) - np.min(rgb_array)),
        }

    # Apply strict format validation rules if strict_validation enabled
    if strict_validation:
        # Check for common rendering artifacts
        if np.all(rgb_array == 0):
            validation_results["issues"].append(
                "Strict validation: All pixels are black (potential rendering failure)"
            )

        # Verify reasonable color distribution
        if len(np.unique(rgb_array)) < 10:
            validation_results["issues"].append(
                "Strict validation: Very limited color palette detected"
            )

    # Calculate overall quality score based on validation results
    valid_count = sum(
        [
            validation_results["shape_valid"],
            validation_results["dtype_valid"],
            validation_results["value_range_valid"],
            validation_results["structure_valid"],
        ]
    )
    validation_results["quality_score"] = (valid_count / 4.0) * 100.0

    # Generate recommendations based on validation results
    if validation_results["quality_score"] == 100.0:
        validation_results["recommendations"].append(
            "RGB array passes all format validation tests"
        )
    else:
        validation_results["recommendations"].append(
            "Address format issues before using RGB array"
        )
        if not validation_results["shape_valid"]:
            validation_results["recommendations"].append(
                "Fix array shape to match grid dimensions"
            )
        if not validation_results["dtype_valid"]:
            validation_results["recommendations"].append("Convert array to uint8 dtype")

    return validation_results


def validate_marker_presence(
    rgb_array: np.ndarray,
    agent_position: Coordinates,
    source_position: Coordinates,
    color_tolerance: int = COLOR_VALIDATION_TOLERANCE,
    check_marker_shapes: bool = True,
) -> Dict[str, Any]:
    """
    Marker presence and accuracy validation ensuring agent and source markers are correctly
    placed with proper colors, sizes, and shapes for visual design specification compliance.

    Args:
        rgb_array: RGB array to check for marker presence
        agent_position: Expected agent marker position
        source_position: Expected source marker position
        color_tolerance: RGB color matching tolerance for validation
        check_marker_shapes: Whether to validate marker shape patterns

    Returns:
        Marker validation results with accuracy metrics, positioning analysis, and visual compliance
    """
    marker_validation = {
        "agent_marker_found": False,
        "source_marker_found": False,
        "agent_color_match": False,
        "source_color_match": False,
        "agent_size_correct": False,
        "source_size_correct": False,
        "positioning_accurate": False,
        "shape_compliance": {},
        "issues": [],
        "recommendations": [],
    }

    height, width = rgb_array.shape[:2]

    # Validate agent marker presence at agent_position with AGENT_MARKER_COLOR
    if 0 <= agent_position.y < height and 0 <= agent_position.x < width:
        actual_agent_color = rgb_array[agent_position.y, agent_position.x]
        expected_agent_color = np.array(AGENT_MARKER_COLOR, dtype=RGB_DTYPE)

        # Check color match within tolerance
        color_diff = np.abs(
            actual_agent_color.astype(int) - expected_agent_color.astype(int)
        )
        if np.all(color_diff <= color_tolerance):
            marker_validation["agent_color_match"] = True
            marker_validation["agent_marker_found"] = True

    # Check agent marker size matches AGENT_MARKER_SIZE (3×3 square) specification
    if marker_validation["agent_marker_found"] and check_marker_shapes:
        agent_size_correct = True
        expected_size_h, expected_size_w = AGENT_MARKER_SIZE

        # Check 3x3 square around agent position
        for dy in range(-expected_size_h // 2, expected_size_h // 2 + 1):
            for dx in range(-expected_size_w // 2, expected_size_w // 2 + 1):
                y_pos = agent_position.y + dy
                x_pos = agent_position.x + dx

                if 0 <= y_pos < height and 0 <= x_pos < width:
                    pixel_color = rgb_array[y_pos, x_pos]
                    expected_color = np.array(AGENT_MARKER_COLOR, dtype=RGB_DTYPE)
                    color_diff = np.abs(
                        pixel_color.astype(int) - expected_color.astype(int)
                    )

                    if not np.all(color_diff <= color_tolerance):
                        agent_size_correct = False
                        break
            if not agent_size_correct:
                break

        marker_validation["agent_size_correct"] = agent_size_correct

    # Validate source marker presence at source_position with SOURCE_MARKER_COLOR
    if 0 <= source_position.y < height and 0 <= source_position.x < width:
        actual_source_color = rgb_array[source_position.y, source_position.x]
        expected_source_color = np.array(SOURCE_MARKER_COLOR, dtype=RGB_DTYPE)

        # Check color match within tolerance
        color_diff = np.abs(
            actual_source_color.astype(int) - expected_source_color.astype(int)
        )
        if np.all(color_diff <= color_tolerance):
            marker_validation["source_color_match"] = True
            marker_validation["source_marker_found"] = True

    # Check source marker pattern matches SOURCE_MARKER_SIZE (5×5 cross) specification
    if marker_validation["source_marker_found"] and check_marker_shapes:
        source_cross_correct = True
        expected_size_h, expected_size_w = SOURCE_MARKER_SIZE

        # Check horizontal line of cross pattern
        for dx in range(-expected_size_w // 2, expected_size_w // 2 + 1):
            x_pos = source_position.x + dx
            y_pos = source_position.y

            if 0 <= y_pos < height and 0 <= x_pos < width:
                pixel_color = rgb_array[y_pos, x_pos]
                expected_color = np.array(SOURCE_MARKER_COLOR, dtype=RGB_DTYPE)
                color_diff = np.abs(
                    pixel_color.astype(int) - expected_color.astype(int)
                )

                if not np.all(color_diff <= color_tolerance):
                    source_cross_correct = False
                    break

        # Check vertical line of cross pattern
        if source_cross_correct:
            for dy in range(-expected_size_h // 2, expected_size_h // 2 + 1):
                y_pos = source_position.y + dy
                x_pos = source_position.x

                if 0 <= y_pos < height and 0 <= x_pos < width:
                    pixel_color = rgb_array[y_pos, x_pos]
                    expected_color = np.array(SOURCE_MARKER_COLOR, dtype=RGB_DTYPE)
                    color_diff = np.abs(
                        pixel_color.astype(int) - expected_color.astype(int)
                    )

                    if not np.all(color_diff <= color_tolerance):
                        source_cross_correct = False
                        break

        marker_validation["source_size_correct"] = source_cross_correct

    # Check markers don't overlap or interfere with each other
    marker_distance = (
        (agent_position.x - source_position.x) ** 2
        + (agent_position.y - source_position.y) ** 2
    ) ** 0.5
    min_safe_distance = max(AGENT_MARKER_SIZE[0], SOURCE_MARKER_SIZE[0]) + 2

    if marker_distance >= min_safe_distance:
        marker_validation["positioning_accurate"] = True
    else:
        marker_validation["issues"].append(
            f"Markers too close: distance {marker_distance:.1f} < {min_safe_distance}"
        )

    # Generate comprehensive validation issues and recommendations
    if not marker_validation["agent_marker_found"]:
        marker_validation["issues"].append(
            "Agent marker not found at expected position"
        )
        marker_validation["recommendations"].append(
            "Verify agent marker placement and color"
        )

    if not marker_validation["source_marker_found"]:
        marker_validation["issues"].append(
            "Source marker not found at expected position"
        )
        marker_validation["recommendations"].append(
            "Verify source marker placement and cross pattern"
        )

    if (
        marker_validation["agent_marker_found"]
        and not marker_validation["agent_size_correct"]
    ):
        marker_validation["issues"].append("Agent marker size/shape incorrect")
        marker_validation["recommendations"].append(
            "Fix agent marker 3×3 square pattern"
        )

    if (
        marker_validation["source_marker_found"]
        and not marker_validation["source_size_correct"]
    ):
        marker_validation["issues"].append("Source marker cross pattern incorrect")
        marker_validation["recommendations"].append(
            "Fix source marker 5×5 cross pattern"
        )

    return marker_validation


def benchmark_rgb_generation_performance(
    renderer: NumpyRGBRenderer,
    test_context: RenderContext,
    iterations: int = BENCHMARK_ITERATIONS,
    include_memory_profiling: bool = False,
    validate_targets: bool = True,
) -> Dict[str, Any]:
    """
    Performance benchmarking function measuring RGB array generation timing, memory usage, and
    efficiency metrics for performance target validation and optimization analysis.

    Args:
        renderer: NumpyRGBRenderer instance to benchmark
        test_context: RenderContext for consistent benchmarking conditions
        iterations: Number of benchmark iterations for statistical analysis
        include_memory_profiling: Whether to include memory usage analysis
        validate_targets: Whether to validate against performance targets

    Returns:
        Performance benchmark results with timing statistics, memory analysis, and target compliance
    """
    benchmark_results = {
        "iterations": iterations,
        "timing_stats": {},
        "memory_stats": {},
        "performance_validation": {},
        "optimization_analysis": {},
        "recommendations": [],
    }

    # Initialize performance monitoring and memory profiling if requested
    timing_results = []
    memory_baseline = 0

    if include_memory_profiling:
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_baseline = process.memory_info().rss / (1024 * 1024)  # MB

    # Execute RGB generation for specified iterations with precise timing measurement
    for i in range(iterations):
        # Force garbage collection between iterations for consistent memory measurement
        import gc

        gc.collect()

        # Measure generation time with high precision
        start_time = time.perf_counter()

        try:
            rgb_array = renderer.render(test_context)
            generation_time_ms = (time.perf_counter() - start_time) * 1000
            timing_results.append(generation_time_ms)

            # Clean up array to prevent memory accumulation
            del rgb_array

        except Exception as e:
            benchmark_results["error"] = (
                f"Benchmark failed on iteration {i+1}: {str(e)}"
            )
            return benchmark_results

    # Collect timing statistics including min, max, mean, median, and standard deviation
    timing_array = np.array(timing_results)
    benchmark_results["timing_stats"] = {
        "mean_ms": float(np.mean(timing_array)),
        "median_ms": float(np.median(timing_array)),
        "min_ms": float(np.min(timing_array)),
        "max_ms": float(np.max(timing_array)),
        "std_dev_ms": float(np.std(timing_array)),
        "percentile_95_ms": float(np.percentile(timing_array, 95)),
        "percentile_99_ms": float(np.percentile(timing_array, 99)),
    }

    # Monitor memory usage patterns if include_memory_profiling enabled
    if include_memory_profiling:
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        benchmark_results["memory_stats"] = {
            "baseline_mb": memory_baseline,
            "final_mb": final_memory,
            "delta_mb": final_memory - memory_baseline,
            "within_threshold": abs(final_memory - memory_baseline)
            <= MEMORY_USAGE_THRESHOLD_MB,
        }

    # Validate performance against PERFORMANCE_TARGET_RGB_RENDER_MS if validate_targets enabled
    if validate_targets:
        mean_time = benchmark_results["timing_stats"]["mean_ms"]
        target_time = PERFORMANCE_TARGET_RGB_RENDER_MS

        benchmark_results["performance_validation"] = {
            "target_ms": target_time,
            "meets_target": mean_time <= target_time,
            "performance_ratio": mean_time / target_time,
            "within_tolerance": mean_time
            <= (target_time + TEST_PERFORMANCE_TOLERANCE_MS),
            "performance_grade": (
                "excellent"
                if mean_time <= target_time * 0.8
                else (
                    "good"
                    if mean_time <= target_time
                    else (
                        "acceptable"
                        if mean_time <= target_time + TEST_PERFORMANCE_TOLERANCE_MS
                        else "needs_optimization"
                    )
                )
            ),
        }

    # Calculate performance metrics and optimization recommendations
    performance_variability = (
        np.std(timing_array) / np.mean(timing_array) if np.mean(timing_array) > 0 else 0
    )

    benchmark_results["optimization_analysis"] = {
        "performance_consistency": performance_variability
        < 0.1,  # Low variability is good
        "variability_coefficient": performance_variability,
        "outlier_count": np.sum(timing_array > np.percentile(timing_array, 95)),
        "optimization_potential": (
            max(0, np.mean(timing_array) - target_time) if validate_targets else 0
        ),
    }

    # Generate performance recommendations
    if (
        validate_targets
        and not benchmark_results["performance_validation"]["meets_target"]
    ):
        benchmark_results["recommendations"].append(
            "Performance target not met - consider optimization"
        )

    if performance_variability > 0.2:
        benchmark_results["recommendations"].append(
            "High timing variability - check system load"
        )

    if (
        include_memory_profiling
        and benchmark_results["memory_stats"]["delta_mb"] > MEMORY_USAGE_THRESHOLD_MB
    ):
        benchmark_results["recommendations"].append(
            "Memory usage increase detected - check for leaks"
        )

    return benchmark_results


def create_edge_case_test_data(
    edge_case_type: str,
    grid_size: GridSize,
    custom_parameters: Optional[Dict[str, Any]] = None,
) -> Tuple[RenderContext, str, Dict[str, Any]]:
    """
    Edge case test data generator creating challenging scenarios including boundary positions,
    extreme values, empty fields, and stress conditions for robust renderer testing.

    Args:
        edge_case_type: Type of edge case scenario to generate
        grid_size: Grid dimensions for test data creation
        custom_parameters: Optional custom parameters for specialized edge cases

    Returns:
        Tuple of (render_context, expected_behavior, validation_criteria) for edge case testing
    """
    custom_params = custom_parameters or {}

    if edge_case_type == "boundary_positions":
        # Create render context with agent and source at grid boundaries
        agent_pos = create_coordinates(0, 0)  # Top-left corner
        source_pos = create_coordinates(
            grid_size.width - 1, grid_size.height - 1
        )  # Bottom-right corner
        field = create_test_concentration_field(grid_size, "uniform", None, 0.5)

        context = create_test_render_context(grid_size, agent_pos, source_pos, field)
        expected_behavior = "markers_at_boundaries"
        validation_criteria = {
            "agent_visible": True,
            "source_visible": True,
            "no_clipping_errors": True,
            "marker_shapes_intact": True,
        }

    elif edge_case_type == "extreme_values":
        # Create field with extreme concentration values and test normalization
        field = np.full(
            (grid_size.height, grid_size.width), 1.0, dtype=FIELD_DTYPE
        )  # Maximum concentration
        agent_pos = create_coordinates(grid_size.width // 2, grid_size.height // 2)
        source_pos = create_coordinates(5, 5)

        context = create_test_render_context(grid_size, agent_pos, source_pos, field)
        expected_behavior = "extreme_concentration_handling"
        validation_criteria = {
            "no_value_overflow": True,
            "proper_normalization": True,
            "markers_visible_on_white": True,
            "no_rendering_artifacts": True,
        }

    elif edge_case_type == "empty_field":
        # Create completely empty concentration field
        field = np.zeros((grid_size.height, grid_size.width), dtype=FIELD_DTYPE)
        agent_pos = create_coordinates(grid_size.width // 2, grid_size.height // 2)
        source_pos = create_coordinates(10, 10)

        context = create_test_render_context(grid_size, agent_pos, source_pos, field)
        expected_behavior = "zero_concentration_rendering"
        validation_criteria = {
            "black_background": True,
            "markers_visible_on_black": True,
            "no_nan_values": True,
            "proper_zero_handling": True,
        }

    elif edge_case_type == "overlapping_markers":
        # Create scenario where agent and source markers overlap
        overlap_pos = create_coordinates(grid_size.width // 2, grid_size.height // 2)
        field = create_test_concentration_field(grid_size, "gaussian", overlap_pos, 0.8)

        context = create_test_render_context(grid_size, overlap_pos, overlap_pos, field)
        expected_behavior = "marker_overlap_resolution"
        validation_criteria = {
            "visible_marker_present": True,
            "no_rendering_corruption": True,
            "deterministic_precedence": True,  # One marker should take precedence
        }

    elif edge_case_type == "large_grid":
        # Create large grid size to test memory and performance limits
        large_grid = create_grid_size(
            custom_params.get("width", 256), custom_params.get("height", 256)
        )
        field = create_test_concentration_field(large_grid, "gaussian")
        agent_pos = create_coordinates(large_grid.width // 2, large_grid.height // 2)
        source_pos = create_coordinates(large_grid.width // 4, large_grid.height // 4)

        context = create_test_render_context(large_grid, agent_pos, source_pos, field)
        expected_behavior = "large_scale_rendering"
        validation_criteria = {
            "memory_within_limits": True,
            "performance_acceptable": True,
            "no_memory_errors": True,
            "scaling_linear": True,
        }

    else:
        # Default edge case with minimal viable grid
        minimal_grid = create_grid_size(8, 8)  # Very small grid
        field = create_test_concentration_field(minimal_grid, "uniform", None, 0.3)
        agent_pos = create_coordinates(2, 2)
        source_pos = create_coordinates(5, 5)

        context = create_test_render_context(minimal_grid, agent_pos, source_pos, field)
        expected_behavior = "minimal_grid_rendering"
        validation_criteria = {
            "markers_fit_in_grid": True,
            "no_size_errors": True,
            "readable_output": True,
        }

    return context, expected_behavior, validation_criteria


@contextmanager
def performance_monitor(operation_name: str = "test_operation"):
    """
    Context manager for performance monitoring providing timing and resource tracking
    during test operations.

    Args:
        operation_name: Name of operation being monitored for logging

    Yields:
        Performance monitoring context with timing and resource information
    """
    start_time = time.perf_counter()
    start_memory = 0

    # Get baseline memory if available
    try:
        import os

        import psutil

        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
    except ImportError:
        start_memory = 0

    performance_data = {
        "operation_name": operation_name,
        "start_time": start_time,
        "start_memory_mb": start_memory,
        "duration_ms": 0,
        "memory_delta_mb": 0,
        "within_performance_target": False,
    }

    try:
        yield performance_data
    finally:
        # Calculate final performance metrics
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        end_memory = start_memory
        if start_memory > 0:
            try:
                end_memory = process.memory_info().rss / (1024 * 1024)  # MB
            except:
                end_memory = start_memory

        performance_data.update(
            {
                "duration_ms": duration_ms,
                "memory_delta_mb": end_memory - start_memory,
                "within_performance_target": duration_ms
                <= PERFORMANCE_TARGET_RGB_RENDER_MS,
            }
        )


# Pytest fixtures for test setup and configuration


@pytest.fixture(scope="function")
def rgb_renderer(request):
    """
    Provides fresh NumpyRGBRenderer instance for each test with standard configuration
    and proper initialization.

    Yields:
        NumpyRGBRenderer configured for test execution with performance monitoring
    """
    # Get test grid size from parametrized fixture or default
    grid_size = getattr(request, "param", create_grid_size(64, 64))

    # Create NumpyRGBRenderer with test-optimized parameters
    renderer = NumpyRGBRenderer(
        grid_size=grid_size,
        color_scheme_name="standard",
        renderer_options={
            "enable_caching": True,
            "optimize_for_performance": True,
            "use_vectorized_operations": True,
        },
    )

    # Initialize renderer with performance monitoring enabled
    renderer.initialize()

    # Validate renderer initialization and resource setup
    assert renderer.supports_render_mode(RenderMode.RGB_ARRAY)

    yield renderer

    # Cleanup renderer resources and memory allocations
    try:
        renderer.cleanup_resources()
    except:
        pass  # Ignore cleanup errors in tests

    # Clear any cached data or temporary state
    clear_rgb_cache(
        clear_array_cache=True, clear_performance_stats=False, force_gc=True
    )


@pytest.fixture(scope="function")
def test_render_context():
    """
    Provides validated RenderContext with test concentration field and positions
    for RGB rendering tests.

    Yields:
        RenderContext configured with test data for comprehensive rendering validation
    """
    # Create test grid size optimized for test performance
    grid_size = create_grid_size(64, 64)

    # Generate test concentration field with gaussian pattern
    concentration_field = create_test_concentration_field(
        grid_size, "gaussian", None, 1.0
    )

    # Set agent position and source position for marker testing
    agent_position = create_coordinates(32, 32)  # Grid center
    source_position = create_coordinates(48, 16)  # Offset from center

    # Create validated RenderContext using create_test_render_context
    context = create_test_render_context(
        grid_size=grid_size,
        agent_position=agent_position,
        source_position=source_position,
        concentration_field=concentration_field,
        validate_immediately=True,
    )

    yield context

    # Clear concentration field data and context references
    del concentration_field
    del context


@pytest.fixture(scope="function")
def performance_test_env():
    """
    Provides performance testing environment configuration for benchmark validation.

    Yields:
        Performance testing configuration with optimized settings
    """
    # Store original configuration
    original_config = get_testing_constants()

    # Set performance-optimized test configuration
    test_config = {
        "benchmark_iterations": BENCHMARK_ITERATIONS,
        "performance_target_ms": PERFORMANCE_TARGET_RGB_RENDER_MS,
        "tolerance_ms": TEST_PERFORMANCE_TOLERANCE_MS,
        "memory_threshold_mb": MEMORY_USAGE_THRESHOLD_MB,
        "enable_profiling": True,
    }

    yield test_config

    # Restore original configuration after test


@pytest.fixture(scope="session")
def edge_case_test_data():
    """
    Provides collection of edge case test scenarios including boundary conditions
    and extreme parameters for comprehensive validation.

    Yields:
        Dictionary of edge case test scenarios with render contexts and validation criteria
    """
    grid_size = create_grid_size(128, 128)

    edge_cases = {
        "boundary": create_edge_case_test_data("boundary_positions", grid_size),
        "extreme_values": create_edge_case_test_data("extreme_values", grid_size),
        "empty_field": create_edge_case_test_data("empty_field", grid_size),
        "overlapping": create_edge_case_test_data("overlapping_markers", grid_size),
        "large_grid": create_edge_case_test_data(
            "large_grid", grid_size, {"width": 256, "height": 256}
        ),
        "minimal_grid": create_edge_case_test_data("minimal_grid", grid_size),
    }

    yield edge_cases

    # Clean up edge case test data and temporary resources
    for case_name, (context, behavior, criteria) in edge_cases.items():
        del context


# Parametrized fixtures for comprehensive testing


@pytest.fixture(params=TEST_GRID_SIZES, ids=["small_grid", "medium_grid", "large_grid"])
def test_grid_sizes(request):
    """
    Parametrized fixture providing various grid sizes for comprehensive testing
    across different RGB renderer configurations.
    """
    width, height = request.param
    return create_grid_size(width, height)


@pytest.fixture(
    params=TEST_CONCENTRATION_VALUES,
    ids=["zero_conc", "quarter_conc", "half_conc", "three_quarter_conc", "full_conc"],
)
def test_concentration_values(request):
    """
    Parametrized fixture providing concentration values for testing grayscale
    conversion and visualization accuracy.
    """
    return request.param


@pytest.fixture(
    params=EDGE_CASE_POSITIONS,
    ids=["origin", "corner_small", "corner_medium", "corner_large"],
)
def edge_case_positions(request):
    """
    Parametrized fixture providing boundary and edge positions for marker
    placement testing.
    """
    x, y = request.param
    return create_coordinates(x, y)


# Main test classes implementing comprehensive test coverage


class TestNumpyRGBRenderer:
    """
    Comprehensive test class for NumpyRGBRenderer functionality including initialization,
    rendering operations, performance validation, error handling, and integration testing
    with fixtures and parameterized test methods.
    """

    def test_renderer_initialization(self, test_grid_sizes):
        """
        Test NumpyRGBRenderer initialization with various grid sizes and configurations
        ensuring proper setup, resource allocation, and inheritance compliance.
        """
        # Create NumpyRGBRenderer instance with parameterized grid_size
        renderer = NumpyRGBRenderer(
            grid_size=test_grid_sizes,
            color_scheme_name="standard",
            renderer_options={"enable_caching": True},
        )

        # Validate renderer inherits from BaseRenderer correctly
        assert isinstance(renderer, BaseRenderer)
        assert isinstance(renderer, NumpyRGBRenderer)

        # Check initialization sets up proper grid configuration
        assert renderer.grid_size == test_grid_sizes
        assert renderer.grid_size.width == test_grid_sizes.width
        assert renderer.grid_size.height == test_grid_sizes.height

        # Verify renderer supports RGB_ARRAY mode using supports_render_mode method
        assert renderer.supports_render_mode(RenderMode.RGB_ARRAY) == True
        assert renderer.supports_render_mode("rgb_array") == True  # String mode support

        # Validate resource initialization after calling initialize()
        renderer.initialize()
        assert renderer._array_buffer is not None
        assert renderer._array_buffer.shape == (
            test_grid_sizes.height,
            test_grid_sizes.width,
            3,
        )
        assert renderer._array_buffer.dtype == RGB_DTYPE

        # Test renderer properties and configuration accessibility
        assert hasattr(renderer, "performance_config")
        assert hasattr(renderer, "generation_stats")
        assert hasattr(renderer, "current_color_scheme")

        # Cleanup
        renderer.cleanup_resources()

    @pytest.mark.unit
    def test_rgb_array_generation(self, rgb_renderer, test_render_context):
        """
        Test RGB array generation with various render contexts ensuring proper format,
        color accuracy, and visual element placement.
        """
        # Generate RGB array using render method with test_render_context
        with performance_monitor("rgb_array_generation") as perf:
            rgb_array = rgb_renderer.render(test_render_context)

        # Validate RGB array format using validate_rgb_array_format function
        format_validation = validate_rgb_array_format(
            rgb_array,
            test_render_context.grid_size,
            strict_validation=True,
            check_value_distribution=True,
        )

        assert format_validation["shape_valid"] == True
        assert format_validation["dtype_valid"] == True
        assert format_validation["value_range_valid"] == True
        assert format_validation["quality_score"] == 100.0

        # Check concentration field visualization and grayscale conversion
        # Background areas should reflect concentration values
        concentration_field = test_render_context.concentration_field
        expected_grayscale = (concentration_field * 255).astype(RGB_DTYPE)

        # Sample a few background pixels (avoiding marker areas)
        sample_y, sample_x = 5, 5  # Top-left area, likely background
        actual_pixel = rgb_array[sample_y, sample_x]
        expected_pixel_value = expected_grayscale[sample_y, sample_x]

        # For grayscale, R=G=B channels should match expected value
        assert abs(actual_pixel[0] - expected_pixel_value) <= 10  # Allow some tolerance

        # Validate agent and source marker placement using validate_marker_presence
        marker_validation = validate_marker_presence(
            rgb_array,
            test_render_context.agent_position,
            test_render_context.source_position,
            color_tolerance=COLOR_VALIDATION_TOLERANCE,
            check_marker_shapes=True,
        )

        assert marker_validation["agent_marker_found"] == True
        assert marker_validation["source_marker_found"] == True
        assert marker_validation["agent_color_match"] == True
        assert marker_validation["source_color_match"] == True

        # Verify performance meets requirements
        assert perf["within_performance_target"] == True
        assert (
            perf["duration_ms"]
            <= PERFORMANCE_TARGET_RGB_RENDER_MS + TEST_PERFORMANCE_TOLERANCE_MS
        )

    @pytest.mark.performance
    @pytest.mark.slow
    def test_performance_benchmarks(
        self, rgb_renderer, test_render_context, performance_test_env
    ):
        """
        Performance benchmark testing validating RGB generation meets <5ms target with
        comprehensive timing analysis and resource monitoring.
        """
        # Execute benchmark_rgb_generation_performance with multiple iterations
        benchmark_results = benchmark_rgb_generation_performance(
            renderer=rgb_renderer,
            test_context=test_render_context,
            iterations=performance_test_env["benchmark_iterations"],
            include_memory_profiling=True,
            validate_targets=True,
        )

        # Validate generation timing meets PERFORMANCE_TARGET_RGB_RENDER_MS target
        timing_stats = benchmark_results["timing_stats"]
        performance_validation = benchmark_results["performance_validation"]

        assert (
            "error" not in benchmark_results
        ), f"Benchmark error: {benchmark_results.get('error', 'Unknown')}"
        assert (
            performance_validation["meets_target"] == True
        ), f"Performance target not met: {timing_stats['mean_ms']:.2f}ms > {PERFORMANCE_TARGET_RGB_RENDER_MS}ms"

        # Check performance consistency across iterations
        assert (
            timing_stats["std_dev_ms"] < timing_stats["mean_ms"] * 0.3
        ), "High performance variability detected"

        # Analyze memory usage patterns and resource efficiency
        if "memory_stats" in benchmark_results:
            memory_stats = benchmark_results["memory_stats"]
            assert (
                memory_stats["within_threshold"] == True
            ), f"Memory usage exceeded threshold: {memory_stats['delta_mb']:.2f}MB"

        # Verify optimization analysis results
        optimization_analysis = benchmark_results["optimization_analysis"]
        assert (
            optimization_analysis["performance_consistency"] == True
        ), "Performance consistency issues detected"

        # Assert performance grade is acceptable
        acceptable_grades = ["excellent", "good", "acceptable"]
        assert (
            performance_validation["performance_grade"] in acceptable_grades
        ), f"Unacceptable performance grade: {performance_validation['performance_grade']}"

    @pytest.mark.unit
    def test_color_scheme_integration(self, rgb_renderer, test_render_context):
        """
        Test color scheme integration and updates ensuring proper color application,
        marker accuracy, and visual consistency across scheme changes.
        """
        # Generate baseline RGB array with default color scheme
        baseline_rgb = rgb_renderer.render(test_render_context)

        # Validate default colors match expected scheme
        baseline_validation = validate_marker_presence(
            baseline_rgb,
            test_render_context.agent_position,
            test_render_context.source_position,
            color_tolerance=COLOR_VALIDATION_TOLERANCE,
        )

        assert baseline_validation["agent_color_match"] == True
        assert baseline_validation["source_color_match"] == True

        # Update color scheme using update_color_scheme method
        # Note: We'll test with the same scheme to avoid dependency on multiple schemes
        rgb_renderer.update_color_scheme(
            color_scheme="standard", optimize_for_rgb=True, clear_cache=True
        )

        # Generate new RGB array and verify consistency
        updated_rgb = rgb_renderer.render(test_render_context)

        # Validate updated array format and marker presence
        updated_validation = validate_marker_presence(
            updated_rgb,
            test_render_context.agent_position,
            test_render_context.source_position,
            color_tolerance=COLOR_VALIDATION_TOLERANCE,
        )

        assert updated_validation["agent_marker_found"] == True
        assert updated_validation["source_marker_found"] == True

        # Check consistency between baseline and updated arrays (same scheme)
        np.testing.assert_array_equal(
            baseline_rgb,
            updated_rgb,
            "Color scheme update should not change output for same scheme",
        )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "position_fixture",
        EDGE_CASE_POSITIONS,
        ids=["origin", "corner_small", "corner_large"],
    )
    def test_marker_placement_accuracy(self, rgb_renderer, position_fixture):
        """
        Test agent and source marker placement accuracy across various positions ensuring
        proper positioning, size, and visual specification compliance.
        """
        x, y = position_fixture

        # Create render context with markers at parameterized positions
        grid_size = create_grid_size(128, 128)  # Large enough for all edge positions

        # Ensure positions are within bounds
        agent_position = create_coordinates(
            min(x, grid_size.width - 3), min(y, grid_size.height - 3)
        )
        source_position = create_coordinates(
            min(max(5, x + 10), grid_size.width - 6),
            min(max(5, y + 10), grid_size.height - 6),
        )

        concentration_field = create_test_concentration_field(
            grid_size, "uniform", None, 0.3
        )

        context = create_test_render_context(
            grid_size=grid_size,
            agent_position=agent_position,
            source_position=source_position,
            concentration_field=concentration_field,
        )

        # Generate RGB array with markers at edge case positions
        rgb_array = rgb_renderer.render(context)

        # Validate agent marker (red 3×3 square) placement accuracy
        marker_validation = validate_marker_presence(
            rgb_array,
            agent_position,
            source_position,
            color_tolerance=COLOR_VALIDATION_TOLERANCE,
            check_marker_shapes=True,
        )

        assert (
            marker_validation["agent_marker_found"] == True
        ), f"Agent marker not found at position ({x}, {y})"
        assert (
            marker_validation["agent_color_match"] == True
        ), f"Agent marker color mismatch at position ({x}, {y})"

        # Validate source marker (white 5×5 cross) placement accuracy
        assert (
            marker_validation["source_marker_found"] == True
        ), f"Source marker not found near position ({x}, {y})"
        assert (
            marker_validation["source_color_match"] == True
        ), f"Source marker color mismatch near position ({x}, {y})"

        # Check marker boundaries and clipping for edge positions
        if x == 0 or y == 0 or x >= grid_size.width - 3 or y >= grid_size.height - 3:
            # For boundary positions, verify no array access errors occurred
            format_validation = validate_rgb_array_format(rgb_array, grid_size)
            assert (
                format_validation["shape_valid"] == True
            ), "Boundary marker placement caused shape issues"
            assert (
                format_validation["value_range_valid"] == True
            ), "Boundary marker placement caused value range issues"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "concentration",
        TEST_CONCENTRATION_VALUES,
        ids=[
            "zero_conc",
            "quarter_conc",
            "half_conc",
            "three_quarter_conc",
            "full_conc",
        ],
    )
    def test_concentration_field_visualization(self, rgb_renderer, concentration):
        """
        Test concentration field visualization accuracy ensuring proper grayscale conversion,
        value mapping, and visual representation of plume data.
        """
        # Create uniform concentration field with parameterized concentration value
        grid_size = create_grid_size(64, 64)
        concentration_field = np.full(
            (grid_size.height, grid_size.width), concentration, dtype=FIELD_DTYPE
        )

        # Position markers away from center to avoid interference
        agent_position = create_coordinates(10, 10)
        source_position = create_coordinates(50, 50)

        context = create_test_render_context(
            grid_size=grid_size,
            agent_position=agent_position,
            source_position=source_position,
            concentration_field=concentration_field,
        )

        # Generate RGB array with known concentration pattern
        rgb_array = rgb_renderer.render(context)

        # Validate grayscale conversion from concentration [0,1] to pixel values [0,255]
        expected_pixel_value = int(concentration * 255)

        # Sample background pixels (avoiding marker areas)
        sample_pixels = []
        for y in range(5, 8):  # Top area, away from markers
            for x in range(5, 8):
                if (x, y) != (agent_position.x, agent_position.y) and (x, y) != (
                    source_position.x,
                    source_position.y,
                ):
                    sample_pixels.append(rgb_array[y, x])

        # Check concentration-to-pixel mapping accuracy for sampled pixels
        for pixel in sample_pixels[:5]:  # Check first 5 sample pixels
            # For grayscale conversion, all RGB channels should be approximately equal
            assert (
                abs(pixel[0] - pixel[1]) <= 2
            ), f"RGB channels not equal for grayscale: {pixel}"
            assert (
                abs(pixel[1] - pixel[2]) <= 2
            ), f"RGB channels not equal for grayscale: {pixel}"

            # Check approximate match to expected pixel value (allowing some tolerance)
            avg_pixel_value = np.mean(pixel.astype(float))
            assert (
                abs(avg_pixel_value - expected_pixel_value) <= 10
            ), f"Pixel value {avg_pixel_value} doesn't match expected {expected_pixel_value}"

        # Verify background areas have proper concentration representation
        if concentration == 0.0:
            # For zero concentration, background should be black (except for markers)
            background_mask = np.ones((grid_size.height, grid_size.width), dtype=bool)

            # Exclude marker areas from background check
            agent_area = slice(
                max(0, agent_position.y - 1),
                min(grid_size.height, agent_position.y + 2),
            ), slice(
                max(0, agent_position.x - 1), min(grid_size.width, agent_position.x + 2)
            )
            source_area = slice(
                max(0, source_position.y - 2),
                min(grid_size.height, source_position.y + 3),
            ), slice(
                max(0, source_position.x - 2),
                min(grid_size.width, source_position.x + 3),
            )

            background_mask[agent_area] = False
            background_mask[source_area] = False

            # Check that background pixels are approximately black
            background_pixels = rgb_array[background_mask]
            if len(background_pixels) > 0:
                avg_background = np.mean(background_pixels)
                assert (
                    avg_background <= 20
                ), f"Background not dark enough for zero concentration: {avg_background}"

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_error_handling(self, rgb_renderer):
        """
        Test comprehensive error handling including invalid inputs, resource failures,
        and edge cases ensuring graceful degradation and meaningful error messages.
        """
        # Test invalid render context handling with malformed data
        invalid_grid = create_grid_size(64, 64)
        invalid_field = np.ones((32, 32), dtype=FIELD_DTYPE)  # Wrong size
        invalid_agent = create_coordinates(32, 32)
        invalid_source = create_coordinates(48, 48)

        # This should raise ValidationError for mismatched dimensions
        with pytest.raises(ValidationError) as exc_info:
            invalid_context = create_render_context(
                concentration_field=invalid_field,
                agent_position=invalid_agent,
                source_position=invalid_source,
                grid_size=invalid_grid,
            )
            rgb_renderer.render(invalid_context)

        assert (
            "shape" in str(exc_info.value).lower()
            or "dimension" in str(exc_info.value).lower()
        )

        # Test out-of-bounds position handling
        valid_grid = create_grid_size(32, 32)
        valid_field = create_test_concentration_field(valid_grid, "uniform", None, 0.5)

        # Agent position outside grid bounds
        oob_agent = create_coordinates(100, 100)  # Way outside 32x32 grid
        valid_source = create_coordinates(16, 16)

        # Create context - validation should catch this
        with pytest.raises((ValidationError, ValueError)) as exc_info:
            oob_context = create_render_context(
                concentration_field=valid_field,
                agent_position=oob_agent,
                source_position=valid_source,
                grid_size=valid_grid,
            )
            rgb_renderer.render(oob_context)

        # Verify error message clarity and debugging information
        error_message = str(exc_info.value)
        assert len(error_message) > 0, "Error message should not be empty"
        assert any(
            keyword in error_message.lower()
            for keyword in ["bounds", "position", "grid", "invalid"]
        )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "grid_size_fixture",
        TEST_GRID_SIZES,
        ids=["small_grid", "medium_grid", "large_grid"],
    )
    def test_memory_usage(self, grid_size_fixture):
        """
        Test memory usage patterns and resource management ensuring efficient allocation,
        cleanup, and memory constraint compliance.
        """
        width, height = grid_size_fixture
        grid_size = create_grid_size(width, height)

        # Monitor baseline memory usage before renderer creation
        import os

        import psutil

        try:
            process = psutil.Process(os.getpid())
            baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
        except ImportError:
            pytest.skip("psutil not available for memory monitoring")

        # Create NumpyRGBRenderer and measure memory allocation
        renderer = NumpyRGBRenderer(
            grid_size=grid_size,
            color_scheme_name="standard",
            renderer_options={"enable_caching": True},
        )
        renderer.initialize()

        after_creation_memory = process.memory_info().rss / (1024 * 1024)  # MB
        creation_memory_delta = after_creation_memory - baseline_memory

        # Generate multiple RGB arrays and track memory usage patterns
        context = create_test_render_context(grid_size)

        memory_samples = []
        for i in range(5):  # Generate 5 arrays to test memory patterns
            rgb_array = renderer.render(context)

            current_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_samples.append(current_memory)

            # Clean up array to test memory management
            del rgb_array

        final_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # Validate memory usage stays within MEMORY_USAGE_THRESHOLD_MB limits
        peak_memory_usage = max(memory_samples)
        memory_growth = peak_memory_usage - after_creation_memory

        assert (
            memory_growth <= MEMORY_USAGE_THRESHOLD_MB
        ), f"Memory growth {memory_growth:.2f}MB exceeds threshold {MEMORY_USAGE_THRESHOLD_MB}MB"

        # Test memory cleanup after renderer disposal
        renderer.cleanup_resources()
        del renderer

        # Force garbage collection
        import gc

        gc.collect()

        cleanup_memory = process.memory_info().rss / (1024 * 1024)  # MB
        cleanup_delta = cleanup_memory - baseline_memory

        # Check for memory leaks - some residual usage is acceptable
        assert (
            cleanup_delta <= creation_memory_delta + 2
        ), f"Potential memory leak detected: {cleanup_delta:.2f}MB residual"

        # Assert efficient memory usage relative to array size
        expected_array_size_mb = (width * height * 3) / (1024 * 1024)  # RGB array size
        efficiency_ratio = (
            creation_memory_delta / expected_array_size_mb
            if expected_array_size_mb > 0
            else 0
        )

        # Memory usage should be reasonable compared to array size (allow for overhead)
        assert (
            efficiency_ratio <= 10
        ), f"Memory efficiency poor: {efficiency_ratio:.2f}x array size"

    @pytest.mark.unit
    @pytest.mark.performance
    def test_optimization_features(
        self, rgb_renderer, test_render_context, performance_test_env
    ):
        """
        Test optimization features including caching, vectorized operations, and performance
        enhancements ensuring efficiency and correctness.
        """
        # Test generate_optimized_array method with various optimization settings
        optimized_array = rgb_renderer.generate_optimized_array(
            context=test_render_context,
            reuse_buffer=True,
            skip_validation=False,  # Keep validation for correctness
        )

        # Validate optimized array format and quality
        format_validation = validate_rgb_array_format(
            optimized_array, test_render_context.grid_size
        )
        assert format_validation["shape_valid"] == True
        assert format_validation["dtype_valid"] == True
        assert format_validation["value_range_valid"] == True

        # Compare optimized vs standard rendering for correctness
        standard_array = rgb_renderer.render(test_render_context)

        # Arrays should be very similar (allowing for minor optimization differences)
        difference = np.mean(
            np.abs(optimized_array.astype(float) - standard_array.astype(float))
        )
        assert (
            difference <= 5.0
        ), f"Optimization changed output too much: difference {difference}"

        # Test performance improvement with buffer reuse
        iterations = 20

        # Benchmark without buffer reuse
        standard_times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            _ = rgb_renderer.render(test_render_context)
            standard_times.append((time.perf_counter() - start_time) * 1000)

        # Benchmark with optimization
        optimized_times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            _ = rgb_renderer.generate_optimized_array(
                test_render_context, reuse_buffer=True, skip_validation=True
            )
            optimized_times.append((time.perf_counter() - start_time) * 1000)

        # Calculate performance improvement
        avg_standard_time = np.mean(standard_times)
        avg_optimized_time = np.mean(optimized_times)

        # Optimization should provide some performance benefit or at least not hurt performance significantly
        performance_ratio = avg_optimized_time / avg_standard_time
        assert (
            performance_ratio <= 1.2
        ), f"Optimization made performance worse: {performance_ratio:.2f}x slower"

        # Ideally, optimization should be faster
        if performance_ratio < 0.9:
            print(
                f"Optimization provided {((1 - performance_ratio) * 100):.1f}% performance improvement"
            )

    @pytest.mark.integration
    def test_integration_with_base_renderer(self, rgb_renderer, test_render_context):
        """
        Test integration with BaseRenderer abstract class ensuring proper inheritance,
        method implementation, and interface compliance.
        """
        # Validate NumpyRGBRenderer properly inherits from BaseRenderer
        assert isinstance(rgb_renderer, BaseRenderer)
        assert issubclass(NumpyRGBRenderer, BaseRenderer)

        # Test abstract method implementations
        assert rgb_renderer.supports_render_mode(RenderMode.RGB_ARRAY) == True
        assert (
            rgb_renderer.supports_render_mode(RenderMode.HUMAN) == False
        )  # NumpyRGBRenderer is RGB-only

        # Check base class method integration
        # initialize() should work correctly
        rgb_renderer.initialize()  # Should not raise exception

        # render() should return proper RGB array
        result = rgb_renderer.render(test_render_context)
        assert isinstance(result, np.ndarray)
        assert result.shape == (
            test_render_context.grid_size.height,
            test_render_context.grid_size.width,
            3,
        )

        # Test context validation integration
        test_render_context.validate()  # Should work correctly

        # Verify performance metrics integration
        stats = rgb_renderer.get_generation_statistics(include_memory_analysis=True)
        assert isinstance(stats, dict)
        assert "total_generations" in stats
        assert "average_time_ms" in stats

        # Test cleanup integration
        rgb_renderer.cleanup_resources()  # Should not raise exception

        # Check polymorphic usage through base class interface
        def render_via_base_class(
            renderer: BaseRenderer, context: RenderContext
        ) -> np.ndarray:
            """Test polymorphic renderer usage through base class."""
            if renderer.supports_render_mode(RenderMode.RGB_ARRAY):
                return renderer.render(context)
            else:
                raise ValueError("Renderer doesn't support RGB_ARRAY mode")

        # This should work correctly with NumpyRGBRenderer through BaseRenderer interface
        result = render_via_base_class(rgb_renderer, test_render_context)
        assert isinstance(result, np.ndarray)

        # Validate error handling consistency with base class patterns
        # Test with invalid context to ensure consistent error handling
        with pytest.raises((ValidationError, ValueError, TypeError)):
            rgb_renderer.render(None)  # Should raise appropriate exception


class TestRGBUtilityFunctions:
    """
    Test class for RGB utility functions including factory functions, validation utilities,
    performance monitoring, and cache management ensuring comprehensive utility function coverage.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "grid_size_fixture",
        TEST_GRID_SIZES,
        ids=["small_grid", "medium_grid", "large_grid"],
    )
    def test_create_rgb_renderer_factory(self, grid_size_fixture):
        """
        Test create_rgb_renderer factory function with various configurations ensuring
        proper renderer creation and parameter validation.
        """
        width, height = grid_size_fixture
        grid_size = create_grid_size(width, height)

        # Test create_rgb_renderer with various configurations
        renderer = create_rgb_renderer(
            grid_size=grid_size,
            color_scheme_name="standard",
            enable_caching=True,
            optimize_for_performance=True,
            renderer_options={"use_vectorized_operations": True},
        )

        # Validate returned renderer is properly initialized NumpyRGBRenderer instance
        assert isinstance(renderer, NumpyRGBRenderer)
        assert isinstance(renderer, BaseRenderer)
        assert renderer.grid_size == grid_size

        # Check performance configuration options are applied correctly
        assert renderer.performance_config["enable_caching"] == True
        assert renderer.performance_config["vectorization_enabled"] == True

        # Test factory function parameter validation
        with pytest.raises(ValidationError):
            # Invalid grid size should raise ValidationError
            invalid_grid = create_grid_size(0, 0)  # Invalid dimensions
            create_rgb_renderer(invalid_grid)

        # Test renderer functionality after factory creation
        test_context = create_test_render_context(grid_size)
        rgb_array = renderer.render(test_context)

        # Validate factory-created renderer produces valid output
        format_validation = validate_rgb_array_format(rgb_array, grid_size)
        assert format_validation["shape_valid"] == True
        assert format_validation["dtype_valid"] == True

        # Cleanup
        renderer.cleanup_resources()

    @pytest.mark.unit
    @pytest.mark.performance
    def test_generate_rgb_array_fast(self, test_render_context):
        """
        Test generate_rgb_array_fast utility function for direct RGB generation bypassing
        renderer overhead with performance validation.
        """
        # Test generate_rgb_array_fast with test render context
        concentration_field = test_render_context.concentration_field
        agent_position = test_render_context.agent_position
        source_position = test_render_context.source_position

        # Measure performance of fast generation
        with performance_monitor("fast_rgb_generation") as perf:
            fast_rgb = generate_rgb_array_fast(
                concentration_field=concentration_field,
                agent_position=agent_position,
                source_position=source_position,
                color_scheme=None,  # Use default colors
                validate_inputs=True,
            )

        # Validate RGB array format and quality
        format_validation = validate_rgb_array_format(
            fast_rgb, test_render_context.grid_size
        )
        assert format_validation["shape_valid"] == True
        assert format_validation["dtype_valid"] == True
        assert format_validation["value_range_valid"] == True

        # Check marker presence in fast-generated array
        marker_validation = validate_marker_presence(
            fast_rgb,
            agent_position,
            source_position,
            color_tolerance=COLOR_VALIDATION_TOLERANCE,
        )
        assert marker_validation["agent_marker_found"] == True
        assert marker_validation["source_marker_found"] == True

        # Validate performance improvement over standard renderer creation
        # Fast generation should be very fast since it bypasses renderer setup
        assert perf["within_performance_target"] == True
        assert perf["duration_ms"] <= PERFORMANCE_TARGET_RGB_RENDER_MS

        # Test function parameter validation
        with pytest.raises(ValidationError):
            # Invalid concentration field should raise ValidationError
            invalid_field = np.ones(
                (10, 20, 3), dtype=FIELD_DTYPE
            )  # Wrong shape (3D instead of 2D)
            generate_rgb_array_fast(
                invalid_field, agent_position, source_position, validate_inputs=True
            )

        # Test with validation disabled for maximum performance
        fast_rgb_no_validation = generate_rgb_array_fast(
            concentration_field, agent_position, source_position, validate_inputs=False
        )

        # Output should still be valid even without validation
        assert fast_rgb_no_validation.shape == fast_rgb.shape
        assert fast_rgb_no_validation.dtype == fast_rgb.dtype

    @pytest.mark.unit
    def test_validate_rgb_array_output(self):
        """
        Test validate_rgb_array_output utility function ensuring comprehensive RGB array
        validation and quality assurance functionality.
        """
        # Create test RGB array with known properties
        grid_size = create_grid_size(32, 32)
        agent_position = create_coordinates(16, 16)
        source_position = create_coordinates(8, 8)

        # Create valid RGB array
        valid_rgb = np.random.randint(0, 256, (32, 32, 3), dtype=RGB_DTYPE)

        # Add agent marker (red square)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                y_pos = agent_position.y + dy
                x_pos = agent_position.x + dx
                if 0 <= y_pos < 32 and 0 <= x_pos < 32:
                    valid_rgb[y_pos, x_pos] = AGENT_MARKER_COLOR

        # Add source marker (white cross)
        for dx in range(-2, 3):
            x_pos = source_position.x + dx
            if 0 <= x_pos < 32:
                valid_rgb[source_position.y, x_pos] = SOURCE_MARKER_COLOR

        for dy in range(-2, 3):
            y_pos = source_position.y + dy
            if 0 <= y_pos < 32:
                valid_rgb[y_pos, source_position.x] = SOURCE_MARKER_COLOR

        # Test validate_rgb_array_output with valid array
        is_valid, validation_report = validate_rgb_array_output(
            rgb_array=valid_rgb,
            expected_grid_size=grid_size,
            expected_agent_position=agent_position,
            expected_source_position=source_position,
            strict_validation=True,
        )

        assert is_valid == True
        assert validation_report["format_validation"]["shape_correct"] == True
        assert validation_report["format_validation"]["dtype_correct"] == True
        assert validation_report["marker_validation"]["agent_marker_present"] == True
        assert validation_report["marker_validation"]["source_marker_present"] == True

        # Test with invalid RGB array
        invalid_rgb = np.ones((20, 20, 3), dtype=np.float32)  # Wrong size and dtype

        is_valid_invalid, validation_report_invalid = validate_rgb_array_output(
            rgb_array=invalid_rgb,
            expected_grid_size=grid_size,
            expected_agent_position=agent_position,
            expected_source_position=source_position,
        )

        assert is_valid_invalid == False
        assert len(validation_report_invalid["recommendations"]) > 0
        assert "shape" in str(validation_report_invalid["recommendations"]).lower()

        # Test validation report completeness
        required_keys = [
            "format_validation",
            "marker_validation",
            "quality_metrics",
            "performance_indicators",
        ]
        for key in required_keys:
            assert key in validation_report, f"Missing key in validation report: {key}"

    @pytest.mark.unit
    def test_cache_management(self):
        """
        Test RGB cache management utilities including cache clearing, performance statistics,
        and memory optimization functionality.
        """
        # Test clear_rgb_cache function
        cleanup_report = clear_rgb_cache(
            clear_array_cache=True, clear_performance_stats=True, force_gc=True
        )

        assert isinstance(cleanup_report, dict)
        assert "cache_cleared" in cleanup_report
        assert "stats_cleared" in cleanup_report
        assert "cleanup_timestamp" in cleanup_report
        assert "gc_forced" in cleanup_report

        # Verify cache clearing effectiveness
        assert cleanup_report["cache_cleared"] == True
        assert cleanup_report["stats_cleared"] == True
        assert cleanup_report["gc_forced"] == True

        # Test get_rgb_performance_stats utility function
        performance_stats = get_rgb_performance_stats(
            include_cache_analysis=True,
            include_memory_usage=True,
            include_optimization_suggestions=True,
        )

        # Check performance statistics collection and analysis functionality
        assert isinstance(performance_stats, dict)
        required_stats_keys = [
            "global_statistics",
            "timing_analysis",
            "cache_analysis",
            "memory_analysis",
            "optimization_suggestions",
        ]

        for key in required_stats_keys:
            assert key in performance_stats, f"Missing key in performance stats: {key}"

        # Validate global statistics structure
        global_stats = performance_stats["global_statistics"]
        assert "performance_baseline_ms" in global_stats
        assert "caching_enabled_globally" in global_stats
        assert (
            global_stats["performance_baseline_ms"] == PERFORMANCE_TARGET_RGB_RENDER_MS
        )

        # Test cache analysis functionality
        cache_analysis = performance_stats["cache_analysis"]
        assert "cache_enabled" in cache_analysis
        assert "cache_size" in cache_analysis
        assert "cache_utilization" in cache_analysis

        # Test optimization suggestions generation
        assert isinstance(performance_stats["optimization_suggestions"], list)

        # Test incremental cache clearing
        partial_cleanup = clear_rgb_cache(
            clear_array_cache=True,
            clear_performance_stats=False,  # Keep stats
            force_gc=False,
        )

        assert partial_cleanup["cache_cleared"] == True
        assert partial_cleanup["stats_cleared"] == False
        assert partial_cleanup["gc_forced"] == False


class TestRGBRenderingEdgeCases:
    """
    Test class for edge cases and boundary conditions including extreme parameters,
    boundary positions, resource constraints, and stress testing for robust renderer validation.
    """

    @pytest.mark.edge_case
    def test_boundary_conditions(self, rgb_renderer):
        """
        Test renderer behavior at grid boundaries ensuring proper clipping, marker
        placement, and edge handling.
        """
        # Create small grid to test boundary conditions more easily
        boundary_grid = create_grid_size(16, 16)

        # Test agent and source marker placement at grid boundaries
        boundary_cases = [
            (create_coordinates(0, 0), create_coordinates(15, 15)),  # Corners
            (create_coordinates(0, 8), create_coordinates(15, 8)),  # Edges
            (create_coordinates(8, 0), create_coordinates(8, 15)),  # Other edges
        ]

        for agent_pos, source_pos in boundary_cases:
            concentration_field = create_test_concentration_field(
                boundary_grid, "uniform", None, 0.4
            )

            context = create_test_render_context(
                grid_size=boundary_grid,
                agent_position=agent_pos,
                source_position=source_pos,
                concentration_field=concentration_field,
            )

            # Render should not fail at boundary positions
            rgb_array = rgb_renderer.render(context)

            # Validate no rendering errors occurred
            format_validation = validate_rgb_array_format(rgb_array, boundary_grid)
            assert format_validation["shape_valid"] == True
            assert format_validation["value_range_valid"] == True

            # Check that markers are visible (even if partially clipped)
            marker_validation = validate_marker_presence(
                rgb_array,
                agent_pos,
                source_pos,
                color_tolerance=COLOR_VALIDATION_TOLERANCE,
                check_marker_shapes=False,  # Don't check shapes at boundaries (may be clipped)
            )

            # At minimum, center pixels of markers should be visible
            assert (
                marker_validation["agent_marker_found"] == True
            ), f"Agent marker not visible at boundary {agent_pos.to_tuple()}"
            assert (
                marker_validation["source_marker_found"] == True
            ), f"Source marker not visible at boundary {source_pos.to_tuple()}"

    @pytest.mark.edge_case
    @pytest.mark.slow
    def test_extreme_grid_sizes(self):
        """
        Test renderer with extreme grid sizes including very small and very large
        configurations ensuring scalability and resource management.
        """
        # Test very small grid sizes
        tiny_grid = create_grid_size(4, 4)
        tiny_renderer = create_rgb_renderer(
            tiny_grid, enable_caching=False
        )  # Disable caching for small grids

        tiny_context = create_test_render_context(
            grid_size=tiny_grid,
            agent_position=create_coordinates(1, 1),
            source_position=create_coordinates(2, 2),
        )

        tiny_rgb = tiny_renderer.render(tiny_context)

        # Validate tiny grid rendering
        assert tiny_rgb.shape == (4, 4, 3)
        assert tiny_rgb.dtype == RGB_DTYPE

        tiny_renderer.cleanup_resources()

        # Test large grid size (memory and performance considerations)
        # Use smaller size than maximum to avoid CI/CD memory issues
        large_grid = create_grid_size(256, 256)

        # Monitor memory usage during large grid operations
        import os

        import psutil

        try:
            process = psutil.Process(os.getpid())
            baseline_memory = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            pytest.skip("psutil not available for memory monitoring")

        large_renderer = create_rgb_renderer(large_grid, enable_caching=True)

        large_context = create_test_render_context(
            grid_size=large_grid,
            agent_position=create_coordinates(128, 128),  # Center
            source_position=create_coordinates(64, 192),
        )

        # Test large grid rendering with performance monitoring
        with performance_monitor("large_grid_rendering") as perf:
            large_rgb = large_renderer.render(large_context)

        # Validate large grid results
        assert large_rgb.shape == (256, 256, 3)
        assert large_rgb.dtype == RGB_DTYPE

        # Check memory usage scaling
        current_memory = process.memory_info().rss / (1024 * 1024)
        memory_usage = current_memory - baseline_memory

        # Memory usage should be reasonable for large grid (allow up to 200MB)
        assert (
            memory_usage <= 200
        ), f"Large grid memory usage too high: {memory_usage:.2f}MB"

        # Performance should still be reasonable (allow more time for large grids)
        large_grid_target = (
            PERFORMANCE_TARGET_RGB_RENDER_MS * 10
        )  # 10x target for 16x larger grid
        assert (
            perf["duration_ms"] <= large_grid_target
        ), f"Large grid rendering too slow: {perf['duration_ms']:.2f}ms"

        large_renderer.cleanup_resources()

        # Force cleanup
        del large_rgb
        import gc

        gc.collect()

    @pytest.mark.edge_case
    def test_concentration_field_edge_cases(self, rgb_renderer):
        """
        Test rendering with edge case concentration fields including uniform fields,
        extreme values, and pathological cases.
        """
        grid_size = create_grid_size(32, 32)

        edge_case_fields = [
            ("all_zeros", np.zeros((32, 32), dtype=FIELD_DTYPE)),
            ("all_ones", np.ones((32, 32), dtype=FIELD_DTYPE)),
            (
                "checkerboard",
                create_test_concentration_field(grid_size, "checkerboard", None, 1.0),
            ),
            (
                "edge_peaks",
                create_test_concentration_field(grid_size, "edge_peak", None, 1.0),
            ),
        ]

        agent_position = create_coordinates(16, 16)
        source_position = create_coordinates(8, 24)

        for field_name, concentration_field in edge_case_fields:
            context = create_test_render_context(
                grid_size=grid_size,
                agent_position=agent_position,
                source_position=source_position,
                concentration_field=concentration_field,
            )

            # Render should handle all edge case fields
            rgb_array = rgb_renderer.render(context)

            # Basic format validation
            format_validation = validate_rgb_array_format(rgb_array, grid_size)
            assert (
                format_validation["shape_valid"] == True
            ), f"Shape invalid for {field_name} field"
            assert (
                format_validation["dtype_valid"] == True
            ), f"Dtype invalid for {field_name} field"
            assert (
                format_validation["value_range_valid"] == True
            ), f"Value range invalid for {field_name} field"

            # Check for NaN or infinite values
            assert np.all(
                np.isfinite(rgb_array)
            ), f"Non-finite values in {field_name} field rendering"

            # Ensure markers are still visible
            marker_validation = validate_marker_presence(
                rgb_array,
                agent_position,
                source_position,
                color_tolerance=COLOR_VALIDATION_TOLERANCE
                * 2,  # Allow more tolerance for edge cases
            )

            # Markers should be visible even in pathological cases
            assert (
                marker_validation["agent_marker_found"] == True
            ), f"Agent marker not found in {field_name} field"
            assert (
                marker_validation["source_marker_found"] == True
            ), f"Source marker not found in {field_name} field"

            # Field-specific validations
            if field_name == "all_zeros":
                # Background should be mostly black
                background_mask = (
                    (rgb_array[:, :, 0] < 50)
                    & (rgb_array[:, :, 1] < 50)
                    & (rgb_array[:, :, 2] < 50)
                )
                assert np.sum(background_mask) > (
                    32 * 32 * 0.7
                ), "Background not dark enough for all-zeros field"

            elif field_name == "all_ones":
                # Background should be mostly white/bright
                background_mask = (
                    (rgb_array[:, :, 0] > 200)
                    & (rgb_array[:, :, 1] > 200)
                    & (rgb_array[:, :, 2] > 200)
                )
                assert np.sum(background_mask) > (
                    32 * 32 * 0.5
                ), "Background not bright enough for all-ones field"

    @pytest.mark.edge_case
    def test_resource_constraint_handling(self, rgb_renderer):
        """
        Test renderer behavior under resource constraints including memory limits,
        processing constraints, and system resource availability.
        """
        # Test with resource-constrained configuration
        constrained_grid = create_grid_size(128, 128)

        # Create multiple render contexts to stress test resource usage
        contexts = []
        for i in range(10):  # Create 10 different contexts
            context = create_test_render_context(
                grid_size=constrained_grid,
                agent_position=create_coordinates(i * 10 + 10, i * 8 + 16),
                source_position=create_coordinates(64 + i * 5, 64 - i * 3),
            )
            contexts.append(context)

        # Monitor memory usage during batch rendering
        import os

        import psutil

        try:
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            start_memory = 0

        rendered_arrays = []
        peak_memory = start_memory

        # Render all contexts while monitoring memory
        for i, context in enumerate(contexts):
            rgb_array = rgb_renderer.render(context)
            rendered_arrays.append(rgb_array)

            if start_memory > 0:
                current_memory = process.memory_info().rss / (1024 * 1024)
                peak_memory = max(peak_memory, current_memory)

            # Validate each rendering
            format_validation = validate_rgb_array_format(rgb_array, constrained_grid)
            assert (
                format_validation["shape_valid"] == True
            ), f"Rendering {i} failed format validation"

            # Check for memory growth patterns
            if start_memory > 0:
                memory_growth = current_memory - start_memory
                assert (
                    memory_growth <= MEMORY_USAGE_THRESHOLD_MB * 5
                ), f"Excessive memory growth: {memory_growth:.2f}MB"

        # Test resource cleanup effectiveness
        del rendered_arrays
        import gc

        gc.collect()

        if start_memory > 0:
            cleanup_memory = process.memory_info().rss / (1024 * 1024)
            memory_released = peak_memory - cleanup_memory

            # Some memory should be released after cleanup
            assert (
                memory_released >= 0
            ), f"Memory not properly released: {memory_released:.2f}MB"

        # Test error handling for simulated resource exhaustion
        # Mock a memory constraint scenario
        with patch("numpy.zeros") as mock_zeros:
            mock_zeros.side_effect = MemoryError("Simulated memory exhaustion")

            # Renderer should handle memory errors gracefully
            with pytest.raises((MemoryError, ComponentError, RenderingError)):
                rgb_renderer.render(contexts[0])

            # Renderer should still be functional after error
            mock_zeros.side_effect = None  # Restore normal behavior
            mock_zeros.side_effect = np.zeros  # Reset to original function

        # Verify renderer can continue working after error recovery
        test_context = create_test_render_context(create_grid_size(32, 32))
        recovery_rgb = rgb_renderer.render(test_context)

        recovery_validation = validate_rgb_array_format(
            recovery_rgb, test_context.grid_size
        )
        assert (
            recovery_validation["shape_valid"] == True
        ), "Renderer not functional after error recovery"


# Pytest markers for test categorization
pytest_markers = [
    pytest.mark.unit,  # Unit tests for individual components
    pytest.mark.integration,  # Integration tests for component interaction
    pytest.mark.performance,  # Performance and benchmark tests
    pytest.mark.edge_case,  # Edge case and boundary condition tests
    pytest.mark.slow,  # Slow-running tests that may be skipped in fast CI
]
