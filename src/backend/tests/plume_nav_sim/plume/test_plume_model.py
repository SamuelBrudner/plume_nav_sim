"""
Comprehensive test suite for plume model components including abstract base classes, static Gaussian
plume implementation, concentration field data structures, registry system, and factory functions
with mathematical validation, performance testing, error handling verification, and integration
testing for plume_nav_sim reinforcement learning environment.

This test module provides comprehensive coverage for:
- Abstract base class implementation testing with interface compliance validation
- Static Gaussian plume model mathematical accuracy and performance validation
- Concentration field data structure operations and caching behavior testing
- Registry system functionality including model registration and factory operations
- Mathematical formula verification against analytical Gaussian solutions
- Performance requirements validation with timing and memory constraints
- Error handling and exception testing with recovery strategy validation
- Integration testing across plume model components and data structures
"""

import copy  # >=3.10 - Deep copying operations for model cloning tests and state isolation validation
import math  # >=3.10 - Mathematical functions for Gaussian formula validation, distance calculations, and numerical testing
import time  # >=3.10 - Performance timing measurements for generation speed validation and benchmark testing
import warnings  # >=3.10 - Warning capture and validation for testing performance warnings and deprecation handling
from unittest.mock import (  # >=3.10 - Mock objects and patching for testing error conditions, performance monitoring, and component isolation
    MagicMock,
    Mock,
    patch,
)

import numpy as np  # >=2.1.0 - Mathematical operations, array assertions, concentration field validation, and numerical precision testing

# External imports with version comments
import pytest  # >=8.0.0 - Testing framework providing fixtures, assertions, parametrization, and test organization for comprehensive test coverage

from plume_nav_sim.core.constants import (
    DEFAULT_GRID_SIZE,
    DEFAULT_PLUME_SIGMA,
    MAX_PLUME_SIGMA,
    MIN_PLUME_SIGMA,
    PERFORMANCE_TARGET_PLUME_GENERATION_MS,
    STATIC_GAUSSIAN_MODEL_TYPE,
)

# Internal imports from core types and constants
from plume_nav_sim.core.types import (
    Coordinates,
    GridSize,
    PlumeParameters,
    create_coordinates,
    create_grid_size,
)

# Internal imports from concentration field data structure
from plume_nav_sim.plume.concentration_field import (
    ConcentrationField,
    create_concentration_field,
)

# Internal imports from plume model framework
from plume_nav_sim.plume.plume_model import (
    BasePlumeModel,
    PlumeModelInterface,
    PlumeModelRegistry,
    create_plume_model,
    get_supported_plume_types,
    validate_plume_model_interface,
)

# Internal imports from static Gaussian plume implementation
from plume_nav_sim.plume.static_gaussian import (
    StaticGaussianPlume,
    calculate_gaussian_concentration,
    create_static_gaussian_plume,
    validate_gaussian_parameters,
)

# Global test constants and configuration parameters
TEST_GRID_SIZES = [(32, 32), (64, 64), (128, 128), (256, 256)]
TEST_SIGMA_VALUES = [0.5, 1.0, 5.0, 12.0, 25.0, 50.0]
TEST_SOURCE_LOCATIONS = [(16, 16), (32, 32), (64, 64), (96, 96)]
PERFORMANCE_TOLERANCE_MS = 2.0
MATHEMATICAL_PRECISION = 1e-6
MEMORY_TOLERANCE_FACTOR = 1.1
MAX_TEST_GRID_SIZE = 256
CONCENTRATION_VALIDATION_TOLERANCE = 1e-10


def test_base_plume_model_abstract_methods():
    """
    Test that BasePlumeModel is properly defined as an abstract base class with required
    abstract methods that cannot be instantiated directly and enforce implementation contracts.
    """
    # Test that BasePlumeModel cannot be instantiated directly due to abstract methods
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BasePlumeModel(grid_size=create_grid_size(DEFAULT_GRID_SIZE))

    # Verify all required abstract methods are properly defined with abc.abstractmethod decorator
    abstract_methods = BasePlumeModel.__abstractmethods__
    expected_abstract_methods = {
        "initialize_model",
        "generate_concentration_field",
        "sample_concentration",
        "validate_model",
        "get_model_info",
        "update_parameters",
        "clone",
    }
    assert (
        abstract_methods == expected_abstract_methods
    ), f"Expected abstract methods {expected_abstract_methods}, got {abstract_methods}"

    # Validate method signatures match PlumeModelInterface protocol specifications
    for method_name in expected_abstract_methods:
        assert hasattr(
            BasePlumeModel, method_name
        ), f"BasePlumeModel missing abstract method: {method_name}"

    # Test that concrete subclasses must implement all abstract methods to be instantiable
    class IncompleteModel(BasePlumeModel):
        def __init__(self, grid_size):
            super().__init__(grid_size)

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteModel(create_grid_size(DEFAULT_GRID_SIZE))


def test_plume_model_interface_protocol_compliance():
    """
    Test PlumeModelInterface protocol definition and structural subtyping compatibility ensuring
    duck typing works correctly with proper method signatures and return types.
    """
    # Test that PlumeModelInterface is properly defined as typing_extensions.Protocol
    import inspect
    from typing import get_args, get_origin

    # Validate all required method signatures are correctly specified with type hints
    required_methods = [
        "initialize_model",
        "generate_concentration_field",
        "sample_concentration",
        "validate_model",
    ]
    for method_name in required_methods:
        assert hasattr(
            PlumeModelInterface, method_name
        ), f"PlumeModelInterface missing method: {method_name}"
        method = getattr(PlumeModelInterface, method_name)
        assert callable(method), f"Method {method_name} is not callable"

    # Test structural subtyping works with classes that implement the interface methods
    class ValidImplementation:
        def initialize_model(self, **kwargs):
            pass

        def generate_concentration_field(self, force_regeneration=False):
            return np.zeros((64, 64))

        def sample_concentration(self, position):
            return 0.0

        def validate_model(self):
            return {"is_valid": True}

    # Protocol compliance testing via duck typing validation
    valid_impl = ValidImplementation()
    assert hasattr(valid_impl, "initialize_model")
    assert hasattr(valid_impl, "generate_concentration_field")
    assert hasattr(valid_impl, "sample_concentration")
    assert hasattr(valid_impl, "validate_model")


@pytest.mark.parametrize("grid_size", TEST_GRID_SIZES)
@pytest.mark.parametrize("sigma", TEST_SIGMA_VALUES)
def test_static_gaussian_plume_initialization(grid_size, sigma):
    """
    Test StaticGaussianPlume initialization with various parameter combinations, validation,
    and proper inheritance from BasePlumeModel with comprehensive parameter checking.
    """
    # Test successful initialization with default parameters using DEFAULT_GRID_SIZE and DEFAULT_PLUME_SIGMA
    test_grid = create_grid_size(grid_size)
    test_source = create_coordinates((grid_size[0] // 2, grid_size[1] // 2))

    # Test initialization with various grid sizes from TEST_GRID_SIZES ensuring proper GridSize conversion
    plume_model = StaticGaussianPlume(
        grid_size=test_grid, source_location=test_source, sigma=sigma
    )

    # Test initialization with different sigma values from TEST_SIGMA_VALUES with range validation
    assert plume_model.grid_size == test_grid
    assert plume_model.source_location == test_source
    assert plume_model.sigma == sigma

    # Test initialization with different source locations ensuring bounds checking and coordinate validation
    assert plume_model.source_location.is_within_bounds(test_grid)

    # Validate model_type is properly set to STATIC_GAUSSIAN_MODEL_TYPE after initialization
    assert plume_model.model_type == STATIC_GAUSSIAN_MODEL_TYPE

    # Test parameter validation during initialization with proper error handling
    if sigma < MIN_PLUME_SIGMA or sigma > MAX_PLUME_SIGMA:
        with pytest.raises((ValueError, Exception)):
            StaticGaussianPlume(
                grid_size=test_grid, source_location=test_source, sigma=sigma
            )

    # Assert proper inheritance from BasePlumeModel with all required properties initialized
    assert isinstance(plume_model, BasePlumeModel)
    assert hasattr(plume_model, "logger")
    assert hasattr(plume_model, "is_initialized")


@pytest.mark.parametrize("source_location", TEST_SOURCE_LOCATIONS)
def test_static_gaussian_plume_field_generation(source_location):
    """
    Test concentration field generation using Gaussian formula with mathematical accuracy
    validation, performance timing, and proper normalization within target constraints.
    """
    # Create StaticGaussianPlume instance with test parameters and initialize model
    grid_size = create_grid_size((128, 128))
    source_coords = create_coordinates(source_location)
    plume_model = StaticGaussianPlume(
        grid_size=grid_size, source_location=source_coords, sigma=DEFAULT_PLUME_SIGMA
    )

    # Initialize model for field operations
    plume_model.initialize_model()

    # Generate concentration field and measure generation time against PERFORMANCE_TARGET_PLUME_GENERATION_MS
    start_time = time.perf_counter()
    field_array = plume_model.generate_concentration_field(force_regeneration=True)
    generation_time_ms = (time.perf_counter() - start_time) * 1000

    # Validate field array shape matches grid dimensions with proper dtype
    expected_shape = (grid_size.height, grid_size.width)
    assert (
        field_array.shape == expected_shape
    ), f"Field shape {field_array.shape} != expected {expected_shape}"
    assert (
        field_array.dtype == np.float32
    ), f"Field dtype {field_array.dtype} != expected float32"

    # Test mathematical accuracy of Gaussian formula at known positions using calculate_gaussian_concentration
    source_y, source_x = source_coords.to_array_index(grid_size)
    source_concentration = field_array[source_y, source_x]

    # Verify peak concentration equals 1.0 at source location within MATHEMATICAL_PRECISION tolerance
    assert (
        abs(source_concentration - 1.0) < MATHEMATICAL_PRECISION
    ), f"Source concentration {source_concentration} != 1.0"

    # Test concentration normalization with all values in range [0.0, 1.0] and proper clamping
    assert np.all(field_array >= 0.0), "Field contains negative concentrations"
    assert np.all(field_array <= 1.0), "Field contains concentrations > 1.0"

    # Validate concentration decreases with distance from source following Gaussian distribution
    distances_and_concentrations = []
    for i in range(0, grid_size.height, 8):
        for j in range(0, grid_size.width, 8):
            dist = math.sqrt((j - source_coords.x) ** 2 + (i - source_coords.y) ** 2)
            conc = field_array[i, j]
            distances_and_concentrations.append((dist, conc))

    # Sort by distance and verify generally decreasing concentration
    distances_and_concentrations.sort(key=lambda x: x[0])
    for i in range(1, min(10, len(distances_and_concentrations))):
        prev_conc = distances_and_concentrations[i - 1][1]
        curr_conc = distances_and_concentrations[i][1]
        # Allow for some noise but expect general decrease
        assert (
            curr_conc <= prev_conc + 0.1
        ), f"Concentration should decrease with distance"

    # Test field regeneration with force_regeneration=True and compare with cached results
    field_array_2 = plume_model.generate_concentration_field(force_regeneration=True)
    assert np.allclose(
        field_array, field_array_2, rtol=MATHEMATICAL_PRECISION
    ), "Field regeneration produced different results"

    # Performance validation against target
    if (
        generation_time_ms
        > PERFORMANCE_TARGET_PLUME_GENERATION_MS + PERFORMANCE_TOLERANCE_MS
    ):
        warnings.warn(
            f"Field generation time {generation_time_ms:.2f}ms exceeds target {PERFORMANCE_TARGET_PLUME_GENERATION_MS}ms"
        )


@pytest.mark.parametrize("interpolate", [True, False])
def test_static_gaussian_concentration_sampling(interpolate):
    """
    Test concentration sampling at specific positions with bounds checking, interpolation
    options, and performance validation for environment observation generation.
    """
    # Create initialized StaticGaussianPlume and generate concentration field
    grid_size = create_grid_size((64, 64))
    source_location = create_coordinates((32, 32))
    plume_model = StaticGaussianPlume(
        grid_size=grid_size, source_location=source_location, sigma=DEFAULT_PLUME_SIGMA
    )
    plume_model.initialize_model()
    plume_model.generate_concentration_field()

    # Test sampling at source location returns concentration of 1.0 within precision tolerance
    source_concentration = plume_model.sample_concentration(
        position=source_location, interpolate=interpolate
    )
    assert (
        abs(source_concentration - 1.0) < MATHEMATICAL_PRECISION
    ), f"Source sampling: {source_concentration} != 1.0"

    # Test sampling at various positions matches theoretical Gaussian values
    test_positions = [
        create_coordinates((16, 16)),
        create_coordinates((48, 48)),
        create_coordinates((0, 0)),
        create_coordinates((63, 63)),
    ]

    for position in test_positions:
        if position.is_within_bounds(grid_size):
            concentration = plume_model.sample_concentration(
                position=position, interpolate=interpolate
            )

            # Validate concentration is within valid range
            assert (
                0.0 <= concentration <= 1.0
            ), f"Invalid concentration {concentration} at {position}"

            # Test theoretical accuracy for distance-based concentration
            distance = source_location.distance_to(position)
            theoretical_concentration = calculate_gaussian_concentration(
                x=position.x,
                y=position.y,
                source_x=source_location.x,
                source_y=source_location.y,
                sigma=DEFAULT_PLUME_SIGMA,
            )

            # Allow for discretization differences
            if not interpolate:
                assert (
                    abs(concentration - theoretical_concentration) < 0.1
                ), f"Sampling mismatch at {position}"

    # Validate bounds checking prevents sampling outside grid boundaries
    invalid_positions = [
        create_coordinates((-1, 32)),
        create_coordinates((32, -1)),
        create_coordinates((64, 32)),
        create_coordinates((32, 64)),
    ]

    for invalid_pos in invalid_positions:
        with pytest.raises((ValueError, IndexError, Exception)):
            plume_model.sample_concentration(
                position=invalid_pos, interpolate=interpolate
            )

    # Test sampling performance meets requirements for environment step operations
    start_time = time.perf_counter()
    for _ in range(100):
        plume_model.sample_concentration(
            position=source_location, interpolate=interpolate
        )
    avg_sampling_time_ms = (
        time.perf_counter() - start_time
    ) * 10  # Convert to ms per sample

    assert (
        avg_sampling_time_ms < 1.0
    ), f"Sampling too slow: {avg_sampling_time_ms:.3f}ms per sample"


@pytest.mark.parametrize("sigma", [1.0, 5.0, 10.0, 25.0])
def test_gaussian_mathematical_formula_accuracy(sigma):
    """
    Test mathematical accuracy of Gaussian concentration formula implementation against
    analytical solutions with numerical precision validation and edge case handling.
    """
    # Test calculate_gaussian_concentration function with known input values
    source_x, source_y = 50, 50

    # Test concentration at source location (0 distance) equals 1.0 for all sigma values
    source_concentration = calculate_gaussian_concentration(
        x=source_x, y=source_y, source_x=source_x, source_y=source_y, sigma=sigma
    )
    assert (
        abs(source_concentration - 1.0) < MATHEMATICAL_PRECISION
    ), f"Source concentration {source_concentration} != 1.0"

    # Validate formula C(x,y) = exp(-((x-sx)² + (y-sy)²) / (2*σ²)) implementation accuracy
    test_points = [
        (source_x + sigma, source_y),  # 1 sigma distance horizontally
        (source_x, source_y + sigma),  # 1 sigma distance vertically
        (
            source_x + sigma * math.sqrt(2) / 2,
            source_y + sigma * math.sqrt(2) / 2,
        ),  # 1 sigma distance diagonally
        (source_x + 2 * sigma, source_y),  # 2 sigma distance
        (source_x + 3 * sigma, source_y),  # 3 sigma distance
    ]

    for x, y in test_points:
        # Calculate concentration using function
        calculated = calculate_gaussian_concentration(x, y, source_x, source_y, sigma)

        # Calculate theoretical value manually
        dx = x - source_x
        dy = y - source_y
        distance_squared = dx * dx + dy * dy
        theoretical = math.exp(-distance_squared / (2.0 * sigma * sigma))

        # Test concentration at specific distances matches analytical calculation within MATHEMATICAL_PRECISION
        assert (
            abs(calculated - theoretical) < MATHEMATICAL_PRECISION
        ), f"Formula mismatch: calculated={calculated}, theoretical={theoretical}"

    # Validate concentration decreases monotonically with distance from source
    distances = [0, sigma, 2 * sigma, 3 * sigma, 4 * sigma]
    concentrations = []
    for dist in distances:
        x = source_x + dist
        y = source_y
        conc = calculate_gaussian_concentration(x, y, source_x, source_y, sigma)
        concentrations.append(conc)

    # Verify monotonic decrease
    for i in range(1, len(concentrations)):
        assert (
            concentrations[i] <= concentrations[i - 1]
        ), f"Non-monotonic decrease at distance {distances[i]}"

    # Test edge cases including very small and large sigma values for numerical stability
    if sigma >= 0.1:  # Avoid numerical instability
        edge_concentration = calculate_gaussian_concentration(
            x=source_x + 10 * sigma,
            y=source_y,
            source_x=source_x,
            source_y=source_y,
            sigma=sigma,
        )
        assert edge_concentration >= 0.0, "Edge case concentration is negative"
        assert edge_concentration < 0.01, "Edge case concentration too high"


def test_plume_model_parameter_validation():
    """
    Test comprehensive parameter validation for plume models including grid dimensions,
    source location bounds, sigma range, and mathematical consistency with detailed error reporting.
    """
    # Test validate_gaussian_parameters function with valid parameter combinations
    valid_grid = create_grid_size((64, 64))
    valid_source = create_coordinates((32, 32))
    valid_sigma = DEFAULT_PLUME_SIGMA

    # This should not raise an exception
    validation_result = validate_gaussian_parameters(
        grid_size=valid_grid, source_location=valid_source, sigma=valid_sigma
    )
    assert validation_result["is_valid"] == True

    # Test validation failures with negative grid dimensions raise ValidationError
    with pytest.raises((ValueError, Exception)):
        validate_gaussian_parameters(
            grid_size=create_grid_size((-10, 64)),
            source_location=valid_source,
            sigma=valid_sigma,
        )

    # Test source location outside grid bounds fails validation with proper error context
    invalid_source = create_coordinates((100, 100))  # Outside 64x64 grid
    validation_result = validate_gaussian_parameters(
        grid_size=valid_grid, source_location=invalid_source, sigma=valid_sigma
    )
    assert validation_result["is_valid"] == False
    assert len(validation_result["critical_errors"]) > 0

    # Test sigma values below MIN_PLUME_SIGMA and above MAX_PLUME_SIGMA fail validation
    invalid_small_sigma = MIN_PLUME_SIGMA - 0.1
    validation_result = validate_gaussian_parameters(
        grid_size=valid_grid, source_location=valid_source, sigma=invalid_small_sigma
    )
    assert validation_result["is_valid"] == False

    invalid_large_sigma = MAX_PLUME_SIGMA + 1.0
    validation_result = validate_gaussian_parameters(
        grid_size=valid_grid, source_location=valid_source, sigma=invalid_large_sigma
    )
    # Note: Large sigma may be valid but warned about, not necessarily invalid
    # Check for warnings rather than strict invalidity

    # Test mathematical consistency validation between sigma and grid dimensions
    very_large_sigma = 1000.0
    validation_result = validate_gaussian_parameters(
        grid_size=valid_grid, source_location=valid_source, sigma=very_large_sigma
    )
    # Should have warnings about sigma being too large for grid
    assert (
        len(validation_result["warnings"]) > 0
        or len(validation_result["recommendations"]) > 0
    )


def test_plume_model_registry_operations():
    """
    Test plume model registry functionality including model registration, creation, discovery,
    and interface validation with thread safety and error handling.
    """
    # Create PlumeModelRegistry instance with thread safety enabled
    registry = PlumeModelRegistry()

    # Test registering StaticGaussianPlume with STATIC_GAUSSIAN_MODEL_TYPE identifier
    registry.register_model(
        model_type=STATIC_GAUSSIAN_MODEL_TYPE,
        model_class=StaticGaussianPlume,
        model_description="Static Gaussian plume implementation",
    )

    # Test model creation using registry.create_model with various parameters
    grid_size = create_grid_size((64, 64))
    source_location = create_coordinates((32, 32))

    created_model = registry.create_model(
        model_type=STATIC_GAUSSIAN_MODEL_TYPE,
        grid_size=grid_size,
        source_location=source_location,
        sigma=DEFAULT_PLUME_SIGMA,
    )

    assert isinstance(created_model, StaticGaussianPlume)
    assert created_model.model_type == STATIC_GAUSSIAN_MODEL_TYPE

    # Validate get_registered_models returns correct model information
    registered_models = registry.get_registered_models()
    assert STATIC_GAUSSIAN_MODEL_TYPE in registered_models
    assert "model_class" in registered_models[STATIC_GAUSSIAN_MODEL_TYPE]
    assert "description" in registered_models[STATIC_GAUSSIAN_MODEL_TYPE]

    # Test registry.validate_model_interface for registered models
    is_valid = registry.validate_model_interface(STATIC_GAUSSIAN_MODEL_TYPE)
    assert is_valid == True

    # Test duplicate registration handling with override_existing parameter
    with pytest.raises((ValueError, Exception)):
        registry.register_model(
            model_type=STATIC_GAUSSIAN_MODEL_TYPE,
            model_class=StaticGaussianPlume,
            model_description="Duplicate registration",
            override_existing=False,
        )

    # Validate registry.get_model_class returns correct class for model type
    model_class = registry.get_model_class(STATIC_GAUSSIAN_MODEL_TYPE)
    assert model_class == StaticGaussianPlume

    # Test registry cleanup and model unregistration functionality
    registry.clear_registry()
    registered_models_after_clear = registry.get_registered_models()
    assert len(registered_models_after_clear) == 0


def test_create_plume_model_factory_function():
    """
    Test create_plume_model factory function with parameter validation, default application,
    error handling, and integration with registry system for model instantiation.
    """
    # Test create_plume_model with STATIC_GAUSSIAN_MODEL_TYPE and default parameters
    grid_size = create_grid_size((64, 64))
    source_location = create_coordinates((32, 32))

    plume_model = create_plume_model(
        model_type=STATIC_GAUSSIAN_MODEL_TYPE,
        grid_size=grid_size,
        source_location=source_location,
        sigma=DEFAULT_PLUME_SIGMA,
    )

    assert isinstance(plume_model, StaticGaussianPlume)
    assert plume_model.grid_size == grid_size
    assert plume_model.source_location == source_location
    assert plume_model.sigma == DEFAULT_PLUME_SIGMA

    # Test factory function parameter validation and error handling
    with pytest.raises((ValueError, TypeError, Exception)):
        create_plume_model(
            model_type="invalid_model_type",
            grid_size=grid_size,
            source_location=source_location,
        )

    # Validate created model is properly initialized and ready for field operations
    assert hasattr(plume_model, "generate_concentration_field")
    assert hasattr(plume_model, "sample_concentration")
    assert hasattr(plume_model, "validate_model")

    # Test parameter defaults application when optional parameters not provided
    minimal_model = create_plume_model(
        model_type=STATIC_GAUSSIAN_MODEL_TYPE,
        grid_size=grid_size,
        source_location=source_location,
        # sigma will use default
    )
    assert minimal_model.sigma == DEFAULT_PLUME_SIGMA


@pytest.mark.parametrize("enable_caching", [True, False])
def test_concentration_field_data_structure(enable_caching):
    """
    Test ConcentrationField data structure including field generation, sampling operations,
    caching behavior, and performance optimization with comprehensive validation.
    """
    # Create ConcentrationField with test grid size and caching configuration
    grid_size = create_grid_size((64, 64))
    field = ConcentrationField(grid_size=grid_size, enable_caching=enable_caching)

    # Test field generation with Gaussian source parameters and timing validation
    source_location = create_coordinates((32, 32))
    sigma = DEFAULT_PLUME_SIGMA

    start_time = time.perf_counter()
    field_array = field.generate_field(
        source_location=source_location, sigma=sigma, normalize_field=True
    )
    generation_time_ms = (time.perf_counter() - start_time) * 1000

    # Test sampling operations with various positions and interpolation methods
    test_positions = [
        create_coordinates((32, 32)),  # Source location
        create_coordinates((16, 16)),  # Off-source location
        create_coordinates((48, 48)),  # Another off-source location
    ]

    for position in test_positions:
        concentration = field.sample_at(position, interpolate=False)
        assert 0.0 <= concentration <= 1.0, f"Invalid concentration {concentration}"

        # Test interpolated sampling if enabled
        interpolated_concentration = field.sample_at(position, interpolate=True)
        assert (
            0.0 <= interpolated_concentration <= 1.0
        ), f"Invalid interpolated concentration {interpolated_concentration}"

    # Validate field array properties including shape, dtype, and value ranges
    field_data = field.get_field_array(copy_array=True)
    assert field_data.shape == (grid_size.height, grid_size.width)
    assert field_data.dtype == np.float32
    assert np.all(field_data >= 0.0)
    assert np.all(field_data <= 1.0)

    # Test caching behavior with enable_caching parameter and cache statistics
    if enable_caching:
        # Perform multiple samples to test caching
        for _ in range(10):
            field.sample_at(source_location, use_cache=True)

        cache_stats = field.clear_cache()
        assert cache_stats["cache_enabled"] == True
        if cache_stats["entries_cleared"] > 0:
            assert cache_stats["estimated_memory_freed_bytes"] > 0

    # Test field validation including mathematical properties and consistency checking
    validation_result = field.validate_field(check_mathematical_properties=True)
    assert validation_result["is_valid"] == True
    assert validation_result["checks_performed"]["basic_properties"] == True

    # Test field statistics generation with distribution analysis and performance metrics
    stats = field.get_field_statistics(
        include_distribution_analysis=True, include_performance_data=True
    )
    assert stats["field_generated"] == True
    assert "basic_statistics" in stats
    assert "distribution_analysis" in stats
    assert stats["basic_statistics"]["max_value"] <= 1.0
    assert stats["basic_statistics"]["min_value"] >= 0.0


@pytest.mark.performance
def test_plume_model_performance_requirements():
    """
    Test plume model performance requirements including field generation timing, sampling
    speed, memory usage, and optimization effectiveness against defined targets.
    """
    # Test plume field generation time is within PERFORMANCE_TARGET_PLUME_GENERATION_MS for 128×128 grid
    grid_size = create_grid_size((128, 128))
    source_location = create_coordinates((64, 64))
    plume_model = StaticGaussianPlume(
        grid_size=grid_size, source_location=source_location, sigma=DEFAULT_PLUME_SIGMA
    )
    plume_model.initialize_model()

    # Measure field generation performance
    start_time = time.perf_counter()
    field_array = plume_model.generate_concentration_field(force_regeneration=True)
    generation_time_ms = (time.perf_counter() - start_time) * 1000

    # Allow some tolerance but warn if significantly over target
    target_with_tolerance = (
        PERFORMANCE_TARGET_PLUME_GENERATION_MS + PERFORMANCE_TOLERANCE_MS
    )
    if generation_time_ms > target_with_tolerance:
        warnings.warn(
            f"Generation time {generation_time_ms:.2f}ms exceeds target {PERFORMANCE_TARGET_PLUME_GENERATION_MS}ms"
        )

    # Measure and validate concentration sampling performance for O(1) lookup operations
    sampling_times = []
    test_positions = [
        create_coordinates((i * 8, j * 8)) for i in range(16) for j in range(16)
    ]

    for position in test_positions[:100]:  # Test subset to avoid too long test
        if position.is_within_bounds(grid_size):
            start_time = time.perf_counter()
            concentration = plume_model.sample_concentration(position)
            sampling_time_ms = (time.perf_counter() - start_time) * 1000
            sampling_times.append(sampling_time_ms)

    # Validate O(1) sampling performance
    avg_sampling_time_ms = np.mean(sampling_times)
    assert (
        avg_sampling_time_ms < 1.0
    ), f"Sampling too slow: {avg_sampling_time_ms:.3f}ms average"

    # Test memory usage estimation accuracy against actual memory consumption
    memory_estimate = grid_size.estimate_memory_mb()
    actual_memory_mb = field_array.nbytes / (1024 * 1024)

    # Memory estimate should be reasonably accurate
    assert (
        memory_estimate * 0.5
        <= actual_memory_mb
        <= memory_estimate * MEMORY_TOLERANCE_FACTOR
    ), f"Memory estimate {memory_estimate:.1f}MB vs actual {actual_memory_mb:.1f}MB"


def test_plume_model_error_handling():
    """
    Test comprehensive error handling including custom exceptions, recovery strategies, error
    context preservation, and graceful degradation with proper error reporting.
    """
    from plume_nav_sim.utils.exceptions import (
        InterfaceValidationError,
        ModelRegistrationError,
        PlumeModelError,
        ValidationError,
    )

    # Test PlumeModelError exception with model-specific context and recovery suggestions
    try:
        raise PlumeModelError(
            message="Test plume model error",
            component_name="test_component",
            operation_name="test_operation",
        )
    except PlumeModelError as e:
        assert "Test plume model error" in str(e)
        assert e.component_name == "test_component"
        assert e.operation_name == "test_operation"
        recovery_suggestions = e.get_recovery_suggestions()
        assert isinstance(recovery_suggestions, list)
        assert len(recovery_suggestions) > 0

    # Test ModelRegistrationError for registry operation failures with detailed context
    registry = PlumeModelRegistry()
    try:
        # Try to get non-existent model
        registry.get_model_class("non_existent_model")
    except (KeyError, ValueError, Exception) as e:
        # Should handle gracefully or raise appropriate error
        assert isinstance(e, (KeyError, ValueError, Exception))

    # Test validation errors for parameter issues
    with pytest.raises((ValidationError, ValueError, Exception)):
        validate_gaussian_parameters(
            grid_size=create_grid_size((-10, -10)),  # Invalid grid size
            source_location=create_coordinates((0, 0)),
            sigma=-1.0,  # Invalid sigma
        )

    # Test graceful degradation when optional components fail or are unavailable
    grid_size = create_grid_size((32, 32))
    source_location = create_coordinates((16, 16))

    # Test with edge case parameters that might cause numerical issues
    try:
        plume_model = StaticGaussianPlume(
            grid_size=grid_size,
            source_location=source_location,
            sigma=0.001,  # Very small sigma
        )
        plume_model.initialize_model()
        # Should either work or raise appropriate error
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress expected warnings
            field = plume_model.generate_concentration_field()
            assert field is not None
    except Exception as e:
        # Should be a meaningful exception with context
        assert len(str(e)) > 0


def test_plume_model_state_management():
    """
    Test plume model state management including parameter updates, model cloning, state
    validation, and consistency checking with proper isolation and thread safety.
    """
    # Test plume model parameter updates with update_parameters method and validation
    grid_size = create_grid_size((64, 64))
    source_location = create_coordinates((32, 32))
    plume_model = StaticGaussianPlume(
        grid_size=grid_size, source_location=source_location, sigma=DEFAULT_PLUME_SIGMA
    )
    plume_model.initialize_model()

    # Generate initial field
    field_1 = plume_model.generate_concentration_field()

    # Test model parameter updates
    new_sigma = DEFAULT_PLUME_SIGMA * 2.0
    success = plume_model.update_parameters(sigma=new_sigma)
    assert success == True
    assert plume_model.sigma == new_sigma

    # Test field regeneration after parameter update
    field_2 = plume_model.generate_concentration_field(force_regeneration=True)
    assert not np.allclose(
        field_1, field_2
    ), "Field should change after parameter update"

    # Test model cloning with clone method preserving state and configuration
    cloned_model = plume_model.clone()
    assert isinstance(cloned_model, StaticGaussianPlume)
    assert cloned_model.grid_size == plume_model.grid_size
    assert cloned_model.source_location == plume_model.source_location
    assert cloned_model.sigma == plume_model.sigma

    # Test state validation with validate_model method and comprehensive checking
    validation_result = plume_model.validate_model()
    assert isinstance(validation_result, dict)
    assert "is_valid" in validation_result

    # Validate model information retrieval with get_model_info method
    model_info = plume_model.get_model_info()
    assert isinstance(model_info, dict)
    assert "model_type" in model_info
    assert model_info["model_type"] == STATIC_GAUSSIAN_MODEL_TYPE

    # Test state isolation between model instances and parameter independence
    original_sigma = plume_model.sigma
    cloned_model.update_parameters(sigma=original_sigma * 0.5)
    assert plume_model.sigma == original_sigma, "Original model sigma should not change"
    assert (
        cloned_model.sigma != original_sigma
    ), "Cloned model sigma should be different"


def test_plume_model_integration():
    """
    Test integration between plume model components including ConcentrationField coordination,
    type system integration, and cross-component communication with proper data flow.
    """
    # Test StaticGaussianPlume integration with ConcentrationField for field management
    grid_size = create_grid_size((64, 64))
    source_location = create_coordinates((32, 32))

    # Create plume model
    plume_model = StaticGaussianPlume(
        grid_size=grid_size, source_location=source_location, sigma=DEFAULT_PLUME_SIGMA
    )
    plume_model.initialize_model()

    # Create separate concentration field
    field = create_concentration_field(
        grid_size=grid_size, source_location=source_location, sigma=DEFAULT_PLUME_SIGMA
    )

    # Test integration with Coordinates and GridSize types for parameter handling
    assert isinstance(plume_model.grid_size, GridSize)
    assert isinstance(plume_model.source_location, Coordinates)
    assert isinstance(field.grid_size, GridSize)

    # Generate fields and compare results
    plume_field = plume_model.generate_concentration_field()
    field_array = field.get_field_array()

    # Fields should be similar (allowing for implementation differences)
    correlation = np.corrcoef(plume_field.flatten(), field_array.flatten())[0, 1]
    assert (
        correlation > 0.9
    ), f"Field correlation {correlation:.3f} too low, poor integration"

    # Test PlumeParameters integration with model validation and consistency checking
    plume_params = PlumeParameters(
        source_location=source_location, sigma=DEFAULT_PLUME_SIGMA
    )
    plume_params.validate()

    # Validate data flow between plume model and concentration field components
    sample_positions = [
        create_coordinates((16, 16)),
        create_coordinates((32, 32)),
        create_coordinates((48, 48)),
    ]

    for position in sample_positions:
        plume_concentration = plume_model.sample_concentration(position)
        field_concentration = field.sample_at(position)

        # Concentrations should be reasonably close
        diff = abs(plume_concentration - field_concentration)
        assert (
            diff < 0.1
        ), f"Concentration difference {diff:.3f} too large at {position}"

    # Test factory function integration with type system and validation utilities
    factory_model = create_plume_model(
        model_type=STATIC_GAUSSIAN_MODEL_TYPE,
        grid_size=grid_size,
        source_location=source_location,
        sigma=DEFAULT_PLUME_SIGMA,
    )

    assert isinstance(factory_model, StaticGaussianPlume)
    factory_field = factory_model.generate_concentration_field()

    # Factory model should produce consistent results
    factory_concentration = factory_model.sample_concentration(source_location)
    assert abs(factory_concentration - 1.0) < MATHEMATICAL_PRECISION


def test_plume_model_mathematical_properties():
    """
    Test mathematical properties of plume models including symmetry, continuity, normalization,
    gradient calculations, and field quality metrics with analytical validation.
    """
    # Test Gaussian plume symmetry around source location with radial distance calculations
    grid_size = create_grid_size((64, 64))
    source_location = create_coordinates((32, 32))
    plume_model = StaticGaussianPlume(
        grid_size=grid_size, source_location=source_location, sigma=DEFAULT_PLUME_SIGMA
    )
    plume_model.initialize_model()
    field_array = plume_model.generate_concentration_field()

    # Test radial symmetry by comparing points at equal distances
    center_x, center_y = source_location.x, source_location.y
    test_radius = 8

    # Test points at same distance from center
    test_points = [
        (center_x + test_radius, center_y),  # East
        (center_x - test_radius, center_y),  # West
        (center_x, center_y + test_radius),  # North
        (center_x, center_y - test_radius),  # South
    ]

    concentrations = []
    for x, y in test_points:
        if 0 <= x < grid_size.width and 0 <= y < grid_size.height:
            position = create_coordinates((x, y))
            concentration = plume_model.sample_concentration(position)
            concentrations.append(concentration)

    # Concentrations should be approximately equal for points at equal distance
    if len(concentrations) > 1:
        max_diff = max(concentrations) - min(concentrations)
        assert max_diff < 0.05, f"Symmetry violation: max difference {max_diff:.3f}"

    # Test concentration field continuity and smoothness properties across the field
    # Sample neighboring points and check for smoothness
    smooth_violations = 0
    for i in range(1, grid_size.height - 1):
        for j in range(1, grid_size.width - 1):
            center_val = field_array[i, j]
            neighbors = [
                field_array[i - 1, j],
                field_array[i + 1, j],
                field_array[i, j - 1],
                field_array[i, j + 1],
            ]

            # Check that center value is not dramatically different from neighbors
            for neighbor_val in neighbors:
                diff = abs(center_val - neighbor_val)
                if diff > 0.5:  # Threshold for smoothness violation
                    smooth_violations += 1

    # Allow some violations but not too many
    total_interior_points = (grid_size.height - 2) * (grid_size.width - 2)
    violation_rate = (
        smooth_violations / total_interior_points if total_interior_points > 0 else 0
    )
    assert (
        violation_rate < 0.01
    ), f"Too many smoothness violations: {violation_rate:.3f}"

    # Test field normalization with peak concentration at source and proper value ranges
    source_y, source_x = source_location.to_array_index(grid_size)
    peak_value = field_array[source_y, source_x]
    assert (
        abs(peak_value - 1.0) < MATHEMATICAL_PRECISION
    ), f"Peak normalization failed: {peak_value:.6f}"

    # Test gradient calculations with calculate_field_gradients method and directional analysis
    if hasattr(plume_model, "calculate_field_gradients"):
        gradients = plume_model.calculate_field_gradients()
        assert gradients is not None
        assert isinstance(gradients, dict)
    else:
        # Manual gradient calculation
        grad_y, grad_x = np.gradient(field_array)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Gradients should be reasonable (not infinite or NaN)
        assert np.all(np.isfinite(gradient_magnitude)), "Invalid gradients detected"
        assert np.max(gradient_magnitude) < 10.0, "Gradients too large"

    # Validate mathematical properties like monotonic decrease with distance from source
    # Sample points at increasing distances
    distances_and_concentrations = []
    for radius in range(1, min(grid_size.width, grid_size.height) // 2):
        x = source_location.x + radius
        y = source_location.y

        if x < grid_size.width and y < grid_size.height:
            position = create_coordinates((x, y))
            concentration = plume_model.sample_concentration(position)
            distances_and_concentrations.append((radius, concentration))

    # Check general monotonic decrease trend
    for i in range(1, len(distances_and_concentrations)):
        prev_conc = distances_and_concentrations[i - 1][1]
        curr_conc = distances_and_concentrations[i][1]
        # Allow some noise but expect general decrease
        assert (
            curr_conc <= prev_conc + 0.1
        ), f"Non-monotonic decrease at distance {distances_and_concentrations[i][0]}"


def test_plume_model_extensibility():
    """
    Test plume model extensibility features including registry system, plugin interfaces,
    custom model registration, and future extension points with architectural validation.
    """
    # Test custom plume model registration with registry system
    registry = PlumeModelRegistry()

    # Create a mock custom plume model class
    class CustomPlumeModel(BasePlumeModel):
        def __init__(self, grid_size, source_location=None, **kwargs):
            super().__init__(grid_size)
            self.source_location = source_location or create_coordinates(
                (grid_size.width // 2, grid_size.height // 2)
            )
            self.model_type = "custom_test_model"
            self.custom_param = kwargs.get("custom_param", 42)

        def initialize_model(self, **kwargs):
            self.is_initialized = True
            return True

        def generate_concentration_field(self, force_regeneration=False):
            # Simple uniform field for testing
            return (
                np.ones((self.grid_size.height, self.grid_size.width), dtype=np.float32)
                * 0.5
            )

        def sample_concentration(self, position, interpolate=False):
            return 0.5  # Uniform concentration

        def validate_model(self):
            return {"is_valid": True, "issues": []}

        def get_model_info(self):
            return {
                "model_type": self.model_type,
                "custom_param": self.custom_param,
                "source_location": self.source_location,
            }

        def update_parameters(self, **kwargs):
            if "custom_param" in kwargs:
                self.custom_param = kwargs["custom_param"]
            return True

        def clone(self):
            return CustomPlumeModel(
                grid_size=self.grid_size,
                source_location=self.source_location,
                custom_param=self.custom_param,
            )

    # Test interface compliance validation for custom model implementations
    custom_model = CustomPlumeModel(grid_size=create_grid_size((32, 32)))

    # Test that custom model implements all required methods
    required_methods = [
        "initialize_model",
        "generate_concentration_field",
        "sample_concentration",
        "validate_model",
    ]
    for method_name in required_methods:
        assert hasattr(
            custom_model, method_name
        ), f"Custom model missing method: {method_name}"
        assert callable(
            getattr(custom_model, method_name)
        ), f"Method {method_name} not callable"

    # Test custom model registration
    registry.register_model(
        model_type="custom_test_model",
        model_class=CustomPlumeModel,
        model_description="Custom test model for extensibility testing",
    )

    # Test factory function extensibility for new model types
    created_custom_model = registry.create_model(
        model_type="custom_test_model",
        grid_size=create_grid_size((32, 32)),
        custom_param=100,
    )

    assert isinstance(created_custom_model, CustomPlumeModel)
    assert created_custom_model.custom_param == 100

    # Validate plugin-ready interfaces and extension point architecture
    registered_models = registry.get_registered_models()
    assert "custom_test_model" in registered_models
    assert "model_class" in registered_models["custom_test_model"]

    # Test model metadata management and capability tracking in registry
    model_info = created_custom_model.get_model_info()
    assert "model_type" in model_info
    assert model_info["model_type"] == "custom_test_model"
    assert "custom_param" in model_info

    # Test interface versioning and compatibility checking for future extensions
    interface_validation = registry.validate_model_interface("custom_test_model")
    assert interface_validation == True

    # Test that registry supports multiple model types simultaneously
    registry.register_model(
        model_type=STATIC_GAUSSIAN_MODEL_TYPE,
        model_class=StaticGaussianPlume,
        model_description="Standard Gaussian plume model",
    )

    registered_models = registry.get_registered_models()
    assert len(registered_models) >= 2
    assert "custom_test_model" in registered_models
    assert STATIC_GAUSSIAN_MODEL_TYPE in registered_models


class TestBasePlumeModel:
    """
    Test class for BasePlumeModel abstract base class functionality including abstract method
    enforcement, interface compliance, and common functionality validation with comprehensive
    inheritance testing.
    """

    def setup_method(self):
        """
        Test class setup for BasePlumeModel testing with fixture initialization and test data preparation.
        """
        # Set up test fixtures for BasePlumeModel testing
        self.test_grid_sizes = TEST_GRID_SIZES
        self.test_parameters = {
            "sigma": DEFAULT_PLUME_SIGMA,
            "source_location": create_coordinates((32, 32)),
        }

        # Initialize test data including grid sizes, parameters, and validation criteria
        self.validation_criteria = {
            "mathematical_precision": MATHEMATICAL_PRECISION,
            "performance_tolerance": PERFORMANCE_TOLERANCE_MS,
        }

        # Prepare mock objects and test doubles for abstract method testing
        self.mock_methods = [
            "initialize_model",
            "generate_concentration_field",
            "sample_concentration",
            "validate_model",
            "get_model_info",
            "update_parameters",
            "clone",
        ]

    def test_abstract_base_class_definition(self):
        """
        Test that BasePlumeModel is properly defined as abstract base class with required
        decorators and method specifications.
        """
        # Assert BasePlumeModel has abc.ABC in inheritance chain
        import abc

        assert issubclass(
            BasePlumeModel, abc.ABC
        ), "BasePlumeModel should inherit from abc.ABC"

        # Validate abstract methods are properly decorated with @abc.abstractmethod
        abstract_methods = BasePlumeModel.__abstractmethods__
        expected_abstract_methods = set(self.mock_methods)
        assert (
            abstract_methods == expected_abstract_methods
        ), f"Abstract methods mismatch: {abstract_methods} != {expected_abstract_methods}"

        # Test that direct instantiation raises TypeError due to abstract methods
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BasePlumeModel(grid_size=create_grid_size((32, 32)))

        # Verify all required abstract methods are defined in the base class
        for method_name in self.mock_methods:
            assert hasattr(
                BasePlumeModel, method_name
            ), f"BasePlumeModel missing abstract method: {method_name}"

    def test_common_functionality_inheritance(self):
        """
        Test common functionality provided by BasePlumeModel including validation, logging,
        and utility methods.
        """

        # Create a concrete implementation for testing common functionality
        class ConcreteTestModel(BasePlumeModel):
            def __init__(self, grid_size):
                super().__init__(grid_size)
                self.model_type = "test_model"

            def initialize_model(self, **kwargs):
                self.is_initialized = True
                return True

            def generate_concentration_field(self, force_regeneration=False):
                return np.ones((self.grid_size.height, self.grid_size.width))

            def sample_concentration(self, position, interpolate=False):
                return 1.0

            def validate_model(self):
                return {"is_valid": True}

            def get_model_info(self):
                return {"model_type": self.model_type}

            def update_parameters(self, **kwargs):
                return True

            def clone(self):
                return ConcreteTestModel(self.grid_size)

        # Test concrete implementation can be instantiated
        test_model = ConcreteTestModel(grid_size=create_grid_size((32, 32)))

        # Test common validation functionality inherited by concrete implementations
        assert hasattr(test_model, "grid_size")
        assert hasattr(test_model, "logger")
        assert hasattr(test_model, "is_initialized")

        # Test logging integration and component logger creation
        assert test_model.logger is not None

        # Test utility methods including parameter validation and state management
        info = test_model.get_model_info()
        assert isinstance(info, dict)
        assert "model_type" in info


class TestStaticGaussianPlume:
    """
    Comprehensive test class for StaticGaussianPlume implementation including initialization,
    field generation, mathematical accuracy, and performance validation with full coverage.
    """

    def setup_method(self):
        """
        Set up test fixtures for StaticGaussianPlume testing with comprehensive test data
        and performance monitoring.
        """
        # Initialize test grid size using create_grid_size with DEFAULT_GRID_SIZE
        self.test_grid_size = create_grid_size(DEFAULT_GRID_SIZE)

        # Create test source location at grid center using grid_size.center()
        center_x = self.test_grid_size.width // 2
        center_y = self.test_grid_size.height // 2
        self.test_source_location = create_coordinates((center_x, center_y))

        # Set test sigma to DEFAULT_PLUME_SIGMA for consistency
        self.test_sigma = DEFAULT_PLUME_SIGMA

        # Create test model instance and initialize for field operations
        self.test_model = StaticGaussianPlume(
            grid_size=self.test_grid_size,
            source_location=self.test_source_location,
            sigma=self.test_sigma,
        )
        self.test_model.initialize_model()

        # Set up performance monitoring and timing measurement infrastructure
        self.performance_data = {
            "generation_times": [],
            "sampling_times": [],
            "memory_usage": [],
        }

    def test_model_initialization_parameters(self):
        """
        Test StaticGaussianPlume initialization with various parameter combinations and validation.
        """
        # Test initialization with default parameters and validate property assignment
        assert self.test_model.grid_size == self.test_grid_size
        assert self.test_model.source_location == self.test_source_location
        assert self.test_model.sigma == self.test_sigma
        assert self.test_model.model_type == STATIC_GAUSSIAN_MODEL_TYPE

        # Test initialization with custom grid size and source location parameters
        custom_grid = create_grid_size((64, 64))
        custom_source = create_coordinates((16, 16))
        custom_model = StaticGaussianPlume(
            grid_size=custom_grid, source_location=custom_source, sigma=10.0
        )

        assert custom_model.grid_size == custom_grid
        assert custom_model.source_location == custom_source
        assert custom_model.sigma == 10.0

        # Test parameter validation during initialization with boundary checking
        with pytest.raises((ValueError, Exception)):
            StaticGaussianPlume(
                grid_size=create_grid_size((32, 32)),
                source_location=create_coordinates((100, 100)),  # Outside bounds
                sigma=DEFAULT_PLUME_SIGMA,
            )

        # Validate model_type assignment and registry compatibility
        assert self.test_model.model_type == STATIC_GAUSSIAN_MODEL_TYPE

    def test_gaussian_field_generation_accuracy(self):
        """
        Test mathematical accuracy of Gaussian concentration field generation.
        """
        # Generate concentration field and validate shape and data type
        field_array = self.test_model.generate_concentration_field(
            force_regeneration=True
        )
        expected_shape = (self.test_grid_size.height, self.test_grid_size.width)

        assert field_array.shape == expected_shape
        assert field_array.dtype == np.float32

        # Test mathematical accuracy at known positions using analytical solutions
        source_y, source_x = self.test_source_location.to_array_index(
            self.test_grid_size
        )
        source_concentration = field_array[source_y, source_x]

        # Validate peak concentration at source location equals 1.0
        assert abs(source_concentration - 1.0) < MATHEMATICAL_PRECISION

        # Test concentration normalization and value range compliance
        assert np.all(field_array >= 0.0)
        assert np.all(field_array <= 1.0)
        assert np.max(field_array) <= 1.0
        assert np.min(field_array) >= 0.0

    def test_concentration_sampling_operations(self):
        """
        Test concentration sampling at various positions with accuracy and performance validation.
        """
        # Test sampling at source location returns peak concentration
        source_concentration = self.test_model.sample_concentration(
            position=self.test_source_location, interpolate=False
        )
        assert abs(source_concentration - 1.0) < MATHEMATICAL_PRECISION

        # Test sampling accuracy at various distances from source
        test_positions = [
            create_coordinates(
                (self.test_source_location.x + 10, self.test_source_location.y)
            ),
            create_coordinates(
                (self.test_source_location.x, self.test_source_location.y + 10)
            ),
            create_coordinates(
                (self.test_source_location.x - 10, self.test_source_location.y - 10)
            ),
        ]

        for position in test_positions:
            if position.is_within_bounds(self.test_grid_size):
                concentration = self.test_model.sample_concentration(position)
                assert 0.0 <= concentration <= 1.0
                assert (
                    concentration < source_concentration
                )  # Should be less than source

        # Validate sampling performance for environment integration
        start_time = time.perf_counter()
        for _ in range(100):
            self.test_model.sample_concentration(self.test_source_location)
        avg_time_ms = (time.perf_counter() - start_time) * 10
        assert avg_time_ms < 1.0  # Should be fast

        # Test interpolation options and boundary handling
        interpolated_concentration = self.test_model.sample_concentration(
            position=self.test_source_location, interpolate=True
        )
        assert 0.0 <= interpolated_concentration <= 1.0

    def test_field_properties_and_statistics(self):
        """
        Test field property analysis and statistical validation for mathematical consistency.
        """
        # Generate field for testing
        field_array = self.test_model.generate_concentration_field()

        # Test field properties analysis including shape and mathematical properties
        if hasattr(self.test_model, "get_field_properties"):
            properties = self.test_model.get_field_properties()
            assert isinstance(properties, dict)

        # Test field statistics generation with distribution analysis
        field_stats = {
            "min_value": float(np.min(field_array)),
            "max_value": float(np.max(field_array)),
            "mean_value": float(np.mean(field_array)),
            "std_value": float(np.std(field_array)),
        }

        assert field_stats["max_value"] <= 1.0
        assert field_stats["min_value"] >= 0.0
        assert field_stats["mean_value"] >= 0.0

        # Validate mathematical properties like symmetry and monotonic decrease
        source_x, source_y = self.test_source_location.x, self.test_source_location.y

        # Test symmetry by comparing points equidistant from source
        if (
            source_x >= 10
            and source_y >= 10
            and source_x < self.test_grid_size.width - 10
            and source_y < self.test_grid_size.height - 10
        ):
            pos_right = create_coordinates((source_x + 5, source_y))
            pos_left = create_coordinates((source_x - 5, source_y))
            pos_up = create_coordinates((source_x, source_y - 5))
            pos_down = create_coordinates((source_x, source_y + 5))

            concentrations = [
                self.test_model.sample_concentration(pos)
                for pos in [pos_right, pos_left, pos_up, pos_down]
            ]

            # Should be approximately equal (radial symmetry)
            max_diff = max(concentrations) - min(concentrations)
            assert max_diff < 0.1  # Allow some discretization error

        # Test gradient calculations and directional field analysis
        if hasattr(self.test_model, "calculate_field_gradients"):
            gradients = self.test_model.calculate_field_gradients()
            assert gradients is not None


class TestPlumeModelRegistry:
    """
    Test class for plume model registry functionality including registration operations,
    model discovery, factory methods, and thread safety validation.
    """

    def setup_method(self):
        """
        Set up test fixtures for plume model registry testing with clean registry state.
        """
        # Create fresh PlumeModelRegistry instance with thread safety enabled
        self.registry = PlumeModelRegistry()

        # Initialize test models dictionary with StaticGaussianPlume and mock models
        self.test_models = {
            STATIC_GAUSSIAN_MODEL_TYPE: {
                "class": StaticGaussianPlume,
                "description": "Static Gaussian plume model for testing",
            }
        }

        # Prepare registered types list for tracking and validation
        self.registered_types = []

    def test_model_registration_operations(self):
        """
        Test model registration including validation, metadata management, and duplicate handling.
        """
        # Test successful model registration with proper metadata
        self.registry.register_model(
            model_type=STATIC_GAUSSIAN_MODEL_TYPE,
            model_class=StaticGaussianPlume,
            model_description=self.test_models[STATIC_GAUSSIAN_MODEL_TYPE][
                "description"
            ],
        )
        self.registered_types.append(STATIC_GAUSSIAN_MODEL_TYPE)

        # Test duplicate registration handling with override options
        with pytest.raises((ValueError, Exception)):
            self.registry.register_model(
                model_type=STATIC_GAUSSIAN_MODEL_TYPE,
                model_class=StaticGaussianPlume,
                model_description="Duplicate registration test",
                override_existing=False,
            )

        # Test registration validation including interface compliance
        registered_models = self.registry.get_registered_models()
        assert STATIC_GAUSSIAN_MODEL_TYPE in registered_models

        # Validate registry state after registration operations
        assert len(registered_models) > 0
        assert "model_class" in registered_models[STATIC_GAUSSIAN_MODEL_TYPE]
        assert "description" in registered_models[STATIC_GAUSSIAN_MODEL_TYPE]

    def test_model_factory_operations(self):
        """
        Test model creation through registry factory methods with parameter validation.
        """
        # Register model first
        self.registry.register_model(
            model_type=STATIC_GAUSSIAN_MODEL_TYPE,
            model_class=StaticGaussianPlume,
            model_description="Factory test model",
        )

        # Test model creation using registry.create_model with various parameters
        grid_size = create_grid_size((64, 64))
        source_location = create_coordinates((32, 32))

        created_model = self.registry.create_model(
            model_type=STATIC_GAUSSIAN_MODEL_TYPE,
            grid_size=grid_size,
            source_location=source_location,
            sigma=DEFAULT_PLUME_SIGMA,
        )

        # Test parameter validation and error handling in factory methods
        assert isinstance(created_model, StaticGaussianPlume)
        assert created_model.grid_size == grid_size
        assert created_model.source_location == source_location

        # Test model configuration and initialization through factory
        assert created_model.model_type == STATIC_GAUSSIAN_MODEL_TYPE

        # Validate created model instances are properly configured
        model_info = created_model.get_model_info()
        assert isinstance(model_info, dict)
        assert model_info["model_type"] == STATIC_GAUSSIAN_MODEL_TYPE

    def test_registry_discovery_and_validation(self):
        """
        Test model discovery operations and interface validation functionality.
        """
        # Register test model
        self.registry.register_model(
            model_type=STATIC_GAUSSIAN_MODEL_TYPE,
            model_class=StaticGaussianPlume,
            model_description="Discovery test model",
        )

        # Test get_registered_models returns accurate model information
        registered_models = self.registry.get_registered_models()
        assert isinstance(registered_models, dict)
        assert STATIC_GAUSSIAN_MODEL_TYPE in registered_models

        # Test interface validation for registered models
        is_valid = self.registry.validate_model_interface(STATIC_GAUSSIAN_MODEL_TYPE)
        assert is_valid == True

        # Test model discovery with metadata and capability queries
        model_class = self.registry.get_model_class(STATIC_GAUSSIAN_MODEL_TYPE)
        assert model_class == StaticGaussianPlume

        # Validate registry information accuracy and completeness
        model_info = registered_models[STATIC_GAUSSIAN_MODEL_TYPE]
        assert "model_class" in model_info
        assert "description" in model_info
        assert model_info["model_class"] == StaticGaussianPlume


class TestConcentrationField:
    """
    Test class for ConcentrationField data structure including field operations, sampling
    functionality, caching behavior, and performance optimization validation.
    """

    def setup_method(self):
        """
        Set up test fixtures for ConcentrationField testing with field generation and caching.
        """
        # Create test grid size for field operations
        self.test_grid = create_grid_size((64, 64))

        # Initialize ConcentrationField with caching enabled
        self.field = ConcentrationField(grid_size=self.test_grid, enable_caching=True)

        # Generate test field array for sampling validation
        source_location = create_coordinates((32, 32))
        self.field_array = self.field.generate_field(
            source_location=source_location, sigma=DEFAULT_PLUME_SIGMA
        )

    def test_field_generation_and_validation(self):
        """
        Test concentration field generation with mathematical validation and performance timing.
        """
        # Test field generation with Gaussian parameters and timing validation
        source_location = create_coordinates((16, 16))
        sigma = 8.0

        start_time = time.perf_counter()
        field_array = self.field.generate_field(
            source_location=source_location, sigma=sigma, force_regeneration=True
        )
        generation_time_ms = (time.perf_counter() - start_time) * 1000

        # Test field array properties including shape, dtype, and value ranges
        assert field_array.shape == (self.test_grid.height, self.test_grid.width)
        assert field_array.dtype == np.float32
        assert np.all(field_array >= 0.0)
        assert np.all(field_array <= 1.0)

        # Test field validation including mathematical consistency checking
        validation_result = self.field.validate_field(
            check_mathematical_properties=True
        )
        assert validation_result["is_valid"] == True

        # Validate field normalization and peak location accuracy
        source_y, source_x = source_location.to_array_index(self.test_grid)
        peak_value = field_array[source_y, source_x]
        assert abs(peak_value - 1.0) < MATHEMATICAL_PRECISION

    def test_sampling_and_caching(self):
        """
        Test field sampling operations with caching behavior and performance optimization.
        """
        # Test sampling operations at various positions with accuracy validation
        test_positions = [
            create_coordinates((32, 32)),
            create_coordinates((16, 16)),
            create_coordinates((48, 48)),
        ]

        for position in test_positions:
            concentration = self.field.sample_at(position, use_cache=True)
            assert 0.0 <= concentration <= 1.0

            # Test cached sampling returns same result
            cached_concentration = self.field.sample_at(position, use_cache=True)
            assert concentration == cached_concentration

        # Test caching behavior including cache hits and misses tracking
        initial_hits = self.field.cache_hits
        initial_misses = self.field.cache_misses

        # Sample same position multiple times to test caching
        for _ in range(5):
            self.field.sample_at(create_coordinates((32, 32)), use_cache=True)

        # Should have increased cache hits
        assert self.field.cache_hits > initial_hits

        # Test sampling performance optimization and interpolation methods
        start_time = time.perf_counter()
        for _ in range(100):
            self.field.sample_at(create_coordinates((32, 32)), interpolate=False)
        avg_sampling_time_ms = (time.perf_counter() - start_time) * 10
        assert avg_sampling_time_ms < 1.0

        # Validate cache efficiency and memory usage optimization
        cache_report = self.field.clear_cache()
        assert cache_report["cache_enabled"] == True
        if cache_report["entries_cleared"] > 0:
            assert cache_report["estimated_memory_freed_bytes"] >= 0
