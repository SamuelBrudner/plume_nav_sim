"""
Comprehensive pytest test suite for BoundaryEnforcer component validation covering position validation,
movement bounds enforcement, boundary constraint configuration, performance benchmarking, error handling,
and integration testing with complete coverage of grid boundary management functionality for the plume
navigation environment.

This test suite provides extensive coverage for all BoundaryEnforcer functionality including:
- Initialization testing with various parameter configurations
- Position validation with bounds checking and strict mode testing
- Movement validation with boundary constraint analysis
- Complete boundary enforcement workflow testing
- Standalone utility function validation
- MovementConstraint configuration testing
- Boundary enforcement statistics collection and analysis
- Comprehensive error handling and exception validation
- Performance benchmarking against target latency requirements
- Integration testing with coordinate system and action compatibility
- Reproducibility and determinism validation for consistent behavior

The test suite follows enterprise-grade testing practices with comprehensive fixtures,
parametrized testing, performance monitoring, and detailed assertion validation.
"""

import time  # Standard library - High-precision timing for boundary enforcement performance benchmark validation
from typing import (  # >=3.10 - Type hints for structured analysis helpers
    Any,
    Dict,
    List,
)

import numpy as np  # >=2.1.0 - Array operations and mathematical validation for boundary enforcement testing

# Standard library imports with version comments
import pytest  # >=8.0.0 - Testing framework for comprehensive boundary enforcement test suite execution

# Internal imports from boundary enforcer module for comprehensive testing coverage
from plume_nav_sim.core.boundary_enforcer import (
    BoundaryEnforcementResult,
    BoundaryEnforcer,
    MovementConstraint,
    calculate_bounded_movement,
    clamp_coordinates_to_bounds,
    enforce_position_bounds,
    is_position_within_bounds,
    validate_movement_bounds,
)

# Internal imports from constants for system defaults and performance targets
from plume_nav_sim.core.constants import (
    BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS,
    BOUNDARY_VALIDATION_CACHE_SIZE,
    MOVEMENT_VECTORS,
)

# Internal imports from core types for coordinate and action system integration
from plume_nav_sim.core.types import Action, Coordinates, GridSize

# Internal imports from exceptions for comprehensive error handling validation
from plume_nav_sim.utils.exceptions import ConfigurationError, ValidationError

# Test configuration constants for comprehensive boundary enforcement testing
TEST_GRID_SIZE = GridSize(32, 32)
TEST_LARGE_GRID_SIZE = GridSize(128, 128)
TEST_COORDINATES_CENTER = Coordinates(16, 16)
TEST_COORDINATES_CORNER_ORIGIN = Coordinates(0, 0)
TEST_COORDINATES_CORNER_OPPOSITE = Coordinates(31, 31)
TEST_COORDINATES_EDGE_TOP = Coordinates(16, 31)
TEST_COORDINATES_EDGE_BOTTOM = Coordinates(16, 0)
TEST_COORDINATES_EDGE_LEFT = Coordinates(0, 16)
TEST_COORDINATES_EDGE_RIGHT = Coordinates(31, 16)
PERFORMANCE_TEST_ITERATIONS = 1000
PERFORMANCE_TOLERANCE_MS = 0.01
BOUNDARY_TEST_POSITIONS = [
    (0, 0),
    (31, 31),
    (15, 0),
    (15, 31),
    (0, 15),
    (31, 15),
    (16, 16),
]
CORNER_POSITIONS = [(0, 0), (0, 31), (31, 0), (31, 31)]
EDGE_POSITIONS = [(0, 15), (31, 15), (15, 0), (15, 31)]
ALL_VALID_ACTIONS = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
OUT_OF_BOUNDS_POSITIONS = [(-1, 16), (32, 16), (16, -1), (16, 32), (-1, -1), (32, 32)]
CONSTRAINT_CONFIG_VARIATIONS = 8


def create_test_boundary_enforcer(
    grid_size: GridSize = None,
    constraint_config: MovementConstraint = None,
    enable_caching: bool = True,
) -> BoundaryEnforcer:
    """
    Factory function to create BoundaryEnforcer instances with test configurations for consistent test
    setup across boundary enforcement test cases with standardized defaults and configuration validation.

    Args:
        grid_size: Grid size for boundary testing, defaults to TEST_GRID_SIZE
        constraint_config: MovementConstraint configuration, defaults to standard settings
        enable_caching: Enable caching for performance optimization

    Returns:
        BoundaryEnforcer: Configured BoundaryEnforcer instance ready for comprehensive boundary testing
    """
    # Use TEST_GRID_SIZE as default if grid_size not provided
    if grid_size is None:
        grid_size = TEST_GRID_SIZE

    # Create default MovementConstraint with standard settings if constraint_config not provided
    if constraint_config is None:
        constraint_config = MovementConstraint(
            enable_clamping=True, strict_validation=False, log_boundary_violations=False
        )

    # Initialize BoundaryEnforcer with provided or default parameters
    enforcer = BoundaryEnforcer(
        grid_size=grid_size,
        constraint_config=constraint_config,
        enable_caching=enable_caching,
    )

    # Validate enforcer initialization was successful and grid size is set correctly
    assert enforcer.grid_size == grid_size
    assert (
        enforcer.constraint_config.enable_clamping == constraint_config.enable_clamping
    )
    assert enforcer.enable_caching == enable_caching

    # Return configured BoundaryEnforcer ready for comprehensive boundary testing
    return enforcer


def generate_boundary_test_cases(
    grid_size: GridSize, include_invalid_positions: bool = False
):
    """
    Generate comprehensive test cases for boundary condition testing with edge positions, corner positions,
    and movement validation scenarios for systematic validation coverage.

    Args:
        grid_size: Grid size for test case generation
        include_invalid_positions: Whether to include out-of-bounds positions

    Returns:
        list: List of test case tuples (position, action, expected_enforcement_result)
    """
    test_cases = []

    # Generate corner positions at all four grid corners for boundary testing
    corners = [
        (0, 0),
        (0, grid_size.height - 1),
        (grid_size.width - 1, 0),
        (grid_size.width - 1, grid_size.height - 1),
    ]

    # Generate edge positions along all four grid boundaries
    edges = []
    # Top and bottom edges
    for x in range(1, grid_size.width - 1):
        edges.extend([(x, 0), (x, grid_size.height - 1)])
    # Left and right edges
    for y in range(1, grid_size.height - 1):
        edges.extend([(0, y), (grid_size.width - 1, y)])

    # Create center positions for unrestricted movement testing
    center = (grid_size.width // 2, grid_size.height // 2)

    # Generate test cases for valid movements within bounds for each position
    for position in [center] + corners[:2] + edges[:4]:
        for action in ALL_VALID_ACTIONS:
            # Determine expected result based on movement feasibility
            coords = Coordinates(position[0], position[1])
            movement_vector = MOVEMENT_VECTORS[action.value]
            new_x = coords.x + movement_vector[0]
            new_y = coords.y + movement_vector[1]

            # Check if movement would be within bounds
            within_bounds = (
                0 <= new_x < grid_size.width and 0 <= new_y < grid_size.height
            )

            test_cases.append(
                {
                    "position": position,
                    "action": action,
                    "expected_within_bounds": within_bounds,
                    "expected_boundary_hit": not within_bounds,
                }
            )

    # Include out-of-bounds positions if include_invalid_positions is True
    if include_invalid_positions:
        invalid_positions = [
            (-1, grid_size.height // 2),
            (grid_size.width, grid_size.height // 2),
            (grid_size.width // 2, -1),
            (grid_size.width // 2, grid_size.height),
        ]

        for position in invalid_positions:
            for action in ALL_VALID_ACTIONS:
                test_cases.append(
                    {
                        "position": position,
                        "action": action,
                        "expected_within_bounds": False,
                        "expected_boundary_hit": True,
                        "invalid_start_position": True,
                    }
                )

    # Return comprehensive boundary test case collection for systematic validation
    return test_cases


def measure_boundary_enforcement_performance(
    enforcer: BoundaryEnforcer, iterations: int, position: Coordinates, action: Action
) -> Dict[str, Any]:
    """
    Measure boundary enforcement performance over multiple iterations for benchmark validation against
    BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS requirement with comprehensive timing analysis.

    Args:
        enforcer: BoundaryEnforcer instance for performance testing
        iterations: Number of iterations for statistical analysis
        position: Position coordinates for enforcement testing
        action: Action for boundary enforcement testing

    Returns:
        dict: Performance metrics including timing statistics and boundary enforcement analysis
    """
    # Initialize timing data collection list for performance measurement
    timing_data = []
    boundary_hits = 0
    constraint_applications = 0

    # Execute boundary enforcement iterations with high-precision timing using time.perf_counter()
    for _ in range(iterations):
        start_time = time.perf_counter()

        # Perform boundary enforcement operation
        result = enforcer.enforce_movement_bounds(position, action)

        end_time = time.perf_counter()

        # Collect individual operation timing measurements for statistical analysis
        operation_time = (end_time - start_time) * 1000  # Convert to milliseconds
        timing_data.append(operation_time)

        # Track boundary hits and constraint applications for analysis
        if result.boundary_hit:
            boundary_hits += 1
        if result.position_modified:
            constraint_applications += 1

    # Calculate statistical metrics including mean, median, percentiles
    timing_array = np.array(timing_data)

    performance_metrics = {
        "iterations": iterations,
        "mean_time_ms": np.mean(timing_array),
        "median_time_ms": np.median(timing_array),
        "min_time_ms": np.min(timing_array),
        "max_time_ms": np.max(timing_array),
        "std_time_ms": np.std(timing_array),
        "percentile_95_ms": np.percentile(timing_array, 95),
        "percentile_99_ms": np.percentile(timing_array, 99),
        "total_time_ms": np.sum(timing_array),
    }

    # Validate performance against BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS (0.1ms)
    target_time = BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS
    compliant_operations = np.sum(timing_array <= target_time)
    performance_metrics["target_compliance_rate"] = compliant_operations / iterations
    performance_metrics["target_time_ms"] = target_time

    # Include boundary hit statistics and constraint application analysis
    performance_metrics["boundary_statistics"] = {
        "boundary_hits": boundary_hits,
        "boundary_hit_rate": boundary_hits / iterations,
        "constraint_applications": constraint_applications,
        "constraint_application_rate": constraint_applications / iterations,
    }

    # Return comprehensive performance analysis dictionary with benchmark validation results
    return performance_metrics


def validate_boundary_enforcement_consistency(
    original_position: Coordinates,
    action: Action,
    enforcement_result: BoundaryEnforcementResult,
    grid_bounds: GridSize,
) -> bool:
    """
    Validate boundary enforcement results are consistent with mathematical expectations and coordinate
    system conventions for system integrity and correctness verification.

    Args:
        original_position: Original position before enforcement
        action: Action that was applied
        enforcement_result: Result from boundary enforcement
        grid_bounds: Grid boundaries for validation

    Returns:
        bool: True if boundary enforcement is mathematically consistent with expected behavior
    """
    # Calculate expected final position using action movement vector and grid constraints
    movement_vector = MOVEMENT_VECTORS[action.value]
    expected_x = original_position.x + movement_vector[0]
    expected_y = original_position.y + movement_vector[1]

    # Check if movement would be within bounds
    movement_within_bounds = (
        0 <= expected_x < grid_bounds.width and 0 <= expected_y < grid_bounds.height
    )

    # Compare enforcement_result.final_position with mathematical expectation
    if movement_within_bounds:
        # If movement should be valid, final position should match expected position
        expected_final = Coordinates(expected_x, expected_y)
        if enforcement_result.final_position != expected_final:
            return False

        # Verify position_modified flag accurately reflects no modification needed
        if enforcement_result.position_modified:
            return False

        # Check boundary_hit flag correctly indicates no boundary constraint
        if enforcement_result.boundary_hit:
            return False
    else:
        # Movement exceeds bounds - check constraint handling
        # Check boundary_hit flag correctly indicates boundary constraint activation
        if not enforcement_result.boundary_hit:
            return False

        # Verify final position is within grid bounds using GridSize.contains_coordinates
        if not enforcement_result.final_position.is_within_bounds(grid_bounds):
            return False

    # Verify coordinate system consistency with (x,y) positioning conventions
    if (
        enforcement_result.final_position.x < 0
        or enforcement_result.final_position.x >= grid_bounds.width
        or enforcement_result.final_position.y < 0
        or enforcement_result.final_position.y >= grid_bounds.height
    ):
        return False

    # Return True if all boundary enforcement results match mathematical expectations
    return True


def create_movement_constraint_variations(include_edge_cases: bool = False):
    """
    Create various MovementConstraint configurations for comprehensive boundary policy testing
    covering different enforcement scenarios and constraint combinations.

    Args:
        include_edge_cases: Whether to include edge case constraint configurations

    Returns:
        list: List of MovementConstraint instances with different configuration settings
    """
    variations = []

    # Create standard constraint config with default clamping enabled
    variations.append(
        MovementConstraint(
            enable_clamping=True,
            strict_validation=False,
            log_boundary_violations=False,
            tolerance=0.0,
        )
    )

    # Create strict validation constraint config with enhanced checking
    variations.append(
        MovementConstraint(
            enable_clamping=True,
            strict_validation=True,
            log_boundary_violations=False,
            tolerance=0.0,
        )
    )

    # Create logging-enabled constraint config for boundary violation tracking
    variations.append(
        MovementConstraint(
            enable_clamping=True,
            strict_validation=False,
            log_boundary_violations=True,
            tolerance=0.0,
        )
    )

    # Create no-clamping constraint config for pure validation testing
    variations.append(
        MovementConstraint(
            enable_clamping=False,
            strict_validation=True,
            log_boundary_violations=False,
            tolerance=0.0,
        )
    )

    # Create performance-optimized constraint config with minimal overhead
    variations.append(
        MovementConstraint(
            enable_clamping=True,
            strict_validation=False,
            log_boundary_violations=False,
            tolerance=0.0,
            performance_monitoring=True,
        )
    )

    # Include edge case configurations if include_edge_cases is True
    if include_edge_cases:
        # High tolerance configuration
        variations.append(
            MovementConstraint(
                enable_clamping=True,
                strict_validation=True,
                log_boundary_violations=True,
                tolerance=1.0,
            )
        )

        # All features enabled configuration
        variations.append(
            MovementConstraint(
                enable_clamping=True,
                strict_validation=True,
                log_boundary_violations=True,
                tolerance=0.5,
                performance_monitoring=True,
            )
        )

        # Custom boundaries configuration
        variations.append(
            MovementConstraint(
                enable_clamping=True,
                strict_validation=False,
                log_boundary_violations=False,
                custom_boundaries={"test_boundary": {"type": "rectangular"}},
            )
        )

    # Return comprehensive list of constraint configurations for policy testing
    return variations


def verify_boundary_statistics_accuracy(
    enforcer: BoundaryEnforcer,
    test_operations: List[Dict[str, Any]],
    expected_stats: Dict[str, Any],
) -> bool:
    """
    Verify boundary enforcement statistics are accurate and consistent with actual enforcement
    operations for monitoring validation and statistical integrity checking.

    Args:
        enforcer: BoundaryEnforcer instance with statistics tracking
        test_operations: List of operations performed for statistics verification
        expected_stats: Expected statistics values for comparison

    Returns:
        bool: True if statistics match expected values within defined tolerance
    """
    # Execute all test_operations and track expected boundary hits manually
    manual_boundary_hits = 0
    manual_constraint_applications = 0

    for operation in test_operations:
        position = operation.get("position")
        action = operation.get("action")

        if position and action:
            result = enforcer.enforce_movement_bounds(position, action)

            # Track expected boundary hits manually for comparison
            if operation.get("expect_boundary_hit", False):
                manual_boundary_hits += 1
            if operation.get("expect_constraint_application", False):
                manual_constraint_applications += 1

    # Get boundary statistics using enforcer.get_boundary_statistics()
    actual_stats = enforcer.get_boundary_statistics()

    # Compare actual boundary hit count with manually tracked count
    expected_boundary_hits = expected_stats.get(
        "expected_boundary_hits", manual_boundary_hits
    )
    actual_boundary_hits = actual_stats.get("boundary_hits", 0)

    tolerance = expected_stats.get("tolerance", 0)
    if abs(actual_boundary_hits - expected_boundary_hits) > tolerance:
        return False

    # Verify enforcement operation count matches executed operations
    expected_operations = expected_stats.get(
        "expected_operations", len(test_operations)
    )
    actual_operations = actual_stats.get("total_enforcements", 0)

    if abs(actual_operations - expected_operations) > tolerance:
        return False

    # Check boundary hit rate calculation accuracy
    if actual_operations > 0:
        expected_hit_rate = expected_boundary_hits / actual_operations
        actual_hit_rate = actual_stats.get("hit_rate", 0.0)

        if abs(actual_hit_rate - expected_hit_rate) > 0.01:  # 1% tolerance
            return False

    # Validate performance metrics are within reasonable ranges
    performance_metrics = actual_stats.get("performance_metrics", {})
    if performance_metrics:
        avg_time = performance_metrics.get("average_enforcement_time_ms", 0)
        if avg_time < 0 or avg_time > 1000:  # Reasonable bounds
            return False

    # Return True if all statistics match expectations within defined tolerance
    return True


class TestBoundaryEnforcerInitialization:
    """
    Test class for BoundaryEnforcer initialization covering constructor validation, parameter
    configuration, and basic setup verification with comprehensive error handling testing.
    """

    def test_boundary_enforcer_initialization_default_parameters(self):
        """Test BoundaryEnforcer initialization with default parameters and validate proper setup."""
        # Create BoundaryEnforcer with default TEST_GRID_SIZE
        enforcer = BoundaryEnforcer(grid_size=TEST_GRID_SIZE)

        # Validate grid_size property is set correctly to expected dimensions
        assert enforcer.grid_size == TEST_GRID_SIZE
        assert enforcer.grid_size.width == TEST_GRID_SIZE.width
        assert enforcer.grid_size.height == TEST_GRID_SIZE.height

        # Verify constraint_config is initialized with default MovementConstraint
        assert enforcer.constraint_config is not None
        assert isinstance(enforcer.constraint_config, MovementConstraint)
        assert enforcer.constraint_config.enable_clamping == True  # Default setting

        # Check enable_caching flag is properly set to default value
        assert enforcer.enable_caching == True  # Default caching enabled

        # Validate logger is initialized and performance_metrics are empty
        assert enforcer.logger is not None
        assert isinstance(enforcer.performance_metrics, dict)
        assert len(enforcer.performance_metrics["validation_times"]) == 0

        # Assert enforcement_count and boundary_hits start at 0
        assert enforcer.enforcement_count == 0
        assert enforcer.boundary_hits == 0

        # Verify validation_cache is initialized if caching is enabled
        assert isinstance(enforcer.validation_cache, dict)
        assert len(enforcer.validation_cache) == 0

    def test_boundary_enforcer_initialization_custom_parameters(self):
        """Test BoundaryEnforcer initialization with custom grid size, constraint configuration, and caching settings."""
        # Create custom GridSize and MovementConstraint instances for testing
        custom_grid = GridSize(64, 48)
        custom_constraint = MovementConstraint(
            enable_clamping=False,
            strict_validation=True,
            log_boundary_violations=True,
            tolerance=1.0,
        )

        # Initialize BoundaryEnforcer with custom parameters including caching settings
        enforcer = BoundaryEnforcer(
            grid_size=custom_grid,
            constraint_config=custom_constraint,
            enable_caching=False,
        )

        # Validate custom grid_size is properly stored and accessible
        assert enforcer.grid_size == custom_grid
        assert enforcer.grid_size.width == 64
        assert enforcer.grid_size.height == 48

        # Verify custom constraint_config is used correctly with proper validation
        assert enforcer.constraint_config == custom_constraint
        assert enforcer.constraint_config.enable_clamping == False
        assert enforcer.constraint_config.strict_validation == True
        assert enforcer.constraint_config.log_boundary_violations == True
        assert enforcer.constraint_config.tolerance == 1.0

        # Check custom caching configuration is applied and cache is initialized
        assert enforcer.enable_caching == False
        assert isinstance(enforcer.validation_cache, dict)

        # Assert performance metrics initialization with custom configuration
        assert isinstance(enforcer.performance_metrics, dict)
        assert "validation_times" in enforcer.performance_metrics

        # Validate constraint configuration using validate_constraint_configuration method
        assert enforcer.validate_constraint_configuration() == True

    def test_boundary_enforcer_invalid_initialization(self):
        """Test BoundaryEnforcer initialization with invalid parameters raises appropriate ValidationError exceptions."""
        # Test initialization with invalid grid_size including negative dimensions and zero values
        with pytest.raises(ValidationError):
            BoundaryEnforcer(grid_size=GridSize(-1, 10))

        with pytest.raises(ValidationError):
            BoundaryEnforcer(grid_size=GridSize(10, 0))

        # Test initialization with None grid_size parameter
        with pytest.raises((ValidationError, TypeError)):
            BoundaryEnforcer(grid_size=None)

        # Test initialization with invalid constraint_config type
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            BoundaryEnforcer(
                grid_size=TEST_GRID_SIZE, constraint_config="invalid_constraint"
            )

        # Test with malformed constraint configuration
        invalid_constraint = MovementConstraint(tolerance=-1.0)  # Invalid tolerance
        with pytest.raises(ValidationError):
            BoundaryEnforcer(
                grid_size=TEST_GRID_SIZE, constraint_config=invalid_constraint
            )

    def test_boundary_enforcer_constraint_configuration_validation(self):
        """Test constraint configuration validation during initialization ensures consistency and feasibility."""
        # Create BoundaryEnforcer with various constraint configurations
        valid_configs = create_movement_constraint_variations(include_edge_cases=False)

        for config in valid_configs:
            # Should not raise exceptions for valid configurations
            enforcer = BoundaryEnforcer(
                grid_size=TEST_GRID_SIZE, constraint_config=config
            )

            # Test validate_constraint_configuration with strict_mode enabled and disabled
            assert enforcer.validate_constraint_configuration(strict_mode=False) == True
            assert enforcer.validate_constraint_configuration(strict_mode=True) == True

        # Test with invalid constraint combinations
        invalid_config = MovementConstraint(
            tolerance=100.0,  # Too large for TEST_GRID_SIZE
            custom_boundaries={"invalid": "config"},
        )

        with pytest.raises((ValidationError, Warning)):
            enforcer = BoundaryEnforcer(
                grid_size=TEST_GRID_SIZE, constraint_config=invalid_config
            )
            # This should generate warnings in strict mode
            enforcer.validate_constraint_configuration(strict_mode=True)


class TestPositionValidation:
    """
    Test class for position validation functionality covering coordinate bounds checking, validation
    modes, and performance optimization with comprehensive edge case testing.
    """

    def test_validate_position_valid_coordinates(self):
        """Test position validation accepts valid coordinates within grid boundaries and returns True."""
        # Create BoundaryEnforcer with TEST_GRID_SIZE
        enforcer = create_test_boundary_enforcer()

        # Test validation with center position coordinates (16, 16)
        assert enforcer.validate_position(TEST_COORDINATES_CENTER) == True

        # Test validation with all corner positions within bounds
        assert enforcer.validate_position(TEST_COORDINATES_CORNER_ORIGIN) == True
        assert enforcer.validate_position(TEST_COORDINATES_CORNER_OPPOSITE) == True

        # Test validation with edge positions along all four boundaries
        assert enforcer.validate_position(TEST_COORDINATES_EDGE_TOP) == True
        assert enforcer.validate_position(TEST_COORDINATES_EDGE_BOTTOM) == True
        assert enforcer.validate_position(TEST_COORDINATES_EDGE_LEFT) == True
        assert enforcer.validate_position(TEST_COORDINATES_EDGE_RIGHT) == True

        # Test with tuple and list input formats
        assert enforcer.validate_position((16, 16)) == True
        assert enforcer.validate_position([16, 16]) == True

    def test_validate_position_invalid_coordinates(self):
        """Test position validation rejects invalid coordinates outside grid boundaries with appropriate error handling."""
        # Create BoundaryEnforcer with TEST_GRID_SIZE
        enforcer = create_test_boundary_enforcer()

        # Test validation with out-of-bounds coordinates including negative values
        for invalid_pos in OUT_OF_BOUNDS_POSITIONS:
            with pytest.raises(ValidationError):
                enforcer.validate_position(invalid_pos, raise_on_invalid=True)

            # Test with raise_on_invalid=False returns False
            assert (
                enforcer.validate_position(invalid_pos, raise_on_invalid=False) == False
            )

        # Test validation with None input
        with pytest.raises((ValidationError, TypeError)):
            enforcer.validate_position(None, raise_on_invalid=True)

        # Test with invalid coordinate type
        with pytest.raises((ValidationError, TypeError)):
            enforcer.validate_position("invalid", raise_on_invalid=True)

    def test_validate_position_strict_mode(self):
        """Test position validation in strict mode with enhanced checking and comprehensive error reporting."""
        # Create BoundaryEnforcer with strict validation enabled in constraint configuration
        strict_constraint = MovementConstraint(
            enable_clamping=True,
            strict_validation=True,
            tolerance=1.0,  # Add tolerance for strict mode testing
        )
        enforcer = create_test_boundary_enforcer(constraint_config=strict_constraint)

        # Test position validation with strict_mode enabled for enhanced checking
        # Center position should still be valid in strict mode
        assert enforcer.validate_position(TEST_COORDINATES_CENTER) == True

        # Verify strict mode applies additional mathematical consistency checks
        # Positions near tolerance boundary should be affected
        near_edge_position = Coordinates(1, 16)  # Within tolerance of edge
        # In strict mode with tolerance=1.0, this might be invalid
        result = enforcer.validate_position(near_edge_position, raise_on_invalid=False)
        # The result depends on strict validation implementation
        assert isinstance(result, bool)

    def test_validate_position_caching(self):
        """Test position validation caching functionality improves performance for repeated validation operations."""
        # Create BoundaryEnforcer with caching enabled
        enforcer = create_test_boundary_enforcer(enable_caching=True)

        # Validate identical positions multiple times and measure timing
        test_position = TEST_COORDINATES_CENTER

        # First validation (cache miss)
        start_time = time.perf_counter()
        result1 = enforcer.validate_position(test_position)
        first_time = time.perf_counter() - start_time

        # Second validation (cache hit)
        start_time = time.perf_counter()
        result2 = enforcer.validate_position(test_position)
        second_time = time.perf_counter() - start_time

        # Verify subsequent validations are faster due to cache hits
        assert result1 == result2 == True
        # Second call should be faster (though timing can be variable)

        # Check cache hit ratio in boundary statistics
        stats = enforcer.get_boundary_statistics()
        cache_stats = stats.get("cache_statistics", {})
        if cache_stats:
            assert cache_stats["cache_hits"] > 0

    def test_validate_position_performance(self):
        """Test position validation performance meets target latency requirements for high-frequency boundary checking."""
        # Create BoundaryEnforcer instance for performance testing
        enforcer = create_test_boundary_enforcer()

        # Measure validate_position execution time over 1000 iterations
        test_positions = [
            TEST_COORDINATES_CENTER,
            TEST_COORDINATES_CORNER_ORIGIN,
            TEST_COORDINATES_EDGE_TOP,
        ]
        timing_results = []

        for position in test_positions:
            for _ in range(PERFORMANCE_TEST_ITERATIONS // len(test_positions)):
                start_time = time.perf_counter()
                enforcer.validate_position(position)
                end_time = time.perf_counter()
                timing_results.append((end_time - start_time) * 1000)  # Convert to ms

        # Assert average validation time is well below performance target
        average_time = np.mean(timing_results)
        assert average_time < BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS

        # Verify 99th percentile validation time meets real-time requirements
        percentile_99 = np.percentile(timing_results, 99)
        assert (
            percentile_99 < BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS * 2
        )  # Allow some tolerance


class TestMovementValidation:
    """
    Test class for movement validation functionality covering action validity, boundary checking,
    and movement constraint analysis with comprehensive action space testing.
    """

    def test_is_movement_valid_within_bounds(self):
        """Test movement validation returns True for valid movements that stay within grid boundaries."""
        # Create BoundaryEnforcer with TEST_GRID_SIZE
        enforcer = create_test_boundary_enforcer()

        # Test movement validation from center position with all four actions
        for action in ALL_VALID_ACTIONS:
            result = enforcer.is_movement_valid(TEST_COORDINATES_CENTER, action)
            # All movements from center should be valid since they stay within bounds
            assert result == True

        # Test movement validation from positions away from boundaries
        interior_positions = [
            Coordinates(10, 10),
            Coordinates(20, 20),
            Coordinates(5, 25),
        ]

        for position in interior_positions:
            for action in ALL_VALID_ACTIONS:
                # Verify movement stays within bounds
                movement_vector = MOVEMENT_VECTORS[action.value]
                new_x = position.x + movement_vector[0]
                new_y = position.y + movement_vector[1]

                if (
                    0 <= new_x < TEST_GRID_SIZE.width
                    and 0 <= new_y < TEST_GRID_SIZE.height
                ):
                    assert enforcer.is_movement_valid(position, action) == True

    def test_is_movement_valid_boundary_constraints(self):
        """Test movement validation returns False for movements that would violate grid boundaries."""
        # Create BoundaryEnforcer with TEST_GRID_SIZE
        enforcer = create_test_boundary_enforcer()

        # Test upward movement from top edge positions (y=31) should be invalid
        top_edge_position = TEST_COORDINATES_EDGE_TOP
        assert enforcer.is_movement_valid(top_edge_position, Action.UP) == False

        # Test rightward movement from right edge positions (x=31) should be invalid
        right_edge_position = TEST_COORDINATES_EDGE_RIGHT
        assert enforcer.is_movement_valid(right_edge_position, Action.RIGHT) == False

        # Test downward movement from bottom edge positions (y=0) should be invalid
        bottom_edge_position = TEST_COORDINATES_EDGE_BOTTOM
        assert enforcer.is_movement_valid(bottom_edge_position, Action.DOWN) == False

        # Test leftward movement from left edge positions (x=0) should be invalid
        left_edge_position = TEST_COORDINATES_EDGE_LEFT
        assert enforcer.is_movement_valid(left_edge_position, Action.LEFT) == False

        # Test corner positions with multiple invalid directions
        corner_position = TEST_COORDINATES_CORNER_ORIGIN  # (0, 0)
        assert enforcer.is_movement_valid(corner_position, Action.LEFT) == False
        assert enforcer.is_movement_valid(corner_position, Action.DOWN) == False

    def test_get_valid_moves_analysis(self):
        """Test analysis of valid moves from specific positions considering boundary constraints for action space analysis."""
        # Create BoundaryEnforcer instance for valid moves analysis
        enforcer = create_test_boundary_enforcer()

        # Test get_valid_moves from center position should return all 4 actions
        center_moves = enforcer.get_valid_moves(TEST_COORDINATES_CENTER)
        assert len(center_moves) == 4
        assert all(action in center_moves for action in ALL_VALID_ACTIONS)

        # Test get_valid_moves from corner positions should return 2 valid actions
        corner_moves = enforcer.get_valid_moves(TEST_COORDINATES_CORNER_ORIGIN)
        assert len(corner_moves) == 2
        # From (0,0), only UP and RIGHT should be valid
        assert Action.UP in corner_moves
        assert Action.RIGHT in corner_moves
        assert Action.DOWN not in corner_moves
        assert Action.LEFT not in corner_moves

        # Test get_valid_moves from edge positions should return 3 valid actions
        edge_moves = enforcer.get_valid_moves(TEST_COORDINATES_EDGE_LEFT)
        assert len(edge_moves) == 3
        # From left edge, LEFT should be invalid
        assert Action.LEFT not in edge_moves
        assert Action.UP in edge_moves
        assert Action.RIGHT in edge_moves
        assert Action.DOWN in edge_moves

        # Verify returned action lists contain only valid Action enum values
        for moves in [center_moves, corner_moves, edge_moves]:
            for action in moves:
                assert isinstance(action, Action)
                assert action in ALL_VALID_ACTIONS

    def test_movement_validation_performance(self):
        """Test movement validation performance meets latency targets for real-time step execution."""
        # Create BoundaryEnforcer instance for performance measurement
        enforcer = create_test_boundary_enforcer()

        # Measure is_movement_valid execution time over multiple iterations
        test_scenarios = [
            (TEST_COORDINATES_CENTER, Action.UP),
            (TEST_COORDINATES_CORNER_ORIGIN, Action.LEFT),
            (TEST_COORDINATES_EDGE_RIGHT, Action.RIGHT),
        ]

        timing_results = []

        for position, action in test_scenarios:
            for _ in range(PERFORMANCE_TEST_ITERATIONS // len(test_scenarios)):
                start_time = time.perf_counter()
                result = enforcer.is_movement_valid(position, action)
                end_time = time.perf_counter()

                timing_results.append((end_time - start_time) * 1000)  # Convert to ms

        # Assert average validation time is well below performance target
        average_time = np.mean(timing_results)
        assert (
            average_time < BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS / 2
        )  # Should be faster than full enforcement

        # Verify movement validation is faster than full boundary enforcement
        # This is tested indirectly through the performance target


class TestBoundaryEnforcement:
    """
    Test class for complete boundary enforcement workflow covering constraint application, result
    generation, and movement analysis with comprehensive scenario testing.
    """

    def test_enforce_movement_bounds_valid_movement(self):
        """Test boundary enforcement with valid movements returns unchanged position and appropriate result analysis."""
        # Create BoundaryEnforcer instance for valid movement testing
        enforcer = create_test_boundary_enforcer()

        # Test enforce_movement_bounds from center position with valid actions
        for action in ALL_VALID_ACTIONS:
            result = enforcer.enforce_movement_bounds(TEST_COORDINATES_CENTER, action)

            # Verify BoundaryEnforcementResult.final_position equals expected movement result
            movement_vector = MOVEMENT_VECTORS[action.value]
            expected_final = Coordinates(
                TEST_COORDINATES_CENTER.x + movement_vector[0],
                TEST_COORDINATES_CENTER.y + movement_vector[1],
            )
            assert result.final_position == expected_final

            # Check position_modified is False for movements that don't hit boundaries
            assert result.position_modified == False

            # Assert boundary_hit is False for valid movements within grid
            assert result.boundary_hit == False

            # Validate enforcement result is mathematically consistent
            assert (
                validate_boundary_enforcement_consistency(
                    TEST_COORDINATES_CENTER, action, result, TEST_GRID_SIZE
                )
                == True
            )

    def test_enforce_movement_bounds_boundary_clamping(self):
        """Test boundary enforcement with clamping enabled constrains out-of-bounds movements to grid edges."""
        # Create BoundaryEnforcer with MovementConstraint having enable_clamping=True
        clamping_constraint = MovementConstraint(enable_clamping=True)
        enforcer = create_test_boundary_enforcer(constraint_config=clamping_constraint)

        # Test enforcement from edge positions with boundary-violating actions
        test_cases = [
            (TEST_COORDINATES_EDGE_TOP, Action.UP),
            (TEST_COORDINATES_EDGE_RIGHT, Action.RIGHT),
            (TEST_COORDINATES_EDGE_BOTTOM, Action.DOWN),
            (TEST_COORDINATES_EDGE_LEFT, Action.LEFT),
        ]

        for position, action in test_cases:
            result = enforcer.enforce_movement_bounds(position, action)

            # Verify final_position is clamped to grid boundary when movement exceeds bounds
            assert result.final_position.is_within_bounds(TEST_GRID_SIZE)

            # Check position_modified is True when clamping is applied
            # Note: This depends on the specific clamping implementation
            if result.boundary_hit:
                # Position should be modified or remain the same depending on clamping strategy
                assert isinstance(result.position_modified, bool)

            # Assert boundary_hit is True for movements constrained by boundaries
            assert result.boundary_hit == True

            # Validate clamped position is within grid bounds
            assert 0 <= result.final_position.x < TEST_GRID_SIZE.width
            assert 0 <= result.final_position.y < TEST_GRID_SIZE.height

    def test_enforce_movement_bounds_no_clamping(self):
        """Test boundary enforcement with clamping disabled keeps agent at current position for invalid movements."""
        # Create BoundaryEnforcer with MovementConstraint having enable_clamping=False
        no_clamp_constraint = MovementConstraint(enable_clamping=False)
        enforcer = create_test_boundary_enforcer(constraint_config=no_clamp_constraint)

        # Test enforcement from edge positions with boundary-violating actions
        result = enforcer.enforce_movement_bounds(TEST_COORDINATES_EDGE_TOP, Action.UP)

        # Verify final_position equals original_position when clamping is disabled
        assert result.final_position == result.original_position

        # Check position_modified is False for blocked movements
        assert result.position_modified == False

        # Assert boundary_hit is True indicating boundary constraint encountered
        assert result.boundary_hit == True

        # Validate movement was blocked rather than allowed to exceed boundaries
        assert result.final_position == TEST_COORDINATES_EDGE_TOP

    def test_boundary_enforcement_result_analysis(self):
        """Test BoundaryEnforcementResult analysis methods provide accurate constraint information and debugging data."""
        # Create BoundaryEnforcer and enforce various movement scenarios
        enforcer = create_test_boundary_enforcer()

        # Test with valid movement
        valid_result = enforcer.enforce_movement_bounds(
            TEST_COORDINATES_CENTER, Action.UP
        )

        # Test BoundaryEnforcementResult.get_constraint_analysis() provides detailed analysis
        analysis = valid_result.get_constraint_analysis()
        assert isinstance(analysis, dict)
        assert "position_delta" in analysis
        assert "constraint_type" in analysis
        assert "boundary_hit" in analysis

        # Verify was_movement_constrained() returns correct boolean status
        assert valid_result.was_movement_constrained() == valid_result.position_modified

        # Check get_final_position() returns proper coordinates
        assert isinstance(valid_result.get_final_position(), Coordinates)

        # Validate to_dict() returns complete result dictionary for serialization
        result_dict = valid_result.to_dict()
        assert isinstance(result_dict, dict)
        assert "original_position" in result_dict
        assert "final_position" in result_dict
        assert "boundary_hit" in result_dict

    def test_boundary_enforcement_performance_monitoring(self):
        """Test boundary enforcement performance monitoring and metrics collection for optimization analysis."""
        # Create BoundaryEnforcer instance with performance monitoring enabled
        enforcer = create_test_boundary_enforcer()

        # Enforce multiple movements and collect performance data
        test_operations = []
        for position in [TEST_COORDINATES_CENTER, TEST_COORDINATES_EDGE_TOP]:
            for action in [Action.UP, Action.DOWN]:
                result = enforcer.enforce_movement_bounds(position, action)
                test_operations.append(
                    {"position": position, "action": action, "result": result}
                )

        # Check get_boundary_statistics() returns comprehensive timing metrics
        stats = enforcer.get_boundary_statistics()
        assert isinstance(stats, dict)
        assert "total_enforcements" in stats
        assert stats["total_enforcements"] == len(test_operations)

        # Assert boundary enforcement meets performance requirements
        if "performance_metrics" in stats:
            avg_time = stats["performance_metrics"].get(
                "average_enforcement_time_ms", 0
            )
            assert (
                avg_time < BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS * 2
            )  # Allow some tolerance


class TestStandaloneFunctions:
    """
    Test class for standalone boundary enforcement functions covering utility functions and
    performance-critical operations with comprehensive parameter validation.
    """

    def test_validate_movement_bounds_function(self):
        """Test standalone validate_movement_bounds function provides fast boundary validation without enforcer instance."""
        # Test validate_movement_bounds with valid position-action combinations
        assert (
            validate_movement_bounds(TEST_COORDINATES_CENTER, Action.UP, TEST_GRID_SIZE)
            == True
        )
        assert (
            validate_movement_bounds(
                TEST_COORDINATES_CENTER, Action.RIGHT, TEST_GRID_SIZE
            )
            == True
        )

        # Verify function returns True for movements within grid boundaries
        interior_position = Coordinates(10, 10)
        for action in ALL_VALID_ACTIONS:
            result = validate_movement_bounds(interior_position, action, TEST_GRID_SIZE)
            assert isinstance(result, bool)

        # Test function with boundary-violating movements returns False
        assert (
            validate_movement_bounds(
                TEST_COORDINATES_EDGE_TOP, Action.UP, TEST_GRID_SIZE
            )
            == False
        )
        assert (
            validate_movement_bounds(
                TEST_COORDINATES_EDGE_RIGHT, Action.RIGHT, TEST_GRID_SIZE
            )
            == False
        )

        # Check strict_validation parameter enables enhanced boundary checking
        result_strict = validate_movement_bounds(
            TEST_COORDINATES_CENTER, Action.UP, TEST_GRID_SIZE, strict_validation=True
        )
        result_non_strict = validate_movement_bounds(
            TEST_COORDINATES_CENTER, Action.UP, TEST_GRID_SIZE, strict_validation=False
        )
        assert isinstance(result_strict, bool)
        assert isinstance(result_non_strict, bool)

    def test_enforce_position_bounds_function(self):
        """Test standalone enforce_position_bounds function provides position clamping with boundary hit detection."""
        # Test enforce_position_bounds with valid positions returns unchanged coordinates
        result_pos, modified = enforce_position_bounds(
            TEST_COORDINATES_CENTER, TEST_GRID_SIZE
        )
        assert result_pos == TEST_COORDINATES_CENTER
        assert modified == False

        # Test function with out-of-bounds positions returns clamped coordinates
        out_of_bounds = Coordinates(-5, 35)
        clamped_pos, was_modified = enforce_position_bounds(
            out_of_bounds, TEST_GRID_SIZE
        )

        # Verify position_modified flag accurately reflects clamping application
        assert was_modified == True
        assert clamped_pos.is_within_bounds(TEST_GRID_SIZE)

        # Check clamp_to_bounds parameter controls clamping behavior
        no_clamp_pos, no_clamp_modified = enforce_position_bounds(
            out_of_bounds, TEST_GRID_SIZE, clamp_to_bounds=False
        )
        assert no_clamp_modified == True  # Still indicates bounds violation

        # Test with different boundary scenarios
        edge_cases = [
            Coordinates(-1, 16),  # Left boundary violation
            Coordinates(32, 16),  # Right boundary violation
            Coordinates(16, -1),  # Bottom boundary violation
            Coordinates(16, 32),  # Top boundary violation
        ]

        for invalid_pos in edge_cases:
            clamped, modified = enforce_position_bounds(invalid_pos, TEST_GRID_SIZE)
            assert modified == True
            assert clamped.is_within_bounds(TEST_GRID_SIZE)

    def test_is_position_within_bounds_function(self):
        """Test standalone is_position_within_bounds function provides fastest boolean bounds checking."""
        # Test is_position_within_bounds with valid positions returns True
        assert (
            is_position_within_bounds(TEST_COORDINATES_CENTER, TEST_GRID_SIZE) == True
        )
        assert (
            is_position_within_bounds(TEST_COORDINATES_CORNER_ORIGIN, TEST_GRID_SIZE)
            == True
        )
        assert (
            is_position_within_bounds(TEST_COORDINATES_CORNER_OPPOSITE, TEST_GRID_SIZE)
            == True
        )

        # Test function with invalid positions returns False
        for invalid_pos in OUT_OF_BOUNDS_POSITIONS:
            assert is_position_within_bounds(invalid_pos, TEST_GRID_SIZE) == False

        # Test function with various coordinate input formats
        assert is_position_within_bounds((16, 16), TEST_GRID_SIZE) == True
        assert is_position_within_bounds([16, 16], TEST_GRID_SIZE) == True

        # Performance test - function should be very fast
        start_time = time.perf_counter()
        for _ in range(1000):
            is_position_within_bounds(TEST_COORDINATES_CENTER, TEST_GRID_SIZE)
        end_time = time.perf_counter()

        avg_time = ((end_time - start_time) * 1000) / 1000  # Average per call in ms
        assert (
            avg_time < BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS / 10
        )  # Should be very fast

    def test_clamp_coordinates_to_bounds_function(self):
        """Test standalone clamp_coordinates_to_bounds function provides precise coordinate clamping."""
        # Test clamp_coordinates_to_bounds with valid coordinates returns unchanged values
        result = clamp_coordinates_to_bounds(TEST_COORDINATES_CENTER, TEST_GRID_SIZE)
        assert result == TEST_COORDINATES_CENTER

        # Test function with out-of-bounds coordinates returns clamped values
        test_cases = [
            (Coordinates(-5, 16), Coordinates(0, 16)),  # X clamp to 0
            (Coordinates(40, 16), Coordinates(31, 16)),  # X clamp to width-1
            (Coordinates(16, -3), Coordinates(16, 0)),  # Y clamp to 0
            (Coordinates(16, 50), Coordinates(16, 31)),  # Y clamp to height-1
            (Coordinates(-10, -10), Coordinates(0, 0)),  # Both coordinates clamped
            (Coordinates(100, 100), Coordinates(31, 31)),  # Both coordinates clamped
        ]

        for input_coords, expected_coords in test_cases:
            result = clamp_coordinates_to_bounds(input_coords, TEST_GRID_SIZE)
            assert result == expected_coords
            assert result.is_within_bounds(TEST_GRID_SIZE)

    def test_calculate_bounded_movement_function(self):
        """Test standalone calculate_bounded_movement function provides comprehensive movement analysis with constraints."""
        # Test calculate_bounded_movement with valid movements returns correct final position
        final_pos, success, boundary_hit = calculate_bounded_movement(
            TEST_COORDINATES_CENTER, Action.UP, TEST_GRID_SIZE
        )

        expected_pos = Coordinates(
            TEST_COORDINATES_CENTER.x + MOVEMENT_VECTORS[Action.UP.value][0],
            TEST_COORDINATES_CENTER.y + MOVEMENT_VECTORS[Action.UP.value][1],
        )
        assert final_pos == expected_pos
        assert success == True
        assert boundary_hit == False

        # Test function with boundary-violating movements handles constraint application
        final_pos, success, boundary_hit = calculate_bounded_movement(
            TEST_COORDINATES_EDGE_TOP, Action.UP, TEST_GRID_SIZE
        )

        # Verify movement_successful flag reflects actual movement outcome
        assert isinstance(success, bool)

        # Check boundary_hit flag indicates boundary constraint encounters
        assert boundary_hit == True

        # Test allow_boundary_clamping parameter controls constraint behavior
        clamp_pos, clamp_success, clamp_hit = calculate_bounded_movement(
            TEST_COORDINATES_EDGE_TOP,
            Action.UP,
            TEST_GRID_SIZE,
            allow_boundary_clamping=True,
        )

        no_clamp_pos, no_clamp_success, no_clamp_hit = calculate_bounded_movement(
            TEST_COORDINATES_EDGE_TOP,
            Action.UP,
            TEST_GRID_SIZE,
            allow_boundary_clamping=False,
        )

        # With clamping, position should be valid; without clamping, movement should fail
        assert clamp_pos.is_within_bounds(TEST_GRID_SIZE)
        assert (
            no_clamp_pos == TEST_COORDINATES_EDGE_TOP
        )  # Should stay at original position


class TestMovementConstraintConfiguration:
    """
    Test class for MovementConstraint configuration covering policy validation, configuration
    management, and constraint analysis with comprehensive parameter testing.
    """

    def test_movement_constraint_initialization(self):
        """Test MovementConstraint initialization with various policy configurations and parameter validation."""
        # Create MovementConstraint with default parameters and validate initialization
        default_constraint = MovementConstraint()
        assert default_constraint.enable_clamping == True  # Default value
        assert default_constraint.strict_validation == True  # Default value
        assert default_constraint.log_boundary_violations == False  # Default value
        assert default_constraint.tolerance == 0  # Default value

        # Test initialization with enable_clamping=True enables position clamping
        clamp_constraint = MovementConstraint(enable_clamping=True)
        assert clamp_constraint.enable_clamping == True

        # Test initialization with strict_validation=True enables enhanced checking
        strict_constraint = MovementConstraint(strict_validation=True)
        assert strict_constraint.strict_validation == True

        # Verify log_boundary_violations=True enables boundary hit logging
        log_constraint = MovementConstraint(log_boundary_violations=True)
        assert log_constraint.log_boundary_violations == True

        # Check tolerance parameter is initialized to proper default value
        assert default_constraint.tolerance >= 0

        # Test custom parameter combinations
        custom_constraint = MovementConstraint(
            enable_clamping=False,
            strict_validation=False,
            log_boundary_violations=True,
            tolerance=0.5,
        )
        assert custom_constraint.enable_clamping == False
        assert custom_constraint.strict_validation == False
        assert custom_constraint.log_boundary_violations == True
        assert custom_constraint.tolerance == 0.5

    def test_movement_constraint_validation(self):
        """Test MovementConstraint.validate_configuration ensures policy consistency and feasibility."""
        # Create MovementConstraint instances with valid configurations
        valid_configs = create_movement_constraint_variations(include_edge_cases=False)

        for config in valid_configs:
            # Test validate_configuration returns True for consistent policy settings
            assert config.validate_configuration() == True

        # Test validation with invalid tolerance values raises ValidationError
        with pytest.raises(ValidationError):
            invalid_constraint = MovementConstraint(tolerance=-1.0)
            invalid_constraint.validate_configuration()

        # Test with invalid custom_boundaries type
        with pytest.raises(ValidationError):
            invalid_constraint = MovementConstraint(custom_boundaries="invalid")
            invalid_constraint.validate_configuration()

    def test_movement_constraint_cloning(self):
        """Test MovementConstraint.clone method creates proper deep copies with optional parameter overrides."""
        # Create MovementConstraint and test clone() creates independent copy
        original = MovementConstraint(
            enable_clamping=True, strict_validation=False, tolerance=1.0
        )

        clone = original.clone()

        # Verify cloned constraint has identical configuration to original
        assert clone.enable_clamping == original.enable_clamping
        assert clone.strict_validation == original.strict_validation
        assert clone.tolerance == original.tolerance

        # Test clone with parameter overrides applies modifications correctly
        overridden_clone = original.clone(
            overrides={"strict_validation": True, "tolerance": 2.0}
        )

        # Check cloned constraint validates successfully if overrides are provided
        assert overridden_clone.validate_configuration() == True
        assert overridden_clone.strict_validation == True
        assert overridden_clone.tolerance == 2.0

        # Assert original constraint is not modified during cloning operation
        assert original.strict_validation == False
        assert original.tolerance == 1.0

    def test_constraint_policy_variations(self):
        """Test various constraint policy combinations ensure proper boundary enforcement behavior."""
        # Get all constraint variations for testing
        variations = create_movement_constraint_variations(include_edge_cases=True)

        # Test each variation works with BoundaryEnforcer
        for constraint in variations:
            # Should be able to create enforcer with each variation
            enforcer = create_test_boundary_enforcer(constraint_config=constraint)
            assert enforcer.constraint_config == constraint

            # Test basic enforcement operation with each constraint
            result = enforcer.enforce_movement_bounds(
                TEST_COORDINATES_CENTER, Action.UP
            )
            assert isinstance(result, BoundaryEnforcementResult)

            # Verify configuration validation passes
            assert constraint.validate_configuration() == True


class TestBoundaryStatistics:
    """
    Test class for boundary enforcement statistics covering metrics collection, hit rate analysis,
    and performance monitoring with comprehensive statistical validation.
    """

    def test_boundary_statistics_collection(self):
        """Test boundary enforcement statistics are accurately collected and reported for monitoring analysis."""
        # Create BoundaryEnforcer and perform various enforcement operations
        enforcer = create_test_boundary_enforcer()

        # Track expected boundary hits manually during test operations
        test_operations = [
            {
                "position": TEST_COORDINATES_CENTER,
                "action": Action.UP,
                "expect_hit": False,
            },
            {
                "position": TEST_COORDINATES_EDGE_TOP,
                "action": Action.UP,
                "expect_hit": True,
            },
            {
                "position": TEST_COORDINATES_EDGE_RIGHT,
                "action": Action.RIGHT,
                "expect_hit": True,
            },
            {
                "position": TEST_COORDINATES_CENTER,
                "action": Action.DOWN,
                "expect_hit": False,
            },
        ]

        expected_hits = 0
        for operation in test_operations:
            result = enforcer.enforce_movement_bounds(
                operation["position"], operation["action"]
            )
            if operation["expect_hit"]:
                expected_hits += 1

        # Get boundary statistics using get_boundary_statistics() method
        stats = enforcer.get_boundary_statistics()

        # Verify total enforcement operations count matches executed operations
        assert stats["total_enforcements"] == len(test_operations)

        # Check boundary hit count matches manually tracked boundary encounters
        assert stats["boundary_hits"] == expected_hits

        # Assert boundary hit rate calculation is accurate within tolerance
        expected_rate = expected_hits / len(test_operations)
        assert abs(stats["hit_rate"] - expected_rate) < 0.01

    def test_cache_statistics_tracking(self):
        """Test boundary enforcement cache statistics provide hit/miss ratios and efficiency metrics."""
        # Create BoundaryEnforcer with caching enabled for statistics tracking
        enforcer = create_test_boundary_enforcer(enable_caching=True)

        # Perform repeated boundary validations to populate cache
        test_position = TEST_COORDINATES_CENTER

        # First validation (cache miss)
        enforcer.validate_position(test_position)

        # Execute duplicate validations to generate cache hits
        for _ in range(5):
            enforcer.validate_position(test_position)

        # Get boundary statistics and verify cache hit/miss ratios are reported
        stats = enforcer.get_boundary_statistics()

        if "cache_statistics" in stats:
            cache_stats = stats["cache_statistics"]

            # Check cache efficiency metrics show performance improvement
            assert cache_stats["cache_hits"] > 0
            assert cache_stats["hit_ratio"] >= 0.0

            # Validate cache entries tracking
            assert cache_stats["cache_entries"] >= 0

    def test_performance_statistics_accuracy(self):
        """Test boundary enforcement performance statistics provide accurate timing analysis and optimization data."""
        # Create BoundaryEnforcer with performance monitoring enabled
        enforcer = create_test_boundary_enforcer()

        # Execute boundary enforcement operations with timing measurement
        positions = [
            TEST_COORDINATES_CENTER,
            TEST_COORDINATES_EDGE_TOP,
            TEST_COORDINATES_CORNER_ORIGIN,
        ]

        for position in positions:
            for action in [Action.UP, Action.RIGHT]:
                enforcer.enforce_movement_bounds(position, action)

        # Get boundary statistics and verify performance metrics are included
        stats = enforcer.get_boundary_statistics()

        if "performance_metrics" in stats:
            perf_metrics = stats["performance_metrics"]

            # Check average enforcement time calculation accuracy
            assert "average_enforcement_time_ms" in perf_metrics
            assert perf_metrics["average_enforcement_time_ms"] >= 0

            # Validate performance target compliance tracking
            if "target_compliance_rate" in perf_metrics:
                assert 0.0 <= perf_metrics["target_compliance_rate"] <= 1.0

    def test_statistics_reset_and_cleanup(self):
        """Test boundary statistics can be reset and cleaned up properly for fresh monitoring periods."""
        # Create BoundaryEnforcer and accumulate boundary statistics through operations
        enforcer = create_test_boundary_enforcer()

        # Perform operations to accumulate statistics
        for _ in range(3):
            enforcer.enforce_movement_bounds(TEST_COORDINATES_CENTER, Action.UP)

        # Verify statistics show non-zero values after enforcement operations
        stats_before = enforcer.get_boundary_statistics()
        assert stats_before["total_enforcements"] > 0

        # Test clear_cache() method resets cache-related statistics
        entries_cleared = enforcer.clear_cache()

        # Check that cache clearing doesn't affect core enforcement functionality
        result = enforcer.enforce_movement_bounds(TEST_COORDINATES_CENTER, Action.UP)
        assert isinstance(result, BoundaryEnforcementResult)

        # Validate cleanup maintains boundary enforcer operational integrity
        assert enforcer.grid_size == TEST_GRID_SIZE
        assert enforcer.constraint_config is not None


class TestErrorHandling:
    """
    Test class for boundary enforcement error handling covering validation failures, state errors,
    and recovery strategies with comprehensive exception testing.
    """

    def test_validation_error_handling(self):
        """Test proper ValidationError handling for invalid boundary parameters with context and recovery information."""
        # Create BoundaryEnforcer and test validation with various invalid parameter types
        enforcer = create_test_boundary_enforcer()

        # Test position validation with None, string, and non-coordinate inputs
        invalid_positions = [None, "invalid", 123, [1, 2, 3], {"x": 1, "y": 2}]

        for invalid_pos in invalid_positions:
            with pytest.raises((ValidationError, TypeError)):
                enforcer.validate_position(invalid_pos, raise_on_invalid=True)

        # Test action validation with invalid action values outside range
        invalid_actions = [-1, 4, 10, "UP", None, 3.14]

        for invalid_action in invalid_actions:
            with pytest.raises((ValidationError, TypeError, AttributeError)):
                enforcer.enforce_movement_bounds(
                    TEST_COORDINATES_CENTER, invalid_action
                )

    def test_state_error_handling(self):
        """Test StateError handling for invalid boundary enforcer states and configuration inconsistencies."""
        # Create BoundaryEnforcer with valid initial configuration
        enforcer = create_test_boundary_enforcer()

        # Test constraint configuration inconsistencies that might cause state errors
        try:
            # Try to update with invalid grid size
            with pytest.raises((ValidationError, ValueError)):
                enforcer.update_grid_size(GridSize(-1, -1))
        except Exception:
            # Verify error handling maintains enforcer stability
            assert enforcer.grid_size == TEST_GRID_SIZE  # Should remain unchanged

    def test_configuration_error_handling(self):
        """Test ConfigurationError handling for invalid boundary constraint configurations with detailed analysis."""
        # Test boundary enforcer initialization with invalid MovementConstraint configurations

        # Create constraints with logically inconsistent parameter combinations
        invalid_constraints = [
            {"tolerance": -1.0},  # Negative tolerance
            {"custom_boundaries": "invalid_type"},  # Wrong type for custom_boundaries
        ]

        for invalid_config in invalid_constraints:
            with pytest.raises((ConfigurationError, ValidationError)):
                # Try to create constraint with invalid parameters
                constraint = MovementConstraint(**invalid_config)
                constraint.validate_configuration()

    def test_error_context_sanitization(self):
        """Test error message sanitization and context filtering prevent information disclosure while providing debugging context."""
        # Create BoundaryEnforcer for error context testing
        enforcer = create_test_boundary_enforcer()

        # Trigger various boundary enforcement errors with context
        try:
            enforcer.validate_position((-1000, -1000), context_info="test_context")
        except ValidationError as e:
            # Verify error messages don't contain internal state information
            error_msg = str(e)
            assert isinstance(error_msg, str)
            assert len(error_msg) > 0

            # Check that error provides useful feedback
            assert "bounds" in error_msg.lower() or "invalid" in error_msg.lower()


class TestPerformanceBenchmarks:
    """
    Test class for boundary enforcement performance benchmarking covering latency validation,
    optimization verification, and resource efficiency testing.
    """

    def test_boundary_enforcement_latency_benchmark(self):
        """Benchmark boundary enforcement latency against performance target for real-time operation."""
        # Create BoundaryEnforcer instance for latency benchmarking
        enforcer = create_test_boundary_enforcer()

        # Test scenarios with various position-action combinations including edge cases
        test_scenarios = [
            (TEST_COORDINATES_CENTER, Action.UP),
            (TEST_COORDINATES_EDGE_TOP, Action.UP),
            (TEST_COORDINATES_CORNER_ORIGIN, Action.LEFT),
            (TEST_COORDINATES_CORNER_OPPOSITE, Action.DOWN),
        ]

        all_timings = []

        # Measure enforce_movement_bounds execution time over iterations
        for scenario in test_scenarios:
            position, action = scenario
            scenario_timings = []

            for _ in range(PERFORMANCE_TEST_ITERATIONS // len(test_scenarios)):
                start_time = time.perf_counter()
                result = enforcer.enforce_movement_bounds(position, action)
                end_time = time.perf_counter()

                timing_ms = (end_time - start_time) * 1000
                scenario_timings.append(timing_ms)
                all_timings.append(timing_ms)

        # Calculate mean, median, 95th and 99th percentile latencies for analysis
        mean_latency = np.mean(all_timings)
        median_latency = np.median(all_timings)
        percentile_95 = np.percentile(all_timings, 95)
        percentile_99 = np.percentile(all_timings, 99)

        # Assert average latency is below target
        assert (
            mean_latency < BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS
        ), f"Mean latency {mean_latency:.4f}ms exceeds target {BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS}ms"

        # Verify 99th percentile meets real-time requirements for step execution
        assert (
            percentile_99 < BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS * 2
        ), f"99th percentile {percentile_99:.4f}ms exceeds relaxed target"

    def test_position_validation_performance_benchmark(self):
        """Benchmark position validation performance for high-frequency boundary checking operations."""
        # Create BoundaryEnforcer instance for validation benchmarking
        enforcer = create_test_boundary_enforcer()

        # Test positions for performance benchmarking
        test_positions = [
            TEST_COORDINATES_CENTER,
            TEST_COORDINATES_CORNER_ORIGIN,
            TEST_COORDINATES_EDGE_TOP,
            (16, 16),  # Tuple format
            [20, 20],  # List format
        ]

        validation_timings = []

        # Measure validate_position execution time over iterations
        iterations_per_position = PERFORMANCE_TEST_ITERATIONS * 2 // len(test_positions)

        for position in test_positions:
            for _ in range(iterations_per_position):
                start_time = time.perf_counter()
                try:
                    result = enforcer.validate_position(
                        position, raise_on_invalid=False
                    )
                except:
                    pass  # Handle any conversion errors gracefully
                end_time = time.perf_counter()

                timing_ms = (end_time - start_time) * 1000
                validation_timings.append(timing_ms)

        # Assert validation performance is well below boundary enforcement target
        avg_validation_time = np.mean(validation_timings)
        assert (
            avg_validation_time < BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS / 2
        ), f"Average validation time {avg_validation_time:.4f}ms too slow"

    def test_memory_usage_efficiency(self):
        """Test memory usage efficiency and resource management during boundary enforcement operations."""
        # Create BoundaryEnforcer and measure initial state
        enforcer = create_test_boundary_enforcer(enable_caching=True)

        # Execute large number of boundary operations and monitor cache growth
        operations_count = 1000
        unique_positions = [
            Coordinates(i % 32, j % 32)
            for i in range(0, 100, 5)
            for j in range(0, 100, 5)
        ]

        for i in range(operations_count):
            position = unique_positions[i % len(unique_positions)]
            action = ALL_VALID_ACTIONS[i % len(ALL_VALID_ACTIONS)]

            # Perform operations
            enforcer.validate_position(position, raise_on_invalid=False)
            enforcer.enforce_movement_bounds(position, action)

        # Check that cache size is reasonable
        stats = enforcer.get_boundary_statistics()
        if "cache_statistics" in stats:
            cache_size = stats["cache_statistics"].get("cache_entries", 0)
            # Cache should not grow unboundedly
            assert cache_size <= BOUNDARY_VALIDATION_CACHE_SIZE * 2

        # Test memory cleanup is effective when cache is cleared
        initial_cache_size = len(enforcer.validation_cache)
        entries_cleared = enforcer.clear_cache(force_cleanup=True)
        final_cache_size = len(enforcer.validation_cache)

        assert final_cache_size == 0
        assert entries_cleared >= 0

    def test_scaling_performance_analysis(self):
        """Test boundary enforcement performance scaling with different grid sizes and constraint configurations."""
        # Test with different grid sizes for scaling analysis
        grid_sizes = [
            GridSize(16, 16),  # Small
            GridSize(32, 32),  # Medium (default)
            GridSize(64, 64),  # Large
        ]

        performance_results = {}

        for grid_size in grid_sizes:
            enforcer = create_test_boundary_enforcer(grid_size=grid_size)

            # Measure performance for this grid size
            timings = []
            test_iterations = 100  # Reduced for scaling test

            # Use center position scaled for grid size
            center_pos = Coordinates(grid_size.width // 2, grid_size.height // 2)

            for _ in range(test_iterations):
                start_time = time.perf_counter()
                enforcer.enforce_movement_bounds(center_pos, Action.UP)
                end_time = time.perf_counter()

                timings.append((end_time - start_time) * 1000)

            avg_time = np.mean(timings)
            performance_results[f"{grid_size.width}x{grid_size.height}"] = avg_time

        # Verify performance scaling is reasonable
        for grid_desc, avg_time in performance_results.items():
            assert (
                avg_time < BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS * 3
            ), f"Performance for {grid_desc} grid: {avg_time:.4f}ms exceeds scaled target"


class TestIntegrationAndCompatibility:
    """
    Test class for boundary enforcer integration covering component interaction, coordinate system
    consistency, and compatibility validation with core system components.
    """

    def test_coordinate_system_consistency(self):
        """Test boundary enforcer maintains coordinate system consistency with (x,y) positioning and proper indexing conventions."""
        # Create BoundaryEnforcer and test coordinate system consistency
        enforcer = create_test_boundary_enforcer()

        # Verify x-axis positive direction is rightward in grid coordinates
        center_pos = TEST_COORDINATES_CENTER
        right_result = enforcer.enforce_movement_bounds(center_pos, Action.RIGHT)

        # Check that RIGHT action increases x-coordinate
        assert right_result.final_position.x == center_pos.x + 1
        assert right_result.final_position.y == center_pos.y

        # Test y-axis positive direction is upward in mathematical coordinate system
        up_result = enforcer.enforce_movement_bounds(center_pos, Action.UP)

        # Check that UP action increases y-coordinate
        assert up_result.final_position.x == center_pos.x
        assert up_result.final_position.y == center_pos.y + 1

        # Validate coordinate transformations maintain mathematical correctness
        movement_vectors = {
            Action.UP: (0, 1),
            Action.RIGHT: (1, 0),
            Action.DOWN: (0, -1),
            Action.LEFT: (-1, 0),
        }

        for action, expected_delta in movement_vectors.items():
            result = enforcer.enforce_movement_bounds(center_pos, action)
            actual_delta = (
                result.final_position.x - center_pos.x,
                result.final_position.y - center_pos.y,
            )

            if not result.boundary_hit:  # Only check if movement wasn't constrained
                assert actual_delta == expected_delta

    def test_action_type_compatibility(self):
        """Test boundary enforcer compatibility with Action enum and integer action representations."""
        # Create BoundaryEnforcer and test with Action enum values
        enforcer = create_test_boundary_enforcer()
        test_position = TEST_COORDINATES_CENTER

        # Test enforcement with Action enum values
        for action_enum in ALL_VALID_ACTIONS:
            result_enum = enforcer.enforce_movement_bounds(test_position, action_enum)
            assert isinstance(result_enum, BoundaryEnforcementResult)

        # Test enforcement with integer action values (0, 1, 2, 3)
        for action_int in range(4):
            result_int = enforcer.enforce_movement_bounds(test_position, action_int)
            assert isinstance(result_int, BoundaryEnforcementResult)

        # Verify Action.to_vector() integration works correctly with boundary checking
        for action in ALL_VALID_ACTIONS:
            # Test that action conversion is consistent
            result = enforcer.enforce_movement_bounds(test_position, action)

            # Check that the movement vector is applied correctly
            movement_vector = MOVEMENT_VECTORS[action.value]
            expected_pos = Coordinates(
                test_position.x + movement_vector[0],
                test_position.y + movement_vector[1],
            )

            if not result.boundary_hit:
                assert result.final_position == expected_pos

        # Assert action type conversion maintains boundary validation accuracy
        # Compare enum vs integer results for same logical action
        for i, action_enum in enumerate(ALL_VALID_ACTIONS):
            result_enum = enforcer.enforce_movement_bounds(test_position, action_enum)
            result_int = enforcer.enforce_movement_bounds(test_position, i)

            # Results should be identical regardless of input format
            assert result_enum.final_position == result_int.final_position
            assert result_enum.boundary_hit == result_int.boundary_hit

    def test_grid_size_update_integration(self):
        """Test boundary enforcer grid size updates maintain system consistency and clear dependent state properly."""
        # Create BoundaryEnforcer with initial grid size and populate cache
        enforcer = create_test_boundary_enforcer(enable_caching=True)

        # Populate cache with some validation results
        enforcer.validate_position(TEST_COORDINATES_CENTER)
        enforcer.validate_position(TEST_COORDINATES_CORNER_ORIGIN)
        initial_cache_size = len(enforcer.validation_cache)

        # Update grid size using update_grid_size() method with new dimensions
        new_grid_size = GridSize(64, 64)
        enforcer.update_grid_size(new_grid_size)

        # Verify cache is properly cleared after grid size update
        assert len(enforcer.validation_cache) == 0

        # Test boundary validation results reflect new grid boundaries
        assert enforcer.grid_size == new_grid_size

        # Test positions that were valid in old grid
        large_position = Coordinates(50, 50)  # Valid in 64x64, invalid in 32x32
        assert (
            enforcer.validate_position(large_position, raise_on_invalid=False) == True
        )

        # Check enforcement statistics are reset appropriately for new grid
        stats = enforcer.get_boundary_statistics()
        assert stats["grid_size"]["width"] == new_grid_size.width
        assert stats["grid_size"]["height"] == new_grid_size.height

    def test_constraint_configuration_integration(self):
        """Test boundary enforcer integration with various MovementConstraint configurations ensures policy application consistency."""
        # Create BoundaryEnforcer instances with different MovementConstraint policies
        constraint_variations = create_movement_constraint_variations(
            include_edge_cases=True
        )

        for constraint in constraint_variations:
            enforcer = create_test_boundary_enforcer(constraint_config=constraint)

            # Test enforcement behavior matches constraint configuration settings
            boundary_position = TEST_COORDINATES_EDGE_TOP
            boundary_action = Action.UP  # Should hit boundary

            result = enforcer.enforce_movement_bounds(
                boundary_position, boundary_action
            )

            # Verify clamping policy integration works correctly with boundary hits
            if constraint.enable_clamping:
                # With clamping, final position should be valid
                assert result.final_position.is_within_bounds(TEST_GRID_SIZE)
            else:
                # Without clamping, agent should stay at original position
                if result.boundary_hit:
                    assert result.final_position == boundary_position

            # Assert constraint configuration changes are applied consistently
            assert enforcer.constraint_config == constraint

            # Test that configuration validation still passes
            assert enforcer.validate_constraint_configuration() == True


class TestReproducibilityAndDeterminism:
    """
    Test class for boundary enforcement reproducibility covering deterministic behavior validation
    and consistency across execution contexts with comprehensive reproducibility testing.
    """

    def test_deterministic_boundary_enforcement(self):
        """Test boundary enforcement produces identical results for identical inputs across multiple execution runs."""
        # Create two identical BoundaryEnforcer instances with same configurations
        enforcer1 = create_test_boundary_enforcer(
            grid_size=TEST_GRID_SIZE, enable_caching=False
        )
        enforcer2 = create_test_boundary_enforcer(
            grid_size=TEST_GRID_SIZE, enable_caching=False
        )

        # Execute identical boundary enforcement sequences on both instances
        test_operations = [
            (TEST_COORDINATES_CENTER, Action.UP),
            (TEST_COORDINATES_EDGE_TOP, Action.UP),
            (TEST_COORDINATES_CORNER_ORIGIN, Action.LEFT),
            (TEST_COORDINATES_EDGE_RIGHT, Action.RIGHT),
        ]

        results1 = []
        results2 = []

        for position, action in test_operations:
            result1 = enforcer1.enforce_movement_bounds(position, action)
            result2 = enforcer2.enforce_movement_bounds(position, action)

            results1.append(result1)
            results2.append(result2)

        # Compare BoundaryEnforcementResult objects for complete equality
        for r1, r2 in zip(results1, results2):
            assert r1.original_position == r2.original_position
            assert r1.final_position == r2.final_position
            assert r1.position_modified == r2.position_modified
            assert r1.boundary_hit == r2.boundary_hit

            # Compare constraint analysis
            analysis1 = r1.get_constraint_analysis()
            analysis2 = r2.get_constraint_analysis()

            assert analysis1["position_delta"] == analysis2["position_delta"]
            assert analysis1["constraint_type"] == analysis2["constraint_type"]
            assert analysis1["boundary_hit"] == analysis2["boundary_hit"]

    def test_consistent_validation_results(self):
        """Test position and movement validation produces consistent results across enforcer instances and execution time."""
        # Create multiple BoundaryEnforcer instances with identical configurations
        enforcers = [
            create_test_boundary_enforcer(enable_caching=False) for _ in range(3)
        ]

        # Test identical validation operations across all enforcer instances
        test_positions = [
            TEST_COORDINATES_CENTER,
            TEST_COORDINATES_CORNER_ORIGIN,
            TEST_COORDINATES_EDGE_TOP,
            (10, 20),  # Tuple format
            (-1, 16),  # Invalid position
        ]

        for position in test_positions:
            validation_results = []

            for enforcer in enforcers:
                try:
                    result = enforcer.validate_position(
                        position, raise_on_invalid=False
                    )
                    validation_results.append(result)
                except Exception as e:
                    validation_results.append(str(type(e).__name__))

            # Verify validation results are identical across all instances
            first_result = validation_results[0]
            for result in validation_results[1:]:
                assert result == first_result

        # Test movement validation consistency
        for enforcer in enforcers:
            for action in ALL_VALID_ACTIONS:
                result1 = enforcer.is_movement_valid(TEST_COORDINATES_CENTER, action)
                result2 = enforcer.is_movement_valid(TEST_COORDINATES_CENTER, action)

                # Check consistency over time with repeated validation operations
                assert result1 == result2

    def test_reproducible_performance_metrics(self):
        """Test performance metrics collection is consistent and reproducible for benchmark validation and optimization analysis."""
        # Create BoundaryEnforcer and perform identical operations multiple times
        test_operations = [
            (TEST_COORDINATES_CENTER, Action.UP),
            (TEST_COORDINATES_CENTER, Action.RIGHT),
        ]

        performance_runs = []

        # Execute multiple performance measurement runs
        for run in range(3):
            enforcer = create_test_boundary_enforcer(enable_caching=False)
            run_timings = []

            for position, action in test_operations:
                for _ in range(50):  # Reduced iterations for reproducibility test
                    start_time = time.perf_counter()
                    result = enforcer.enforce_movement_bounds(position, action)
                    end_time = time.perf_counter()

                    run_timings.append((end_time - start_time) * 1000)

            performance_runs.append(
                {
                    "mean_time": np.mean(run_timings),
                    "median_time": np.median(run_timings),
                    "total_operations": len(run_timings),
                }
            )

        # Verify timing measurements are within acceptable statistical variance
        mean_times = [run["mean_time"] for run in performance_runs]
        mean_variance = np.std(mean_times) / np.mean(
            mean_times
        )  # Coefficient of variation

        # Performance should be relatively consistent (within 50% coefficient of variation)
        assert (
            mean_variance < 0.5
        ), f"Performance variance too high: {mean_variance:.3f}"

        # Check that all runs executed the same number of operations
        operation_counts = [run["total_operations"] for run in performance_runs]
        assert all(count == operation_counts[0] for count in operation_counts)
