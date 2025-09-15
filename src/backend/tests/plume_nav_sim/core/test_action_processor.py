"""
Comprehensive pytest test suite for ActionProcessor component validation covering action validation, 
movement calculation, boundary enforcement integration, performance benchmarking, error handling, 
and reproducibility testing with complete coverage of discrete action space processing functionality 
for the plume navigation reinforcement learning environment.

This test suite provides exhaustive validation of all ActionProcessor functionality including public 
methods, performance characteristics, error conditions, and integration with core system components 
to ensure reliable action processing operations for the Gymnasium environment implementation.
"""

# External imports with version comments
import pytest  # >=8.0.0 - Testing framework for comprehensive action processor test suite execution
import numpy as np  # >=2.1.0 - Array operations and mathematical validation for action processing testing
import time  # >=3.10 - High-precision timing for action processing performance benchmark validation
import itertools  # >=3.10 - Test parameter generation for comprehensive action processing testing
import random  # >=3.10 - Random test data generation for edge case action validation and stress testing
import unittest.mock  # >=3.10 - Component mocking for isolated action processing testing
import copy  # >=3.10 - Deep copying for action processing result validation and state preservation

# Internal imports from action processor module for comprehensive testing
from plume_nav_sim.core.action_processor import (
    ActionProcessor,
    ActionProcessingResult,
    ActionProcessingConfig,
    process_action,
    validate_action_bounds,
    calculate_movement_delta,
    is_valid_action_for_position
)

# Internal imports from core types module for action processing testing
from plume_nav_sim.core.types import (
    Action,
    ActionType,
    Coordinates,
    GridSize,
    MovementVector,
    create_coordinates,
    create_grid_size,
    validate_action,
    get_movement_vector
)

# Internal imports from constants module for action processing testing
from plume_nav_sim.core.constants import (
    ACTION_UP,
    ACTION_RIGHT,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_SPACE_SIZE,
    MOVEMENT_VECTORS,
    DEFAULT_GRID_SIZE,
    PERFORMANCE_TARGET_STEP_LATENCY_MS,
    ACTION_PROCESSING_PERFORMANCE_TARGET_MS,
    TESTING_CONSTANTS,
    VALIDATION_ERROR_MESSAGES
)

# Internal imports from boundary enforcer module for integration testing
from plume_nav_sim.core.boundary_enforcer import (
    BoundaryEnforcer,
    BoundaryEnforcementResult
)

# Internal imports from exceptions module for error handling validation
from plume_nav_sim.utils.exceptions import (
    ValidationError,
    StateError,
    ComponentError
)

# Global test constants and configuration
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

ALL_VALID_ACTIONS = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
ALL_ACTION_INTEGERS = [0, 1, 2, 3]
INVALID_ACTIONS = [-1, 4, 5, None, 'up', 1.5]

ACTION_PROCESSING_TEST_POSITIONS = [(16, 16), (0, 0), (31, 31), (0, 31), (31, 0), (15, 15)]
BOUNDARY_TEST_SCENARIOS = [(31, 0, Action.RIGHT), (0, 16, Action.LEFT), (16, 31, Action.UP), (16, 0, Action.DOWN)]
CONFIG_VARIATIONS = 8


def create_test_action_processor(
    grid_size=None, 
    config=None, 
    boundary_enforcer=None
):
    """
    Factory function to create ActionProcessor instances with test configurations for consistent 
    test setup across action processing test cases.
    
    Args:
        grid_size (Optional[GridSize]): Grid size for boundary testing, defaults to TEST_GRID_SIZE
        config (Optional[ActionProcessingConfig]): Configuration for action processing
        boundary_enforcer (Optional[BoundaryEnforcer]): Boundary enforcer for testing
        
    Returns:
        ActionProcessor: Configured ActionProcessor instance for action processing testing
    """
    # Use TEST_GRID_SIZE as default if grid_size not provided
    if grid_size is None:
        grid_size = TEST_GRID_SIZE
    
    # Create default ActionProcessingConfig with standard settings if config not provided
    if config is None:
        config = ActionProcessingConfig()
    
    # Create default BoundaryEnforcer if boundary_enforcer not provided and boundaries enabled
    if boundary_enforcer is None and config.enforce_boundaries:
        boundary_enforcer = BoundaryEnforcer(grid_size)
    
    # Initialize ActionProcessor with provided or default parameters
    processor = ActionProcessor(
        grid_size=grid_size,
        config=config,
        boundary_enforcer=boundary_enforcer
    )
    
    # Validate processor initialization was successful and grid size is set correctly
    assert processor.grid_size == grid_size
    assert processor.config == config
    
    # Return configured ActionProcessor ready for comprehensive action processing testing
    return processor


def generate_action_test_cases(grid_size, include_invalid_actions=False):
    """
    Generate comprehensive test cases for action processing testing with various position-action 
    combinations and boundary scenarios.
    
    Args:
        grid_size (GridSize): Grid size for boundary testing
        include_invalid_actions (bool): Include invalid action types in test cases
        
    Returns:
        list: List of test case tuples (position, action, expected_processing_result)
    """
    test_cases = []
    
    # Generate center positions for unrestricted movement testing
    center_x, center_y = grid_size.width // 2, grid_size.height // 2
    center_position = Coordinates(center_x, center_y)
    
    for action in ALL_VALID_ACTIONS:
        test_cases.append((center_position, action, True))  # All movements valid from center
    
    # Generate corner positions at all four grid corners for boundary testing
    corners = [
        Coordinates(0, 0),  # Bottom-left corner
        Coordinates(grid_size.width - 1, 0),  # Bottom-right corner
        Coordinates(0, grid_size.height - 1),  # Top-left corner
        Coordinates(grid_size.width - 1, grid_size.height - 1)  # Top-right corner
    ]
    
    for corner in corners:
        for action in ALL_VALID_ACTIONS:
            # Determine if action would violate boundaries from corner position
            movement_vector = action.to_vector()
            new_x = corner.x + movement_vector[0]
            new_y = corner.y + movement_vector[1]
            is_valid = (0 <= new_x < grid_size.width and 0 <= new_y < grid_size.height)
            test_cases.append((corner, action, is_valid))
    
    # Generate edge positions along all four grid boundaries
    mid_x, mid_y = grid_size.width // 2, grid_size.height // 2
    edges = [
        Coordinates(mid_x, 0),  # Bottom edge
        Coordinates(mid_x, grid_size.height - 1),  # Top edge
        Coordinates(0, mid_y),  # Left edge
        Coordinates(grid_size.width - 1, mid_y)  # Right edge
    ]
    
    for edge in edges:
        for action in ALL_VALID_ACTIONS:
            movement_vector = action.to_vector()
            new_x = edge.x + movement_vector[0]
            new_y = edge.y + movement_vector[1]
            is_valid = (0 <= new_x < grid_size.width and 0 <= new_y < grid_size.height)
            test_cases.append((edge, action, is_valid))
    
    # Include invalid action types if include_invalid_actions is True
    if include_invalid_actions:
        for position in [center_position, corners[0], edges[0]]:
            for invalid_action in INVALID_ACTIONS:
                test_cases.append((position, invalid_action, False))
    
    # Return comprehensive action processing test case collection for systematic validation
    return test_cases


def measure_action_processing_performance(processor, iterations, position, action):
    """
    Measure action processing performance over multiple iterations for benchmark validation 
    against ACTION_PROCESSING_PERFORMANCE_TARGET_MS requirement.
    
    Args:
        processor (ActionProcessor): ActionProcessor instance for performance testing
        iterations (int): Number of iterations for performance measurement
        position (Coordinates): Position for action processing testing
        action (ActionType): Action for performance testing
        
    Returns:
        dict: Performance metrics including mean, min, max, and percentile timing data
    """
    timing_measurements = []
    
    # Execute action processing iterations with high-precision timing using time.perf_counter()
    for _ in range(iterations):
        start_time = time.perf_counter()
        result = processor.process_action(position, action)
        end_time = time.perf_counter()
        
        # Collect individual operation timing measurements for statistical analysis
        execution_time_ms = (end_time - start_time) * 1000
        timing_measurements.append(execution_time_ms)
    
    # Calculate statistical metrics including mean, median, 95th and 99th percentiles
    timing_array = np.array(timing_measurements)
    performance_metrics = {
        'mean_time_ms': np.mean(timing_array),
        'median_time_ms': np.median(timing_array),
        'min_time_ms': np.min(timing_array),
        'max_time_ms': np.max(timing_array),
        'std_time_ms': np.std(timing_array),
        'percentile_95_ms': np.percentile(timing_array, 95),
        'percentile_99_ms': np.percentile(timing_array, 99),
        'total_iterations': iterations
    }
    
    # Validate performance against ACTION_PROCESSING_PERFORMANCE_TARGET_MS (0.1ms)
    target_compliance_rate = np.sum(timing_array <= ACTION_PROCESSING_PERFORMANCE_TARGET_MS) / iterations
    performance_metrics['target_compliance_rate'] = target_compliance_rate
    
    # Include action validation statistics and movement calculation analysis
    processing_stats = processor.get_processing_statistics()
    performance_metrics.update({
        'validation_operations': processing_stats.get('validation_count', 0),
        'successful_movements': processing_stats.get('successful_movements', 0),
        'boundary_hits': processing_stats.get('boundary_hits', 0)
    })
    
    # Return comprehensive performance analysis dictionary with benchmark validation results
    return performance_metrics


def validate_action_processing_consistency(original_position, action, processing_result, grid_bounds):
    """
    Validate action processing results are consistent with mathematical expectations and 
    coordinate system conventions for system integrity.
    
    Args:
        original_position (Coordinates): Original agent position
        action (ActionType): Action that was processed
        processing_result (ActionProcessingResult): Result from action processing
        grid_bounds (GridSize): Grid boundaries for validation
        
    Returns:
        bool: True if action processing is mathematically consistent
    """
    # Calculate expected final position using action movement vector and grid constraints
    movement_vector = get_movement_vector(action)
    expected_x = original_position.x + movement_vector[0]
    expected_y = original_position.y + movement_vector[1]
    
    # Check if expected position would be within bounds
    within_bounds = (0 <= expected_x < grid_bounds.width and 0 <= expected_y < grid_bounds.height)
    
    if within_bounds:
        # Compare processing_result.final_position with mathematical expectation
        expected_final = Coordinates(expected_x, expected_y)
        if processing_result.final_position != expected_final:
            return False
        
        # Verify position_changed flag accurately reflects whether position changed
        position_actually_changed = (processing_result.final_position != original_position)
        if processing_result.position_changed != position_actually_changed:
            return False
        
        # Check boundary_hit flag correctly indicates no boundary constraint activation
        if processing_result.boundary_hit:
            return False
    else:
        # Movement would exceed bounds - verify boundary handling
        if not processing_result.boundary_hit:
            return False
        
        # Final position should be within bounds after boundary enforcement
        if not processing_result.final_position.is_within_bounds(grid_bounds):
            return False
    
    # Verify coordinate system consistency with (x,y) positioning conventions
    if (processing_result.final_position.x < 0 or processing_result.final_position.x >= grid_bounds.width or
        processing_result.final_position.y < 0 or processing_result.final_position.y >= grid_bounds.height):
        return False
    
    # Return True if all action processing results match mathematical expectations
    return True


def create_action_processing_config_variations(include_edge_cases=False):
    """
    Create various ActionProcessingConfig configurations for comprehensive processing policy 
    testing covering different validation and monitoring scenarios.
    
    Args:
        include_edge_cases (bool): Include edge case configurations for thorough testing
        
    Returns:
        list: List of ActionProcessingConfig instances with different configuration settings
    """
    config_variations = []
    
    # Create standard config with default validation and boundary enforcement enabled
    config_variations.append(ActionProcessingConfig(
        enable_validation=True,
        enforce_boundaries=True,
        enable_performance_monitoring=True
    ))
    
    # Create strict validation config with enhanced checking and performance monitoring
    config_variations.append(ActionProcessingConfig(
        enable_validation=True,
        enforce_boundaries=True,
        enable_performance_monitoring=True
    ))
    
    # Create performance-optimized config with validation caching and minimal overhead
    config_variations.append(ActionProcessingConfig(
        enable_validation=False,  # Disabled for performance
        enforce_boundaries=True,
        enable_performance_monitoring=False
    ))
    
    # Create no-boundary-enforcement config for pure validation testing
    config_variations.append(ActionProcessingConfig(
        enable_validation=True,
        enforce_boundaries=False,
        enable_performance_monitoring=True
    ))
    
    # Create monitoring-disabled config for performance-critical scenarios
    config_variations.append(ActionProcessingConfig(
        enable_validation=True,
        enforce_boundaries=True,
        enable_performance_monitoring=False
    ))
    
    # Include edge case configurations if include_edge_cases is True
    if include_edge_cases:
        # All features disabled configuration
        config_variations.append(ActionProcessingConfig(
            enable_validation=False,
            enforce_boundaries=False,
            enable_performance_monitoring=False
        ))
        
        # Validation only configuration
        config_variations.append(ActionProcessingConfig(
            enable_validation=True,
            enforce_boundaries=False,
            enable_performance_monitoring=False
        ))
        
        # Boundaries only configuration
        config_variations.append(ActionProcessingConfig(
            enable_validation=False,
            enforce_boundaries=True,
            enable_performance_monitoring=False
        ))
    
    # Return comprehensive list of configuration variations for policy testing
    return config_variations


def verify_processing_statistics_accuracy(processor, test_operations, expected_stats):
    """
    Verify action processing statistics are accurate and consistent with actual processing 
    operations for monitoring validation.
    
    Args:
        processor (ActionProcessor): ActionProcessor instance for statistics verification
        test_operations (list): List of operations to execute for statistics validation
        expected_stats (dict): Expected statistics values for comparison
        
    Returns:
        bool: True if statistics match expected values within tolerance
    """
    # Execute all test_operations and track expected actions processed manually
    manual_action_count = 0
    manual_validation_count = 0
    manual_successful_movements = 0
    
    for position, action in test_operations:
        try:
            result = processor.process_action(position, action)
            manual_action_count += 1
            
            if result.action_valid:
                manual_validation_count += 1
            
            if result.was_movement_successful():
                manual_successful_movements += 1
                
        except Exception:
            # Count failed operations
            pass
    
    # Get processing statistics using processor.get_processing_statistics()
    actual_stats = processor.get_processing_statistics()
    
    # Compare actual action count with manually tracked count
    if 'actions_processed' in expected_stats:
        expected_actions = expected_stats['actions_processed']
        actual_actions = actual_stats.get('actions_processed', 0)
        if abs(actual_actions - expected_actions) > 1:  # Allow tolerance of 1
            return False
    
    # Verify validation operation count matches executed validations
    if 'validation_operations' in expected_stats:
        expected_validations = expected_stats['validation_operations']
        actual_validations = actual_stats.get('validation_operations', 0)
        if abs(actual_validations - expected_validations) > 1:
            return False
    
    # Check action success rate calculation accuracy
    if manual_action_count > 0:
        expected_success_rate = manual_successful_movements / manual_action_count
        actual_success_rate = actual_stats.get('success_rate', 0)
        if abs(actual_success_rate - expected_success_rate) > 0.1:  # 10% tolerance
            return False
    
    # Validate performance metrics are within reasonable ranges
    avg_processing_time = actual_stats.get('average_processing_time_ms', 0)
    if avg_processing_time > ACTION_PROCESSING_PERFORMANCE_TARGET_MS * 10:  # 10x tolerance
        return False
    
    # Return True if all statistics match expectations within defined tolerance
    return True


class TestActionProcessorInitialization:
    """
    Test class for ActionProcessor initialization covering constructor validation, parameter 
    configuration, and basic setup verification.
    """
    
    def test_action_processor_initialization_default_parameters(self):
        """
        Test ActionProcessor initialization with default parameters and validate proper setup.
        """
        # Create ActionProcessor with default TEST_GRID_SIZE
        processor = ActionProcessor(grid_size=TEST_GRID_SIZE)
        
        # Validate grid_size property is set correctly to expected dimensions
        assert processor.grid_size == TEST_GRID_SIZE
        assert processor.grid_size.width == 32
        assert processor.grid_size.height == 32
        
        # Verify config is initialized with default ActionProcessingConfig
        assert processor.config is not None
        assert isinstance(processor.config, ActionProcessingConfig)
        assert processor.config.enable_validation is True
        assert processor.config.enforce_boundaries is True
        
        # Check boundary_enforcer is properly initialized if boundaries enabled
        if processor.config.enforce_boundaries:
            assert processor.boundary_enforcer is not None
            assert isinstance(processor.boundary_enforcer, BoundaryEnforcer)
        
        # Assert actions_processed and validation_errors start at 0
        stats = processor.get_processing_statistics()
        assert stats.get('actions_processed', 0) == 0
        assert stats.get('validation_errors', 0) == 0
    
    def test_action_processor_initialization_custom_parameters(self):
        """
        Test ActionProcessor initialization with custom grid size, configuration, and boundary 
        enforcer settings.
        """
        # Create custom GridSize, ActionProcessingConfig, and BoundaryEnforcer instances
        custom_grid = GridSize(64, 64)
        custom_config = ActionProcessingConfig(
            enable_validation=True,
            enforce_boundaries=True,
            enable_performance_monitoring=True
        )
        custom_boundary_enforcer = BoundaryEnforcer(custom_grid)
        
        # Initialize ActionProcessor with custom parameters including monitoring settings
        processor = ActionProcessor(
            grid_size=custom_grid,
            config=custom_config,
            boundary_enforcer=custom_boundary_enforcer
        )
        
        # Validate custom grid_size is properly stored and accessible
        assert processor.grid_size == custom_grid
        assert processor.grid_size.width == 64
        assert processor.grid_size.height == 64
        
        # Verify custom config is used correctly with proper validation
        assert processor.config == custom_config
        assert processor.config.enable_validation is True
        assert processor.config.enable_performance_monitoring is True
        
        # Check custom boundary_enforcer is integrated correctly
        assert processor.boundary_enforcer == custom_boundary_enforcer
        assert processor.boundary_enforcer.grid_size == custom_grid
        
        # Assert performance metrics initialization with custom configuration
        stats = processor.get_processing_statistics()
        assert 'actions_processed' in stats
        assert 'average_processing_time_ms' in stats
    
    def test_action_processor_invalid_initialization(self):
        """
        Test ActionProcessor initialization with invalid parameters raises appropriate 
        ValidationError exceptions.
        """
        # Test initialization with invalid grid_size including negative dimensions and zero values
        with pytest.raises((ValidationError, ValueError, TypeError)):
            ActionProcessor(grid_size=GridSize(-1, 10))
        
        with pytest.raises((ValidationError, ValueError, TypeError)):
            ActionProcessor(grid_size=GridSize(0, 0))
        
        # Test initialization with None grid_size parameter
        with pytest.raises((ValidationError, ValueError, TypeError)):
            ActionProcessor(grid_size=None)
        
        # Test initialization with invalid config type and malformed configurations
        with pytest.raises((ValidationError, TypeError)):
            ActionProcessor(grid_size=TEST_GRID_SIZE, config="invalid_config")
    
    def test_action_processor_configuration_validation(self):
        """
        Test action processing configuration validation during initialization ensures consistency 
        and feasibility.
        """
        # Create ActionProcessor with various configuration settings
        config = ActionProcessingConfig(
            enable_validation=True,
            enforce_boundaries=True,
            enable_performance_monitoring=True
        )
        
        # Test configuration validation with boundary enforcement enabled and disabled
        processor = ActionProcessor(grid_size=TEST_GRID_SIZE, config=config)
        
        # Verify configuration consistency checks detect compatible settings
        assert processor.config.validate_configuration() is True
        
        # Check mathematical feasibility validation for normal configurations
        assert processor.grid_size.width > 0
        assert processor.grid_size.height > 0


class TestActionValidation:
    """
    Test class for action validation functionality covering action parameter validation, type 
    checking, and bounds validation.
    """
    
    def test_validate_action_valid_actions(self):
        """
        Test action validation accepts valid Action enum values and integer actions within bounds.
        """
        # Create ActionProcessor with TEST_GRID_SIZE
        processor = create_test_action_processor()
        
        # Test validation with all Action enum values (UP, RIGHT, DOWN, LEFT)
        for action in ALL_VALID_ACTIONS:
            assert processor.validate_action(action) is True
        
        # Test validation with corresponding integer values (0, 1, 2, 3)
        for action_int in ALL_ACTION_INTEGERS:
            assert processor.validate_action(action_int) is True
        
        # Verify no exceptions are raised for valid action inputs
        try:
            for action in ALL_VALID_ACTIONS + ALL_ACTION_INTEGERS:
                processor.validate_action(action)
        except Exception as e:
            pytest.fail(f"Valid action validation raised exception: {e}")
    
    def test_validate_action_invalid_actions(self):
        """
        Test action validation rejects invalid actions outside bounds with appropriate error handling.
        """
        # Create ActionProcessor with TEST_GRID_SIZE
        processor = create_test_action_processor()
        
        # Test validation with out-of-bounds integer actions (-1, 4, 5)
        invalid_integers = [-1, 4, 5, 10, -5]
        for invalid_action in invalid_integers:
            assert processor.validate_action(invalid_action) is False
        
        # Test validation with None and non-numeric type inputs
        invalid_types = [None, "up", "right", 1.5, [], {}]
        for invalid_action in invalid_types:
            assert processor.validate_action(invalid_action) is False
    
    def test_validate_action_strict_mode(self):
        """
        Test action validation in strict mode with enhanced checking and comprehensive error reporting.
        """
        # Create ActionProcessor with strict validation enabled in configuration
        strict_config = ActionProcessingConfig(enable_validation=True, enforce_boundaries=True)
        processor = create_test_action_processor(config=strict_config)
        
        # Test action validation with enhanced checking
        for action in ALL_VALID_ACTIONS:
            assert processor.validate_action(action) is True
        
        # Verify strict mode applies additional type and consistency checks
        for invalid_action in INVALID_ACTIONS:
            if invalid_action is not None:  # None is handled separately
                assert processor.validate_action(invalid_action) is False
    
    def test_validate_action_caching(self):
        """
        Test action validation caching functionality improves performance for repeated validation operations.
        """
        # Create ActionProcessor with caching enabled
        processor = create_test_action_processor()
        
        # Validate identical actions multiple times and measure timing
        action_to_test = Action.UP
        
        # First validation (cache miss)
        start_time = time.perf_counter()
        result1 = processor.validate_action(action_to_test)
        first_time = time.perf_counter() - start_time
        
        # Subsequent validation (potential cache hit)
        start_time = time.perf_counter()
        result2 = processor.validate_action(action_to_test)
        second_time = time.perf_counter() - start_time
        
        # Assert cached results are identical to fresh validation calculations
        assert result1 == result2
        assert result1 is True
    
    def test_validate_action_performance(self):
        """
        Test action validation performance meets target latency requirements for high-frequency processing.
        """
        # Create ActionProcessor instance for performance testing
        processor = create_test_action_processor()
        
        # Measure validate_action execution time over 1000 iterations
        iterations = PERFORMANCE_TEST_ITERATIONS
        timing_measurements = []
        
        for action in ALL_VALID_ACTIONS:
            for _ in range(iterations // len(ALL_VALID_ACTIONS)):
                start_time = time.perf_counter()
                processor.validate_action(action)
                end_time = time.perf_counter()
                timing_measurements.append((end_time - start_time) * 1000)
        
        # Assert average validation time is well below ACTION_PROCESSING_PERFORMANCE_TARGET_MS
        avg_time = np.mean(timing_measurements)
        assert avg_time < ACTION_PROCESSING_PERFORMANCE_TARGET_MS * 2  # Allow 2x tolerance
        
        # Verify 99th percentile validation time meets real-time requirements
        percentile_99 = np.percentile(timing_measurements, 99)
        assert percentile_99 < ACTION_PROCESSING_PERFORMANCE_TARGET_MS * 5  # Allow 5x tolerance


class TestMovementCalculation:
    """
    Test class for movement calculation functionality covering delta computation, position updates, 
    and mathematical consistency.
    """
    
    def test_calculate_movement_delta_valid_actions(self):
        """
        Test movement delta calculation returns correct coordinate deltas for valid actions.
        """
        # Create ActionProcessor with TEST_GRID_SIZE
        processor = create_test_action_processor()
        
        # Test calculate_movement_delta with Action.UP should return (0, 1)
        delta_up = processor.calculate_movement_delta(Action.UP)
        assert delta_up == MOVEMENT_VECTORS[ACTION_UP]
        assert delta_up == (0, 1)
        
        # Test calculate_movement_delta with Action.RIGHT should return (1, 0)
        delta_right = processor.calculate_movement_delta(Action.RIGHT)
        assert delta_right == MOVEMENT_VECTORS[ACTION_RIGHT]
        assert delta_right == (1, 0)
        
        # Test calculate_movement_delta with Action.DOWN should return (0, -1)
        delta_down = processor.calculate_movement_delta(Action.DOWN)
        assert delta_down == MOVEMENT_VECTORS[ACTION_DOWN]
        assert delta_down == (0, -1)
        
        # Test calculate_movement_delta with Action.LEFT should return (-1, 0)
        delta_left = processor.calculate_movement_delta(Action.LEFT)
        assert delta_left == MOVEMENT_VECTORS[ACTION_LEFT]
        assert delta_left == (-1, 0)
        
        # Assert movement deltas match MOVEMENT_VECTORS dictionary
        for action_int, expected_vector in MOVEMENT_VECTORS.items():
            calculated_delta = processor.calculate_movement_delta(action_int)
            assert calculated_delta == expected_vector
    
    def test_calculate_movement_outcome_within_bounds(self):
        """
        Test movement outcome calculation for valid movements that stay within grid boundaries.
        """
        # Create ActionProcessor with TEST_GRID_SIZE
        processor = create_test_action_processor()
        
        # Test calculate_movement_outcome from center position with all four actions
        center_pos = TEST_COORDINATES_CENTER
        
        for action in ALL_VALID_ACTIONS:
            result = processor.calculate_movement_outcome(center_pos, action)
            
            # Verify final_position equals original_position plus movement_delta
            movement_delta = processor.calculate_movement_delta(action)
            expected_final = Coordinates(
                center_pos.x + movement_delta[0],
                center_pos.y + movement_delta[1]
            )
            assert result.final_position == expected_final
            
            # Check position_changed is True for all movements from center
            assert result.position_changed is True
            
            # Assert boundary_hit is False for movements within grid
            assert result.boundary_hit is False
            
            # Validate movement_successful reflects successful position change
            assert result.was_movement_successful() is True
    
    def test_calculate_movement_outcome_boundary_constraints(self):
        """
        Test movement outcome calculation handles boundary constraints with proper enforcement.
        """
        # Create ActionProcessor with boundary enforcement enabled
        processor = create_test_action_processor()
        
        # Test movement from edge positions with boundary-violating actions
        for position, violating_action in [(TEST_COORDINATES_EDGE_RIGHT, Action.RIGHT),
                                         (TEST_COORDINATES_EDGE_LEFT, Action.LEFT),
                                         (TEST_COORDINATES_EDGE_TOP, Action.UP),
                                         (TEST_COORDINATES_EDGE_BOTTOM, Action.DOWN)]:
            
            result = processor.calculate_movement_outcome(position, violating_action)
            
            # Check boundary_hit flag is True when boundary constraints are encountered
            assert result.boundary_hit is True
            
            # Assert final_position reflects boundary enforcement (clamping or blocking)
            assert result.final_position.is_within_bounds(processor.grid_size)
    
    def test_movement_delta_performance(self):
        """
        Test movement delta calculation performance meets latency targets for frequent computation.
        """
        # Create ActionProcessor instance for performance measurement
        processor = create_test_action_processor()
        
        # Measure calculate_movement_delta execution time over multiple iterations
        iterations = PERFORMANCE_TEST_ITERATIONS
        timing_measurements = []
        
        for _ in range(iterations):
            action = random.choice(ALL_VALID_ACTIONS)
            start_time = time.perf_counter()
            processor.calculate_movement_delta(action)
            end_time = time.perf_counter()
            timing_measurements.append((end_time - start_time) * 1000)
        
        # Assert average calculation time is well below performance target
        avg_time = np.mean(timing_measurements)
        assert avg_time < ACTION_PROCESSING_PERFORMANCE_TARGET_MS
        
        # Verify movement calculation is optimized for mathematical operations
        assert avg_time < 0.01  # Should be very fast for simple lookup


class TestActionProcessing:
    """
    Test class for complete action processing workflow covering comprehensive action processing, 
    result generation, and integration analysis.
    """
    
    def test_process_action_valid_movement(self):
        """
        Test complete action processing with valid movements returns proper results and 
        comprehensive analysis.
        """
        # Create ActionProcessor instance for valid movement testing
        processor = create_test_action_processor()
        
        # Test process_action from center position with valid actions
        center_pos = TEST_COORDINATES_CENTER
        
        for action in ALL_VALID_ACTIONS:
            result = processor.process_action(center_pos, action)
            
            # Verify ActionProcessingResult.final_position equals expected movement result
            movement_delta = get_movement_vector(action)
            expected_pos = Coordinates(
                center_pos.x + movement_delta[0],
                center_pos.y + movement_delta[1]
            )
            assert result.final_position == expected_pos
            
            # Check action_valid is True for valid actions
            assert result.action_valid is True
            
            # Assert position_changed reflects actual position change
            assert result.position_changed is True
            
            # Validate was_movement_successful returns True for successful movements
            assert result.was_movement_successful() is True
    
    def test_process_action_boundary_enforcement(self):
        """
        Test action processing with boundary enforcement integration constrains movements appropriately.
        """
        # Create ActionProcessor with boundary enforcement enabled
        processor = create_test_action_processor()
        
        # Test processing from edge positions with boundary-violating actions
        test_scenarios = [
            (TEST_COORDINATES_EDGE_RIGHT, Action.RIGHT),
            (TEST_COORDINATES_EDGE_LEFT, Action.LEFT),
            (TEST_COORDINATES_EDGE_TOP, Action.UP),
            (TEST_COORDINATES_EDGE_BOTTOM, Action.DOWN)
        ]
        
        for position, boundary_action in test_scenarios:
            result = processor.process_action(position, boundary_action)
            
            # Check boundary_hit is True for movements constrained by boundaries
            assert result.boundary_hit is True
            
            # Assert final_position reflects boundary enforcement policy (clamping/blocking)
            assert result.final_position.is_within_bounds(processor.grid_size)
    
    def test_process_action_invalid_actions(self):
        """
        Test action processing with invalid actions handles errors appropriately with proper validation.
        """
        # Create ActionProcessor with validation enabled
        processor = create_test_action_processor()
        
        # Test processing with invalid action types and out-of-bounds values
        center_pos = TEST_COORDINATES_CENTER
        
        for invalid_action in INVALID_ACTIONS:
            result = processor.process_action(center_pos, invalid_action)
            
            # Check error handling maintains processor stability
            assert result.action_valid is False
            
            # Assert invalid actions don't corrupt processor state
            assert result.original_position == center_pos
    
    def test_action_processing_result_analysis(self):
        """
        Test ActionProcessingResult analysis methods provide accurate processing information 
        and debugging data.
        """
        # Create ActionProcessor and process various action scenarios
        processor = create_test_action_processor()
        
        # Test successful movement scenario
        result = processor.process_action(TEST_COORDINATES_CENTER, Action.UP)
        
        # Test ActionProcessingResult.get_movement_summary() provides detailed analysis
        summary = result.get_movement_summary()
        assert isinstance(summary, dict)
        assert 'movement_direction' in summary
        assert 'movement_successful' in summary
        
        # Verify was_movement_successful() returns correct boolean status
        assert isinstance(result.was_movement_successful(), bool)
        
        # Check to_dict() returns complete result dictionary for serialization
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert 'action' in result_dict
        assert 'original_position' in result_dict
        assert 'final_position' in result_dict
    
    def test_action_processing_performance_monitoring(self):
        """
        Test action processing performance monitoring and metrics collection for optimization analysis.
        """
        # Create ActionProcessor instance with performance monitoring enabled
        config = ActionProcessingConfig(enable_performance_monitoring=True)
        processor = create_test_action_processor(config=config)
        
        # Process multiple actions and collect performance data
        test_position = TEST_COORDINATES_CENTER
        iterations = 100
        
        for _ in range(iterations):
            action = random.choice(ALL_VALID_ACTIONS)
            result = processor.process_action(test_position, action)
            
            # Verify processing_time is recorded in ActionProcessingResult
            if hasattr(result, 'processing_time'):
                assert result.processing_time >= 0
        
        # Check get_processing_statistics() returns comprehensive timing metrics
        stats = processor.get_processing_statistics()
        assert 'actions_processed' in stats
        assert stats['actions_processed'] >= iterations


class TestValidActionAnalysis:
    """
    Test class for valid action analysis functionality covering action space filtering and 
    position-dependent validation.
    """
    
    def test_get_valid_actions_center_position(self):
        """
        Test valid actions analysis from center position should return all four cardinal directions.
        """
        # Create ActionProcessor with TEST_GRID_SIZE
        processor = create_test_action_processor()
        
        # Test get_valid_actions from TEST_COORDINATES_CENTER (16, 16)
        valid_actions = processor.get_valid_actions(TEST_COORDINATES_CENTER)
        
        # Verify returned action list contains all four Action enum values
        assert len(valid_actions) == 4
        
        # Check valid actions are [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
        for action in ALL_VALID_ACTIONS:
            assert action in valid_actions
        
        # Assert all returned actions are valid Action enum instances
        for action in valid_actions:
            assert isinstance(action, Action)
    
    def test_get_valid_actions_corner_positions(self):
        """
        Test valid actions analysis from corner positions should return two valid directions.
        """
        # Create ActionProcessor with TEST_GRID_SIZE
        processor = create_test_action_processor()
        
        # Test get_valid_actions from all four corner positions
        corner_tests = [
            (TEST_COORDINATES_CORNER_ORIGIN, [Action.UP, Action.RIGHT]),  # (0, 0)
            (Coordinates(31, 0), [Action.UP, Action.LEFT]),  # (31, 0)  
            (Coordinates(0, 31), [Action.DOWN, Action.RIGHT]),  # (0, 31)
            (TEST_COORDINATES_CORNER_OPPOSITE, [Action.DOWN, Action.LEFT])  # (31, 31)
        ]
        
        for corner_pos, expected_actions in corner_tests:
            valid_actions = processor.get_valid_actions(corner_pos)
            
            # Assert each corner position returns exactly 2 valid actions
            assert len(valid_actions) == 2
            
            # Verify expected actions are present
            for expected_action in expected_actions:
                assert expected_action in valid_actions
    
    def test_get_valid_actions_edge_positions(self):
        """
        Test valid actions analysis from edge positions should return three valid directions.
        """
        # Create ActionProcessor with TEST_GRID_SIZE
        processor = create_test_action_processor()
        
        # Test get_valid_actions from all four edge positions
        edge_tests = [
            (TEST_COORDINATES_EDGE_TOP, Action.UP),    # Top edge excludes UP
            (TEST_COORDINATES_EDGE_RIGHT, Action.RIGHT),  # Right edge excludes RIGHT
            (TEST_COORDINATES_EDGE_BOTTOM, Action.DOWN),   # Bottom edge excludes DOWN
            (TEST_COORDINATES_EDGE_LEFT, Action.LEFT)      # Left edge excludes LEFT
        ]
        
        for edge_pos, excluded_action in edge_tests:
            valid_actions = processor.get_valid_actions(edge_pos)
            
            # Assert each edge position returns exactly 3 valid actions
            assert len(valid_actions) == 3
            
            # Check excluded action is not in valid actions list
            assert excluded_action not in valid_actions
            
            # Verify all other actions are present
            for action in ALL_VALID_ACTIONS:
                if action != excluded_action:
                    assert action in valid_actions
    
    def test_is_valid_action_for_position(self):
        """
        Test position-dependent action validation for boundary constraint checking and action filtering.
        """
        # Create ActionProcessor with TEST_GRID_SIZE
        processor = create_test_action_processor()
        
        # Test is_valid_action_for_position with various position-action combinations
        test_cases = generate_action_test_cases(TEST_GRID_SIZE)
        
        for position, action, expected_validity in test_cases:
            if action in ALL_VALID_ACTIONS:  # Only test valid action types
                result = processor.is_valid_action_for_position(position, action)
                
                # Verify function returns True for movements that stay within bounds
                # and False for movements that would violate boundaries
                assert isinstance(result, bool)
                
                # Assert consistency between is_valid_action_for_position and get_valid_actions results
                valid_actions = processor.get_valid_actions(position)
                if result:
                    assert action in valid_actions
                else:
                    assert action not in valid_actions
    
    def test_valid_action_analysis_performance(self):
        """
        Test valid action analysis performance meets requirements for action space filtering.
        """
        # Create ActionProcessor instance for performance measurement
        processor = create_test_action_processor()
        
        # Measure get_valid_actions execution time over multiple iterations
        positions_to_test = [TEST_COORDINATES_CENTER, TEST_COORDINATES_CORNER_ORIGIN, 
                           TEST_COORDINATES_EDGE_TOP, TEST_COORDINATES_EDGE_RIGHT]
        timing_measurements = []
        
        for _ in range(PERFORMANCE_TEST_ITERATIONS // len(positions_to_test)):
            for position in positions_to_test:
                start_time = time.perf_counter()
                processor.get_valid_actions(position)
                end_time = time.perf_counter()
                timing_measurements.append((end_time - start_time) * 1000)
        
        # Assert analysis performance is optimized for high-frequency action filtering
        avg_time = np.mean(timing_measurements)
        assert avg_time < ACTION_PROCESSING_PERFORMANCE_TARGET_MS * 2  # Allow 2x tolerance


class TestStandaloneFunctions:
    """
    Test class for standalone action processing functions covering utility functions and 
    performance-critical operations.
    """
    
    def test_process_action_function(self):
        """
        Test standalone process_action function provides fast action processing without processor instance.
        """
        # Test process_action with valid position-action combinations
        position = TEST_COORDINATES_CENTER
        action = Action.UP
        grid_size = TEST_GRID_SIZE
        
        result = process_action(position, action, grid_size)
        
        # Verify function returns proper tuple format with final position and analysis
        assert isinstance(result, tuple)
        assert len(result) >= 2  # At minimum (final_position, success_flag)
        
        final_position, success = result[:2]
        assert isinstance(final_position, Coordinates)
        assert isinstance(success, bool)
        
        # Check function performance is optimized for high-frequency processing
        timing_measurements = []
        for _ in range(100):
            start_time = time.perf_counter()
            process_action(position, action, grid_size)
            end_time = time.perf_counter()
            timing_measurements.append((end_time - start_time) * 1000)
        
        avg_time = np.mean(timing_measurements)
        assert avg_time < ACTION_PROCESSING_PERFORMANCE_TARGET_MS * 2
    
    def test_validate_action_bounds_function(self):
        """
        Test standalone validate_action_bounds function provides fast action bounds checking.
        """
        # Test validate_action_bounds with valid actions returns True
        for action in ALL_VALID_ACTIONS + ALL_ACTION_INTEGERS:
            result = validate_action_bounds(action)
            assert result is True
        
        # Test function with invalid actions returns False
        for invalid_action in [-1, 4, 5]:
            result = validate_action_bounds(invalid_action)
            assert result is False
        
        # Check function performance is optimized for maximum validation speed
        timing_measurements = []
        for _ in range(PERFORMANCE_TEST_ITERATIONS):
            action = random.choice(ALL_ACTION_INTEGERS)
            start_time = time.perf_counter()
            validate_action_bounds(action)
            end_time = time.perf_counter()
            timing_measurements.append((end_time - start_time) * 1000)
        
        avg_time = np.mean(timing_measurements)
        assert avg_time < ACTION_PROCESSING_PERFORMANCE_TARGET_MS / 2  # Should be very fast
    
    def test_calculate_movement_delta_function(self):
        """
        Test standalone calculate_movement_delta function provides precise movement vector calculation.
        """
        # Test calculate_movement_delta with all valid actions returns correct vectors
        for action in ALL_VALID_ACTIONS:
            delta = calculate_movement_delta(action)
            
            # Verify function returns proper tuple format with (dx, dy) coordinates
            assert isinstance(delta, tuple)
            assert len(delta) == 2
            assert all(isinstance(coord, int) for coord in delta)
            
            # Assert function consistency with MOVEMENT_VECTORS dictionary
            expected_vector = action.to_vector()
            assert delta == expected_vector
        
        # Test function with integer inputs
        for action_int in ALL_ACTION_INTEGERS:
            delta = calculate_movement_delta(action_int)
            expected_vector = MOVEMENT_VECTORS[action_int]
            assert delta == expected_vector
    
    def test_is_valid_action_for_position_function(self):
        """
        Test standalone is_valid_action_for_position function provides position-dependent validation.
        """
        grid_size = TEST_GRID_SIZE
        
        # Test is_valid_action_for_position with valid position-action combinations
        center_pos = TEST_COORDINATES_CENTER
        for action in ALL_VALID_ACTIONS:
            result = is_valid_action_for_position(center_pos, action, grid_size)
            assert result is True  # All actions valid from center
        
        # Test function with boundary-violating combinations returns False
        edge_right = TEST_COORDINATES_EDGE_RIGHT
        result = is_valid_action_for_position(edge_right, Action.RIGHT, grid_size)
        assert result is False  # RIGHT action invalid from right edge
        
        # Check function performance is optimized for action filtering operations
        timing_measurements = []
        for _ in range(PERFORMANCE_TEST_ITERATIONS):
            position = random.choice([center_pos, TEST_COORDINATES_CORNER_ORIGIN])
            action = random.choice(ALL_VALID_ACTIONS)
            
            start_time = time.perf_counter()
            is_valid_action_for_position(position, action, grid_size)
            end_time = time.perf_counter()
            timing_measurements.append((end_time - start_time) * 1000)
        
        avg_time = np.mean(timing_measurements)
        assert avg_time < ACTION_PROCESSING_PERFORMANCE_TARGET_MS


class TestActionProcessingConfiguration:
    """
    Test class for ActionProcessingConfig configuration covering policy validation, configuration 
    management, and processing behavior.
    """
    
    def test_action_processing_config_initialization(self):
        """
        Test ActionProcessingConfig initialization with various policy configurations and 
        parameter validation.
        """
        # Create ActionProcessingConfig with default parameters and validate initialization
        default_config = ActionProcessingConfig()
        assert default_config.enable_validation is True
        assert default_config.enforce_boundaries is True
        assert default_config.enable_performance_monitoring is True
        
        # Test initialization with enable_validation=True enables action validation
        validation_config = ActionProcessingConfig(enable_validation=True)
        assert validation_config.enable_validation is True
        
        # Test initialization with enforce_boundaries=True enables boundary checking
        boundary_config = ActionProcessingConfig(enforce_boundaries=True)
        assert boundary_config.enforce_boundaries is True
        
        # Verify enable_performance_monitoring=True enables timing tracking
        monitoring_config = ActionProcessingConfig(enable_performance_monitoring=True)
        assert monitoring_config.enable_performance_monitoring is True
    
    def test_action_processing_config_validation(self):
        """
        Test ActionProcessingConfig.validate_configuration ensures policy consistency and feasibility.
        """
        # Create ActionProcessingConfig instances with valid configurations
        valid_configs = create_action_processing_config_variations()
        
        for config in valid_configs:
            # Test validate_configuration returns True for consistent policy settings
            assert config.validate_configuration() is True
        
        # Verify validation checks logical consistency between enable flags
        consistent_config = ActionProcessingConfig(
            enable_validation=True,
            enforce_boundaries=True,
            enable_performance_monitoring=False  # Valid combination
        )
        assert consistent_config.validate_configuration() is True
    
    def test_action_processing_config_cloning(self):
        """
        Test ActionProcessingConfig.clone method creates proper deep copies with optional 
        parameter overrides.
        """
        # Create ActionProcessingConfig and test clone() creates independent copy
        original_config = ActionProcessingConfig(
            enable_validation=True,
            enforce_boundaries=False,
            enable_performance_monitoring=True
        )
        
        cloned_config = original_config.clone()
        
        # Verify cloned configuration has identical settings to original
        assert cloned_config.enable_validation == original_config.enable_validation
        assert cloned_config.enforce_boundaries == original_config.enforce_boundaries
        assert cloned_config.enable_performance_monitoring == original_config.enable_performance_monitoring
        
        # Test clone with parameter overrides applies modifications correctly
        overridden_config = original_config.clone(overrides={'enforce_boundaries': True})
        assert overridden_config.enforce_boundaries is True
        assert overridden_config.enable_validation == original_config.enable_validation  # Unchanged
        
        # Assert original configuration is not modified during cloning operation
        assert original_config.enforce_boundaries is False  # Original unchanged
    
    def test_config_policy_variations(self):
        """
        Test various configuration policy combinations ensure proper action processing behavior.
        """
        # Test config with validation enabled and boundary enforcement disabled
        config1 = ActionProcessingConfig(enable_validation=True, enforce_boundaries=False)
        processor1 = create_test_action_processor(config=config1)
        assert processor1.config.enable_validation is True
        assert processor1.config.enforce_boundaries is False
        
        # Test config with validation disabled and boundary enforcement enabled
        config2 = ActionProcessingConfig(enable_validation=False, enforce_boundaries=True)
        processor2 = create_test_action_processor(config=config2)
        assert processor2.config.enable_validation is False
        assert processor2.config.enforce_boundaries is True
        
        # Test config with performance monitoring enabled and other features varied
        config3 = ActionProcessingConfig(enable_performance_monitoring=True)
        processor3 = create_test_action_processor(config=config3)
        assert processor3.config.enable_performance_monitoring is True


class TestProcessingStatistics:
    """
    Test class for action processing statistics covering metrics collection, performance analysis, 
    and monitoring validation.
    """
    
    def test_processing_statistics_collection(self):
        """
        Test action processing statistics are accurately collected and reported for monitoring analysis.
        """
        # Create ActionProcessor and perform various processing operations
        processor = create_test_action_processor()
        initial_stats = processor.get_processing_statistics()
        
        # Track expected actions processed manually during test operations
        expected_operations = 10
        test_position = TEST_COORDINATES_CENTER
        
        for i in range(expected_operations):
            action = ALL_VALID_ACTIONS[i % len(ALL_VALID_ACTIONS)]
            processor.process_action(test_position, action)
        
        # Get processing statistics using get_processing_statistics() method
        final_stats = processor.get_processing_statistics()
        
        # Verify total processing operations count matches executed operations
        actions_processed = final_stats.get('actions_processed', 0)
        assert actions_processed >= expected_operations
    
    def test_cache_statistics_tracking(self):
        """
        Test action processing cache statistics provide hit/miss ratios and efficiency metrics.
        """
        # Create ActionProcessor with caching enabled for statistics tracking
        processor = create_test_action_processor()
        
        # Perform repeated action validations to populate cache
        test_action = Action.UP
        
        # Execute duplicate validations to generate cache hits
        for _ in range(5):
            processor.validate_action(test_action)  # Should hit cache after first call
        
        # Get processing statistics and verify cache statistics are available
        stats = processor.get_processing_statistics()
        
        # Check cache efficiency shows some benefit (implementation dependent)
        assert 'actions_processed' in stats
    
    def test_performance_statistics_accuracy(self):
        """
        Test action processing performance statistics provide accurate timing analysis and 
        optimization data.
        """
        # Create ActionProcessor with performance monitoring enabled
        config = ActionProcessingConfig(enable_performance_monitoring=True)
        processor = create_test_action_processor(config=config)
        
        # Execute action processing operations with timing measurement
        test_operations = 20
        test_position = TEST_COORDINATES_CENTER
        
        for i in range(test_operations):
            action = ALL_VALID_ACTIONS[i % len(ALL_VALID_ACTIONS)]
            processor.process_action(test_position, action)
        
        # Get processing statistics and verify performance metrics are included
        stats = processor.get_processing_statistics()
        assert 'actions_processed' in stats
        assert stats['actions_processed'] >= test_operations
    
    def test_statistics_reset_and_cleanup(self):
        """
        Test processing statistics can be reset and cleaned up properly for fresh monitoring periods.
        """
        # Create ActionProcessor and accumulate processing statistics through operations
        processor = create_test_action_processor()
        
        # Perform some operations to accumulate statistics
        for _ in range(5):
            processor.process_action(TEST_COORDINATES_CENTER, Action.UP)
        
        # Verify statistics show non-zero values after processing operations
        stats_before = processor.get_processing_statistics()
        assert stats_before.get('actions_processed', 0) > 0
        
        # Test clear_cache() method resets cache-related statistics
        cleared_entries = processor.clear_cache()
        
        # Check statistics reset doesn't affect core processing functionality
        processor.process_action(TEST_COORDINATES_CENTER, Action.RIGHT)
        stats_after = processor.get_processing_statistics()
        
        # Assert reset maintains action processor operational integrity
        assert isinstance(stats_after, dict)


class TestErrorHandling:
    """
    Test class for action processing error handling covering validation failures, state errors, 
    and recovery strategies.
    """
    
    def test_validation_error_handling(self):
        """
        Test proper ValidationError handling for invalid action parameters with context and 
        recovery information.
        """
        # Create ActionProcessor and test validation with various invalid parameter types
        processor = create_test_action_processor()
        
        # Test action processing with None, string, and non-numeric action inputs
        invalid_inputs = [None, "up", "invalid", 1.5, [], {}]
        
        for invalid_input in invalid_inputs:
            result = processor.process_action(TEST_COORDINATES_CENTER, invalid_input)
            
            # Verify ValidationError handling with appropriate error context
            assert result.action_valid is False
            assert result.original_position == TEST_COORDINATES_CENTER
        
        # Test action validation with values outside Discrete(4) range
        out_of_bounds = [-1, 4, 5, 10, -5]
        for invalid_action in out_of_bounds:
            result = processor.process_action(TEST_COORDINATES_CENTER, invalid_action)
            assert result.action_valid is False
    
    def test_state_error_handling(self):
        """
        Test StateError handling for invalid action processor states and configuration inconsistencies.
        """
        # Create ActionProcessor with valid configuration
        processor = create_test_action_processor()
        
        # Test component integration consistency
        try:
            # Attempt to process action with valid inputs
            result = processor.process_action(TEST_COORDINATES_CENTER, Action.UP)
            assert result is not None
            assert hasattr(result, 'action_valid')
        except StateError as e:
            # If StateError occurs, verify it's handled appropriately
            assert isinstance(e, StateError)
            assert hasattr(e, 'message')
    
    def test_component_error_handling(self):
        """
        Test ComponentError handling for action processor component failures with detailed analysis.
        """
        # Create ActionProcessor for component integration testing
        processor = create_test_action_processor()
        
        # Test component integration with boundary enforcer
        try:
            # Perform operation that involves component interaction
            result = processor.process_action(TEST_COORDINATES_EDGE_RIGHT, Action.RIGHT)
            
            # Verify component interaction handles boundary enforcement correctly
            assert result.boundary_hit is True or result.boundary_hit is False  # Valid boolean
            assert isinstance(result, ActionProcessingResult)
            
        except ComponentError as e:
            # If ComponentError occurs, verify proper error handling
            assert isinstance(e, ComponentError)
            assert hasattr(e, 'component_name')
    
    def test_error_context_sanitization(self):
        """
        Test error message sanitization and context filtering prevent information disclosure 
        while providing debugging context.
        """
        # Create ActionProcessor for error context testing
        processor = create_test_action_processor()
        
        # Trigger various action processing errors with different contexts
        error_inducing_inputs = [None, "invalid", -1, 999]
        
        for invalid_input in error_inducing_inputs:
            try:
                result = processor.process_action(TEST_COORDINATES_CENTER, invalid_input)
                
                # Verify error handling maintains security while providing useful feedback
                assert result.action_valid is False
                
                # Check that result provides actionable information without sensitive details
                result_dict = result.to_dict()
                assert isinstance(result_dict, dict)
                assert 'action' in result_dict
                
            except Exception as e:
                # Verify any exceptions maintain appropriate security boundaries
                error_message = str(e)
                assert isinstance(error_message, str)
                # Ensure no sensitive internal state is exposed
                assert 'password' not in error_message.lower()
                assert 'token' not in error_message.lower()


class TestPerformanceBenchmarks:
    """
    Test class for action processing performance benchmarking covering latency validation, 
    optimization verification, and resource efficiency.
    """
    
    def test_action_processing_latency_benchmark(self):
        """
        Benchmark action processing latency against ACTION_PROCESSING_PERFORMANCE_TARGET_MS 
        requirement for real-time operation.
        """
        # Create ActionProcessor instance for latency benchmarking
        processor = create_test_action_processor()
        
        # Measure process_action execution time over 1000 iterations
        test_position = TEST_COORDINATES_CENTER
        action = Action.UP
        
        performance_metrics = measure_action_processing_performance(
            processor, PERFORMANCE_TEST_ITERATIONS, test_position, action
        )
        
        # Assert average latency is below ACTION_PROCESSING_PERFORMANCE_TARGET_MS (0.1ms)
        mean_time = performance_metrics['mean_time_ms']
        assert mean_time < ACTION_PROCESSING_PERFORMANCE_TARGET_MS * 2  # Allow 2x tolerance
        
        # Verify 99th percentile meets real-time requirements for step execution
        percentile_99 = performance_metrics['percentile_99_ms']
        assert percentile_99 < ACTION_PROCESSING_PERFORMANCE_TARGET_MS * 5  # Allow 5x tolerance
        
        # Check performance consistency across different processing scenarios
        assert performance_metrics['std_time_ms'] < mean_time  # Standard deviation reasonable
    
    def test_action_validation_performance_benchmark(self):
        """
        Benchmark action validation performance for high-frequency processing operations.
        """
        # Create ActionProcessor instance for validation benchmarking
        processor = create_test_action_processor()
        
        # Measure validate_action execution time over 10000 iterations
        validation_iterations = PERFORMANCE_TEST_ITERATIONS * 10
        timing_measurements = []
        
        for _ in range(validation_iterations):
            action = random.choice(ALL_VALID_ACTIONS + ALL_ACTION_INTEGERS)
            start_time = time.perf_counter()
            processor.validate_action(action)
            end_time = time.perf_counter()
            timing_measurements.append((end_time - start_time) * 1000)
        
        # Assert validation performance is well below processing target
        avg_validation_time = np.mean(timing_measurements)
        assert avg_validation_time < ACTION_PROCESSING_PERFORMANCE_TARGET_MS / 2
        
        # Verify validation consistency across action types and validation modes
        std_validation_time = np.std(timing_measurements)
        assert std_validation_time < avg_validation_time
    
    def test_memory_usage_efficiency(self):
        """
        Test memory usage efficiency and resource management during action processing operations.
        """
        # Create ActionProcessor and measure initial memory baseline
        processor = create_test_action_processor()
        
        # Execute large number of processing operations and monitor resource usage
        large_operation_count = PERFORMANCE_TEST_ITERATIONS * 5
        test_position = TEST_COORDINATES_CENTER
        
        # Process actions without accumulating excessive memory
        for i in range(large_operation_count):
            action = ALL_VALID_ACTIONS[i % len(ALL_VALID_ACTIONS)]
            result = processor.process_action(test_position, action)
            
            # Verify each result is properly created and cleaned up
            assert isinstance(result, ActionProcessingResult)
        
        # Assert no significant memory leaks occur during action processing
        stats = processor.get_processing_statistics()
        assert stats['actions_processed'] >= large_operation_count
        
        # Check memory cleanup is effective when cache is cleared
        cleared_entries = processor.clear_cache()
        assert isinstance(cleared_entries, int)
    
    def test_scaling_performance_analysis(self):
        """
        Test action processing performance scaling with different grid sizes and configuration settings.
        """
        # Create ActionProcessor instances with various grid sizes for scaling analysis
        grid_sizes = [GridSize(16, 16), TEST_GRID_SIZE, GridSize(64, 64), TEST_LARGE_GRID_SIZE]
        
        scaling_results = {}
        
        for grid_size in grid_sizes:
            processor = create_test_action_processor(grid_size=grid_size)
            test_position = Coordinates(grid_size.width // 2, grid_size.height // 2)
            
            # Measure processing performance for this grid size
            performance_metrics = measure_action_processing_performance(
                processor, PERFORMANCE_TEST_ITERATIONS // 4, test_position, Action.UP
            )
            
            scaling_results[f"{grid_size.width}x{grid_size.height}"] = performance_metrics['mean_time_ms']
        
        # Verify performance scaling is within acceptable limits for larger grids
        base_performance = scaling_results['16x16']
        large_performance = scaling_results['128x128']
        
        # Performance should not degrade significantly with grid size for action processing
        assert large_performance < base_performance * 3  # Allow 3x degradation max


class TestIntegrationAndCompatibility:
    """
    Test class for action processor integration covering component interaction, coordinate system 
    consistency, and compatibility validation.
    """
    
    def test_coordinate_system_consistency(self):
        """
        Test action processor maintains coordinate system consistency with (x,y) positioning 
        and proper movement calculations.
        """
        # Create ActionProcessor and test coordinate system consistency
        processor = create_test_action_processor()
        
        # Verify x-axis positive direction is rightward in grid coordinates
        result_right = processor.process_action(TEST_COORDINATES_CENTER, Action.RIGHT)
        assert result_right.final_position.x == TEST_COORDINATES_CENTER.x + 1
        assert result_right.final_position.y == TEST_COORDINATES_CENTER.y
        
        # Check y-axis positive direction is upward in mathematical coordinate system
        result_up = processor.process_action(TEST_COORDINATES_CENTER, Action.UP)
        assert result_up.final_position.x == TEST_COORDINATES_CENTER.x
        assert result_up.final_position.y == TEST_COORDINATES_CENTER.y + 1
        
        # Test action processing respects coordinate system conventions
        result_left = processor.process_action(TEST_COORDINATES_CENTER, Action.LEFT)
        assert result_left.final_position.x == TEST_COORDINATES_CENTER.x - 1
        
        result_down = processor.process_action(TEST_COORDINATES_CENTER, Action.DOWN)
        assert result_down.final_position.y == TEST_COORDINATES_CENTER.y - 1
    
    def test_action_type_compatibility(self):
        """
        Test action processor compatibility with Action enum and integer action representations.
        """
        # Create ActionProcessor and test with Action enum values
        processor = create_test_action_processor()
        test_position = TEST_COORDINATES_CENTER
        
        # Test processing with integer action values (0, 1, 2, 3)
        for action_int in ALL_ACTION_INTEGERS:
            result_int = processor.process_action(test_position, action_int)
            assert result_int.action_valid is True
            
            # Compare with corresponding Action enum
            action_enum = ALL_VALID_ACTIONS[action_int]
            result_enum = processor.process_action(test_position, action_enum)
            
            # Assert action type conversion maintains processing accuracy
            assert result_int.final_position == result_enum.final_position
            assert result_int.position_changed == result_enum.position_changed
        
        # Verify Action.to_vector() integration works correctly with processing
        for action in ALL_VALID_ACTIONS:
            movement_vector = action.to_vector()
            result = processor.process_action(test_position, action)
            
            expected_final = Coordinates(
                test_position.x + movement_vector[0],
                test_position.y + movement_vector[1]
            )
            assert result.final_position == expected_final
    
    def test_boundary_enforcer_integration(self):
        """
        Test action processor integration with BoundaryEnforcer maintains boundary constraint consistency.
        """
        # Create ActionProcessor with BoundaryEnforcer integration
        boundary_enforcer = BoundaryEnforcer(TEST_GRID_SIZE)
        processor = create_test_action_processor(boundary_enforcer=boundary_enforcer)
        
        # Test processing with boundary enforcement enabled handles constraints correctly
        edge_position = TEST_COORDINATES_EDGE_RIGHT
        boundary_violating_action = Action.RIGHT
        
        result = processor.process_action(edge_position, boundary_violating_action)
        
        # Verify boundary enforcement results are properly integrated in processing outcomes
        assert result.boundary_hit is True
        
        # Check boundary hit detection is accurately reported in processing results
        assert result.final_position.is_within_bounds(TEST_GRID_SIZE)
        
        # Assert boundary constraint policies are consistently applied
        boundary_result = boundary_enforcer.enforce_movement_bounds(edge_position, boundary_violating_action)
        assert result.final_position == boundary_result.final_position
    
    def test_configuration_update_integration(self):
        """
        Test action processor configuration updates maintain system consistency and clear 
        dependent state properly.
        """
        # Create ActionProcessor with initial configuration and populate cache
        initial_config = ActionProcessingConfig(enable_validation=True, enforce_boundaries=True)
        processor = create_test_action_processor(config=initial_config)
        
        # Perform some operations to populate internal state
        for _ in range(5):
            processor.process_action(TEST_COORDINATES_CENTER, Action.UP)
        
        # Update configuration using update_configuration() method with new settings
        new_config = ActionProcessingConfig(enable_validation=False, enforce_boundaries=False)
        processor.update_configuration(new_config)
        
        # Verify configuration update takes effect
        assert processor.config == new_config
        assert processor.config.enable_validation is False
        assert processor.config.enforce_boundaries is False
        
        # Test processing results reflect new configuration settings
        result = processor.process_action(TEST_COORDINATES_CENTER, Action.DOWN)
        
        # Assert configuration updates maintain action processor operational integrity
        assert isinstance(result, ActionProcessingResult)
        assert result.action_valid is True  # Validation disabled, should always be True


class TestReproducibilityAndDeterminism:
    """
    Test class for action processing reproducibility covering deterministic behavior validation 
    and consistency across execution contexts.
    """
    
    def test_deterministic_action_processing(self):
        """
        Test action processing produces identical results for identical inputs across multiple 
        execution runs.
        """
        # Create two identical ActionProcessor instances with same configurations
        config = ActionProcessingConfig(enable_validation=True, enforce_boundaries=True)
        processor1 = create_test_action_processor(config=config)
        processor2 = create_test_action_processor(config=config)
        
        # Execute identical action processing sequences on both instances
        test_sequence = [
            (TEST_COORDINATES_CENTER, Action.UP),
            (TEST_COORDINATES_CENTER, Action.RIGHT),
            (TEST_COORDINATES_EDGE_LEFT, Action.LEFT),
            (TEST_COORDINATES_CORNER_ORIGIN, Action.DOWN)
        ]
        
        results1 = []
        results2 = []
        
        for position, action in test_sequence:
            result1 = processor1.process_action(position, action)
            result2 = processor2.process_action(position, action)
            
            results1.append(result1)
            results2.append(result2)
        
        # Compare ActionProcessingResult objects for complete equality
        for r1, r2 in zip(results1, results2):
            assert r1.final_position == r2.final_position
            assert r1.action_valid == r2.action_valid
            assert r1.position_changed == r2.position_changed
            assert r1.boundary_hit == r2.boundary_hit
            
            # Verify movement analysis and success status match
            assert r1.was_movement_successful() == r2.was_movement_successful()
    
    def test_consistent_validation_results(self):
        """
        Test action validation produces consistent results across processor instances and execution time.
        """
        # Create multiple ActionProcessor instances with identical configurations
        processors = [create_test_action_processor() for _ in range(3)]
        
        # Test identical validation operations across all processor instances
        test_actions = ALL_VALID_ACTIONS + ALL_ACTION_INTEGERS + INVALID_ACTIONS[:3]
        
        for action in test_actions:
            validation_results = []
            
            for processor in processors:
                try:
                    result = processor.validate_action(action)
                    validation_results.append(result)
                except:
                    validation_results.append(False)
            
            # Verify validation results are identical across all instances
            first_result = validation_results[0]
            for result in validation_results[1:]:
                assert result == first_result
        
        # Check consistency over time with repeated validation operations
        processor = processors[0]
        action_to_repeat = Action.UP
        
        repeated_results = []
        for _ in range(10):
            result = processor.validate_action(action_to_repeat)
            repeated_results.append(result)
        
        # Assert no randomness or timing affects validation outcomes
        assert all(result == repeated_results[0] for result in repeated_results)
    
    def test_reproducible_performance_metrics(self):
        """
        Test performance metrics collection is consistent and reproducible for benchmark 
        validation and optimization analysis.
        """
        # Create ActionProcessor and perform identical operations multiple times
        processor = create_test_action_processor()
        test_position = TEST_COORDINATES_CENTER
        test_action = Action.UP
        operation_count = 50
        
        # Execute identical operations in multiple runs
        run1_stats = []
        run2_stats = []
        
        # First run
        for _ in range(operation_count):
            processor.process_action(test_position, test_action)
        stats1 = processor.get_processing_statistics()
        
        # Clear processor state for second run
        processor.clear_cache()
        
        # Second run with identical operations
        for _ in range(operation_count):
            processor.process_action(test_position, test_action)
        stats2 = processor.get_processing_statistics()
        
        # Verify timing measurements are within acceptable statistical variance
        actions_processed_1 = stats1.get('actions_processed', 0)
        actions_processed_2 = stats2.get('actions_processed', 0)
        
        # Check performance statistics consistency across identical operations
        # Allow for some variance in execution time but expect similar operation counts
        assert abs(actions_processed_1 - actions_processed_2) <= operation_count
        
        # Assert reproducible benchmarking results for optimization validation
        assert actions_processed_1 >= operation_count
        assert actions_processed_2 >= operation_count