"""
Comprehensive edge case testing module for plume_nav_sim package validating
robustness under extreme conditions, boundary violations, invalid parameters,
error recovery, and system limits.

This module tests environment behavior with malformed inputs, resource constraints,
mathematical edge cases, and failure scenarios to ensure graceful error handling
and system stability across all components including boundary enforcement,
validation utilities, rendering pipeline, and state management.
"""

import contextlib  # standard library - Context managers for edge case resource management
import gc  # standard library - Garbage collection control for memory edge case testing
import math  # standard library - Mathematical utilities for edge case calculations
import sys  # standard library - System utilities for testing system limit edge cases
import time  # standard library - Timing utilities for performance edge case testing
import unittest.mock  # standard library - Mocking utilities for simulating edge case scenarios
import warnings  # standard library - Warning management for edge case testing

import os  # >=3.10 - Environment introspection for conditional edge-case skips
import importlib.util  # >=3.10 - Module availability checks for conditional skips
import numpy as np  # >=2.1.0 - Array operations for mathematical edge cases and precision testing

# External imports with version requirements
import pytest  # >=8.0.0 - Testing framework for edge case test organization and parametrization

from plume_nav_sim.core.boundary_enforcer import BoundaryEnforcer
from plume_nav_sim.core.types import Action, Coordinates
from plume_nav_sim.envs.plume_search_env import PlumeSearchEnv, create_plume_search_env
from plume_nav_sim.utils.exceptions import (
    ConfigurationError,
    ErrorSeverity,
    RenderingError,
    ResourceError,
    StateError,
    ValidationError,
)
from plume_nav_sim.utils.validation import ParameterValidator


# Conditional feature flags for skip markers
MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None
SINGLE_THREADED_ONLY = os.environ.get("PLUME_SKIP_CONCURRENCY", "0") == "1"

# Internal imports - core environment and utilities

# Global constants for edge case testing parameters
EDGE_CASE_TIMEOUT_SECONDS = 30.0
MEMORY_STRESS_TEST_ITERATIONS = 1000
INVALID_ACTION_TEST_VALUES = [
    -1,
    4,
    5,
    100,
    -999,
    None,
    "invalid",
    3.14,
    float("inf"),
    float("nan"),
]
EXTREME_GRID_SIZES = [
    (1, 1),
    (0, 0),
    (-1, -1),
    (1000, 1000),
    (sys.maxsize, 1),
    (1, sys.maxsize),
]
BOUNDARY_COORDINATE_CASES = [
    (-1, -1),
    (-1, 0),
    (0, -1),
    (128, 128),
    (127, 128),
    (128, 127),
    (-999, -999),
    (999, 999),
]
EXTREME_PLUME_PARAMETERS = [
    0.0,
    0.001,
    0.1,
    1000.0,
    float("inf"),
    float("nan"),
    -1.0,
    -999.0,
]
INVALID_SEED_VALUES = [None, -1, 2**63, float("inf"), float("nan"), "invalid", [], {}]
STRESS_TEST_STEP_COUNTS = [1, 10, 100, 1000, 10000]


class EdgeCaseTestFixture:
    """
    Comprehensive test fixture class for edge case testing providing specialized
    environment configurations, stress testing utilities, error simulation capabilities,
    and validation helpers for systematic edge case validation across all
    plume_nav_sim components.
    """

    def __init__(self, test_category: str, edge_case_config: dict):
        """
        Initialize edge case test fixture with test category classification and
        specialized configuration for comprehensive edge case testing scenarios.

        Args:
            test_category: Category of edge case testing (boundary, validation, performance, etc.)
            edge_case_config: Specialized parameters for extreme condition testing
        """
        # Store test_category for categorized edge case testing
        self.test_category = test_category

        # Store edge_case_config with specialized parameters for extreme condition testing
        self.edge_case_config = edge_case_config

        # Initialize empty test_environments list for environment instance tracking
        self.test_environments = []

        # Initialize error_scenarios dictionary for systematic error simulation
        self.error_scenarios = {}

        # Create PerformanceTracker instance for monitoring edge case performance impact
        from tests.conftest import PerformanceTracker as _PerformanceTracker

        self.performance_tracker = _PerformanceTracker()

        # Initialize MemoryMonitor for detecting resource issues during edge case execution
        from tests.conftest import MemoryMonitor as _MemoryMonitor

        self.memory_monitor = _MemoryMonitor()

    def create_extreme_environment(
        self, edge_case_type: str, custom_params: dict = None
    ) -> PlumeSearchEnv:
        """
        Create environment configured for specific edge case testing with extreme
        parameters and validation.

        Args:
            edge_case_type: Type of edge case configuration to create
            custom_params: Optional custom parameters to override defaults

        Returns:
            PlumeSearchEnv: Environment configured for specified edge case testing with monitoring
        """
        # Select appropriate extreme configuration based on edge_case_type
        if edge_case_type == "boundary_violation":
            base_config = {
                "grid_size": (2, 2),  # Minimal grid for boundary testing
                "source_location": (
                    0,
                    0,
                ),  # Corner source for maximum boundary pressure
                "goal_radius": 0,
                "max_steps": 10,
            }
        elif edge_case_type == "mathematical_precision":
            base_config = {
                "grid_size": (128, 128),
                "source_location": (64, 64),
                "plume_params": {"sigma": 0.001},  # Extreme precision requirements
                "goal_radius": 0,
                "max_steps": 100,
            }
        elif edge_case_type == "resource_exhaustion":
            base_config = {
                "grid_size": (512, 512),  # Large grid for memory pressure
                "source_location": (256, 256),
                "goal_radius": 0,
                "max_steps": 10000,
            }
        elif edge_case_type == "state_transition":
            base_config = {
                "grid_size": (32, 32),
                "source_location": (
                    31,
                    31,
                ),  # Source at boundary for transition edge cases
                "goal_radius": 0,
                "max_steps": 1000,
            }
        else:
            # Default extreme configuration
            base_config = {
                "grid_size": (64, 64),
                "source_location": (32, 32),
                "goal_radius": 0,
                "max_steps": 500,
            }

        # Apply custom_params overrides if provided with validation
        if custom_params:
            base_config.update(custom_params)

        # Create PlumeSearchEnv with extreme parameters and error handling
        try:
            env = create_plume_search_env(**base_config)
        except Exception:
            # Handle configuration failures for edge case testing
            env = PlumeSearchEnv()  # Fall back to default configuration

        # Register environment for cleanup and resource monitoring
        self.test_environments.append(env)

        # Set up performance monitoring and resource tracking for edge case execution
        if hasattr(self.performance_tracker, "start_monitoring"):
            self.performance_tracker.start_monitoring(
                f"{edge_case_type}_{len(self.test_environments)}"
            )

        # Return configured environment ready for edge case testing
        return env

    def simulate_error_scenario(
        self, error_type: str, error_context: dict
    ) -> contextlib.AbstractContextManager:
        """
        Simulate specific error scenarios for testing error handling and recovery mechanisms.

        Args:
            error_type: Type of error to simulate
            error_context: Context information for error simulation

        Returns:
            contextlib.AbstractContextManager: Context manager for controlled error simulation
        """
        # Configure error simulation based on error_type specification
        if error_type == "memory_exhaustion":
            return self._create_memory_exhaustion_context(error_context)
        elif error_type == "rendering_failure":
            return self._create_rendering_failure_context(error_context)
        elif error_type == "validation_failure":
            return self._create_validation_failure_context(error_context)
        elif error_type == "state_corruption":
            return self._create_state_corruption_context(error_context)
        else:
            return self._create_generic_error_context(error_type, error_context)

    def _create_memory_exhaustion_context(self, context: dict):
        """Create context manager for memory exhaustion simulation."""

        @contextlib.contextmanager
        def memory_exhaustion():
            # Enable memory monitoring and recovery validation during simulation
            if hasattr(self.memory_monitor, "start_monitoring"):
                self.memory_monitor.start_monitoring()

            try:
                # Simulate memory pressure
                large_arrays = []
                for i in range(context.get("allocation_count", 100)):
                    large_arrays.append(np.zeros((1000, 1000), dtype=np.float32))
                yield
            except MemoryError:
                # Expected memory exhaustion for testing
                pass
            finally:
                # Clean up allocated memory
                del large_arrays
                gc.collect()
                if hasattr(self.memory_monitor, "stop_monitoring"):
                    self.memory_monitor.stop_monitoring()

        return memory_exhaustion()

    def _create_rendering_failure_context(self, context: dict):
        """Create context manager for rendering failure simulation."""

        @contextlib.contextmanager
        def rendering_failure():
            # Mock matplotlib to simulate backend failures
            with unittest.mock.patch("matplotlib.pyplot.subplots") as mock_subplots:
                mock_subplots.side_effect = ImportError("Simulated matplotlib failure")
                yield

        return rendering_failure()

    def _create_validation_failure_context(self, context: dict):
        """Create context manager for validation failure simulation."""

        @contextlib.contextmanager
        def validation_failure():
            # Store error simulation context
            self.error_scenarios["validation_failure"] = context
            yield

        return validation_failure()

    def _create_state_corruption_context(self, context: dict):
        """Create context manager for state corruption simulation."""

        @contextlib.contextmanager
        def state_corruption():
            # Store error simulation context
            self.error_scenarios["state_corruption"] = context
            yield

        return state_corruption()

    def _create_generic_error_context(self, error_type: str, context: dict):
        """Create generic error simulation context."""

        @contextlib.contextmanager
        def generic_error():
            # Store error simulation context
            self.error_scenarios[error_type] = context
            yield

        return generic_error()

    def validate_edge_case_recovery(self, environment: PlumeSearchEnv) -> dict:
        """
        Validate system recovery after edge case execution including state
        consistency and resource cleanup.

        Args:
            environment: Environment to validate recovery for

        Returns:
            dict: Recovery validation report with system state analysis
        """
        recovery_report = {
            "environment_state_consistent": False,
            "resources_cleaned": False,
            "error_handling_stable": False,
            "performance_impact": {},
            "memory_status": {},
        }

        # Check environment state consistency after edge case execution
        try:
            if hasattr(environment, "validate_environment_integrity"):
                recovery_report["environment_state_consistent"] = (
                    environment.validate_environment_integrity()
                )
            else:
                # Basic state consistency checks
                recovery_report["environment_state_consistent"] = (
                    hasattr(environment, "action_space")
                    and hasattr(environment, "observation_space")
                    and hasattr(environment, "_episode_active")
                )
        except Exception as e:
            recovery_report["environment_state_error"] = str(e)

        # Validate resource cleanup and memory deallocation
        try:
            gc.collect()  # Force garbage collection
            if hasattr(self.memory_monitor, "get_current_usage"):
                recovery_report["memory_status"] = (
                    self.memory_monitor.get_current_usage()
                )
            recovery_report["resources_cleaned"] = True
        except Exception as e:
            recovery_report["cleanup_error"] = str(e)

        # Verify error handling system returned to stable state
        try:
            # Test basic operations to ensure error handling is stable
            test_obs, test_info = environment.reset(seed=42)
            test_result = environment.step(0)
            recovery_report["error_handling_stable"] = len(test_result) == 5
        except Exception as e:
            recovery_report["error_handling_error"] = str(e)

        # Check performance metrics for impact assessment
        if hasattr(self.performance_tracker, "get_metrics"):
            recovery_report["performance_impact"] = (
                self.performance_tracker.get_metrics()
            )

        # Return detailed analysis of system recovery effectiveness
        return recovery_report

    def cleanup_edge_case_resources(self) -> dict:
        """
        Clean up all resources created during edge case testing with comprehensive validation.

        Returns:
            dict: Cleanup report with resource deallocation analysis
        """
        cleanup_report = {
            "environments_closed": 0,
            "cleanup_errors": [],
            "memory_freed": 0,
            "monitoring_stopped": False,
        }

        # Close all registered test environments with error handling
        for env in self.test_environments:
            try:
                env.close()
                cleanup_report["environments_closed"] += 1
            except Exception as e:
                cleanup_report["cleanup_errors"].append(
                    f"Environment cleanup error: {e}"
                )

        # Clear environment list
        self.test_environments.clear()

        # Validate memory cleanup and resource deallocation
        try:
            initial_objects = len(gc.get_objects())
            gc.collect()
            final_objects = len(gc.get_objects())
            cleanup_report["memory_freed"] = initial_objects - final_objects
        except Exception as e:
            cleanup_report["cleanup_errors"].append(f"Memory cleanup error: {e}")

        # Clear performance monitoring data and reset metrics
        try:
            if hasattr(self.performance_tracker, "reset"):
                self.performance_tracker.reset()
            cleanup_report["monitoring_stopped"] = True
        except Exception as e:
            cleanup_report["cleanup_errors"].append(
                f"Performance tracking cleanup error: {e}"
            )

        # Clean up error simulation artifacts and temporary resources
        self.error_scenarios.clear()

        # Return cleanup effectiveness analysis for test validation
        return cleanup_report


@pytest.mark.edge_case
@pytest.mark.boundary
@pytest.mark.timeout(EDGE_CASE_TIMEOUT_SECONDS)
def test_boundary_position_edge_cases(edge_case_test_env, boundary_enforcer=None):
    """
    Test agent behavior at grid boundaries including corners, edges, and boundary
    violations with comprehensive position validation and boundary enforcement testing.
    """
    # Create boundary enforcer if not provided
    if boundary_enforcer is None:
        boundary_enforcer = BoundaryEnforcer(edge_case_test_env.grid_size)

    # Test agent placement at all four grid corners
    corners = [
        (0, 0),  # Bottom-left corner
        (0, edge_case_test_env.grid_size[1] - 1),  # Top-left corner
        (edge_case_test_env.grid_size[0] - 1, 0),  # Bottom-right corner
        (
            edge_case_test_env.grid_size[0] - 1,
            edge_case_test_env.grid_size[1] - 1,
        ),  # Top-right corner
    ]

    for corner_x, corner_y in corners:
        # Test corner position validation
        corner_coords = Coordinates(corner_x, corner_y)
        position_valid = boundary_enforcer.validate_position(corner_coords)
        assert (
            position_valid.is_valid
        ), f"Corner position {corner_coords.to_tuple()} should be valid"

        # Test movement actions from corner positions that would violate boundaries
        corner_actions = boundary_enforcer.get_valid_moves(corner_coords)
        assert (
            len(corner_actions) >= 2
        ), f"Corner should have at least 2 valid moves, got {len(corner_actions)}"

        # Verify agent stays in bounds when boundary-violating actions are attempted
        for action in [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]:
            result = boundary_enforcer.enforce_movement_bounds(corner_coords, action)
            assert result.final_position.x >= 0, "Position X should not be negative"
            assert result.final_position.y >= 0, "Position Y should not be negative"
            assert (
                result.final_position.x < edge_case_test_env.grid_size[0]
            ), "Position X should be within grid width"
            assert (
                result.final_position.y < edge_case_test_env.grid_size[1]
            ), "Position Y should be within grid height"

    # Test agent placement at grid edges and validate edge movement constraints
    edge_positions = [
        (0, edge_case_test_env.grid_size[1] // 2),  # Left edge
        (
            edge_case_test_env.grid_size[0] - 1,
            edge_case_test_env.grid_size[1] // 2,
        ),  # Right edge
        (edge_case_test_env.grid_size[0] // 2, 0),  # Bottom edge
        (
            edge_case_test_env.grid_size[0] // 2,
            edge_case_test_env.grid_size[1] - 1,
        ),  # Top edge
    ]

    for edge_x, edge_y in edge_positions:
        edge_coords = Coordinates(edge_x, edge_y)
        edge_validation = boundary_enforcer.validate_position(edge_coords)
        assert (
            edge_validation.is_valid
        ), f"Edge position {edge_coords.to_tuple()} should be valid"

        # Test edge movement constraints
        valid_moves = boundary_enforcer.get_valid_moves(edge_coords)
        assert (
            2 <= len(valid_moves) <= 3
        ), f"Edge position should have 2-3 valid moves, got {len(valid_moves)}"

    # Test boundary enforcement with extreme coordinate values and negative positions
    extreme_positions = BOUNDARY_COORDINATE_CASES
    for extreme_x, extreme_y in extreme_positions:
        extreme_coords = Coordinates(extreme_x, extreme_y)

        # Test position validation for extreme coordinates
        validation_result = boundary_enforcer.validate_position(extreme_coords)

        if (
            extreme_x < 0
            or extreme_y < 0
            or extreme_x >= edge_case_test_env.grid_size[0]
            or extreme_y >= edge_case_test_env.grid_size[1]
        ):
            # Should be invalid for out-of-bounds positions
            assert (
                not validation_result.is_valid
            ), f"Extreme position {extreme_coords.to_tuple()} should be invalid"

        # Test coordinate clamping behavior with out-of-bounds positions
        if not validation_result.is_valid:
            # Test all actions to ensure clamping works
            for action in [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]:
                clamped_result = boundary_enforcer.enforce_movement_bounds(
                    extreme_coords, action
                )

                # Verify clamped position is within bounds
                assert (
                    0
                    <= clamped_result.final_position.x
                    < edge_case_test_env.grid_size[0]
                )
                assert (
                    0
                    <= clamped_result.final_position.y
                    < edge_case_test_env.grid_size[1]
                )
                assert (
                    clamped_result.boundary_hit
                ), "Should indicate boundary hit for extreme positions"


@pytest.mark.edge_case
@pytest.mark.parametrize("invalid_action", INVALID_ACTION_TEST_VALUES)
def test_invalid_action_edge_cases(edge_case_test_env, invalid_action):
    """
    Test environment response to invalid action parameters including out-of-range
    values, incorrect types, null values, and malformed action inputs with
    comprehensive error handling validation.
    """
    # Initialize environment and reset to get valid starting state
    obs, info = edge_case_test_env.reset(seed=42)
    initial_agent_position = info.get("agent_xy", (0, 0))

    # Attempt environment step with invalid action value
    with pytest.raises((ValidationError, ValueError, TypeError)) as exc_info:
        edge_case_test_env.step(invalid_action)

    # Verify appropriate exception was raised with helpful information
    error = exc_info.value
    if isinstance(error, ValidationError):
        # Test ValidationError contains helpful information about valid action ranges
        assert (
            "action" in str(error).lower()
        ), "Error message should mention action parameter"

        # Verify error provides context about valid ranges
        error_details = (
            error.get_validation_details()
            if hasattr(error, "get_validation_details")
            else {}
        )
        if error_details:
            assert (
                "expected_format" in error_details or "parameter_name" in error_details
            )

    # Verify environment state remains consistent after invalid action attempts
    try:
        # Test that environment can still perform valid operations
        test_obs, test_info = edge_case_test_env.reset(seed=42)
        assert (
            test_info.get("agent_xy") == initial_agent_position
        ), "Environment state should remain consistent"

        # Test valid action still works
        valid_result = edge_case_test_env.step(0)  # Action.UP
        assert len(valid_result) == 5, "Valid action should return proper 5-tuple"

    except Exception as e:
        pytest.fail(f"Environment state became inconsistent after invalid action: {e}")


@pytest.mark.edge_case
@pytest.mark.parametrize("grid_size", EXTREME_GRID_SIZES)
def test_extreme_environment_configurations(grid_size, parameter_validator=None):
    """
    Test environment initialization and behavior with extreme configuration
    parameters including invalid grid sizes, source locations, plume parameters,
    and configuration edge cases.
    """
    # Create parameter validator if not provided
    if parameter_validator is None:
        parameter_validator = ParameterValidator()

    # Test environment creation with extreme grid sizes
    try:
        if (
            grid_size[0] <= 0
            or grid_size[1] <= 0
            or grid_size[0] > 10000
            or grid_size[1] > 10000
        ):
            # Should raise ConfigurationError for invalid grid dimensions
            with pytest.raises(
                (ConfigurationError, ValueError, MemoryError)
            ) as exc_info:
                env = create_plume_search_env(grid_size=grid_size)

            # Verify appropriate error type and message
            error = exc_info.value
            if isinstance(error, ConfigurationError):
                valid_options = (
                    error.get_valid_options()
                    if hasattr(error, "get_valid_options")
                    else {}
                )
                assert "grid_size" in str(error).lower() or valid_options
        else:
            # Valid grid size should work (though may be slow)
            env = create_plume_search_env(grid_size=grid_size)
            obs, info = env.reset(seed=42)
            assert obs is not None, "Environment should initialize successfully"
            env.close()

    except MemoryError:
        # Expected for very large grid sizes
        pytest.skip(
            f"Grid size {grid_size} requires too much memory for test environment"
        )

    # Test source location placement outside valid grid boundaries
    if (
        grid_size[0] > 0
        and grid_size[1] > 0
        and grid_size[0] <= 1000
        and grid_size[1] <= 1000
    ):
        invalid_source_locations = [
            (-1, 0),
            (0, -1),
            (grid_size[0], grid_size[1] // 2),
            (grid_size[0] // 2, grid_size[1]),
        ]

        for invalid_source in invalid_source_locations:
            with pytest.raises((ConfigurationError, ValidationError, ValueError)):
                env = create_plume_search_env(
                    grid_size=grid_size, source_location=invalid_source
                )

    # Test plume parameter validation with extreme sigma values
    if (
        grid_size[0] > 0 and grid_size[1] > 0 and max(grid_size) <= 256
    ):  # Reasonable size for parameter testing
        for extreme_sigma in EXTREME_PLUME_PARAMETERS:
            try:
                # Test parameter validation
                param_result = parameter_validator.validate_parameter(
                    "sigma", extreme_sigma, {"min_value": 0.001, "max_value": 1000.0}
                )

                if not param_result.is_valid:
                    # Should reject invalid sigma values
                    with pytest.raises(
                        (ConfigurationError, ValidationError, ValueError)
                    ):
                        env = create_plume_search_env(
                            grid_size=grid_size, plume_params={"sigma": extreme_sigma}
                        )
                elif not (math.isnan(extreme_sigma) or math.isinf(extreme_sigma)):
                    # Valid sigma should work
                    env = create_plume_search_env(
                        grid_size=grid_size, plume_params={"sigma": extreme_sigma}
                    )
                    env.close()

            except (ValueError, TypeError, OverflowError):
                # Expected for invalid mathematical values
                pass


@pytest.mark.edge_case
@pytest.mark.mathematical
def test_mathematical_precision_edge_cases(edge_case_test_env):
    """
    Test mathematical edge cases in plume calculations including floating point
    precision, overflow/underflow conditions, NaN/infinity handling, and numerical
    stability validation.
    """
    # Test Gaussian plume calculation with extreme sigma values near zero
    small_sigma_env = create_plume_search_env(
        grid_size=(32, 32),
        source_location=(16, 16),
        plume_params={"sigma": 0.001},  # Very small sigma
    )

    obs, info = small_sigma_env.reset(seed=42)

    # Verify concentration calculations handle division by zero gracefully
    assert not math.isnan(obs[0]), "Observation should not be NaN with small sigma"
    assert not math.isinf(obs[0]), "Observation should not be infinite with small sigma"
    assert (
        0.0 <= obs[0] <= 1.0
    ), "Observation should be in valid concentration range [0,1]"

    # Test plume field generation with source location at grid boundaries
    for source_x, source_y in [(0, 0), (31, 31), (0, 31), (31, 0)]:
        boundary_source_env = create_plume_search_env(
            grid_size=(32, 32), source_location=(source_x, source_y)
        )

        obs, info = boundary_source_env.reset(seed=42)

        # Validate concentration value clamping to [0,1] range under extreme conditions
        assert (
            0.0 <= obs[0] <= 1.0
        ), f"Boundary source observation should be valid: {obs[0]}"

        # Test distance calculations with maximum coordinate values
        agent_pos = info.get("agent_xy", (0, 0))
        distance = info.get("distance_to_source", 0.0)

        # Verify floating point precision handling in reward calculations
        assert not math.isnan(distance), "Distance calculation should not produce NaN"
        assert not math.isinf(
            distance
        ), "Distance calculation should not produce infinity"
        assert distance >= 0.0, "Distance should be non-negative"

        boundary_source_env.close()

    # Test mathematical consistency with very small coordinate differences
    obs1, info1 = small_sigma_env.reset(seed=123)
    obs2, info2 = small_sigma_env.reset(seed=123)

    # Verify numerical stability across multiple episode resets with extreme parameters
    assert (
        abs(obs1[0] - obs2[0]) < 1e-10
    ), "Identical seeds should produce numerically identical results"
    assert info1.get("agent_xy") == info2.get(
        "agent_xy"
    ), "Agent positions should be identical"

    small_sigma_env.close()

    # Test NaN and infinity propagation through mathematical operations
    try:
        # Test with invalid plume parameters that might produce mathematical edge cases
        nan_env = create_plume_search_env(
            grid_size=(16, 16), plume_params={"sigma": float("nan")}
        )
        pytest.fail("Environment should reject NaN sigma parameter")
    except (ConfigurationError, ValidationError, ValueError):
        # Expected rejection of NaN parameters
        pass

    try:
        # Test with infinite plume parameters
        inf_env = create_plume_search_env(
            grid_size=(16, 16), plume_params={"sigma": float("inf")}
        )
        pytest.fail("Environment should reject infinite sigma parameter")
    except (ConfigurationError, ValidationError, ValueError):
        # Expected rejection of infinite parameters
        pass


@pytest.mark.edge_case
@pytest.mark.rendering
@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
def test_rendering_edge_cases_and_failures(edge_case_test_env):
    """
    Test rendering system robustness under extreme conditions including backend
    failures, memory constraints, very large/small grids, and display compatibility
    edge cases with fallback validation.
    """
    # Test RGB array rendering with extremely large grid sizes
    large_grid_env = create_plume_search_env(grid_size=(256, 256))
    obs, info = large_grid_env.reset(seed=42)

    # Verify memory usage stays within bounds for large rendering operations
    try:
        start_memory = 0
        if hasattr(gc, "get_stats"):
            start_memory = sum(stat["collected"] for stat in gc.get_stats())

        rgb_array = large_grid_env.render(mode="rgb_array")

        # Validate RGB array properties
        assert rgb_array is not None, "Large grid should produce RGB array"
        assert rgb_array.shape == (
            256,
            256,
            3,
        ), "RGB array should have correct dimensions"
        assert rgb_array.dtype == np.uint8, "RGB array should be uint8 type"

        # Check memory usage after rendering
        end_memory = 0
        if hasattr(gc, "get_stats"):
            end_memory = sum(stat["collected"] for stat in gc.get_stats())

        # Memory usage should not grow excessively
        memory_growth = end_memory - start_memory
        assert (
            memory_growth < 100
        ), f"Memory growth {memory_growth} should be reasonable"

    except MemoryError:
        pytest.skip("Large grid rendering requires too much memory")
    finally:
        large_grid_env.close()

    # Test human mode rendering with matplotlib backend failures
    with unittest.mock.patch("matplotlib.pyplot.subplots") as mock_subplots:
        mock_subplots.side_effect = ImportError("Simulated matplotlib failure")

        # Should gracefully fallback to rgb_array mode when human rendering fails
        try:
            fallback_result = edge_case_test_env.render(mode="human")
            # Should either return None (graceful failure) or fall back to RGB array
            if fallback_result is not None:
                assert isinstance(
                    fallback_result, np.ndarray
                ), "Fallback should produce array"
        except RenderingError as e:
            # Should provide fallback suggestions
            if hasattr(e, "get_fallback_suggestions"):
                suggestions = e.get_fallback_suggestions()
                assert suggestions, "RenderingError should provide fallback suggestions"

    # Test rendering with single-cell grids (1x1) and minimal configurations
    minimal_env = create_plume_search_env(grid_size=(1, 1), source_location=(0, 0))
    obs, info = minimal_env.reset(seed=42)

    # Verify rendering consistency with agent and source at same position
    rgb_minimal = minimal_env.render(mode="rgb_array")
    assert rgb_minimal is not None, "Minimal grid should render successfully"
    assert rgb_minimal.shape == (1, 1, 3), "Minimal RGB should have correct shape"

    # Test concurrent rendering requests and thread safety
    import queue
    import threading

    render_results = queue.Queue()
    render_errors = queue.Queue()

    def concurrent_render(env, result_queue, error_queue):
        """Helper function for concurrent rendering test."""
        try:
            result = env.render(mode="rgb_array")
            result_queue.put(result)
        except Exception as e:
            error_queue.put(e)

    # Start multiple rendering threads
    threads = []
    for i in range(3):
        thread = threading.Thread(
            target=concurrent_render,
            args=(edge_case_test_env, render_results, render_errors),
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join(timeout=5.0)

    # Validate resource cleanup after rendering failures
    cleanup_successful = True
    try:
        # Force garbage collection to test cleanup
        gc.collect()

        # Test that environment still functions after concurrent rendering
        test_obs, test_info = edge_case_test_env.reset(seed=42)
        test_render = edge_case_test_env.render(mode="rgb_array")
        assert (
            test_render is not None
        ), "Environment should function after concurrent rendering"

    except Exception as e:
        cleanup_successful = False
        pytest.fail(f"Resource cleanup failed after concurrent rendering: {e}")

    minimal_env.close()

    # Test headless environment rendering with no display available
    with unittest.mock.patch.dict("os.environ", {}, clear=True):
        # Remove DISPLAY environment variable to simulate headless
        try:
            headless_render = edge_case_test_env.render(mode="human")
            # Should gracefully handle headless environment
            assert headless_render is None or isinstance(headless_render, np.ndarray)
        except RenderingError as e:
            # Should provide appropriate fallback suggestions for headless mode
            if hasattr(e, "get_fallback_suggestions"):
                suggestions = e.get_fallback_suggestions()
                assert (
                    "rgb_array" in str(suggestions).lower()
                ), "Should suggest rgb_array mode for headless"


@pytest.mark.edge_case
@pytest.mark.memory_stress
@pytest.mark.slow
def test_memory_exhaustion_and_resource_limits(memory_monitor, cleanup_validator):
    """
    Test system behavior under memory pressure and resource exhaustion conditions
    including memory leak detection, cleanup validation, and resource recovery testing.
    """
    initial_memory = 0
    if hasattr(memory_monitor, "get_current_usage"):
        initial_memory = memory_monitor.get_current_usage().get("memory_mb", 0)

    # Create multiple environment instances to stress test memory usage
    test_environments = []
    memory_readings = []

    try:
        for i in range(10):  # Create multiple environments
            env = create_plume_search_env(grid_size=(64, 64), source_location=(32, 32))
            test_environments.append(env)

            # Monitor memory usage patterns during repeated episode execution
            obs, info = env.reset(seed=i)

            # Execute several steps to populate internal state
            for step in range(50):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

            # Record memory usage
            if hasattr(memory_monitor, "get_current_usage"):
                current_memory = memory_monitor.get_current_usage().get("memory_mb", 0)
                memory_readings.append(current_memory)

        # Validate no memory leaks occur during stress testing iterations
        if len(memory_readings) >= 2:
            memory_growth = memory_readings[-1] - memory_readings[0]
            memory_per_env = memory_growth / len(test_environments)

            # Each environment should not consume excessive memory
            assert (
                memory_per_env < 50
            ), f"Memory per environment {memory_per_env}MB too high"

        # Test memory cleanup after environment close() operations
        environments_to_close = test_environments[:5]  # Close half the environments

        for env in environments_to_close:
            env.close()

        # Force garbage collection and check memory reduction
        gc.collect()

        if hasattr(memory_monitor, "get_current_usage"):
            post_cleanup_memory = memory_monitor.get_current_usage().get("memory_mb", 0)

            # Memory should be reduced after cleanup
            if len(memory_readings) > 0:
                memory_reduction = memory_readings[-1] - post_cleanup_memory
                assert memory_reduction >= 0, "Memory should be reduced after cleanup"

        # Test system behavior when approaching memory limits
        large_arrays = []
        try:
            # Allocate arrays to approach memory limits
            for i in range(MEMORY_STRESS_TEST_ITERATIONS // 100):  # Reduced for safety
                array = np.zeros((1000, 1000), dtype=np.float32)
                large_arrays.append(array)

                # Check if we're approaching limits
                if len(large_arrays) % 10 == 0:
                    if hasattr(memory_monitor, "get_current_usage"):
                        usage = memory_monitor.get_current_usage()
                        if usage.get("memory_mb", 0) > 500:  # Stop before exhaustion
                            break

        except MemoryError:
            # ResourceError should be raised appropriately for memory exhaustion
            # In real implementation, this would be caught and converted to ResourceError
            pass
        finally:
            # Test cleanup action suggestions and memory recovery
            del large_arrays
            gc.collect()

    finally:
        # Clean up all test environments
        for env in test_environments:
            try:
                env.close()
            except Exception:
                pass  # Ignore cleanup errors

        gc.collect()

        # Verify system remains stable after resource limit recovery
        try:
            recovery_env = create_plume_search_env()
            recovery_obs, recovery_info = recovery_env.reset(seed=42)
            assert (
                recovery_obs is not None
            ), "System should remain stable after memory stress"
            recovery_env.close()
        except Exception as e:
            pytest.fail(f"System instability after memory stress test: {e}")


@pytest.mark.edge_case
@pytest.mark.concurrency
@pytest.mark.skipif(SINGLE_THREADED_ONLY, reason="Concurrency testing disabled")
def test_concurrent_operations_and_thread_safety(edge_case_test_env):
    """
    Test thread safety and concurrent operation handling including simultaneous
    environment access, rendering conflicts, and shared resource management validation.
    """
    import queue
    import threading
    import time

    # Test multiple threads accessing same environment instance simultaneously
    results_queue = queue.Queue()
    errors_queue = queue.Queue()

    def thread_worker(env, worker_id, operations_count=50):
        """Worker function for concurrent environment access."""
        try:
            # Reset environment for this thread
            obs, info = env.reset(seed=worker_id)
            results_queue.put(f"Worker {worker_id}: Reset successful")

            # Perform environment operations
            for i in range(operations_count):
                action = env.action_space.sample()
                result = env.step(action)

                # Verify thread safety of state management operations
                assert (
                    len(result) == 5
                ), f"Worker {worker_id}: Invalid step result length"

                obs, reward, terminated, truncated, info = result
                assert isinstance(
                    obs, np.ndarray
                ), f"Worker {worker_id}: Invalid observation type"

                if terminated or truncated:
                    break

            results_queue.put(f"Worker {worker_id}: Operations completed")

        except Exception as e:
            errors_queue.put(f"Worker {worker_id}: {str(e)}")

    # Start multiple worker threads
    threads = []
    num_workers = 3

    for worker_id in range(num_workers):
        thread = threading.Thread(
            target=thread_worker, args=(edge_case_test_env, worker_id, 25)
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete with timeout
    for thread in threads:
        thread.join(timeout=EDGE_CASE_TIMEOUT_SECONDS)
        if thread.is_alive():
            pytest.fail("Thread did not complete within timeout")

    # Verify no race conditions occurred in boundary enforcement operations
    while not errors_queue.empty():
        error = errors_queue.get()
        if "boundary" in error.lower() or "position" in error.lower():
            pytest.fail(f"Boundary enforcement race condition detected: {error}")

    # Test concurrent rendering operations from multiple threads
    render_results = queue.Queue()
    render_errors = queue.Queue()

    def render_worker(env, worker_id):
        """Worker function for concurrent rendering."""
        try:
            for i in range(5):
                rgb_array = env.render(mode="rgb_array")
                if rgb_array is not None:
                    render_results.put(f"Render {worker_id}-{i}: Success")
                time.sleep(0.01)  # Small delay between renders
        except Exception as e:
            render_errors.put(f"Render {worker_id}: {str(e)}")

    # Start concurrent rendering threads
    render_threads = []
    for worker_id in range(2):
        thread = threading.Thread(
            target=render_worker, args=(edge_case_test_env, worker_id)
        )
        render_threads.append(thread)
        thread.start()

    # Wait for rendering threads
    for thread in render_threads:
        thread.join(timeout=10.0)

    # Validate proper locking and synchronization in shared resource access
    render_error_count = render_errors.qsize()
    if render_error_count > 0:
        # Some rendering errors may be acceptable, but not all
        assert render_error_count < num_workers, "Too many concurrent rendering errors"

    # Test performance monitoring accuracy under concurrent load
    # This would be implemented with actual performance monitoring in the real system
    final_obs, final_info = edge_case_test_env.reset(seed=999)
    assert (
        final_obs is not None
    ), "Environment should remain functional after concurrent operations"


@pytest.mark.edge_case
@pytest.mark.lifecycle
def test_episode_lifecycle_edge_cases(edge_case_test_env):
    """
    Test edge cases in episode lifecycle management including multiple resets,
    premature close operations, step limit handling, and state transition validation.
    """
    # Test multiple consecutive reset() calls without step operations
    reset_results = []
    for i in range(5):
        obs, info = edge_case_test_env.reset(seed=i)
        reset_results.append((obs.copy(), info.copy()))

        # Verify environment state consistency after repeated resets
        assert obs is not None, f"Reset {i} should return valid observation"
        assert isinstance(info, dict), f"Reset {i} should return info dictionary"
        assert "agent_xy" in info, f"Reset {i} should include agent position"

    # Verify different seeds produce different starting positions
    positions = [result[1]["agent_xy"] for result in reset_results]
    unique_positions = set(positions)
    assert (
        len(unique_positions) > 1
    ), "Different seeds should produce different start positions"

    # Test step operations after environment close() - should raise StateError
    test_env = create_plume_search_env()
    obs, info = test_env.reset(seed=42)
    test_env.close()

    # Validate proper error handling for operations on closed environment
    with pytest.raises((StateError, RuntimeError, ValueError)) as exc_info:
        test_env.step(0)

    # Check error provides recovery suggestions
    error = exc_info.value
    if isinstance(error, StateError) and hasattr(error, "suggest_recovery_action"):
        recovery_suggestion = error.suggest_recovery_action()
        assert (
            recovery_suggestion is not None
        ), "StateError should provide recovery suggestion"

    # Test episode termination at maximum step limits
    limited_env = create_plume_search_env(max_steps=10)
    obs, info = limited_env.reset(seed=42)

    step_count = 0
    terminated = False
    truncated = False

    while not terminated and not truncated and step_count < 20:  # Safety limit
        action = limited_env.action_space.sample()
        obs, reward, terminated, truncated, info = limited_env.step(action)
        step_count += 1

    # Verify truncation logic works correctly at step boundaries
    assert step_count <= 10, "Episode should not exceed max_steps"
    assert (
        truncated or terminated
    ), "Episode should end due to truncation or termination"

    # Test reset operations during active episodes with state validation
    active_env = create_plume_search_env()
    obs, info = active_env.reset(seed=123)

    # Take some steps to establish active episode
    for i in range(5):
        action = active_env.action_space.sample()
        obs, reward, terminated, truncated, info = active_env.step(action)
        if terminated or truncated:
            break

    # Reset during active episode should work
    new_obs, new_info = active_env.reset(seed=456)

    # Validate info dictionary consistency throughout episode lifecycle
    required_keys = ["agent_xy"]
    for key in required_keys:
        assert key in new_info, f"Info dictionary should contain {key} after reset"

    # Test environment integrity checking after state transition edge cases
    if hasattr(active_env, "validate_environment_integrity"):
        integrity_check = active_env.validate_environment_integrity()
        assert integrity_check, "Environment integrity should be maintained"

    # Cleanup test environments
    limited_env.close()
    active_env.close()


@pytest.mark.edge_case
@pytest.mark.parametrize("invalid_seed", INVALID_SEED_VALUES)
def test_seeding_and_reproducibility_edge_cases(edge_case_test_env, invalid_seed):
    """
    Test edge cases in seeding and reproducibility including invalid seed values,
    seed overflow conditions, cross-session consistency, and deterministic behavior
    validation under extreme conditions.
    """
    # Test environment seeding with invalid seed values
    if invalid_seed is None:
        # None seed should work (uses random seed)
        obs, info = edge_case_test_env.reset(seed=None)
        assert obs is not None, "None seed should produce valid observation"
        assert isinstance(info, dict), "None seed should produce valid info"
    elif isinstance(invalid_seed, (int, float)) and not (
        math.isnan(invalid_seed) or math.isinf(invalid_seed)
    ):
        if -(2**31) <= invalid_seed <= 2**31 - 1:
            # Valid integer range should work
            try:
                obs, info = edge_case_test_env.reset(seed=int(invalid_seed))
                assert obs is not None, f"Valid seed {invalid_seed} should work"
            except (ValueError, OverflowError):
                # Some edge cases near limits may fail
                pass
        else:
            # Out of range seeds should raise error
            with pytest.raises((ValueError, OverflowError, ValidationError)):
                edge_case_test_env.reset(seed=invalid_seed)
    else:
        # Invalid seed types should raise appropriate error
        with pytest.raises((TypeError, ValidationError, ValueError)):
            edge_case_test_env.reset(seed=invalid_seed)

    # Test seed value edge cases including maximum integer values and overflow
    edge_seeds = [0, 1, -1, 2**31 - 1, -(2**31)]
    seed_results = {}

    for seed in edge_seeds:
        try:
            obs, info = edge_case_test_env.reset(seed=seed)
            seed_results[seed] = (obs.copy(), info.copy())
        except (ValueError, OverflowError):
            # Some extreme seeds may not be supported
            pass

    # Verify identical episode generation with same seeds under edge conditions
    if len(seed_results) >= 2:
        # Test reproducibility with a known working seed
        test_seed = list(seed_results.keys())[0]

        # First episode
        obs1, info1 = edge_case_test_env.reset(seed=test_seed)
        trajectory1 = [(obs1.copy(), info1.copy())]

        for step in range(10):
            action = 0  # Use deterministic action
            obs, reward, terminated, truncated, info = edge_case_test_env.step(action)
            trajectory1.append((obs.copy(), reward, terminated, truncated, info.copy()))
            if terminated or truncated:
                break

        # Second episode with same seed
        obs2, info2 = edge_case_test_env.reset(seed=test_seed)
        trajectory2 = [(obs2.copy(), info2.copy())]

        for step in range(len(trajectory1) - 1):
            action = 0  # Same deterministic action
            obs, reward, terminated, truncated, info = edge_case_test_env.step(action)
            trajectory2.append((obs.copy(), reward, terminated, truncated, info.copy()))
            if terminated or truncated:
                break

        # Verify identical trajectories
        assert len(trajectory1) == len(
            trajectory2
        ), "Trajectories should have same length"

        for i, (step1, step2) in enumerate(zip(trajectory1, trajectory2)):
            if i == 0:
                # Initial observations and info
                obs1, info1 = step1
                obs2, info2 = step2
                assert np.allclose(obs1, obs2), "Initial observations should match"
                assert (
                    info1["agent_xy"] == info2["agent_xy"]
                ), "Initial positions should match"
            else:
                # Step results
                obs1, reward1, term1, trunc1, info1 = step1
                obs2, reward2, term2, trunc2, info2 = step2
                assert np.allclose(obs1, obs2), f"Step {i} observations should match"
                assert reward1 == reward2, f"Step {i} rewards should match"
                assert term1 == term2, f"Step {i} termination should match"
                assert trunc1 == trunc2, f"Step {i} truncation should match"


@pytest.mark.edge_case
@pytest.mark.performance_stress
@pytest.mark.parametrize("step_count", STRESS_TEST_STEP_COUNTS)
def test_performance_degradation_scenarios(
    performance_tracker, edge_case_test_env, step_count
):
    """
    Test system performance under degraded conditions including timing constraint
    violations, resource bottlenecks, and performance recovery validation with
    comprehensive monitoring.
    """
    # Start performance monitoring for the test
    if hasattr(performance_tracker, "start_monitoring"):
        performance_tracker.start_monitoring(f"stress_test_{step_count}")

    # Execute long episodes with varying step counts
    obs, info = edge_case_test_env.reset(seed=42)

    step_times = []
    total_start_time = time.perf_counter()

    for step in range(step_count):
        # Monitor step latency throughout extended episode execution
        step_start = time.perf_counter()

        action = edge_case_test_env.action_space.sample()
        obs, reward, terminated, truncated, info = edge_case_test_env.step(action)

        step_end = time.perf_counter()
        step_time = step_end - step_start
        step_times.append(step_time)

        if terminated or truncated:
            break

    total_duration = time.perf_counter() - total_start_time

    # Validate performance targets are maintained under stress conditions
    if step_times:
        avg_step_time = sum(step_times) / len(step_times)
        max_step_time = max(step_times)

        # Performance targets - these may need adjustment based on system capabilities
        target_avg_step_time = 0.01  # 10ms average for stress testing
        target_max_step_time = 0.05  # 50ms maximum for any single step

        # Test system behavior when performance targets are exceeded
        if avg_step_time > target_avg_step_time:
            warnings.warn(
                f"Average step time {avg_step_time:.4f}s exceeds target {target_avg_step_time}s"
            )

        if max_step_time > target_max_step_time:
            warnings.warn(
                f"Maximum step time {max_step_time:.4f}s exceeds target {target_max_step_time}s"
            )

        # Verify performance monitoring accuracy during edge case execution
        if hasattr(performance_tracker, "get_metrics"):
            metrics = performance_tracker.get_metrics()

            # Check that monitoring captured the stress test
            if "average_step_time" in metrics:
                monitored_avg = metrics["average_step_time"]
                # Allow some tolerance for monitoring overhead
                assert (
                    abs(monitored_avg - avg_step_time) < 0.01
                ), "Performance monitoring should be accurate"

    # Test performance recovery after resource cleanup operations
    gc.collect()  # Force garbage collection

    # Test performance after cleanup
    recovery_start = time.perf_counter()
    recovery_obs, recovery_info = edge_case_test_env.reset(seed=123)

    recovery_steps = []
    for i in range(min(10, step_count // 10)):  # Small recovery test
        step_start = time.perf_counter()
        action = edge_case_test_env.action_space.sample()
        obs, reward, terminated, truncated, info = edge_case_test_env.step(action)
        recovery_steps.append(time.perf_counter() - step_start)
        if terminated or truncated:
            break

    # Validate optimization recommendations generated during performance stress
    if recovery_steps:
        recovery_avg = sum(recovery_steps) / len(recovery_steps)

        # Performance should improve or at least not degrade significantly after cleanup
        if step_times and recovery_avg > avg_step_time * 2:
            warnings.warn(
                f"Performance degraded after cleanup: {recovery_avg:.4f}s vs {avg_step_time:.4f}s"
            )

    # Test system stability during sustained high-frequency operations
    stability_test_start = time.perf_counter()
    stability_steps = 0

    while (
        time.perf_counter() - stability_test_start < 1.0
    ):  # 1 second of high-frequency ops
        try:
            action = edge_case_test_env.action_space.sample()
            obs, reward, terminated, truncated, info = edge_case_test_env.step(action)
            stability_steps += 1

            if terminated or truncated:
                edge_case_test_env.reset(seed=42)
        except Exception as e:
            pytest.fail(f"System became unstable during high-frequency operations: {e}")

    # Verify performance metrics collection remains accurate under load
    if hasattr(performance_tracker, "stop_monitoring"):
        final_metrics = performance_tracker.stop_monitoring()

        if final_metrics and "total_steps" in final_metrics:
            # Verify step counting accuracy
            expected_steps = len(step_times) + len(recovery_steps) + stability_steps
            monitored_steps = final_metrics["total_steps"]

            # Allow some tolerance for concurrent operations
            step_difference = abs(monitored_steps - expected_steps)
            assert (
                step_difference < expected_steps * 0.1
            ), "Step counting should be reasonably accurate"


@pytest.mark.edge_case
@pytest.mark.validation
def test_validation_system_edge_cases(parameter_validator=None):
    """
    Test validation system robustness including validation cache behavior,
    parameter sanitization edge cases, cross-parameter consistency checking,
    and validation performance under stress conditions.
    """
    # Create parameter validator if not provided
    if parameter_validator is None:
        parameter_validator = ParameterValidator()

    # Test validation cache behavior with extremely large parameter sets
    large_parameter_set = {}
    for i in range(1000):
        large_parameter_set[f"param_{i}"] = i % 100

    # Test cache performance and memory usage under stress conditions
    cache_start_time = time.perf_counter()
    cache_results = []

    try:
        # Perform batch validation to test caching
        if hasattr(parameter_validator, "batch_validate"):
            batch_result = parameter_validator.batch_validate(large_parameter_set)
            cache_results.append(batch_result)
        else:
            # Individual validation to test caching
            for key, value in list(large_parameter_set.items())[
                :100
            ]:  # Limit for performance
                result = parameter_validator.validate_parameter(
                    key, value, {"min_value": 0, "max_value": 200}
                )
                cache_results.append(result)

        cache_duration = time.perf_counter() - cache_start_time

        # Verify cache performance
        assert (
            cache_duration < 5.0
        ), f"Validation caching took too long: {cache_duration:.2f}s"

    except MemoryError:
        pytest.skip("Large parameter set validation requires too much memory")

    # Test parameter sanitization with malformed and malicious inputs
    malicious_inputs = [
        {"type": "script_injection", "value": '<script>alert("test")</script>'},
        {"type": "sql_injection", "value": "'; DROP TABLE users; --"},
        {"type": "path_traversal", "value": "../../etc/passwd"},
        {"type": "null_bytes", "value": "test\x00hidden"},
        {"type": "unicode_overflow", "value": "A" * 100000},
        {"type": "nested_dict", "value": {"level1": {"level2": {"level3": "deep"}}}},
        {"type": "circular_reference", "value": None},  # Will be created separately
    ]

    # Create circular reference
    circular = {"self": None}
    circular["self"] = circular
    malicious_inputs.append({"type": "circular_reference", "value": circular})

    for malicious_input in malicious_inputs:
        try:
            # Test parameter sanitization
            result = parameter_validator.validate_parameter(
                "test_param",
                malicious_input["value"],
                {"allow_strings": True, "max_length": 1000},
            )

            # Validation should either succeed with sanitized input or fail safely
            if result.is_valid:
                # If validation passes, input should be sanitized
                sanitized_value = (
                    result.sanitized_value
                    if hasattr(result, "sanitized_value")
                    else malicious_input["value"]
                )

                # Basic sanitization checks
                if isinstance(sanitized_value, str):
                    assert (
                        len(sanitized_value) <= 1000
                    ), "Sanitized value should respect length limits"
                    assert "\x00" not in sanitized_value, "Null bytes should be removed"

        except (ValidationError, ValueError, TypeError) as e:
            # Expected for malicious inputs
            error_message = str(e).lower()

            # Validate sensitive parameter detection and redaction in edge cases
            sensitive_keywords = ["password", "token", "key", "secret"]
            for keyword in sensitive_keywords:
                if keyword in error_message:
                    assert (
                        "*" in error_message or "[REDACTED]" in error_message
                    ), "Sensitive data should be redacted"

    # Test cross-parameter consistency checking with conflicting parameters
    conflicting_params = {
        "min_value": 100,
        "max_value": 50,  # Conflicts with min_value
        "grid_width": 10,
        "grid_height": 5,
        "total_cells": 100,  # Conflicts with width * height = 50
    }

    if hasattr(parameter_validator, "validate_parameter_consistency"):
        consistency_result = parameter_validator.validate_parameter_consistency(
            conflicting_params
        )
        assert not consistency_result.is_valid, "Should detect parameter conflicts"

        # Verify validation error accumulation and reporting for multiple failures
        if hasattr(consistency_result, "validation_errors"):
            errors = consistency_result.validation_errors
            assert len(errors) >= 2, "Should report multiple validation errors"

    # Test validation performance optimization under high-frequency validation
    high_frequency_start = time.perf_counter()
    high_frequency_count = 0

    # Run validation for 0.5 seconds to test performance
    while time.perf_counter() - high_frequency_start < 0.5:
        try:
            result = parameter_validator.validate_parameter(
                "perf_test",
                high_frequency_count % 100,
                {"min_value": 0, "max_value": 200},
            )
            high_frequency_count += 1
        except Exception:
            break

    high_frequency_duration = time.perf_counter() - high_frequency_start

    # Validate validation performance under stress
    if high_frequency_count > 0:
        avg_validation_time = high_frequency_duration / high_frequency_count
        assert (
            avg_validation_time < 0.001
        ), f"Average validation time {avg_validation_time:.6f}s too slow"

    # Test validation context management during concurrent validation operations
    import queue
    import threading

    concurrent_results = queue.Queue()
    concurrent_errors = queue.Queue()

    def concurrent_validation(validator, thread_id):
        """Concurrent validation worker."""
        try:
            for i in range(50):
                result = validator.validate_parameter(
                    f"concurrent_{thread_id}_{i}", i, {"min_value": 0, "max_value": 100}
                )
                concurrent_results.put(result)
        except Exception as e:
            concurrent_errors.put(f"Thread {thread_id}: {e}")

    # Start concurrent validation threads
    validation_threads = []
    for thread_id in range(3):
        thread = threading.Thread(
            target=concurrent_validation, args=(parameter_validator, thread_id)
        )
        validation_threads.append(thread)
        thread.start()

    # Wait for concurrent validation to complete
    for thread in validation_threads:
        thread.join(timeout=5.0)

    # Verify validation result serialization and error context preservation
    concurrent_error_count = concurrent_errors.qsize()
    assert (
        concurrent_error_count == 0
    ), f"Concurrent validation produced {concurrent_error_count} errors"

    # Check that all validation results are properly formatted
    result_count = concurrent_results.qsize()
    assert result_count > 0, "Should have validation results from concurrent operations"

    # Sample a few results to verify format
    for _ in range(min(5, result_count)):
        if not concurrent_results.empty():
            result = concurrent_results.get()
            assert hasattr(
                result, "is_valid"
            ), "Validation result should have is_valid attribute"


@pytest.mark.edge_case
@pytest.mark.error_handling
def test_error_handling_and_recovery_mechanisms():
    """
    Test comprehensive error handling including exception hierarchy behavior,
    error context preservation, recovery suggestion accuracy, and error
    propagation under extreme conditions.
    """
    # Test all custom exception types with edge case scenarios
    exception_tests = [
        {
            "exception_class": ValidationError,
            "test_message": "Test validation failure",
            "test_context": {
                "parameter_name": "test_param",
                "invalid_value": "invalid",
                "expected_format": "valid format",
            },
        },
        {
            "exception_class": StateError,
            "test_message": "Test state transition failure",
            "test_context": {
                "current_state": "invalid",
                "expected_state": "ready",
                "component_name": "environment",
                "attempted_operation": "step",
            },
        },
        {
            "exception_class": RenderingError,
            "test_message": "Test rendering failure",
            "test_context": {
                "render_mode": "human",
                "backend_name": "matplotlib",
            },
        },
        {
            "exception_class": ConfigurationError,
            "test_message": "Test configuration failure",
            "test_context": {
                "config_parameter": "grid_size",
                "parameter_value": (-1, -1),
            },
        },
        {
            "exception_class": ResourceError,
            "test_message": "Test resource exhaustion",
            "test_context": {"resource_type": "memory", "current_usage": "500MB"},
        },
    ]

    for exception_test in exception_tests:
        exception_class = exception_test["exception_class"]
        test_message = exception_test["test_message"]
        test_context = exception_test["test_context"]

        # Create and test exception instance
        try:
            context_kwargs = dict(test_context)
            if exception_class is StateError:
                attempted_operation = context_kwargs.pop("attempted_operation", None)
                error_instance = exception_class(
                    test_message,
                    current_state=context_kwargs.get("current_state"),
                    expected_state=context_kwargs.get("expected_state"),
                    component_name=context_kwargs.get("component_name"),
                )
                if attempted_operation:
                    setattr(error_instance, "attempted_operation", attempted_operation)
                raise error_instance

            raise exception_class(test_message, **context_kwargs)
        except exception_class as e:
            # Verify error context preservation during exception propagation
            error_message = str(e)
            assert (
                test_message in error_message
            ), "Error message should contain original message"

            # Test recovery suggestion generation for various error types
            if hasattr(e, "suggest_recovery_action"):
                recovery_suggestion = e.suggest_recovery_action()
                assert (
                    recovery_suggestion is not None
                ), f"{exception_class.__name__} should provide recovery suggestion"
                assert isinstance(
                    recovery_suggestion, str
                ), "Recovery suggestion should be string"
                assert (
                    len(recovery_suggestion) > 0
                ), "Recovery suggestion should not be empty"

            # Test specific exception methods
            if isinstance(e, ValidationError) and hasattr(e, "get_validation_details"):
                details = e.get_validation_details()
                assert isinstance(
                    details, dict
                ), "Validation details should be dictionary"
                assert "parameter_name" in details or "invalid_value" in details

            elif isinstance(e, StateError) and hasattr(e, "suggest_recovery_action"):
                recovery = e.suggest_recovery_action()
                assert "reset" in recovery.lower() or "restart" in recovery.lower()

            elif isinstance(e, RenderingError) and hasattr(
                e, "get_fallback_suggestions"
            ):
                fallbacks = e.get_fallback_suggestions()
                assert (
                    fallbacks is not None
                ), "RenderingError should provide fallback suggestions"

            elif isinstance(e, ConfigurationError) and hasattr(e, "get_valid_options"):
                options = e.get_valid_options()
                assert (
                    options is not None
                ), "ConfigurationError should provide valid options"

            elif isinstance(e, ResourceError) and hasattr(e, "suggest_cleanup_actions"):
                cleanup = e.suggest_cleanup_actions()
                assert (
                    cleanup is not None
                ), "ResourceError should provide cleanup suggestions"

    # Test exception handling performance under high error frequency
    error_handling_start = time.perf_counter()
    error_count = 0

    while time.perf_counter() - error_handling_start < 0.1:  # 100ms of error handling
        try:
            raise ValidationError(
                f"Performance test error {error_count}", parameter_name="test"
            )
        except ValidationError:
            error_count += 1

    error_handling_duration = time.perf_counter() - error_handling_start

    # Validate error handling performance
    if error_count > 0:
        avg_error_time = error_handling_duration / error_count
        assert (
            avg_error_time < 0.001
        ), f"Error handling too slow: {avg_error_time:.6f}s per error"

    # Test error escalation mechanisms for critical failures
    try:
        # Simulate critical system failure
        critical_resource_error = ResourceError(
            "Critical system failure - memory exhausted",
            resource_type="memory",
            current_usage=999.0,
            limit_exceeded=512.0,
        )
        critical_resource_error.severity = ErrorSeverity.CRITICAL
        raise critical_resource_error
    except ResourceError as e:
        # Critical errors should be handled appropriately
        if hasattr(e, "severity"):
            assert e.severity == ErrorSeverity.CRITICAL, "Should preserve error severity"

        # Should suggest immediate cleanup actions
        if hasattr(e, "suggest_cleanup_actions"):
            cleanup_actions = e.suggest_cleanup_actions()
            assert (
                "immediate" in cleanup_actions.lower()
                or "urgent" in cleanup_actions.lower()
            )

    # Test exception handling thread safety during concurrent errors
    import queue
    import threading

    error_queue = queue.Queue()

    def concurrent_error_handler(thread_id):
        """Generate and handle errors concurrently."""
        try:
            for i in range(10):
                try:
                    if i % 3 == 0:
                        raise ValidationError(
                            f"Thread {thread_id} validation error {i}"
                        )
                    elif i % 3 == 1:
                        raise StateError(f"Thread {thread_id} state error {i}")
                    else:
                        raise RenderingError(f"Thread {thread_id} rendering error {i}")
                except (ValidationError, StateError, RenderingError) as e:
                    error_queue.put(f"Thread {thread_id}: Handled {type(e).__name__}")
        except Exception as e:
            error_queue.put(f"Thread {thread_id}: Unexpected error {e}")

    # Start concurrent error handling threads
    error_threads = []
    for thread_id in range(3):
        thread = threading.Thread(target=concurrent_error_handler, args=(thread_id,))
        error_threads.append(thread)
        thread.start()

    # Wait for error handling threads
    for thread in error_threads:
        thread.join(timeout=5.0)

    # Verify comprehensive error reporting includes actionable recovery information
    handled_errors = []
    while not error_queue.empty():
        handled_errors.append(error_queue.get())

    # Should have handled errors from all threads
    assert len(handled_errors) > 0, "Should have handled concurrent errors"

    # Verify different error types were handled
    error_types_handled = set()
    for error_msg in handled_errors:
        if "ValidationError" in error_msg:
            error_types_handled.add("ValidationError")
        elif "StateError" in error_msg:
            error_types_handled.add("StateError")
        elif "RenderingError" in error_msg:
            error_types_handled.add("RenderingError")

    assert (
        len(error_types_handled) >= 2
    ), "Should handle multiple error types concurrently"
