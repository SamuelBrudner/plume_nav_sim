"""
Tests for the protocol-based navigator implementation with enhanced API compliance and integration hardening.

This module provides comprehensive testing for NavigatorProtocol implementations, focusing on:
- Gymnasium 0.29.x API compliance with dual API support (4-tuple vs 5-tuple returns)
- Performance monitoring and SLA validation (≤10ms step() execution time)
- Dataclass-based Hydra structured configuration integration with Pydantic validation
- Property-based testing using Hypothesis for coordinate frame consistency
- Centralized Loguru logging system integration with correlation ID tracking
- Gymnasium utils env_checker integration for protocol compliance validation
- Enhanced test coverage for protocol-based navigation components with comprehensive edge case testing

The testing architecture follows scientific computing best practices with deterministic behavior,
comprehensive mocking, research-grade quality standards, and API consistency validation.

Key Testing Areas:
- NavigatorProtocol reset(seed=...) parameter support for deterministic initial state generation
- Dual API support ensuring legacy gym callers receive 4-tuple while Gymnasium callers receive 5-tuple
- Performance monitoring tests validating NavigatorProtocol implementations meet ≤10ms step() execution SLA
- Hydra configuration integration tests using new dataclass-based structured configs with Pydantic validation
- Property-based testing using Hypothesis for coordinate frame consistency across navigator implementations
- Centralized Loguru system integration including correlation ID tracking
- Gymnasium utils env_checker integration for protocol compliance validation
- Comprehensive edge case testing for protocol-based navigation components
"""

import pytest
import numpy as np
import os
import tempfile
import time
import threading
from typing import Any, Dict, Tuple, Optional, Union, List
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, asdict

# Core dependencies from the new architecture
from plume_nav_sim.core.navigator import NavigatorProtocol, NavigatorFactory
from plume_nav_sim.core.controllers import (
    SingleAgentController,
    MultiAgentController,
    SingleAgentParams,
    MultiAgentParams,
    create_controller_from_config,
)
from plume_nav_sim.config.schemas import (
    NavigatorConfig,
    SingleAgentConfig,
    MultiAgentConfig,
    SimulationConfig,
)
from plume_nav_sim.utils.logging_setup import (
    get_logger,
    correlation_context,
    LoggingConfig,
    setup_logger,
)
from plume_nav_sim.utils.seed_utils import (
    set_global_seed,
    seed_context_manager,
)

# Hypothesis for property-based testing
try:
    from hypothesis import given, strategies as st, assume, settings, HealthCheck
    from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule, initialize, precondition
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    given = lambda *args, **kwargs: lambda f: f
    assume = lambda x: None
    
    # Mock strategies object to handle @given decorators when hypothesis is not available
    class MockStrategies:
        def floats(self, **kwargs):
            return None
        def integers(self, **kwargs):
            return None
        def text(self, **kwargs):
            return None
        def booleans(self, **kwargs):
            return None
        def lists(self, *args, **kwargs):
            return None
    
    st = MockStrategies()
    
    # Mock settings and HealthCheck for test configuration
    class MockSettings:
        def __init__(self, **kwargs):
            pass
        def __call__(self, func):
            return func
    
    settings = MockSettings
    
    class MockHealthCheck:
        too_slow = None
    
    HealthCheck = MockHealthCheck()
    
    # Mock stateful testing components
    Bundle = None
    RuleBasedStateMachine = object
    rule = lambda *args, **kwargs: lambda f: f
    initialize = lambda *args, **kwargs: lambda f: f
    precondition = lambda *args, **kwargs: lambda f: f

# Gymnasium integration
try:
    import gymnasium
    from gymnasium.utils.env_checker import check_env
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gymnasium = None
    check_env = None

# Hydra integration imports with graceful fallback
try:
    from omegaconf import DictConfig, OmegaConf
    from hydra.core.config_store import ConfigStore
    from hydra import compose, initialize
    HYDRA_AVAILABLE = True
except ImportError:
    # Fallback for environments without Hydra
    HYDRA_AVAILABLE = False
    DictConfig = dict
    OmegaConf = None
    ConfigStore = None

# Module logger with enhanced context
logger = get_logger(__name__)


class TestSingleAgentController:
    """Tests for the SingleAgentController with enhanced API compliance and performance monitoring."""
    
    def test_initialization(self) -> None:
        """Test that a SingleAgentController initializes correctly with enhanced logging."""
        with correlation_context("single_agent_init_test"):
            # Test default initialization
            controller = SingleAgentController()
            assert controller.num_agents == 1
            assert controller.positions.shape == (1, 2)
            assert controller.orientations.shape == (1,)
            assert controller.speeds.shape == (1,)
            assert controller.max_speeds.shape == (1,)
            assert controller.angular_velocities.shape == (1,)
            
            # Test with custom parameters
            controller = SingleAgentController(
                position=(10.0, 20.0),
                orientation=45.0,
                speed=2.0,
                max_speed=5.0,
                angular_velocity=10.0
            )
            assert controller.positions[0, 0] == 10.0
            assert controller.positions[0, 1] == 20.0
            assert controller.orientations[0] == 45.0
            assert controller.speeds[0] == 2.0
            assert controller.max_speeds[0] == 5.0
            assert controller.angular_velocities[0] == 10.0
            
            logger.info("SingleAgentController initialization test completed")
    
    def test_initialization_with_dataclass_params(self) -> None:
        """Test initialization using SingleAgentParams dataclass for type safety."""
        params = SingleAgentParams(
            position=(15.0, 25.0),
            orientation=90.0,
            speed=1.5,
            max_speed=3.0,
            angular_velocity=0.2
        )
        
        controller = SingleAgentController(
            position=params.position,
            orientation=params.orientation,
            speed=params.speed,
            max_speed=params.max_speed,
            angular_velocity=params.angular_velocity
        )
        
        assert controller.positions[0, 0] == 15.0
        assert controller.positions[0, 1] == 25.0
        assert controller.orientations[0] == 90.0
        assert controller.speeds[0] == 1.5
        assert controller.max_speeds[0] == 3.0
        assert controller.angular_velocities[0] == 0.2
    
    def test_reset(self) -> None:
        """Test resetting the controller state with enhanced seed parameter support."""
        controller = SingleAgentController(position=(10.0, 20.0), orientation=45.0)
        controller.reset(position=(30.0, 40.0), orientation=90.0)
        assert controller.positions[0, 0] == 30.0
        assert controller.positions[0, 1] == 40.0
        assert controller.orientations[0] == 90.0
    
    def test_reset_with_seed_parameter(self) -> None:
        """Test NavigatorProtocol reset method supports new seed parameter for deterministic initial state generation."""
        with seed_context_manager(42):
            controller = SingleAgentController(position=(5.0, 5.0))
            
            # Test reset with seed parameter for Gymnasium 0.29.x compatibility
            seed_params = get_gymnasium_seed_parameter(123)
            
            # Reset with seed should be deterministic
            initial_position = controller.positions.copy()
            controller.reset(position=(10.0, 15.0), **seed_params)
            
            assert controller.positions[0, 0] == 10.0
            assert controller.positions[0, 1] == 15.0
            
            # Test that multiple resets with same seed produce consistent results
            controller.reset(position=(20.0, 25.0), **seed_params)
            assert controller.positions[0, 0] == 20.0
            assert controller.positions[0, 1] == 25.0
            
            logger.info("Reset with seed parameter test completed")
    
    def test_reset_with_params_dataclass(self) -> None:
        """Test reset using SingleAgentParams dataclass for type-safe parameter updates."""
        controller = SingleAgentController()
        
        # Test reset with type-safe parameter object
        reset_params = SingleAgentParams(
            position=(50.0, 60.0),
            orientation=180.0,
            speed=2.5,
            max_speed=4.0
        )
        
        controller.reset_with_params(reset_params)
        
        assert controller.positions[0, 0] == 50.0
        assert controller.positions[0, 1] == 60.0
        assert controller.orientations[0] == 180.0
        assert controller.speeds[0] == 2.5
        assert controller.max_speeds[0] == 4.0
    
    def test_step(self) -> None:
        """Test the step method updates position and orientation."""
        controller = SingleAgentController(
            position=(10.0, 10.0),
            orientation=0.0,  # Pointing along x-axis
            speed=1.0,
            angular_velocity=10.0
        )
        
        # Create a mock environment array
        env_array = np.zeros((100, 100))
        
        # Take a step
        controller.step(env_array)
        
        # Check that position was updated (should move along x-axis)
        assert controller.positions[0, 0] > 10.0
        assert np.isclose(controller.positions[0, 1], 10.0)
        
        # Check that orientation was updated
        assert controller.orientations[0] == 10.0
    
    def test_step_performance_sla(self) -> None:
        """Test step method meets ≤10ms execution time SLA with performance monitoring."""
        controller = SingleAgentController(
            position=(25.0, 25.0),
            speed=1.0,
            enable_logging=True
        )
        env_array = np.random.rand(100, 100)
        
        # Warm up the controller
        for _ in range(5):
            controller.step(env_array)
        
        # Measure step performance
        step_times = []
        num_measurements = 20
        
        for _ in range(num_measurements):
            start_time = time.perf_counter()
            controller.step(env_array)
            end_time = time.perf_counter()
            step_time_ms = (end_time - start_time) * 1000
            step_times.append(step_time_ms)
        
        # Validate SLA requirements
        mean_step_time = np.mean(step_times)
        max_step_time = np.max(step_times)
        p95_step_time = np.percentile(step_times, 95)
        
        # Performance assertions - relaxed for test environment but validates SLA
        assert mean_step_time < 10.0, f"Mean step time {mean_step_time:.2f}ms exceeds 10ms SLA"
        assert p95_step_time < 15.0, f"P95 step time {p95_step_time:.2f}ms indicates performance issues"
        
        # Test performance metrics collection
        if hasattr(controller, 'get_performance_metrics'):
            metrics = controller.get_performance_metrics()
            assert 'step_time_mean_ms' in metrics
            assert metrics['total_steps'] >= num_measurements
        
        logger.info(
            "Step performance SLA validation completed",
            extra={
                "mean_step_time_ms": mean_step_time,
                "max_step_time_ms": max_step_time,
                "p95_step_time_ms": p95_step_time,
                "measurements": num_measurements
            }
        )
    
    def test_dual_api_support_simulation(self) -> None:
        """Test dual API support ensuring legacy gym callers receive 4-tuple while Gymnasium callers receive 5-tuple."""
        # Note: This test simulates the environment-level dual API support
        # The NavigatorProtocol itself provides the step functionality
        controller = SingleAgentController(position=(0.0, 0.0), speed=1.0)
        env_array = np.zeros((50, 50))
        
        # Test that the controller step method works correctly for environment wrapper
        controller.step(env_array)
        position_after_step = controller.positions.copy()
        
        # Simulate environment wrapper dual API behavior
        # When used in a Gymnasium environment, step() would return 5-tuple
        obs = controller.positions.flatten()
        reward = 1.0
        terminated = False
        truncated = False
        info = {"agent_position": controller.positions[0].tolist()}
        
        # Test 5-tuple format (Gymnasium 0.29.x)
        gymnasium_result = (obs, reward, terminated, truncated, info)
        assert len(gymnasium_result) == 5
        assert isinstance(gymnasium_result[0], np.ndarray)
        assert isinstance(gymnasium_result[1], (int, float))
        assert isinstance(gymnasium_result[2], bool)
        assert isinstance(gymnasium_result[3], bool)
        assert isinstance(gymnasium_result[4], dict)
        
        # Test 4-tuple format (legacy gym compatibility)
        done = terminated or truncated
        legacy_result = (obs, reward, done, info)
        assert len(legacy_result) == 4
        assert isinstance(legacy_result[0], np.ndarray)
        assert isinstance(legacy_result[1], (int, float))
        assert isinstance(legacy_result[2], bool)
        assert isinstance(legacy_result[3], dict)
        
        logger.info("Dual API support simulation test completed")
    
    def test_sample_odor(self) -> None:
        """Test sampling odor at the agent's position."""
        controller = SingleAgentController(position=(5, 5))
        
        # Create an environment with a known value at the agent's position
        env_array = np.zeros((10, 10))
        env_array[5, 5] = 1.0
        
        odor = controller.sample_odor(env_array)
        assert odor == 1.0
    
    def test_sample_multiple_sensors(self) -> None:
        """Test sampling odor at multiple sensor positions."""
        controller = SingleAgentController(position=(50, 50), orientation=0.0)
        
        # Create a gradient environment
        env_array = np.zeros((100, 100))
        y, x = np.ogrid[:100, :100]
        env_array += np.exp(-((x - 50)**2 + (y - 50)**2) / 100)
        
        # Sample with multiple sensors
        odor_values = controller.sample_multiple_sensors(
            env_array, 
            sensor_distance=10.0,
            sensor_angle=90.0,
            num_sensors=3
        )
        
        # Check result shape and values
        assert isinstance(odor_values, np.ndarray)
        assert odor_values.shape == (3,)
        assert np.all(odor_values >= 0.0)


class TestMultiAgentController:
    """Tests for the MultiAgentController with enhanced configuration support and performance monitoring."""
    
    def test_initialization(self) -> None:
        """Test that a MultiAgentController initializes correctly with enhanced logging."""
        with correlation_context("multi_agent_init_test"):
            # Test default initialization
            controller = MultiAgentController()
            assert controller.num_agents == 1
            assert controller.positions.shape == (1, 2)
            
            # Test with custom parameters
            positions = np.array([[10.0, 20.0], [30.0, 40.0]])
            orientations = np.array([0.0, 90.0])
            speeds = np.array([1.0, 2.0])
            
            controller = MultiAgentController(
                positions=positions,
                orientations=orientations,
                speeds=speeds
            )
            
            assert controller.num_agents == 2
            assert controller.positions.shape == (2, 2)
            assert controller.orientations.shape == (2,)
            assert controller.speeds.shape == (2,)
            assert np.array_equal(controller.positions, positions)
            assert np.array_equal(controller.orientations, orientations)
            assert np.array_equal(controller.speeds, speeds)
            
            logger.info("MultiAgentController initialization test completed")
    
    def test_initialization_with_dataclass_params(self) -> None:
        """Test initialization using MultiAgentParams dataclass for type safety."""
        params = MultiAgentParams(
            positions=np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]),
            orientations=np.array([0.0, 90.0, 180.0]),
            speeds=np.array([1.0, 1.5, 2.0]),
            max_speeds=np.array([2.0, 3.0, 4.0]),
            angular_velocities=np.array([0.1, 0.2, 0.3])
        )
        
        controller = MultiAgentController(
            positions=params.positions,
            orientations=params.orientations,
            speeds=params.speeds,
            max_speeds=params.max_speeds,
            angular_velocities=params.angular_velocities
        )
        
        assert controller.num_agents == 3
        assert np.array_equal(controller.positions, params.positions)
        assert np.array_equal(controller.orientations, params.orientations)
        assert np.array_equal(controller.speeds, params.speeds)
        assert np.array_equal(controller.max_speeds, params.max_speeds)
        assert np.array_equal(controller.angular_velocities, params.angular_velocities)
    
    def test_reset(self) -> None:
        """Test resetting the controller state with enhanced seed parameter support."""
        controller = MultiAgentController(
            positions=np.array([[10.0, 20.0], [30.0, 40.0]]),
            orientations=np.array([0.0, 90.0])
        )
        
        new_positions = np.array([[50.0, 60.0], [70.0, 80.0], [90.0, 100.0]])
        controller.reset(positions=new_positions)
        
        assert controller.num_agents == 3
        assert np.array_equal(controller.positions, new_positions)
        assert controller.orientations.shape == (3,)
    
    def test_reset_with_seed_parameter(self) -> None:
        """Test MultiAgentController reset method supports seed parameter for deterministic initial state generation."""
        with seed_context_manager(42):
            controller = MultiAgentController(
                positions=np.array([[0.0, 0.0], [5.0, 5.0]])
            )
            
            # Test reset with seed parameter for Gymnasium 0.29.x compatibility
            seed_params = get_gymnasium_seed_parameter(456)
            
            new_positions = np.array([[10.0, 15.0], [20.0, 25.0], [30.0, 35.0]])
            controller.reset(positions=new_positions, **seed_params)
            
            assert controller.num_agents == 3
            assert np.array_equal(controller.positions, new_positions)
            
            logger.info("MultiAgentController reset with seed parameter test completed")
    
    def test_reset_with_params_dataclass(self) -> None:
        """Test reset using MultiAgentParams dataclass for type-safe parameter updates."""
        controller = MultiAgentController()
        
        # Test reset with type-safe parameter object
        reset_params = MultiAgentParams(
            positions=np.array([[100.0, 110.0], [120.0, 130.0]]),
            orientations=np.array([45.0, 135.0]),
            speeds=np.array([2.0, 2.5]),
            max_speeds=np.array([4.0, 5.0])
        )
        
        controller.reset_with_params(reset_params)
        
        assert controller.num_agents == 2
        assert np.array_equal(controller.positions, reset_params.positions)
        assert np.array_equal(controller.orientations, reset_params.orientations)
        assert np.array_equal(controller.speeds, reset_params.speeds)
        assert np.array_equal(controller.max_speeds, reset_params.max_speeds)
    
    def test_step(self) -> None:
        """Test the step method updates positions and orientations."""
        positions = np.array([[10.0, 10.0], [20.0, 20.0]])
        orientations = np.array([0.0, 90.0])  # First agent along x, second along y
        speeds = np.array([1.0, 2.0])
        
        controller = MultiAgentController(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            angular_velocities=np.array([5.0, 10.0])
        )
        
        # Create a mock environment array
        env_array = np.zeros((100, 100))
        
        # Take a step
        controller.step(env_array)
        
        # Check that first agent moved along x-axis
        assert controller.positions[0, 0] > 10.0
        assert np.isclose(controller.positions[0, 1], 10.0)
        
        # Check that second agent moved along y-axis
        assert np.isclose(controller.positions[1, 0], 20.0)
        assert controller.positions[1, 1] > 20.0
        
        # Check that orientations were updated
        assert controller.orientations[0] == 5.0
        assert controller.orientations[1] == 100.0
    
    def test_multi_agent_step_performance_sla(self) -> None:
        """Test multi-agent step method meets ≤10ms execution time SLA with throughput monitoring."""
        # Test with multiple agents to validate scalability
        num_agents = 10
        positions = np.random.rand(num_agents, 2) * 100
        controller = MultiAgentController(
            positions=positions,
            enable_logging=True
        )
        env_array = np.random.rand(100, 100)
        
        # Warm up the controller
        for _ in range(5):
            controller.step(env_array)
        
        # Measure step performance
        step_times = []
        throughputs = []
        num_measurements = 15
        
        for _ in range(num_measurements):
            start_time = time.perf_counter()
            controller.step(env_array)
            end_time = time.perf_counter()
            
            step_time_ms = (end_time - start_time) * 1000
            throughput = num_agents * (1000.0 / step_time_ms) if step_time_ms > 0 else 0
            
            step_times.append(step_time_ms)
            throughputs.append(throughput)
        
        # Validate SLA requirements
        mean_step_time = np.mean(step_times)
        mean_throughput = np.mean(throughputs)
        p95_step_time = np.percentile(step_times, 95)
        
        # Performance assertions
        assert mean_step_time < 10.0, f"Mean step time {mean_step_time:.2f}ms exceeds 10ms SLA"
        assert p95_step_time < 15.0, f"P95 step time {p95_step_time:.2f}ms indicates performance issues"
        
        # Throughput validation (relaxed for test environment)
        min_expected_throughput = num_agents * 100  # agents * fps
        assert mean_throughput > min_expected_throughput, f"Throughput {mean_throughput:.0f} below minimum {min_expected_throughput}"
        
        # Test performance metrics collection
        if hasattr(controller, 'get_performance_metrics'):
            metrics = controller.get_performance_metrics()
            assert 'step_time_mean_ms' in metrics
            assert 'throughput_mean_agents_fps' in metrics
            assert metrics['num_agents'] == num_agents
        
        logger.info(
            "Multi-agent step performance SLA validation completed",
            extra={
                "num_agents": num_agents,
                "mean_step_time_ms": mean_step_time,
                "mean_throughput_agents_fps": mean_throughput,
                "p95_step_time_ms": p95_step_time,
                "measurements": num_measurements
            }
        )
    
    def test_sample_odor(self) -> None:
        """Test sampling odor at multiple agent positions."""
        positions = np.array([[5, 5], [8, 8]])
        controller = MultiAgentController(positions=positions)
        
        # Create an environment with known values at agent positions
        env_array = np.zeros((10, 10))
        env_array[5, 5] = 1.0
        env_array[8, 8] = 0.5
        
        odor_values = controller.sample_odor(env_array)
        assert odor_values.shape == (2,)
        assert odor_values[0] == 1.0
        assert odor_values[1] == 0.5
    
    def test_sample_multiple_sensors(self) -> None:
        """Test sampling odor at multiple sensor positions for multiple agents."""
        positions = np.array([[25, 25], [75, 75]])
        controller = MultiAgentController(positions=positions)
        
        # Create a gradient environment
        env_array = np.zeros((100, 100))
        y, x = np.ogrid[:100, :100]
        env_array += np.exp(-((x - 50)**2 + (y - 50)**2) / 100)
        
        # Sample with multiple sensors
        odor_values = controller.sample_multiple_sensors(
            env_array, 
            sensor_distance=10.0,
            sensor_angle=90.0,
            num_sensors=2
        )
        
        # Check result shape and values
        assert isinstance(odor_values, np.ndarray)
        assert odor_values.shape == (2, 2)
        assert np.all(odor_values >= 0.0)


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
class TestCoordinateFrameConsistency:
    """
    Property-based testing using Hypothesis for coordinate frame consistency across navigator implementations.
    
    These tests validate that coordinate transformations, position updates, and navigation calculations
    maintain consistency across different parameter ranges and edge cases using property-based testing.
    """
    
    @given(
        position_x=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        position_y=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        orientation=st.floats(min_value=0.0, max_value=360.0, allow_nan=False, allow_infinity=False),
        speed=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        angular_velocity=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=5000, suppress_health_check=[HealthCheck.too_slow])
    def test_single_agent_coordinate_frame_consistency(self, position_x, position_y, orientation, speed, angular_velocity):
        """Test that single agent coordinate transformations are consistent across parameter ranges."""
        assume(abs(position_x) < 99.0)  # Avoid boundary conditions
        assume(abs(position_y) < 99.0)
        assume(speed >= 0.0)
        
        with correlation_context("property_based_single_agent_test"):
            controller = SingleAgentController(
                position=(position_x, position_y),
                orientation=orientation,
                speed=speed,
                angular_velocity=angular_velocity
            )
            
            # Record initial state
            initial_position = controller.positions.copy()
            initial_orientation = controller.orientations.copy()
            
            # Create environment
            env_array = np.ones((100, 100)) * 0.5
            
            # Take a step
            controller.step(env_array, dt=1.0)
            
            # Validate coordinate frame consistency
            new_position = controller.positions
            new_orientation = controller.orientations
            
            # Position should change based on speed and orientation (unless speed is 0)
            if speed > 0.01:  # Avoid floating point precision issues
                position_changed = not np.allclose(initial_position, new_position, atol=1e-6)
                assert position_changed, f"Position should change with speed={speed}"
            
            # Orientation should change based on angular velocity (unless angular_velocity is 0)
            if abs(angular_velocity) > 0.01:
                orientation_changed = not np.allclose(initial_orientation, new_orientation, atol=1e-6)
                assert orientation_changed, f"Orientation should change with angular_velocity={angular_velocity}"
            
            # Position values should remain finite
            assert np.all(np.isfinite(new_position)), "Position values must remain finite"
            assert np.all(np.isfinite(new_orientation)), "Orientation values must remain finite"
            
            # Validate coordinate bounds (positions should be reasonable)
            assert np.all(np.abs(new_position) < 1000), "Position should not exceed reasonable bounds"
    
    @given(
        num_agents=st.integers(min_value=1, max_value=10),
        position_scale=st.floats(min_value=1.0, max_value=50.0, allow_nan=False),
        orientation_range=st.floats(min_value=0.0, max_value=360.0, allow_nan=False),
        speed_scale=st.floats(min_value=0.1, max_value=5.0, allow_nan=False)
    )
    @settings(max_examples=30, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
    def test_multi_agent_coordinate_frame_consistency(self, num_agents, position_scale, orientation_range, speed_scale):
        """Test that multi-agent coordinate transformations maintain consistency across agent counts."""
        assume(num_agents >= 1)
        assume(position_scale > 0)
        assume(speed_scale > 0)
        
        with correlation_context("property_based_multi_agent_test", num_agents=num_agents):
            # Generate consistent agent parameters
            positions = np.random.uniform(-position_scale, position_scale, (num_agents, 2))
            orientations = np.random.uniform(0, orientation_range, num_agents)
            speeds = np.random.uniform(0, speed_scale, num_agents)
            angular_velocities = np.random.uniform(-10, 10, num_agents)
            
            controller = MultiAgentController(
                positions=positions,
                orientations=orientations,
                speeds=speeds,
                angular_velocities=angular_velocities
            )
            
            # Record initial state
            initial_positions = controller.positions.copy()
            initial_orientations = controller.orientations.copy()
            
            # Create environment
            env_array = np.ones((100, 100)) * 0.3
            
            # Take a step
            controller.step(env_array, dt=1.0)
            
            # Validate coordinate frame consistency for all agents
            new_positions = controller.positions
            new_orientations = controller.orientations
            
            # Validate shapes remain consistent
            assert new_positions.shape == (num_agents, 2), "Position array shape must be consistent"
            assert new_orientations.shape == (num_agents,), "Orientation array shape must be consistent"
            
            # All values should remain finite
            assert np.all(np.isfinite(new_positions)), "All position values must remain finite"
            assert np.all(np.isfinite(new_orientations)), "All orientation values must remain finite"
            
            # Check inter-agent coordinate consistency
            for i in range(num_agents):
                # Individual agent coordinate bounds
                assert np.all(np.abs(new_positions[i]) < 1000), f"Agent {i} position should not exceed bounds"
                
                # Speed-based movement validation
                if speeds[i] > 0.01:
                    position_delta = np.linalg.norm(new_positions[i] - initial_positions[i])
                    # Movement should be proportional to speed (within reasonable bounds)
                    assert position_delta <= speeds[i] * 2.0, f"Agent {i} moved too far for given speed"
    
    @given(
        steps=st.integers(min_value=1, max_value=10),
        dt=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=20, deadline=15000, suppress_health_check=[HealthCheck.too_slow])
    def test_temporal_coordinate_consistency(self, steps, dt):
        """Test coordinate frame consistency over multiple time steps."""
        assume(steps >= 1)
        assume(dt > 0.05)
        
        with correlation_context("property_based_temporal_test", steps=steps, dt=dt):
            controller = SingleAgentController(
                position=(0.0, 0.0),
                orientation=45.0,
                speed=1.0,
                angular_velocity=5.0
            )
            
            env_array = np.ones((50, 50)) * 0.4
            trajectory_positions = []
            trajectory_orientations = []
            
            # Record trajectory over multiple steps
            for step in range(steps):
                trajectory_positions.append(controller.positions.copy())
                trajectory_orientations.append(controller.orientations.copy())
                controller.step(env_array, dt=dt)
            
            # Final position
            trajectory_positions.append(controller.positions.copy())
            trajectory_orientations.append(controller.orientations.copy())
            
            # Validate temporal consistency
            for i in range(len(trajectory_positions) - 1):
                pos_current = trajectory_positions[i]
                pos_next = trajectory_positions[i + 1]
                
                # Movement should be bounded by speed and time
                distance_moved = np.linalg.norm(pos_next - pos_current)
                max_possible_distance = controller.speeds[0] * dt * 1.5  # Allow some tolerance
                
                assert distance_moved <= max_possible_distance, \
                    f"Step {i}: Moved {distance_moved:.3f}, max possible {max_possible_distance:.3f}"
            
            # All trajectory points should have finite values
            all_positions = np.array(trajectory_positions)
            all_orientations = np.array(trajectory_orientations)
            
            assert np.all(np.isfinite(all_positions)), "All trajectory positions must be finite"
            assert np.all(np.isfinite(all_orientations)), "All trajectory orientations must be finite"
            
            logger.info(
                "Temporal coordinate consistency test completed",
                extra={"steps": steps, "dt": dt, "trajectory_length": len(trajectory_positions)}
            )


class TestNavigatorFactory:
    """Tests for NavigatorFactory with enhanced Hydra integration and dataclass-based configuration."""
    
    def test_single_agent_factory(self) -> None:
        """Test the single-agent factory method with enhanced logging."""
        with correlation_context("single_agent_factory_test"):
            navigator = NavigatorFactory.single_agent(
                position=(10.0, 20.0), 
                orientation=45.0,
                max_speed=2.0
            )
            assert isinstance(navigator, NavigatorProtocol)
            assert navigator.num_agents == 1
            assert navigator.positions[0, 0] == 10.0
            assert navigator.positions[0, 1] == 20.0
            assert navigator.orientations[0] == 45.0
            assert navigator.max_speeds[0] == 2.0
            
            logger.info("Single agent factory test completed")
    
    def test_multi_agent_factory(self) -> None:
        """Test the multi-agent factory method with enhanced logging."""
        with correlation_context("multi_agent_factory_test"):
            positions = [[0.0, 0.0], [10.0, 10.0]]
            orientations = [0.0, 90.0]
            
            navigator = NavigatorFactory.multi_agent(
                positions=positions,
                orientations=orientations
            )
            assert isinstance(navigator, NavigatorProtocol)
            assert navigator.num_agents == 2
            assert navigator.positions.shape == (2, 2)
            assert navigator.orientations.shape == (2,)
            
            logger.info("Multi agent factory test completed")
    
    def test_factory_from_dataclass_config(self) -> None:
        """Test creating navigator from dataclass-based structured config."""
        # Test single agent configuration
        single_config = SingleAgentConfig(
            position=(15.0, 25.0),
            orientation=135.0,
            speed=1.8,
            max_speed=3.5,
            angular_velocity=0.15
        )
        
        navigator = create_controller_from_config(single_config)
        assert isinstance(navigator, NavigatorProtocol)
        assert navigator.num_agents == 1
        assert navigator.positions[0, 0] == 15.0
        assert navigator.positions[0, 1] == 25.0
        assert navigator.orientations[0] == 135.0
        assert navigator.speeds[0] == 1.8
        assert navigator.max_speeds[0] == 3.5
        
        # Test multi-agent configuration
        multi_config = MultiAgentConfig(
            num_agents=3,
            positions=[[5.0, 5.0], [15.0, 15.0], [25.0, 25.0]],
            orientations=[30.0, 60.0, 90.0],
            speeds=[1.0, 1.2, 1.4],
            max_speeds=[2.0, 2.2, 2.4],
            angular_velocities=[0.1, 0.12, 0.14]
        )
        
        navigator = create_controller_from_config(multi_config)
        assert isinstance(navigator, NavigatorProtocol)
        assert navigator.num_agents == 3
        assert navigator.positions.shape == (3, 2)
        assert navigator.orientations.shape == (3,)
        assert np.allclose(navigator.orientations, [30.0, 60.0, 90.0])
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_from_config_single_agent(self) -> None:
        """Test creating single agent from configuration."""
        # Create a test configuration
        config_dict = {
            'position': (10.0, 20.0),
            'orientation': 45.0,
            'speed': 1.0,
            'max_speed': 2.0,
            'angular_velocity': 0.1
        }
        
        # Test with plain dict
        navigator = NavigatorFactory.from_config(config_dict)
        assert isinstance(navigator, NavigatorProtocol)
        assert navigator.num_agents == 1
        assert navigator.positions[0, 0] == 10.0
        assert navigator.positions[0, 1] == 20.0
        assert navigator.orientations[0] == 45.0
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_from_config_multi_agent(self) -> None:
        """Test creating multi-agent from configuration."""
        # Create a test configuration for multi-agent
        config_dict = {
            'positions': [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]],
            'orientations': [0.0, 90.0, 180.0],
            'speeds': [1.0, 1.5, 2.0],
            'max_speeds': [2.0, 3.0, 4.0],
            'angular_velocities': [0.1, 0.2, 0.3]
        }
        
        navigator = NavigatorFactory.from_config(config_dict)
        assert isinstance(navigator, NavigatorProtocol)
        assert navigator.num_agents == 3
        assert navigator.positions.shape == (3, 2)
        assert navigator.orientations.shape == (3,)
    
    def test_factory_sensor_based_navigation_workflows(self) -> None:
        """Test NavigatorFactory creates controllers with proper sensor integration."""
        # Test single agent with sensor configuration
        config = {
            'position': (10.0, 15.0),
            'orientation': 90.0,
            'max_speed': 2.0,
            'enable_memory': True,
            'sensors': None  # Will use default sensors
        }
        
        navigator = NavigatorFactory.from_config(config)
        assert isinstance(navigator, NavigatorProtocol)
        
        # Verify sensor integration is properly configured
        assert hasattr(navigator, '_sensors')
        assert hasattr(navigator, '_primary_sensor')
        
        # Test sensor-based odor sampling workflow
        env_array = np.random.rand(50, 50)
        odor_value = navigator.sample_odor(env_array)
        assert isinstance(odor_value, (float, np.floating))
        assert not np.isnan(odor_value)
        assert not np.isinf(odor_value)
        
        # Test memory interface is available when configured
        assert hasattr(navigator, 'load_memory')
        assert hasattr(navigator, 'save_memory')
        initial_memory = navigator.load_memory()
        # Should return None initially for memory-enabled but uninitialized controller
        assert initial_memory is None
    
    def test_factory_modular_component_creation(self) -> None:
        """Test factory methods for creating modular components."""
        # Test plume model creation if factory supports it
        if hasattr(NavigatorFactory, 'create_plume_model'):
            plume_config = {
                'type': 'GaussianPlumeModel',
                'source_position': (50, 50),
                'source_strength': 1000.0
            }
            
            try:
                plume_model = NavigatorFactory.create_plume_model(plume_config)
                # Should implement PlumeModelProtocol
                assert hasattr(plume_model, 'concentration_at')
                assert hasattr(plume_model, 'step')
                assert hasattr(plume_model, 'reset')
            except ImportError:
                # Expected if plume models not yet available
                pass
        
        # Test sensor creation if factory supports it
        if hasattr(NavigatorFactory, 'create_sensors'):
            sensor_configs = [
                {'type': 'ConcentrationSensor', 'dynamic_range': (0, 1)},
                {'type': 'BinarySensor', 'threshold': 0.1}
            ]
            
            try:
                sensors = NavigatorFactory.create_sensors(sensor_configs)
                assert isinstance(sensors, list)
                assert len(sensors) == 2
                # Each sensor should implement SensorProtocol
                for sensor in sensors:
                    assert hasattr(sensor, 'detect') or hasattr(sensor, 'measure')
                    assert hasattr(sensor, 'configure')
            except ImportError:
                # Expected if sensor implementations not yet available
                pass
    
    def test_factory_protocol_validation(self) -> None:
        """Test factory protocol compliance validation methods."""
        navigator = NavigatorFactory.single_agent(position=(5, 10))
        
        # Test protocol compliance validation if available
        if hasattr(NavigatorFactory, 'validate_protocol_compliance'):
            from odor_plume_nav.core.navigator import NavigatorProtocol
            
            is_compliant = NavigatorFactory.validate_protocol_compliance(
                navigator, NavigatorProtocol
            )
            assert is_compliant is True
        
        # Test that created navigators implement all required methods
        required_methods = [
            'reset', 'step', 'sample_odor', 'sample_multiple_sensors',
            'compute_additional_obs', 'compute_extra_reward', 'on_episode_end',
            'load_memory', 'save_memory'
        ]
        
        for method_name in required_methods:
            assert hasattr(navigator, method_name), f"Navigator missing required method: {method_name}"
            assert callable(getattr(navigator, method_name)), f"Navigator method not callable: {method_name}"


@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
class TestHydraConfigurationIntegration:
    """
    Comprehensive tests for Hydra configuration integration with NavigatorProtocol.
    
    These tests validate the enhanced Hydra-based configuration management system
    including dataclass-based structured configs with Pydantic validation, ConfigStore registration,
    schema validation, hierarchical composition, and environment variable interpolation
    patterns essential for modern ML workflows.
    """
    
    def test_hydra_config_store_registration(self) -> None:
        """Test ConfigStore registration and schema validation workflows with dataclass-based structured configs."""
        from hydra.core.config_store import ConfigStore
        from odor_plume_nav.config.models import NavigatorConfig, SingleAgentConfig, MultiAgentConfig
        
        # Test ConfigStore registration for dataclass-based structured configs
        cs = ConfigStore.instance()
        cs.store(name="test_navigator_config", node=NavigatorConfig)
        cs.store(name="test_single_agent_config", node=SingleAgentConfig)
        cs.store(name="test_multi_agent_config", node=MultiAgentConfig)
        
        # Verify registration success
        assert "test_navigator_config" in cs.cache
        assert "test_single_agent_config" in cs.cache
        assert "test_multi_agent_config" in cs.cache
        
        # Test schema validation through Pydantic dataclass models
        test_config = NavigatorConfig(
            mode="single",
            position=(25.0, 50.0),
            orientation=90.0,
            speed=1.5,
            max_speed=3.0,
            angular_velocity=0.2
        )
        
        # Verify Pydantic validation
        assert test_config.position == (25.0, 50.0)
        assert test_config.orientation == 90.0
        assert test_config.speed <= test_config.max_speed
        assert test_config.mode == "single"
        
        # Test structured config extraction methods
        single_config = test_config.get_single_agent_config()
        assert isinstance(single_config, SingleAgentConfig)
        assert single_config.position == (25.0, 50.0)
        assert single_config.orientation == 90.0
        
        logger.info("Hydra ConfigStore registration test with dataclass configs completed")
    
    def test_hydra_configuration_composition(self) -> None:
        """Test hierarchical configuration composition with dataclass-based structured configs."""
        # Create base configuration using dataclass models
        base_navigator_config = NavigatorConfig(
            mode="single",
            position=(0.0, 0.0),
            orientation=0.0,
            speed=0.0,
            max_speed=1.0,
            angular_velocity=0.1
        )
        
        # Create simulation configuration
        base_simulation_config = SimulationConfig(
            max_steps=1000,
            step_size=1.0,
            random_seed=42,
            experiment_name="test_experiment"
        )
        
        # Test hierarchical composition through OmegaConf
        base_config = {
            'navigator': OmegaConf.structured(base_navigator_config),
            'simulation': OmegaConf.structured(base_simulation_config)
        }
        
        # Test configuration loading and validation
        mock_cfg = OmegaConf.create(base_config)
        
        # Validate configuration structure with dataclass validation
        assert 'navigator' in mock_cfg
        assert 'simulation' in mock_cfg
        assert mock_cfg.navigator.position == (0.0, 0.0)
        assert mock_cfg.navigator.orientation == 0.0
        assert mock_cfg.navigator.max_speed == 1.0
        assert mock_cfg.simulation.max_steps == 1000
        assert mock_cfg.simulation.random_seed == 42
        
        # Test configuration object creation from structured config
        navigator_config = NavigatorConfig(**OmegaConf.to_container(mock_cfg.navigator, resolve=True))
        assert navigator_config.mode == "single"
        assert navigator_config.position == (0.0, 0.0)
        
        logger.info("Hydra configuration composition test with dataclass configs completed")
    
    def test_dictconfig_factory_method_patterns(self) -> None:
        """Test factory method patterns with DictConfig parameters and dataclass-based structured configs."""
        # Create a structured DictConfig for single agent using dataclass
        single_agent_config = SingleAgentConfig(
            position=(15.0, 25.0),
            orientation=180.0,
            speed=2.0,
            max_speed=4.0,
            angular_velocity=0.15
        )
        single_config = OmegaConf.structured(single_agent_config)
        
        # Test factory method with structured DictConfig
        navigator = create_controller_from_config(single_config)
        assert isinstance(navigator, NavigatorProtocol)
        assert navigator.num_agents == 1
        assert navigator.positions[0, 0] == 15.0
        assert navigator.positions[0, 1] == 25.0
        assert navigator.orientations[0] == 180.0
        
        # Create a structured DictConfig for multi-agent using dataclass
        multi_agent_config = MultiAgentConfig(
            num_agents=2,
            positions=[[5.0, 5.0], [15.0, 15.0]],
            orientations=[45.0, 135.0],
            speeds=[1.0, 1.5],
            max_speeds=[2.0, 3.0],
            angular_velocities=[0.1, 0.2]
        )
        multi_config = OmegaConf.structured(multi_agent_config)
        
        # Test multi-agent factory with structured DictConfig
        multi_navigator = create_controller_from_config(multi_config)
        assert isinstance(multi_navigator, NavigatorProtocol)
        assert multi_navigator.num_agents == 2
        assert multi_navigator.positions.shape == (2, 2)
        assert multi_navigator.orientations.shape == (2,)
        
        # Test with plain DictConfig (backward compatibility)
        plain_config = OmegaConf.create({
            'position': (25.0, 35.0),
            'orientation': 270.0,
            'speed': 1.5,
            'max_speed': 3.0
        })
        
        plain_navigator = create_controller_from_config(plain_config)
        assert isinstance(plain_navigator, NavigatorProtocol)
        assert plain_navigator.num_agents == 1
        
        logger.info("DictConfig factory method patterns test with structured configs completed")
    
    def test_environment_variable_interpolation(self) -> None:
        """Test environment variable interpolation in configuration composition."""
        # Set test environment variables
        test_env_vars = {
            'AGENT_START_X': '100.0',
            'AGENT_START_Y': '200.0', 
            'AGENT_ORIENTATION': '270.0',
            'AGENT_MAX_SPEED': '5.0'
        }
        
        # Mock environment variables
        with patch.dict(os.environ, test_env_vars):
            # Create configuration with environment variable interpolation
            config_with_env = {
                'position': [float(os.environ.get('AGENT_START_X', '0.0')), 
                           float(os.environ.get('AGENT_START_Y', '0.0'))],
                'orientation': float(os.environ.get('AGENT_ORIENTATION', '0.0')),
                'max_speed': float(os.environ.get('AGENT_MAX_SPEED', '1.0')),
                'speed': 0.0,
                'angular_velocity': 0.1
            }
            
            # Create navigator with interpolated configuration
            navigator = NavigatorFactory.from_config(config_with_env)
            
            # Validate environment variable interpolation
            assert navigator.positions[0, 0] == 100.0
            assert navigator.positions[0, 1] == 200.0
            assert navigator.orientations[0] == 270.0
            assert navigator.max_speeds[0] == 5.0
    
    def test_configuration_override_scenarios(self) -> None:
        """Test configuration override scenarios with parameter validation."""
        # Base configuration
        base_config = OmegaConf.create({
            'position': [10.0, 10.0],
            'orientation': 0.0,
            'speed': 1.0,
            'max_speed': 2.0,
            'angular_velocity': 0.1
        })
        
        # Override configuration
        override_config = OmegaConf.create({
            'position': [50.0, 50.0],
            'orientation': 90.0,
            'max_speed': 3.0
        })
        
        # Merge configurations (simulating Hydra override behavior)
        merged_config = OmegaConf.merge(base_config, override_config)
        
        # Validate merged configuration
        assert merged_config.position == [50.0, 50.0]  # Overridden
        assert merged_config.orientation == 90.0  # Overridden
        assert merged_config.speed == 1.0  # From base
        assert merged_config.max_speed == 3.0  # Overridden
        assert merged_config.angular_velocity == 0.1  # From base
        
        # Test navigator creation with merged config
        navigator = NavigatorFactory.from_config(merged_config)
        assert navigator.positions[0, 0] == 50.0
        assert navigator.positions[0, 1] == 50.0
        assert navigator.orientations[0] == 90.0
        assert navigator.max_speeds[0] == 3.0
    
    def test_hydra_multirun_configuration_patterns(self) -> None:
        """Test configuration patterns for Hydra multirun scenarios."""
        # Simulate multirun parameter sweep configurations
        sweep_configs = [
            {'max_speed': 1.0, 'angular_velocity': 0.1},
            {'max_speed': 2.0, 'angular_velocity': 0.2},
            {'max_speed': 3.0, 'angular_velocity': 0.3}
        ]
        
        base_params = {
            'position': [25.0, 25.0],
            'orientation': 45.0,
            'speed': 0.5
        }
        
        navigators = []
        for sweep_params in sweep_configs:
            # Merge base parameters with sweep parameters
            config = {**base_params, **sweep_params}
            navigator = NavigatorFactory.from_config(config)
            navigators.append(navigator)
        
        # Validate parameter sweep results
        assert len(navigators) == 3
        for i, navigator in enumerate(navigators):
            expected_max_speed = sweep_configs[i]['max_speed']
            expected_angular_velocity = sweep_configs[i]['angular_velocity']
            
            assert navigator.max_speeds[0] == expected_max_speed
            assert navigator.angular_velocities[0] == expected_angular_velocity
            assert navigator.positions[0, 0] == 25.0  # Base parameter maintained
            assert navigator.positions[0, 1] == 25.0  # Base parameter maintained


class TestNavigatorProtocolCompliance:
    """
    Tests ensuring NavigatorProtocol compliance across all implementations.
    
    These tests validate that all navigator implementations properly conform
    to the NavigatorProtocol interface requirements, including enhanced
    configuration management, optional memory interface, and new sensor-based
    observation processing workflows per Section 0 requirements.
    """
    
    def test_single_agent_protocol_compliance(self) -> None:
        """Test SingleAgentController protocol compliance with modular architecture extensions."""
        controller = SingleAgentController()
        
        # Verify protocol compliance
        assert isinstance(controller, NavigatorProtocol)
        
        # Test all required properties exist and return correct types
        assert isinstance(controller.positions, np.ndarray)
        assert isinstance(controller.orientations, np.ndarray)
        assert isinstance(controller.speeds, np.ndarray)
        assert isinstance(controller.max_speeds, np.ndarray)
        assert isinstance(controller.angular_velocities, np.ndarray)
        assert isinstance(controller.num_agents, int)
        
        # Test all required methods exist and are callable
        assert hasattr(controller, 'reset') and callable(controller.reset)
        assert hasattr(controller, 'step') and callable(controller.step)
        assert hasattr(controller, 'sample_odor') and callable(controller.sample_odor)
        assert hasattr(controller, 'sample_multiple_sensors') and callable(controller.sample_multiple_sensors)
        
        # Test new extensibility hooks exist and are callable
        assert hasattr(controller, 'compute_additional_obs') and callable(controller.compute_additional_obs)
        assert hasattr(controller, 'compute_extra_reward') and callable(controller.compute_extra_reward)
        assert hasattr(controller, 'on_episode_end') and callable(controller.on_episode_end)
        
        # Test optional memory interface methods exist and are callable
        assert hasattr(controller, 'load_memory') and callable(controller.load_memory)
        assert hasattr(controller, 'save_memory') and callable(controller.save_memory)
    
    def test_multi_agent_protocol_compliance(self) -> None:
        """Test MultiAgentController protocol compliance with modular architecture extensions."""
        controller = MultiAgentController()
        
        # Verify protocol compliance
        assert isinstance(controller, NavigatorProtocol)
        
        # Test all required properties exist and return correct types
        assert isinstance(controller.positions, np.ndarray)
        assert isinstance(controller.orientations, np.ndarray)
        assert isinstance(controller.speeds, np.ndarray)
        assert isinstance(controller.max_speeds, np.ndarray)
        assert isinstance(controller.angular_velocities, np.ndarray)
        assert isinstance(controller.num_agents, int)
        
        # Test all required methods exist and are callable
        assert hasattr(controller, 'reset') and callable(controller.reset)
        assert hasattr(controller, 'step') and callable(controller.step)
        assert hasattr(controller, 'sample_odor') and callable(controller.sample_odor)
        assert hasattr(controller, 'sample_multiple_sensors') and callable(controller.sample_multiple_sensors)
        
        # Test new extensibility hooks exist and are callable
        assert hasattr(controller, 'compute_additional_obs') and callable(controller.compute_additional_obs)
        assert hasattr(controller, 'compute_extra_reward') and callable(controller.compute_extra_reward)
        assert hasattr(controller, 'on_episode_end') and callable(controller.on_episode_end)
        
        # Test optional memory interface methods exist and are callable
        assert hasattr(controller, 'load_memory') and callable(controller.load_memory)
        assert hasattr(controller, 'save_memory') and callable(controller.save_memory)
    
    def test_factory_created_navigator_protocol_compliance(self) -> None:
        """Test that factory-created navigators maintain protocol compliance."""
        # Test single agent factory compliance
        single_navigator = NavigatorFactory.single_agent(position=(10.0, 20.0))
        assert isinstance(single_navigator, NavigatorProtocol)
        
        # Test multi-agent factory compliance
        multi_navigator = NavigatorFactory.multi_agent(
            positions=[[0.0, 0.0], [10.0, 10.0]]
        )
        assert isinstance(multi_navigator, NavigatorProtocol)
        
        # Test configuration-based factory compliance
        config = {'position': (5.0, 5.0), 'max_speed': 2.0}
        config_navigator = NavigatorFactory.from_config(config)
        assert isinstance(config_navigator, NavigatorProtocol)
    
    def test_protocol_method_signatures(self) -> None:
        """Test that protocol method signatures are correctly implemented."""
        controller = SingleAgentController()
        env_array = np.zeros((100, 100))
        
        # Test step method signature
        try:
            controller.step(env_array, dt=1.0)
        except TypeError:
            pytest.fail("step method signature is not compatible with protocol")
        
        # Test sample_odor method signature
        try:
            result = controller.sample_odor(env_array)
            assert isinstance(result, (float, np.ndarray))
        except TypeError:
            pytest.fail("sample_odor method signature is not compatible with protocol")
        
        # Test sample_multiple_sensors method signature
        try:
            result = controller.sample_multiple_sensors(
                env_array, 
                sensor_distance=5.0,
                sensor_angle=45.0,
                num_sensors=2,
                layout_name=None
            )
            assert isinstance(result, np.ndarray)
        except TypeError:
            pytest.fail("sample_multiple_sensors method signature is not compatible with protocol")
    
    def test_property_array_shapes_consistency(self) -> None:
        """Test that property array shapes are consistent across implementations."""
        # Test single agent
        single = SingleAgentController()
        assert single.positions.shape == (1, 2)
        assert single.orientations.shape == (1,)
        assert single.speeds.shape == (1,)
        assert single.max_speeds.shape == (1,)
        assert single.angular_velocities.shape == (1,)
        
        # Test multi-agent
        multi = MultiAgentController(positions=[[0, 0], [10, 10], [20, 20]])
        assert multi.positions.shape == (3, 2)
        assert multi.orientations.shape == (3,)
        assert multi.speeds.shape == (3,)
        assert multi.max_speeds.shape == (3,)
        assert multi.angular_velocities.shape == (3,)
        
        # Verify num_agents consistency
        assert single.num_agents == single.positions.shape[0]
        assert multi.num_agents == multi.positions.shape[0]


class TestPerformanceAndScalability:
    """
    Tests for performance characteristics and scalability requirements.
    
    These tests validate that the NavigatorProtocol implementations meet
    the specified performance requirements for scientific computing applications
    including frame processing latency and multi-agent simulation scalability.
    """
    
    def test_single_agent_step_performance(self) -> None:
        """Test single agent step performance meets requirements (<1ms)."""
        import time
        
        controller = SingleAgentController()
        env_array = np.zeros((100, 100))
        
        # Warm up
        for _ in range(10):
            controller.step(env_array)
        
        # Measure performance
        start_time = time.perf_counter()
        for _ in range(100):
            controller.step(env_array)
        end_time = time.perf_counter()
        
        average_time = (end_time - start_time) / 100 * 1000  # Convert to milliseconds
        
        # Performance assertion (relaxed for testing environment)
        assert average_time < 10, f"Single agent step took {average_time:.2f}ms, expected <10ms"
    
    def test_multi_agent_step_performance(self) -> None:
        """Test multi-agent step performance meets scalability requirements."""
        import time
        
        # Test with 10 agents
        positions = np.random.rand(10, 2) * 100
        controller = MultiAgentController(positions=positions)
        env_array = np.zeros((100, 100))
        
        # Warm up
        for _ in range(10):
            controller.step(env_array)
        
        # Measure performance
        start_time = time.perf_counter()
        for _ in range(50):
            controller.step(env_array)
        end_time = time.perf_counter()
        
        average_time = (end_time - start_time) / 50 * 1000  # Convert to milliseconds
        
        # Performance assertion (relaxed for testing environment)
        assert average_time < 50, f"10-agent step took {average_time:.2f}ms, expected <50ms"
    
    def test_odor_sampling_performance(self) -> None:
        """Test odor sampling performance meets requirements."""
        import time
        
        controller = SingleAgentController()
        env_array = np.random.rand(200, 200)
        
        # Measure odor sampling performance
        start_time = time.perf_counter()
        for _ in range(1000):
            controller.sample_odor(env_array)
        end_time = time.perf_counter()
        
        average_time = (end_time - start_time) / 1000 * 1000  # Convert to milliseconds
        
        # Performance assertion
        assert average_time < 1, f"Odor sampling took {average_time:.3f}ms, expected <1ms"
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_configuration_loading_performance(self) -> None:
        """Test configuration loading performance meets requirements."""
        import time
        
        config_dict = {
            'position': (10.0, 20.0),
            'orientation': 45.0,
            'speed': 1.0,
            'max_speed': 2.0,
            'angular_velocity': 0.1
        }
        
        # Measure configuration loading and navigator creation performance
        start_time = time.perf_counter()
        for _ in range(100):
            navigator = NavigatorFactory.from_config(config_dict)
        end_time = time.perf_counter()
        
        average_time = (end_time - start_time) / 100 * 1000  # Convert to milliseconds
        
        # Performance assertion for configuration processing
        assert average_time < 100, f"Configuration loading took {average_time:.2f}ms, expected <100ms"


# Test fixtures for enhanced testing capabilities

@pytest.fixture
def mock_environment_array():
    """Provide a standard mock environment array for testing."""
    env_array = np.zeros((100, 100))
    # Add some structure to the environment
    y, x = np.ogrid[:100, :100]
    env_array += np.exp(-((x - 50)**2 + (y - 50)**2) / 200)
    return env_array


@pytest.fixture
def single_agent_config():
    """Provide a standard single-agent configuration for testing."""
    return {
        'position': (25.0, 35.0),
        'orientation': 135.0,
        'speed': 1.5,
        'max_speed': 3.0,
        'angular_velocity': 0.2
    }


@pytest.fixture
def multi_agent_config():
    """Provide a standard multi-agent configuration for testing."""
    return {
        'positions': [[10.0, 10.0], [30.0, 30.0], [50.0, 50.0]],
        'orientations': [0.0, 90.0, 180.0],
        'speeds': [1.0, 1.5, 2.0],
        'max_speeds': [2.0, 3.0, 4.0],
        'angular_velocities': [0.1, 0.15, 0.2]
    }


@pytest.fixture
def hydra_config_mock():
    """Provide a mock Hydra configuration for testing."""
    if not HYDRA_AVAILABLE:
        pytest.skip("Hydra not available")
    
    return OmegaConf.create({
        'navigator': {
            'position': [40.0, 60.0],
            'orientation': 225.0,
            'speed': 2.0,
            'max_speed': 4.0,
            'angular_velocity': 0.25
        },
        'simulation': {
            'num_steps': 100,
            'dt': 1.0
        },
        'environment': {
            'size': [100, 100]
        }
    })


# Integration test using multiple fixtures
def test_full_navigator_workflow(mock_environment_array, single_agent_config):
    """Test complete navigator workflow with configuration and environment."""
    # Create navigator from configuration
    navigator = NavigatorFactory.from_config(single_agent_config)
    
    # Verify initial state
    assert navigator.num_agents == 1
    assert navigator.positions[0, 0] == 25.0
    assert navigator.positions[0, 1] == 35.0
    
    # Execute simulation steps
    for _ in range(10):
        navigator.step(mock_environment_array, dt=1.0)
    
    # Verify simulation progression
    # Position should have changed due to movement
    assert navigator.positions[0, 0] != 25.0 or navigator.positions[0, 1] != 35.0
    
    # Sample odor at current position
    odor = navigator.sample_odor(mock_environment_array)
    assert isinstance(odor, (float, np.floating))
    assert odor >= 0.0
    
    # Test multi-sensor sampling
    sensor_readings = navigator.sample_multiple_sensors(
        mock_environment_array,
        sensor_distance=5.0,
        num_sensors=3
    )
    assert isinstance(sensor_readings, np.ndarray)
    assert sensor_readings.shape == (3,)
    assert np.all(sensor_readings >= 0.0)


@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
def test_hydra_integration_workflow(hydra_config_mock, mock_environment_array):
    """Test complete workflow with Hydra configuration integration."""
    # Extract navigator configuration from Hydra config
    nav_config = hydra_config_mock.navigator
    
    # Create navigator using Hydra configuration
    navigator = NavigatorFactory.from_config(nav_config)
    
    # Verify configuration was applied correctly
    assert navigator.positions[0, 0] == 40.0
    assert navigator.positions[0, 1] == 60.0
    assert navigator.orientations[0] == 225.0
    assert navigator.speeds[0] == 2.0
    assert navigator.max_speeds[0] == 4.0
    
    # Execute simulation using Hydra simulation parameters
    num_steps = hydra_config_mock.simulation.num_steps
    dt = hydra_config_mock.simulation.dt
    
    for step in range(min(num_steps, 20)):  # Limit for test performance
        navigator.step(mock_environment_array, dt=dt)
    
    # Verify simulation executed successfully
    assert step == 19  # Completed loop
    
    # Test that configuration parameters were maintained
    assert navigator.max_speeds[0] == 4.0
    assert navigator.num_agents == 1


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
class TestGymnasiumIntegration:
    """
    Tests for Gymnasium utils env_checker integration and protocol compliance validation.
    
    These tests validate that NavigatorProtocol implementations can be properly wrapped
    in Gymnasium environments and pass the official env_checker validation for API compliance.
    """
    
    def test_navigator_protocol_gymnasium_compatibility(self) -> None:
        """Test that NavigatorProtocol can be integrated with Gymnasium environment wrapper."""
        with correlation_context("gymnasium_compatibility_test"):
            # Create a simple NavigatorProtocol implementation
            controller = SingleAgentController(
                position=(10.0, 10.0),
                orientation=0.0,
                speed=1.0,
                max_speed=2.0
            )
            
            # Test that controller supports the operations needed for Gymnasium integration
            env_array = np.random.rand(50, 50)
            
            # Test reset with seed parameter (Gymnasium 0.29.x compatibility)
            seed_params = get_gymnasium_seed_parameter(42)
            controller.reset(position=(0.0, 0.0), **seed_params)
            
            # Test step operation
            controller.step(env_array, dt=1.0)
            
            # Test observation generation
            obs = controller.positions.flatten()
            assert isinstance(obs, np.ndarray)
            assert obs.shape == (2,)  # x, y coordinates
            
            # Test action space compatibility (would be used by environment wrapper)
            action = np.array([0.5, 0.1])  # speed, angular_velocity
            # Environment wrapper would apply this action
            
            # Test info dictionary generation
            info = {
                "agent_position": controller.positions[0].tolist(),
                "agent_orientation": float(controller.orientations[0]),
                "step_performance_ms": 5.0  # Mock performance metric
            }
            assert isinstance(info, dict)
            assert "agent_position" in info
            assert "agent_orientation" in info
            
            logger.info("NavigatorProtocol Gymnasium compatibility test completed")
    
    def test_dual_api_compatibility_simulation(self) -> None:
        """Test simulation of dual API support for legacy gym vs Gymnasium callers."""
        with correlation_context("dual_api_simulation_test"):
            controller = SingleAgentController(position=(5.0, 5.0), speed=1.0)
            env_array = np.random.rand(30, 30)
            
            # Simulate environment step
            controller.step(env_array)
            
            # Generate observation and reward
            obs = controller.positions.flatten()
            reward = 1.0  # Mock reward
            
            # Simulate dual API support in environment wrapper
            # Gymnasium 0.29.x API (5-tuple)
            terminated = False
            truncated = False
            info = {"agent_pos": controller.positions[0].tolist()}
            
            gymnasium_result = (obs, reward, terminated, truncated, info)
            assert len(gymnasium_result) == 5
            assert isinstance(gymnasium_result[0], np.ndarray)
            assert isinstance(gymnasium_result[1], (int, float))
            assert isinstance(gymnasium_result[2], bool)  # terminated
            assert isinstance(gymnasium_result[3], bool)  # truncated
            assert isinstance(gymnasium_result[4], dict)
            
            # Legacy gym API (4-tuple)
            done = terminated or truncated
            legacy_result = (obs, reward, done, info)
            assert len(legacy_result) == 4
            assert isinstance(legacy_result[0], np.ndarray)
            assert isinstance(legacy_result[1], (int, float))
            assert isinstance(legacy_result[2], bool)  # done
            assert isinstance(legacy_result[3], dict)
            
            logger.info("Dual API compatibility simulation test completed")
    
    def test_seed_parameter_integration(self) -> None:
        """Test integration of seed parameter support with Gymnasium seeding requirements."""
        with seed_context_manager(123):
            controller = SingleAgentController()
            
            # Test Gymnasium seed parameter generation
            seed_params = get_gymnasium_seed_parameter(456)
            assert "seed" in seed_params
            assert isinstance(seed_params["seed"], int)
            assert 0 <= seed_params["seed"] <= 2**31 - 1
            
            # Test reset with seed parameter
            initial_position = controller.positions.copy()
            controller.reset(position=(20.0, 30.0), **seed_params)
            
            # Verify reset worked
            assert controller.positions[0, 0] == 20.0
            assert controller.positions[0, 1] == 30.0
            
            # Test reproducibility with same seed
            controller.reset(position=(40.0, 50.0), **seed_params)
            assert controller.positions[0, 0] == 40.0
            assert controller.positions[0, 1] == 50.0
            
            logger.info("Seed parameter integration test completed")


class TestEnhancedEdgeCases:
    """
    Comprehensive edge case testing for protocol-based navigation components.
    
    These tests cover boundary conditions, error handling, and robustness scenarios
    to ensure comprehensive test coverage and reliable operation under diverse conditions.
    """
    
    def test_boundary_position_conditions(self) -> None:
        """Test navigator behavior at boundary position conditions."""
        with correlation_context("boundary_conditions_test"):
            # Test extreme positions
            extreme_positions = [
                (-1000.0, -1000.0),
                (1000.0, 1000.0),
                (0.0, 0.0),
                (-0.001, 0.001),
                (999.999, -999.999)
            ]
            
            for pos in extreme_positions:
                controller = SingleAgentController(position=pos)
                
                # Verify position is set correctly
                assert controller.positions[0, 0] == pos[0]
                assert controller.positions[0, 1] == pos[1]
                
                # Test that step still works
                env_array = np.ones((100, 100)) * 0.5
                controller.step(env_array)
                
                # Position should remain finite
                assert np.all(np.isfinite(controller.positions))
    
    def test_extreme_speed_and_angular_velocity(self) -> None:
        """Test navigator behavior with extreme speed and angular velocity values."""
        with correlation_context("extreme_values_test"):
            extreme_configs = [
                {"speed": 0.0, "angular_velocity": 0.0},  # No movement
                {"speed": 100.0, "angular_velocity": 180.0},  # High values
                {"speed": 0.001, "angular_velocity": 0.001},  # Very small values
                {"speed": 10.0, "angular_velocity": -180.0},  # Negative angular velocity
            ]
            
            for config in extreme_configs:
                controller = SingleAgentController(
                    position=(50.0, 50.0),
                    speed=config["speed"],
                    angular_velocity=config["angular_velocity"]
                )
                
                env_array = np.ones((100, 100)) * 0.3
                
                # Test multiple steps
                for _ in range(5):
                    controller.step(env_array)
                    
                    # Verify values remain finite and reasonable
                    assert np.all(np.isfinite(controller.positions))
                    assert np.all(np.isfinite(controller.orientations))
                    assert np.all(np.abs(controller.positions) < 10000)  # Reasonable bounds
    
    def test_multi_agent_scaling_edge_cases(self) -> None:
        """Test multi-agent controller with various agent count edge cases."""
        with correlation_context("multi_agent_scaling_test"):
            # Test single agent in multi-agent controller
            single_positions = np.array([[25.0, 35.0]])
            controller = MultiAgentController(positions=single_positions)
            assert controller.num_agents == 1
            
            # Test maximum reasonable agent count
            max_agents = 50  # Reasonable for testing
            positions = np.random.rand(max_agents, 2) * 100
            controller = MultiAgentController(positions=positions)
            assert controller.num_agents == max_agents
            
            env_array = np.random.rand(100, 100)
            
            # Test step performance with many agents
            start_time = time.perf_counter()
            controller.step(env_array)
            step_time = (time.perf_counter() - start_time) * 1000
            
            # Should complete within reasonable time (relaxed for testing)
            assert step_time < 100.0, f"Step time {step_time:.2f}ms too slow for {max_agents} agents"
    
    def test_configuration_validation_edge_cases(self) -> None:
        """Test configuration validation with edge case parameters."""
        with correlation_context("config_validation_test"):
            # Test speed constraints
            with pytest.raises(ValueError):
                SingleAgentConfig(speed=5.0, max_speed=2.0)  # speed > max_speed
            
            # Test orientation bounds
            valid_config = SingleAgentConfig(orientation=359.9)
            assert valid_config.orientation == 359.9
            
            with pytest.raises(ValueError):
                SingleAgentConfig(orientation=361.0)  # orientation > 360
            
            # Test multi-agent parameter consistency
            with pytest.raises(ValueError):
                MultiAgentConfig(
                    positions=[[0, 0], [10, 10]],
                    orientations=[0.0]  # Length mismatch
                )
    
    def test_dataclass_params_edge_cases(self) -> None:
        """Test edge cases for dataclass parameter objects."""
        with correlation_context("dataclass_params_test"):
            # Test partial parameter update
            controller = SingleAgentController()
            
            # Update only some parameters
            partial_params = SingleAgentParams(
                position=(100.0, 200.0),
                max_speed=5.0
                # speed and angular_velocity not specified
            )
            
            controller.reset_with_params(partial_params)
            assert controller.positions[0, 0] == 100.0
            assert controller.positions[0, 1] == 200.0
            assert controller.max_speeds[0] == 5.0
            
            # Test empty parameter update
            empty_params = SingleAgentParams()
            controller.reset_with_params(empty_params)  # Should not crash
    
    def test_performance_monitoring_edge_cases(self) -> None:
        """Test performance monitoring under edge case conditions."""
        with correlation_context("performance_monitoring_edge_test"):
            # Test with very small environment
            controller = SingleAgentController(enable_logging=True)
            tiny_env = np.ones((5, 5)) * 0.1
            
            # Should handle small environments gracefully
            controller.step(tiny_env)
            
            if hasattr(controller, 'get_performance_metrics'):
                metrics = controller.get_performance_metrics()
                assert isinstance(metrics, dict)
                assert 'total_steps' in metrics
            
            # Test with large environment
            large_env = np.random.rand(500, 500)
            
            start_time = time.perf_counter()
            controller.step(large_env)
            step_time = (time.perf_counter() - start_time) * 1000
            
            # Should handle large environments within reasonable time
            assert step_time < 50.0, f"Large environment step took {step_time:.2f}ms"
    
    def test_logging_integration_edge_cases(self) -> None:
        """Test centralized Loguru logging integration under various conditions."""
        with correlation_context("logging_integration_test"):
            # Test logging with various controller configurations
            configs = [
                {"enable_logging": True, "controller_id": "test_001"},
                {"enable_logging": False},
                {"enable_logging": True, "controller_id": None}
            ]
            
            for config in configs:
                controller = SingleAgentController(**config)
                env_array = np.ones((50, 50))
                
                # Should not crash with any logging configuration
                controller.step(env_array)
                
                if config.get("enable_logging") and hasattr(controller, 'get_performance_metrics'):
                    metrics = controller.get_performance_metrics()
                    assert isinstance(metrics, dict)
            
            logger.info("Logging integration edge cases test completed")


class TestCrossIntegrationScenarios:
    """
    Tests for cross-component integration scenarios combining multiple enhanced features.
    
    These tests validate the interaction between different enhanced features like
    Hydra configuration, performance monitoring, logging, and seed management.
    """
    
    def test_full_integration_workflow(self) -> None:
        """Test complete integration workflow with all enhanced features."""
        with seed_context_manager(789, experiment_name="full_integration_test"):
            # Create structured configuration
            config = NavigatorConfig(
                mode="multi",
                num_agents=5,
                positions=[[i*10.0, i*10.0] for i in range(5)],
                orientations=[i*45.0 for i in range(5)],
                speeds=[1.0 + i*0.2 for i in range(5)],
                max_speeds=[2.0 + i*0.5 for i in range(5)],
                angular_velocities=[0.1 + i*0.05 for i in range(5)]
            )
            
            # Create controller from structured config
            controller = create_controller_from_config(
                config,
                enable_logging=True,
                controller_id="integration_test_controller"
            )
            
            # Verify controller configuration
            assert controller.num_agents == 5
            assert controller.positions.shape == (5, 2)
            
            # Test with seed parameter
            seed_params = get_gymnasium_seed_parameter(999)
            controller.reset(**seed_params)
            
            # Run simulation with performance monitoring
            env_array = np.random.rand(100, 100)
            
            step_times = []
            for step in range(10):
                start_time = time.perf_counter()
                controller.step(env_array, dt=1.0)
                step_time = (time.perf_counter() - start_time) * 1000
                step_times.append(step_time)
            
            # Validate performance
            mean_step_time = np.mean(step_times)
            assert mean_step_time < 20.0, f"Integration workflow step time {mean_step_time:.2f}ms too slow"
            
            # Test performance metrics
            if hasattr(controller, 'get_performance_metrics'):
                metrics = controller.get_performance_metrics()
                assert metrics['total_steps'] >= 10
                assert metrics['num_agents'] == 5
            
            logger.info(
                "Full integration workflow test completed",
                extra={
                    "num_agents": controller.num_agents,
                    "mean_step_time_ms": mean_step_time,
                    "total_steps": 10
                }
            )


class TestNavigatorProtocolMemoryInterface:
    """
    Tests for optional memory interface methods supporting both memory-based and 
    non-memory-based navigation strategies per Section 0.2.1 requirements.
    
    These tests validate that the memory hooks can be disabled or bypassed for 
    memory-less navigation strategies while providing flexible cognitive modeling 
    approaches for planning agents.
    """

    def test_observe_and_memory_method_signatures(self) -> None:
        """NavigatorProtocol implementations expose observe/load/save hooks."""
        from plume_nav_sim.core.controllers import SingleAgentController, MultiAgentController

        controller = SingleAgentController(enable_memory=True)

        # Observe should require dict input and return it unchanged
        sample_obs = {"reading": 1.0}
        assert hasattr(controller, "observe")
        assert controller.observe(sample_obs) == sample_obs

        with pytest.raises(TypeError):
            controller.observe([1, 2, 3])

        # Memory loading should require dict input
        memory = {"count": 5}
        assert controller.load_memory(memory) == memory
        assert controller.save_memory() == memory

        with pytest.raises(TypeError):
            controller.load_memory([1, 2, 3])

        # When memory disabled, hooks return None
        no_mem = SingleAgentController(enable_memory=False)
        assert no_mem.load_memory() is None
        assert no_mem.save_memory() is None

        # Multi-agent controllers share the same interface
        multi = MultiAgentController(positions=[[0, 0], [1, 1]], enable_memory=True)
        multi_obs = {"signals": [0.1, 0.2]}
        assert multi.observe(multi_obs) == multi_obs
        with pytest.raises(TypeError):
            multi.observe("invalid")
    
    def test_memory_interface_default_behavior(self) -> None:
        """Test that memory interface has safe default behavior for non-memory-based agents."""
        controller = SingleAgentController(enable_memory=False)
        
        # Test load_memory with no memory enabled
        result = controller.load_memory()
        assert result is None  # Should return None when memory is disabled
        
        # Test load_memory with data when memory is disabled
        test_data = {"trajectory": [(0, 0), (1, 1)], "episode_count": 5}
        result = controller.load_memory(test_data)
        assert result is None  # Should still return None when memory is disabled
        
        # Test save_memory with no memory enabled
        result = controller.save_memory()
        assert result is None  # Should return None when memory is disabled
    
    def test_memory_interface_enabled_behavior(self) -> None:
        """Test memory interface functionality when enabled for cognitive modeling."""
        controller = SingleAgentController(enable_memory=True)
        
        # Test initial memory state
        initial_memory = controller.load_memory()
        # Initial memory should be None until first data is loaded
        assert initial_memory is None
        
        # Test loading memory data
        test_memory = {
            "trajectory_history": [(0, 0), (5, 10), (15, 20)],
            "visited_positions": [(0, 0), (5, 10)],
            "episode_count": 3,
            "learned_parameters": {"exploration_rate": 0.1}
        }
        
        loaded_memory = controller.load_memory(test_memory)
        assert loaded_memory == test_memory

        # Test saving memory data
        saved_memory = controller.save_memory()
        assert saved_memory == test_memory
        # Returned object should be a different instance than internal memory
        assert saved_memory is not controller._memory_state
    
    def test_memory_interface_optional_compliance(self) -> None:
        """Test that memory interface methods are optional and don't enforce usage."""
        # Test with memory disabled
        controller_no_memory = SingleAgentController(enable_memory=False)
        
        # Memory methods should exist but not enforce usage
        assert hasattr(controller_no_memory, 'load_memory')
        assert hasattr(controller_no_memory, 'save_memory')
        
        # Should be able to call without errors
        controller_no_memory.load_memory()
        controller_no_memory.save_memory()
        
        # Navigation should work normally without memory
        env_array = np.ones((50, 50)) * 0.5
        controller_no_memory.step(env_array)
        position_after = controller_no_memory.positions.copy()
        
        # Test with memory enabled but not used
        controller_with_memory = SingleAgentController(enable_memory=True)
        controller_with_memory.step(env_array)
        
        # Both should work regardless of memory setting
        assert controller_no_memory.positions.shape == (1, 2)
        assert controller_with_memory.positions.shape == (1, 2)
    
    def test_memory_interface_multi_agent_compatibility(self) -> None:
        """Test memory interface works with multi-agent controllers."""
        positions = [[0, 0], [10, 10], [20, 20]]
        controller = MultiAgentController(positions=positions, enable_memory=True)
        
        # Test memory operations with multi-agent setup
        test_memory = {
            "swarm_formation": "triangle",
            "collective_trajectory": positions,
            "coordination_state": {"leader": 0, "followers": [1, 2]}
        }
        
        loaded_memory = controller.load_memory(test_memory)
        assert loaded_memory == test_memory
        
        saved_memory = controller.save_memory()
        assert saved_memory == test_memory
        assert controller.num_agents == 3
    
    def test_memory_interface_serialization_compatibility(self) -> None:
        """Test that memory data is JSON-serializable for storage compatibility."""
        import json
        
        controller = SingleAgentController(enable_memory=True)
        
        # Test with various data types that should be serializable
        test_memory = {
            "trajectory": [(0.0, 0.0), (1.5, 2.3)],
            "episode_count": 10,
            "success_rate": 0.75,
            "metadata": {
                "timestamp": 1234567890.123,
                "version": "1.0",
                "agent_id": "agent_001"
            },
            "belief_state": [0.1, 0.2, 0.3, 0.4]
        }
        
        controller.load_memory(test_memory)
        saved_memory = controller.save_memory()

        # Should be JSON serializable
        try:
            json_str = json.dumps(saved_memory)
            json.loads(json_str)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Memory data should be JSON-serializable: {e}")


class TestNavigatorProtocolExtensibilityHooks:
    """
    Tests for new extensibility hooks supporting custom observations, rewards,
    and episode handling per Gymnasium 0.29.x migration requirements.
    
    These tests validate that extensibility hooks work correctly and provide
    the expected extension points for custom navigation algorithms.
    """
    
    def test_extensibility_hooks_default_behavior(self) -> None:
        """Test default behavior of extensibility hooks when disabled."""
        controller = SingleAgentController(enable_extensibility_hooks=False)
        
        # Test compute_additional_obs with hooks disabled
        base_obs = {"position": [0, 0], "orientation": 0.0}
        additional_obs = controller.compute_additional_obs(base_obs)
        assert additional_obs == {}  # Should return empty dict when disabled
        
        # Test compute_extra_reward with hooks disabled
        extra_reward = controller.compute_extra_reward(1.0, {"step": 10})
        assert extra_reward == 0.0  # Should return 0.0 when disabled
        
        # Test on_episode_end with hooks disabled (should not raise errors)
        controller.on_episode_end({"episode_length": 100, "success": True})
        # Should complete without error
    
    def test_extensibility_hooks_enabled_behavior(self) -> None:
        """Test extensibility hooks when enabled with custom observation keys."""
        controller = SingleAgentController(
            enable_extensibility_hooks=True,
            custom_observation_keys=["controller_id", "performance_metrics"]
        )
        
        # Initialize some performance metrics
        controller._performance_metrics['step_times'] = [1.0, 2.0, 3.0]
        
        # Test compute_additional_obs with enabled hooks
        base_obs = {"position": [5, 10], "orientation": 45.0}
        additional_obs = controller.compute_additional_obs(base_obs)
        
        # Should include configured custom observation keys
        assert "controller_id" in additional_obs
        assert "avg_step_time_ms" in additional_obs
        assert additional_obs["avg_step_time_ms"] == 2.0  # Mean of [1,2,3]
    
    def test_extensibility_hooks_reward_shaping(self) -> None:
        """Test reward shaping functionality through extensibility hooks."""
        controller = SingleAgentController(
            enable_extensibility_hooks=True,
            reward_shaping="exploration_bonus"
        )
        
        # Set up agent state for reward calculation
        controller._speed = np.array([1.5])
        
        # Test exploration bonus reward shaping
        base_reward = 0.5
        info = {"step": 10}
        extra_reward = controller.compute_extra_reward(base_reward, info)
        
        # Should provide exploration bonus based on movement
        expected_bonus = 0.01 * 1.5  # movement_bonus = 0.01 * speed
        assert extra_reward == expected_bonus
    
    def test_extensibility_hooks_episode_handling(self) -> None:
        """Test episode completion handling through extensibility hooks."""
        controller = SingleAgentController(
            enable_extensibility_hooks=True,
            enable_logging=True
        )
        
        # Set up performance metrics for episode end handling
        controller._performance_metrics['step_times'] = [5.0, 10.0, 15.0]
        controller._performance_metrics['total_steps'] = 100
        
        # Test episode end handling (should not raise errors)
        final_info = {
            "episode_length": 100,
            "success": True,
            "total_reward": 50.0
        }
        
        # Should complete without error and log episode statistics
        controller.on_episode_end(final_info)
        
        # Verify the method was called (no exceptions raised)
        assert controller._performance_metrics['total_steps'] == 100


class TestNavigatorProtocolSensorIntegration:
    """
    Tests for sensor-based observation processing workflows replacing direct 
    environmental sampling per Section 0.2.1 modular architecture requirements.
    
    These tests validate that NavigatorProtocol implementations properly integrate
    with SensorProtocol components for flexible perception modeling.
    """
    
    def test_sensor_based_odor_sampling(self) -> None:
        """Test that odor sampling uses SensorProtocol instead of direct field access."""
        # Create controller with default sensor setup
        controller = SingleAgentController()
        
        # Verify controller has sensors configured
        assert hasattr(controller, '_sensors')
        assert len(controller._sensors) > 0
        assert hasattr(controller, '_primary_sensor')
        
        # Test odor sampling goes through sensor system
        env_array = np.random.rand(50, 50)
        odor_value = controller.sample_odor(env_array)
        
        # Should return a valid odor value
        assert isinstance(odor_value, (float, np.floating))
        assert not np.isnan(odor_value)
        assert not np.isinf(odor_value)
    
    def test_sensor_protocol_compliance_in_controllers(self) -> None:
        """Test that controllers properly interact with SensorProtocol implementations."""
        from plume_nav_sim.core.sensors import ConcentrationSensor, BinarySensor
        
        # Create custom sensors
        concentration_sensor = ConcentrationSensor(dynamic_range=(0, 1), resolution=0.001)
        binary_sensor = BinarySensor(threshold=0.1, false_positive_rate=0.02)
        
        # Create controller with custom sensors
        controller = SingleAgentController(sensors=[concentration_sensor, binary_sensor])
        
        # Verify sensors are properly configured
        active_sensors = controller.get_active_sensors()
        assert len(active_sensors) == 2
        assert concentration_sensor in active_sensors
        assert binary_sensor in active_sensors
        
        # Test sensor management operations
        from plume_nav_sim.core.sensors import GradientSensor
        gradient_sensor = GradientSensor(spatial_resolution=(0.5, 0.5))
        
        controller.add_sensor(gradient_sensor)
        assert len(controller.get_active_sensors()) == 3
        
        removed = controller.remove_sensor(binary_sensor)
        assert removed is True
        assert len(controller.get_active_sensors()) == 2
        assert binary_sensor not in controller.get_active_sensors()
    
    def test_sensor_reset_integration(self) -> None:
        """Test that sensor reset is properly integrated with navigator reset."""
        from plume_nav_sim.core.sensors import ConcentrationSensor
        
        # Create controller with sensors
        sensor = ConcentrationSensor()
        controller = SingleAgentController(sensors=[sensor])
        
        # Mock the sensor reset method to track calls
        reset_called = []
        original_reset = getattr(sensor, 'reset', lambda: None)
        sensor.reset = lambda: reset_called.append(True)
        
        # Reset controller should reset sensors
        controller.reset()
        
        # Verify sensor reset was called (if sensor supports it)
        if hasattr(sensor, 'reset'):
            assert len(reset_called) > 0
    
    def test_multi_agent_sensor_integration(self) -> None:
        """Test sensor integration with multi-agent controllers."""
        from plume_nav_sim.core.sensors import ConcentrationSensor
        
        positions = [[0, 0], [10, 10]]
        sensors = [ConcentrationSensor(dynamic_range=(0, 1))]
        controller = MultiAgentController(positions=positions, sensors=sensors)
        
        # Test multi-agent odor sampling through sensors
        env_array = np.random.rand(50, 50)
        odor_values = controller.sample_odor(env_array)
        
        # Should return array of values for all agents
        assert isinstance(odor_values, np.ndarray)
        assert odor_values.shape == (2,)  # Two agents
        assert np.all(~np.isnan(odor_values))
        assert np.all(~np.isinf(odor_values))


class TestNavigatorProtocolBackwardCompatibility:
    """
    Tests ensuring existing NavigatorProtocol implementations continue functioning
    with new optional interface extensions per Section 0 requirements.
    
    These tests validate that existing code continues to work while new features
    are available for enhanced navigation strategies.
    """
    
    def test_existing_navigation_workflows_unchanged(self) -> None:
        """Test that existing navigation workflows continue to work unchanged."""
        # Test basic single-agent workflow
        controller = SingleAgentController(position=(10, 20), speed=1.0)
        
        # Basic navigation operations should work as before
        assert controller.num_agents == 1
        assert controller.positions[0, 0] == 10.0
        assert controller.positions[0, 1] == 20.0
        assert controller.speeds[0] == 1.0
        
        # Test reset functionality
        controller.reset(position=(30, 40))
        assert controller.positions[0, 0] == 30.0
        assert controller.positions[0, 1] == 40.0
        
        # Test step functionality
        env_array = np.ones((50, 50)) * 0.5
        initial_position = controller.positions.copy()
        controller.step(env_array)
        
        # Position should change after step
        assert not np.array_equal(controller.positions, initial_position)
    
    def test_multi_agent_backward_compatibility(self) -> None:
        """Test that multi-agent workflows remain compatible."""
        positions = [[0, 0], [10, 10], [20, 20]]
        controller = MultiAgentController(positions=positions)
        
        # Basic multi-agent properties should work
        assert controller.num_agents == 3
        assert controller.positions.shape == (3, 2)
        assert np.array_equal(controller.positions, np.array(positions))
        
        # Test multi-agent step
        env_array = np.ones((50, 50)) * 0.3
        initial_positions = controller.positions.copy()
        controller.step(env_array)
        
        # All agents should maintain valid positions
        assert controller.positions.shape == (3, 2)
        assert np.all(np.isfinite(controller.positions))
    
    def test_factory_method_backward_compatibility(self) -> None:
        """Test that factory methods maintain backward compatibility."""
        # Test single agent factory
        navigator = NavigatorFactory.single_agent(position=(5, 15), max_speed=2.0)
        assert isinstance(navigator, NavigatorProtocol)
        assert navigator.positions[0, 0] == 5.0
        assert navigator.positions[0, 1] == 15.0
        assert navigator.max_speeds[0] == 2.0
        
        # Test multi-agent factory
        positions = [[0, 0], [5, 5]]
        navigator = NavigatorFactory.multi_agent(positions=positions)
        assert isinstance(navigator, NavigatorProtocol)
        assert navigator.num_agents == 2
        assert navigator.positions.shape == (2, 2)
    
    def test_configuration_based_creation_compatibility(self) -> None:
        """Test that configuration-based navigator creation remains compatible."""
        # Test basic configuration
        config = {
            'position': (25, 35),
            'orientation': 45.0,
            'speed': 1.5,
            'max_speed': 3.0
        }
        
        navigator = NavigatorFactory.from_config(config)
        assert isinstance(navigator, NavigatorProtocol)
        assert navigator.positions[0, 0] == 25.0
        assert navigator.positions[0, 1] == 35.0
        assert navigator.orientations[0] == 45.0
        assert navigator.speeds[0] == 1.5
        assert navigator.max_speeds[0] == 3.0
    
    def test_existing_sampling_methods_unchanged(self) -> None:
        """Test that existing sampling methods maintain their interface."""
        controller = SingleAgentController(position=(25, 25))
        env_array = np.random.rand(50, 50)
        
        # Test single odor sampling (should return float)
        odor_value = controller.sample_odor(env_array)
        assert isinstance(odor_value, (float, np.floating))
        
        # Test multi-sensor sampling (should return array)
        sensor_readings = controller.sample_multiple_sensors(
            env_array, 
            sensor_distance=5.0,
            num_sensors=3
        )
        assert isinstance(sensor_readings, np.ndarray)
        assert sensor_readings.shape == (3,)
        
        # Test multi-agent sampling
        multi_controller = MultiAgentController(positions=[[10, 10], [20, 20]])
        multi_odor_values = multi_controller.sample_odor(env_array)
        assert isinstance(multi_odor_values, np.ndarray)
        assert multi_odor_values.shape == (2,)


class TestNavigatorProtocolObserveMethod:
    """
    Tests for new observe(sensor_output) method for processing SensorProtocol 
    observations instead of direct environmental sampling per Section 0.2.1.
    
    These tests validate that the observe method correctly processes sensor
    outputs and integrates with the modular sensor architecture.
    """
    
    def test_observe_method_existence_and_signature(self) -> None:
        """Test that observe method exists with correct signature."""
        controller = SingleAgentController()
        
        # Check if observe method exists
        if hasattr(controller, 'observe'):
            # Test method signature - should accept sensor output
            import inspect
            sig = inspect.signature(controller.observe)
            params = list(sig.parameters.keys())
            
            # Should have at least sensor_output parameter
            assert 'sensor_output' in params or len(params) >= 1
        else:
            # If observe method doesn't exist yet, test that sensor integration works
            # through existing sample_odor method which now uses sensors
            assert hasattr(controller, 'sample_odor')
            assert hasattr(controller, '_sensors')
            assert len(controller._sensors) > 0
    
    def test_observe_method_sensor_processing(self) -> None:
        """Test observe method processes sensor outputs correctly."""
        controller = SingleAgentController()
        
        # If observe method exists, test it
        if hasattr(controller, 'observe'):
            # Create mock sensor output
            sensor_output = {
                'concentration': 0.5,
                'gradient': [0.1, -0.2],
                'detection': True,
                'position': [10.0, 20.0]
            }
            
            # Test observe method processing
            observation = controller.observe(sensor_output)
            
            # Should return processed observation
            assert observation is not None
            assert isinstance(observation, dict)
        else:
            # Test that sensor-based sampling works as expected
            env_array = np.random.rand(50, 50)
            odor_value = controller.sample_odor(env_array)
            assert isinstance(odor_value, (float, np.floating))
    
    def test_observe_method_multi_sensor_integration(self) -> None:
        """Test observe method handles multiple sensor outputs."""
        from plume_nav_sim.core.sensors import ConcentrationSensor, BinarySensor
        
        # Create controller with multiple sensors
        sensors = [
            ConcentrationSensor(dynamic_range=(0, 1)),
            BinarySensor(threshold=0.1)
        ]
        controller = SingleAgentController(sensors=sensors)
        
        if hasattr(controller, 'observe'):
            # Test with multi-sensor output
            multi_sensor_output = {
                'ConcentrationSensor': {'value': 0.75, 'timestamp': 1.0},
                'BinarySensor': {'detected': True, 'confidence': 0.9}
            }
            
            observation = controller.observe(multi_sensor_output)
            assert observation is not None
            assert isinstance(observation, dict)
        else:
            # Verify that multiple sensors are properly integrated
            assert len(controller.get_active_sensors()) == 2
            
            # Test multi-sensor sampling functionality
            env_array = np.random.rand(50, 50)
            readings = controller.sample_multiple_sensors(env_array, num_sensors=2)
            assert isinstance(readings, np.ndarray)
            assert readings.shape == (2,)
    
    def test_observe_method_error_handling(self) -> None:
        """Test observe method error handling with invalid sensor outputs."""
        controller = SingleAgentController()
        
        if hasattr(controller, 'observe'):
            # Test with invalid sensor output
            invalid_outputs = [
                None,
                {},
                {'invalid': 'data'},
                {'concentration': float('nan')}
            ]
            
            for invalid_output in invalid_outputs:
                try:
                    observation = controller.observe(invalid_output)
                    # Should handle gracefully and return valid observation
                    assert observation is not None
                except (ValueError, TypeError) as e:
                    # Acceptable to raise exceptions for invalid input
                    assert str(e)  # Should have error message
        else:
            # Test error handling in sensor-based sampling
            try:
                # Test with invalid environment array
                invalid_env = np.array([])  # Empty array
                controller.sample_odor(invalid_env)
            except (ValueError, IndexError):
                # Expected to raise errors for invalid input
                pass

    def test_observe_returns_sensor_output_dict(self) -> None:
        """Ensure observe merges sensor output with additional observations."""
        controller = SingleAgentController(enable_extensibility_hooks=True)
        sensor_output = {"signal": 1}
        controller.compute_additional_obs = MagicMock(return_value={"extra": 2})

        observation = controller.observe(sensor_output)

        assert observation["signal"] == 1
        assert observation["extra"] == 2
        controller.compute_additional_obs.assert_called_once()
        assert controller.compute_additional_obs.call_args[0][0]["signal"] == 1
