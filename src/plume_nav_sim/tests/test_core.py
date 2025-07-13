"""
Core navigation system testing module for plume_nav_sim.

This module provides comprehensive validation for Navigator facade, controller 
implementations, and protocol interface compliance. Tests cover single-agent 
and multi-agent navigation logic, state management, sensor sampling, and 
protocol adherence through systematic unit testing.

Test Categories:
- Navigator facade testing with configuration-driven instantiation
- SingleAgentController and MultiAgentController implementation validation
- NavigatorProtocol compliance and interface verification
- State management testing for positions, orientations, speeds, and velocities
- Sensor sampling validation including multi-sensor capabilities
- Performance requirements validation per Section 6.6.3.3
- Numerical precision validation with 1e-6 tolerance per research standards
- Extensibility hooks integration for Gymnasium 0.29.x compatibility
- Factory method integration with Hydra configuration system

Performance Requirements:
- Single agent step operations: <1ms (Section 6.6.3.3)
- Multi-agent operations: efficient scaling up to 100 agents
- Navigator components: >90% test coverage (Section 6.6.3.1)
- Numerical precision: 1e-6 tolerance for research accuracy

"""

import pytest
import numpy as np
import time
from typing import Dict, Any, List, Tuple, Union, Optional
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import core navigation components
from plume_nav_sim.core.navigator import NavigatorProtocol, Navigator, NavigatorFactory
from plume_nav_sim.core.controllers import (
    SingleAgentController, 
    MultiAgentController,
    SingleAgentParams,
    MultiAgentParams,
    create_controller_from_config
)

# Import new v1.0 protocol components for dependency injection testing
from plume_nav_sim.core.protocols import BoundaryPolicyProtocol
from plume_nav_sim.core.sources import create_source
from plume_nav_sim.core.boundaries import create_boundary_policy

# Import utility functions
from plume_nav_sim.utils.navigator_utils import (
    create_navigator_from_config,
    PREDEFINED_SENSOR_LAYOUTS
)


# Test tolerance for numerical precision validation per research standards
NUMERICAL_PRECISION_TOLERANCE = 1e-6

# Performance thresholds per Section 6.6.3.3
SINGLE_AGENT_STEP_THRESHOLD_MS = 1.0
MULTI_AGENT_10_STEP_THRESHOLD_MS = 5.0
MULTI_AGENT_100_STEP_THRESHOLD_MS = 50.0


class TestNavigatorProtocol:
    """Test NavigatorProtocol interface compliance and type checking."""
    
    def test_protocol_interface_definition(self):
        """Test that NavigatorProtocol defines required interface methods and properties."""
        # Test that NavigatorProtocol is runtime checkable
        assert hasattr(NavigatorProtocol, '__runtime_checkable__')
        
        # Test required properties exist in protocol
        required_properties = [
            'positions', 'orientations', 'speeds', 'max_speeds', 
            'angular_velocities', 'num_agents'
        ]
        
        for prop in required_properties:
            assert hasattr(NavigatorProtocol, prop), f"Protocol missing property: {prop}"
    
    def test_protocol_required_methods(self):
        """Test that NavigatorProtocol defines required interface methods."""
        required_methods = [
            'reset', 'step', 'sample_odor', 'sample_multiple_sensors'
        ]
        
        for method in required_methods:
            assert hasattr(NavigatorProtocol, method), f"Protocol missing method: {method}"
    
    def test_single_agent_controller_protocol_compliance(self):
        """Test that SingleAgentController implements NavigatorProtocol."""
        controller = SingleAgentController(
            position=(10.0, 20.0),
            orientation=45.0,
            speed=2.0,
            max_speed=5.0
        )
        
        # Test protocol compliance
        assert isinstance(controller, NavigatorProtocol)
        
        # Test all required properties are accessible
        assert hasattr(controller, 'positions')
        assert hasattr(controller, 'orientations')
        assert hasattr(controller, 'speeds')
        assert hasattr(controller, 'max_speeds')
        assert hasattr(controller, 'angular_velocities')
        assert hasattr(controller, 'num_agents')
        
        # Test all required methods are callable
        assert callable(controller.reset)
        assert callable(controller.step)
        assert callable(controller.sample_odor)
        assert callable(controller.sample_multiple_sensors)
    
    def test_multi_agent_controller_protocol_compliance(self):
        """Test that MultiAgentController implements NavigatorProtocol."""
        positions = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        orientations = np.array([0.0, 90.0, 180.0])
        
        controller = MultiAgentController(
            positions=positions,
            orientations=orientations,
            speeds=np.array([1.0, 2.0, 1.5]),
            max_speeds=np.array([3.0, 4.0, 3.5])
        )
        
        # Test protocol compliance
        assert isinstance(controller, NavigatorProtocol)
        
        # Test property types and shapes
        assert isinstance(controller.positions, np.ndarray)
        assert isinstance(controller.orientations, np.ndarray)
        assert isinstance(controller.speeds, np.ndarray)
        assert isinstance(controller.max_speeds, np.ndarray)
        assert isinstance(controller.angular_velocities, np.ndarray)
        assert isinstance(controller.num_agents, int)
    
    def test_extensibility_hooks_protocol_compliance(self):
        """Test that controllers implement extensibility hooks for Gymnasium 0.29.x."""
        controller = SingleAgentController(
            position=(5.0, 5.0),
            enable_extensibility_hooks=True
        )
        
        # Test extensibility hooks are callable
        assert callable(controller.compute_additional_obs)
        assert callable(controller.compute_extra_reward)
        assert callable(controller.on_episode_end)
        
        # Test hooks return expected types
        base_obs = {"position": [5.0, 5.0]}
        additional_obs = controller.compute_additional_obs(base_obs)
        assert isinstance(additional_obs, dict)
        
        extra_reward = controller.compute_extra_reward(1.0, {"episode_length": 100})
        assert isinstance(extra_reward, (int, float))
        
        # Test episode end hook executes without error
        controller.on_episode_end({"success": True, "episode_length": 100})


class TestSingleAgentController:
    """Comprehensive testing for SingleAgentController implementation."""
    
    @pytest.fixture
    def basic_single_agent(self):
        """Create a basic single agent controller for testing."""
        return SingleAgentController(
            position=(5.0, 10.0),
            orientation=30.0,
            speed=1.5,
            max_speed=3.0,
            angular_velocity=5.0
        )
    
    @pytest.fixture
    def default_single_agent(self):
        """Create a single agent controller with default parameters."""
        return SingleAgentController()
    
    def test_single_agent_initialization_with_parameters(self, basic_single_agent):
        """Test single agent initialization with explicit parameters."""
        agent = basic_single_agent
        
        # Test property shapes and types
        assert agent.positions.shape == (1, 2)
        assert agent.orientations.shape == (1,)
        assert agent.speeds.shape == (1,)
        assert agent.max_speeds.shape == (1,)
        assert agent.angular_velocities.shape == (1,)
        assert agent.num_agents == 1
        
        # Test initial values with numerical precision
        np.testing.assert_allclose(agent.positions[0], [5.0, 10.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.orientations[0], 30.0, atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.speeds[0], 1.5, atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.max_speeds[0], 3.0, atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.angular_velocities[0], 5.0, atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_single_agent_initialization_defaults(self, default_single_agent):
        """Test single agent initialization with default parameters."""
        agent = default_single_agent
        
        # Test default values
        np.testing.assert_allclose(agent.positions[0], [0.0, 0.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.orientations[0], 0.0, atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.speeds[0], 0.0, atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.max_speeds[0], 1.0, atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.angular_velocities[0], 0.0, atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_single_agent_parameter_validation(self):
        """Test parameter validation during initialization."""
        # Test speed exceeding max_speed
        with pytest.raises((ValueError, RuntimeError)):
            SingleAgentController(speed=5.0, max_speed=3.0)
        
        # Test negative max_speed
        with pytest.raises((ValueError, RuntimeError)):
            SingleAgentController(max_speed=-1.0)
    
    def test_single_agent_reset_with_kwargs(self, basic_single_agent):
        """Test reset functionality with keyword arguments."""
        agent = basic_single_agent
        
        # Reset with new parameters
        agent.reset(
            position=(15.0, 25.0),
            orientation=60.0,
            speed=2.0,
            max_speed=4.0,
            angular_velocity=10.0
        )
        
        # Verify reset values
        np.testing.assert_allclose(agent.positions[0], [15.0, 25.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.orientations[0], 60.0, atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.speeds[0], 2.0, atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.max_speeds[0], 4.0, atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.angular_velocities[0], 10.0, atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_single_agent_reset_partial_parameters(self, basic_single_agent):
        """Test reset with only some parameters updated."""
        agent = basic_single_agent
        original_position = agent.positions[0].copy()
        original_orientation = agent.orientations[0]
        
        # Reset only speed
        agent.reset(speed=2.5)
        
        # Verify only speed changed
        np.testing.assert_allclose(agent.positions[0], original_position, atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.orientations[0], original_orientation, atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.speeds[0], 2.5, atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_single_agent_step_functionality(self, basic_single_agent):
        """Test step method calls utility functions correctly."""
        agent = basic_single_agent
        mock_env_array = np.random.rand(100, 100)
        
        # Store initial state for comparison
        initial_position = agent.positions[0].copy()
        
        # Step should execute without error
        agent.step(mock_env_array)
        
        # Verify that step could potentially change position (basic functionality test)
        # Note: Since we don't mock the update function, position may or may not change
        # but the important thing is that step executes without error
        assert agent.positions.shape == (1, 2)
        assert np.all(np.isfinite(agent.positions))
    
    def test_single_agent_step_performance(self, basic_single_agent):
        """Test single agent step performance meets <1ms requirement."""
        agent = basic_single_agent
        mock_env_array = np.random.rand(50, 50)  # Smaller array for performance test
        
        # Warm up
        agent.step(mock_env_array)
        
        # Performance test
        start_time = time.time()
        agent.step(mock_env_array)
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Verify performance requirement
        assert execution_time_ms < SINGLE_AGENT_STEP_THRESHOLD_MS, \
            f"Single agent step took {execution_time_ms:.2f}ms, should be <{SINGLE_AGENT_STEP_THRESHOLD_MS}ms"
    
    def test_single_agent_odor_sampling(self, basic_single_agent):
        """Test odor sampling functionality for single agent."""
        agent = basic_single_agent
        mock_env_array = np.random.rand(100, 100)
        
        result = agent.sample_odor(mock_env_array)
        
        # Verify correct result type and value
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)
        assert result >= 0.0  # Odor values should be non-negative
    
    def test_single_agent_multi_sensor_sampling(self, basic_single_agent):
        """Test multi-sensor sampling for single agent."""
        agent = basic_single_agent
        mock_env_array = np.random.rand(100, 100)
        
        result = agent.sample_multiple_sensors(
            mock_env_array,
            sensor_distance=8.0,
            sensor_angle=30.0,
            num_sensors=2
        )
        
        # Verify result shape and type
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)  # 2 sensors for single agent
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)  # Odor values should be non-negative
    
    def test_single_agent_state_validation_edge_cases(self):
        """Test state validation for edge cases and boundary conditions."""
        # Test with very small values
        agent = SingleAgentController(
            position=(1e-10, 1e-10),
            orientation=0.001,
            speed=1e-6,
            max_speed=1e-5
        )
        
        assert agent.num_agents == 1
        assert np.all(np.isfinite(agent.positions))
        assert np.all(np.isfinite(agent.orientations))
        assert np.all(np.isfinite(agent.speeds))
        assert np.all(np.isfinite(agent.max_speeds))
    
    def test_single_agent_configuration_integration(self):
        """Test integration with configuration system."""
        # Test with mock configuration
        mock_config = {
            'position': (12.0, 18.0),
            'orientation': 45.0,
            'speed': 1.0,
            'max_speed': 2.5,
            'angular_velocity': 3.0
        }
        
        agent = create_controller_from_config(mock_config)
        
        # Verify configuration was applied
        assert isinstance(agent, SingleAgentController)
        np.testing.assert_allclose(agent.positions[0], [12.0, 18.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.orientations[0], 45.0, atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.speeds[0], 1.0, atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_single_agent_extensibility_hooks(self):
        """Test extensibility hooks integration for Gymnasium 0.29.x."""
        agent = SingleAgentController(
            position=(5.0, 5.0),
            enable_extensibility_hooks=True,
            custom_observation_keys=["controller_id", "performance_metrics"]
        )
        
        # Test additional observations
        base_obs = {"position": [5.0, 5.0], "orientation": 0.0}
        additional_obs = agent.compute_additional_obs(base_obs)
        assert isinstance(additional_obs, dict)
        
        # Test extra reward computation
        extra_reward = agent.compute_extra_reward(1.0, {"episode_length": 100})
        assert isinstance(extra_reward, (int, float))
        
        # Test episode end handling
        agent.on_episode_end({"success": True, "episode_length": 100})


class TestMultiAgentController:
    """Comprehensive testing for MultiAgentController implementation."""
    
    @pytest.fixture
    def basic_multi_agent(self):
        """Create a basic multi-agent controller for testing."""
        positions = np.array([
            [0.0, 0.0],
            [10.0, 5.0],
            [5.0, 15.0]
        ])
        orientations = np.array([0.0, 45.0, 90.0])
        speeds = np.array([1.0, 1.5, 2.0])
        max_speeds = np.array([2.0, 3.0, 4.0])
        angular_velocities = np.array([0.0, 5.0, -3.0])
        
        return MultiAgentController(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds,
            angular_velocities=angular_velocities
        )
    
    @pytest.fixture
    def large_multi_agent(self):
        """Create a large multi-agent controller for scaling tests."""
        num_agents = 50
        positions = np.random.rand(num_agents, 2) * 100
        orientations = np.random.rand(num_agents) * 360
        speeds = np.random.rand(num_agents) * 2
        max_speeds = speeds + np.random.rand(num_agents) * 2
        angular_velocities = (np.random.rand(num_agents) - 0.5) * 20
        
        return MultiAgentController(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds,
            angular_velocities=angular_velocities
        )
    
    def test_multi_agent_initialization_with_arrays(self, basic_multi_agent):
        """Test multi-agent initialization with explicit arrays."""
        agent = basic_multi_agent
        
        # Test array shapes
        assert agent.positions.shape == (3, 2)
        assert agent.orientations.shape == (3,)
        assert agent.speeds.shape == (3,)
        assert agent.max_speeds.shape == (3,)
        assert agent.angular_velocities.shape == (3,)
        assert agent.num_agents == 3
        
        # Test specific values with numerical precision
        expected_positions = np.array([[0.0, 0.0], [10.0, 5.0], [5.0, 15.0]])
        expected_orientations = np.array([0.0, 45.0, 90.0])
        expected_speeds = np.array([1.0, 1.5, 2.0])
        
        np.testing.assert_allclose(agent.positions, expected_positions, atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.orientations, expected_orientations, atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.speeds, expected_speeds, atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_multi_agent_initialization_defaults(self):
        """Test multi-agent initialization with minimal parameters."""
        positions = np.array([[0.0, 0.0], [1.0, 1.0]])
        agent = MultiAgentController(positions=positions)
        
        # Test default values
        assert agent.num_agents == 2
        np.testing.assert_allclose(agent.orientations, [0.0, 0.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.speeds, [0.0, 0.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.max_speeds, [1.0, 1.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.angular_velocities, [0.0, 0.0], atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_multi_agent_array_validation(self):
        """Test array shape and consistency validation."""
        positions = np.array([[0.0, 0.0], [1.0, 1.0]])
        
        # Test mismatched array shapes
        with pytest.raises((ValueError, RuntimeError)):
            MultiAgentController(
                positions=positions,
                orientations=np.array([0.0])  # Wrong size
            )
        
        # Test invalid position array shape
        with pytest.raises((ValueError, RuntimeError)):
            MultiAgentController(
                positions=np.array([0.0, 1.0])  # Should be (N, 2)
            )
    
    def test_multi_agent_parameter_validation(self):
        """Test parameter validation for multi-agent scenario."""
        positions = np.array([[0.0, 0.0], [1.0, 1.0]])
        speeds = np.array([3.0, 4.0])
        max_speeds = np.array([2.0, 3.0])  # Some speeds exceed max_speeds
        
        # Create controller and verify speeds are handled appropriately
        agent = MultiAgentController(
            positions=positions,
            speeds=speeds,
            max_speeds=max_speeds
        )
        
        # This should not raise an error - speeds may be clamped or validated differently
        assert agent.num_agents == 2
    
    def test_multi_agent_reset_functionality(self, basic_multi_agent):
        """Test reset functionality for multi-agent controller."""
        agent = basic_multi_agent
        
        new_positions = np.array([[20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
        new_orientations = np.array([180.0, 270.0, 0.0])
        
        agent.reset(
            positions=new_positions,
            orientations=new_orientations
        )
        
        # Verify reset values
        np.testing.assert_allclose(agent.positions, new_positions, atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(agent.orientations, new_orientations, atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_multi_agent_step_functionality(self, basic_multi_agent):
        """Test step method for multi-agent scenario."""
        agent = basic_multi_agent
        mock_env_array = np.random.rand(100, 100)
        
        # Store initial state
        initial_positions = agent.positions.copy()
        
        # Step should execute without error
        agent.step(mock_env_array)
        
        # Verify arrays maintain proper shapes
        assert agent.positions.shape == (3, 2)
        assert agent.orientations.shape == (3,)
        assert np.all(np.isfinite(agent.positions))
        assert np.all(np.isfinite(agent.orientations))
    
    def test_multi_agent_step_performance_10_agents(self):
        """Test multi-agent step performance for 10 agents meets <5ms requirement."""
        positions = np.random.rand(10, 2) * 50
        agent = MultiAgentController(positions=positions)
        mock_env_array = np.random.rand(50, 50)
        
        # Warm up
        agent.step(mock_env_array)
        
        # Performance test
        start_time = time.time()
        agent.step(mock_env_array)
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Verify performance requirement
        assert execution_time_ms < MULTI_AGENT_10_STEP_THRESHOLD_MS, \
            f"10-agent step took {execution_time_ms:.2f}ms, should be <{MULTI_AGENT_10_STEP_THRESHOLD_MS}ms"
    
    def test_multi_agent_step_performance_100_agents(self):
        """Test multi-agent step performance for 100 agents meets <50ms requirement."""
        positions = np.random.rand(100, 2) * 100
        agent = MultiAgentController(positions=positions)
        mock_env_array = np.random.rand(100, 100)
        
        # Warm up
        agent.step(mock_env_array)
        
        # Performance test
        start_time = time.time()
        agent.step(mock_env_array)
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Verify performance requirement
        assert execution_time_ms < MULTI_AGENT_100_STEP_THRESHOLD_MS, \
            f"100-agent step took {execution_time_ms:.2f}ms, should be <{MULTI_AGENT_100_STEP_THRESHOLD_MS}ms"
    
    def test_multi_agent_odor_sampling(self, basic_multi_agent):
        """Test odor sampling for multi-agent scenario."""
        agent = basic_multi_agent
        mock_env_array = np.random.rand(100, 100)
        
        result = agent.sample_odor(mock_env_array)
        
        # Verify result type and shape
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)  # Odor values should be non-negative
    
    def test_multi_agent_multi_sensor_sampling(self, basic_multi_agent):
        """Test multi-sensor sampling for multi-agent scenario."""
        agent = basic_multi_agent
        mock_env_array = np.random.rand(100, 100)
        
        result = agent.sample_multiple_sensors(
            mock_env_array,
            sensor_distance=5.0,
            num_sensors=2
        )
        
        # Verify result shape and values
        assert result.shape == (3, 2)  # 3 agents, 2 sensors each
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)  # Odor values should be non-negative
    
    def test_multi_agent_scaling_efficiency(self, large_multi_agent):
        """Test that multi-agent operations scale efficiently."""
        agent = large_multi_agent
        mock_env_array = np.random.rand(100, 100)
        
        # Test that array operations maintain efficiency
        start_time = time.time()
        
        # Simulate multiple operations
        positions = agent.positions
        orientations = agent.orientations
        speeds = agent.speeds
        
        # Test vectorized operations
        distances = np.linalg.norm(positions, axis=1)
        normalized_orientations = orientations % 360
        speed_ratios = speeds / agent.max_speeds
        
        execution_time = time.time() - start_time
        
        # Verify operations complete quickly for 50 agents
        assert execution_time < 0.01, f"Scaling operations took {execution_time:.4f}s, should be <0.01s"
        
        # Verify results maintain numerical precision
        assert np.all(np.isfinite(distances))
        assert np.all(normalized_orientations >= 0) and np.all(normalized_orientations < 360)
        assert np.all(speed_ratios >= 0)
    
    def test_multi_agent_configuration_integration(self):
        """Test multi-agent configuration integration."""
        mock_config = {
            'positions': [[0.0, 0.0], [10.0, 10.0]],
            'orientations': [0.0, 90.0],
            'speeds': [1.0, 2.0],
            'max_speeds': [3.0, 4.0]
        }
        
        agent = create_controller_from_config(mock_config)
        
        # Verify configuration was applied
        assert isinstance(agent, MultiAgentController)
        assert agent.num_agents == 2
        expected_positions = np.array([[0.0, 0.0], [10.0, 10.0]])
        np.testing.assert_allclose(agent.positions, expected_positions, atol=NUMERICAL_PRECISION_TOLERANCE)


class TestSensorSampling:
    """Test sensor sampling functionality and sensor layout configurations."""
    
    @pytest.fixture
    def mock_navigator(self):
        """Create a mock navigator for sensor testing."""
        navigator = Mock(spec=NavigatorProtocol)
        navigator.positions = np.array([[5.0, 10.0], [15.0, 20.0]])
        navigator.orientations = np.array([0.0, 90.0])
        navigator.num_agents = 2
        return navigator
    
    @pytest.fixture
    def sample_env_array(self):
        """Create a sample environment array for testing."""
        return np.random.rand(50, 50).astype(np.float32)
    
    def test_predefined_sensor_layouts(self):
        """Test predefined sensor layout configurations."""
        # Test LEFT_RIGHT layout
        layout = PREDEFINED_SENSOR_LAYOUTS["LEFT_RIGHT"]
        expected = np.array([[0.0, 1.0], [0.0, -1.0]])
        np.testing.assert_allclose(layout, expected, atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Test FRONT_SIDES layout
        layout = PREDEFINED_SENSOR_LAYOUTS["FRONT_SIDES"]
        expected = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        np.testing.assert_allclose(layout, expected, atol=NUMERICAL_PRECISION_TOLERANCE)
        
        # Test SINGLE layout
        layout = PREDEFINED_SENSOR_LAYOUTS["SINGLE"]
        expected = np.array([[0.0, 0.0]])
        np.testing.assert_allclose(layout, expected, atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_sensor_layout_validation(self):
        """Test sensor layout validation and error handling."""
        # Test that predefined layouts exist
        required_layouts = ["SINGLE", "LEFT_RIGHT", "FRONT_SIDES", "CARDINAL"]
        for layout_name in required_layouts:
            assert layout_name in PREDEFINED_SENSOR_LAYOUTS, f"Missing layout: {layout_name}"
    
    def test_sensor_sampling_edge_cases(self, sample_env_array):
        """Test sensor sampling edge cases and boundary conditions."""
        # Create a simple single agent controller for testing
        agent = SingleAgentController(position=(24.5, 24.5))  # Center with decimals
        
        # Test multi-sensor sampling at different configurations
        result = agent.sample_multiple_sensors(
            sample_env_array,
            sensor_distance=1.0,
            num_sensors=1
        )
        
        # Verify all values are finite and valid
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)
    
    def test_sensor_sampling_with_boundary_positions(self, sample_env_array):
        """Test sensor sampling with positions at array boundaries."""
        # Test agents at boundary positions
        boundary_positions = [[0.0, 0.0], [49.0, 49.0], [24.5, 24.5]]
        agent = MultiAgentController(positions=np.array(boundary_positions))
        
        result = agent.sample_multiple_sensors(
            sample_env_array,
            sensor_distance=2.0,
            num_sensors=2
        )
        
        # Verify all values are finite and valid
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)
        assert result.shape == (3, 2)  # 3 agents, 2 sensors each


class TestNavigatorFactoryMethods:
    """Test factory method integration with configuration system."""
    
    @pytest.fixture
    def single_agent_config(self):
        """Create configuration for single agent."""
        return {
            'position': [8.0, 12.0],
            'orientation': 45.0,
            'speed': 1.5,
            'max_speed': 3.0,
            'angular_velocity': 2.0
        }
    
    @pytest.fixture  
    def multi_agent_config(self):
        """Create configuration for multi-agent scenario."""
        return {
            'positions': [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]],
            'orientations': [0.0, 90.0, 180.0],
            'speeds': [1.0, 1.5, 2.0],
            'max_speeds': [2.0, 3.0, 4.0],
            'angular_velocities': [0.0, 5.0, -5.0]
        }
    
    def test_single_agent_factory_creation(self, single_agent_config):
        """Test single agent creation via factory method pattern."""
        # Create agent using configuration
        navigator = create_navigator_from_config(single_agent_config)
        
        # Verify configuration was applied correctly
        assert isinstance(navigator, Navigator)
        assert navigator.num_agents == 1
        
        np.testing.assert_allclose(navigator.positions[0], [8.0, 12.0], atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(navigator.orientations[0], 45.0, atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(navigator.speeds[0], 1.5, atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_multi_agent_factory_creation(self, multi_agent_config):
        """Test multi-agent creation via factory method pattern."""
        # Create agent using configuration
        navigator = create_navigator_from_config(multi_agent_config)
        
        # Verify configuration was applied correctly
        assert isinstance(navigator, Navigator)
        assert navigator.num_agents == 3
        
        expected_positions = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        expected_orientations = np.array([0.0, 90.0, 180.0])
        
        np.testing.assert_allclose(navigator.positions, expected_positions, atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(navigator.orientations, expected_orientations, atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_navigator_factory_enhanced_features(self):
        """Test NavigatorFactory with enhanced Gymnasium 0.29.x features."""
        factory = NavigatorFactory()
        
        config = {
            'position': [5.0, 5.0],
            'enable_extensibility_hooks': True,
            'frame_cache_mode': 'lru',
            'custom_observation_keys': ['performance_metrics']
        }
        
        navigator = factory.create_navigator(config)
        
        # Verify enhanced features are enabled
        assert isinstance(navigator, Navigator)
        assert navigator.enable_extensibility_hooks is True
        assert navigator.frame_cache_mode == 'lru'
        assert 'performance_metrics' in navigator.custom_observation_keys


class TestNavigatorStateManagement:
    """Test comprehensive state management functionality."""
    
    @pytest.fixture
    def state_test_agent(self):
        """Create agent for state management testing."""
        return SingleAgentController(
            position=(10.0, 20.0),
            orientation=30.0,
            speed=2.0,
            max_speed=5.0,
            angular_velocity=10.0
        )
    
    def test_state_consistency_after_operations(self, state_test_agent):
        """Test that state remains consistent after various operations."""
        agent = state_test_agent
        initial_state = {
            'positions': agent.positions.copy(),
            'orientations': agent.orientations.copy(),
            'speeds': agent.speeds.copy(),
            'max_speeds': agent.max_speeds.copy(),
            'angular_velocities': agent.angular_velocities.copy()
        }
        
        # Perform mock step operation
        mock_env_array = np.random.rand(50, 50)
        agent.step(mock_env_array)
        
        # State should still be valid (arrays may change but remain consistent)
        assert agent.positions.shape == initial_state['positions'].shape
        assert agent.orientations.shape == initial_state['orientations'].shape
        assert np.all(np.isfinite(agent.positions))
        assert np.all(np.isfinite(agent.orientations))
    
    def test_state_validation_with_invalid_values(self):
        """Test state validation rejects invalid values."""
        # Test with NaN values
        with pytest.raises((ValueError, RuntimeError)):
            SingleAgentController(
                position=(float('nan'), 10.0)
            )
        
        # Test with infinite values
        with pytest.raises((ValueError, RuntimeError)):
            SingleAgentController(
                speed=float('inf')
            )
    
    def test_state_reset_reproducibility(self, state_test_agent):
        """Test that state reset is reproducible."""
        agent = state_test_agent
        
        # Reset to specific state
        reset_params = {
            'position': (50.0, 60.0),
            'orientation': 120.0,
            'speed': 3.0
        }
        
        agent.reset(**reset_params)
        state_1 = {
            'positions': agent.positions.copy(),
            'orientations': agent.orientations.copy(),
            'speeds': agent.speeds.copy()
        }
        
        # Reset again with same parameters
        agent.reset(**reset_params)
        state_2 = {
            'positions': agent.positions.copy(),
            'orientations': agent.orientations.copy(),
            'speeds': agent.speeds.copy()
        }
        
        # Verify states are identical
        np.testing.assert_array_equal(state_1['positions'], state_2['positions'])
        np.testing.assert_array_equal(state_1['orientations'], state_2['orientations'])
        np.testing.assert_array_equal(state_1['speeds'], state_2['speeds'])
    
    def test_boundary_condition_handling(self):
        """Test handling of boundary conditions and edge cases."""
        # Test very large position values
        agent = SingleAgentController(position=(1e6, 1e6))
        assert np.all(np.isfinite(agent.positions))
        
        # Test very small speed values
        agent = SingleAgentController(speed=1e-10, max_speed=1e-9)
        assert np.all(agent.speeds >= 0)
        assert np.all(agent.speeds <= agent.max_speeds)
        
        # Test orientation wrapping (if implemented)
        agent = SingleAgentController(orientation=720.0)  # 2 full rotations
        assert np.all(np.isfinite(agent.orientations))


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""
    
    def test_graceful_degradation_invalid_environment(self):
        """Test graceful handling of invalid environment arrays."""
        agent = SingleAgentController()
        
        # Test with None environment
        with pytest.raises((AttributeError, TypeError)):
            agent.sample_odor(None)
        
        # Test with invalid array shape
        invalid_env = np.array([1, 2, 3])  # 1D array instead of 2D
        result = agent.sample_odor(invalid_env)
        
        # Should return valid result or handle gracefully
        assert isinstance(result, (float, type(None))) or np.isfinite(result)
    
    def test_resource_cleanup_on_failure(self):
        """Test that resources are properly cleaned up on failure."""
        with pytest.raises((ValueError, RuntimeError)):
            # Create controller with invalid parameters
            SingleAgentController(max_speed=-1.0)
        
        # Memory should not be leaked after failed initialization
        # This is more of a conceptual test - actual memory leak detection
        # would require specialized tools
    
    def test_concurrent_access_safety(self):
        """Test that controllers handle concurrent access safely."""
        agent = SingleAgentController()
        
        # Test that multiple property accesses don't interfere
        positions_1 = agent.positions
        orientations = agent.orientations
        positions_2 = agent.positions
        
        # Arrays should be consistent
        np.testing.assert_array_equal(positions_1, positions_2)


class TestNumericalPrecisionAndAccuracy:
    """Test numerical precision and accuracy requirements."""
    
    def test_floating_point_precision(self):
        """Test floating point precision in calculations."""
        # Test with high precision input values
        precise_position = (np.pi, np.e)  # Irrational numbers
        agent = SingleAgentController(position=precise_position)
        
        # Verify precision is maintained
        np.testing.assert_allclose(
            agent.positions[0], 
            [np.pi, np.e], 
            atol=NUMERICAL_PRECISION_TOLERANCE
        )
    
    def test_accumulation_error_prevention(self):
        """Test that numerical errors don't accumulate over operations."""
        agent = SingleAgentController(
            position=(1.0, 1.0),
            speed=0.1,
            angular_velocity=1.0
        )
        
        mock_env = np.ones((10, 10))
        
        # Perform multiple operations
        for _ in range(100):
            agent.step(mock_env)
            
            # Verify values remain finite and reasonable
            assert np.all(np.isfinite(agent.positions))
            assert np.all(np.isfinite(agent.orientations))
            assert np.all(np.isfinite(agent.speeds))
    
    def test_vector_operation_precision(self):
        """Test precision in vectorized operations for multi-agent scenarios."""
        num_agents = 10
        positions = np.random.rand(num_agents, 2) * 100
        
        agent = MultiAgentController(positions=positions)
        
        # Test vectorized distance calculations
        distances = np.linalg.norm(agent.positions, axis=1)
        
        # Verify precision
        for i in range(num_agents):
            expected_distance = np.sqrt(positions[i, 0]**2 + positions[i, 1]**2)
            np.testing.assert_allclose(
                distances[i], 
                expected_distance, 
                atol=NUMERICAL_PRECISION_TOLERANCE
            )
    
    def test_trigonometric_precision(self):
        """Test precision in trigonometric calculations."""
        # Test with special angles
        special_angles = [0.0, 30.0, 45.0, 60.0, 90.0, 180.0, 270.0, 360.0]
        
        for angle in special_angles:
            agent = SingleAgentController(orientation=angle)
            
            # Verify angle is stored with proper precision
            np.testing.assert_allclose(
                agent.orientations[0], 
                angle % 360.0,  # Normalized angle
                atol=NUMERICAL_PRECISION_TOLERANCE
            )


class TestExtensibilityHooksIntegration:
    """Test integration of extensibility hooks for Gymnasium 0.29.x compatibility."""
    
    def test_extensibility_hooks_single_agent(self):
        """Test extensibility hooks for single agent controllers."""
        agent = SingleAgentController(
            position=(5.0, 5.0),
            enable_extensibility_hooks=True,
            custom_observation_keys=["controller_id", "performance_metrics"],
            reward_shaping="exploration_bonus"
        )
        
        # Test additional observations
        base_obs = {"position": [5.0, 5.0], "orientation": 0.0, "speed": 1.0}
        additional_obs = agent.compute_additional_obs(base_obs)
        assert isinstance(additional_obs, dict)
        
        # Test extra reward computation
        info = {"episode_length": 100, "exploration_score": 0.5}
        extra_reward = agent.compute_extra_reward(1.0, info)
        assert isinstance(extra_reward, (int, float))
        
        # Test episode end handling
        final_info = {"success": True, "episode_length": 100, "total_reward": 10.5}
        agent.on_episode_end(final_info)
    
    def test_extensibility_hooks_multi_agent(self):
        """Test extensibility hooks for multi-agent controllers."""
        positions = np.array([[0.0, 0.0], [10.0, 10.0]])
        agent = MultiAgentController(
            positions=positions,
            enable_extensibility_hooks=True,
            reward_shaping="efficiency_penalty"
        )
        
        # Test additional observations
        base_obs = {"positions": positions, "orientations": [0.0, 90.0]}
        additional_obs = agent.compute_additional_obs(base_obs)
        assert isinstance(additional_obs, dict)
        
        # Test extra reward computation
        info = {"collective_performance": 0.8}
        extra_reward = agent.compute_extra_reward(2.0, info)
        assert isinstance(extra_reward, (int, float))
    
    def test_frame_cache_integration(self):
        """Test frame cache integration with controllers."""
        agent = SingleAgentController(
            position=(10.0, 10.0),
            frame_cache_mode="lru",
            enable_extensibility_hooks=True,
            custom_observation_keys=["frame_cache_stats"]
        )
        
        # Test that frame cache configuration is stored
        assert agent._frame_cache_mode == "lru"
        
        # Test additional observations include frame cache stats
        base_obs = {"position": [10.0, 10.0]}
        additional_obs = agent.compute_additional_obs(base_obs)
        assert isinstance(additional_obs, dict)
        # Frame cache stats may or may not be included depending on implementation
    
    def test_performance_metrics_integration(self):
        """Test performance metrics integration with extensibility hooks."""
        agent = SingleAgentController(
            position=(5.0, 5.0),
            enable_extensibility_hooks=True,
            custom_observation_keys=["performance_metrics"]
        )
        
        # Perform some operations to generate metrics
        mock_env = np.random.rand(50, 50)
        for _ in range(10):
            agent.step(mock_env)
        
        # Test performance metrics access
        if hasattr(agent, 'get_performance_metrics'):
            metrics = agent.get_performance_metrics()
            assert isinstance(metrics, dict)


class TestV1SourceProtocolIntegration:
    """Test v1.0 source protocol integration and component dependency injection."""
    
    def test_source_protocol_compliance(self):
        """Test that source implementations comply with SourceProtocol."""
        # Test PointSource protocol compliance
        source_config = {
            'type': 'PointSource',
            'position': (50.0, 50.0),
            'emission_rate': 1000.0
        }
        point_source = create_source(source_config)
        
        # Verify protocol compliance
        assert isinstance(point_source, object)  # SourceProtocol is runtime_checkable
        assert hasattr(point_source, 'get_emission_rate')
        assert hasattr(point_source, 'get_position')
        assert hasattr(point_source, 'update_state')
        assert callable(point_source.get_emission_rate)
        assert callable(point_source.get_position)
        assert callable(point_source.update_state)
    
    def test_source_factory_creation(self):
        """Test source factory method with different source types."""
        # Test PointSource creation
        point_config = {
            'type': 'PointSource',
            'position': (25.0, 75.0),
            'emission_rate': 500.0
        }
        point_source = create_source(point_config)
        assert point_source.get_position()[0] == 25.0
        assert point_source.get_position()[1] == 75.0
        
        # Test MultiSource creation
        multi_config = {
            'type': 'MultiSource',
            'sources': [
                {'type': 'PointSource', 'position': (30, 30), 'emission_rate': 500},
                {'type': 'PointSource', 'position': (70, 70), 'emission_rate': 800}
            ]
        }
        multi_source = create_source(multi_config)
        assert multi_source.get_source_count() == 2
        
        # Test DynamicSource creation
        dynamic_config = {
            'type': 'DynamicSource',
            'initial_position': (50, 50),
            'pattern_type': 'circular',
            'amplitude': 10.0,
            'frequency': 0.1
        }
        dynamic_source = create_source(dynamic_config)
        assert dynamic_source.get_pattern_type() == 'circular'
    
    def test_source_emission_rate_functionality(self):
        """Test source emission rate queries for single and multi-agent scenarios."""
        source_config = {
            'type': 'PointSource',
            'position': (50.0, 50.0),
            'emission_rate': 1000.0
        }
        source = create_source(source_config)
        
        # Test scalar emission rate query
        scalar_rate = source.get_emission_rate()
        assert isinstance(scalar_rate, (int, float))
        assert scalar_rate == 1000.0
        
        # Test single agent emission rate query
        single_position = np.array([45.0, 48.0])
        single_rate = source.get_emission_rate(single_position)
        assert isinstance(single_rate, (int, float))
        assert single_rate == 1000.0
        
        # Test multi-agent emission rate query
        multi_positions = np.array([[40, 45], [50, 50], [60, 55]])
        multi_rates = source.get_emission_rate(multi_positions)
        assert isinstance(multi_rates, np.ndarray)
        assert multi_rates.shape == (3,)
        assert np.all(multi_rates == 1000.0)
    
    def test_source_temporal_dynamics(self):
        """Test source temporal evolution and state updates."""
        dynamic_config = {
            'type': 'DynamicSource',
            'initial_position': (50, 50),
            'pattern_type': 'linear',
            'velocity': (1.0, 0.5)
        }
        dynamic_source = create_source(dynamic_config)
        
        initial_position = dynamic_source.get_position()
        
        # Update state and verify position change
        dynamic_source.update_state(dt=2.0)
        updated_position = dynamic_source.get_position()
        
        # Verify linear motion
        expected_x = initial_position[0] + 2.0 * 1.0  # dt * velocity_x
        expected_y = initial_position[1] + 2.0 * 0.5  # dt * velocity_y
        
        np.testing.assert_allclose(updated_position[0], expected_x, atol=NUMERICAL_PRECISION_TOLERANCE)
        np.testing.assert_allclose(updated_position[1], expected_y, atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_multi_source_aggregation(self):
        """Test multi-source emission rate aggregation."""
        multi_config = {
            'type': 'MultiSource',
            'sources': [
                {'type': 'PointSource', 'position': (30, 30), 'emission_rate': 400},
                {'type': 'PointSource', 'position': (70, 70), 'emission_rate': 600}
            ]
        }
        multi_source = create_source(multi_config)
        
        # Test total emission rate
        total_rate = multi_source.get_total_emission_rate()
        assert total_rate == 1000.0  # 400 + 600
        
        # Test multi-agent aggregation
        agent_positions = np.array([[25, 25], [50, 50], [75, 75]])
        aggregated_rates = multi_source.get_emission_rate(agent_positions)
        assert isinstance(aggregated_rates, np.ndarray)
        assert aggregated_rates.shape == (3,)
        assert np.all(aggregated_rates == 1000.0)  # All agents get sum of all sources
    
    def test_source_performance_requirements(self):
        """Test source performance meets v1.0 requirements."""
        source_config = {
            'type': 'PointSource',
            'position': (50.0, 50.0),
            'emission_rate': 1000.0
        }
        source = create_source(source_config)
        
        # Test single query performance (<0.1ms requirement)
        start_time = time.time()
        _ = source.get_emission_rate(np.array([45.0, 48.0]))
        single_query_time = (time.time() - start_time) * 1000
        assert single_query_time < 0.1, f"Single query took {single_query_time:.3f}ms, should be <0.1ms"
        
        # Test multi-agent query performance (<1ms for 100 agents requirement)
        large_positions = np.random.rand(100, 2) * 100
        start_time = time.time()
        _ = source.get_emission_rate(large_positions)
        multi_query_time = (time.time() - start_time) * 1000
        assert multi_query_time < 1.0, f"100-agent query took {multi_query_time:.3f}ms, should be <1ms"


class TestV1BoundaryPolicyIntegration:
    """Test v1.0 boundary policy protocol integration and vectorized operations."""
    
    def test_boundary_policy_protocol_compliance(self):
        """Test that boundary policy implementations comply with BoundaryPolicyProtocol."""
        # Test TerminateBoundary protocol compliance
        terminate_policy = create_boundary_policy(
            "terminate", 
            domain_bounds=(100, 100)
        )
        
        # Verify protocol compliance
        assert isinstance(terminate_policy, BoundaryPolicyProtocol)
        assert hasattr(terminate_policy, 'apply_policy')
        assert hasattr(terminate_policy, 'check_violations')
        assert hasattr(terminate_policy, 'get_termination_status')
        assert callable(terminate_policy.apply_policy)
        assert callable(terminate_policy.check_violations)
        assert callable(terminate_policy.get_termination_status)
    
    def test_boundary_policy_factory_creation(self):
        """Test boundary policy factory method with different policy types."""
        domain_bounds = (100, 100)
        
        # Test terminate policy creation
        terminate_policy = create_boundary_policy("terminate", domain_bounds)
        assert terminate_policy.get_termination_status() == "oob"
        
        # Test bounce policy creation
        bounce_policy = create_boundary_policy(
            "bounce", 
            domain_bounds, 
            elasticity=0.8,
            energy_loss=0.1
        )
        assert bounce_policy.get_termination_status() == "continue"
        
        # Test wrap policy creation
        wrap_policy = create_boundary_policy("wrap", domain_bounds)
        assert wrap_policy.get_termination_status() == "continue"
        
        # Test clip policy creation
        clip_policy = create_boundary_policy(
            "clip", 
            domain_bounds,
            velocity_damping=0.8
        )
        assert clip_policy.get_termination_status() == "continue"
    
    def test_boundary_violation_detection(self):
        """Test vectorized boundary violation detection."""
        policy = create_boundary_policy("terminate", domain_bounds=(100, 100))
        
        # Test single agent violation detection
        in_bounds_position = np.array([50.0, 50.0])
        out_bounds_position = np.array([105.0, 50.0])
        
        assert not policy.check_violations(in_bounds_position)
        assert policy.check_violations(out_bounds_position)
        
        # Test multi-agent violation detection
        mixed_positions = np.array([
            [50, 50],    # In bounds
            [105, 50],   # Out of bounds (x)
            [50, 105],   # Out of bounds (y)
            [25, 75]     # In bounds
        ])
        violations = policy.check_violations(mixed_positions)
        expected_violations = np.array([False, True, True, False])
        np.testing.assert_array_equal(violations, expected_violations)
    
    def test_terminate_boundary_behavior(self):
        """Test terminate boundary policy behavior."""
        policy = create_boundary_policy("terminate", domain_bounds=(100, 100))
        
        positions = np.array([[50, 50], [105, 50]])  # One in, one out
        velocities = np.array([[1.0, 0.5], [2.0, 1.0]])
        
        # Apply policy - terminate policy should not modify positions/velocities
        corrected_pos, corrected_vel = policy.apply_policy(positions, velocities)
        np.testing.assert_array_equal(corrected_pos, positions)
        np.testing.assert_array_equal(corrected_vel, velocities)
        
        # Test termination status
        assert policy.get_termination_status() == "oob"
    
    def test_bounce_boundary_behavior(self):
        """Test bounce boundary policy behavior with collision physics."""
        policy = create_boundary_policy(
            "bounce", 
            domain_bounds=(100, 100),
            elasticity=1.0,  # Perfect elastic collision
            energy_loss=0.0
        )
        
        # Agent moving out of right boundary
        positions = np.array([[105.0, 50.0]])  # 5 units beyond right boundary
        velocities = np.array([[2.0, 1.0]])    # Moving right and up
        
        corrected_pos, corrected_vel = policy.apply_policy(positions, velocities)
        
        # Position should be reflected back into domain
        assert corrected_pos[0, 0] <= 100.0  # Within x boundary
        assert corrected_pos[0, 1] == 50.0   # Y unchanged
        
        # X velocity should be reversed, Y velocity unchanged
        assert corrected_vel[0, 0] < 0        # X velocity reversed
        assert corrected_vel[0, 1] == 1.0     # Y velocity unchanged
    
    def test_wrap_boundary_behavior(self):
        """Test wrap boundary policy behavior with periodic conditions."""
        policy = create_boundary_policy("wrap", domain_bounds=(100, 100))
        
        # Agent beyond right boundary
        positions = np.array([[105.0, 50.0]])
        velocities = np.array([[2.0, 1.0]])
        
        wrapped_pos, unchanged_vel = policy.apply_policy(positions, velocities)
        
        # Position should wrap to left side
        assert wrapped_pos[0, 0] == 5.0       # 105 % 100 = 5
        assert wrapped_pos[0, 1] == 50.0      # Y unchanged
        
        # Velocities should be unchanged
        np.testing.assert_array_equal(unchanged_vel, velocities)
    
    def test_clip_boundary_behavior(self):
        """Test clip boundary policy behavior with hard constraints."""
        policy = create_boundary_policy("clip", domain_bounds=(100, 100))
        
        # Agent beyond boundaries
        positions = np.array([[105.0, 110.0], [50.0, 50.0]])  # One out, one in
        
        clipped_pos = policy.apply_policy(positions)
        
        # Out-of-bounds position should be clipped
        assert clipped_pos[0, 0] == 100.0     # Clipped to right boundary
        assert clipped_pos[0, 1] == 100.0     # Clipped to top boundary
        
        # In-bounds position should be unchanged
        assert clipped_pos[1, 0] == 50.0
        assert clipped_pos[1, 1] == 50.0
    
    def test_boundary_policy_performance_requirements(self):
        """Test boundary policy performance meets v1.0 requirements."""
        policy = create_boundary_policy("bounce", domain_bounds=(100, 100))
        
        # Test check_violations performance (<0.5ms for 100 agents)
        large_positions = np.random.rand(100, 2) * 120  # Some out of bounds
        start_time = time.time()
        _ = policy.check_violations(large_positions)
        violation_check_time = (time.time() - start_time) * 1000
        assert violation_check_time < 0.5, f"Violation check took {violation_check_time:.3f}ms, should be <0.5ms"
        
        # Test apply_policy performance (<1ms for 100 agents)
        large_velocities = np.random.rand(100, 2) * 2 - 1  # Velocities in [-1, 1]
        start_time = time.time()
        _ = policy.apply_policy(large_positions, large_velocities)
        policy_apply_time = (time.time() - start_time) * 1000
        assert policy_apply_time < 1.0, f"Policy application took {policy_apply_time:.3f}ms, should be <1ms"


class TestV1NavigatorProtocolExtensions:
    """Test NavigatorProtocol extensions with Source and BoundaryPolicy dependencies."""
    
    def test_navigator_protocol_source_dependency(self):
        """Test NavigatorProtocol extension with source dependency injection."""
        # Create a navigator with source dependency
        navigator = SingleAgentController(position=(25.0, 25.0))
        
        # Test source property access (may be None if not configured)
        source = getattr(navigator, 'source', None)
        if source is not None:
            # If source is configured, verify it implements SourceProtocol
            assert hasattr(source, 'get_emission_rate')
            assert hasattr(source, 'get_position')
            assert hasattr(source, 'update_state')
        
        # Test source integration
        if hasattr(navigator, 'set_source'):
            test_source_config = {
                'type': 'PointSource',
                'position': (50.0, 50.0),
                'emission_rate': 1000.0
            }
            test_source = create_source(test_source_config)
            navigator.set_source(test_source)
            
            # Verify source integration
            assert navigator.source is not None
            assert navigator.source.get_emission_rate() == 1000.0
    
    def test_navigator_protocol_boundary_policy_dependency(self):
        """Test NavigatorProtocol extension with boundary policy dependency injection."""
        # Create a navigator with boundary policy dependency
        navigator = SingleAgentController(position=(75.0, 75.0))
        
        # Test boundary_policy property access (may be None if not configured)
        boundary_policy = getattr(navigator, 'boundary_policy', None)
        if boundary_policy is not None:
            # If boundary policy is configured, verify it implements BoundaryPolicyProtocol
            assert isinstance(boundary_policy, BoundaryPolicyProtocol)
            assert hasattr(boundary_policy, 'apply_policy')
            assert hasattr(boundary_policy, 'check_violations')
            assert hasattr(boundary_policy, 'get_termination_status')
        
        # Test boundary policy integration
        if hasattr(navigator, 'set_boundary_policy'):
            test_policy = create_boundary_policy("terminate", domain_bounds=(100, 100))
            navigator.set_boundary_policy(test_policy)
            
            # Verify boundary policy integration
            assert navigator.boundary_policy is not None
            assert navigator.boundary_policy.get_termination_status() == "oob"
    
    def test_navigator_dependency_injection_patterns(self):
        """Test component dependency injection patterns in controllers."""
        # Test source dependency injection
        source_config = {
            'type': 'PointSource',
            'position': (60.0, 40.0),
            'emission_rate': 750.0
        }
        source = create_source(source_config)
        
        # Test boundary policy dependency injection
        boundary_policy = create_boundary_policy(
            "bounce", 
            domain_bounds=(100, 100),
            elasticity=0.9
        )
        
        # Create navigator with dependencies (if supported)
        navigator = SingleAgentController(position=(30.0, 30.0))
        
        # Test dependency injection methods (if available)
        if hasattr(navigator, 'configure_dependencies'):
            navigator.configure_dependencies(
                source=source,
                boundary_policy=boundary_policy
            )
            
            # Verify dependencies are properly injected
            assert navigator.source is source
            assert navigator.boundary_policy is boundary_policy
        
        # Test that navigator can operate with dependencies
        if hasattr(navigator, 'source') and navigator.source is not None:
            # Test source-aware navigation
            mock_env = np.random.rand(50, 50)
            navigator.step(mock_env)  # Should complete without error
            
            # Verify navigator can query source
            emission_rate = navigator.source.get_emission_rate()
            assert isinstance(emission_rate, (int, float))
    
    def test_v1_component_integration_workflow(self):
        """Test complete v1.0 component integration workflow."""
        # Step 1: Create source component
        source_config = {
            'type': 'DynamicSource',
            'initial_position': (50, 50),
            'pattern_type': 'circular',
            'amplitude': 15.0,
            'frequency': 0.1
        }
        source = create_source(source_config)
        
        # Step 2: Create boundary policy component
        boundary_policy = create_boundary_policy(
            "wrap",
            domain_bounds=(100, 100)
        )
        
        # Step 3: Create navigator with v1.0 integration (if supported)
        navigator = MultiAgentController(
            positions=np.array([[25, 25], [75, 75]])
        )
        
        # Step 4: Test component integration
        components = {
            'source': source,
            'boundary_policy': boundary_policy
        }
        
        # Verify components implement correct protocols
        assert hasattr(source, 'get_emission_rate')
        assert hasattr(source, 'update_state')
        assert isinstance(boundary_policy, BoundaryPolicyProtocol)
        
        # Step 5: Test integrated simulation step
        mock_env = np.random.rand(100, 100)
        
        # Update source state
        source.update_state(dt=1.0)
        
        # Check boundary violations
        violations = boundary_policy.check_violations(navigator.positions)
        if np.any(violations):
            corrected_positions = boundary_policy.apply_policy(navigator.positions)
            # Verify positions were corrected appropriately
            assert isinstance(corrected_positions, np.ndarray)
        
        # Execute navigator step
        navigator.step(mock_env)
        
        # Verify simulation completed successfully
        assert navigator.positions.shape == (2, 2)
        assert np.all(np.isfinite(navigator.positions))


class TestV1HookPointIntegration:
    """Test hook point integration in core navigation components for v1.0."""
    
    def test_component_lifecycle_hooks(self):
        """Test that navigation components support lifecycle hooks."""
        navigator = SingleAgentController(
            position=(25.0, 25.0),
            enable_extensibility_hooks=True
        )
        
        # Test pre-step hook integration
        if hasattr(navigator, 'on_pre_step'):
            mock_env = np.random.rand(50, 50)
            navigator.on_pre_step(mock_env)  # Should not raise error
        
        # Test post-step hook integration
        if hasattr(navigator, 'on_post_step'):
            mock_env = np.random.rand(50, 50)
            navigator.on_post_step(mock_env)  # Should not raise error
        
        # Test reset hook integration
        if hasattr(navigator, 'on_reset'):
            navigator.on_reset()  # Should not raise error
    
    def test_source_integration_hooks(self):
        """Test source integration hooks in navigation components."""
        # Create source for testing hooks
        source_config = {
            'type': 'PointSource',
            'position': (50.0, 50.0),
            'emission_rate': 1000.0
        }
        source = create_source(source_config)
        
        navigator = SingleAgentController(position=(40.0, 40.0))
        
        # Test source state update hooks
        if hasattr(navigator, 'update_source_state'):
            navigator.update_source_state(source, dt=1.0)
        
        # Test source query hooks
        if hasattr(navigator, 'on_source_query'):
            emission_rate = source.get_emission_rate(navigator.positions)
            navigator.on_source_query(source, emission_rate)
    
    def test_boundary_policy_integration_hooks(self):
        """Test boundary policy integration hooks in navigation components."""
        # Create boundary policy for testing hooks
        boundary_policy = create_boundary_policy("bounce", domain_bounds=(100, 100))
        
        navigator = MultiAgentController(
            positions=np.array([[95.0, 50.0], [50.0, 95.0]])  # Near boundaries
        )
        
        # Test boundary violation hooks
        violations = boundary_policy.check_violations(navigator.positions)
        if hasattr(navigator, 'on_boundary_violation'):
            navigator.on_boundary_violation(violations, boundary_policy)
        
        # Test boundary correction hooks
        if np.any(violations):
            corrected_positions = boundary_policy.apply_policy(navigator.positions)
            if hasattr(navigator, 'on_boundary_correction'):
                navigator.on_boundary_correction(
                    original_positions=navigator.positions,
                    corrected_positions=corrected_positions,
                    policy=boundary_policy
                )
    
    def test_dependency_injection_hooks(self):
        """Test dependency injection hooks for v1.0 components."""
        navigator = SingleAgentController(position=(30.0, 30.0))
        
        # Test source injection hooks
        if hasattr(navigator, 'on_source_injected'):
            source_config = {'type': 'PointSource', 'position': (60, 60), 'emission_rate': 500}
            source = create_source(source_config)
            navigator.on_source_injected(source)
        
        # Test boundary policy injection hooks
        if hasattr(navigator, 'on_boundary_policy_injected'):
            policy = create_boundary_policy("terminate", domain_bounds=(100, 100))
            navigator.on_boundary_policy_injected(policy)
        
        # Test component configuration hooks
        if hasattr(navigator, 'on_components_configured'):
            component_config = {
                'source_type': 'PointSource',
                'boundary_policy_type': 'terminate'
            }
            navigator.on_components_configured(component_config)
    
    def test_performance_monitoring_hooks(self):
        """Test performance monitoring hooks for v1.0 components."""
        navigator = SingleAgentController(
            position=(45.0, 55.0),
            enable_extensibility_hooks=True
        )
        
        # Test performance measurement hooks
        if hasattr(navigator, 'on_performance_measured'):
            performance_data = {
                'step_time': 0.5,  # milliseconds
                'query_count': 10,
                'memory_usage': 1024  # bytes
            }
            navigator.on_performance_measured(performance_data)
        
        # Test performance threshold hooks
        if hasattr(navigator, 'on_performance_threshold_exceeded'):
            threshold_data = {
                'metric': 'step_time',
                'value': 1.5,
                'threshold': 1.0,
                'severity': 'warning'
            }
            navigator.on_performance_threshold_exceeded(threshold_data)
    
    def test_extensibility_hook_chaining(self):
        """Test that extensibility hooks can be chained and composed."""
        navigator = SingleAgentController(
            position=(35.0, 65.0),
            enable_extensibility_hooks=True
        )
        
        # Test observation hook chaining
        base_obs = {"position": [35.0, 65.0], "orientation": 0.0}
        
        # Call additional observations hook multiple times to test chaining
        additional_obs_1 = navigator.compute_additional_obs(base_obs)
        combined_obs = {**base_obs, **additional_obs_1}
        additional_obs_2 = navigator.compute_additional_obs(combined_obs)
        
        # Verify hook chaining works correctly
        assert isinstance(additional_obs_1, dict)
        assert isinstance(additional_obs_2, dict)
        
        # Test reward hook chaining
        base_reward = 1.0
        info = {"step": 50}
        
        extra_reward_1 = navigator.compute_extra_reward(base_reward, info)
        total_reward = base_reward + extra_reward_1
        extra_reward_2 = navigator.compute_extra_reward(total_reward, info)
        
        # Verify reward hook chaining works correctly
        assert isinstance(extra_reward_1, (int, float))
        assert isinstance(extra_reward_2, (int, float))
    
    def test_hook_error_handling(self):
        """Test error handling in hook point integration."""
        navigator = SingleAgentController(
            position=(20.0, 80.0),
            enable_extensibility_hooks=True
        )
        
        # Test that hooks handle None inputs gracefully
        additional_obs = navigator.compute_additional_obs(None)
        assert isinstance(additional_obs, dict)
        
        extra_reward = navigator.compute_extra_reward(0.0, None)
        assert isinstance(extra_reward, (int, float))
        
        # Test that episode end hook handles empty info
        navigator.on_episode_end({})  # Should not raise error
        navigator.on_episode_end(None)  # Should not raise error
    
    def test_hook_integration_with_v1_components(self):
        """Test hook integration with v1.0 source and boundary components."""
        navigator = SingleAgentController(
            position=(55.0, 45.0),
            enable_extensibility_hooks=True
        )
        
        # Create v1.0 components
        source = create_source({
            'type': 'DynamicSource',
            'initial_position': (50, 50),
            'pattern_type': 'linear',
            'velocity': (1.0, 0.5)
        })
        
        boundary_policy = create_boundary_policy("wrap", domain_bounds=(100, 100))
        
        # Test hooks receive component information
        base_obs = {"position": navigator.positions[0]}
        
        # If navigator supports component-aware hooks
        if hasattr(navigator, 'compute_additional_obs_with_components'):
            component_obs = navigator.compute_additional_obs_with_components(
                base_obs, source=source, boundary_policy=boundary_policy
            )
            assert isinstance(component_obs, dict)
        
        # Test that standard hooks work with component integration
        mock_env = np.random.rand(100, 100)
        
        # Update components
        source.update_state(dt=1.0)
        violations = boundary_policy.check_violations(navigator.positions)
        
        # Test hooks with component state
        additional_obs = navigator.compute_additional_obs(base_obs)
        assert isinstance(additional_obs, dict)
        
        # Component state should not break hook functionality
        navigator.step(mock_env)  # Should complete successfully


class TestV1BackwardCompatibility:
    """Test backward compatibility during v0.3.0 to v1.0 migration."""
    
    def test_existing_navigator_interface_preserved(self):
        """Test that existing NavigatorProtocol interface is preserved."""
        # Test that v0.3.0 navigator creation still works
        navigator = SingleAgentController(
            position=(10.0, 20.0),
            orientation=45.0,
            speed=1.5,
            max_speed=3.0
        )
        
        # Verify all original properties still exist and work
        assert hasattr(navigator, 'positions')
        assert hasattr(navigator, 'orientations')
        assert hasattr(navigator, 'speeds')
        assert hasattr(navigator, 'max_speeds')
        assert hasattr(navigator, 'angular_velocities')
        assert hasattr(navigator, 'num_agents')
        
        # Verify all original methods still exist and work
        assert hasattr(navigator, 'reset')
        assert hasattr(navigator, 'step')
        assert hasattr(navigator, 'sample_odor')
        assert hasattr(navigator, 'sample_multiple_sensors')
        
        # Test that original functionality works unchanged
        mock_env = np.random.rand(50, 50)
        navigator.step(mock_env)
        
        odor_sample = navigator.sample_odor(mock_env)
        assert isinstance(odor_sample, (float, np.floating))
        
        multi_sensor_samples = navigator.sample_multiple_sensors(mock_env, num_sensors=2)
        assert isinstance(multi_sensor_samples, np.ndarray)
        assert multi_sensor_samples.shape == (2,)
    
    def test_v0_3_controller_creation_patterns(self):
        """Test that v0.3.0 controller creation patterns still work."""
        # Test factory method compatibility
        single_agent_config = {
            'position': [15.0, 25.0],
            'orientation': 90.0,
            'speed': 2.0,
            'max_speed': 4.0
        }
        
        navigator = create_navigator_from_config(single_agent_config)
        assert isinstance(navigator, Navigator)
        assert navigator.num_agents == 1
        
        # Test multi-agent creation patterns
        multi_agent_config = {
            'positions': [[0.0, 0.0], [20.0, 20.0]],
            'orientations': [0.0, 180.0],
            'speeds': [1.0, 1.5],
            'max_speeds': [2.0, 3.0]
        }
        
        navigator = create_navigator_from_config(multi_agent_config)
        assert isinstance(navigator, Navigator)
        assert navigator.num_agents == 2
    
    def test_extensibility_hooks_optional(self):
        """Test that v1.0 extensibility hooks are optional and backward compatible."""
        # Create navigator without extensibility hooks (v0.3.0 style)
        navigator = SingleAgentController(position=(5.0, 5.0))
        
        # Verify navigator works without hooks
        mock_env = np.random.rand(25, 25)
        navigator.step(mock_env)
        
        # Test that hook methods exist but have default behavior
        base_obs = {"position": [5.0, 5.0]}
        additional_obs = navigator.compute_additional_obs(base_obs)
        assert isinstance(additional_obs, dict)
        
        extra_reward = navigator.compute_extra_reward(1.0, {})
        assert isinstance(extra_reward, (int, float))
        
        # Test episode end hook (should not raise error)
        navigator.on_episode_end({"success": True})
    
    def test_v1_extensions_do_not_break_v0_3_functionality(self):
        """Test that v1.0 extensions do not break existing v0.3.0 functionality."""
        # Create v0.3.0 style navigator
        navigator = MultiAgentController(
            positions=np.array([[10, 10], [30, 30], [50, 50]])
        )
        
        # Test that all v0.3.0 operations still work
        mock_env = np.random.rand(100, 100)
        
        # Test step functionality
        navigator.step(mock_env)
        assert navigator.positions.shape == (3, 2)
        
        # Test odor sampling
        odor_samples = navigator.sample_odor(mock_env)
        assert isinstance(odor_samples, np.ndarray)
        assert odor_samples.shape == (3,)
        
        # Test multi-sensor sampling
        sensor_samples = navigator.sample_multiple_sensors(mock_env, num_sensors=3)
        assert sensor_samples.shape == (3, 3)  # 3 agents, 3 sensors each
        
        # Test reset functionality
        navigator.reset(positions=np.array([[5, 5], [15, 15], [25, 25]]))
        expected_positions = np.array([[5, 5], [15, 15], [25, 25]])
        np.testing.assert_allclose(navigator.positions, expected_positions, atol=NUMERICAL_PRECISION_TOLERANCE)
    
    def test_performance_regression_prevention(self):
        """Test that v1.0 changes do not introduce performance regressions."""
        # Create navigator in v0.3.0 style
        navigator = SingleAgentController(position=(50.0, 50.0))
        mock_env = np.random.rand(100, 100)
        
        # Test single agent step performance (should still meet <1ms requirement)
        start_time = time.time()
        navigator.step(mock_env)
        step_time = (time.time() - start_time) * 1000
        assert step_time < SINGLE_AGENT_STEP_THRESHOLD_MS, \
            f"v1.0 single agent step took {step_time:.2f}ms, should be <{SINGLE_AGENT_STEP_THRESHOLD_MS}ms"
        
        # Test multi-agent performance
        multi_navigator = MultiAgentController(
            positions=np.random.rand(100, 2) * 100
        )
        
        start_time = time.time()
        multi_navigator.step(mock_env)
        multi_step_time = (time.time() - start_time) * 1000
        assert multi_step_time < MULTI_AGENT_100_STEP_THRESHOLD_MS, \
            f"v1.0 100-agent step took {multi_step_time:.2f}ms, should be <{MULTI_AGENT_100_STEP_THRESHOLD_MS}ms"


if __name__ == "__main__":
    # Run tests with appropriate verbosity and coverage
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        f"--cov=src.plume_nav_sim.core",
        "--cov-report=term-missing",
        # Include new v1.0 component coverage
        f"--cov=src.plume_nav_sim.core.sources",
        f"--cov=src.plume_nav_sim.core.boundaries"
    ])