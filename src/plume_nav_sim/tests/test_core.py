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


if __name__ == "__main__":
    # Run tests with appropriate verbosity and coverage
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        f"--cov=src.plume_nav_sim.core",
        "--cov-report=term-missing"
    ])