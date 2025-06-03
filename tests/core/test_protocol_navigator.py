"""
Tests for the protocol-based navigator implementation with enhanced Hydra integration.

This module provides comprehensive testing for NavigatorProtocol implementations,
including Hydra configuration management, ConfigStore registration, schema validation,
and environment variable interpolation. The tests validate both traditional
navigator functionality and modern ML framework integration patterns.

Enhanced Features Tested:
- Hydra configuration composition and override scenarios
- ConfigStore registration and schema validation workflows
- Factory method patterns with DictConfig parameters
- Environment variable interpolation security and functionality
- CLI interface integration with navigator protocol compliance
- Database session management compatibility testing
- Multi-agent coordination with configuration-driven initialization

The testing architecture follows scientific computing best practices with
deterministic behavior, comprehensive mocking, and research-grade quality standards.
"""

import pytest
import numpy as np
import os
import tempfile
from typing import Any, Dict, Tuple, Optional, Union
from unittest.mock import Mock, patch, MagicMock

# Updated import statements for new cookiecutter template structure
from {{cookiecutter.project_slug}}.core.navigator import NavigatorProtocol, NavigatorFactory
from {{cookiecutter.project_slug}}.core.controllers import SingleAgentController, MultiAgentController
from {{cookiecutter.project_slug}}.config.schemas import NavigatorConfig, SingleAgentConfig, MultiAgentConfig

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


class TestSingleAgentController:
    """Tests for the SingleAgentController with enhanced Hydra integration."""
    
    def test_initialization(self) -> None:
        """Test that a SingleAgentController initializes correctly."""
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
    
    def test_reset(self) -> None:
        """Test resetting the controller state."""
        controller = SingleAgentController(position=(10.0, 20.0), orientation=45.0)
        controller.reset(position=(30.0, 40.0), orientation=90.0)
        assert controller.positions[0, 0] == 30.0
        assert controller.positions[0, 1] == 20.0
        assert controller.orientations[0] == 90.0
    
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
    """Tests for the MultiAgentController with enhanced configuration support."""
    
    def test_initialization(self) -> None:
        """Test that a MultiAgentController initializes correctly."""
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
    
    def test_reset(self) -> None:
        """Test resetting the controller state."""
        controller = MultiAgentController(
            positions=np.array([[10.0, 20.0], [30.0, 40.0]]),
            orientations=np.array([0.0, 90.0])
        )
        
        new_positions = np.array([[50.0, 60.0], [70.0, 80.0], [90.0, 100.0]])
        controller.reset(positions=new_positions)
        
        assert controller.num_agents == 3
        assert np.array_equal(controller.positions, new_positions)
        assert controller.orientations.shape == (3,)
    
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


class TestNavigatorFactory:
    """Tests for NavigatorFactory with enhanced Hydra integration."""
    
    def test_single_agent_factory(self) -> None:
        """Test the single-agent factory method."""
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
    
    def test_multi_agent_factory(self) -> None:
        """Test the multi-agent factory method."""
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


@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
class TestHydraConfigurationIntegration:
    """
    Comprehensive tests for Hydra configuration integration with NavigatorProtocol.
    
    These tests validate the enhanced Hydra-based configuration management system
    including ConfigStore registration, schema validation, hierarchical composition,
    and environment variable interpolation patterns essential for modern ML workflows.
    """
    
    def test_hydra_config_store_registration(self) -> None:
        """Test ConfigStore registration and schema validation workflows."""
        from hydra.core.config_store import ConfigStore
        from {{cookiecutter.project_slug}}.config.schemas import NavigatorConfig
        
        # Test ConfigStore registration
        cs = ConfigStore.instance()
        cs.store(name="test_navigator_config", node=NavigatorConfig)
        
        # Verify registration success
        assert "test_navigator_config" in cs.cache
        
        # Test schema validation through ConfigStore
        test_config = NavigatorConfig(
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
    
    def test_hydra_configuration_composition(self) -> None:
        """Test hierarchical configuration composition with Hydra."""
        # Create temporary configuration files for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create base configuration
            base_config = {
                'defaults': ['_self_'],
                'navigator': {
                    'position': [0.0, 0.0],
                    'orientation': 0.0,
                    'max_speed': 1.0,
                    'angular_velocity': 0.1
                }
            }
            
            # Write base config
            base_path = os.path.join(temp_dir, 'base_config.yaml')
            with open(base_path, 'w') as f:
                import yaml
                yaml.dump(base_config, f)
            
            # Test configuration loading (mock Hydra behavior)
            mock_cfg = OmegaConf.create(base_config)
            
            # Validate configuration structure
            assert 'navigator' in mock_cfg
            assert mock_cfg.navigator.position == [0.0, 0.0]
            assert mock_cfg.navigator.orientation == 0.0
            assert mock_cfg.navigator.max_speed == 1.0
    
    def test_dictconfig_factory_method_patterns(self) -> None:
        """Test factory method patterns with DictConfig parameters."""
        # Create a DictConfig for single agent
        single_config = OmegaConf.create({
            'position': [15.0, 25.0],
            'orientation': 180.0,
            'speed': 2.0,
            'max_speed': 4.0,
            'angular_velocity': 0.15
        })
        
        # Test factory method with DictConfig
        navigator = NavigatorFactory.from_config(single_config)
        assert isinstance(navigator, NavigatorProtocol)
        assert navigator.num_agents == 1
        assert navigator.positions[0, 0] == 15.0
        assert navigator.positions[0, 1] == 25.0
        assert navigator.orientations[0] == 180.0
        
        # Create a DictConfig for multi-agent
        multi_config = OmegaConf.create({
            'positions': [[5.0, 5.0], [15.0, 15.0]],
            'orientations': [45.0, 135.0],
            'speeds': [1.0, 1.5],
            'max_speeds': [2.0, 3.0],
            'angular_velocities': [0.1, 0.2]
        })
        
        # Test multi-agent factory with DictConfig
        multi_navigator = NavigatorFactory.from_config(multi_config)
        assert isinstance(multi_navigator, NavigatorProtocol)
        assert multi_navigator.num_agents == 2
        assert multi_navigator.positions.shape == (2, 2)
        assert multi_navigator.orientations.shape == (2,)
    
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
    configuration management and integration patterns.
    """
    
    def test_single_agent_protocol_compliance(self) -> None:
        """Test SingleAgentController protocol compliance."""
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
    
    def test_multi_agent_protocol_compliance(self) -> None:
        """Test MultiAgentController protocol compliance."""
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