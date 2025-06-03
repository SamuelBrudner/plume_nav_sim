"""Tests for the protocol-based navigator implementation with Hydra integration.

This module provides comprehensive testing for NavigatorProtocol compliance, controller
implementations, and enhanced Hydra configuration integration. Tests validate the
refactored cookiecutter-based architecture with configuration composition, factory
method patterns, and environment variable interpolation.

Enhanced test coverage includes:
- Hydra ConfigStore registration and schema validation
- DictConfig parameter handling in factory methods
- Environment variable interpolation in configuration composition
- Protocol interface compliance across single and multi-agent scenarios
- Configuration-driven parameter validation and type safety
"""

import pytest
import numpy as np
from typing import Any, Dict, Tuple, Optional
from unittest.mock import patch, MagicMock
import tempfile
import os

# Updated imports reflecting new cookiecutter template structure
from {{cookiecutter.project_slug}}.core.navigator import NavigatorProtocol
from {{cookiecutter.project_slug}}.core.controllers import SingleAgentController, MultiAgentController
from {{cookiecutter.project_slug}}.config.schemas import NavigatorConfig, SingleAgentConfig, MultiAgentConfig

# Hydra integration imports for enhanced configuration testing
try:
    from hydra import compose, initialize
    from hydra.core.config_store import ConfigStore
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    # Fallback for environments without Hydra
    DictConfig = dict
    OmegaConf = None


class TestNavigatorProtocolCompliance:
    """Test NavigatorProtocol interface compliance across all implementations."""
    
    def test_single_agent_controller_protocol_compliance(self) -> None:
        """Test that SingleAgentController implements NavigatorProtocol correctly."""
        controller = SingleAgentController()
        
        # Verify protocol compliance
        assert isinstance(controller, NavigatorProtocol)
        
        # Test all required properties exist and return correct types
        assert hasattr(controller, 'positions')
        assert hasattr(controller, 'orientations') 
        assert hasattr(controller, 'speeds')
        assert hasattr(controller, 'max_speeds')
        assert hasattr(controller, 'angular_velocities')
        assert hasattr(controller, 'num_agents')
        
        # Test all required methods exist
        assert hasattr(controller, 'reset')
        assert hasattr(controller, 'step')
        assert hasattr(controller, 'sample_odor')
        assert hasattr(controller, 'read_single_antenna_odor')
        assert hasattr(controller, 'sample_multiple_sensors')
        
        # Test property return types
        assert isinstance(controller.positions, np.ndarray)
        assert isinstance(controller.orientations, np.ndarray)
        assert isinstance(controller.speeds, np.ndarray)
        assert isinstance(controller.max_speeds, np.ndarray)
        assert isinstance(controller.angular_velocities, np.ndarray)
        assert isinstance(controller.num_agents, int)
    
    def test_multi_agent_controller_protocol_compliance(self) -> None:
        """Test that MultiAgentController implements NavigatorProtocol correctly."""
        positions = np.array([[10.0, 20.0], [30.0, 40.0]])
        controller = MultiAgentController(positions=positions)
        
        # Verify protocol compliance
        assert isinstance(controller, NavigatorProtocol)
        
        # Test all required properties and methods exist (same as single agent)
        assert hasattr(controller, 'positions')
        assert hasattr(controller, 'orientations')
        assert hasattr(controller, 'speeds')
        assert hasattr(controller, 'max_speeds')
        assert hasattr(controller, 'angular_velocities')
        assert hasattr(controller, 'num_agents')
        
        assert hasattr(controller, 'reset')
        assert hasattr(controller, 'step')
        assert hasattr(controller, 'sample_odor')
        assert hasattr(controller, 'read_single_antenna_odor')
        assert hasattr(controller, 'sample_multiple_sensors')
        
        # Test multi-agent specific behavior
        assert controller.num_agents == 2
        assert controller.positions.shape == (2, 2)
        assert controller.orientations.shape == (2,)


class TestSingleAgentController:
    """Tests for the SingleAgentController with enhanced configuration support."""
    
    def test_initialization_default(self) -> None:
        """Test that a SingleAgentController initializes correctly with defaults."""
        controller = SingleAgentController()
        assert controller.num_agents == 1
        assert controller.positions.shape == (1, 2)
        assert controller.orientations.shape == (1,)
        assert controller.speeds.shape == (1,)
        assert controller.max_speeds.shape == (1,)
        assert controller.angular_velocities.shape == (1,)
    
    def test_initialization_with_parameters(self) -> None:
        """Test SingleAgentController initialization with custom parameters."""
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
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_initialization_with_pydantic_config(self) -> None:
        """Test SingleAgentController initialization with Pydantic configuration."""
        config = SingleAgentConfig(
            position=(15.0, 25.0),
            orientation=90.0,
            speed=1.5,
            max_speed=3.0,
            angular_velocity=5.0
        )
        
        controller = SingleAgentController(config=config)
        assert controller.positions[0, 0] == 15.0
        assert controller.positions[0, 1] == 25.0
        assert controller.orientations[0] == 90.0
        assert controller.speeds[0] == 1.5
        assert controller.max_speeds[0] == 3.0
        assert controller.angular_velocities[0] == 5.0
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_initialization_with_dict_config(self) -> None:
        """Test SingleAgentController initialization with DictConfig."""
        config_dict = {
            "position": [20.0, 30.0],
            "orientation": 180.0,
            "speed": 2.5,
            "max_speed": 4.0,
            "angular_velocity": 8.0
        }
        
        if OmegaConf is not None:
            config = OmegaConf.create(config_dict)
        else:
            config = config_dict
        
        controller = SingleAgentController(config=config)
        assert controller.positions[0, 0] == 20.0
        assert controller.positions[0, 1] == 30.0
        assert controller.orientations[0] == 180.0
        assert controller.speeds[0] == 2.5
        assert controller.max_speeds[0] == 4.0
        assert controller.angular_velocities[0] == 8.0
    
    def test_reset_with_parameters(self) -> None:
        """Test resetting the controller state with new parameters."""
        controller = SingleAgentController(position=(10.0, 20.0), orientation=45.0)
        controller.reset(position=(30.0, 40.0), orientation=90.0)
        assert controller.positions[0, 0] == 30.0
        assert controller.positions[0, 1] == 40.0
        assert controller.orientations[0] == 90.0
    
    def test_step_updates_state(self) -> None:
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
    
    def test_sample_odor_at_position(self) -> None:
        """Test sampling odor at the agent's position."""
        controller = SingleAgentController(position=(5, 5))
        
        # Create an environment with a known value at the agent's position
        env_array = np.zeros((10, 10))
        env_array[5, 5] = 1.0
        
        odor = controller.sample_odor(env_array)
        assert odor == 1.0
    
    def test_sample_multiple_sensors_arrangement(self) -> None:
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
    
    def test_predefined_sensor_layout(self) -> None:
        """Test using a predefined sensor layout."""
        controller = SingleAgentController(position=(50, 50), orientation=0.0)
        
        env_array = np.zeros((100, 100))
        y, x = np.ogrid[:100, :100]
        env_array += np.exp(-((x - 50)**2 + (y - 50)**2) / 100)
        
        odor_values = controller.sample_multiple_sensors(
            env_array, 
            layout_name="LEFT_RIGHT",
            sensor_distance=10.0
        )
        
        # LEFT_RIGHT has 2 sensors
        assert isinstance(odor_values, np.ndarray)
        assert odor_values.shape == (2,)


class TestMultiAgentController:
    """Tests for the MultiAgentController with enhanced configuration support."""
    
    def test_initialization_default(self) -> None:
        """Test that a MultiAgentController initializes correctly with defaults."""
        controller = MultiAgentController()
        assert controller.num_agents == 1
        assert controller.positions.shape == (1, 2)
    
    def test_initialization_with_parameters(self) -> None:
        """Test MultiAgentController initialization with custom parameters."""
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
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_initialization_with_pydantic_config(self) -> None:
        """Test MultiAgentController initialization with Pydantic configuration."""
        config = MultiAgentConfig(
            positions=[[5.0, 10.0], [15.0, 20.0], [25.0, 30.0]],
            orientations=[0.0, 120.0, 240.0],
            speeds=[1.0, 1.5, 2.0],
            max_speeds=[2.0, 3.0, 4.0],
            angular_velocities=[5.0, 7.5, 10.0]
        )
        
        controller = MultiAgentController(config=config)
        assert controller.num_agents == 3
        assert controller.positions.shape == (3, 2)
        assert controller.orientations.shape == (3,)
        assert np.allclose(controller.positions[0], [5.0, 10.0])
        assert np.allclose(controller.orientations, [0.0, 120.0, 240.0])
    
    def test_reset_with_new_agent_configuration(self) -> None:
        """Test resetting the controller state with new agent configuration."""
        controller = MultiAgentController(
            positions=np.array([[10.0, 20.0], [30.0, 40.0]]),
            orientations=np.array([0.0, 90.0])
        )
        
        new_positions = np.array([[50.0, 60.0], [70.0, 80.0], [90.0, 100.0]])
        controller.reset(positions=new_positions)
        
        assert controller.num_agents == 3
        assert np.array_equal(controller.positions, new_positions)
        assert controller.orientations.shape == (3,)
    
    def test_step_updates_multiple_agents(self) -> None:
        """Test the step method updates positions and orientations for multiple agents."""
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
    
    def test_sample_odor_multiple_agents(self) -> None:
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
    
    def test_sample_multiple_sensors_multiple_agents(self) -> None:
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


@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
class TestHydraConfigurationIntegration:
    """Tests for Hydra configuration integration with NavigatorProtocol and controllers."""
    
    def setup_method(self):
        """Set up test configuration for each test method."""
        # Clear any existing Hydra instance
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Clear Hydra instance after each test
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
    
    def test_hydra_config_store_registration(self) -> None:
        """Test that configuration schemas are properly registered with Hydra ConfigStore."""
        cs = ConfigStore.instance()
        
        # Register schemas with ConfigStore
        cs.store(name="navigator_config", node=NavigatorConfig)
        cs.store(name="single_agent_config", node=SingleAgentConfig)
        cs.store(name="multi_agent_config", node=MultiAgentConfig)
        
        # Verify schemas are registered
        assert cs.load("navigator_config.yaml") is not None
        assert cs.load("single_agent_config.yaml") is not None
        assert cs.load("multi_agent_config.yaml") is not None
    
    def test_hydra_configuration_composition(self) -> None:
        """Test Hydra configuration composition with hierarchical parameters."""
        # Create temporary configuration files
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = os.path.join(temp_dir, "conf")
            os.makedirs(config_dir)
            
            # Create base configuration
            base_config = """
defaults:
  - _self_

navigator:
  type: "single"
  position: [50.0, 50.0]
  orientation: 0.0
  speed: 1.0
  max_speed: 5.0
  angular_velocity: 0.1
"""
            
            # Create override configuration
            override_config = """
# @package _global_
navigator:
  position: [100.0, 100.0]
  speed: 2.0
"""
            
            # Write configuration files
            with open(os.path.join(config_dir, "config.yaml"), "w") as f:
                f.write(base_config)
            
            with open(os.path.join(config_dir, "override.yaml"), "w") as f:
                f.write(override_config)
            
            # Test configuration composition
            with initialize(config_path=config_dir, version_base=None):
                cfg = compose(config_name="config")
                
                # Verify base configuration is loaded
                assert cfg.navigator.type == "single"
                assert cfg.navigator.position == [50.0, 50.0]
                assert cfg.navigator.orientation == 0.0
                assert cfg.navigator.speed == 1.0
                assert cfg.navigator.max_speed == 5.0
                assert cfg.navigator.angular_velocity == 0.1
                
                # Test override composition
                cfg_override = compose(config_name="config", overrides=["override"])
                assert cfg_override.navigator.position == [100.0, 100.0]
                assert cfg_override.navigator.speed == 2.0
                # Other values should remain from base
                assert cfg_override.navigator.max_speed == 5.0
    
    def test_environment_variable_interpolation(self) -> None:
        """Test environment variable interpolation in configuration composition."""
        # Set test environment variables
        test_env = {
            "NAVIGATOR_MAX_SPEED": "7.5",
            "NAVIGATOR_POSITION_X": "75.0",
            "NAVIGATOR_POSITION_Y": "85.0",
            "ANGULAR_VELOCITY": "0.25"
        }
        
        with patch.dict(os.environ, test_env):
            # Create temporary configuration with environment variable interpolation
            with tempfile.TemporaryDirectory() as temp_dir:
                config_dir = os.path.join(temp_dir, "conf")
                os.makedirs(config_dir)
                
                env_config = """
defaults:
  - _self_

navigator:
  type: "single"
  position: [${oc.env:NAVIGATOR_POSITION_X,50.0}, ${oc.env:NAVIGATOR_POSITION_Y,50.0}]
  orientation: 0.0
  speed: 1.0
  max_speed: ${oc.env:NAVIGATOR_MAX_SPEED,5.0}
  angular_velocity: ${oc.env:ANGULAR_VELOCITY,0.1}
"""
                
                with open(os.path.join(config_dir, "config.yaml"), "w") as f:
                    f.write(env_config)
                
                # Test environment variable interpolation
                with initialize(config_path=config_dir, version_base=None):
                    cfg = compose(config_name="config")
                    
                    # Verify environment variables are interpolated
                    assert cfg.navigator.position == [75.0, 85.0]
                    assert cfg.navigator.max_speed == 7.5
                    assert cfg.navigator.angular_velocity == 0.25
                    # Non-interpolated values should remain as defaults
                    assert cfg.navigator.orientation == 0.0
                    assert cfg.navigator.speed == 1.0
    
    def test_schema_validation_with_hydra_config(self) -> None:
        """Test that Pydantic schema validation works with Hydra DictConfig objects."""
        # Create a valid configuration dictionary
        config_dict = {
            "position": [25.0, 35.0],
            "orientation": 45.0,
            "speed": 1.5,
            "max_speed": 3.0,
            "angular_velocity": 0.2
        }
        
        # Convert to DictConfig
        dict_config = OmegaConf.create(config_dict)
        
        # Test that Pydantic schema can validate DictConfig
        try:
            validated_config = SingleAgentConfig(**dict_config)
            assert validated_config.position == (25.0, 35.0)
            assert validated_config.orientation == 45.0
            assert validated_config.speed == 1.5
            assert validated_config.max_speed == 3.0
            assert validated_config.angular_velocity == 0.2
        except Exception as e:
            pytest.fail(f"Schema validation failed with DictConfig: {e}")
    
    def test_configuration_override_scenarios(self) -> None:
        """Test various Hydra configuration override scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = os.path.join(temp_dir, "conf")
            os.makedirs(config_dir)
            
            base_config = """
defaults:
  - _self_

navigator:
  type: "single"
  position: [0.0, 0.0]
  orientation: 0.0
  speed: 1.0
  max_speed: 5.0
  angular_velocity: 0.1

simulation:
  steps: 100
  recording: true
"""
            
            with open(os.path.join(config_dir, "config.yaml"), "w") as f:
                f.write(base_config)
            
            with initialize(config_path=config_dir, version_base=None):
                # Test single parameter override
                cfg1 = compose(config_name="config", overrides=[
                    "navigator.max_speed=10.0"
                ])
                assert cfg1.navigator.max_speed == 10.0
                assert cfg1.navigator.speed == 1.0  # Should remain unchanged
                
                # Test multiple parameter overrides
                cfg2 = compose(config_name="config", overrides=[
                    "navigator.position=[100.0,200.0]",
                    "navigator.orientation=90.0",
                    "simulation.steps=500"
                ])
                assert cfg2.navigator.position == [100.0, 200.0]
                assert cfg2.navigator.orientation == 90.0
                assert cfg2.simulation.steps == 500
                
                # Test nested override with dot notation
                cfg3 = compose(config_name="config", overrides=[
                    "navigator.position=[50.0,75.0]",
                    "navigator.speed=2.5"
                ])
                assert cfg3.navigator.position == [50.0, 75.0]
                assert cfg3.navigator.speed == 2.5


@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
class TestFactoryMethodPatternsWithDictConfig:
    """Test factory method patterns that accept DictConfig parameters."""
    
    def setup_method(self):
        """Set up test configuration for each test method."""
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
    
    def teardown_method(self):
        """Clean up after each test method."""
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
    
    def test_single_agent_controller_from_dict_config(self) -> None:
        """Test creating SingleAgentController from DictConfig through factory pattern."""
        config_dict = {
            "position": [20.0, 30.0],
            "orientation": 135.0,
            "speed": 2.0,
            "max_speed": 4.0,
            "angular_velocity": 0.15
        }
        
        dict_config = OmegaConf.create(config_dict)
        controller = SingleAgentController(config=dict_config)
        
        # Verify controller was created with correct parameters
        assert controller.num_agents == 1
        assert np.allclose(controller.positions[0], [20.0, 30.0])
        assert controller.orientations[0] == 135.0
        assert controller.speeds[0] == 2.0
        assert controller.max_speeds[0] == 4.0
        assert controller.angular_velocities[0] == 0.15
    
    def test_multi_agent_controller_from_dict_config(self) -> None:
        """Test creating MultiAgentController from DictConfig through factory pattern."""
        config_dict = {
            "positions": [[10.0, 15.0], [20.0, 25.0], [30.0, 35.0]],
            "orientations": [0.0, 120.0, 240.0],
            "speeds": [1.0, 1.5, 2.0],
            "max_speeds": [3.0, 4.0, 5.0],
            "angular_velocities": [0.1, 0.15, 0.2]
        }
        
        dict_config = OmegaConf.create(config_dict)
        controller = MultiAgentController(config=dict_config)
        
        # Verify controller was created with correct parameters
        assert controller.num_agents == 3
        assert controller.positions.shape == (3, 2)
        assert np.allclose(controller.positions[0], [10.0, 15.0])
        assert np.allclose(controller.positions[1], [20.0, 25.0])
        assert np.allclose(controller.positions[2], [30.0, 35.0])
        assert np.allclose(controller.orientations, [0.0, 120.0, 240.0])
    
    def test_factory_method_with_hydra_composition(self) -> None:
        """Test factory methods work with full Hydra configuration composition."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = os.path.join(temp_dir, "conf")
            os.makedirs(config_dir)
            
            # Create single agent configuration
            single_config = """
defaults:
  - _self_

type: "single"
position: [40.0, 60.0]
orientation: 270.0
speed: 1.8
max_speed: 6.0
angular_velocity: 0.12
"""
            
            # Create multi-agent configuration
            multi_config = """
defaults:
  - _self_

type: "multi"
positions: 
  - [10.0, 20.0]
  - [30.0, 40.0]
orientations: [45.0, 225.0]
speeds: [1.2, 1.8]
max_speeds: [3.5, 4.5]
angular_velocities: [0.08, 0.16]
"""
            
            with open(os.path.join(config_dir, "single_navigator.yaml"), "w") as f:
                f.write(single_config)
            
            with open(os.path.join(config_dir, "multi_navigator.yaml"), "w") as f:
                f.write(multi_config)
            
            with initialize(config_path=config_dir, version_base=None):
                # Test single agent factory method
                single_cfg = compose(config_name="single_navigator")
                single_controller = SingleAgentController(config=single_cfg)
                
                assert single_controller.num_agents == 1
                assert np.allclose(single_controller.positions[0], [40.0, 60.0])
                assert single_controller.orientations[0] == 270.0
                assert single_controller.speeds[0] == 1.8
                
                # Test multi-agent factory method
                multi_cfg = compose(config_name="multi_navigator")
                multi_controller = MultiAgentController(config=multi_cfg)
                
                assert multi_controller.num_agents == 2
                assert np.allclose(multi_controller.positions[0], [10.0, 20.0])
                assert np.allclose(multi_controller.positions[1], [30.0, 40.0])
                assert np.allclose(multi_controller.orientations, [45.0, 225.0])
    
    def test_factory_method_parameter_override_precedence(self) -> None:
        """Test that direct parameters override DictConfig parameters in factory methods."""
        config_dict = {
            "position": [50.0, 60.0],
            "orientation": 180.0,
            "speed": 2.5,
            "max_speed": 5.5,
            "angular_velocity": 0.18
        }
        
        dict_config = OmegaConf.create(config_dict)
        
        # Create controller with config and direct parameter overrides
        controller = SingleAgentController(
            config=dict_config,
            position=(100.0, 200.0),  # Should override config
            speed=3.5,  # Should override config
            # orientation, max_speed, angular_velocity should come from config
        )
        
        # Verify override precedence
        assert np.allclose(controller.positions[0], [100.0, 200.0])  # Direct override
        assert controller.speeds[0] == 3.5  # Direct override
        assert controller.orientations[0] == 180.0  # From config
        assert controller.max_speeds[0] == 5.5  # From config
        assert controller.angular_velocities[0] == 0.18  # From config
    
    def test_error_handling_in_factory_methods(self) -> None:
        """Test error handling when factory methods receive invalid DictConfig."""
        # Test with invalid configuration - speed exceeds max_speed
        invalid_config_dict = {
            "position": [25.0, 35.0],
            "orientation": 45.0,
            "speed": 10.0,  # Invalid: exceeds max_speed
            "max_speed": 5.0,
            "angular_velocity": 0.1
        }
        
        dict_config = OmegaConf.create(invalid_config_dict)
        
        # Should handle validation error gracefully
        with pytest.raises((ValueError, RuntimeError)):
            SingleAgentController(config=dict_config)
        
        # Test with missing required parameters for multi-agent
        incomplete_multi_config = {
            "positions": [[10.0, 20.0], [30.0, 40.0]],
            "orientations": [0.0]  # Missing orientation for second agent
        }
        
        dict_config_multi = OmegaConf.create(incomplete_multi_config)
        
        with pytest.raises((ValueError, RuntimeError)):
            MultiAgentController(config=dict_config_multi)


class TestConfigurationDrivenBehavior:
    """Test configuration-driven behavior across different parameter combinations."""
    
    def test_single_agent_behavior_variations(self) -> None:
        """Test single agent behavior with different configuration parameters."""
        # Test slow agent configuration
        slow_config = {
            "position": [25.0, 25.0],
            "orientation": 0.0,
            "speed": 0.5,
            "max_speed": 1.0,
            "angular_velocity": 0.05
        }
        
        slow_controller = SingleAgentController(**slow_config)
        
        env_array = np.zeros((100, 100))
        initial_position = slow_controller.positions[0].copy()
        
        # Take a step
        slow_controller.step(env_array)
        
        # Verify slow movement
        distance_moved = np.linalg.norm(slow_controller.positions[0] - initial_position)
        assert distance_moved <= 1.0  # Should move slowly
        
        # Test fast agent configuration
        fast_config = {
            "position": [25.0, 25.0],
            "orientation": 0.0,
            "speed": 5.0,
            "max_speed": 10.0,
            "angular_velocity": 0.5
        }
        
        fast_controller = SingleAgentController(**fast_config)
        initial_position = fast_controller.positions[0].copy()
        
        # Take a step
        fast_controller.step(env_array)
        
        # Verify fast movement
        distance_moved = np.linalg.norm(fast_controller.positions[0] - initial_position)
        assert distance_moved >= 3.0  # Should move quickly
    
    def test_multi_agent_coordination_scenarios(self) -> None:
        """Test multi-agent behavior with different coordination scenarios."""
        # Test formation flying configuration
        formation_positions = np.array([
            [40.0, 50.0],  # Leader
            [35.0, 45.0],  # Left wing
            [45.0, 45.0]   # Right wing
        ])
        
        formation_orientations = np.array([0.0, 0.0, 0.0])  # All facing same direction
        formation_speeds = np.array([2.0, 2.0, 2.0])  # Same speed
        
        formation_controller = MultiAgentController(
            positions=formation_positions,
            orientations=formation_orientations,
            speeds=formation_speeds
        )
        
        env_array = np.zeros((100, 100))
        initial_positions = formation_controller.positions.copy()
        
        # Take several steps
        for _ in range(5):
            formation_controller.step(env_array)
        
        # Verify formation is maintained (relative positions similar)
        position_changes = formation_controller.positions - initial_positions
        
        # All agents should move in similar direction (x-axis)
        assert np.all(position_changes[:, 0] > 0)  # All moving positive x
        assert np.std(position_changes[:, 0]) < 1.0  # Similar x movement
        assert np.std(position_changes[:, 1]) < 0.5  # Minimal y deviation
    
    def test_boundary_condition_handling(self) -> None:
        """Test agent behavior at environment boundaries."""
        # Test agent near boundary
        boundary_controller = SingleAgentController(
            position=(95.0, 95.0),  # Near edge of 100x100 environment
            orientation=45.0,  # Moving toward boundary
            speed=2.0,
            max_speed=5.0
        )
        
        env_array = np.zeros((100, 100))
        
        # Take multiple steps
        for _ in range(10):
            boundary_controller.step(env_array)
            
            # Verify agent stays within reasonable bounds
            x, y = boundary_controller.positions[0]
            assert -10 <= x <= 110  # Allow some buffer outside environment
            assert -10 <= y <= 110
    
    def test_sensor_configuration_effects(self) -> None:
        """Test how different sensor configurations affect odor sampling."""
        controller = SingleAgentController(position=(50, 50), orientation=0.0)
        
        # Create environment with gradient
        env_array = np.zeros((100, 100))
        y, x = np.ogrid[:100, :100]
        env_array += np.exp(-((x - 60)**2 + (y - 50)**2) / 100)  # Peak at (60, 50)
        
        # Test different sensor configurations
        configs = [
            {"num_sensors": 2, "sensor_distance": 5.0, "sensor_angle": 45.0},
            {"num_sensors": 4, "sensor_distance": 10.0, "sensor_angle": 90.0},
            {"num_sensors": 8, "sensor_distance": 3.0, "sensor_angle": 22.5}
        ]
        
        for config in configs:
            odor_values = controller.sample_multiple_sensors(env_array, **config)
            
            # Verify correct shape
            assert odor_values.shape == (config["num_sensors"],)
            
            # Verify all values are reasonable
            assert np.all(odor_values >= 0.0)
            assert np.all(odor_values <= 1.0)
            
            # With agent at (50,50) and peak at (60,50), right sensors should read higher
            # This is a basic sanity check for sensor positioning
            if config["num_sensors"] >= 2:
                assert np.any(odor_values > 0.1)  # Should detect some odor


if __name__ == "__main__":
    pytest.main([__file__])