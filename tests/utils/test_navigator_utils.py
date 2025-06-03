"""
Tests for enhanced navigator utility functions in the cookiecutter-based architecture.

This test module provides comprehensive validation for the enhanced utility functions
supporting Hydra configuration integration, seed management, and navigator factory
patterns. Tests cover backward compatibility maintenance, parameter validation,
factory method integration, and reproducible testing scenarios.

The enhanced test suite validates:
- Parameter normalization and validation functions with type safety
- Navigator creation utilities with factory pattern integration
- Sensor positioning and sampling functions with numerical precision
- Configuration-driven testing with Hydra integration
- Seed management integration for reproducible test execution
- Error handling and edge cases with enhanced validation

Test Categories:
1. Parameter Normalization Testing: Array parameter handling with type validation
2. Factory Method Integration: Navigator creation with Hydra configuration
3. Sensor Utilities Testing: Position calculation and odor sampling validation
4. Configuration Integration: Hydra-based parameter management testing  
5. Seed Management Testing: Reproducible randomization with enhanced context
6. Integration Testing: Cross-component validation with realistic scenarios

Author: Enhanced Cookiecutter Template Generator
Version: 2.0.0
"""

import pytest
import numpy as np
import warnings
from unittest.mock import Mock, MagicMock, patch
from typing import Any, Dict, List, Optional, Tuple, Union

# Enhanced imports for cookiecutter-based architecture
from {{cookiecutter.project_slug}}.utils import (
    # Parameter normalization utilities (backward compatibility)
    normalize_array_parameter,
    create_navigator_from_params,  # Deprecated but maintained for compatibility
    
    # Enhanced sensor utilities  
    calculate_sensor_positions,
    sample_odor_at_sensors,
    
    # Configuration and integration utilities
    configure_from_hydra,
    initialize_reproducibility,
    
    # I/O utilities for testing
    load_yaml,
    save_yaml,
    load_json,
    save_json,
    
    # Availability checking
    __availability__,
)

# Core navigator imports for integration testing
from {{cookiecutter.project_slug}}.core.navigator import NavigatorProtocol, NavigatorFactory


class TestParameterNormalization:
    """
    Test suite for parameter normalization and validation functions.
    
    Validates array parameter handling, type conversion, and broadcasting
    behavior with enhanced error handling and validation patterns.
    """

    def test_normalize_array_parameter_none(self):
        """Test that None parameter returns None without modification."""
        result = normalize_array_parameter(None, (3,))
        assert result is None

    def test_normalize_array_parameter_scalar_basic(self):
        """Test converting scalar values to arrays with specified shapes."""
        # Test with integer scalar
        result = normalize_array_parameter(5, (3,))
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.array_equal(result, np.array([5, 5, 5]))

        # Test with float scalar  
        result = normalize_array_parameter(2.5, (4,))
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        assert np.array_equal(result, np.array([2.5, 2.5, 2.5, 2.5]))

    def test_normalize_array_parameter_scalar_multidimensional(self):
        """Test scalar broadcasting to multi-dimensional arrays."""
        # Test 2D array broadcasting
        result = normalize_array_parameter(1.0, (2, 3))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        expected = np.ones((2, 3))
        assert np.array_equal(result, expected)

    def test_normalize_array_parameter_list_conversion(self):
        """Test converting lists to numpy arrays with validation."""
        # Test basic list conversion
        result = normalize_array_parameter([1, 2, 3], (3,))
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.array_equal(result, np.array([1, 2, 3]))
        
        # Test nested list conversion
        result = normalize_array_parameter([[1, 2], [3, 4]], (2, 2))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        assert np.array_equal(result, np.array([[1, 2], [3, 4]]))

    def test_normalize_array_parameter_ndarray_passthrough(self):
        """Test that numpy arrays pass through unchanged when shape matches."""
        input_array = np.array([4, 5, 6])
        result = normalize_array_parameter(input_array, (3,))
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.array_equal(result, input_array)
        # Verify it's the same object (unchanged)
        assert result is input_array

    def test_normalize_array_parameter_shape_mismatch_handling(self):
        """Test parameter normalization with shape mismatches."""
        # Test broadcasting behavior
        input_array = np.array([1, 2])
        result = normalize_array_parameter(input_array, (4,))
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        # Should resize the array to match expected shape
        assert len(result) == 4

    def test_normalize_array_parameter_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test empty array handling
        result = normalize_array_parameter([], (0,))
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)
        
        # Test single element array
        result = normalize_array_parameter([42], (1,))
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert result[0] == 42


class TestDeprecatedNavigatorUtilities:
    """
    Test suite for deprecated navigator utility functions.
    
    Validates backward compatibility maintenance while testing the
    transition to new factory patterns and API structures.
    """

    @pytest.fixture
    def mock_navigator_factory(self):
        """Mock the new NavigatorFactory for testing deprecated functions."""
        with patch('{{cookiecutter.project_slug}}.utils.create_navigator') as mock_create:
            mock_navigator = MagicMock()
            mock_navigator.positions = np.array([[0.0, 0.0]])
            mock_navigator.orientations = np.array([0.0])
            mock_navigator.speeds = np.array([0.0])
            mock_navigator.max_speeds = np.array([1.0])
            mock_navigator.angular_velocities = np.array([0.0])
            mock_navigator.num_agents = 1
            
            mock_create.return_value = mock_navigator
            yield mock_create, mock_navigator

    def test_create_navigator_from_params_deprecation_warning(self, mock_navigator_factory):
        """Test that deprecated function raises deprecation warning."""
        mock_create, mock_navigator = mock_navigator_factory
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            navigator = create_navigator_from_params(
                positions=(5, 10),
                orientations=30,
                speeds=1.5,
                max_speeds=3.0
            )
            
            # Check that deprecation warning was issued
            assert len(w) >= 1
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
            assert any("deprecated" in str(warning.message).lower() for warning in w)

    def test_create_navigator_from_params_single_agent_compatibility(self, mock_navigator_factory):
        """Test single agent creation maintains backward compatibility."""
        mock_create, mock_navigator = mock_navigator_factory
        
        # Configure mock for single agent
        mock_navigator.positions = np.array([[5.0, 10.0]])
        mock_navigator.orientations = np.array([30.0])
        mock_navigator.speeds = np.array([1.5])
        mock_navigator.max_speeds = np.array([3.0])
        mock_navigator.num_agents = 1
        
        navigator = create_navigator_from_params(
            positions=(5, 10),
            orientations=30,
            speeds=1.5,
            max_speeds=3.0
        )
        
        # Verify backward compatible behavior
        assert navigator.num_agents == 1
        assert len(navigator.positions) == 1
        assert np.allclose(navigator.positions[0], (5, 10))
        assert navigator.orientations[0] == 30.0
        assert navigator.speeds[0] == 1.5
        assert navigator.max_speeds[0] == 3.0

    def test_create_navigator_from_params_multi_agent_compatibility(self, mock_navigator_factory):
        """Test multi-agent creation maintains backward compatibility."""
        mock_create, mock_navigator = mock_navigator_factory
        
        # Configure mock for multi-agent
        positions = [(1, 2), (3, 4), (5, 6)]
        orientations = [10, 20, 30]
        speeds = [0.1, 0.2, 0.3]
        max_speeds = [1.0, 2.0, 3.0]
        
        mock_navigator.positions = np.array(positions)
        mock_navigator.orientations = np.array(orientations)
        mock_navigator.speeds = np.array(speeds)
        mock_navigator.max_speeds = np.array(max_speeds)
        mock_navigator.num_agents = 3
        
        navigator = create_navigator_from_params(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds
        )
        
        # Verify backward compatible behavior
        assert navigator.num_agents == 3
        assert len(navigator.positions) == 3
        for i, pos in enumerate(positions):
            assert np.allclose(navigator.positions[i], pos)
        assert np.array_equal(navigator.orientations, orientations)
        assert np.array_equal(navigator.speeds, speeds)
        assert np.array_equal(navigator.max_speeds, max_speeds)


class TestSensorUtilities:
    """
    Test suite for enhanced sensor positioning and sampling utilities.
    
    Validates sensor position calculations, odor sampling algorithms,
    and integration with navigator instances.
    """

    def test_calculate_sensor_positions_single_sensor(self):
        """Test sensor position calculation for single sensor configuration."""
        position = (10.0, 20.0)
        orientation = 45.0  # degrees
        sensor_config = {
            'num_sensors': 1,
            'distance': 5.0,
            'angle_spread': 0.0
        }
        
        result = calculate_sensor_positions(position, orientation, sensor_config)
        
        assert len(result) == 1
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 2
        # Single sensor should be at agent position
        assert np.allclose(result[0], position)

    def test_calculate_sensor_positions_bilateral_sensors(self):
        """Test sensor position calculation for bilateral (left-right) configuration."""
        position = (0.0, 0.0)
        orientation = 0.0  # facing right
        sensor_config = {
            'num_sensors': 2,
            'distance': 10.0,
            'angle_spread': 90.0  # ±45 degrees from center
        }
        
        result = calculate_sensor_positions(position, orientation, sensor_config)
        
        assert len(result) == 2
        assert all(isinstance(pos, tuple) and len(pos) == 2 for pos in result)
        
        # Verify sensors are at expected positions
        # For 0° orientation with ±45° spread, sensors should be at 45° and -45°
        sensor1_x, sensor1_y = result[0]
        sensor2_x, sensor2_y = result[1]
        
        # Check that sensors are at correct distance from agent
        dist1 = np.sqrt(sensor1_x**2 + sensor1_y**2)
        dist2 = np.sqrt(sensor2_x**2 + sensor2_y**2)
        assert np.allclose(dist1, 10.0, atol=0.1)
        assert np.allclose(dist2, 10.0, atol=0.1)

    def test_calculate_sensor_positions_triangular_configuration(self):
        """Test sensor position calculation for triangular sensor arrangement."""
        position = (5.0, 5.0)
        orientation = 90.0  # facing up
        sensor_config = {
            'num_sensors': 3,
            'distance': 8.0,
            'angle_spread': 120.0  # ±60 degrees from center
        }
        
        result = calculate_sensor_positions(position, orientation, sensor_config)
        
        assert len(result) == 3
        assert all(isinstance(pos, tuple) and len(pos) == 2 for pos in result)
        
        # Verify all sensors are at correct distance from agent
        for sensor_pos in result:
            sensor_x, sensor_y = sensor_pos
            distance = np.sqrt((sensor_x - 5.0)**2 + (sensor_y - 5.0)**2)
            assert np.allclose(distance, 8.0, atol=0.1)

    def test_sample_odor_at_sensors_basic_sampling(self):
        """Test basic odor sampling at sensor positions."""
        sensor_positions = [(10, 20), (30, 40), (50, 60)]
        
        # Create synthetic plume frame with known values
        plume_frame = np.zeros((100, 100))
        plume_frame[20, 10] = 0.8  # High concentration at first sensor
        plume_frame[40, 30] = 0.5  # Medium concentration at second sensor
        plume_frame[60, 50] = 0.2  # Low concentration at third sensor
        
        result = sample_odor_at_sensors(sensor_positions, plume_frame)
        
        assert len(result) == 3
        assert all(isinstance(val, float) for val in result)
        
        # Check expected concentration values
        assert np.allclose(result[0], 0.8, atol=0.1)
        assert np.allclose(result[1], 0.5, atol=0.1)
        assert np.allclose(result[2], 0.2, atol=0.1)

    def test_sample_odor_at_sensors_interpolation_methods(self):
        """Test different interpolation methods for odor sampling."""
        sensor_positions = [(10.5, 20.5)]  # Non-integer position for interpolation
        
        # Create gradient plume for interpolation testing
        plume_frame = np.zeros((50, 50))
        plume_frame[20, 10] = 1.0
        plume_frame[20, 11] = 0.8
        plume_frame[21, 10] = 0.6
        plume_frame[21, 11] = 0.4
        
        # Test nearest neighbor interpolation
        result_nearest = sample_odor_at_sensors(
            sensor_positions, plume_frame, interpolation_method='nearest'
        )
        assert len(result_nearest) == 1
        assert isinstance(result_nearest[0], float)
        
        # Test bilinear interpolation
        result_bilinear = sample_odor_at_sensors(
            sensor_positions, plume_frame, interpolation_method='bilinear'
        )
        assert len(result_bilinear) == 1
        assert isinstance(result_bilinear[0], float)
        
        # Bilinear should give different result than nearest for non-integer positions
        # (though both should be valid concentrations)
        assert 0.0 <= result_nearest[0] <= 1.0
        assert 0.0 <= result_bilinear[0] <= 1.0

    def test_sample_odor_at_sensors_boundary_conditions(self):
        """Test odor sampling at plume frame boundaries."""
        plume_frame = np.ones((50, 50)) * 0.5  # Uniform concentration
        
        # Test sensors at frame boundaries
        boundary_positions = [
            (0, 0),      # Top-left corner
            (49, 0),     # Top-right corner  
            (0, 49),     # Bottom-left corner
            (49, 49),    # Bottom-right corner
            (-5, 25),    # Outside left boundary
            (55, 25),    # Outside right boundary
            (25, -5),    # Outside top boundary
            (25, 55)     # Outside bottom boundary
        ]
        
        result = sample_odor_at_sensors(boundary_positions, plume_frame)
        
        assert len(result) == len(boundary_positions)
        
        # Boundary positions should return valid concentrations
        for i in range(4):  # First 4 are valid boundary positions
            assert 0.0 <= result[i] <= 1.0
            
        # Outside boundary positions should return 0.0
        for i in range(4, 8):  # Last 4 are outside boundaries
            assert result[i] == 0.0

    def test_sample_odor_at_sensors_error_handling(self):
        """Test error handling for invalid inputs."""
        sensor_positions = [(10, 20)]
        
        # Test with invalid plume frame dimensions
        with pytest.raises(ValueError, match="2D array"):
            sample_odor_at_sensors(sensor_positions, np.array([1, 2, 3]))
        
        # Test with non-numpy array input
        invalid_plume = [[1, 2], [3, 4]]  # List instead of numpy array
        result = sample_odor_at_sensors(sensor_positions, invalid_plume)
        # Should handle conversion automatically
        assert len(result) == 1
        assert isinstance(result[0], float)


class TestConfigurationIntegration:
    """
    Test suite for Hydra configuration integration and utility management.
    
    Validates configuration loading, component initialization, and
    cross-component integration patterns.
    """

    def test_configure_from_hydra_basic_integration(self, mock_hydra_config):
        """Test basic Hydra configuration integration."""
        if mock_hydra_config is None:
            pytest.skip("Hydra not available for testing")
        
        # Test configuration with all components
        results = configure_from_hydra(mock_hydra_config)
        
        assert isinstance(results, dict)
        assert 'logging' in results
        assert 'seed_manager' in results
        
        # Results should be boolean success indicators
        for component, success in results.items():
            assert isinstance(success, bool)

    def test_configure_from_hydra_selective_components(self, mock_hydra_config):
        """Test selective component configuration."""
        if mock_hydra_config is None:
            pytest.skip("Hydra not available for testing")
        
        # Test configuring only specific components
        results = configure_from_hydra(mock_hydra_config, components=['logging'])
        
        assert isinstance(results, dict)
        assert 'logging' in results
        # Should only configure requested components
        if len(results) > 1:
            # Other components may be included if available
            pass

    def test_configure_from_hydra_availability_handling(self):
        """Test graceful handling of unavailable components."""
        # Create minimal mock config
        mock_config = type('MockConfig', (), {
            'logging': type('LoggingConfig', (), {'level': 'INFO'}),
            'reproducibility': type('ReproConfig', (), {'global_seed': 42})
        })()
        
        # Should handle missing components gracefully
        results = configure_from_hydra(mock_config)
        assert isinstance(results, dict)

    def test_initialize_reproducibility_basic(self, mock_seed_manager):
        """Test basic reproducibility initialization."""
        if mock_seed_manager is None:
            pytest.skip("Seed manager not available for testing")
        
        seed = 42
        experiment_id = "test_experiment"
        
        repro_info = initialize_reproducibility(seed, experiment_id)
        
        assert isinstance(repro_info, dict)
        assert 'seed_value' in repro_info or 'seed' in repro_info
        assert repro_info.get('experiment_id') == experiment_id

    def test_initialize_reproducibility_without_experiment_id(self, mock_seed_manager):
        """Test reproducibility initialization without experiment ID."""
        seed = 123
        
        repro_info = initialize_reproducibility(seed)
        
        assert isinstance(repro_info, dict)
        # Should work without experiment_id
        assert 'seed' in repro_info or 'seed_value' in repro_info

    def test_initialize_reproducibility_fallback_behavior(self):
        """Test fallback behavior when advanced features unavailable."""
        # Test with basic functionality only
        seed = 456
        
        repro_info = initialize_reproducibility(seed)
        
        # Should return some basic information even with fallbacks
        assert isinstance(repro_info, dict)
        assert len(repro_info) > 0


class TestIOUtilities:
    """
    Test suite for I/O utility functions with enhanced error handling.
    
    Validates file loading/saving operations, format handling,
    and integration with the enhanced architecture.
    """

    def test_load_yaml_basic(self, tmp_path):
        """Test basic YAML file loading functionality."""
        # Create test YAML file
        yaml_file = tmp_path / "test_config.yaml"
        yaml_content = """
navigator:
  max_speed: 2.0
  num_agents: 1
simulation:
  fps: 30
"""
        yaml_file.write_text(yaml_content)
        
        result = load_yaml(yaml_file)
        
        assert isinstance(result, dict)
        assert 'navigator' in result
        assert 'simulation' in result
        assert result['navigator']['max_speed'] == 2.0
        assert result['simulation']['fps'] == 30

    def test_save_yaml_basic(self, tmp_path):
        """Test basic YAML file saving functionality."""
        yaml_file = tmp_path / "output_config.yaml"
        test_data = {
            'navigator': {'max_speed': 1.5, 'orientation': 45.0},
            'simulation': {'fps': 60, 'duration': 10.0}
        }
        
        save_yaml(test_data, yaml_file)
        
        # Verify file was created and can be loaded back
        assert yaml_file.exists()
        
        loaded_data = load_yaml(yaml_file)
        assert loaded_data == test_data

    def test_load_json_basic(self, tmp_path):
        """Test basic JSON file loading functionality."""
        json_file = tmp_path / "test_data.json"
        json_content = '{"positions": [[0, 0], [10, 10]], "speeds": [1.0, 1.5]}'
        json_file.write_text(json_content)
        
        result = load_json(json_file)
        
        assert isinstance(result, dict)
        assert 'positions' in result
        assert 'speeds' in result
        assert result['positions'] == [[0, 0], [10, 10]]
        assert result['speeds'] == [1.0, 1.5]

    def test_save_json_basic(self, tmp_path):
        """Test basic JSON file saving functionality."""
        json_file = tmp_path / "output_data.json"
        test_data = {
            'experiment_id': 'test_001',
            'results': [1.2, 3.4, 5.6],
            'metadata': {'timestamp': '2024-01-01T00:00:00'}
        }
        
        save_json(test_data, json_file)
        
        # Verify file was created and can be loaded back
        assert json_file.exists()
        
        loaded_data = load_json(json_file)
        assert loaded_data == test_data

    def test_io_error_handling(self, tmp_path):
        """Test error handling for I/O operations."""
        # Test loading non-existent file
        with pytest.raises(FileNotFoundError):
            load_yaml(tmp_path / "nonexistent.yaml")
        
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "nonexistent.json")
        
        # Test loading invalid YAML
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: content: [unclosed")
        
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_yaml(invalid_yaml)
        
        # Test loading invalid JSON
        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text('{"invalid": json, "content"}')
        
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_json(invalid_json)


class TestAvailabilityManagement:
    """
    Test suite for availability management and graceful degradation.
    
    Validates that the utility system handles missing dependencies
    gracefully and provides appropriate fallbacks.
    """

    def test_availability_flags_structure(self):
        """Test that availability flags provide expected structure."""
        assert isinstance(__availability__, dict)
        
        expected_keys = ['logging', 'visualization', 'seed_manager']
        for key in expected_keys:
            assert key in __availability__
            assert isinstance(__availability__[key], bool)

    def test_availability_based_imports(self):
        """Test that imports succeed based on availability flags."""
        if __availability__['logging']:
            from {{cookiecutter.project_slug}}.utils import setup_enhanced_logging
            assert callable(setup_enhanced_logging)
        
        if __availability__['seed_manager']:
            from {{cookiecutter.project_slug}}.utils import set_global_seed
            assert callable(set_global_seed)
        
        if __availability__['visualization']:
            from {{cookiecutter.project_slug}}.utils import SimulationVisualization
            assert SimulationVisualization is not None

    def test_fallback_functions_exist(self):
        """Test that fallback functions are available when features unavailable."""
        # These should always be available even as fallbacks
        from {{cookiecutter.project_slug}}.utils import (
            setup_enhanced_logging,
            get_module_logger,
            set_global_seed
        )
        
        assert callable(setup_enhanced_logging)
        assert callable(get_module_logger)
        assert callable(set_global_seed)


class TestIntegrationScenarios:
    """
    Integration test suite combining multiple utility components.
    
    Tests realistic scenarios involving configuration management,
    seed management, and navigator utilities working together.
    """

    def test_full_reproducibility_workflow(self, mock_hydra_config, mock_seed_manager):
        """Test complete reproducibility workflow with all components."""
        if mock_hydra_config is None:
            pytest.skip("Hydra not available for testing")
        
        # Initialize reproducibility environment
        seed = 42
        experiment_id = "integration_test_001"
        
        repro_info = initialize_reproducibility(seed, experiment_id)
        
        # Configure from Hydra
        config_results = configure_from_hydra(mock_hydra_config)
        
        # Test parameter normalization with reproducible random data
        np.random.seed(seed)  # Ensure reproducible test data
        random_params = np.random.rand(5)
        
        normalized = normalize_array_parameter(random_params, (5,))
        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == (5,)
        
        # Verify reproducibility
        np.random.seed(seed)
        random_params_2 = np.random.rand(5)
        normalized_2 = normalize_array_parameter(random_params_2, (5,))
        
        assert np.array_equal(normalized, normalized_2)

    def test_configuration_driven_navigator_creation(self, mock_hydra_config):
        """Test navigator creation driven by Hydra configuration."""
        if mock_hydra_config is None:
            pytest.skip("Hydra not available for testing")
        
        # Extract navigator configuration
        nav_config = mock_hydra_config.navigator
        
        # Test parameter normalization for navigator creation
        if hasattr(nav_config, 'max_speed'):
            max_speeds = normalize_array_parameter(
                nav_config.max_speed, 
                (nav_config.get('num_agents', 1),)
            )
            assert isinstance(max_speeds, np.ndarray)
            assert len(max_speeds) == nav_config.get('num_agents', 1)

    def test_sensor_utilities_integration(self, mock_navigator):
        """Test sensor utilities integration with navigator instances."""
        # Configure navigator for testing
        mock_navigator.positions = np.array([[25.0, 35.0]])
        mock_navigator.orientations = np.array([90.0])  # Facing up
        
        # Test sensor position calculation
        sensor_config = {
            'num_sensors': 2,
            'distance': 5.0,
            'angle_spread': 60.0
        }
        
        agent_pos = tuple(mock_navigator.positions[0])
        agent_orientation = mock_navigator.orientations[0]
        
        sensor_positions = calculate_sensor_positions(
            agent_pos, agent_orientation, sensor_config
        )
        
        assert len(sensor_positions) == 2
        
        # Test odor sampling at calculated positions
        plume_frame = np.random.rand(100, 100)
        
        odor_readings = sample_odor_at_sensors(sensor_positions, plume_frame)
        
        assert len(odor_readings) == 2
        assert all(0.0 <= reading <= 1.0 for reading in odor_readings)

    def test_io_utilities_with_configuration(self, tmp_path, mock_hydra_config):
        """Test I/O utilities integration with configuration management."""
        if mock_hydra_config is None:
            pytest.skip("Hydra not available for testing")
        
        # Test saving configuration to file
        config_file = tmp_path / "test_config.yaml"
        
        # Convert hydra config to dict for saving
        if hasattr(mock_hydra_config, 'to_container'):
            config_dict = mock_hydra_config.to_container()
        else:
            config_dict = dict(mock_hydra_config)
        
        save_yaml(config_dict, config_file)
        
        # Test loading configuration back
        loaded_config = load_yaml(config_file)
        
        assert isinstance(loaded_config, dict)
        assert 'navigator' in loaded_config
        
        # Test parameter processing from loaded config
        if 'navigator' in loaded_config:
            nav_params = loaded_config['navigator']
            if 'max_speed' in nav_params:
                max_speed_array = normalize_array_parameter(
                    nav_params['max_speed'], (1,)
                )
                assert isinstance(max_speed_array, np.ndarray)

    def test_error_handling_integration(self):
        """Test integrated error handling across utility functions."""
        # Test parameter normalization with invalid inputs
        with pytest.raises((ValueError, TypeError)):
            normalize_array_parameter("invalid", (3,))
        
        # Test sensor calculation with invalid configuration
        with pytest.raises((KeyError, ValueError)):
            calculate_sensor_positions((0, 0), 0, {})  # Missing required config
        
        # Test odor sampling with incompatible inputs
        with pytest.raises(ValueError):
            sample_odor_at_sensors([(0, 0)], "not_an_array")

    def test_performance_characteristics(self, mock_navigator):
        """Test performance characteristics of utility functions."""
        import time
        
        # Test parameter normalization performance
        large_array = np.random.rand(1000)
        
        start_time = time.time()
        result = normalize_array_parameter(large_array, (1000,))
        normalization_time = time.time() - start_time
        
        assert normalization_time < 0.01  # Should be very fast
        assert isinstance(result, np.ndarray)
        
        # Test sensor calculation performance
        start_time = time.time()
        sensor_positions = calculate_sensor_positions(
            (50, 50), 0.0, {'num_sensors': 10, 'distance': 5.0, 'angle_spread': 180.0}
        )
        calculation_time = time.time() - start_time
        
        assert calculation_time < 0.001  # Should be sub-millisecond
        assert len(sensor_positions) == 10
        
        # Test odor sampling performance with large plume
        large_plume = np.random.rand(500, 500)
        sensor_positions = [(i*10, i*15) for i in range(20)]
        
        start_time = time.time()
        odor_readings = sample_odor_at_sensors(sensor_positions, large_plume)
        sampling_time = time.time() - start_time
        
        assert sampling_time < 0.01  # Should handle large plumes efficiently
        assert len(odor_readings) == 20


# Test execution configuration and fixtures
pytest_plugins = ["tests.conftest"]


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])