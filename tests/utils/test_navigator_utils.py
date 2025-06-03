"""Tests for navigator utility functions and utils consolidation scenarios.

This test module validates the utility functions that support navigator creation,
parameter normalization, sensor positioning, and odor sampling. It includes
comprehensive testing for the enhanced utils module organization with Hydra
configuration integration, seed management, and factory method validation.

The tests cover:
- Parameter normalization utilities for multi-agent scenarios
- Navigator factory functions with configuration validation
- Sensor positioning and odor sampling utilities
- Integration with the seed manager for reproducible testing
- Hydra configuration integration patterns
- Error handling and edge cases for all utility functions
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import tempfile
import yaml

# Updated imports to new package structure
from {{cookiecutter.project_slug}}.utils import (
    # Parameter normalization utilities
    normalize_array_parameter,
    create_navigator_from_params,
    
    # Sensor and odor sampling utilities 
    calculate_sensor_positions,
    sample_odor_at_sensors,
    
    # Seed management utilities
    set_global_seed,
    get_current_seed,
    get_numpy_generator,
    
    # I/O utilities for configuration testing
    load_yaml,
    save_yaml,
    load_json,
    save_json,
    
    # Comprehensive utility information
    get_module_info,
    initialize_reproducibility,
)

# Updated imports from core module
from {{cookiecutter.project_slug}}.core.navigator import NavigatorProtocol


class TestNormalizeArrayParameter:
    """Test suite for parameter normalization utility function."""
    
    def test_normalize_array_parameter_none(self):
        """Test that None parameter returns None."""
        result = normalize_array_parameter(None, 3)
        assert result is None

    def test_normalize_array_parameter_scalar_integer(self):
        """Test converting a scalar integer value to array of desired length."""
        result = normalize_array_parameter(5, 3)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.array_equal(result, np.array([5.0, 5.0, 5.0]))
        assert result.dtype == float  # Ensure dtype is float

    def test_normalize_array_parameter_scalar_float(self):
        """Test converting a scalar float value to array of desired length."""
        result = normalize_array_parameter(2.5, 4)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        assert np.array_equal(result, np.array([2.5, 2.5, 2.5, 2.5]))
        assert result.dtype == float

    def test_normalize_array_parameter_list(self):
        """Test converting a list to a numpy array."""
        result = normalize_array_parameter([1, 2, 3], 3)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))
        assert result.dtype == float

    def test_normalize_array_parameter_ndarray(self):
        """Test that a numpy array is returned unchanged."""
        input_array = np.array([4.0, 5.0, 6.0])
        result = normalize_array_parameter(input_array, 3)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.array_equal(result, input_array)
        assert result.dtype == float

    def test_normalize_array_parameter_single_element_array(self):
        """Test single-element array broadcasting to target length."""
        input_array = np.array([7.5])
        result = normalize_array_parameter(input_array, 5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)
        assert np.array_equal(result, np.array([7.5, 7.5, 7.5, 7.5, 7.5]))

    def test_normalize_array_parameter_length_mismatch_error(self):
        """Test error when parameter array length doesn't match target."""
        with pytest.raises(ValueError, match="Parameter array length 2 does not match target length 3"):
            normalize_array_parameter([1, 2], 3)

    def test_normalize_array_parameter_empty_array(self):
        """Test behavior with empty arrays."""
        with pytest.raises(ValueError):
            normalize_array_parameter([], 3)

    def test_normalize_array_parameter_zero_target_length(self):
        """Test behavior with zero target length."""
        result = normalize_array_parameter(5.0, 0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)
        assert result.dtype == float


class TestCreateNavigatorFromParams:
    """Test suite for navigator factory function with parameter validation."""
    
    @pytest.fixture
    def mock_navigator_config(self):
        """Mock configuration for navigator creation testing."""
        with patch('{{cookiecutter.project_slug}}.utils.NavigatorConfig') as mock_config:
            # Configure the mock to return a valid config instance
            mock_instance = Mock()
            mock_config.return_value = mock_instance
            yield mock_config, mock_instance
    
    @pytest.fixture
    def mock_navigator_class(self):
        """Mock Navigator class for factory testing."""
        with patch('{{cookiecutter.project_slug}}.utils.Navigator') as mock_navigator:
            # Configure mock methods
            mock_single_instance = Mock()
            mock_multi_instance = Mock()
            
            mock_navigator.single.return_value = mock_single_instance
            mock_navigator.multi.return_value = mock_multi_instance
            
            # Configure basic properties for testing
            mock_single_instance.positions = np.array([[5.0, 10.0]])
            mock_single_instance.orientations = np.array([30.0])
            mock_single_instance.speeds = np.array([1.5])
            mock_single_instance.max_speeds = np.array([3.0])
            mock_single_instance.num_agents = 1
            
            mock_multi_instance.positions = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            mock_multi_instance.orientations = np.array([10.0, 20.0, 30.0])
            mock_multi_instance.speeds = np.array([0.1, 0.2, 0.3])
            mock_multi_instance.max_speeds = np.array([1.0, 2.0, 3.0])
            mock_multi_instance.num_agents = 3
            
            yield mock_navigator, mock_single_instance, mock_multi_instance

    def test_create_navigator_from_params_single_agent_defaults(self, mock_navigator_config, mock_navigator_class):
        """Test creating a single agent navigator with default parameters."""
        mock_config_class, mock_config_instance = mock_navigator_config
        mock_navigator, mock_single, mock_multi = mock_navigator_class
        
        navigator = create_navigator_from_params()
        
        # Verify configuration was created with defaults
        mock_config_class.assert_called_once()
        call_args = mock_config_class.call_args[1]
        assert call_args["type"] == "single"
        assert call_args["initial_position"] == [50.0, 50.0]
        assert call_args["initial_orientation"] == 0.0
        assert call_args["max_speed"] == 10.0
        assert call_args["angular_velocity"] == 0.1
        
        # Verify single navigator was created
        mock_navigator.single.assert_called_once_with(mock_config_instance)
        assert navigator == mock_single

    def test_create_navigator_from_params_single_agent_custom(self, mock_navigator_config, mock_navigator_class):
        """Test creating a single agent navigator with custom parameters."""
        mock_config_class, mock_config_instance = mock_navigator_config
        mock_navigator, mock_single, mock_multi = mock_navigator_class
        
        navigator = create_navigator_from_params(
            navigator_type="single",
            initial_position=[5.0, 10.0],
            initial_orientation=30.0,
            max_speed=3.0,
            angular_velocity=0.2,
            custom_param="test_value"
        )
        
        # Verify configuration was created with custom parameters
        mock_config_class.assert_called_once()
        call_args = mock_config_class.call_args[1]
        assert call_args["type"] == "single"
        assert call_args["initial_position"] == [5.0, 10.0]
        assert call_args["initial_orientation"] == 30.0
        assert call_args["max_speed"] == 3.0
        assert call_args["angular_velocity"] == 0.2
        assert call_args["custom_param"] == "test_value"
        
        # Verify single navigator was created
        mock_navigator.single.assert_called_once_with(mock_config_instance)
        assert navigator == mock_single

    def test_create_navigator_from_params_multi_agent(self, mock_navigator_config, mock_navigator_class):
        """Test creating a multi-agent navigator."""
        mock_config_class, mock_config_instance = mock_navigator_config
        mock_navigator, mock_single, mock_multi = mock_navigator_class
        
        navigator = create_navigator_from_params(
            navigator_type="multi",
            initial_position=[[1.0, 2.0], [3.0, 4.0]],
            initial_orientation=45.0,
            max_speed=5.0
        )
        
        # Verify configuration was created for multi-agent
        mock_config_class.assert_called_once()
        call_args = mock_config_class.call_args[1]
        assert call_args["type"] == "multi"
        assert call_args["initial_position"] == [[1.0, 2.0], [3.0, 4.0]]
        
        # Verify multi navigator was created
        mock_navigator.multi.assert_called_once_with(mock_config_instance)
        assert navigator == mock_multi

    def test_create_navigator_from_params_invalid_type(self, mock_navigator_config, mock_navigator_class):
        """Test error handling for invalid navigator type."""
        mock_config_class, mock_config_instance = mock_navigator_config
        
        with pytest.raises(ValueError, match="Unsupported navigator type: invalid"):
            create_navigator_from_params(navigator_type="invalid")

    def test_create_navigator_from_params_import_error(self):
        """Test handling of import errors."""
        with patch('{{cookiecutter.project_slug}}.utils.Navigator', side_effect=ImportError("Module not found")):
            with pytest.raises(ImportError, match="Navigator modules unavailable"):
                create_navigator_from_params()

    def test_create_navigator_from_params_config_validation_error(self, mock_navigator_class):
        """Test handling of configuration validation errors."""
        mock_navigator, mock_single, mock_multi = mock_navigator_class
        
        with patch('{{cookiecutter.project_slug}}.utils.NavigatorConfig', side_effect=ValueError("Invalid config")):
            with pytest.raises(ValueError, match="Invalid config"):
                create_navigator_from_params(max_speed=-1.0)  # Invalid negative speed


class TestSensorPositionCalculation:
    """Test suite for sensor position calculation utilities."""
    
    def test_calculate_sensor_positions_single_sensor(self):
        """Test single sensor configuration."""
        agent_pos = np.array([10.0, 20.0])
        agent_orientation = 0.0
        
        sensor_positions = calculate_sensor_positions(
            agent_pos, agent_orientation, "SINGLE"
        )
        
        assert sensor_positions.shape == (1, 2)
        np.testing.assert_array_almost_equal(sensor_positions[0], agent_pos)

    def test_calculate_sensor_positions_left_right(self):
        """Test left-right sensor configuration."""
        agent_pos = np.array([10.0, 20.0])
        agent_orientation = 0.0  # Facing east
        
        sensor_positions = calculate_sensor_positions(
            agent_pos, agent_orientation, "LEFT_RIGHT"
        )
        
        assert sensor_positions.shape == (2, 2)
        # Left sensor should be at (9, 20), right sensor at (11, 20)
        np.testing.assert_array_almost_equal(sensor_positions[0], [9.0, 20.0])
        np.testing.assert_array_almost_equal(sensor_positions[1], [11.0, 20.0])

    def test_calculate_sensor_positions_triangle(self):
        """Test triangular sensor configuration."""
        agent_pos = np.array([0.0, 0.0])
        agent_orientation = 0.0
        
        sensor_positions = calculate_sensor_positions(
            agent_pos, agent_orientation, "TRIANGLE"
        )
        
        assert sensor_positions.shape == (3, 2)
        # Front sensor at (0, 1), left-back at (-0.866, -0.5), right-back at (0.866, -0.5)
        np.testing.assert_array_almost_equal(sensor_positions[0], [0.0, 1.0], decimal=3)
        np.testing.assert_array_almost_equal(sensor_positions[1], [-0.866, -0.5], decimal=3)
        np.testing.assert_array_almost_equal(sensor_positions[2], [0.866, -0.5], decimal=3)

    def test_calculate_sensor_positions_cross(self):
        """Test cross sensor configuration."""
        agent_pos = np.array([5.0, 5.0])
        agent_orientation = 90.0  # Facing north
        
        sensor_positions = calculate_sensor_positions(
            agent_pos, agent_orientation, "CROSS"
        )
        
        assert sensor_positions.shape == (4, 2)
        # After 90° rotation: front=(4,5), right=(5,6), back=(6,5), left=(5,4)
        expected_positions = np.array([
            [4.0, 5.0],  # Front sensor (rotated)
            [5.0, 6.0],  # Right sensor (rotated)
            [6.0, 5.0],  # Back sensor (rotated)
            [5.0, 4.0]   # Left sensor (rotated)
        ])
        np.testing.assert_array_almost_equal(sensor_positions, expected_positions, decimal=6)

    def test_calculate_sensor_positions_custom_dict(self):
        """Test custom sensor configuration via dictionary."""
        agent_pos = np.array([0.0, 0.0])
        agent_orientation = 0.0
        
        custom_config = {
            "positions": [[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]]
        }
        
        sensor_positions = calculate_sensor_positions(
            agent_pos, agent_orientation, custom_config
        )
        
        assert sensor_positions.shape == (3, 2)
        expected_positions = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]])
        np.testing.assert_array_almost_equal(sensor_positions, expected_positions)

    def test_calculate_sensor_positions_custom_list(self):
        """Test custom sensor configuration via list."""
        agent_pos = np.array([1.0, 1.0])
        agent_orientation = 45.0  # 45-degree rotation
        
        custom_positions = [[1.0, 0.0], [0.0, 1.0]]
        
        sensor_positions = calculate_sensor_positions(
            agent_pos, agent_orientation, custom_positions
        )
        
        assert sensor_positions.shape == (2, 2)
        # After 45° rotation and translation
        sqrt2_half = np.sqrt(2) / 2
        expected_positions = np.array([
            [1.0 + sqrt2_half, 1.0 + sqrt2_half],  # Rotated (1,0) + agent_pos
            [1.0 - sqrt2_half, 1.0 + sqrt2_half]   # Rotated (0,1) + agent_pos
        ])
        np.testing.assert_array_almost_equal(sensor_positions, expected_positions, decimal=6)

    def test_calculate_sensor_positions_unknown_config(self):
        """Test error handling for unknown configuration."""
        agent_pos = np.array([0.0, 0.0])
        agent_orientation = 0.0
        
        with pytest.raises(ValueError, match="Unknown sensor configuration: INVALID"):
            calculate_sensor_positions(agent_pos, agent_orientation, "INVALID")


class TestOdorSampling:
    """Test suite for odor sampling utilities."""
    
    @pytest.fixture
    def sample_plume_frame(self):
        """Create a sample plume frame for testing."""
        # Create a 10x10 gradient frame
        frame = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                frame[i, j] = (i + j) / 18.0  # Normalized gradient
        return frame

    def test_sample_odor_at_sensors_nearest(self, sample_plume_frame):
        """Test nearest neighbor sampling."""
        sensor_positions = np.array([
            [2.0, 3.0],
            [5.5, 4.5],
            [8.0, 7.0]
        ])
        
        concentrations = sample_odor_at_sensors(
            sensor_positions, sample_plume_frame, sampling_method="nearest"
        )
        
        assert concentrations.shape == (3,)
        # Check expected values (rounded to nearest indices)
        assert concentrations[0] == sample_plume_frame[3, 2]  # (2,3) -> [3,2]
        assert concentrations[1] == sample_plume_frame[4, 6]  # (5.5,4.5) -> [4,6]
        assert concentrations[2] == sample_plume_frame[7, 8]  # (8,7) -> [7,8]

    def test_sample_odor_at_sensors_bilinear(self, sample_plume_frame):
        """Test bilinear interpolation sampling."""
        sensor_positions = np.array([
            [2.5, 3.5],  # Exactly between four pixels
            [0.0, 0.0],  # Corner position
            [9.0, 9.0]   # Edge position
        ])
        
        concentrations = sample_odor_at_sensors(
            sensor_positions, sample_plume_frame, sampling_method="bilinear"
        )
        
        assert concentrations.shape == (3,)
        # For the middle position (2.5, 3.5), it should be average of 4 corners
        expected_middle = (
            sample_plume_frame[3, 2] + sample_plume_frame[3, 3] +
            sample_plume_frame[4, 2] + sample_plume_frame[4, 3]
        ) / 4.0
        np.testing.assert_almost_equal(concentrations[0], expected_middle)
        
        # Corner and edge positions should match exact values
        assert concentrations[1] == sample_plume_frame[0, 0]
        assert concentrations[2] == sample_plume_frame[9, 9]

    def test_sample_odor_at_sensors_bicubic(self, sample_plume_frame):
        """Test bicubic interpolation sampling."""
        sensor_positions = np.array([
            [4.3, 5.7],
            [1.1, 2.9]
        ])
        
        concentrations = sample_odor_at_sensors(
            sensor_positions, sample_plume_frame, sampling_method="bicubic"
        )
        
        assert concentrations.shape == (2,)
        # Bicubic should provide smooth interpolation (exact values depend on scipy implementation)
        assert np.all(concentrations >= 0.0)
        assert np.all(concentrations <= 1.0)

    def test_sample_odor_at_sensors_boundary_clipping(self, sample_plume_frame):
        """Test that out-of-bounds positions are clipped properly."""
        sensor_positions = np.array([
            [-1.0, 5.0],   # Negative x
            [5.0, -1.0],   # Negative y
            [15.0, 5.0],   # x beyond bounds
            [5.0, 15.0]    # y beyond bounds
        ])
        
        concentrations = sample_odor_at_sensors(
            sensor_positions, sample_plume_frame, sampling_method="nearest"
        )
        
        assert concentrations.shape == (4,)
        # All should be clipped to valid frame boundaries
        assert concentrations[0] == sample_plume_frame[5, 0]  # Clipped to x=0
        assert concentrations[1] == sample_plume_frame[0, 5]  # Clipped to y=0
        assert concentrations[2] == sample_plume_frame[5, 9]  # Clipped to x=9
        assert concentrations[3] == sample_plume_frame[9, 5]  # Clipped to y=9

    def test_sample_odor_at_sensors_invalid_method(self, sample_plume_frame):
        """Test error handling for invalid sampling method."""
        sensor_positions = np.array([[5.0, 5.0]])
        
        with pytest.raises(ValueError, match="Unsupported sampling method: invalid"):
            sample_odor_at_sensors(
                sensor_positions, sample_plume_frame, sampling_method="invalid"
            )


class TestSeedManagerIntegration:
    """Test suite for seed manager integration with navigator utilities."""
    
    def test_reproducible_parameter_normalization(self):
        """Test that parameter normalization is deterministic with seed management."""
        # Set a specific seed
        seed = set_global_seed(42)
        assert seed == 42
        
        # Generate some random data for normalization
        np.random.seed(42)
        random_params = np.random.rand(5)
        
        result1 = normalize_array_parameter(random_params, 5)
        
        # Reset seed and repeat
        set_global_seed(42)
        np.random.seed(42)
        random_params_repeat = np.random.rand(5)
        result2 = normalize_array_parameter(random_params_repeat, 5)
        
        # Results should be identical
        np.testing.assert_array_equal(result1, result2)

    def test_reproducible_sensor_calculations(self):
        """Test reproducible sensor position calculations."""
        seed = set_global_seed(123)
        assert seed == 123
        
        # Use numpy generator for reproducible random positions
        rng = get_numpy_generator()
        agent_pos = rng.uniform(0, 100, 2)
        agent_orientation = rng.uniform(0, 360)
        
        positions1 = calculate_sensor_positions(
            agent_pos, agent_orientation, "TRIANGLE"
        )
        
        # Reset and repeat with same seed
        set_global_seed(123)
        rng = get_numpy_generator()
        agent_pos_repeat = rng.uniform(0, 100, 2)
        agent_orientation_repeat = rng.uniform(0, 360)
        
        positions2 = calculate_sensor_positions(
            agent_pos_repeat, agent_orientation_repeat, "TRIANGLE"
        )
        
        # Results should be identical
        np.testing.assert_array_almost_equal(positions1, positions2)

    def test_seed_state_tracking(self):
        """Test seed state tracking capabilities."""
        # Initialize with a known seed
        original_seed = set_global_seed(999)
        assert original_seed == 999
        
        # Check that we can retrieve the current seed
        current_seed = get_current_seed()
        assert current_seed == 999
        
        # Use some random operations
        rng = get_numpy_generator()
        _ = rng.random(10)
        
        # Seed should still be tracked correctly
        tracked_seed = get_current_seed()
        assert tracked_seed == 999

    def test_initialize_reproducibility_integration(self):
        """Test the convenience function for reproducibility setup."""
        # Test with explicit seed
        seed = initialize_reproducibility(seed=456)
        assert seed == 456
        assert get_current_seed() == 456
        
        # Test with auto-generated seed
        auto_seed = initialize_reproducibility()
        assert auto_seed is not None
        assert isinstance(auto_seed, int)
        assert get_current_seed() == auto_seed


class TestHydraConfigurationIntegration:
    """Test suite for Hydra configuration integration with navigator utilities."""
    
    def test_config_file_loading(self, temp_config_files):
        """Test loading YAML configuration files for navigator creation."""
        configs = temp_config_files
        config_path = configs["config_path"]
        
        # Test loading configuration
        loaded_config = load_yaml(config_path)
        assert isinstance(loaded_config, dict)
        assert "navigator" in loaded_config
        
        # Test that configuration contains expected structure
        navigator_config = loaded_config["navigator"]
        assert "orientation" in navigator_config
        assert "speed" in navigator_config
        assert "max_speed" in navigator_config

    def test_hierarchical_config_composition(self, temp_config_files):
        """Test hierarchical configuration file composition patterns."""
        configs = temp_config_files
        base_config = configs["base_config"]
        user_config = configs["user_config"]
        
        # Simulate Hydra-style configuration composition
        composed_config = {**base_config, **user_config}
        
        # Navigator section should have user overrides
        assert composed_config["navigator"]["orientation"] == 45.0  # User override
        assert composed_config["navigator"]["max_speed"] == 2.0     # User override
        
        # Video plume section should also have user overrides
        assert composed_config["video_plume"]["flip"] is True       # User override
        assert composed_config["video_plume"]["kernel_size"] == 5   # User override
        assert composed_config["video_plume"]["kernel_sigma"] == 1.0  # From base

    def test_config_validation_patterns(self, temp_config_files):
        """Test configuration validation patterns for navigator utilities."""
        configs = temp_config_files
        config_data = configs["user_config"]
        
        # Test valid configuration
        navigator_params = config_data["navigator"]
        
        # These should not raise errors for valid configurations
        assert isinstance(navigator_params["orientation"], (int, float))
        assert isinstance(navigator_params["speed"], (int, float))
        assert isinstance(navigator_params["max_speed"], (int, float))
        
        # Test constraint validation
        assert navigator_params["speed"] >= 0.0
        assert navigator_params["max_speed"] > 0.0
        assert 0 <= navigator_params["orientation"] <= 360

    def test_environment_variable_interpolation_patterns(self, temp_config_files):
        """Test patterns for environment variable interpolation in configurations."""
        configs = temp_config_files
        paths_template = configs["paths_path"]
        
        # Load the paths template
        paths_config = load_yaml(paths_template)
        
        # Verify template contains environment variable patterns
        assert "${oc.env:DATA_DIR,/tmp/data}" in str(paths_config["data_dir"])
        assert "${oc.env:OUTPUT_DIR,/tmp/outputs}" in str(paths_config["output_dir"])
        
        # Test that we can handle these patterns (simulation of Hydra interpolation)
        import os
        test_data_dir = "/custom/data/path"
        os.environ["DATA_DIR"] = test_data_dir
        
        # Simulate environment variable resolution
        resolved_path = os.environ.get("DATA_DIR", "/tmp/data")
        assert resolved_path == test_data_dir


class TestUtilsConsolidationScenarios:
    """Test suite for utils consolidation scenarios and integration patterns."""
    
    def test_complete_navigation_workflow_integration(self, mock_navigator_config, mock_navigator_class):
        """Test complete workflow integration with all utilities."""
        mock_config_class, mock_config_instance = mock_navigator_config
        mock_navigator, mock_single, mock_multi = mock_navigator_class
        
        # Set reproducible seed
        seed = set_global_seed(789)
        
        # Create navigator with normalized parameters
        positions = normalize_array_parameter([10.0, 20.0], 2)
        orientations = normalize_array_parameter(45.0, 2)
        speeds = normalize_array_parameter(1.5, 2)
        
        # Create navigator instance
        navigator = create_navigator_from_params(
            navigator_type="multi",
            initial_position=positions.tolist(),
            initial_orientation=orientations[0],
            max_speed=3.0
        )
        
        # Calculate sensor positions for first agent
        agent_pos = np.array([10.0, 20.0])
        sensor_positions = calculate_sensor_positions(
            agent_pos, 45.0, "LEFT_RIGHT"
        )
        
        # Create sample plume for odor sampling
        plume_frame = np.random.rand(100, 100)
        concentrations = sample_odor_at_sensors(
            sensor_positions, plume_frame, "bilinear"
        )
        
        # Verify all components integrated successfully
        assert navigator is not None
        assert sensor_positions.shape == (2, 2)
        assert concentrations.shape == (2,)
        assert get_current_seed() == seed

    def test_parameter_validation_integration(self):
        """Test parameter validation across utility functions."""
        # Test coordinate validation for sensor calculations
        with pytest.raises(ValueError):
            calculate_sensor_positions(
                np.array([1, 2, 3]),  # Invalid shape
                0.0,
                "SINGLE"
            )
        
        # Test sampling method validation
        sensor_pos = np.array([[5.0, 5.0]])
        plume = np.ones((10, 10))
        
        with pytest.raises(ValueError, match="Unsupported sampling method"):
            sample_odor_at_sensors(sensor_pos, plume, "invalid_method")

    def test_factory_integration_with_error_handling(self):
        """Test factory method integration with comprehensive error handling."""
        # Test import error handling
        with patch('{{cookiecutter.project_slug}}.utils.Navigator', side_effect=ImportError):
            with pytest.raises(ImportError, match="Navigator modules unavailable"):
                create_navigator_from_params()
        
        # Test configuration validation error handling
        with patch('{{cookiecutter.project_slug}}.utils.NavigatorConfig', side_effect=ValueError("Bad config")):
            with pytest.raises(ValueError, match="Bad config"):
                create_navigator_from_params(max_speed=-1.0)

    def test_module_info_and_health_checking(self):
        """Test module information and health checking utilities."""
        module_info = get_module_info()
        
        # Verify expected structure
        assert isinstance(module_info, dict)
        assert "available_modules" in module_info
        assert "import_errors" in module_info
        assert "total_modules" in module_info
        assert "total_errors" in module_info
        assert "status" in module_info
        
        # Verify data types
        assert isinstance(module_info["available_modules"], list)
        assert isinstance(module_info["import_errors"], dict)
        assert isinstance(module_info["total_modules"], int)
        assert isinstance(module_info["total_errors"], int)
        assert module_info["status"] in ["healthy", "degraded"]


class TestIOUtilitiesIntegration:
    """Test suite for I/O utilities integration with navigator testing."""
    
    def test_yaml_configuration_roundtrip(self, tmp_path):
        """Test YAML configuration save/load roundtrip."""
        test_config = {
            "navigator": {
                "type": "single",
                "initial_position": [25.0, 30.0],
                "initial_orientation": 90.0,
                "max_speed": 5.0,
                "angular_velocity": 0.15
            },
            "sensors": {
                "layout": "LEFT_RIGHT",
                "distance": 3.0,
                "angle": 45.0
            }
        }
        
        config_file = tmp_path / "test_config.yaml"
        
        # Save configuration
        save_yaml(test_config, config_file)
        assert config_file.exists()
        
        # Load configuration
        loaded_config = load_yaml(config_file)
        
        # Verify roundtrip accuracy
        assert loaded_config == test_config
        assert loaded_config["navigator"]["initial_position"] == [25.0, 30.0]
        assert loaded_config["sensors"]["layout"] == "LEFT_RIGHT"

    def test_json_trajectory_data_handling(self, tmp_path):
        """Test JSON handling for trajectory data."""
        trajectory_data = {
            "agent_id": 1,
            "timestamps": [0.0, 0.1, 0.2, 0.3],
            "positions": [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0], [13.0, 23.0]],
            "orientations": [0.0, 5.0, 10.0, 15.0],
            "odor_concentrations": [0.1, 0.2, 0.3, 0.4]
        }
        
        trajectory_file = tmp_path / "trajectory.json"
        
        # Save trajectory data
        save_json(trajectory_data, trajectory_file)
        assert trajectory_file.exists()
        
        # Load trajectory data
        loaded_trajectory = load_json(trajectory_file)
        
        # Verify data integrity
        assert loaded_trajectory == trajectory_data
        assert len(loaded_trajectory["positions"]) == 4
        assert loaded_trajectory["agent_id"] == 1

    def test_numpy_array_persistence(self, tmp_path):
        """Test NumPy array persistence for simulation results."""
        # Create sample trajectory data
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        concentrations = np.array([0.1, 0.2, 0.3])
        
        pos_file = tmp_path / "positions.npy"
        conc_file = tmp_path / "concentrations.npy"
        
        # Save arrays
        save_numpy(positions, pos_file)
        save_numpy(concentrations, conc_file)
        
        assert pos_file.exists()
        assert conc_file.exists()
        
        # Load arrays
        loaded_positions = load_numpy(pos_file)
        loaded_concentrations = load_numpy(conc_file)
        
        # Verify data integrity
        np.testing.assert_array_equal(loaded_positions, positions)
        np.testing.assert_array_equal(loaded_concentrations, concentrations)
        assert loaded_positions.shape == (3, 2)
        assert loaded_concentrations.shape == (3,)

    def test_file_error_handling(self, tmp_path):
        """Test error handling for file operations."""
        nonexistent_file = tmp_path / "nonexistent.yaml"
        
        # Test FileNotFoundError
        with pytest.raises(FileNotFoundError, match="YAML file not found"):
            load_yaml(nonexistent_file)
        
        with pytest.raises(FileNotFoundError, match="JSON file not found"):
            load_json(nonexistent_file)
        
        with pytest.raises(FileNotFoundError, match="NumPy file not found"):
            load_numpy(nonexistent_file)


class TestNavigatorProtocolIntegration:
    """Test suite for NavigatorProtocol integration with utilities."""
    
    def test_navigator_protocol_type_checking(self):
        """Test NavigatorProtocol type checking with utility functions."""
        # Create a mock navigator that implements the protocol
        mock_navigator = Mock(spec=NavigatorProtocol)
        
        # Configure protocol properties
        mock_navigator.positions = np.array([[10.0, 20.0]])
        mock_navigator.orientations = np.array([45.0])
        mock_navigator.speeds = np.array([1.5])
        mock_navigator.max_speeds = np.array([3.0])
        mock_navigator.angular_velocities = np.array([0.1])
        mock_navigator.num_agents = 1
        
        # Test that mock satisfies protocol interface
        assert hasattr(mock_navigator, 'positions')
        assert hasattr(mock_navigator, 'orientations')
        assert hasattr(mock_navigator, 'step')
        assert hasattr(mock_navigator, 'sample_odor')
        
        # Test integration with utility functions
        sensor_pos = calculate_sensor_positions(
            mock_navigator.positions[0],
            mock_navigator.orientations[0],
            "LEFT_RIGHT"
        )
        
        assert sensor_pos.shape == (2, 2)

    def test_navigator_protocol_runtime_checking(self):
        """Test runtime protocol checking capabilities."""
        # Test that NavigatorProtocol is runtime checkable
        assert hasattr(NavigatorProtocol, '__protocol_attrs__') or \
               getattr(NavigatorProtocol, '_is_protocol', False) or \
               hasattr(NavigatorProtocol, '__annotations__')
        
        # Create a proper protocol-compliant mock
        class MockNavigator:
            @property
            def positions(self):
                return np.array([[0.0, 0.0]])
            
            @property
            def orientations(self):
                return np.array([0.0])
            
            @property
            def speeds(self):
                return np.array([0.0])
            
            @property
            def max_speeds(self):
                return np.array([1.0])
                
            @property
            def angular_velocities(self):
                return np.array([0.1])
            
            @property
            def num_agents(self):
                return 1
            
            def reset(self, **kwargs):
                pass
            
            def step(self, env_array):
                pass
            
            def sample_odor(self, env_array):
                return 0.0
            
            def read_single_antenna_odor(self, env_array):
                return 0.0
            
            def sample_multiple_sensors(self, env_array, **kwargs):
                return np.array([0.0, 0.0])
        
        mock_nav = MockNavigator()
        
        # Test protocol compliance (runtime checking if available)
        try:
            assert isinstance(mock_nav, NavigatorProtocol)
        except TypeError:
            # Protocol runtime checking not available, just verify interface
            assert hasattr(mock_nav, 'positions')
            assert hasattr(mock_nav, 'step')
            assert hasattr(mock_nav, 'sample_odor')