"""
Tests for the enhanced configuration schema validation module.

This module provides comprehensive testing for Pydantic-based configuration schemas
with Hydra integration, covering hierarchical configuration composition, override
scenarios, and enhanced validation patterns for the enhanced system architecture.
"""

import pytest
from typing import Dict, Any, List
from pathlib import Path
from unittest.mock import patch, MagicMock

# Updated imports for new package structure
from plume_nav_sim.config.schemas import (
    NavigatorConfig,
    SingleAgentConfig,
    MultiAgentConfig,
    VideoPlumeConfig,
)

# Import for validation errors from Pydantic
from pydantic import ValidationError


class TestNavigatorConfig:
    """Test validation of the unified NavigatorConfig schema."""

    def test_single_agent_mode_validation(self):
        """Test NavigatorConfig validation in single-agent mode."""
        # Valid single agent configuration
        config_data = {
            "position": [0.0, 0.0],
            "orientation": 0.0,
            "speed": 1.0,
            "max_speed": 2.0,
            "angular_velocity": 0.0
        }
        config = NavigatorConfig(**config_data)
        
        assert config.position == (0.0, 0.0)
        assert config.orientation == 0.0
        assert config.speed == 1.0
        assert config.max_speed == 2.0
        assert config.angular_velocity == 0.0

    def test_multi_agent_mode_validation(self):
        """Test NavigatorConfig validation in multi-agent mode."""
        # Valid multi-agent configuration
        config_data = {
            "positions": [[0.0, 0.0], [1.0, 1.0]],
            "orientations": [0.0, 90.0],
            "speeds": [1.0, 1.2],
            "max_speeds": [2.0, 2.5],
            "angular_velocities": [0.0, 0.1],
            "num_agents": 2
        }
        config = NavigatorConfig(**config_data)
        
        assert len(config.positions) == 2
        assert len(config.orientations) == 2
        assert len(config.speeds) == 2
        assert config.num_agents == 2

    def test_mixed_mode_validation_error(self):
        """Test that mixed single/multi-agent parameters raise validation error."""
        # Invalid: mixing single and multi-agent parameters
        config_data = {
            "position": [0.0, 0.0],  # Single-agent parameter
            "positions": [[0.0, 0.0], [1.0, 1.0]],  # Multi-agent parameter
            "orientation": 0.0,
            "speeds": [1.0, 1.2],
            "max_speed": 2.0
        }
        
        with pytest.raises(ValidationError) as exc_info:
            NavigatorConfig(**config_data)
        
        assert "Cannot specify both single-agent" in str(exc_info.value)

    def test_speed_constraint_validation(self):
        """Test speed constraint validation for single agent."""
        # Invalid: speed exceeds max_speed
        config_data = {
            "position": [0.0, 0.0],
            "orientation": 0.0,
            "speed": 3.0,  # Exceeds max_speed
            "max_speed": 2.0,
            "angular_velocity": 0.0
        }
        
        with pytest.raises(ValidationError) as exc_info:
            NavigatorConfig(**config_data)
        
        assert "cannot exceed max_speed" in str(exc_info.value)

    def test_multi_agent_speed_constraint_validation(self):
        """Test speed constraint validation for multi-agent mode."""
        # Invalid: one agent's speed exceeds max_speed
        config_data = {
            "positions": [[0.0, 0.0], [1.0, 1.0]],
            "orientations": [0.0, 90.0],
            "speeds": [1.0, 3.0],  # Second agent speed exceeds max_speed
            "max_speeds": [2.0, 2.5],
            "num_agents": 2
        }
        
        with pytest.raises(ValidationError) as exc_info:
            NavigatorConfig(**config_data)
        
        assert "Agent 1: speed" in str(exc_info.value)
        assert "cannot exceed max_speed" in str(exc_info.value)

    def test_orientation_bounds_validation(self):
        """Orientations outside valid range should raise errors."""
        # Single agent orientation above 360
        config_data = {
            "position": [0.0, 0.0],
            "orientation": 450.0,
            "speed": 1.0,
            "max_speed": 2.0
        }
        with pytest.raises(ValidationError, match="less than or equal to 360"):
            NavigatorConfig(**config_data)

        # Multi-agent orientations with invalid values
        config_data = {
            "positions": [[0.0, 0.0], [1.0, 1.0]],
            "orientations": [90.0, -90.0],
            "speeds": [1.0, 1.0],
            "max_speeds": [2.0, 2.0],
            "num_agents": 2
        }
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            NavigatorConfig(**config_data)

    def test_multi_agent_parameter_consistency(self):
        """Test that multi-agent parameter lists have consistent lengths."""
        # Invalid: inconsistent parameter list lengths
        config_data = {
            "positions": [[0.0, 0.0], [1.0, 1.0]],  # 2 agents
            "orientations": [0.0],  # Only 1 orientation
            "speeds": [1.0, 1.0],
            "max_speeds": [2.0, 2.0],
            "num_agents": 2
        }
        
        with pytest.raises(ValidationError) as exc_info:
            NavigatorConfig(**config_data)
        
        assert "orientations length" in str(exc_info.value)
        assert "does not match number of agents" in str(exc_info.value)

    def test_num_agents_consistency(self):
        """Test that num_agents matches position list length."""
        # Invalid: num_agents doesn't match positions length
        config_data = {
            "positions": [[0.0, 0.0], [1.0, 1.0]],  # 2 positions
            "orientations": [0.0, 90.0],
            "speeds": [1.0, 1.0],
            "max_speeds": [2.0, 2.0],
            "num_agents": 3  # Doesn't match positions length
        }
        
        with pytest.raises(ValidationError) as exc_info:
            NavigatorConfig(**config_data)
        
        assert "num_agents (3) does not match number of positions (2)" in str(exc_info.value)


class TestSingleAgentConfig:
    """Test validation of SingleAgentConfig schema."""

    def test_valid_single_agent_config(self):
        """Test that a valid single agent config passes validation."""
        config_data = {
            "position": [10.0, 20.0],
            "orientation": 45.0,
            "speed": 1.5,
            "max_speed": 3.0,
            "angular_velocity": 0.2
        }
        config = SingleAgentConfig(**config_data)
        
        assert config.position == (10.0, 20.0)
        assert config.orientation == 45.0
        assert config.speed == 1.5
        assert config.max_speed == 3.0
        assert config.angular_velocity == 0.2

    def test_speed_validation(self):
        """Test speed validation constraints."""
        # Test negative speed validation
        config_data = {
            "position": [0.0, 0.0],
            "orientation": 0.0,
            "speed": -1.0,  # Invalid: negative speed
            "max_speed": 2.0
        }
        
        with pytest.raises(ValidationError) as exc_info:
            SingleAgentConfig(**config_data)
        
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_max_speed_validation(self):
        """Test max_speed validation constraints."""
        # Test zero max_speed validation
        config_data = {
            "position": [0.0, 0.0],
            "orientation": 0.0,
            "speed": 0.0,
            "max_speed": 0.0  # Invalid: max_speed must be > 0
        }
        
        with pytest.raises(ValidationError) as exc_info:
            SingleAgentConfig(**config_data)
        
        assert "greater than 0" in str(exc_info.value)

    def test_orientation_normalization_edge_cases(self):
        """Edge case orientations outside range should fail."""
        with pytest.raises(ValidationError, match="less than or equal to 360"):
            SingleAgentConfig(orientation=720.0, speed=1.0, max_speed=2.0)

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            SingleAgentConfig(orientation=-180.0, speed=1.0, max_speed=2.0)


class TestMultiAgentConfig:
    """Test validation of MultiAgentConfig schema."""

    def test_valid_multi_agent_config(self):
        """Test that a valid multi-agent config passes validation."""
        config_data = {
            "positions": [[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]],
            "orientations": [0.0, 120.0, 240.0],
            "speeds": [1.0, 1.5, 2.0],
            "max_speeds": [2.0, 3.0, 4.0],
            "angular_velocities": [0.1, 0.2, 0.3],
            "num_agents": 3
        }
        config = MultiAgentConfig(**config_data)
        
        assert len(config.positions) == 3
        assert len(config.orientations) == 3
        assert len(config.speeds) == 3
        assert config.num_agents == 3

    def test_position_structure_validation(self):
        """Test that positions have correct [x, y] structure."""
        # Invalid: position with wrong number of coordinates
        config_data = {
            "positions": [[0.0, 0.0], [5.0]],  # Second position incomplete
            "orientations": [0.0, 90.0],
            "speeds": [1.0, 1.0],
            "max_speeds": [2.0, 2.0],
            "num_agents": 2
        }
        
        with pytest.raises(ValidationError) as exc_info:
            MultiAgentConfig(**config_data)
        
        assert "Position 1 must be a list/tuple of [x, y] coordinates" in str(exc_info.value)

    def test_position_numeric_validation(self):
        """Test that position coordinates are numeric."""
        # Invalid: non-numeric position coordinates
        config_data = {
            "positions": [[0.0, 0.0], ["invalid", 5.0]],  # Second position has non-numeric coordinate
            "orientations": [0.0, 90.0],
            "speeds": [1.0, 1.0],
            "max_speeds": [2.0, 2.0],
            "num_agents": 2
        }
        
        with pytest.raises(ValidationError) as exc_info:
            MultiAgentConfig(**config_data)
        
        assert "Position 1 coordinates must be numeric" in str(exc_info.value)

    def test_negative_speed_validation(self):
        """Test that negative speeds are rejected."""
        config_data = {
            "positions": [[0.0, 0.0], [5.0, 5.0]],
            "orientations": [0.0, 90.0],
            "speeds": [1.0, -1.0],  # Invalid: negative speed
            "max_speeds": [2.0, 2.0],
            "num_agents": 2
        }
        
        with pytest.raises(ValidationError) as exc_info:
            MultiAgentConfig(**config_data)
        
        assert "Speed at index 1 must be non-negative" in str(exc_info.value)

    def test_parameter_list_length_consistency(self):
        """Test comprehensive parameter list length validation."""
        positions = [[0.0, 0.0], [5.0, 5.0]]
        
        # Test each parameter list with wrong length
        test_cases = [
            ("orientations", [0.0]),  # Too short
            ("speeds", [1.0]),  # Too short
            ("max_speeds", [2.0]),  # Too short
            ("angular_velocities", [0.1]),  # Too short
        ]
        
        for param_name, param_value in test_cases:
            config_data = {
                "positions": positions,
                "orientations": [0.0, 90.0],
                "speeds": [1.0, 1.0],
                "max_speeds": [2.0, 2.0],
                "angular_velocities": [0.1, 0.1],
                "num_agents": 2
            }
            config_data[param_name] = param_value
            
            with pytest.raises(ValidationError) as exc_info:
                MultiAgentConfig(**config_data)
            
            assert f"{param_name} length" in str(exc_info.value)
            assert "does not match number of agents" in str(exc_info.value)


class TestVideoPlumeConfig:
    """Test validation of VideoPlumeConfig schema."""

    def test_valid_video_plume_config(self):
        """Test that a valid video plume config passes validation."""
        config_data = {
            "video_path": "data/test_video.mp4",
            "flip": False,
            "grayscale": True,
            "kernel_size": 5,
            "kernel_sigma": 1.0,
            "threshold": 0.5,
            "normalize": True,
            "_skip_validation": True  # Skip file existence for testing
        }
        config = VideoPlumeConfig(**config_data)
        
        assert str(config.video_path) == "data/test_video.mp4"
        assert config.flip == False
        assert config.grayscale == True
        assert config.kernel_size == 5
        assert config.kernel_sigma == 1.0
        assert config.threshold == 0.5
        assert config.normalize == True

    def test_kernel_size_validation(self):
        """Test kernel_size validation (must be odd and positive)."""
        # Test negative kernel size
        config_data = {
            "video_path": "data/test_video.mp4",
            "kernel_size": -5,  # Invalid: negative
            "kernel_sigma": 1.0,
            "_skip_validation": True
        }
        
        with pytest.raises(ValidationError) as exc_info:
            VideoPlumeConfig(**config_data)
        
        assert "kernel_size must be positive" in str(exc_info.value)

        # Test even kernel size
        config_data = {
            "video_path": "data/test_video.mp4",
            "kernel_size": 4,  # Invalid: even
            "kernel_sigma": 1.0,
            "_skip_validation": True
        }
        
        with pytest.raises(ValidationError) as exc_info:
            VideoPlumeConfig(**config_data)
        
        assert "kernel_size must be odd" in str(exc_info.value)

    def test_kernel_sigma_validation(self):
        """Test kernel_sigma validation (must be positive)."""
        config_data = {
            "video_path": "data/test_video.mp4",
            "kernel_size": 5,
            "kernel_sigma": 0.0,  # Invalid: must be > 0
            "_skip_validation": True
        }
        
        with pytest.raises(ValidationError) as exc_info:
            VideoPlumeConfig(**config_data)
        
        assert "greater than 0" in str(exc_info.value)

    def test_threshold_range_validation(self):
        """Test threshold validation (must be between 0 and 1)."""
        # Test threshold > 1
        config_data = {
            "video_path": "data/test_video.mp4",
            "threshold": 1.5,  # Invalid: > 1
            "_skip_validation": True
        }
        
        with pytest.raises(ValidationError) as exc_info:
            VideoPlumeConfig(**config_data)
        
        assert "less than or equal to 1" in str(exc_info.value)

        # Test negative threshold
        config_data = {
            "video_path": "data/test_video.mp4",
            "threshold": -0.1,  # Invalid: < 0
            "_skip_validation": True
        }
        
        with pytest.raises(ValidationError) as exc_info:
            VideoPlumeConfig(**config_data)
        
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_gaussian_parameter_consistency(self):
        """Test that Gaussian parameters are specified together."""
        # Invalid: kernel_size without kernel_sigma
        config_data = {
            "video_path": "data/test_video.mp4",
            "kernel_size": 5,
            # Missing kernel_sigma
            "_skip_validation": True
        }
        
        with pytest.raises(ValidationError) as exc_info:
            VideoPlumeConfig(**config_data)
        
        assert "kernel_sigma must be specified when kernel_size is provided" in str(exc_info.value)

        # Invalid: kernel_sigma without kernel_size
        config_data = {
            "video_path": "data/test_video.mp4",
            "kernel_sigma": 1.0,
            # Missing kernel_size
            "_skip_validation": True
        }
        
        with pytest.raises(ValidationError) as exc_info:
            VideoPlumeConfig(**config_data)
        
        assert "kernel_size must be specified when kernel_sigma is provided" in str(exc_info.value)

    def test_video_path_conversion(self):
        """Test video path conversion to Path object."""
        config_data = {
            "video_path": "data/test_video.mp4",
            "_skip_validation": True
        }
        config = VideoPlumeConfig(**config_data)
        
        assert isinstance(config.video_path, Path)
        assert str(config.video_path) == "data/test_video.mp4"

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    def test_video_file_existence_validation(self, mock_is_file, mock_exists):
        """Test video file existence validation."""
        # Test file doesn't exist
        mock_exists.return_value = False
        mock_is_file.return_value = False
        
        config_data = {
            "video_path": "data/nonexistent.mp4",
            "_skip_validation": False  # Enable validation
        }
        
        with pytest.raises(ValidationError) as exc_info:
            VideoPlumeConfig(**config_data)
        
        assert "Video file not found" in str(exc_info.value)

        # Test path exists but is not a file
        mock_exists.return_value = True
        mock_is_file.return_value = False
        
        with pytest.raises(ValidationError) as exc_info:
            VideoPlumeConfig(**config_data)
        
        assert "Video path is not a file" in str(exc_info.value)

        # Test valid file
        mock_exists.return_value = True
        mock_is_file.return_value = True
        
        config = VideoPlumeConfig(**config_data)
        assert config.video_path == Path("data/nonexistent.mp4")


class TestHydraIntegration:
    """Test Hydra configuration integration and enhanced validation patterns."""

    def test_hydra_configstore_registration(self):
        """Test that schemas are properly registered with Hydra ConfigStore."""
        try:
            from hydra.core.config_store import ConfigStore
            cs = ConfigStore.instance()
            
            # Test that schemas can be retrieved from ConfigStore
            # Note: The actual registration happens during import
            assert cs is not None
            
        except ImportError:
            # Hydra not available in test environment - skip test
            pytest.skip("Hydra not available for ConfigStore testing")

    def test_environment_variable_interpolation_compatibility(self):
        """Test schema compatibility with Hydra environment variable interpolation."""
        # Test that schemas can handle environment variable placeholders
        config_data = {
            "video_path": "${oc.env:VIDEO_PATH,data/default.mp4}",
            "flip": "${oc.env:VIDEO_FLIP,false}",
            "_skip_validation": True
        }
        
        # Should not raise validation error for interpolation strings
        config = VideoPlumeConfig(**config_data)
        assert "${oc.env:VIDEO_PATH,data/default.mp4}" in str(config.video_path)

    def test_configuration_composition_validation(self):
        """Test schema validation with hierarchical configuration composition."""
        # Test multi-level configuration merging scenario
        base_config = {
            "position": [0.0, 0.0],
            "orientation": 0.0,
            "speed": 1.0,
            "max_speed": 2.0
        }
        
        override_config = {
            "speed": 1.5,  # Override speed
            "angular_velocity": 0.2  # Add new parameter
        }
        
        # Merge configurations (simulating Hydra composition)
        merged_config = {**base_config, **override_config}
        
        config = NavigatorConfig(**merged_config)
        assert config.speed == 1.5
        assert config.angular_velocity == 0.2

    @pytest.mark.parametrize("invalid_override", [
        {"speed": "invalid_string"},  # Type error
        {"max_speed": -1.0},  # Value error
        {"position": [1, 2, 3]},  # Structure error
    ])
    def test_configuration_override_validation_errors(self, invalid_override):
        """Test that invalid configuration overrides are properly rejected."""
        base_config = {
            "position": [0.0, 0.0],
            "orientation": 0.0,
            "speed": 1.0,
            "max_speed": 2.0
        }
        
        # Apply invalid override
        merged_config = {**base_config, **invalid_override}
        
        with pytest.raises(ValidationError):
            NavigatorConfig(**merged_config)

    def test_schema_extra_fields_handling(self):
        """Test handling of extra fields in configuration (allow mode)."""
        # Schemas should allow extra fields for flexibility
        config_data = {
            "position": [0.0, 0.0],
            "orientation": 0.0,
            "speed": 1.0,
            "max_speed": 2.0,
            "extra_field": "extra_value",  # Should be allowed
            "custom_parameter": 42
        }
        
        config = NavigatorConfig(**config_data)
        assert config.position == (0.0, 0.0)
        # Extra fields should be preserved
        assert hasattr(config, 'extra_field')
        assert config.extra_field == "extra_value"


class TestAdvancedValidationPatterns:
    """Test advanced validation patterns for enhanced configuration system."""

    def test_cross_field_validation_complex_scenarios(self):
        """Test complex cross-field validation scenarios."""
        # Test multi-agent with complex speed relationships
        config_data = {
            "positions": [[i*10.0, i*10.0] for i in range(5)],  # 5 agents
            "orientations": [i*72.0 for i in range(5)],  # Every 72 degrees
            "speeds": [1.0, 2.0, 3.0, 4.0, 5.0],
            "max_speeds": [2.0, 3.0, 4.0, 5.0, 6.0],
            "num_agents": 5
        }
        
        config = MultiAgentConfig(**config_data)
        assert len(config.positions) == 5
        assert all(speed <= max_speed for speed, max_speed in zip(config.speeds, config.max_speeds))

    def test_validation_error_message_quality(self):
        """Test that validation error messages are informative and actionable."""
        # Test speed constraint error message
        config_data = {
            "positions": [[0.0, 0.0], [1.0, 1.0]],
            "orientations": [0.0, 90.0],
            "speeds": [1.0, 3.0],  # Second speed exceeds max_speed
            "max_speeds": [2.0, 2.5],
            "num_agents": 2
        }
        
        with pytest.raises(ValidationError) as exc_info:
            MultiAgentConfig(**config_data)
        
        error_message = str(exc_info.value)
        # Should specify which agent has the problem
        assert "Agent 1" in error_message
        # Should specify the constraint that was violated
        assert "cannot exceed max_speed" in error_message
        # Should include the actual values
        assert "3.0" in error_message  # Speed value
        assert "2.5" in error_message  # Max speed value

    def test_type_coercion_and_validation(self):
        """Test type coercion behavior in validation."""
        # Test integer to float coercion
        config_data = {
            "position": [0, 0],  # Integers should be coerced to floats
            "orientation": 45,  # Integer angle
            "speed": 1,  # Integer speed
            "max_speed": 2,  # Integer max speed
        }
        
        config = NavigatorConfig(**config_data)
        assert isinstance(config.position[0], float)
        assert isinstance(config.orientation, float)
        assert isinstance(config.speed, float)
        assert isinstance(config.max_speed, float)

    def test_performance_with_large_configurations(self):
        """Test validation performance with large multi-agent configurations."""
        # Create configuration for 100 agents
        num_agents = 100
        config_data = {
            "positions": [[i*1.0, i*1.0] for i in range(num_agents)],
            "orientations": [(i*3.6) % 360 for i in range(num_agents)],
            "speeds": [1.0] * num_agents,
            "max_speeds": [2.0] * num_agents,
            "angular_velocities": [0.1] * num_agents,
            "num_agents": num_agents
        }
        
        # Validation should complete quickly for large configurations
        import time
        start_time = time.time()
        config = MultiAgentConfig(**config_data)
        validation_time = time.time() - start_time
        
        assert len(config.positions) == num_agents
        # Validation should complete in reasonable time (<1 second)
        assert validation_time < 1.0

    def test_edge_case_numeric_values(self):
        """Test validation with edge case numeric values."""
        import math
        
        # Test very small positive values
        config_data = {
            "position": [1e-10, 1e-10],
            "orientation": 0.0,
            "speed": 1e-6,
            "max_speed": 1e-5,
        }
        config = NavigatorConfig(**config_data)
        assert config.speed < config.max_speed

        # Test very large values
        config_data = {
            "position": [1e6, 1e6],
            "orientation": 0.0,
            "speed": 1e3,
            "max_speed": 1e4,
        }
        config = NavigatorConfig(**config_data)
        assert config.speed < config.max_speed

    def test_schema_serialization_compatibility(self):
        """Test that validated schemas can be serialized/deserialized."""
        config_data = {
            "position": [10.0, 20.0],
            "orientation": 45.0,
            "speed": 1.5,
            "max_speed": 3.0,
            "angular_velocity": 0.2
        }
        config = NavigatorConfig(**config_data)
        
        # Test dict conversion
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict['position'] == [10.0, 20.0]
        
        # Test JSON serialization compatibility
        import json
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        
        # Test round-trip conversion
        recreated_config = NavigatorConfig.model_validate_json(json_str)
        assert recreated_config.position == config.position
        assert recreated_config.speed == config.speed


class TestConfigurationMigrationCompatibility:
    """Test compatibility patterns for configuration migration scenarios."""

    def test_legacy_parameter_handling(self):
        """Test handling of legacy parameter names and structures."""
        # Test that new schemas can handle legacy-style configurations
        # with appropriate field mapping or graceful handling
        legacy_config = {
            "position": [0.0, 0.0],
            "orientation": 0.0,
            "speed": 1.0,
            "max_speed": 2.0,
            # Legacy fields that might exist in old configs
            "legacy_field": "old_value"
        }
        
        # Should handle extra fields gracefully
        config = NavigatorConfig(**legacy_config)
        assert config.position == (0.0, 0.0)
        assert hasattr(config, 'legacy_field')

    def test_partial_configuration_validation(self):
        """Test validation with partial configurations using defaults."""
        # Test minimal configuration with defaults
        minimal_config = {
            "max_speed": 2.0  # Only specify required field
        }
        
        config = NavigatorConfig(**minimal_config)
        assert config.max_speed == 2.0
        assert config.orientation == 0.0  # Should use default
        assert config.speed == 0.0  # Should use default

    def test_configuration_upgrade_patterns(self):
        """Test patterns for upgrading configuration schemas."""
        # Test that new fields can be added without breaking existing configs
        old_style_config = {
            "position": [0.0, 0.0],
            "orientation": 0.0,
            "speed": 1.0,
            "max_speed": 2.0
        }
        
        # Should work even without new optional fields
        config = NavigatorConfig(**old_style_config)
        assert config.angular_velocity == 0.0  # Should use default for new field