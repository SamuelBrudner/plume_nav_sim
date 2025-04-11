"""Tests for the configuration validation module."""

import pytest
from odor_plume_nav.config.validator import (
    validate_config, 
    validate_video_plume_config, 
    validate_navigator_config,
    ConfigValidationError
)


class TestVideoPlumeSectionValidation:
    """Test validation of video_plume section."""

    def test_missing_video_plume_section(self):
        """Test error when video_plume section is missing."""
        config = {}
        with pytest.raises(ConfigValidationError, match="Missing required 'video_plume' section"):
            validate_video_plume_config(config)

    def test_missing_required_fields(self):
        """Test error when required fields are missing."""
        config = {"video_plume": {}}
        with pytest.raises(ConfigValidationError, match="Missing required field 'flip'"):
            validate_video_plume_config(config)
            
        config = {"video_plume": {"flip": False}}
        with pytest.raises(ConfigValidationError, match="Missing required field 'kernel_size'"):
            validate_video_plume_config(config)
            
        config = {"video_plume": {"flip": False, "kernel_size": 0}}
        with pytest.raises(ConfigValidationError, match="Missing required field 'kernel_sigma'"):
            validate_video_plume_config(config)

    def test_invalid_field_types(self):
        """Test error when field types are invalid."""
        config = {"video_plume": {"flip": "not-a-bool", "kernel_size": 0, "kernel_sigma": 1.0}}
        with pytest.raises(ConfigValidationError, match="Invalid type for video_plume.flip"):
            validate_video_plume_config(config)
            
        config = {"video_plume": {"flip": False, "kernel_size": 2.5, "kernel_sigma": 1.0}}
        with pytest.raises(ConfigValidationError, match="Invalid type for video_plume.kernel_size"):
            validate_video_plume_config(config)
            
        config = {"video_plume": {"flip": False, "kernel_size": 0, "kernel_sigma": "1.0"}}
        with pytest.raises(ConfigValidationError, match="Invalid type for video_plume.kernel_sigma"):
            validate_video_plume_config(config)

    def test_invalid_field_values(self):
        """Test error when field values are invalid."""
        config = {"video_plume": {"flip": False, "kernel_size": -1, "kernel_sigma": 1.0}}
        with pytest.raises(ConfigValidationError, match="kernel_size must be non-negative"):
            validate_video_plume_config(config)
            
        config = {"video_plume": {"flip": False, "kernel_size": 0, "kernel_sigma": 0.0}}
        with pytest.raises(ConfigValidationError, match="kernel_sigma must be positive"):
            validate_video_plume_config(config)
            
        config = {"video_plume": {"flip": False, "kernel_size": 0, "kernel_sigma": -1.0}}
        with pytest.raises(ConfigValidationError, match="kernel_sigma must be positive"):
            validate_video_plume_config(config)

    def test_valid_config(self):
        """Test that a valid config passes validation."""
        # Minimal valid config
        config = {"video_plume": {"flip": False, "kernel_size": 0, "kernel_sigma": 1.0}}
        validate_video_plume_config(config)  # Should not raise

        # Alternative valid values
        config = {"video_plume": {"flip": True, "kernel_size": 3, "kernel_sigma": 0.5}}
        validate_video_plume_config(config)  # Should not raise


class TestNavigatorSectionValidation:
    """Test validation of navigator section."""

    def test_missing_navigator_section(self):
        """Test error when navigator section is missing."""
        config = {}
        with pytest.raises(ConfigValidationError, match="Missing required 'navigator' section"):
            validate_navigator_config(config)

    def test_missing_required_fields(self):
        """Test error when required fields are missing."""
        config = {"navigator": {}}
        with pytest.raises(ConfigValidationError, match="Missing required field 'orientation'"):
            validate_navigator_config(config)
            
        config = {"navigator": {"orientation": 0.0}}
        with pytest.raises(ConfigValidationError, match="Missing required field 'speed'"):
            validate_navigator_config(config)
            
        config = {"navigator": {"orientation": 0.0, "speed": 0.0}}
        with pytest.raises(ConfigValidationError, match="Missing required field 'max_speed'"):
            validate_navigator_config(config)

    def test_invalid_field_types(self):
        """Test error when field types are invalid."""
        config = {"navigator": {"orientation": "invalid", "speed": 0.0, "max_speed": 1.0}}
        with pytest.raises(ConfigValidationError, match="Invalid type for navigator.orientation"):
            validate_navigator_config(config)
            
        config = {"navigator": {"orientation": 0.0, "speed": "invalid", "max_speed": 1.0}}
        with pytest.raises(ConfigValidationError, match="Invalid type for navigator.speed"):
            validate_navigator_config(config)
            
        config = {"navigator": {"orientation": 0.0, "speed": 0.0, "max_speed": "invalid"}}
        with pytest.raises(ConfigValidationError, match="Invalid type for navigator.max_speed"):
            validate_navigator_config(config)

    def test_invalid_field_values(self):
        """Test error when field values are invalid."""
        config = {"navigator": {"orientation": 0.0, "speed": 0.0, "max_speed": -1.0}}
        with pytest.raises(ConfigValidationError, match="max_speed must be non-negative"):
            validate_navigator_config(config)
            
        config = {"navigator": {"orientation": 0.0, "speed": 2.0, "max_speed": 1.0}}
        with pytest.raises(ConfigValidationError, match="exceeds max_speed"):
            validate_navigator_config(config)
            
        config = {"navigator": {"orientation": 0.0, "speed": -2.0, "max_speed": 1.0}}
        with pytest.raises(ConfigValidationError, match="exceeds max_speed"):
            validate_navigator_config(config)

    def test_valid_config(self):
        """Test that a valid config passes validation."""
        # Minimal valid config with numeric types
        config = {"navigator": {"orientation": 0.0, "speed": 0.0, "max_speed": 1.0}}
        validate_navigator_config(config)  # Should not raise

        # Alternative valid values and integer types
        config = {"navigator": {"orientation": 90, "speed": -0.5, "max_speed": 2.0}}
        validate_navigator_config(config)  # Should not raise
        
        # Mixed integer and float types
        config = {"navigator": {"orientation": 45.5, "speed": 0, "max_speed": 1}}
        validate_navigator_config(config)  # Should not raise


class TestFullConfigValidation:
    """Test validation of complete configurations."""

    def test_invalid_config_type(self):
        """Test error when config is not a dictionary."""
        config = "not-a-dict"
        with pytest.raises(ConfigValidationError, match="Configuration must be a dictionary"):
            validate_config(config)

        config = 42
        with pytest.raises(ConfigValidationError, match="Configuration must be a dictionary"):
            validate_config(config)

    def test_unknown_validation_section(self):
        """Test error when an unknown section is requested for validation."""
        config = {}
        with pytest.raises(ConfigValidationError, match="Unknown section 'unknown'"):
            validate_config(config, required_sections=["unknown"])

    def test_selective_section_validation(self):
        """Test that only requested sections are validated."""
        # Missing video_plume section but only validating navigator
        config = {
            "navigator": {"orientation": 0.0, "speed": 0.0, "max_speed": 1.0}
        }
        validate_config(config, required_sections=["navigator"])  # Should not raise
        
        # Missing navigator section but only validating video_plume
        config = {
            "video_plume": {"flip": False, "kernel_size": 0, "kernel_sigma": 1.0}
        }
        validate_config(config, required_sections=["video_plume"])  # Should not raise

    def test_complete_valid_config(self):
        """Test that a complete valid config passes validation."""
        config = {
            "video_plume": {"flip": False, "kernel_size": 0, "kernel_sigma": 1.0},
            "navigator": {"orientation": 0.0, "speed": 0.0, "max_speed": 1.0}
        }
        validate_config(config)  # Should not raise
