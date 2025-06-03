"""Tests for the navigator factory module with Hydra configuration integration.

This test suite validates the enhanced navigator factory functionality including:
- Legacy navigator factory method migration to API module
- Hydra-based configuration support and validation
- Parameter merging and override scenarios
- Configuration validation and error handling
- Integration with new Pydantic schema validation

The tests ensure backward compatibility while validating new Hydra configuration
features per Feature F-011 requirements.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Import from new package structure
from {{cookiecutter.project_slug}}.api.navigation import (
    create_navigator,
    create_navigator_from_config,
    ConfigurationError,
    _validate_and_merge_config,
    _normalize_positions
)
from {{cookiecutter.project_slug}}.core.navigator import Navigator, NavigatorProtocol
from {{cookiecutter.project_slug}}.config.schemas import NavigatorConfig, SingleAgentConfig

try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    DictConfig = dict
    OmegaConf = None
    HYDRA_AVAILABLE = False


class TestNavigatorFactory:
    """Test suite for navigator factory functions with Hydra integration."""

    def test_create_navigator_with_default_config(self, config_files):
        """Test creating a navigator with default configuration using legacy method."""
        # Create DictConfig if Hydra is available
        if HYDRA_AVAILABLE:
            cfg = OmegaConf.create(config_files["default_config"]["navigator"])
        else:
            cfg = config_files["default_config"]["navigator"]
        
        # Create a navigator with default config
        navigator = create_navigator_from_config(cfg)
        
        # Verify navigator was created and satisfies protocol
        assert isinstance(navigator, NavigatorProtocol)
        assert hasattr(navigator, 'positions')
        assert hasattr(navigator, 'orientations')
        assert hasattr(navigator, 'speeds')
        assert hasattr(navigator, 'max_speeds')
        
        # Check that the navigator was created with default settings
        assert navigator.orientations[0] == 0.0
        assert navigator.speeds[0] == 0.0
        assert navigator.max_speeds[0] == 1.0

    def test_create_navigator_with_user_config(self, config_files):
        """Test creating a navigator with user configuration using legacy method."""
        if HYDRA_AVAILABLE:
            cfg = OmegaConf.create(config_files["user_config"]["navigator"])
        else:
            cfg = config_files["user_config"]["navigator"]
        
        # Create a navigator with user config
        navigator = create_navigator_from_config(cfg)
        
        # Check that the navigator was created with user settings
        assert navigator.orientations[0] == 45.0
        assert navigator.speeds[0] == 0.5
        assert navigator.max_speeds[0] == 2.0

    def test_create_navigator_with_merged_config(self, config_files):
        """Test creating a navigator with merged configuration."""
        # Create a merged config by combining default and parts of user config
        merged_config = config_files["default_config"]["navigator"].copy()
        merged_config["orientation"] = 90.0  # Override just the orientation parameter
        
        if HYDRA_AVAILABLE:
            cfg = OmegaConf.create(merged_config)
        else:
            cfg = merged_config
        
        # Create a navigator with merged config
        navigator = create_navigator_from_config(cfg)
        
        # Check that the navigator was created with merged settings
        assert navigator.orientations[0] == 90.0  # Overridden
        assert navigator.speeds[0] == 0.0        # Default
        assert navigator.max_speeds[0] == 1.0    # Default

    def test_create_navigator_with_parameter_overrides(self, config_files):
        """Test creating a navigator with configuration and parameter overrides."""
        if HYDRA_AVAILABLE:
            cfg = OmegaConf.create(config_files["default_config"]["navigator"])
        else:
            cfg = config_files["default_config"]["navigator"]
        
        # Create a navigator with default config but override some parameters
        navigator = create_navigator_from_config(cfg, orientation=180.0, speed=0.75)
        
        # Check that the navigator was created with overridden settings
        assert navigator.orientations[0] == 180.0  # Explicitly provided
        assert navigator.speeds[0] == 0.75         # Explicitly provided
        assert navigator.max_speeds[0] == 1.0      # From default config


class TestEnhancedNavigatorCreation:
    """Test suite for enhanced create_navigator function with Hydra support."""

    def test_create_navigator_direct_parameters(self):
        """Test creating navigator with direct parameters (no config)."""
        navigator = create_navigator(
            position=(10.0, 20.0),
            orientation=45.0,
            speed=2.0,
            max_speed=5.0,
            angular_velocity=15.0
        )
        
        assert isinstance(navigator, NavigatorProtocol)
        assert np.allclose(navigator.positions[0], [10.0, 20.0])
        assert navigator.orientations[0] == 45.0
        assert navigator.speeds[0] == 2.0
        assert navigator.max_speeds[0] == 5.0

    def test_create_navigator_with_hydra_config(self, config_files):
        """Test creating navigator with Hydra DictConfig."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available")
        
        # Create Hydra DictConfig
        cfg = OmegaConf.create({
            "position": [15.0, 25.0],
            "orientation": 90.0,
            "speed": 1.5,
            "max_speed": 3.0
        })
        
        navigator = create_navigator(cfg=cfg)
        
        assert isinstance(navigator, NavigatorProtocol)
        assert np.allclose(navigator.positions[0], [15.0, 25.0])
        assert navigator.orientations[0] == 90.0
        assert navigator.speeds[0] == 1.5
        assert navigator.max_speeds[0] == 3.0

    def test_create_navigator_config_with_overrides(self):
        """Test creating navigator with config and parameter overrides."""
        config_dict = {
            "position": [5.0, 10.0],
            "orientation": 30.0,
            "speed": 1.0,
            "max_speed": 2.0
        }
        
        # Override some parameters
        navigator = create_navigator(
            cfg=config_dict,
            orientation=60.0,  # Override orientation
            max_speed=4.0      # Override max_speed
        )
        
        assert np.allclose(navigator.positions[0], [5.0, 10.0])  # From config
        assert navigator.orientations[0] == 60.0                # Overridden
        assert navigator.speeds[0] == 1.0                       # From config
        assert navigator.max_speeds[0] == 4.0                   # Overridden

    def test_create_multi_agent_navigator(self):
        """Test creating multi-agent navigator with positions array."""
        positions = np.array([[0, 0], [10, 10], [20, 20]])
        orientations = [0.0, 45.0, 90.0]
        max_speeds = [1.0, 2.0, 3.0]
        
        navigator = create_navigator(
            positions=positions,
            orientations=orientations,
            max_speeds=max_speeds
        )
        
        assert isinstance(navigator, NavigatorProtocol)
        assert navigator.positions.shape == (3, 2)
        assert np.allclose(navigator.positions, positions)
        assert len(navigator.orientations) == 3
        assert navigator.orientations[1] == 45.0
        assert navigator.max_speeds[2] == 3.0

    def test_create_navigator_with_seed(self):
        """Test creating navigator with seed for reproducibility."""
        from {{cookiecutter.project_slug}}.utils.seed_manager import get_current_seed
        
        # Create navigator with seed
        navigator1 = create_navigator(
            position=(0.0, 0.0),
            orientation=0.0,
            seed=42
        )
        
        # Verify seed was set (if seed manager tracks it)
        # Note: This test depends on seed_manager implementation
        navigator2 = create_navigator(
            position=(0.0, 0.0),
            orientation=0.0,
            seed=42
        )
        
        # Both navigators should be created with same seed
        assert isinstance(navigator1, NavigatorProtocol)
        assert isinstance(navigator2, NavigatorProtocol)


class TestConfigurationValidation:
    """Test suite for configuration validation and error handling."""

    def test_validate_and_merge_config_basic(self):
        """Test basic configuration validation and merging."""
        config = {"orientation": 45.0, "speed": 1.0}
        overrides = {"max_speed": 2.0}
        
        result = _validate_and_merge_config(cfg=config, **overrides)
        
        assert result["orientation"] == 45.0
        assert result["speed"] == 1.0
        assert result["max_speed"] == 2.0

    def test_validate_and_merge_config_with_schema(self):
        """Test configuration validation with Pydantic schema."""
        config = {
            "position": [10.0, 20.0],
            "orientation": 45.0,
            "speed": 1.0,
            "max_speed": 2.0
        }
        
        result = _validate_and_merge_config(
            cfg=config,
            config_schema=SingleAgentConfig
        )
        
        assert result["position"] == [10.0, 20.0]
        assert result["orientation"] == 45.0
        assert result["speed"] == 1.0
        assert result["max_speed"] == 2.0

    def test_validate_and_merge_config_invalid_schema(self):
        """Test configuration validation with invalid data."""
        config = {
            "speed": -1.0,  # Invalid: speed should be >= 0
            "max_speed": 2.0
        }
        
        with pytest.raises(ConfigurationError):
            _validate_and_merge_config(
                cfg=config,
                config_schema=SingleAgentConfig
            )

    def test_validate_and_merge_config_override_precedence(self):
        """Test that direct parameters override config values."""
        config = {"orientation": 30.0, "speed": 1.0}
        
        result = _validate_and_merge_config(
            cfg=config,
            orientation=60.0,  # Should override config
            max_speed=3.0      # Should be added
        )
        
        assert result["orientation"] == 60.0  # Overridden
        assert result["speed"] == 1.0         # From config
        assert result["max_speed"] == 3.0     # Added


class TestPositionNormalization:
    """Test suite for position parameter normalization."""

    def test_normalize_positions_single_position(self):
        """Test normalizing single position parameter."""
        position = (10.0, 20.0)
        
        result_positions, is_multi_agent = _normalize_positions(position=position)
        
        assert result_positions.shape == (1, 2)
        assert np.allclose(result_positions[0], [10.0, 20.0])
        assert not is_multi_agent

    def test_normalize_positions_multiple_positions(self):
        """Test normalizing multiple positions parameter."""
        positions = [[0, 0], [10, 10], [20, 20]]
        
        result_positions, is_multi_agent = _normalize_positions(positions=positions)
        
        assert result_positions.shape == (3, 2)
        assert np.allclose(result_positions[0], [0, 0])
        assert np.allclose(result_positions[2], [20, 20])
        assert is_multi_agent

    def test_normalize_positions_single_as_positions(self):
        """Test normalizing single position provided as positions parameter."""
        positions = (10.0, 20.0)  # Single position as tuple
        
        result_positions, is_multi_agent = _normalize_positions(positions=positions)
        
        assert result_positions.shape == (1, 2)
        assert np.allclose(result_positions[0], [10.0, 20.0])
        assert not is_multi_agent

    def test_normalize_positions_both_provided(self):
        """Test error when both position and positions are provided."""
        with pytest.raises(ConfigurationError, match="Cannot specify both"):
            _normalize_positions(
                position=(0.0, 0.0),
                positions=[[10, 10], [20, 20]]
            )

    def test_normalize_positions_invalid_format(self):
        """Test error handling for invalid position formats."""
        with pytest.raises(ConfigurationError):
            _normalize_positions(position=(10.0,))  # Wrong dimension
        
        with pytest.raises(ConfigurationError):
            _normalize_positions(positions=[[10, 10, 10]])  # Wrong dimension


class TestFactoryErrorHandling:
    """Test suite for factory method error handling."""

    def test_create_navigator_from_config_invalid_type(self):
        """Test error handling for invalid configuration type in legacy method."""
        with pytest.raises(ConfigurationError):
            create_navigator_from_config("invalid_string_config")

    def test_create_navigator_configuration_error(self):
        """Test configuration error propagation in create_navigator."""
        with pytest.raises(ConfigurationError):
            create_navigator(speed=-1.0)  # Invalid speed

    def test_create_navigator_conflicting_parameters(self):
        """Test error when conflicting parameters are provided."""
        with pytest.raises(ConfigurationError):
            create_navigator(
                position=(0.0, 0.0),
                positions=[[10, 10], [20, 20]]
            )


class TestBackwardCompatibility:
    """Test suite for backward compatibility with legacy interfaces."""

    def test_legacy_factory_method_compatibility(self, config_files):
        """Test that legacy create_navigator_from_config works as expected."""
        config = config_files["default_config"]["navigator"]
        
        # Should work with both dict and DictConfig
        navigator1 = create_navigator_from_config(config)
        
        if HYDRA_AVAILABLE:
            hydra_config = OmegaConf.create(config)
            navigator2 = create_navigator_from_config(hydra_config)
            
            # Both should create equivalent navigators
            assert navigator1.orientations[0] == navigator2.orientations[0]
            assert navigator1.speeds[0] == navigator2.speeds[0]
            assert navigator1.max_speeds[0] == navigator2.max_speeds[0]

    def test_parameter_naming_compatibility(self):
        """Test that both old and new parameter names work."""
        # Test single-agent parameter names
        navigator = create_navigator(
            position=(10.0, 20.0),
            orientation=45.0,
            speed=1.0,
            max_speed=2.0
        )
        
        assert np.allclose(navigator.positions[0], [10.0, 20.0])
        assert navigator.orientations[0] == 45.0


class TestHydraIntegration:
    """Test suite for enhanced Hydra configuration integration."""

    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_hydra_dictconfig_processing(self):
        """Test processing of Hydra DictConfig objects."""
        # Create nested DictConfig
        cfg = OmegaConf.create({
            "navigator": {
                "position": [5.0, 15.0],
                "orientation": 30.0,
                "speed": 1.5,
                "max_speed": 3.0
            }
        })
        
        navigator = create_navigator(cfg=cfg.navigator)
        
        assert isinstance(navigator, NavigatorProtocol)
        assert np.allclose(navigator.positions[0], [5.0, 15.0])
        assert navigator.orientations[0] == 30.0

    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_hydra_interpolation_simulation(self):
        """Test simulation of Hydra variable interpolation."""
        # Simulate resolved Hydra config (interpolations already resolved)
        cfg = OmegaConf.create({
            "position": [0.0, 0.0],
            "orientation": 0.0,
            "speed": 1.0,
            "max_speed": 2.0  # This would be ${env:MAX_SPEED,2.0} before resolution
        })
        
        navigator = create_navigator(cfg=cfg)
        
        assert navigator.max_speeds[0] == 2.0

    def test_factory_method_parameter_merging(self):
        """Test comprehensive parameter merging scenarios."""
        base_config = {
            "position": [0.0, 0.0],
            "orientation": 0.0,
            "speed": 1.0,
            "max_speed": 2.0
        }
        
        # Test merging with various override patterns
        navigator = create_navigator(
            cfg=base_config,
            orientation=90.0,      # Override scalar
            position=(10.0, 20.0), # Override array
            angular_velocity=30.0  # Add new parameter
        )
        
        assert np.allclose(navigator.positions[0], [10.0, 20.0])
        assert navigator.orientations[0] == 90.0
        assert navigator.speeds[0] == 1.0  # From config
        assert navigator.max_speeds[0] == 2.0  # From config


class TestParameterValidation:
    """Test suite for enhanced parameter validation in factory methods."""

    def test_position_validation(self):
        """Test position parameter validation."""
        # Valid 2D position
        navigator = create_navigator(position=(10.0, 20.0))
        assert np.allclose(navigator.positions[0], [10.0, 20.0])
        
        # Invalid position dimensions
        with pytest.raises(ConfigurationError):
            create_navigator(position=(10.0,))  # 1D
        
        with pytest.raises(ConfigurationError):
            create_navigator(position=(10.0, 20.0, 30.0))  # 3D

    def test_speed_validation(self):
        """Test speed parameter validation."""
        # Valid speed
        navigator = create_navigator(speed=1.5)
        assert navigator.speeds[0] == 1.5
        
        # Invalid negative speed (should be caught by schema validation)
        with pytest.raises(ConfigurationError):
            create_navigator(speed=-1.0)

    def test_orientation_normalization(self):
        """Test orientation normalization to [0, 360) range."""
        # Test various orientation values
        navigator1 = create_navigator(orientation=450.0)  # > 360
        assert navigator1.orientations[0] == 90.0  # 450 % 360
        
        navigator2 = create_navigator(orientation=-90.0)  # Negative
        assert navigator2.orientations[0] == 270.0  # (-90 % 360)

    def test_multi_agent_parameter_consistency(self):
        """Test parameter consistency validation for multi-agent scenarios."""
        positions = [[0, 0], [10, 10], [20, 20]]
        orientations = [0.0, 45.0, 90.0]
        speeds = [1.0, 2.0, 3.0]
        
        navigator = create_navigator(
            positions=positions,
            orientations=orientations,
            speeds=speeds
        )
        
        assert navigator.positions.shape[0] == 3
        assert len(navigator.orientations) == 3
        assert len(navigator.speeds) == 3