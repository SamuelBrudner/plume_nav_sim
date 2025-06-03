"""Tests for the navigator factory module.

Updated for the new package structure with Hydra-based configuration management
and enhanced factory method patterns per Feature F-011.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
import numpy as np
from typing import Dict, Any

# Updated imports for new package structure
from {{cookiecutter.project_slug}}.api.navigation import create_navigator_from_config
from {{cookiecutter.project_slug}}.core.navigator import Navigator


@pytest.fixture
def base_hydra_config():
    """Fixture providing base Hydra configuration for testing."""
    return {
        "navigator": {
            "orientation": 0.0,
            "speed": 0.0,
            "max_speed": 1.0,
            "angular_velocity": 0.0
        }
    }


@pytest.fixture
def multi_agent_hydra_config():
    """Fixture providing multi-agent Hydra configuration for testing."""
    return {
        "navigator": {
            "positions": [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]],
            "orientations": [0.0, 45.0, 90.0],
            "speeds": [0.0, 0.5, 1.0],
            "max_speeds": [1.0, 1.5, 2.0],
            "angular_velocities": [0.0, 0.1, 0.2]
        }
    }


@pytest.fixture
def user_override_config():
    """Fixture providing user configuration overrides."""
    return {
        "navigator": {
            "orientation": 45.0,
            "speed": 0.5,
            "max_speed": 2.0,
            "angular_velocity": 0.1
        }
    }


@pytest.fixture
def invalid_config():
    """Fixture providing invalid configuration for validation testing."""
    return {
        "navigator": {
            "orientation": 0.0,
            "speed": 2.0,  # Invalid: exceeds max_speed
            "max_speed": 1.0,
            "angular_velocity": 0.0
        }
    }


@pytest.fixture
def merged_config():
    """Fixture providing configuration for testing parameter merging."""
    return {
        "navigator": {
            "orientation": 90.0,  # Override only orientation
            "speed": 0.0,         # Keep default
            "max_speed": 1.0,     # Keep default
            "angular_velocity": 0.0  # Keep default
        }
    }


class TestNavigatorFactoryBasicFunctionality:
    """Test basic navigator factory functionality with updated API."""

    def test_create_navigator_with_default_config(self, base_hydra_config):
        """Test creating a navigator with default Hydra configuration."""
        with patch('{{cookiecutter.project_slug}}.api.navigation.load_config', 
                   return_value=base_hydra_config):
            
            # Create a navigator with default config
            navigator = create_navigator_from_config()
            
            # Check that the navigator was created with default settings
            assert navigator.orientations[0] == 0.0
            assert navigator.speeds[0] == 0.0
            assert navigator.max_speeds[0] == 1.0
            assert navigator.angular_velocities[0] == 0.0

    def test_create_navigator_with_user_config(self, user_override_config):
        """Test creating a navigator with user configuration overrides."""
        with patch('{{cookiecutter.project_slug}}.api.navigation.load_config', 
                   return_value=user_override_config):
            
            # Create a navigator with user config
            navigator = create_navigator_from_config()
            
            # Check that the navigator was created with user settings
            assert navigator.orientations[0] == 45.0
            assert navigator.speeds[0] == 0.5
            assert navigator.max_speeds[0] == 2.0
            assert navigator.angular_velocities[0] == 0.1

    def test_create_navigator_with_merged_config(self, merged_config):
        """Test creating a navigator with merged configuration."""
        with patch('{{cookiecutter.project_slug}}.api.navigation.load_config', 
                   return_value=merged_config):
            
            # Create a navigator with merged config
            navigator = create_navigator_from_config()
            
            # Check that the navigator was created with merged settings
            assert navigator.orientations[0] == 90.0  # Overridden
            assert navigator.speeds[0] == 0.0         # Default
            assert navigator.max_speeds[0] == 1.0     # Default
            assert navigator.angular_velocities[0] == 0.0  # Default

    def test_create_navigator_with_additional_params(self, base_hydra_config):
        """Test creating a navigator with additional parameters."""
        with patch('{{cookiecutter.project_slug}}.api.navigation.load_config', 
                   return_value=base_hydra_config):
            
            # Create a navigator with default config but override some parameters
            navigator = create_navigator_from_config(orientation=180.0, speed=0.75)
            
            # Check that the navigator was created with overridden settings
            assert navigator.orientations[0] == 180.0  # Explicitly provided
            assert navigator.speeds[0] == 0.75         # Explicitly provided
            assert navigator.max_speeds[0] == 1.0      # From default config
            assert navigator.angular_velocities[0] == 0.0  # From default config


class TestHydraBasedFactoryPatterns:
    """Test enhanced Hydra-based factory method patterns per Feature F-011."""

    def test_hydra_configuration_loading_integration(self, base_hydra_config):
        """Test integration with Hydra configuration loading system."""
        # Mock Hydra's configuration loading mechanism
        with patch('{{cookiecutter.project_slug}}.api.navigation.load_config') as mock_load:
            mock_load.return_value = base_hydra_config
            
            # Test that the factory integrates with Hydra config loading
            navigator = create_navigator_from_config()
            
            # Verify Hydra configuration was loaded
            mock_load.assert_called_once()
            assert isinstance(navigator, Navigator)

    def test_hydra_configuration_override_patterns(self, base_hydra_config):
        """Test Hydra-style configuration override patterns."""
        # Test CLI-style parameter overrides (Hydra pattern)
        overrides = {
            "navigator.orientation": 45.0,
            "navigator.speed": 1.5,
            "navigator.max_speed": 3.0
        }
        
        with patch('{{cookiecutter.project_slug}}.api.navigation.load_config', 
                   return_value=base_hydra_config):
            with patch('{{cookiecutter.project_slug}}.api.navigation.apply_hydra_overrides') as mock_overrides:
                mock_overrides.return_value = {
                    "navigator": {
                        "orientation": 45.0,
                        "speed": 1.5,
                        "max_speed": 3.0,
                        "angular_velocity": 0.0
                    }
                }
                
                # Create navigator with Hydra-style overrides
                navigator = create_navigator_from_config(hydra_overrides=overrides)
                
                # Verify overrides were applied
                assert navigator.orientations[0] == 45.0
                assert navigator.speeds[0] == 1.5
                assert navigator.max_speeds[0] == 3.0

    def test_hydra_multirun_compatibility(self, base_hydra_config):
        """Test compatibility with Hydra multirun patterns."""
        # Test configuration sweeps (Hydra multirun pattern)
        sweep_configs = [
            {**base_hydra_config, "navigator": {**base_hydra_config["navigator"], "speed": 0.5}},
            {**base_hydra_config, "navigator": {**base_hydra_config["navigator"], "speed": 1.0}},
            {**base_hydra_config, "navigator": {**base_hydra_config["navigator"], "speed": 1.5}}
        ]
        
        navigators = []
        for config in sweep_configs:
            with patch('{{cookiecutter.project_slug}}.api.navigation.load_config', 
                       return_value=config):
                navigators.append(create_navigator_from_config())
        
        # Verify each navigator has different speed settings
        assert navigators[0].speeds[0] == 0.5
        assert navigators[1].speeds[0] == 1.0
        assert navigators[2].speeds[0] == 1.5

    def test_hydra_config_composition(self, base_hydra_config):
        """Test Hydra configuration composition patterns."""
        # Test composed configurations (Hydra composition pattern)
        composed_config = {
            **base_hydra_config,
            "navigator": {
                **base_hydra_config["navigator"],
                "composition": "single_agent",
                "experiment": "test_run"
            }
        }
        
        with patch('{{cookiecutter.project_slug}}.api.navigation.load_config', 
                   return_value=composed_config):
            navigator = create_navigator_from_config()
            
            # Verify composed configuration works
            assert isinstance(navigator, Navigator)
            assert navigator.is_single_agent


class TestMultiAgentFactoryPatterns:
    """Test multi-agent factory patterns with Hydra configuration."""

    def test_multi_agent_hydra_config_creation(self, multi_agent_hydra_config):
        """Test multi-agent navigator creation from Hydra configuration."""
        with patch('{{cookiecutter.project_slug}}.api.navigation.load_config', 
                   return_value=multi_agent_hydra_config):
            
            navigator = create_navigator_from_config()
            
            # Verify multi-agent navigator was created
            assert not navigator.is_single_agent
            assert navigator.num_agents == 3
            
            # Verify positions
            np.testing.assert_array_equal(navigator.positions, 
                                        np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]))
            
            # Verify orientations
            np.testing.assert_array_equal(navigator.orientations, 
                                        np.array([0.0, 45.0, 90.0]))
            
            # Verify speeds
            np.testing.assert_array_equal(navigator.speeds, 
                                        np.array([0.0, 0.5, 1.0]))

    def test_multi_agent_parameter_validation(self, multi_agent_hydra_config):
        """Test parameter validation for multi-agent configurations."""
        # Create invalid multi-agent config (mismatched array lengths)
        invalid_multi_config = {
            "navigator": {
                "positions": [[0.0, 0.0], [10.0, 10.0]],  # 2 agents
                "orientations": [0.0, 45.0, 90.0],        # 3 orientations (mismatch!)
                "speeds": [0.0, 0.5],                      # 2 speeds
                "max_speeds": [1.0, 1.5],                  # 2 max_speeds
                "angular_velocities": [0.0, 0.1]          # 2 angular velocities
            }
        }
        
        with patch('{{cookiecutter.project_slug}}.api.navigation.load_config', 
                   return_value=invalid_multi_config):
            with patch('{{cookiecutter.project_slug}}.api.navigation.validate_config') as mock_validate:
                mock_validate.side_effect = ValueError("Array length mismatch")
                
                # Test that validation error is raised
                with pytest.raises(ValueError, match="Array length mismatch"):
                    create_navigator_from_config(validate=True)


class TestConfigurationValidationAndMerging:
    """Test configuration validation and parameter merging in factory methods."""

    def test_configuration_validation_success(self, base_hydra_config):
        """Test successful configuration validation."""
        with patch('{{cookiecutter.project_slug}}.api.navigation.load_config', 
                   return_value=base_hydra_config):
            with patch('{{cookiecutter.project_slug}}.api.navigation.validate_config') as mock_validate:
                mock_validate.return_value = True
                
                # Test that validation passes for valid config
                navigator = create_navigator_from_config(validate=True)
                
                mock_validate.assert_called_once()
                assert isinstance(navigator, Navigator)

    def test_configuration_validation_failure(self, invalid_config):
        """Test configuration validation with invalid parameters."""
        with patch('{{cookiecutter.project_slug}}.api.navigation.load_config', 
                   return_value=invalid_config):
            with patch('{{cookiecutter.project_slug}}.api.navigation.validate_config') as mock_validate:
                mock_validate.side_effect = ValueError("speed cannot exceed max_speed")
                
                # Test that validation error is raised for invalid config
                with pytest.raises(ValueError, match="speed cannot exceed max_speed"):
                    create_navigator_from_config(validate=True)

    def test_parameter_merging_precedence(self, base_hydra_config):
        """Test parameter merging precedence in factory methods."""
        # Test that explicit parameters override configuration
        with patch('{{cookiecutter.project_slug}}.api.navigation.load_config', 
                   return_value=base_hydra_config):
            
            navigator = create_navigator_from_config(
                speed=2.0,      # Override config speed (0.0)
                max_speed=3.0,  # Override config max_speed (1.0)
                # orientation should remain from config (0.0)
            )
            
            # Verify precedence: explicit params > config
            assert navigator.speeds[0] == 2.0      # Explicit override
            assert navigator.max_speeds[0] == 3.0  # Explicit override
            assert navigator.orientations[0] == 0.0  # From config

    def test_deep_parameter_merging(self, base_hydra_config):
        """Test deep merging of nested configuration parameters."""
        # Test with nested parameter updates
        nested_updates = {
            "navigator": {
                "speed": 1.2,
                "angular_velocity": 0.15
                # Other parameters should remain from base config
            }
        }
        
        merged_config = {
            "navigator": {
                "orientation": 0.0,      # From base
                "speed": 1.2,            # From update
                "max_speed": 1.0,        # From base
                "angular_velocity": 0.15  # From update
            }
        }
        
        with patch('{{cookiecutter.project_slug}}.api.navigation.load_config', 
                   return_value=base_hydra_config):
            with patch('{{cookiecutter.project_slug}}.api.navigation.merge_configs', 
                       return_value=merged_config):
                
                navigator = create_navigator_from_config(config_updates=nested_updates)
                
                # Verify deep merging worked correctly
                assert navigator.orientations[0] == 0.0      # From base
                assert navigator.speeds[0] == 1.2            # From update
                assert navigator.max_speeds[0] == 1.0        # From base
                assert navigator.angular_velocities[0] == 0.15  # From update

    def test_config_path_resolution(self, base_hydra_config):
        """Test configuration file path resolution with Hydra patterns."""
        config_path = Path("conf/navigation/single_agent.yaml")
        
        with patch('{{cookiecutter.project_slug}}.api.navigation.load_config') as mock_load:
            mock_load.return_value = base_hydra_config
            
            # Test with explicit config path
            navigator = create_navigator_from_config(config_path=config_path)
            
            # Verify config path was passed to loader
            mock_load.assert_called_once_with(config_path)
            assert isinstance(navigator, Navigator)

    def test_environment_variable_integration(self, base_hydra_config):
        """Test integration with environment variable overrides."""
        # Test Hydra environment variable patterns
        with patch('{{cookiecutter.project_slug}}.api.navigation.load_config', 
                   return_value=base_hydra_config):
            with patch('{{cookiecutter.project_slug}}.api.navigation.resolve_env_vars') as mock_resolve:
                mock_resolve.return_value = {
                    "navigator": {
                        "orientation": 30.0,  # From env var
                        "speed": 0.0,
                        "max_speed": 1.0,
                        "angular_velocity": 0.0
                    }
                }
                
                navigator = create_navigator_from_config(resolve_env=True)
                
                # Verify environment variables were resolved
                mock_resolve.assert_called_once()
                assert navigator.orientations[0] == 30.0


class TestFactoryErrorHandling:
    """Test error handling in factory methods."""

    def test_missing_configuration_section(self):
        """Test handling of missing navigator configuration section."""
        config_without_navigator = {
            "video_plume": {
                "flip": False,
                "kernel_size": 0
            }
            # Missing navigator section
        }
        
        with patch('{{cookiecutter.project_slug}}.api.navigation.load_config', 
                   return_value=config_without_navigator):
            
            # Should create navigator with defaults when section is missing
            navigator = create_navigator_from_config()
            
            # Verify defaults were applied
            assert isinstance(navigator, Navigator)
            assert navigator.is_single_agent

    def test_config_loading_error(self):
        """Test handling of configuration loading errors."""
        with patch('{{cookiecutter.project_slug}}.api.navigation.load_config') as mock_load:
            mock_load.side_effect = FileNotFoundError("Configuration file not found")
            
            # Test that config loading error is properly handled
            with pytest.raises(FileNotFoundError, match="Configuration file not found"):
                create_navigator_from_config()

    def test_invalid_parameter_types(self, base_hydra_config):
        """Test handling of invalid parameter types."""
        with patch('{{cookiecutter.project_slug}}.api.navigation.load_config', 
                   return_value=base_hydra_config):
            
            # Test with invalid parameter types
            with pytest.raises(TypeError):
                create_navigator_from_config(
                    orientation="invalid_string",  # Should be float
                    speed=None,                     # Should be float
                )

    def test_configuration_schema_validation(self, base_hydra_config):
        """Test Pydantic schema validation integration."""
        with patch('{{cookiecutter.project_slug}}.api.navigation.load_config', 
                   return_value=base_hydra_config):
            with patch('{{cookiecutter.project_slug}}.config.schemas.NavigatorConfig') as mock_schema:
                # Configure mock to validate and return parsed data
                mock_instance = MagicMock()
                mock_instance.model_dump.return_value = base_hydra_config["navigator"]
                mock_schema.return_value = mock_instance
                
                navigator = create_navigator_from_config(validate_schema=True)
                
                # Verify schema validation was called
                mock_schema.assert_called_once()
                assert isinstance(navigator, Navigator)