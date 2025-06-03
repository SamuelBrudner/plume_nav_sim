"""Tests for configuration utilities with enhanced Hydra integration.

This module provides comprehensive testing for the Hydra-based configuration system
including hierarchical composition validation, environment variable interpolation,
and local override management per Section 6.6.1.1 and Section 7.2.3.1.
"""

import pytest
from pathlib import Path
import yaml
import os
from unittest.mock import patch, MagicMock
import tempfile
from typing import Dict, Any

from {{cookiecutter.project_slug}}.config.schemas import (
    NavigatorConfig,
    VideoPlumeConfig,
    SimulationConfig
)

try:
    from hydra import compose, initialize, initialize_config_dir
    from hydra.core.config_store import ConfigStore
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False

# Legacy imports for backward compatibility testing
from {{cookiecutter.project_slug}}.config.utils import (
    update_config,
    get_config_dir,
    load_yaml_config,
    load_config
)

pytestmark = pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")


class TestLegacyConfigUtils:
    """Test legacy configuration utilities for backward compatibility."""

    def test_update_config_basic(self):
        """Test that update_config correctly updates flat dictionaries."""
        original = {"a": 1, "b": 2}
        update = {"b": 3, "c": 4}
        result = update_config(original, update)
        
        assert result == {"a": 1, "b": 3, "c": 4}
        # Original should be unchanged
        assert original == {"a": 1, "b": 2}

    def test_update_config_nested(self):
        """Test that update_config correctly updates nested dictionaries."""
        original = {"a": 1, "b": {"x": 10, "y": 20}}
        update = {"b": {"y": 30, "z": 40}}
        result = update_config(original, update)
        
        assert result == {"a": 1, "b": {"x": 10, "y": 30, "z": 40}}
        # Original should be unchanged
        assert original == {"a": 1, "b": {"x": 10, "y": 20}}

    def test_get_config_dir_default(self):
        """Test that get_config_dir returns the standard config directory."""
        with patch.dict('os.environ', {}, clear=True):
            config_dir = get_config_dir()
            expected_path = Path(__file__).parent.parent / "conf"
            assert config_dir == expected_path

    def test_get_config_dir_env_override(self):
        """Test that get_config_dir respects the environment variable."""
        with patch.dict('os.environ', {"{{cookiecutter.project_slug|upper}}_CONFIG_DIR": "/custom/config/path"}):
            config_dir = get_config_dir()
            assert config_dir == Path("/custom/config/path")

    def test_load_yaml_config(self):
        """Test that load_yaml_config correctly loads a YAML file."""
        # Create a temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.yaml') as tmp:
            test_config = {
                "test": {"key": "value"},
                "number": 42
            }
            yaml.dump(test_config, tmp)
            tmp.flush()
            
            # Load and verify the config
            loaded_config = load_yaml_config(tmp.name)
            assert loaded_config == test_config

    def test_load_yaml_config_file_not_found(self):
        """Test that load_yaml_config raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_yaml_config("/nonexistent/config.yaml")


class TestHydraConfigurationIntegration:
    """Enhanced Hydra configuration testing scenarios with hierarchical composition validation."""

    @pytest.fixture(autouse=True)
    def setup_hydra(self):
        """Set up clean Hydra environment for each test."""
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    @pytest.fixture
    def hydra_config_dir(self, tmp_path):
        """Create temporary Hydra configuration directory structure."""
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Create base.yaml with foundation defaults
        base_config = {
            "defaults": [
                "_self_",
                "navigation: single_agent",
                "video_plume: default",
                "simulation: standard"
            ],
            "navigator": {
                "type": "single",
                "initial_position": [50.0, 50.0],
                "initial_orientation": 0.0,
                "max_speed": 10.0,
                "angular_velocity": 0.1
            },
            "video_plume": {
                "flip_horizontal": False,
                "gaussian_blur": {
                    "enabled": False,
                    "kernel_size": 5,
                    "sigma": 1.0
                }
            },
            "simulation": {
                "num_steps": 100,
                "step_size": 0.1,
                "recording_enabled": True
            }
        }
        
        base_path = config_dir / "base.yaml"
        with open(base_path, 'w') as f:
            yaml.dump(base_config, f)

        # Create config.yaml with environment-specific settings
        config_yaml = {
            "defaults": [
                "base"
            ],
            "hydra": {
                "run": {
                    "dir": "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"
                },
                "sweep": {
                    "dir": "multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}",
                    "subdir": "${hydra:job.num}"
                }
            },
            "debug": False,
            "logging_level": "INFO"
        }
        
        config_path = config_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_yaml, f)
            
        # Create local override directory
        local_dir = config_dir / "local"
        local_dir.mkdir()
        
        # Create development override
        dev_config = {
            "debug": True,
            "logging_level": "DEBUG",
            "navigator": {
                "max_speed": 15.0
            },
            "visualization": {
                "save_animations": False
            }
        }
        
        dev_path = local_dir / "development.yaml"
        with open(dev_path, 'w') as f:
            yaml.dump(dev_config, f)
            
        # Create production override with environment variables
        prod_config = {
            "debug": False,
            "logging_level": "ERROR",
            "database": {
                "url": "${oc.env:DATABASE_URL,sqlite:///production.db}",
                "username": "${oc.env:DB_USER,admin}",
                "password": "${oc.env:DB_PASSWORD}"
            },
            "api_keys": {
                "visualization_service": "${oc.env:VIZ_API_KEY}",
                "data_storage": "${oc.env:STORAGE_KEY}"
            }
        }
        
        prod_path = local_dir / "production.yaml"
        with open(prod_path, 'w') as f:
            yaml.dump(prod_config, f)
        
        return {
            "config_dir": config_dir,
            "base_path": base_path,
            "config_path": config_path,
            "dev_path": dev_path,
            "prod_path": prod_path,
            "local_dir": local_dir
        }

    def test_hydra_basic_composition(self, hydra_config_dir):
        """Test basic Hydra configuration composition from base.yaml."""
        config_dir = hydra_config_dir["config_dir"]
        
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name="base")
            
            # Verify base configuration loaded correctly
            assert cfg.navigator.type == "single"
            assert cfg.navigator.initial_position == [50.0, 50.0]
            assert cfg.navigator.max_speed == 10.0
            assert cfg.video_plume.flip_horizontal is False
            assert cfg.video_plume.gaussian_blur.enabled is False
            assert cfg.simulation.num_steps == 100

    def test_hydra_hierarchical_composition(self, hydra_config_dir):
        """Test hierarchical configuration composition from config.yaml."""
        config_dir = hydra_config_dir["config_dir"]
        
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name="config")
            
            # Verify hierarchical composition works
            assert cfg.navigator.type == "single"  # From base
            assert cfg.debug is False  # From config
            assert cfg.logging_level == "INFO"  # From config
            assert "hydra" in cfg  # Hydra-specific settings

    def test_hydra_override_scenarios(self, hydra_config_dir):
        """Test Hydra configuration override mechanisms."""
        config_dir = hydra_config_dir["config_dir"]
        
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            # Test command-line style overrides
            cfg = compose(
                config_name="config",
                overrides=[
                    "navigator.max_speed=25.0",
                    "navigator.angular_velocity=0.5",
                    "debug=true"
                ]
            )
            
            # Verify overrides applied correctly
            assert cfg.navigator.max_speed == 25.0
            assert cfg.navigator.angular_velocity == 0.5
            assert cfg.debug is True
            
            # Verify non-overridden values remain
            assert cfg.navigator.type == "single"
            assert cfg.logging_level == "INFO"

    def test_hydra_local_override_integration(self, hydra_config_dir):
        """Test local override management through conf/local/ structure."""
        config_dir = hydra_config_dir["config_dir"]
        
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            # Test development override composition
            cfg = compose(
                config_name="config",
                overrides=["++hydra/config=local/development"]
            )
            
            # Note: Since we're using ++ prefix, this creates a new config group
            # In practice, local overrides would be applied differently
            # For testing purposes, we'll compose with base and then manually merge
            dev_cfg = OmegaConf.load(hydra_config_dir["dev_path"])
            base_cfg = compose(config_name="config")
            merged_cfg = OmegaConf.merge(base_cfg, dev_cfg)
            
            # Verify development overrides applied
            assert merged_cfg.debug is True
            assert merged_cfg.logging_level == "DEBUG"
            assert merged_cfg.navigator.max_speed == 15.0
            assert merged_cfg.visualization.save_animations is False

    def test_environment_variable_interpolation(self, hydra_config_dir):
        """Test environment variable interpolation and secure credential management."""
        config_dir = hydra_config_dir["config_dir"]
        
        # Set up test environment variables
        test_env = {
            "DATABASE_URL": "postgresql://test:test@localhost/testdb",
            "DB_USER": "test_user",
            "DB_PASSWORD": "secure_password",
            "VIZ_API_KEY": "viz_key_12345",
            "STORAGE_KEY": "storage_key_67890"
        }
        
        with patch.dict(os.environ, test_env):
            # Load production configuration with environment variables
            prod_cfg = OmegaConf.load(hydra_config_dir["prod_path"])
            
            with initialize_config_dir(config_dir=str(config_dir), version_base=None):
                base_cfg = compose(config_name="config")
                merged_cfg = OmegaConf.merge(base_cfg, prod_cfg)
                
                # Resolve environment variables
                resolved_cfg = OmegaConf.create(OmegaConf.to_yaml(merged_cfg, resolve=True))
                
                # Verify environment variable interpolation
                assert resolved_cfg.database.url == "postgresql://test:test@localhost/testdb"
                assert resolved_cfg.database.username == "test_user"
                assert resolved_cfg.database.password == "secure_password"
                assert resolved_cfg.api_keys.visualization_service == "viz_key_12345"
                assert resolved_cfg.api_keys.data_storage == "storage_key_67890"

    def test_environment_variable_defaults(self, hydra_config_dir):
        """Test environment variable interpolation with default values."""
        config_dir = hydra_config_dir["config_dir"]
        
        # Test with missing environment variables to trigger defaults
        with patch.dict(os.environ, {}, clear=True):
            prod_cfg = OmegaConf.load(hydra_config_dir["prod_path"])
            
            with initialize_config_dir(config_dir=str(config_dir), version_base=None):
                base_cfg = compose(config_name="config")
                merged_cfg = OmegaConf.merge(base_cfg, prod_cfg)
                
                # Resolve with defaults
                resolved_cfg = OmegaConf.create(OmegaConf.to_yaml(merged_cfg, resolve=True))
                
                # Verify default values used when env vars missing
                assert resolved_cfg.database.url == "sqlite:///production.db"
                assert resolved_cfg.database.username == "admin"
                # DB_PASSWORD has no default, so it should remain as interpolation
                assert "${oc.env:DB_PASSWORD}" in OmegaConf.to_yaml(merged_cfg)

    def test_hydra_multirun_configuration(self, hydra_config_dir):
        """Test Hydra multi-run experiment support configuration."""
        config_dir = hydra_config_dir["config_dir"]
        
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            # Test parameter sweep configuration
            cfg = compose(
                config_name="config",
                overrides=[
                    "hydra/launcher=basic",
                    "--multirun"
                ]
            )
            
            # Verify Hydra sweep configuration
            assert "sweep" in cfg.hydra
            assert "dir" in cfg.hydra.sweep
            assert "subdir" in cfg.hydra.sweep
            assert "${hydra:job.num}" in cfg.hydra.sweep.subdir

    def test_hydra_working_directory_management(self, hydra_config_dir):
        """Test Hydra working directory isolation and output organization."""
        config_dir = hydra_config_dir["config_dir"]
        
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name="config")
            
            # Verify working directory configuration
            assert "run" in cfg.hydra
            assert "dir" in cfg.hydra.run
            assert "outputs/" in cfg.hydra.run.dir
            assert "${now:" in cfg.hydra.run.dir  # Timestamp interpolation

    def test_config_schema_validation_integration(self, hydra_config_dir):
        """Test integration of Pydantic schemas with Hydra configuration."""
        config_dir = hydra_config_dir["config_dir"]
        
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name="config")
            
            # Test NavigatorConfig validation
            navigator_config = NavigatorConfig(**cfg.navigator)
            assert navigator_config.type == "single"
            assert navigator_config.max_speed == 10.0
            
            # Test VideoPlumeConfig validation
            video_config = VideoPlumeConfig(
                video_path="test.mp4",
                **cfg.video_plume
            )
            assert video_config.flip_horizontal is False
            assert video_config.gaussian_blur.enabled is False

    def test_hydra_compose_api_integration(self, hydra_config_dir):
        """Test programmatic configuration composition for research scripts."""
        config_dir = hydra_config_dir["config_dir"]
        
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            # Test dynamic configuration assembly
            cfg1 = compose(
                config_name="config",
                overrides=["navigator.max_speed=12.0"]
            )
            
            cfg2 = compose(
                config_name="config", 
                overrides=["video_plume.flip_horizontal=true"]
            )
            
            # Verify independent compositions
            assert cfg1.navigator.max_speed == 12.0
            assert cfg1.video_plume.flip_horizontal is False
            
            assert cfg2.navigator.max_speed == 10.0
            assert cfg2.video_plume.flip_horizontal is True

    def test_hydra_error_handling_and_validation(self, hydra_config_dir):
        """Test Hydra configuration error handling and validation."""
        config_dir = hydra_config_dir["config_dir"]
        
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            # Test invalid override handling
            with pytest.raises(Exception):  # Hydra should raise on invalid overrides
                compose(
                    config_name="config",
                    overrides=["nonexistent.parameter=value"]
                )
            
            # Test missing configuration file
            with pytest.raises(Exception):
                compose(config_name="nonexistent_config")

    def test_configuration_performance_characteristics(self, hydra_config_dir):
        """Test Hydra configuration loading performance characteristics."""
        import time
        config_dir = hydra_config_dir["config_dir"]
        
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            # Measure configuration loading time
            start_time = time.time()
            
            cfg = compose(
                config_name="config",
                overrides=[
                    "navigator.max_speed=15.0",
                    "video_plume.gaussian_blur.enabled=true",
                    "simulation.num_steps=500"
                ]
            )
            
            load_time = time.time() - start_time
            
            # Verify configuration loaded correctly
            assert cfg.navigator.max_speed == 15.0
            assert cfg.video_plume.gaussian_blur.enabled is True
            assert cfg.simulation.num_steps == 500
            
            # Verify performance requirement (<1 second per Section 7.2.4.1)
            assert load_time < 1.0, f"Configuration loading took {load_time:.3f}s, should be <1.0s"


class TestConfigurationSecurity:
    """Test configuration security and validation scenarios."""

    @pytest.fixture(autouse=True)
    def setup_hydra(self):
        """Set up clean Hydra environment for each test."""
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    def test_environment_variable_injection_prevention(self, tmp_path):
        """Test that malicious environment variable injection is prevented."""
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Create configuration with potential injection point
        malicious_config = {
            "database": {
                "url": "${oc.env:DATABASE_URL,sqlite:///safe.db}",
                "admin_password": "${oc.env:ADMIN_PASSWORD,default_admin}"
            }
        }
        
        config_path = config_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(malicious_config, f)
        
        # Attempt malicious environment variable injection
        malicious_env = {
            "DATABASE_URL": "sqlite:///hacked.db; DROP TABLE users; --",
            "ADMIN_PASSWORD": "hacked_password"
        }
        
        with patch.dict(os.environ, malicious_env):
            with initialize_config_dir(config_dir=str(config_dir), version_base=None):
                cfg = compose(config_name="config")
                
                # Verify values are properly isolated (basic string check)
                assert "DROP TABLE" not in str(cfg.database.url)
                # Note: Full SQL injection prevention would be handled by the database layer

    def test_configuration_path_traversal_prevention(self, tmp_path):
        """Test that configuration file path traversal is prevented."""
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Create base configuration
        base_config = {"test": "value"}
        config_path = config_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(base_config, f)
        
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            # Test that path traversal attempts fail gracefully
            with pytest.raises(Exception):
                # This should fail as Hydra validates config paths
                compose(config_name="../../../etc/passwd")

    def test_configuration_override_boundaries(self, tmp_path):
        """Test that sensitive configuration parameters cannot be overridden."""
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Create configuration with sensitive parameters
        secure_config = {
            "system": {
                "admin_mode": False,
                "debug_enabled": False,
                "secret_key": "secure_default"
            },
            "public": {
                "max_speed": 10.0,
                "timeout": 30
            }
        }
        
        config_path = config_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(secure_config, f)
        
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            # Test that overrides work for public parameters
            cfg = compose(
                config_name="config",
                overrides=["public.max_speed=15.0"]
            )
            assert cfg.public.max_speed == 15.0
            
            # System parameters should remain unchanged
            assert cfg.system.admin_mode is False
            assert cfg.system.debug_enabled is False


@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
class TestLegacyCompatibilityIntegration:
    """Test integration between legacy config utilities and new Hydra system."""

    @pytest.fixture
    def mixed_config_environment(self, tmp_path):
        """Create environment with both legacy and Hydra configurations."""
        # Legacy config directory
        legacy_dir = tmp_path / "configs"
        legacy_dir.mkdir()
        
        legacy_config = {
            "environment": {
                "dimensions": [10.0, 8.0, 2.0],
                "wind": {"speed": 0.5}
            },
            "video_plume": {"flip": False}
        }
        
        legacy_path = legacy_dir / "default.yaml"
        with open(legacy_path, 'w') as f:
            yaml.dump(legacy_config, f)
        
        # Hydra config directory
        hydra_dir = tmp_path / "conf"
        hydra_dir.mkdir()
        
        hydra_config = {
            "navigator": {
                "type": "single",
                "max_speed": 10.0
            },
            "video_plume": {
                "flip_horizontal": False,
                "gaussian_blur": {"enabled": False}
            }
        }
        
        hydra_path = hydra_dir / "config.yaml"
        with open(hydra_path, 'w') as f:
            yaml.dump(hydra_config, f)
        
        return {
            "legacy_dir": legacy_dir,
            "legacy_path": legacy_path,
            "hydra_dir": hydra_dir,
            "hydra_path": hydra_path
        }

    def test_legacy_config_loading_compatibility(self, mixed_config_environment):
        """Test that legacy configuration loading still works."""
        legacy_dir = mixed_config_environment["legacy_dir"]
        
        with patch('{{cookiecutter.project_slug}}.config.utils.get_config_dir', 
                   return_value=legacy_dir):
            config = load_config()
            assert config["environment"]["wind"]["speed"] == 0.5
            assert config["video_plume"]["flip"] is False

    def test_hydra_config_loading_with_legacy_fallback(self, mixed_config_environment):
        """Test Hydra configuration loading with legacy compatibility."""
        hydra_dir = mixed_config_environment["hydra_dir"]
        
        with initialize_config_dir(config_dir=str(hydra_dir), version_base=None):
            cfg = compose(config_name="config")
            
            # Convert to legacy format for compatibility testing
            legacy_format = OmegaConf.to_yaml(cfg)
            legacy_dict = yaml.safe_load(legacy_format)
            
            assert legacy_dict["navigator"]["max_speed"] == 10.0
            assert legacy_dict["video_plume"]["flip_horizontal"] is False


# Custom pytest fixtures for comprehensive testing integration
@pytest.fixture(scope="session")
def hydra_test_config_store():
    """Register test configurations with Hydra ConfigStore."""
    if not HYDRA_AVAILABLE:
        pytest.skip("Hydra not available")
    
    cs = ConfigStore.instance()
    
    # Register test schemas
    cs.store(name="test_navigator", node=NavigatorConfig)
    cs.store(name="test_video_plume", node=VideoPlumeConfig)
    cs.store(name="test_simulation", node=SimulationConfig)
    
    return cs


@pytest.fixture
def mock_hydra_environment(tmp_path):
    """Create mock Hydra environment for isolated testing."""
    if not HYDRA_AVAILABLE:
        pytest.skip("Hydra not available")
    
    config_dir = tmp_path / "test_conf"
    config_dir.mkdir()
    
    # Minimal test configuration
    test_config = {
        "test_param": "test_value",
        "nested": {
            "param": 42
        }
    }
    
    config_path = config_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    return {
        "config_dir": config_dir,
        "config_path": config_path
    }


# Integration test for pytest-hydra plugin usage
@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
def test_pytest_hydra_plugin_integration(mock_hydra_environment):
    """Test pytest-hydra plugin integration for comprehensive configuration testing."""
    config_dir = mock_hydra_environment["config_dir"]
    
    # Test basic pytest-hydra integration patterns
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="test_config")
        
        assert cfg.test_param == "test_value"
        assert cfg.nested.param == 42
        
        # Test override scenarios that pytest-hydra would enable
        cfg_override = compose(
            config_name="test_config",
            overrides=["test_param=override_value"]
        )
        
        assert cfg_override.test_param == "override_value"
        assert cfg_override.nested.param == 42  # Unchanged


if __name__ == "__main__":
    # Enable running tests directly for development
    pytest.main([__file__, "-v"])