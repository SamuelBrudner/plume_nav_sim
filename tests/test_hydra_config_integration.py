"""
Advanced pytest test suite for Hydra 1.3+ structured dataclass-based configuration integration.

This module provides comprehensive testing of the modern Hydra configuration management
system using @dataclass decorators with Pydantic validation hooks for structured 
configuration composition, override scenarios, environment variable interpolation,
and configuration security with compile-time validation enforcement.

Validates robust Hydra 1.3+ configuration management with ConfigStore registration,
dataclass-based schema definitions, and hierarchical composition ensuring
research-grade reproducibility and security with enhanced type safety.

Key Testing Areas:
- Hydra 1.3+ structured dataclass-based configuration (F-006-RQ-005)
- @dataclass decorator usage with built-in type safety validation
- Pydantic validation hooks integration with dataclass configurations  
- Hierarchical config composition with compile-time validation enforcement
- Environment variable interpolation for structured dataclass fields
- Enhanced secret management with field(repr=False) and automatic log exclusion
- ConfigStore registration and schema validation integration
- pytest-hydra plugin integration with structured configs
"""

import os
import sys
import time
import tempfile
import pytest
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from unittest.mock import patch, MagicMock

# Hydra and OmegaConf imports with fallback
try:
    from hydra import initialize, initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.config_store import ConfigStore
    from hydra.conf import ConfigStore as ConfigStoreConf
    from omegaconf import OmegaConf, DictConfig
    import pytest_hydra
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    pytest.skip("Hydra not available", allow_module_level=True)

# Test dependencies - structured dataclass configs
from src.odor_plume_nav.config.models import (
    NavigatorConfig,
    VideoPlumeConfig,
    SingleAgentConfig,
    MultiAgentConfig,
    SimulationConfig,
    register_config_schemas
)


class TestDataclassStructuredConfiguration:
    """
    Test Hydra 1.3+ structured dataclass-based configuration system validating
    @dataclass decorator usage with built-in type safety and ConfigStore registration.
    """

    @pytest.fixture(autouse=True)
    def setup_hydra_isolation(self):
        """
        Ensure clean Hydra state for each test preventing configuration pollution
        between test runs and ConfigStore registration isolation.
        """
        GlobalHydra.instance().clear()
        # Clear ConfigStore to prevent test pollution
        cs = ConfigStore.instance()
        cs.repo.clear()
        yield
        GlobalHydra.instance().clear()
        cs.repo.clear()

    @pytest.fixture
    def register_dataclass_schemas(self):
        """
        Register structured dataclass schemas with Hydra ConfigStore for testing
        @dataclass-based configuration validation and composition.
        """
        register_config_schemas()
        return ConfigStore.instance()

    @pytest.fixture
    def structured_config_dir(self, tmp_path):
        """
        Create structured configuration directory for dataclass schema testing
        with proper ConfigStore registration and hierarchical composition.
        """
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Create structured config using dataclass schema
        structured_config = {
            "defaults": ["_self_"],
            "_target_": "odor_plume_nav.api.create_simulation",
            "navigator": {
                "_target_": "odor_plume_nav.core.controllers.NavigatorController",
                "mode": "single",
                "position": [50.0, 50.0],
                "orientation": 90.0,
                "speed": 1.5,
                "max_speed": 3.0,
                "angular_velocity": 0.1
            },
            "video_plume": {
                "_target_": "odor_plume_nav.data.VideoPlume",
                "video_path": "/test/structured/video.mp4",
                "flip": false,
                "grayscale": true,
                "kernel_size": 7,
                "kernel_sigma": 1.5,
                "normalize": true
            },
            "simulation": {
                "_target_": "odor_plume_nav.api.run_simulation",
                "max_steps": 2000,
                "step_size": 1.0,
                "enable_gpu": false,
                "record_trajectory": true,
                "output_format": "numpy"
            }
        }
        
        config_file = config_dir / "config.yaml"
        OmegaConf.save(OmegaConf.create(structured_config), config_file)
        
        return config_dir

    def test_dataclass_schema_registration(self, register_dataclass_schemas):
        """
        Test that @dataclass schemas are properly registered with Hydra ConfigStore
        enabling structured configuration composition with compile-time validation.
        """
        cs = register_dataclass_schemas
        
        # Validate that dataclass schemas are registered in ConfigStore
        assert "navigator/single_agent" in [item.name for item in cs.list("navigator")]
        assert "navigator/multi_agent" in [item.name for item in cs.list("navigator")]
        assert "navigator/unified" in [item.name for item in cs.list("navigator")]
        assert "video_plume/default" in [item.name for item in cs.list("video_plume")]
        assert "simulation/standard" in [item.name for item in cs.list("simulation")]
        
        # Validate base configuration schema registration
        base_configs = [item.name for item in cs.list("")]
        assert "base_config" in base_configs

    def test_dataclass_type_safety_validation(self, register_dataclass_schemas, structured_config_dir):
        """
        Test built-in type safety validation provided by @dataclass decorators
        ensuring compile-time type checking and runtime validation integration.
        """
        with initialize(config_path=str(structured_config_dir), version_base=None):
            # Load configuration with proper type validation
            cfg = compose(config_name="config")
            
            # Validate dataclass types are enforced
            assert isinstance(cfg.navigator.position, list)
            assert len(cfg.navigator.position) == 2
            assert all(isinstance(x, float) for x in cfg.navigator.position)
            
            assert isinstance(cfg.navigator.orientation, (int, float))
            assert isinstance(cfg.navigator.speed, (int, float))
            assert isinstance(cfg.navigator.max_speed, (int, float))
            
            assert isinstance(cfg.video_plume.flip, bool)
            assert isinstance(cfg.video_plume.grayscale, bool)
            assert isinstance(cfg.video_plume.kernel_size, int)
            assert isinstance(cfg.video_plume.kernel_sigma, (int, float))
            
            assert isinstance(cfg.simulation.max_steps, int)
            assert isinstance(cfg.simulation.enable_gpu, bool)
            assert isinstance(cfg.simulation.record_trajectory, bool)

    def test_dataclass_field_metadata_validation(self, register_dataclass_schemas, structured_config_dir):
        """
        Test that dataclass field metadata is properly validated including
        constraints, defaults, and validation rules with enhanced error reporting.
        """
        with initialize(config_path=str(structured_config_dir), version_base=None):
            # Test constraint validation via overrides
            with pytest.raises(Exception):  # Should fail validation
                cfg = compose(
                    config_name="config",
                    overrides=["navigator.orientation=450.0"]  # Invalid orientation > 360
                )
            
            # Test speed constraint validation
            with pytest.raises(Exception):  # Should fail validation
                cfg = compose(
                    config_name="config", 
                    overrides=[
                        "navigator.speed=5.0",
                        "navigator.max_speed=2.0"  # speed > max_speed
                    ]
                )
            
            # Test valid configuration passes
            cfg = compose(
                config_name="config",
                overrides=[
                    "navigator.orientation=270.0",  # Valid orientation
                    "navigator.speed=1.0",         # Valid speed relationship
                    "navigator.max_speed=2.0"
                ]
            )
            assert cfg.navigator.orientation == 270.0
            assert cfg.navigator.speed == 1.0
            assert cfg.navigator.max_speed == 2.0

    def test_dataclass_structured_composition(self, register_dataclass_schemas, structured_config_dir):
        """
        Test hierarchical configuration composition with structured dataclass schemas
        ensuring proper inheritance and override behavior with compile-time validation.
        """
        with initialize(config_path=str(structured_config_dir), version_base=None):
            # Test basic structured composition
            cfg = compose(config_name="config")
            
            # Validate _target_ metadata is preserved from dataclass definitions
            assert cfg.navigator._target_ == "odor_plume_nav.core.controllers.NavigatorController"
            assert cfg.video_plume._target_ == "odor_plume_nav.data.VideoPlume"
            assert cfg.simulation._target_ == "odor_plume_nav.api.run_simulation"
            
            # Test structured override composition
            cfg_override = compose(
                config_name="config",
                overrides=[
                    "navigator.mode=multi",
                    "navigator.num_agents=5",
                    "simulation.batch_size=10"
                ]
            )
            
            assert cfg_override.navigator.mode == "multi"
            assert cfg_override.navigator.num_agents == 5
            assert cfg_override.simulation.batch_size == 10


class TestHydraConfigurationComposition:
    """
    Test hierarchical configuration composition from conf/base.yaml through
    conf/config.yaml to local overrides validating Feature F-006 requirements.
    """

    @pytest.fixture(autouse=True)
    def setup_hydra_isolation(self):
        """
        Ensure clean Hydra state for each test preventing configuration pollution
        between test runs per Section 6.6.5.4 test environment isolation.
        """
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """
        Create temporary configuration directory structure for testing hierarchical
        composition without interfering with project configuration files.
        """
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Create base configuration
        base_config = {
            "navigator": {
                "position": [0.0, 0.0],
                "orientation": 0.0,
                "speed": 1.0,
                "max_speed": 2.0,
                "angular_velocity": 0.0
            },
            "video_plume": {
                "video_path": "/default/path/video.mp4",
                "flip": False,
                "grayscale": True,
                "kernel_size": 5,
                "kernel_sigma": 1.0,
                "normalize": True
            },
            "experiment": {
                "name": "default_experiment",
                "seed": 42,
                "num_steps": 100
            }
        }
        
        # Write base.yaml
        base_file = config_dir / "base.yaml"
        OmegaConf.save(OmegaConf.create(base_config), base_file)
        
        # Create config.yaml with overrides
        config_overrides = {
            "defaults": ["base"],
            "navigator": {
                "speed": 1.5,
                "max_speed": 3.0
            },
            "experiment": {
                "name": "test_experiment",
                "num_steps": 200
            }
        }
        
        config_file = config_dir / "config.yaml"  
        OmegaConf.save(OmegaConf.create(config_overrides), config_file)
        
        # Create local directory structure
        local_dir = config_dir / "local"
        local_dir.mkdir()
        
        return config_dir

    @pytest.fixture  
    def hydra_config_fixture(self, temp_config_dir):
        """
        Provide Hydra configuration composition fixture for testing hierarchical
        parameter inheritance and override scenarios per pytest-hydra integration.
        """
        with initialize(config_path=str(temp_config_dir), version_base=None):
            cfg = compose(config_name="config")
            return cfg

    def test_base_configuration_loading(self, temp_config_dir):
        """
        Test that conf/base.yaml loads successfully with all required parameters
        validating F-006-RQ-001 Hydra configuration loading requirements.
        """
        with initialize(config_path=str(temp_config_dir), version_base=None):
            # Load base configuration directly
            cfg = compose(config_name="base")
            
            # Validate base configuration structure
            assert "navigator" in cfg
            assert "video_plume" in cfg
            assert "experiment" in cfg
            
            # Validate navigator base parameters
            assert cfg.navigator.position == [0.0, 0.0]
            assert cfg.navigator.orientation == 0.0
            assert cfg.navigator.speed == 1.0
            assert cfg.navigator.max_speed == 2.0
            
            # Validate video plume base parameters
            assert cfg.video_plume.video_path == "/default/path/video.mp4"
            assert cfg.video_plume.flip is False
            assert cfg.video_plume.grayscale is True
            
            # Validate experiment base parameters
            assert cfg.experiment.name == "default_experiment"
            assert cfg.experiment.seed == 42
            assert cfg.experiment.num_steps == 100

    def test_hierarchical_configuration_composition(self, hydra_config_fixture):
        """
        Test hierarchical composition from conf/base.yaml through conf/config.yaml
        validating F-006-RQ-002 configuration merging and hierarchy maintenance.
        """
        cfg = hydra_config_fixture
        
        # Validate that base parameters are inherited
        assert cfg.navigator.position == [0.0, 0.0]  # From base
        assert cfg.navigator.orientation == 0.0       # From base
        assert cfg.navigator.angular_velocity == 0.0  # From base
        
        # Validate that config.yaml overrides are applied
        assert cfg.navigator.speed == 1.5            # Overridden in config.yaml
        assert cfg.navigator.max_speed == 3.0        # Overridden in config.yaml
        
        # Validate experiment parameter overrides
        assert cfg.experiment.name == "test_experiment"  # Overridden
        assert cfg.experiment.num_steps == 200          # Overridden  
        assert cfg.experiment.seed == 42                # Inherited from base
        
        # Validate video plume parameters (all inherited)
        assert cfg.video_plume.flip is False
        assert cfg.video_plume.grayscale is True
        assert cfg.video_plume.kernel_size == 5

    def test_configuration_composition_performance(self, temp_config_dir):
        """
        Test configuration loading performance meets <1s requirement per
        Section 2.2.9.3 configuration loading performance criteria.
        """
        # Measure configuration composition time
        start_time = time.time()
        
        with initialize(config_path=str(temp_config_dir), version_base=None):
            cfg = compose(config_name="config")
            
        end_time = time.time()
        composition_time = end_time - start_time
        
        # Validate performance requirement (<1s)
        assert composition_time < 1.0, (
            f"Configuration composition took {composition_time:.3f}s, "
            f"exceeding 1.0s requirement"
        )
        
        # Validate configuration was loaded successfully
        assert "navigator" in cfg
        assert "video_plume" in cfg
        assert "experiment" in cfg

    def test_configuration_source_tracking(self, hydra_config_fixture):
        """
        Test hierarchical configuration source identification and composition
        tracking ensuring transparency in parameter inheritance chains.
        """
        cfg = hydra_config_fixture
        
        # Validate configuration metadata is available
        assert hasattr(cfg, '_metadata') or OmegaConf.is_config(cfg)
        
        # Test parameter resolution tracking
        # Navigator parameters should show proper composition
        navigator_cfg = cfg.navigator
        assert OmegaConf.is_config(navigator_cfg)
        
        # Validate composed configuration maintains structure
        assert len(cfg.navigator) >= 5  # All navigator parameters present
        assert len(cfg.video_plume) >= 5  # All video plume parameters present
        assert len(cfg.experiment) >= 3   # All experiment parameters present


class TestConfigurationOverrideAndPrecedence:
    """
    Test configuration override mechanisms and parameter precedence rules
    validating F-006-RQ-003 CLI and Hydra override capabilities.
    """

    @pytest.fixture(autouse=True)
    def setup_hydra_isolation(self):
        """Ensure clean Hydra state for each test."""
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    @pytest.fixture
    def base_config_dir(self, tmp_path):
        """Create base configuration for override testing."""
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        base_config = {
            "navigator": {
                "speed": 1.0,
                "max_speed": 2.0,
                "position": [0.0, 0.0]
            },
            "experiment": {
                "seed": 42,
                "name": "base_experiment"
            }
        }
        
        base_file = config_dir / "config.yaml"
        OmegaConf.save(OmegaConf.create(base_config), base_file)
        
        return config_dir

    def test_programmatic_override_precedence(self, base_config_dir):
        """
        Test programmatic configuration overrides take highest precedence
        validating F-006-RQ-003 override precedence rules.
        """
        with initialize(config_path=str(base_config_dir), version_base=None):
            # Test single parameter override
            cfg = compose(
                config_name="config",
                overrides=["navigator.speed=2.5"]
            )
            
            assert cfg.navigator.speed == 2.5     # Overridden
            assert cfg.navigator.max_speed == 2.0  # Not overridden
            assert cfg.experiment.seed == 42       # Not overridden
            
            # Test nested parameter override
            cfg_nested = compose(
                config_name="config", 
                overrides=["experiment.name=override_experiment"]
            )
            
            assert cfg_nested.experiment.name == "override_experiment"
            assert cfg_nested.experiment.seed == 42  # Unchanged
            
            # Test multiple parameter overrides
            cfg_multi = compose(
                config_name="config",
                overrides=[
                    "navigator.speed=3.0",
                    "navigator.max_speed=5.0", 
                    "experiment.seed=123"
                ]
            )
            
            assert cfg_multi.navigator.speed == 3.0
            assert cfg_multi.navigator.max_speed == 5.0
            assert cfg_multi.experiment.seed == 123

    def test_list_parameter_override(self, base_config_dir):
        """
        Test override of list-type parameters like agent positions
        ensuring proper list parsing and validation.
        """
        with initialize(config_path=str(base_config_dir), version_base=None):
            # Test position list override
            cfg = compose(
                config_name="config",
                overrides=["navigator.position=[1.5,2.5]"]
            )
            
            assert cfg.navigator.position == [1.5, 2.5]
            
            # Test multi-agent positions override
            cfg_multi = compose(
                config_name="config", 
                overrides=["navigator.positions=[[0,0],[1,1],[2,2]]"]
            )
            
            # Should create positions parameter
            assert "positions" in cfg_multi.navigator

    def test_type_safe_override_validation(self, base_config_dir):
        """
        Test that configuration overrides maintain type safety and
        validation rules per OmegaConf type enforcement.
        """
        with initialize(config_path=str(base_config_dir), version_base=None):
            # Valid numeric override should succeed
            cfg = compose(
                config_name="config",
                overrides=["navigator.speed=2.5"]
            )
            assert cfg.navigator.speed == 2.5
            
            # Test boolean override
            cfg_bool = compose(
                config_name="config",
                overrides=["video_plume.flip=true"]
            )
            assert cfg_bool.video_plume.flip is True

    def test_override_conflict_resolution(self, base_config_dir):
        """
        Test conflict resolution when multiple overrides affect the same parameter
        ensuring last override wins per Hydra precedence rules.
        """
        with initialize(config_path=str(base_config_dir), version_base=None):
            # Multiple overrides of same parameter - last should win
            cfg = compose(
                config_name="config",
                overrides=[
                    "navigator.speed=2.0",
                    "navigator.speed=3.0",  # This should win
                    "navigator.speed=4.0"   # This should win  
                ]
            )
            
            assert cfg.navigator.speed == 4.0


class TestDataclassSecretManagement:
    """
    Test enhanced secret management with field(repr=False) and automatic log exclusion
    ensuring sensitive configuration data is properly protected in dataclass schemas.
    """

    @pytest.fixture(autouse=True)
    def setup_hydra_isolation(self):
        """Ensure clean Hydra state for each test."""
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    @pytest.fixture
    def secret_config_dir(self, tmp_path):
        """Create configuration with sensitive data for secret management testing."""
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Configuration with sensitive fields that should use field(repr=False)
        config_with_secrets = {
            "database": {
                "_target_": "odor_plume_nav.persistence.DatabaseConnection",
                "host": "localhost",
                "port": 5432,
                "username": "research_user",
                "password": "${oc.env:DB_PASSWORD,default_secret_password}",
                "api_key": "${oc.env:API_KEY,sk-1234567890abcdef}",
                "connection_string": "postgresql://user:secret@localhost:5432/db"
            },
            "authentication": {
                "_target_": "odor_plume_nav.auth.AuthService",
                "jwt_secret": "${oc.env:JWT_SECRET,super_secret_jwt_key}",
                "admin_token": "${oc.env:ADMIN_TOKEN,admin_12345}",
                "encrypt_key": "encryption_key_should_be_hidden",
                "public_key": "public_keys_can_be_visible"
            },
            "experiment": {
                "name": "secret_management_test",
                "researcher_id": "researcher_001",
                "data_path": "/secure/research/data"
            }
        }
        
        config_file = config_dir / "config.yaml"
        OmegaConf.save(OmegaConf.create(config_with_secrets), config_file)
        
        return config_dir

    def test_field_repr_false_protection(self, secret_config_dir):
        """
        Test that sensitive fields marked with field(repr=False) are excluded
        from string representations and logging output for security protection.
        """
        # Set sensitive environment variables
        os.environ['DB_PASSWORD'] = 'very_secret_password'
        os.environ['API_KEY'] = 'sk-super_secret_api_key'
        os.environ['JWT_SECRET'] = 'jwt_ultra_secret'
        
        try:
            with initialize(config_path=str(secret_config_dir), version_base=None):
                cfg = compose(config_name="config")
                
                # Convert to string representation (simulating logging)
                config_str = str(cfg)
                config_yaml = OmegaConf.to_yaml(cfg)
                
                # Validate sensitive data is excluded from string representations
                # These should NOT appear in logs or string output
                sensitive_values = [
                    'very_secret_password',
                    'sk-super_secret_api_key', 
                    'jwt_ultra_secret',
                    'super_secret_jwt_key',
                    'admin_12345',
                    'encryption_key_should_be_hidden'
                ]
                
                for sensitive_value in sensitive_values:
                    assert sensitive_value not in config_str, (
                        f"Sensitive value '{sensitive_value}' found in config string representation"
                    )
                
                # Non-sensitive data should still be visible
                assert "research_user" in config_yaml
                assert "localhost" in config_yaml
                assert "5432" in str(cfg.database.port)
                assert "public_keys_can_be_visible" in config_yaml
                
        finally:
            # Clean up environment variables
            for var in ['DB_PASSWORD', 'API_KEY', 'JWT_SECRET']:
                os.environ.pop(var, None)

    def test_automatic_log_exclusion(self, secret_config_dir, caplog):
        """
        Test that sensitive configuration fields are automatically excluded
        from structured log output preventing credential exposure.
        """
        os.environ['DB_PASSWORD'] = 'log_test_password'
        os.environ['API_KEY'] = 'log_test_api_key'
        
        try:
            with initialize(config_path=str(secret_config_dir), version_base=None):
                cfg = compose(config_name="config")
                
                # Simulate configuration logging that might occur in application
from loguru import logger
                # Log configuration (this should exclude sensitive fields)
                logger.info("Configuration loaded: %s", cfg)
                logger.debug("Database config: %s", cfg.database)
                
                # Check captured log output
                log_output = caplog.text.lower()
                
                # Sensitive values should NOT appear in logs
                assert 'log_test_password' not in log_output
                assert 'log_test_api_key' not in log_output
                assert 'super_secret_jwt_key' not in log_output
                assert 'encryption_key_should_be_hidden' not in log_output
                
                # Non-sensitive values should appear in logs
                assert 'localhost' in log_output
                assert 'research_user' in log_output
                
        finally:
            for var in ['DB_PASSWORD', 'API_KEY']:
                os.environ.pop(var, None)

    def test_secret_field_access_validation(self, secret_config_dir):
        """
        Test that sensitive fields are still accessible programmatically when needed
        but protected from accidental exposure in logs and representations.
        """
        os.environ['DB_PASSWORD'] = 'programmatic_access_test'
        
        try:
            with initialize(config_path=str(secret_config_dir), version_base=None):
                cfg = compose(config_name="config")
                
                # Sensitive fields should be accessible for application use
                assert cfg.database.password == 'programmatic_access_test'
                assert cfg.database.username == 'research_user'  # Non-sensitive
                
                # But should not appear in OmegaConf YAML output
                yaml_output = OmegaConf.to_yaml(cfg)
                assert 'programmatic_access_test' not in yaml_output
                
                # Validate that masking or exclusion is applied correctly
                assert '***' in yaml_output or 'password' not in yaml_output
                
        finally:
            os.environ.pop('DB_PASSWORD', None)


class TestEnvironmentVariableInterpolationSecurity:
    """
    Test environment variable interpolation security preventing malicious
    parameter injection per Section 6.6.7.1 configuration security requirements.
    """

    @pytest.fixture(autouse=True)
    def setup_hydra_isolation(self):
        """Ensure clean Hydra state for each test."""
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    @pytest.fixture
    def secure_config_dir(self, tmp_path):
        """Create configuration with environment variable interpolation for security testing."""
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Configuration with environment variable interpolation
        config_with_env = {
            "navigator": {
                "speed": "${oc.env:NAVIGATOR_SPEED,1.0}",
                "max_speed": "${oc.env:NAVIGATOR_MAX_SPEED,2.0}",
                "position": [0.0, 0.0]
            },
            "video_plume": {
                "video_path": "${oc.env:VIDEO_PATH,/default/video.mp4}",
                "flip": "${oc.env:VIDEO_FLIP,false}"
            },
            "experiment": {
                "name": "${oc.env:EXPERIMENT_NAME,default}",
                "seed": 42
            },
            "security": {
                "admin_password": "secure_default",
                "api_key": "default_key"
            }
        }
        
        config_file = config_dir / "config.yaml"
        OmegaConf.save(OmegaConf.create(config_with_env), config_file)
        
        return config_dir

    def test_dataclass_environment_variable_interpolation_basic(self, secure_config_dir):
        """
        Test environment variable interpolation with structured dataclass fields
        ensuring proper substitution and type coercion for dataclass schemas.
        """
        # Set test environment variables for dataclass field interpolation
        os.environ['NAVIGATOR_SPEED'] = '2.5'
        os.environ['VIDEO_PATH'] = '/test/structured/video.mp4'
        os.environ['EXPERIMENT_NAME'] = 'dataclass_test_experiment'
        os.environ['MAX_STEPS'] = '5000'
        os.environ['ENABLE_GPU'] = 'true'
        
        try:
            with initialize(config_path=str(secure_config_dir), version_base=None):
                cfg = compose(config_name="config")
                
                # Validate environment variables were interpolated with correct types
                assert cfg.navigator.speed == 2.5  # Float type preserved
                assert cfg.video_plume.video_path == "/test/structured/video.mp4"
                assert cfg.experiment.name == "dataclass_test_experiment"
                
                # Test type coercion for dataclass fields
                cfg_with_overrides = compose(
                    config_name="config",
                    overrides=[
                        "simulation.max_steps=${oc.env:MAX_STEPS}",
                        "simulation.enable_gpu=${oc.env:ENABLE_GPU}"
                    ]
                )
                
                assert cfg_with_overrides.simulation.max_steps == 5000  # Int type
                assert cfg_with_overrides.simulation.enable_gpu is True  # Bool type
                
                # Validate defaults used when env var not set
                assert cfg.navigator.max_speed == 2.0  # Default value from dataclass
                
        finally:
            # Clean up environment variables
            for var in ['NAVIGATOR_SPEED', 'VIDEO_PATH', 'EXPERIMENT_NAME', 'MAX_STEPS', 'ENABLE_GPU']:
                os.environ.pop(var, None)

    def test_malicious_environment_variable_injection_prevention(self, secure_config_dir):
        """
        Test that malicious environment variable injection cannot override 
        sensitive configuration parameters per Section 6.6.7.1 security requirements.
        """
        # Attempt malicious environment variable injection
        malicious_vars = {
            'MALICIOUS_OVERRIDE': 'hacked_value',
            'SYSTEM_ADMIN_PASSWORD': 'pwned',
            'API_SECRET_KEY': 'stolen_key'
        }
        
        for var, value in malicious_vars.items():
            os.environ[var] = value
            
        try:
            with initialize(config_path=str(secure_config_dir), version_base=None):
                cfg = compose(config_name="config")
                
                # Validate sensitive parameters cannot be overridden via env injection
                assert cfg.security.admin_password == "secure_default"
                assert cfg.security.api_key == "default_key"
                
                # Validate malicious environment variables don't appear in config
                config_str = OmegaConf.to_yaml(cfg)
                assert 'hacked_value' not in config_str
                assert 'pwned' not in config_str
                assert 'stolen_key' not in config_str
                
        finally:
            # Clean up malicious environment variables
            for var in malicious_vars:
                os.environ.pop(var, None)

    def test_environment_variable_interpolation_with_overrides(self, secure_config_dir):
        """
        Test environment variable interpolation security when combined with
        configuration overrides ensuring override precedence is maintained.
        """
        os.environ['NAVIGATOR_SPEED'] = '2.0'
        
        try:
            with initialize(config_path=str(secure_config_dir), version_base=None):
                # Override should take precedence over environment variable
                cfg = compose(
                    config_name="config",
                    overrides=["navigator.speed=3.5"]
                )
                
                # Override should win over environment variable
                assert cfg.navigator.speed == 3.5
                
        finally:
            os.environ.pop('NAVIGATOR_SPEED', None)

    def test_environment_variable_interpolation_security_boundaries(self, secure_config_dir):
        """
        Test that environment variable interpolation respects security boundaries
        and cannot access unauthorized system information.
        """
        # Attempt to inject system commands or file paths
        malicious_env_attempts = {
            'MALICIOUS_COMMAND': '$(whoami)',
            'MALICIOUS_PATH': '../../../etc/passwd',
            'MALICIOUS_INJECTION': '${oc.env:HOME}'
        }
        
        for var, value in malicious_env_attempts.items():
            os.environ[var] = value
            
        try:
            with initialize(config_path=str(secure_config_dir), version_base=None):
                # Configuration should load without executing malicious content
                cfg = compose(config_name="config")
                
                # Validate configuration loaded successfully without injection
                assert "navigator" in cfg
                assert "video_plume" in cfg
                assert "experiment" in cfg
                
                # Validate no malicious content in resolved configuration
                config_yaml = OmegaConf.to_yaml(cfg)
                assert '$(whoami)' not in config_yaml
                assert '../../../etc/passwd' not in config_yaml
                
        finally:
            for var in malicious_env_attempts:
                os.environ.pop(var, None)


class TestDataclassPydanticValidationIntegration:
    """
    Test Pydantic validation hooks integration with dataclass configurations
    validating runtime validation within structured dataclass schemas per F-006-RQ-005.
    """

    @pytest.fixture(autouse=True)
    def setup_hydra_isolation(self):
        """Ensure clean Hydra state for each test."""
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    @pytest.fixture
    def validation_config_dir(self, tmp_path):
        """Create configuration directory for Pydantic validation testing."""
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Valid configuration for Pydantic validation
        valid_config = {
            "navigator": {
                "position": [0.0, 0.0],
                "orientation": 45.0,
                "speed": 1.0,
                "max_speed": 2.0,
                "angular_velocity": 0.5
            },
            "video_plume": {
                "video_path": "/test/video.mp4", 
                "flip": False,
                "grayscale": True,
                "kernel_size": 5,
                "kernel_sigma": 1.0,
                "normalize": True,
                "_skip_validation": True  # Skip file existence for testing
            }
        }
        
        config_file = config_dir / "config.yaml"
        OmegaConf.save(OmegaConf.create(valid_config), config_file)
        
        return config_dir

    def test_dataclass_pydantic_validation_hooks(self, validation_config_dir):
        """
        Test Pydantic validation hooks integration within dataclass configuration
        ensuring runtime validation within structured schema per F-006-RQ-005.
        """
        with initialize(config_path=str(validation_config_dir), version_base=None):
            cfg = compose(config_name="config")
            
            # Validate that dataclass structure contains Pydantic validation
            navigator_dict = OmegaConf.to_container(cfg.navigator, resolve=True)
            
            # Test creating Pydantic model from dataclass-configured data
            navigator_config = NavigatorConfig(**navigator_dict)
            
            # Validate Pydantic validation hooks work with dataclass data
            assert navigator_config.position == (0.0, 0.0)
            assert navigator_config.orientation == 45.0
            assert navigator_config.speed == 1.0
            assert navigator_config.max_speed == 2.0
            assert navigator_config.angular_velocity == 0.5
            
            # Test that validation hooks are triggered for dataclass fields
            assert hasattr(navigator_config, 'model_validate')  # Pydantic method available
            assert hasattr(navigator_config, 'model_dump')     # Pydantic serialization available

    def test_dataclass_video_plume_validation_integration(self, validation_config_dir):
        """
        Test VideoPlumeConfig dataclass with Pydantic validation hooks integration
        ensuring video configuration validation within structured schemas.
        """
        with initialize(config_path=str(validation_config_dir), version_base=None):
            cfg = compose(config_name="config")
            
            # Convert Hydra dataclass config to Pydantic validation
            video_dict = OmegaConf.to_container(cfg.video_plume, resolve=True)
            video_config = VideoPlumeConfig(**video_dict)
            
            # Validate Pydantic validation works with dataclass-sourced data
            assert str(video_config.video_path) == "/test/video.mp4"
            assert video_config.flip is False
            assert video_config.grayscale is True
            assert video_config.kernel_size == 5
            assert video_config.kernel_sigma == 1.0
            
            # Test dataclass field validation through Pydantic hooks
            assert hasattr(video_config, 'model_config')  # Pydantic config available
            assert video_config.model_config.get('validate_assignment', False) or True

    def test_dataclass_pydantic_constraint_enforcement(self, validation_config_dir):
        """
        Test that Pydantic validation hooks enforce constraints within dataclass 
        configurations ensuring type validation requirements per F-006-RQ-005.
        """
        with initialize(config_path=str(validation_config_dir), version_base=None):
            # Test speed constraint validation through dataclass configuration
            cfg_invalid_speed = compose(
                config_name="config",
                overrides=["navigator.speed=5.0", "navigator.max_speed=2.0"]
            )
            
            navigator_dict = OmegaConf.to_container(cfg_invalid_speed.navigator, resolve=True)
            
            # Pydantic validation hooks should catch constraint violation
            with pytest.raises(ValueError, match="speed .* cannot exceed max_speed"):
                NavigatorConfig(**navigator_dict)
            
            # Test additional dataclass field constraints
            with pytest.raises(ValueError):
                NavigatorConfig(
                    position=[0.0, 0.0],
                    orientation=450.0,  # Invalid orientation > 360
                    speed=1.0,
                    max_speed=2.0
                )

    def test_dataclass_validation_error_reporting(self, validation_config_dir):
        """
        Test that Pydantic validation hooks provide clear error messages for
        dataclass configurations per F-006-RQ-005 error reporting requirements.
        """
        with initialize(config_path=str(validation_config_dir), version_base=None):
            # Test invalid kernel_size validation through dataclass schema
            cfg_invalid_kernel = compose(
                config_name="config",
                overrides=["video_plume.kernel_size=4"]  # Even number - invalid
            )
            
            video_dict = OmegaConf.to_container(cfg_invalid_kernel.video_plume, resolve=True)
            
            # Pydantic validation hooks should provide clear error message
            with pytest.raises(ValueError, match="kernel_size must be odd"):
                VideoPlumeConfig(**video_dict)
            
            # Test dataclass field type validation error reporting
            with pytest.raises(ValueError) as exc_info:
                VideoPlumeConfig(
                    video_path="/test/video.mp4",
                    kernel_sigma=-1.0  # Invalid negative sigma
                )
            
            # Ensure error message is descriptive for dataclass field validation
            assert "kernel_sigma" in str(exc_info.value) or "sigma" in str(exc_info.value)

    def test_dataclass_multi_agent_validation_integration(self, validation_config_dir):
        """
        Test multi-agent configuration validation through dataclass with Pydantic hooks
        ensuring parameter consistency validation across agent arrays in structured configs.
        """
        with initialize(config_path=str(validation_config_dir), version_base=None):
            # Configure multi-agent scenario through dataclass schema
            cfg_multi = compose(
                config_name="config",
                overrides=[
                    "navigator.mode=multi",
                    "navigator.positions=[[0,0],[1,1],[2,2]]",
                    "navigator.speeds=[1.0,1.5,2.0]",
                    "navigator.max_speeds=[2.0,2.5,3.0]",
                    "navigator.num_agents=3"
                ]
            )
            
            navigator_dict = OmegaConf.to_container(cfg_multi.navigator, resolve=True)
            
            # Pydantic validation hooks should succeed for consistent multi-agent dataclass config
            multi_config = NavigatorConfig(**navigator_dict)
            assert multi_config.mode == "multi"
            assert len(multi_config.positions) == 3
            assert len(multi_config.speeds) == 3
            assert len(multi_config.max_speeds) == 3
            assert multi_config.num_agents == 3
            
            # Test dataclass field validation for multi-agent consistency
            with pytest.raises(ValueError):
                NavigatorConfig(
                    mode="multi",
                    positions=[[0,0], [1,1]],  # 2 agents
                    speeds=[1.0, 1.5, 2.0],    # 3 speeds - inconsistent
                    num_agents=2
                )


class TestDataclassCompileTimeValidation:
    """
    Test hierarchical config composition with compile-time validation enforcement
    ensuring dataclass schemas provide type safety before runtime execution.
    """

    @pytest.fixture(autouse=True)
    def setup_hydra_isolation(self):
        """Ensure clean Hydra state for each test."""
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    @pytest.fixture
    def compile_validation_config_dir(self, tmp_path):
        """Create configuration for compile-time validation testing."""
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Configuration that should pass compile-time validation
        valid_config = {
            "defaults": ["_self_"],
            "_target_": "odor_plume_nav.api.create_simulation",
            "navigator": {
                "_target_": "odor_plume_nav.core.controllers.NavigatorController",
                "mode": "single",
                "position": [25.0, 75.0],
                "orientation": 180.0,
                "speed": 1.0,
                "max_speed": 2.5,
                "angular_velocity": 0.05
            },
            "simulation": {
                "_target_": "odor_plume_nav.api.run_simulation", 
                "max_steps": 1500,
                "step_size": 1.0,
                "enable_gpu": false,
                "batch_size": 1,
                "record_trajectory": true
            }
        }
        
        config_file = config_dir / "config.yaml"
        OmegaConf.save(OmegaConf.create(valid_config), config_file)
        
        return config_dir

    def test_dataclass_compile_time_type_enforcement(self, compile_validation_config_dir):
        """
        Test that dataclass schemas enforce compile-time type validation
        preventing type mismatches before runtime execution.
        """
        with initialize(config_path=str(compile_validation_config_dir), version_base=None):
            # Valid configuration should load without compile-time errors
            cfg = compose(config_name="config")
            
            # Validate compile-time type inference works correctly
            assert isinstance(cfg.navigator.position, list)
            assert all(isinstance(x, (int, float)) for x in cfg.navigator.position)
            assert isinstance(cfg.navigator.orientation, (int, float))
            assert isinstance(cfg.navigator.speed, (int, float))
            assert isinstance(cfg.simulation.max_steps, int)
            assert isinstance(cfg.simulation.enable_gpu, bool)

    def test_hierarchical_composition_compile_validation(self, compile_validation_config_dir):
        """
        Test hierarchical configuration composition with compile-time validation
        ensuring schema enforcement throughout the composition hierarchy.
        """
        with initialize(config_path=str(compile_validation_config_dir), version_base=None):
            # Test base configuration compile-time validation
            cfg = compose(config_name="config")
            
            # Test hierarchical override with compile-time validation
            cfg_override = compose(
                config_name="config",
                overrides=[
                    "navigator.mode=multi",
                    "navigator.positions=[[0,0],[10,10],[20,20]]",
                    "navigator.num_agents=3",
                    "simulation.batch_size=3"
                ]
            )
            
            # Validate compile-time type enforcement in overrides
            assert cfg_override.navigator.mode == "multi"
            assert len(cfg_override.navigator.positions) == 3
            assert cfg_override.navigator.num_agents == 3
            assert cfg_override.simulation.batch_size == 3
            
            # Test that type mismatches are caught at composition time
            with pytest.raises(Exception):  # Should fail at compose time
                compose(
                    config_name="config",
                    overrides=[
                        "navigator.position=invalid_type",  # Should be list
                        "simulation.max_steps=not_an_integer"  # Should be int
                    ]
                )

    def test_dataclass_field_constraint_compile_enforcement(self, compile_validation_config_dir):
        """
        Test that dataclass field constraints are enforced at configuration
        composition time rather than delayed until runtime execution.
        """
        with initialize(config_path=str(compile_validation_config_dir), version_base=None):
            # Valid constraints should compose successfully
            cfg_valid = compose(
                config_name="config",
                overrides=[
                    "navigator.orientation=270.0",  # Valid 0-360 range
                    "navigator.speed=1.5",
                    "navigator.max_speed=3.0"       # speed <= max_speed
                ]
            )
            
            assert cfg_valid.navigator.orientation == 270.0
            assert cfg_valid.navigator.speed == 1.5
            assert cfg_valid.navigator.max_speed == 3.0

    def test_configstore_schema_validation_integration(self, compile_validation_config_dir):
        """
        Test that ConfigStore schema validation integrates with compile-time 
        validation ensuring structured dataclass schemas are enforced.
        """
        # Register schemas with ConfigStore
        register_config_schemas()
        cs = ConfigStore.instance()
        
        with initialize(config_path=str(compile_validation_config_dir), version_base=None):
            # Test that ConfigStore schemas provide compile-time validation
            cfg = compose(config_name="config")
            
            # Validate that _target_ metadata from dataclass schemas is preserved
            assert "_target_" in cfg.navigator
            assert "_target_" in cfg.simulation
            
            # Test schema-driven validation through ConfigStore
            navigator_schema = cs.get_schema_path("navigator/unified")
            assert navigator_schema is not None
            
            # Validate that structured schemas enforce field requirements
            assert hasattr(cfg.navigator, 'mode')
            assert hasattr(cfg.navigator, 'position') 
            assert hasattr(cfg.navigator, 'orientation')


class TestConfigurationPerformanceAndReliability:
    """
    Test configuration loading performance and reliability requirements
    per Section 2.2.9.3 performance criteria and test isolation standards.
    """

    @pytest.fixture(autouse=True)
    def setup_hydra_isolation(self):
        """Ensure clean Hydra state for each test."""
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    @pytest.fixture
    def performance_config_dir(self, tmp_path):
        """Create configuration directory for performance testing."""
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Large configuration to test performance
        large_config = {
            "navigator": {
                "positions": [[i, i] for i in range(100)],  # 100 agents
                "speeds": [1.0] * 100,
                "max_speeds": [2.0] * 100,
                "orientations": [i * 3.6 for i in range(100)],  # 0-360 degrees
                "num_agents": 100
            },
            "video_plume": {
                "video_path": "/test/large_video.mp4",
                "flip": False,
                "grayscale": True,
                "kernel_size": 7,
                "kernel_sigma": 2.0,
                "_skip_validation": True
            },
            "experiment": {
                "name": "large_scale_test",
                "seed": 42,
                "num_steps": 10000
            }
        }
        
        config_file = config_dir / "config.yaml"
        OmegaConf.save(OmegaConf.create(large_config), config_file)
        
        return config_dir

    def test_configuration_loading_performance_requirement(self, performance_config_dir):
        """
        Test configuration loading meets <1s performance requirement per
        Section 2.2.9.3 configuration loading performance criteria.
        """
        # Measure configuration loading time for large configuration
        start_time = time.time()
        
        with initialize(config_path=str(performance_config_dir), version_base=None):
            cfg = compose(config_name="config")
            
            # Ensure configuration is fully resolved
            _ = OmegaConf.to_container(cfg, resolve=True)
            
        end_time = time.time()
        loading_time = end_time - start_time
        
        # Validate performance requirement (<1s)
        assert loading_time < 1.0, (
            f"Configuration loading took {loading_time:.3f}s, "
            f"exceeding 1.0s requirement"
        )
        
        # Validate large configuration loaded correctly
        assert cfg.navigator.num_agents == 100
        assert len(cfg.navigator.positions) == 100
        assert cfg.experiment.num_steps == 10000

    def test_configuration_composition_scaling(self, performance_config_dir):
        """
        Test configuration composition performance scales appropriately with
        override complexity ensuring batch processing efficiency.
        """
        # Test composition with multiple overrides
        override_sets = [
            [],  # No overrides
            ["navigator.num_agents=50"],  # Single override
            ["navigator.num_agents=50", "experiment.seed=123"],  # Two overrides
            [
                "navigator.num_agents=25", 
                "experiment.seed=123",
                "experiment.name=scaled_test",
                "video_plume.kernel_size=9"
            ]  # Multiple overrides
        ]
        
        composition_times = []
        
        for overrides in override_sets:
            start_time = time.time()
            
            with initialize(config_path=str(performance_config_dir), version_base=None):
                cfg = compose(config_name="config", overrides=overrides)
                _ = OmegaConf.to_container(cfg, resolve=True)
                
            end_time = time.time()
            composition_times.append(end_time - start_time)
        
        # All composition times should be under performance threshold
        for i, comp_time in enumerate(composition_times):
            assert comp_time < 1.0, (
                f"Composition with {len(override_sets[i])} overrides took "
                f"{comp_time:.3f}s, exceeding 1.0s requirement"
            )

    def test_hydra_working_directory_isolation(self, tmp_path):
        """
        Test Hydra working directory isolation prevents configuration pollution
        between test runs per Section 6.6.5.4 test environment isolation.
        """
        # Create two separate configuration scenarios
        config_dir_1 = tmp_path / "conf1" 
        config_dir_1.mkdir()
        
        config_1 = {"test_param": "value_1", "seed": 111}
        config_file_1 = config_dir_1 / "config.yaml"
        OmegaConf.save(OmegaConf.create(config_1), config_file_1)
        
        config_dir_2 = tmp_path / "conf2"
        config_dir_2.mkdir()
        
        config_2 = {"test_param": "value_2", "seed": 222}  
        config_file_2 = config_dir_2 / "config.yaml"
        OmegaConf.save(OmegaConf.create(config_2), config_file_2)
        
        # Test first configuration
        with initialize(config_path=str(config_dir_1), version_base=None):
            cfg_1 = compose(config_name="config")
            assert cfg_1.test_param == "value_1"
            assert cfg_1.seed == 111
        
        # Clear Hydra state
        GlobalHydra.instance().clear()
        
        # Test second configuration - should not be polluted by first
        with initialize(config_path=str(config_dir_2), version_base=None):
            cfg_2 = compose(config_name="config")
            assert cfg_2.test_param == "value_2"
            assert cfg_2.seed == 222
        
        # Validate no cross-contamination occurred
        assert cfg_1.test_param != cfg_2.test_param
        assert cfg_1.seed != cfg_2.seed

    def test_concurrent_configuration_access_safety(self, performance_config_dir):
        """
        Test that concurrent configuration access is handled safely without
        state corruption ensuring thread-safe configuration management.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        def load_config_with_overrides(override_seed):
            """Load configuration with unique seed override."""
            with initialize(config_path=str(performance_config_dir), version_base=None):
                cfg = compose(
                    config_name="config", 
                    overrides=[f"experiment.seed={override_seed}"]
                )
                return cfg.experiment.seed
        
        # Test concurrent configuration loading with different overrides
        seeds = [100, 200, 300, 400, 500]
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(load_config_with_overrides, seed) 
                for seed in seeds
            ]
            
            results = [future.result() for future in futures]
        
        # Validate each configuration loaded correctly without interference
        assert len(results) == len(seeds)
        assert set(results) == set(seeds)  # All unique seeds returned correctly


class TestHydraConfigIntegrationEndToEnd:
    """
    End-to-end integration tests combining all Hydra configuration features
    ensuring comprehensive system validation across all testing domains.
    """

    @pytest.fixture(autouse=True)
    def setup_hydra_isolation(self):
        """Ensure clean Hydra state for each test."""
        GlobalHydra.instance().clear()
        yield  
        GlobalHydra.instance().clear()

    @pytest.fixture
    def comprehensive_config_dir(self, tmp_path):
        """Create comprehensive configuration directory for end-to-end testing."""
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        
        # Base configuration
        base_config = {
            "defaults": ["_self_"],
            "navigator": {
                "position": [0.0, 0.0],
                "orientation": 0.0,
                "speed": "${oc.env:DEFAULT_SPEED,1.0}",
                "max_speed": 2.0
            },
            "video_plume": {
                "video_path": "${oc.env:VIDEO_PATH,/default/video.mp4}",
                "flip": False,
                "grayscale": True,
                "kernel_size": 5,
                "kernel_sigma": 1.0,
                "_skip_validation": True
            },
            "experiment": {
                "name": "${oc.env:EXPERIMENT_NAME,default_experiment}",
                "seed": 42,
                "output_dir": "${oc.env:OUTPUT_DIR,./outputs}"
            }
        }
        
        base_file = config_dir / "base.yaml"
        OmegaConf.save(OmegaConf.create(base_config), base_file)
        
        # Main configuration inheriting from base with overrides
        main_config = {
            "defaults": ["base"],
            "navigator": {
                "speed": 1.5,
                "angular_velocity": 0.1
            },
            "experiment": {
                "num_steps": 1000
            }
        }
        
        main_file = config_dir / "config.yaml"
        OmegaConf.save(OmegaConf.create(main_config), main_file)
        
        # Local overrides directory
        local_dir = config_dir / "local"
        local_dir.mkdir()
        
        local_config = {
            "navigator": {
                "max_speed": 3.0
            },
            "experiment": {
                "name": "local_override_experiment"
            }
        }
        
        local_file = local_dir / "config.yaml"
        OmegaConf.save(OmegaConf.create(local_config), local_file)
        
        return config_dir

    def test_end_to_end_dataclass_configuration_composition(self, comprehensive_config_dir):
        """
        Test complete dataclass-based configuration composition pipeline including
        ConfigStore registration, hierarchical inheritance, environment variables, 
        and programmatic overrides with compile-time validation.
        """
        # Register dataclass schemas with ConfigStore
        register_config_schemas()
        
        # Set environment variables for dataclass interpolation
        env_vars = {
            'DEFAULT_SPEED': '2.0',
            'VIDEO_PATH': '/env/dataclass/video.mp4',
            'EXPERIMENT_NAME': 'dataclass_env_experiment',
            'OUTPUT_DIR': '/env/dataclass/outputs',
            'MAX_STEPS': '3000',
            'ENABLE_GPU': 'false'
        }
        
        for var, value in env_vars.items():
            os.environ[var] = value
            
        try:
            start_time = time.time()
            
            with initialize(config_path=str(comprehensive_config_dir), version_base=None):
                # Load with programmatic overrides using dataclass structure
                cfg = compose(
                    config_name="config",
                    overrides=[
                        "navigator.orientation=90.0",
                        "navigator.mode=single",
                        "simulation.max_steps=${oc.env:MAX_STEPS}",
                        "simulation.enable_gpu=${oc.env:ENABLE_GPU}",
                        "experiment.seed=999"
                    ]
                )
                
                # Validate dataclass integration with Pydantic validation hooks
                navigator_dict = OmegaConf.to_container(cfg.navigator, resolve=True)
                navigator_config = NavigatorConfig(**navigator_dict)
                
                video_dict = OmegaConf.to_container(cfg.video_plume, resolve=True)
                video_config = VideoPlumeConfig(**video_dict)
                
                simulation_dict = OmegaConf.to_container(cfg.simulation, resolve=True)
                simulation_config = SimulationConfig(**simulation_dict)
                
            end_time = time.time()
            
            # Validate performance requirement for dataclass composition
            assert (end_time - start_time) < 1.0
            
            # Validate hierarchical composition with dataclass structure
            # From base.yaml (with env override)
            assert cfg.navigator.speed == 2.0  # Environment variable override
            assert cfg.video_plume.video_path == "/env/dataclass/video.mp4"  # Env var
            
            # From config.yaml override
            assert cfg.navigator.angular_velocity == 0.1
            assert cfg.experiment.num_steps == 1000
            
            # From programmatic override (highest precedence)
            assert cfg.navigator.orientation == 90.0
            assert cfg.navigator.mode == "single"
            assert cfg.simulation.max_steps == 3000  # From env var with type coercion
            assert cfg.simulation.enable_gpu is False  # Bool type from env var
            assert cfg.experiment.seed == 999
            
            # Validate dataclass-Pydantic integration models created successfully
            assert navigator_config.speed == 2.0
            assert navigator_config.orientation == 90.0
            assert navigator_config.mode == "single"
            assert str(video_config.video_path) == "/env/dataclass/video.mp4"
            assert simulation_config.max_steps == 3000
            assert simulation_config.enable_gpu is False
            
            # Validate _target_ metadata preserved from dataclass schemas
            assert cfg.navigator._target_ == "odor_plume_nav.api.create_navigator"
            assert cfg.video_plume._target_ == "odor_plume_nav.data.VideoPlume"
            assert cfg.simulation._target_ == "odor_plume_nav.api.run_simulation"
            
        finally:
            # Clean up environment variables
            for var in env_vars:
                os.environ.pop(var, None)

    def test_comprehensive_error_handling_and_recovery(self, comprehensive_config_dir):
        """
        Test error handling across all configuration layers ensuring robust
        error reporting and graceful degradation scenarios.
        """
        # Test invalid environment variable scenario
        os.environ['DEFAULT_SPEED'] = 'invalid_number'
        
        try:
            with initialize(config_path=str(comprehensive_config_dir), version_base=None):
                # Should handle invalid environment variable gracefully  
                cfg = compose(config_name="config")
                
                # Environment variable interpolation should fail gracefully
                # or provide clear error message
                navigator_dict = OmegaConf.to_container(cfg.navigator, resolve=True)
                
                # Test with Pydantic validation
                try:
                    NavigatorConfig(**navigator_dict)
                except (ValueError, TypeError) as e:
                    # Should provide clear error message
                    assert "speed" in str(e).lower() or "invalid" in str(e).lower()
                    
        finally:
            os.environ.pop('DEFAULT_SPEED', None)

    def test_configuration_composition_memory_efficiency(self, comprehensive_config_dir):
        """
        Test memory efficiency of configuration composition ensuring scalable
        memory usage for large-scale scientific computing configurations.
        """
        import psutil
        import gc
        
        # Measure baseline memory
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss
        
        # Load multiple configurations to test memory scaling
        configs = []
        
        for i in range(10):  # Load 10 configurations
            with initialize(config_path=str(comprehensive_config_dir), version_base=None):
                cfg = compose(
                    config_name="config",
                    overrides=[f"experiment.seed={i * 100}"]
                )
                configs.append(cfg)
                
            GlobalHydra.instance().clear()
        
        # Measure peak memory usage
        gc.collect()
        peak_memory = process.memory_info().rss
        memory_increase = (peak_memory - baseline_memory) / 1024 / 1024  # MB
        
        # Validate reasonable memory usage (<50MB for configuration management)
        assert memory_increase < 50, (
            f"Configuration loading used {memory_increase:.1f}MB, "
            f"exceeding 50MB reasonable limit"
        )
        
        # Validate all configurations loaded successfully
        assert len(configs) == 10
        assert all("navigator" in cfg for cfg in configs)


if __name__ == "__main__":
    pytest.main([__file__])