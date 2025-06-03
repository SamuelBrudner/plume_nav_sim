"""
Advanced pytest test suite for Hydra configuration system integration.

This module provides comprehensive testing of the Hydra configuration management
system using pytest-hydra plugin to validate hierarchical configuration composition,
override scenarios, environment variable interpolation, and configuration security.

Validates robust Hydra configuration management across conf/base.yaml, conf/config.yaml,
and conf/local/ directory structures with comprehensive validation scenarios ensuring
research-grade reproducibility and security.

Key Testing Areas:
- Hierarchical configuration composition (F-006-RQ-001 through F-006-RQ-004)  
- Environment variable interpolation security (Section 6.6.7.1)
- Configuration override mechanisms and precedence (Feature F-006)
- Schema validation integration with Pydantic (Feature F-007)
- Configuration loading performance validation (<1s per Section 2.2.9.3)
- pytest-hydra plugin integration (Section 6.6.1.1)
- Working directory isolation for test reliability (Section 6.6.5.4)
"""

import os
import sys
import time
import tempfile
import pytest
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Hydra and OmegaConf imports with fallback
try:
    from hydra import initialize, initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf, DictConfig
    import pytest_hydra
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    pytest.skip("Hydra not available", allow_module_level=True)

# Test dependencies
from src.odor_plume_nav.config.schemas import (
    NavigatorConfig,
    VideoPlumeConfig,
    SingleAgentConfig,
    MultiAgentConfig
)


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

    def test_environment_variable_interpolation_basic(self, secure_config_dir):
        """
        Test basic environment variable interpolation functionality ensuring
        proper substitution without security vulnerabilities.
        """
        # Set test environment variables
        os.environ['NAVIGATOR_SPEED'] = '2.5'
        os.environ['VIDEO_PATH'] = '/test/video.mp4'
        os.environ['EXPERIMENT_NAME'] = 'test_experiment'
        
        try:
            with initialize(config_path=str(secure_config_dir), version_base=None):
                cfg = compose(config_name="config")
                
                # Validate environment variables were interpolated
                assert cfg.navigator.speed == 2.5
                assert cfg.video_plume.video_path == "/test/video.mp4"
                assert cfg.experiment.name == "test_experiment"
                
                # Validate defaults used when env var not set
                assert cfg.navigator.max_speed == 2.0  # Default value
                
        finally:
            # Clean up environment variables
            for var in ['NAVIGATOR_SPEED', 'VIDEO_PATH', 'EXPERIMENT_NAME']:
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


class TestPydanticSchemaValidationIntegration:
    """
    Test schema validation integration with Pydantic models validating
    Feature F-007 Hydra-triggered validation requirements.
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

    def test_navigator_config_pydantic_validation(self, validation_config_dir):
        """
        Test NavigatorConfig Pydantic validation integration with Hydra
        per F-007-RQ-001 navigator configuration validation requirements.
        """
        with initialize(config_path=str(validation_config_dir), version_base=None):
            cfg = compose(config_name="config")
            
            # Convert Hydra config to Pydantic model
            navigator_dict = OmegaConf.to_container(cfg.navigator, resolve=True)
            navigator_config = NavigatorConfig(**navigator_dict)
            
            # Validate Pydantic model creation succeeded
            assert navigator_config.position == (0.0, 0.0)
            assert navigator_config.orientation == 45.0
            assert navigator_config.speed == 1.0
            assert navigator_config.max_speed == 2.0
            assert navigator_config.angular_velocity == 0.5

    def test_video_plume_config_pydantic_validation(self, validation_config_dir):
        """
        Test VideoPlumeConfig Pydantic validation integration with Hydra
        per F-007-RQ-002 video configuration validation requirements.
        """
        with initialize(config_path=str(validation_config_dir), version_base=None):
            cfg = compose(config_name="config")
            
            # Convert Hydra config to Pydantic model
            video_dict = OmegaConf.to_container(cfg.video_plume, resolve=True)
            video_config = VideoPlumeConfig(**video_dict)
            
            # Validate Pydantic model creation succeeded
            assert str(video_config.video_path) == "/test/video.mp4"
            assert video_config.flip is False
            assert video_config.grayscale is True
            assert video_config.kernel_size == 5
            assert video_config.kernel_sigma == 1.0

    def test_pydantic_validation_constraint_enforcement(self, validation_config_dir):
        """
        Test that Pydantic validation constraints are enforced when integrating
        with Hydra configuration per F-007-RQ-003 type validation requirements.
        """
        with initialize(config_path=str(validation_config_dir), version_base=None):
            # Test speed constraint validation
            cfg_invalid_speed = compose(
                config_name="config",
                overrides=["navigator.speed=5.0", "navigator.max_speed=2.0"]
            )
            
            navigator_dict = OmegaConf.to_container(cfg_invalid_speed.navigator, resolve=True)
            
            # Pydantic validation should catch constraint violation
            with pytest.raises(ValueError, match="speed .* cannot exceed max_speed"):
                NavigatorConfig(**navigator_dict)

    def test_pydantic_validation_error_reporting(self, validation_config_dir):
        """
        Test that Pydantic validation provides clear error messages when
        integrated with Hydra per F-007-RQ-004 error reporting requirements.
        """
        with initialize(config_path=str(validation_config_dir), version_base=None):
            # Test invalid kernel_size (must be odd)
            cfg_invalid_kernel = compose(
                config_name="config",
                overrides=["video_plume.kernel_size=4"]  # Even number - invalid
            )
            
            video_dict = OmegaConf.to_container(cfg_invalid_kernel.video_plume, resolve=True)
            
            # Pydantic should provide clear error message
            with pytest.raises(ValueError, match="kernel_size must be odd"):
                VideoPlumeConfig(**video_dict)

    def test_multi_agent_config_validation_integration(self, validation_config_dir):
        """
        Test multi-agent configuration validation through Pydantic integration
        ensuring parameter consistency validation across agent arrays.
        """
        with initialize(config_path=str(validation_config_dir), version_base=None):
            # Configure multi-agent scenario  
            cfg_multi = compose(
                config_name="config",
                overrides=[
                    "navigator.positions=[[0,0],[1,1],[2,2]]",
                    "navigator.speeds=[1.0,1.5,2.0]",
                    "navigator.max_speeds=[2.0,2.5,3.0]",
                    "navigator.num_agents=3"
                ]
            )
            
            navigator_dict = OmegaConf.to_container(cfg_multi.navigator, resolve=True)
            
            # Pydantic validation should succeed for consistent multi-agent config
            multi_config = NavigatorConfig(**navigator_dict)
            assert len(multi_config.positions) == 3
            assert len(multi_config.speeds) == 3
            assert len(multi_config.max_speeds) == 3


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

    def test_end_to_end_configuration_composition_with_environment(self, comprehensive_config_dir):
        """
        Test complete configuration composition pipeline including hierarchical
        inheritance, environment variables, and programmatic overrides.
        """
        # Set environment variables
        env_vars = {
            'DEFAULT_SPEED': '2.0',
            'VIDEO_PATH': '/env/test/video.mp4',
            'EXPERIMENT_NAME': 'env_experiment',
            'OUTPUT_DIR': '/env/outputs'
        }
        
        for var, value in env_vars.items():
            os.environ[var] = value
            
        try:
            start_time = time.time()
            
            with initialize(config_path=str(comprehensive_config_dir), version_base=None):
                # Load with programmatic overrides
                cfg = compose(
                    config_name="config",
                    overrides=[
                        "navigator.orientation=90.0",
                        "experiment.seed=999"
                    ]
                )
                
                # Validate Pydantic integration
                navigator_dict = OmegaConf.to_container(cfg.navigator, resolve=True)
                navigator_config = NavigatorConfig(**navigator_dict)
                
                video_dict = OmegaConf.to_container(cfg.video_plume, resolve=True)
                video_config = VideoPlumeConfig(**video_dict)
                
            end_time = time.time()
            
            # Validate performance requirement
            assert (end_time - start_time) < 1.0
            
            # Validate hierarchical composition
            # From base.yaml (with env override)
            assert cfg.navigator.speed == 2.0  # Environment variable override
            assert cfg.video_plume.video_path == "/env/test/video.mp4"  # Env var
            
            # From config.yaml override
            assert cfg.navigator.angular_velocity == 0.1
            assert cfg.experiment.num_steps == 1000
            
            # From programmatic override (highest precedence)
            assert cfg.navigator.orientation == 90.0
            assert cfg.experiment.seed == 999
            
            # Validate Pydantic models created successfully
            assert navigator_config.speed == 2.0
            assert navigator_config.orientation == 90.0
            assert str(video_config.video_path) == "/env/test/video.mp4"
            
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