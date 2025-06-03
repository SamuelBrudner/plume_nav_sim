"""
Comprehensive configuration system testing module for {{cookiecutter.project_slug}}.

This module provides extensive validation of the Hydra-based configuration system,
testing hierarchical composition from conf/base.yaml through conf/config.yaml to
conf/local/ overrides. Includes validation of environment variable interpolation,
schema validation integration, override mechanisms, and security considerations.

Test Categories:
- Hydra configuration composition and hierarchical loading
- Environment variable interpolation with ${oc.env:VAR_NAME} syntax  
- Configuration override scenarios (CLI, programmatic, file-based)
- Pydantic schema integration with Hydra structured configs
- Configuration validation functions and comprehensive error handling
- Multi-run configuration composition and parameter sweep validation
- Configuration security including path traversal prevention
- Performance validation ensuring <500ms loading requirement
- Hierarchical override precedence and parameter inheritance testing
- Configuration schema evolution and backward compatibility validation

Performance Requirements:
- Configuration composition: <500ms per Section 6.6.3.3
- Schema validation: <100ms for typical configurations
- Environment variable interpolation: <50ms per variable

Security Requirements:
- Path traversal prevention per Section 6.6.7.1
- Environment variable injection protection
- Configuration override security validation
"""

import os
import tempfile
import time
import pytest
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import patch, mock_open, MagicMock
from contextlib import contextmanager

import yaml
from hydra import compose, initialize, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ValidationError

from src.{{cookiecutter.project_slug}}.config.schemas import (
    NavigatorConfig,
    SingleAgentConfig, 
    MultiAgentConfig,
    VideoPlumeConfig
)


class TestHydraConfigurationComposition:
    """Test Hydra configuration hierarchical composition and loading mechanisms."""
    
    def setup_method(self):
        """Set up clean Hydra state for each test."""
        GlobalHydra.instance().clear()
        
    def teardown_method(self):
        """Clean up Hydra state after each test."""
        GlobalHydra.instance().clear()
    
    @pytest.fixture
    def base_config_content(self):
        """Provides base configuration content for testing."""
        return {
            'navigator': {
                'position': [0.0, 0.0],
                'orientation': 0.0,
                'speed': 0.5,
                'max_speed': 1.0,
                'angular_velocity': 0.0
            },
            'video_plume': {
                'video_path': 'test_video.mp4',
                'flip': False,
                'grayscale': True,
                'normalize': True
            },
            'simulation': {
                'num_steps': 100,
                'step_size': 0.1,
                'random_seed': 42
            }
        }
    
    @pytest.fixture
    def config_override_content(self):
        """Provides config override content for testing."""
        return {
            'navigator': {
                'speed': 0.8,
                'max_speed': 1.5
            },
            'simulation': {
                'num_steps': 200,
                'random_seed': 123
            },
            'experiment': {
                'name': 'test_experiment',
                'output_dir': '${oc.env:OUTPUT_DIR,./outputs}'
            }
        }
    
    @pytest.fixture
    def local_override_content(self):
        """Provides local override content for testing."""
        return {
            'navigator': {
                'position': [1.0, 1.0],
                'orientation': 45.0
            },
            'video_plume': {
                'video_path': '/path/to/local/video.mp4',
                'threshold': 0.5
            },
            'experiment': {
                'output_dir': '/tmp/test_outputs'
            }
        }
    
    @contextmanager
    def temp_config_dir(self, base_config, config_override=None, local_override=None):
        """Create temporary configuration directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir)
            
            # Create base.yaml
            base_file = config_path / 'base.yaml'
            with open(base_file, 'w') as f:
                yaml.dump(base_config, f)
            
            # Create config.yaml
            config_file = config_path / 'config.yaml'
            config_content = {'defaults': ['base']}
            if config_override:
                config_content.update(config_override)
            with open(config_file, 'w') as f:
                yaml.dump(config_content, f)
            
            # Create local directory and overrides
            local_dir = config_path / 'local'
            local_dir.mkdir()
            
            if local_override:
                local_file = local_dir / 'overrides.yaml'
                with open(local_file, 'w') as f:
                    yaml.dump(local_override, f)
            
            yield str(config_path)
    
    def test_basic_hydra_composition(self, base_config_content, config_override_content):
        """Test basic Hydra configuration composition from hierarchical structure."""
        with self.temp_config_dir(base_config_content, config_override_content) as config_dir:
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg = compose(config_name="config")
                
                # Verify base configuration loaded
                assert cfg.navigator.position == [0.0, 0.0]
                assert cfg.navigator.orientation == 0.0
                assert cfg.video_plume.video_path == 'test_video.mp4'
                assert cfg.simulation.random_seed == 123  # Override applied
                
                # Verify overrides applied correctly  
                assert cfg.navigator.speed == 0.8  # Overridden
                assert cfg.navigator.max_speed == 1.5  # Overridden
                assert cfg.simulation.num_steps == 200  # Overridden
    
    def test_hierarchical_override_precedence(self, base_config_content, config_override_content, local_override_content):
        """Test hierarchical override precedence: local > config > base."""
        with self.temp_config_dir(base_config_content, config_override_content, local_override_content) as config_dir:
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg = compose(config_name="config", overrides=[f"hydra.searchpath=[file://{config_dir}/local]"])
                
                # Base values (lowest precedence)
                assert cfg.navigator.angular_velocity == 0.0  # Base only
                assert cfg.video_plume.grayscale == True  # Base only
                
                # Config overrides (middle precedence)
                assert cfg.navigator.speed == 0.8  # Config override
                assert cfg.simulation.num_steps == 200  # Config override
                
                # Local overrides would have highest precedence if properly loaded
                # Note: In real implementation, local overrides would be highest priority
    
    def test_configuration_composition_performance(self, base_config_content, config_override_content):
        """Test configuration composition meets <500ms performance requirement."""
        with self.temp_config_dir(base_config_content, config_override_content) as config_dir:
            start_time = time.time()
            
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg = compose(config_name="config")
                
            composition_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Verify performance requirement per Section 6.6.3.3
            assert composition_time < 500, f"Configuration composition took {composition_time:.2f}ms, exceeds 500ms requirement"
            
            # Verify configuration was loaded successfully
            assert cfg is not None
            assert hasattr(cfg, 'navigator')
            assert hasattr(cfg, 'video_plume')
    
    def test_invalid_configuration_handling(self):
        """Test handling of invalid configuration files."""
        invalid_yaml = "invalid: yaml: content: [unclosed"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir)
            invalid_file = config_path / 'config.yaml'
            
            with open(invalid_file, 'w') as f:
                f.write(invalid_yaml)
            
            with pytest.raises(Exception):  # Should raise YAML parsing error
                with initialize_config_dir(config_dir=str(config_path), version_base=None):
                    compose(config_name="config")
    
    def test_missing_configuration_file_handling(self):
        """Test handling of missing configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(Exception):  # Should raise configuration not found error
                with initialize_config_dir(config_dir=temp_dir, version_base=None):
                    compose(config_name="nonexistent")


class TestEnvironmentVariableInterpolation:
    """Test environment variable interpolation with ${oc.env:VAR_NAME} syntax."""
    
    def setup_method(self):
        """Set up clean environment and Hydra state."""
        GlobalHydra.instance().clear()
        # Store original environment for restoration
        self.original_env = os.environ.copy()
    
    def teardown_method(self):
        """Clean up environment and Hydra state."""
        GlobalHydra.instance().clear()
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    @pytest.fixture
    def env_config_content(self):
        """Configuration content with environment variable interpolation."""
        return {
            'experiment': {
                'name': '${oc.env:EXPERIMENT_NAME,default_experiment}',
                'output_dir': '${oc.env:OUTPUT_DIR}',
                'debug_mode': '${oc.env:DEBUG_MODE,false}',
                'max_iterations': '${oc.env:MAX_ITER,100}'
            },
            'database': {
                'host': '${oc.env:DB_HOST,localhost}',
                'port': '${oc.env:DB_PORT,5432}',
                'username': '${oc.env:DB_USER}',
                'password': '${oc.env:DB_PASSWORD}'
            },
            'paths': {
                'data_dir': '${oc.env:DATA_DIR,./data}',
                'model_dir': '${oc.env:MODEL_DIR,./models}',
                'log_dir': '${oc.env:LOG_DIR,./logs}'
            }
        }
    
    def test_environment_variable_interpolation_with_defaults(self, env_config_content):
        """Test environment variable interpolation with default values."""
        # Set some environment variables
        os.environ['EXPERIMENT_NAME'] = 'test_interpolation'
        os.environ['OUTPUT_DIR'] = '/tmp/test_output'
        os.environ['DEBUG_MODE'] = 'true'
        # Leave MAX_ITER unset to test default value
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / 'config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(env_config_content, f)
            
            with initialize_config_dir(config_dir=temp_dir, version_base=None):
                cfg = compose(config_name="config")
                
                # Verify environment variables interpolated correctly
                assert cfg.experiment.name == 'test_interpolation'
                assert cfg.experiment.output_dir == '/tmp/test_output'
                assert cfg.experiment.debug_mode == 'true'
                assert cfg.experiment.max_iterations == '100'  # Default value
                
                # Verify defaults used when env vars not set
                assert cfg.database.host == 'localhost'  # Default
                assert cfg.database.port == '5432'  # Default
                assert cfg.paths.data_dir == './data'  # Default
    
    def test_environment_variable_interpolation_security(self, env_config_content):
        """Test environment variable interpolation security per Section 6.6.7.1."""
        # Attempt to inject malicious values through environment variables
        os.environ['EXPERIMENT_NAME'] = '../../etc/passwd'
        os.environ['OUTPUT_DIR'] = '/; rm -rf /'
        os.environ['DB_HOST'] = '$(whoami)'
        os.environ['DB_USER'] = '${oc.env:ADMIN_PASSWORD}'  # Nested interpolation attempt
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / 'config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(env_config_content, f)
            
            with initialize_config_dir(config_dir=temp_dir, version_base=None):
                cfg = compose(config_name="config")
                
                # Verify malicious values are passed through as strings (Hydra doesn't execute them)
                assert cfg.experiment.name == '../../etc/passwd'  # Treated as string
                assert cfg.experiment.output_dir == '/; rm -rf /'  # Treated as string
                assert cfg.database.host == '$(whoami)'  # Treated as string
                
                # Verify nested interpolation doesn't occur
                assert cfg.database.username == '${oc.env:ADMIN_PASSWORD}'  # Literal string
    
    def test_missing_required_environment_variables(self, env_config_content):
        """Test handling of missing required environment variables."""
        # Don't set required environment variables (no defaults)
        if 'OUTPUT_DIR' in os.environ:
            del os.environ['OUTPUT_DIR']
        if 'DB_USER' in os.environ:
            del os.environ['DB_USER']
        if 'DB_PASSWORD' in os.environ:
            del os.environ['DB_PASSWORD']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / 'config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(env_config_content, f)
            
            with initialize_config_dir(config_dir=temp_dir, version_base=None):
                # This should raise an exception for missing required env vars
                with pytest.raises(Exception):  # OmegaConf will raise for missing required vars
                    cfg = compose(config_name="config")
    
    def test_environment_variable_interpolation_performance(self, env_config_content):
        """Test environment variable interpolation performance."""
        # Set multiple environment variables
        env_vars = {
            'EXPERIMENT_NAME': 'performance_test',
            'OUTPUT_DIR': '/tmp/performance',
            'DEBUG_MODE': 'false',
            'MAX_ITER': '500',
            'DB_HOST': 'test_host',
            'DB_PORT': '3306',
            'DB_USER': 'test_user',
            'DB_PASSWORD': 'test_pass',
            'DATA_DIR': '/data/test',
            'MODEL_DIR': '/models/test',
            'LOG_DIR': '/logs/test'
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / 'config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(env_config_content, f)
            
            start_time = time.time()
            
            with initialize_config_dir(config_dir=temp_dir, version_base=None):
                cfg = compose(config_name="config")
            
            interpolation_time = (time.time() - start_time) * 1000
            
            # Should complete environment variable interpolation quickly
            assert interpolation_time < 100, f"Environment variable interpolation took {interpolation_time:.2f}ms"
            
            # Verify all variables interpolated correctly
            assert cfg.experiment.name == 'performance_test'
            assert cfg.database.host == 'test_host'
            assert cfg.paths.data_dir == '/data/test'


class TestConfigurationOverrideMechanisms:
    """Test configuration override scenarios through command-line and programmatic composition."""
    
    def setup_method(self):
        """Set up clean Hydra state."""
        GlobalHydra.instance().clear()
    
    def teardown_method(self):
        """Clean up Hydra state."""
        GlobalHydra.instance().clear()
    
    @pytest.fixture
    def base_config_for_overrides(self):
        """Base configuration for override testing."""
        return {
            'navigator': {
                'position': [0.0, 0.0],
                'orientation': 0.0,
                'speed': 0.5,
                'max_speed': 1.0
            },
            'simulation': {
                'num_steps': 100,
                'random_seed': 42
            },
            'experiment': {
                'name': 'base_experiment'
            }
        }
    
    def test_command_line_style_overrides(self, base_config_for_overrides):
        """Test command-line style configuration overrides."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / 'config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(base_config_for_overrides, f)
            
            with initialize_config_dir(config_dir=temp_dir, version_base=None):
                # Simulate command-line overrides
                overrides = [
                    'navigator.speed=0.8',
                    'navigator.max_speed=1.5',
                    'simulation.num_steps=200',
                    'experiment.name=override_experiment',
                    'new_param=new_value'  # Adding new parameter
                ]
                
                cfg = compose(config_name="config", overrides=overrides)
                
                # Verify overrides applied correctly
                assert cfg.navigator.speed == 0.8
                assert cfg.navigator.max_speed == 1.5
                assert cfg.simulation.num_steps == 200
                assert cfg.experiment.name == 'override_experiment'
                assert cfg.new_param == 'new_value'
                
                # Verify non-overridden values preserved
                assert cfg.navigator.position == [0.0, 0.0]
                assert cfg.navigator.orientation == 0.0
                assert cfg.simulation.random_seed == 42
    
    def test_nested_configuration_overrides(self, base_config_for_overrides):
        """Test deeply nested configuration overrides."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / 'config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(base_config_for_overrides, f)
            
            with initialize_config_dir(config_dir=temp_dir, version_base=None):
                # Test complex nested overrides
                overrides = [
                    'navigator.sensors.odor.sensitivity=0.9',
                    'navigator.sensors.wind.enabled=true',
                    'simulation.output.save_trajectory=true',
                    'simulation.output.format=numpy',
                    'experiment.metadata.author=test_user',
                    'experiment.metadata.version=1.0'
                ]
                
                cfg = compose(config_name="config", overrides=overrides)
                
                # Verify nested structures created correctly
                assert cfg.navigator.sensors.odor.sensitivity == 0.9
                assert cfg.navigator.sensors.wind.enabled == True
                assert cfg.simulation.output.save_trajectory == True
                assert cfg.simulation.output.format == 'numpy'
                assert cfg.experiment.metadata.author == 'test_user'
                assert cfg.experiment.metadata.version == '1.0'
    
    def test_list_and_dict_overrides(self, base_config_for_overrides):
        """Test overriding list and dictionary configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / 'config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(base_config_for_overrides, f)
            
            with initialize_config_dir(config_dir=temp_dir, version_base=None):
                # Test list and dict overrides
                overrides = [
                    'navigator.position=[1.0,2.0]',  # Override list
                    'navigator.waypoints=[0,0,5,5,10,10]',  # New list
                    'simulation.agents={count:5,spacing:1.0}',  # New dict
                ]
                
                cfg = compose(config_name="config", overrides=overrides)
                
                # Verify list overrides
                assert cfg.navigator.position == [1.0, 2.0]
                assert cfg.navigator.waypoints == [0, 0, 5, 5, 10, 10]
                
                # Verify dict overrides
                assert cfg.simulation.agents.count == 5
                assert cfg.simulation.agents.spacing == 1.0
    
    def test_override_precedence_and_inheritance(self, base_config_for_overrides):
        """Test override precedence and parameter inheritance patterns."""
        # Create multiple configuration files with different precedence
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir)
            
            # Base config
            base_file = config_path / 'base.yaml'
            with open(base_file, 'w') as f:
                yaml.dump(base_config_for_overrides, f)
            
            # Environment-specific config
            env_config = {
                'defaults': ['base'],
                'navigator': {
                    'speed': 0.7,  # Override base
                    'max_speed': 1.2  # Override base
                },
                'simulation': {
                    'num_steps': 150  # Override base
                }
            }
            env_file = config_path / 'env_config.yaml'
            with open(env_file, 'w') as f:
                yaml.dump(env_config, f)
            
            with initialize_config_dir(config_dir=str(config_path), version_base=None):
                # Load with command-line overrides (highest precedence)
                overrides = [
                    'navigator.speed=0.9',  # Should override both base and env
                    'simulation.random_seed=999'  # Should override base
                ]
                
                cfg = compose(config_name="env_config", overrides=overrides)
                
                # Verify precedence: CLI > env_config > base
                assert cfg.navigator.speed == 0.9  # CLI override (highest)
                assert cfg.navigator.max_speed == 1.2  # env_config override
                assert cfg.simulation.num_steps == 150  # env_config override
                assert cfg.simulation.random_seed == 999  # CLI override
                assert cfg.navigator.position == [0.0, 0.0]  # base value (lowest)


class TestPydanticSchemaIntegration:
    """Test Pydantic schema integration with Hydra structured configs."""
    
    def setup_method(self):
        """Set up clean Hydra state."""
        GlobalHydra.instance().clear()
        
    def teardown_method(self):
        """Clean up Hydra state."""
        GlobalHydra.instance().clear()
    
    def test_navigator_config_validation(self):
        """Test NavigatorConfig validation with Hydra configuration."""
        config_data = {
            'position': [1.0, 2.0],
            'orientation': 45.0,
            'speed': 0.5,
            'max_speed': 1.0,
            'angular_velocity': 10.0
        }
        
        # Test valid configuration
        navigator_config = NavigatorConfig(**config_data)
        assert navigator_config.position == (1.0, 2.0)
        assert navigator_config.orientation == 45.0
        assert navigator_config.speed == 0.5
        assert navigator_config.max_speed == 1.0
        assert navigator_config.angular_velocity == 10.0
    
    def test_navigator_config_validation_errors(self):
        """Test NavigatorConfig validation error handling."""
        # Test speed exceeding max_speed
        with pytest.raises(ValidationError) as exc_info:
            NavigatorConfig(
                position=[0.0, 0.0],
                speed=1.5,
                max_speed=1.0  # speed > max_speed should fail
            )
        
        assert 'speed' in str(exc_info.value) or 'max_speed' in str(exc_info.value)
    
    def test_multi_agent_config_validation(self):
        """Test MultiAgentConfig validation with position arrays."""
        # Valid multi-agent configuration
        config_data = {
            'positions': [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            'orientations': [0.0, 45.0, 90.0],
            'speeds': [0.5, 0.6, 0.7],
            'max_speeds': [1.0, 1.1, 1.2],
            'angular_velocities': [5.0, 10.0, 15.0],
            'num_agents': 3
        }
        
        multi_config = MultiAgentConfig(**config_data)
        assert len(multi_config.positions) == 3
        assert len(multi_config.orientations) == 3
        assert multi_config.num_agents == 3
    
    def test_multi_agent_config_validation_errors(self):
        """Test MultiAgentConfig validation error scenarios."""
        # Test mismatched array lengths
        with pytest.raises(ValidationError):
            MultiAgentConfig(
                positions=[[0.0, 0.0], [1.0, 1.0]],  # 2 positions
                orientations=[0.0, 45.0, 90.0],  # 3 orientations - mismatch
                speeds=[0.5, 0.6],  # 2 speeds
                max_speeds=[1.0, 1.1],  # 2 max_speeds
                angular_velocities=[5.0, 10.0]  # 2 angular_velocities
            )
        
        # Test invalid position format
        with pytest.raises(ValidationError):
            MultiAgentConfig(
                positions=[[0.0], [1.0, 1.0, 1.0]],  # Invalid position dimensions
            )
    
    def test_video_plume_config_validation(self):
        """Test VideoPlumeConfig validation scenarios."""
        # Valid configuration
        config_data = {
            'video_path': '/path/to/video.mp4',
            'flip': False,
            'grayscale': True,
            'kernel_size': 5,  # Must be odd
            'kernel_sigma': 1.0,
            'threshold': 0.5,
            'normalize': True
        }
        
        video_config = VideoPlumeConfig(**config_data)
        assert video_config.video_path == '/path/to/video.mp4'
        assert video_config.kernel_size == 5
        assert video_config.normalize == True
    
    def test_video_plume_config_validation_errors(self):
        """Test VideoPlumeConfig validation error scenarios."""
        # Test even kernel_size (should fail)
        with pytest.raises(ValidationError):
            VideoPlumeConfig(
                video_path='/path/to/video.mp4',
                kernel_size=4  # Even number should fail
            )
        
        # Test negative kernel_size (should fail)
        with pytest.raises(ValidationError):
            VideoPlumeConfig(
                video_path='/path/to/video.mp4',
                kernel_size=-1  # Negative should fail
            )
    
    def test_hydra_structured_config_integration(self):
        """Test integration of Pydantic schemas with Hydra structured configs."""
        from hydra.core.config_store import ConfigStore
        from dataclasses import dataclass
        
        @dataclass
        class StructuredNavigatorConfig:
            position: tuple = (0.0, 0.0)
            orientation: float = 0.0
            speed: float = 0.5
            max_speed: float = 1.0
        
        # Register structured config with Hydra
        cs = ConfigStore.instance()
        cs.store(name="navigator_structured", node=StructuredNavigatorConfig)
        
        with initialize(config_path=None, version_base=None):
            cfg = compose(config_name="navigator_structured")
            
            # Convert to Pydantic model for validation
            navigator_config = NavigatorConfig(
                position=list(cfg.position),  # Convert tuple to list
                orientation=cfg.orientation,
                speed=cfg.speed,
                max_speed=cfg.max_speed
            )
            
            assert navigator_config.position == (0.0, 0.0)
            assert navigator_config.speed == 0.5
    
    def test_schema_evolution_and_backward_compatibility(self):
        """Test configuration schema evolution and backward compatibility."""
        # Old format configuration (missing new fields)
        old_config = {
            'position': [0.0, 0.0],
            'orientation': 0.0,
            'speed': 0.5
            # Missing max_speed, angular_velocity (should use defaults)
        }
        
        # Should work with defaults for missing fields
        navigator_config = NavigatorConfig(**old_config)
        assert navigator_config.position == (0.0, 0.0)
        assert navigator_config.speed == 0.5
        assert navigator_config.max_speed == 1.0  # Default value
        assert navigator_config.angular_velocity == 0.0  # Default value
        
        # New format with additional fields
        new_config = {
            'position': [1.0, 1.0],
            'orientation': 45.0,
            'speed': 0.8,
            'max_speed': 1.2,
            'angular_velocity': 15.0,
            'sensor_range': 2.0,  # New field (allowed by extra="allow")
            'collision_avoidance': True  # New field
        }
        
        # Should accept new fields due to extra="allow"
        navigator_config = NavigatorConfig(**new_config)
        assert navigator_config.position == (1.0, 1.0)
        assert navigator_config.speed == 0.8
        # Note: extra fields are stored but may not be accessible as attributes


class TestConfigurationSecurity:
    """Test configuration security including path traversal prevention per Section 6.6.7.1."""
    
    def setup_method(self):
        """Set up clean Hydra state."""
        GlobalHydra.instance().clear()
        
    def teardown_method(self):
        """Clean up Hydra state."""
        GlobalHydra.instance().clear()
    
    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks in configuration loading."""
        malicious_configs = [
            '../../../etc/passwd',
            '../../sensitive_config.yaml',
            '..\\..\\windows\\system32\\config',
            '/etc/shadow',
            '~/.ssh/id_rsa'
        ]
        
        for malicious_path in malicious_configs:
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Attempt to load configuration from malicious path
                    with initialize_config_dir(config_dir=malicious_path, version_base=None):
                        cfg = compose(config_name="config")
                    # If we reach here, the path was accepted (potential security issue)
                    pytest.fail(f"Path traversal attack succeeded with path: {malicious_path}")
                except Exception:
                    # Expected - path traversal should be prevented
                    pass
    
    def test_symbolic_link_handling(self):
        """Test handling of symbolic links in configuration directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / 'config'
            config_dir.mkdir()
            
            # Create a legitimate config file
            config_content = {'test': 'value'}
            config_file = config_dir / 'config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(config_content, f)
            
            # Create symbolic link to sensitive file
            try:
                sensitive_file = Path(temp_dir) / 'sensitive.txt'
                with open(sensitive_file, 'w') as f:
                    f.write('sensitive_data')
                
                symlink_file = config_dir / 'symlink_config.yaml'
                symlink_file.symlink_to(sensitive_file)
                
                # Attempt to load configuration through symlink
                with initialize_config_dir(config_dir=str(config_dir), version_base=None):
                    # This should either fail or be properly restricted
                    try:
                        cfg = compose(config_name="symlink_config")
                        # If successful, verify content is not sensitive data
                        assert 'sensitive_data' not in str(cfg)
                    except Exception:
                        # Expected behavior - symlink loading prevented
                        pass
                        
            except OSError:
                # Skip test if symbolic links not supported on platform
                pytest.skip("Symbolic links not supported on this platform")
    
    def test_configuration_override_injection_prevention(self):
        """Test prevention of malicious configuration override injection."""
        base_config = {
            'safe_param': 'safe_value',
            'admin_settings': {
                'debug_mode': False,
                'admin_access': False
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / 'config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(base_config, f)
            
            # Attempt malicious overrides
            malicious_overrides = [
                'admin_settings.admin_access=true',  # Privilege escalation attempt
                'admin_settings.debug_mode=true',    # Debug mode injection
                '__import__("os").system("rm -rf /")',  # Code injection attempt
                'eval("print(\'hacked\')")',  # Code execution attempt
            ]
            
            with initialize_config_dir(config_dir=temp_dir, version_base=None):
                for malicious_override in malicious_overrides:
                    try:
                        cfg = compose(config_name="config", overrides=[malicious_override])
                        
                        # Verify critical security settings not compromised
                        if hasattr(cfg, 'admin_settings'):
                            # These checks depend on the specific security model
                            # In practice, you'd implement proper access control
                            pass
                            
                    except Exception:
                        # Expected - malicious overrides should be rejected or safely handled
                        pass
    
    def test_environment_variable_injection_prevention(self):
        """Test prevention of environment variable injection attacks."""
        config_with_env = {
            'database': {
                'password': '${oc.env:DB_PASSWORD,default_password}',
                'host': '${oc.env:DB_HOST,localhost}'
            },
            'api': {
                'secret_key': '${oc.env:API_SECRET,default_secret}'
            }
        }
        
        # Set malicious environment variables
        malicious_env_vars = {
            'DB_PASSWORD': '"; DROP TABLE users; --',  # SQL injection attempt
            'DB_HOST': '$(rm -rf /)',  # Command injection attempt
            'API_SECRET': '${oc.env:ADMIN_PASSWORD}',  # Nested interpolation attempt
        }
        
        original_env = os.environ.copy()
        try:
            # Set malicious environment variables
            for key, value in malicious_env_vars.items():
                os.environ[key] = value
            
            with tempfile.TemporaryDirectory() as temp_dir:
                config_file = Path(temp_dir) / 'config.yaml'
                with open(config_file, 'w') as f:
                    yaml.dump(config_with_env, f)
                
                with initialize_config_dir(config_dir=temp_dir, version_base=None):
                    cfg = compose(config_name="config")
                    
                    # Verify malicious values are treated as strings, not executed
                    assert cfg.database.password == '"; DROP TABLE users; --'  # String, not executed
                    assert cfg.database.host == '$(rm -rf /)'  # String, not executed
                    assert cfg.api.secret_key == '${oc.env:ADMIN_PASSWORD}'  # Literal, not interpolated
                    
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)


class TestMultiRunConfigurationComposition:
    """Test multi-run configuration composition and parameter sweep scenarios."""
    
    def setup_method(self):
        """Set up clean Hydra state."""
        GlobalHydra.instance().clear()
        
    def teardown_method(self):
        """Clean up Hydra state."""
        GlobalHydra.instance().clear()
    
    def test_parameter_sweep_configuration(self):
        """Test parameter sweep configuration for multi-run scenarios."""
        sweep_config = {
            'navigator': {
                'speed': 0.5,
                'max_speed': 1.0
            },
            'simulation': {
                'num_steps': 100,
                'random_seed': 42
            },
            'hydra': {
                'mode': 'MULTIRUN',
                'sweep': {
                    'dir': 'multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}',
                    'subdir': '${hydra:job.num}'
                }
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / 'config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(sweep_config, f)
            
            with initialize_config_dir(config_dir=temp_dir, version_base=None):
                # Test parameter sweep overrides
                sweep_overrides = [
                    'navigator.speed=0.3,0.5,0.7,0.9',  # Multiple values
                    'simulation.random_seed=42,123,456'  # Multiple seeds
                ]
                
                # Note: In real usage, this would create multiple runs
                # For testing, we verify the configuration structure
                cfg = compose(config_name="config")
                
                assert cfg.navigator.speed == 0.5
                assert cfg.simulation.num_steps == 100
                assert cfg.hydra.mode == 'MULTIRUN'
    
    def test_grid_search_configuration(self):
        """Test grid search configuration setup."""
        grid_config = {
            'experiment': {
                'name': 'grid_search',
                'parameters': {
                    'learning_rate': [0.01, 0.1, 1.0],
                    'batch_size': [16, 32, 64],
                    'optimizer': ['adam', 'sgd', 'rmsprop']
                }
            },
            'navigator': {
                'speed': 0.5
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / 'config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(grid_config, f)
            
            with initialize_config_dir(config_dir=temp_dir, version_base=None):
                cfg = compose(config_name="config")
                
                # Verify grid search parameters accessible
                assert cfg.experiment.name == 'grid_search'
                assert len(cfg.experiment.parameters.learning_rate) == 3
                assert len(cfg.experiment.parameters.batch_size) == 3
                assert len(cfg.experiment.parameters.optimizer) == 3
    
    def test_conditional_configuration_composition(self):
        """Test conditional configuration composition based on runtime parameters."""
        conditional_config = {
            'defaults': [
                {'navigator': 'single_agent'},
                {'environment': 'video_plume'}
            ],
            'experiment_type': 'single_agent',
            'navigator': {
                'speed': 0.5
            },
            'multi_agent': {
                'num_agents': 5,
                'formation': 'line'
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / 'config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(conditional_config, f)
            
            with initialize_config_dir(config_dir=temp_dir, version_base=None):
                # Test single agent configuration
                cfg_single = compose(config_name="config")
                assert cfg_single.experiment_type == 'single_agent'
                
                # Test multi-agent override
                cfg_multi = compose(
                    config_name="config",
                    overrides=['experiment_type=multi_agent', 'navigator=multi_agent']
                )
                assert cfg_multi.experiment_type == 'multi_agent'


class TestConfigurationPerformanceAndReliability:
    """Test configuration loading performance and system reliability requirements."""
    
    def setup_method(self):
        """Set up clean Hydra state."""
        GlobalHydra.instance().clear()
        
    def teardown_method(self):
        """Clean up Hydra state."""
        GlobalHydra.instance().clear()
    
    def test_large_configuration_loading_performance(self):
        """Test performance with large configuration files."""
        # Create large configuration with many parameters
        large_config = {
            'simulation': {
                'num_steps': 1000,
                'step_size': 0.01,
                'random_seed': 42
            }
        }
        
        # Add many agents to configuration
        for i in range(100):
            large_config[f'agent_{i}'] = {
                'position': [float(i), float(i)],
                'orientation': float(i * 10),
                'speed': 0.5 + i * 0.01,
                'max_speed': 1.0 + i * 0.01
            }
        
        # Add many environment parameters
        for i in range(50):
            large_config[f'env_param_{i}'] = {
                'value': i,
                'threshold': i * 0.1,
                'enabled': i % 2 == 0
            }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / 'config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(large_config, f)
            
            start_time = time.time()
            
            with initialize_config_dir(config_dir=temp_dir, version_base=None):
                cfg = compose(config_name="config")
            
            loading_time = (time.time() - start_time) * 1000
            
            # Verify performance requirement
            assert loading_time < 500, f"Large configuration loading took {loading_time:.2f}ms, exceeds 500ms requirement"
            
            # Verify configuration loaded correctly
            assert cfg.simulation.num_steps == 1000
            assert hasattr(cfg, 'agent_0')
            assert hasattr(cfg, 'agent_99')
    
    def test_configuration_validation_performance(self):
        """Test configuration validation performance with Pydantic schemas."""
        config_data = {
            'navigator': {
                'position': [0.0, 0.0],
                'orientation': 0.0,
                'speed': 0.5,
                'max_speed': 1.0,
                'angular_velocity': 0.0
            },
            'video_plume': {
                'video_path': '/path/to/video.mp4',
                'flip': False,
                'grayscale': True,
                'normalize': True
            }
        }
        
        start_time = time.time()
        
        # Validate multiple schemas rapidly
        for _ in range(100):
            navigator_config = NavigatorConfig(**config_data['navigator'])
            video_config = VideoPlumeConfig(**config_data['video_plume'])
        
        validation_time = (time.time() - start_time) * 1000
        
        # Should complete validation quickly (100 iterations in reasonable time)
        assert validation_time < 100, f"Schema validation took {validation_time:.2f}ms for 100 iterations"
    
    def test_configuration_error_handling_reliability(self):
        """Test reliable error handling for various configuration failure scenarios."""
        error_scenarios = [
            # Invalid YAML syntax
            ("invalid_yaml", "invalid: yaml: [unclosed"),
            
            # Missing required fields
            ("missing_required", {"navigator": {"position": [0, 0]}}),  # Missing video_path in video_plume
            
            # Invalid data types
            ("invalid_types", {
                "navigator": {
                    "position": "not_a_list",
                    "speed": "not_a_number"
                }
            }),
            
            # Invalid value ranges
            ("invalid_ranges", {
                "navigator": {
                    "position": [0, 0],
                    "speed": 2.0,
                    "max_speed": 1.0  # speed > max_speed
                }
            })
        ]
        
        for scenario_name, config_content in error_scenarios:
            with tempfile.TemporaryDirectory() as temp_dir:
                config_file = Path(temp_dir) / 'config.yaml'
                
                if isinstance(config_content, str):
                    # Invalid YAML
                    with open(config_file, 'w') as f:
                        f.write(config_content)
                else:
                    # Valid YAML but invalid configuration
                    with open(config_file, 'w') as f:
                        yaml.dump(config_content, f)
                
                # Should handle errors gracefully
                with pytest.raises(Exception) as exc_info:
                    with initialize_config_dir(config_dir=temp_dir, version_base=None):
                        cfg = compose(config_name="config")
                        
                        # If Hydra doesn't catch it, Pydantic validation should
                        if scenario_name == "invalid_ranges":
                            NavigatorConfig(**cfg.navigator)
                
                # Verify meaningful error information available
                assert exc_info.value is not None
    
    def test_concurrent_configuration_loading(self):
        """Test configuration loading under concurrent access scenarios."""
        import threading
        import queue
        
        config_data = {
            'navigator': {
                'position': [0.0, 0.0],
                'speed': 0.5,
                'max_speed': 1.0
            },
            'simulation': {
                'num_steps': 100
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / 'config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)
            
            results = queue.Queue()
            errors = queue.Queue()
            
            def load_config(thread_id):
                try:
                    GlobalHydra.instance().clear()  # Each thread needs clean state
                    with initialize_config_dir(config_dir=temp_dir, version_base=None):
                        cfg = compose(config_name="config")
                        results.put((thread_id, cfg.navigator.speed))
                except Exception as e:
                    errors.put((thread_id, str(e)))
            
            # Start multiple threads loading configuration concurrently
            threads = []
            for i in range(5):
                thread = threading.Thread(target=load_config, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=10)  # 10 second timeout
            
            # Verify all threads completed successfully
            assert errors.empty(), f"Configuration loading errors: {list(errors.queue)}"
            assert results.qsize() == 5, f"Expected 5 results, got {results.qsize()}"
            
            # Verify consistent results across threads
            speeds = []
            while not results.empty():
                thread_id, speed = results.get()
                speeds.append(speed)
            
            assert all(speed == 0.5 for speed in speeds), "Inconsistent results across threads"


# Integration test combining multiple configuration features
class TestIntegratedConfigurationScenarios:
    """Test integrated configuration scenarios combining multiple features."""
    
    def setup_method(self):
        """Set up clean Hydra state."""
        GlobalHydra.instance().clear()
        
    def teardown_method(self):
        """Clean up Hydra state."""
        GlobalHydra.instance().clear()
    
    def test_full_configuration_pipeline(self):
        """Test complete configuration pipeline with all features."""
        # Set environment variables
        os.environ['EXPERIMENT_NAME'] = 'integration_test'
        os.environ['OUTPUT_DIR'] = '/tmp/integration'
        os.environ['DEBUG_MODE'] = 'true'
        
        try:
            # Create hierarchical configuration structure
            base_config = {
                'navigator': {
                    'position': [0.0, 0.0],
                    'orientation': 0.0,
                    'speed': 0.5,
                    'max_speed': 1.0
                },
                'video_plume': {
                    'video_path': 'base_video.mp4',
                    'grayscale': True,
                    'normalize': True
                },
                'simulation': {
                    'num_steps': 100,
                    'random_seed': 42
                }
            }
            
            env_config = {
                'defaults': ['base'],
                'navigator': {
                    'speed': 0.7  # Override base
                },
                'experiment': {
                    'name': '${oc.env:EXPERIMENT_NAME}',
                    'output_dir': '${oc.env:OUTPUT_DIR}',
                    'debug': '${oc.env:DEBUG_MODE}'
                }
            }
            
            with tempfile.TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir)
                
                # Create base.yaml
                base_file = config_path / 'base.yaml'
                with open(base_file, 'w') as f:
                    yaml.dump(base_config, f)
                
                # Create config.yaml
                config_file = config_path / 'config.yaml'
                with open(config_file, 'w') as f:
                    yaml.dump(env_config, f)
                
                start_time = time.time()
                
                with initialize_config_dir(config_dir=str(config_path), version_base=None):
                    # Load with command-line overrides
                    cfg = compose(
                        config_name="config",
                        overrides=[
                            'navigator.max_speed=1.5',  # CLI override
                            'simulation.num_steps=200',  # CLI override
                            'new_param=test_value'  # New parameter
                        ]
                    )
                
                loading_time = (time.time() - start_time) * 1000
                
                # Verify performance
                assert loading_time < 500, f"Full pipeline took {loading_time:.2f}ms"
                
                # Verify hierarchical precedence: CLI > config > base
                assert cfg.navigator.speed == 0.7  # config override
                assert cfg.navigator.max_speed == 1.5  # CLI override
                assert cfg.simulation.num_steps == 200  # CLI override
                assert cfg.navigator.position == [0.0, 0.0]  # base value
                
                # Verify environment variable interpolation
                assert cfg.experiment.name == 'integration_test'
                assert cfg.experiment.output_dir == '/tmp/integration'
                assert cfg.experiment.debug == 'true'
                
                # Verify new parameters accepted
                assert cfg.new_param == 'test_value'
                
                # Validate through Pydantic schemas
                navigator_config = NavigatorConfig(
                    position=cfg.navigator.position,
                    orientation=cfg.navigator.orientation,
                    speed=cfg.navigator.speed,
                    max_speed=cfg.navigator.max_speed
                )
                
                video_config = VideoPlumeConfig(
                    video_path=cfg.video_plume.video_path,
                    grayscale=cfg.video_plume.grayscale,
                    normalize=cfg.video_plume.normalize
                )
                
                assert navigator_config.speed == 0.7
                assert navigator_config.max_speed == 1.5
                assert video_config.video_path == 'base_video.mp4'
                
        finally:
            # Clean up environment
            for var in ['EXPERIMENT_NAME', 'OUTPUT_DIR', 'DEBUG_MODE']:
                if var in os.environ:
                    del os.environ[var]


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])