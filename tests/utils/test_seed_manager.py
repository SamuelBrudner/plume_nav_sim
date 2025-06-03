"""
Comprehensive test suite for the Random Seed Manager utility (Feature F-014).

This module validates global seed management, deterministic experiment execution,
cross-platform consistency, and Hydra configuration integration. Tests cover
seed initialization, state preservation, NumPy/Python random module coordination,
and reproducibility validation essential for scientific computing workflows.

Testing Framework:
- pytest for test organization and execution
- pytest-hydra for Hydra configuration testing
- unittest.mock for dependency isolation
- numpy.testing for numerical precision validation
- performance timing validation for <100ms requirement

Test Categories:
- Basic functionality and API validation
- Global seed management and coordination
- Cross-platform consistency testing
- Hydra configuration integration
- State preservation and restoration
- Performance and timing validation
- Error handling and edge cases
- Reproducibility and experiment validation
"""

import pytest
import time
import hashlib
import platform
from unittest.mock import patch, MagicMock, call
from contextlib import contextmanager
from typing import Optional, Dict, Any, Union

import numpy as np
import random
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from src.{{cookiecutter.project_slug}}.utils.seed_manager import (
    SeedManager,
    SeedConfig,
    get_seed_manager,
    set_global_seed,
    get_current_seed,
    get_numpy_generator
)


class TestSeedConfig:
    """Test suite for SeedConfig Pydantic model validation."""
    
    def test_seed_config_default_values(self):
        """Test SeedConfig default parameter values."""
        config = SeedConfig()
        
        assert config.seed is None
        assert config.numpy_seed is None
        assert config.python_seed is None
        assert config.auto_seed is True
        assert config.hash_environment is True
        assert config.validate_initialization is True
        assert config.preserve_state is False
        assert config.log_seed_context is True
    
    def test_seed_config_explicit_values(self):
        """Test SeedConfig with explicit parameter values."""
        config = SeedConfig(
            seed=42,
            numpy_seed=123,
            python_seed=456,
            auto_seed=False,
            hash_environment=False,
            validate_initialization=False,
            preserve_state=True,
            log_seed_context=False
        )
        
        assert config.seed == 42
        assert config.numpy_seed == 123
        assert config.python_seed == 456
        assert config.auto_seed is False
        assert config.hash_environment is False
        assert config.validate_initialization is False
        assert config.preserve_state is True
        assert config.log_seed_context is False
    
    def test_seed_config_validation_strict_mode(self):
        """Test SeedConfig rejects unknown fields in strict mode."""
        with pytest.raises(ValueError, match="extra fields not permitted"):
            SeedConfig(invalid_field="not_allowed")
    
    def test_seed_config_field_descriptions(self):
        """Test SeedConfig field metadata and descriptions."""
        schema = SeedConfig.model_json_schema()
        
        # Verify field descriptions exist
        properties = schema["properties"]
        assert "description" in properties["seed"]
        assert "Global random seed" in properties["seed"]["description"]
        assert "description" in properties["auto_seed"]
        assert "Automatically generate seed" in properties["auto_seed"]["description"]


class TestSeedManagerSingleton:
    """Test suite for SeedManager singleton pattern implementation."""
    
    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()
    
    def teardown_method(self):
        """Clean up singleton state after each test."""
        SeedManager.reset()
    
    def test_singleton_pattern(self):
        """Test SeedManager implements singleton pattern correctly."""
        manager1 = SeedManager()
        manager2 = SeedManager()
        manager3 = get_seed_manager()
        
        assert manager1 is manager2
        assert manager2 is manager3
        assert id(manager1) == id(manager2) == id(manager3)
    
    def test_singleton_reset(self):
        """Test singleton reset functionality for testing purposes."""
        manager1 = SeedManager()
        original_id = id(manager1)
        
        SeedManager.reset()
        
        manager2 = SeedManager()
        new_id = id(manager2)
        
        assert original_id != new_id
        assert manager1 is not manager2
    
    def test_singleton_initialization_once(self):
        """Test singleton initializes internal state only once."""
        with patch('src.{{cookiecutter.project_slug}}.utils.seed_manager.logger.bind') as mock_logger:
            manager1 = SeedManager()
            manager2 = SeedManager()
            
            # Logger binding should only happen once during first initialization
            assert mock_logger.call_count == 1


class TestSeedManagerInitialization:
    """Test suite for SeedManager initialization and configuration loading."""
    
    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()
    
    def teardown_method(self):
        """Clean up singleton state after each test."""
        SeedManager.reset()
    
    def test_initialize_with_explicit_seed(self):
        """Test initialization with explicit seed value."""
        manager = SeedManager()
        config = SeedConfig(seed=42)
        
        result_seed = manager.initialize(config=config)
        
        assert result_seed == 42
        assert manager.current_seed == 42
        assert isinstance(manager.numpy_generator, np.random.Generator)
    
    def test_initialize_with_dict_config(self):
        """Test initialization with dictionary configuration."""
        manager = SeedManager()
        config_dict = {"seed": 123, "validate_initialization": False}
        
        result_seed = manager.initialize(config=config_dict)
        
        assert result_seed == 123
        assert manager.current_seed == 123
    
    def test_initialize_with_omegaconf_config(self):
        """Test initialization with OmegaConf DictConfig."""
        manager = SeedManager()
        config_dict = {"seed": 456, "hash_environment": False}
        omega_config = OmegaConf.create(config_dict)
        
        result_seed = manager.initialize(config=omega_config)
        
        assert result_seed == 456
        assert manager.current_seed == 456
    
    def test_initialize_with_auto_seed_generation(self):
        """Test initialization with automatic seed generation."""
        manager = SeedManager()
        config = SeedConfig(seed=None, auto_seed=True)
        
        result_seed = manager.initialize(config=config)
        
        assert isinstance(result_seed, int)
        assert 0 <= result_seed <= 2**32 - 1
        assert manager.current_seed == result_seed
    
    def test_initialize_performance_requirement(self):
        """Test initialization meets <100ms performance requirement."""
        manager = SeedManager()
        config = SeedConfig(seed=42)
        
        start_time = time.perf_counter()
        manager.initialize(config=config)
        end_time = time.perf_counter()
        
        initialization_time_ms = (end_time - start_time) * 1000
        assert initialization_time_ms < 100, f"Initialization took {initialization_time_ms:.2f}ms > 100ms"
    
    def test_initialize_run_id_generation(self):
        """Test initialization generates proper run identifier."""
        manager = SeedManager()
        config = SeedConfig(seed=42)
        
        manager.initialize(config=config)
        
        assert manager.run_id is not None
        assert manager.run_id.startswith("run_")
        assert len(manager.run_id) == 12  # "run_" + 8 character hash
    
    def test_initialize_with_custom_run_id(self):
        """Test initialization with custom run identifier."""
        manager = SeedManager()
        config = SeedConfig(seed=42)
        custom_run_id = "experiment_2024_001"
        
        manager.initialize(config=config, run_id=custom_run_id)
        
        assert manager.run_id == custom_run_id
    
    def test_initialize_environment_hash_generation(self):
        """Test initialization generates environment hash for consistency."""
        manager = SeedManager()
        config = SeedConfig(seed=42, hash_environment=True)
        
        manager.initialize(config=config)
        
        assert manager.environment_hash is not None
        assert len(manager.environment_hash) == 8  # MD5 hash first 8 characters
        assert isinstance(manager.environment_hash, str)
    
    def test_initialize_without_config_uses_defaults(self):
        """Test initialization without config uses SeedConfig defaults."""
        manager = SeedManager()
        
        with patch.object(manager, '_load_from_hydra', return_value={}):
            result_seed = manager.initialize(config=None)
        
        assert isinstance(result_seed, int)
        assert 0 <= result_seed <= 2**32 - 1


class TestSeedManagerHydraIntegration:
    """Test suite for Hydra configuration system integration."""
    
    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()
    
    def teardown_method(self):
        """Clean up singleton state after each test."""
        SeedManager.reset()
    
    def test_load_from_hydra_with_seed_manager_config(self):
        """Test loading configuration from Hydra seed_manager section."""
        manager = SeedManager()
        
        # Mock Hydra global configuration
        mock_config = OmegaConf.create({
            "seed_manager": {
                "seed": 789,
                "validate_initialization": False
            }
        })
        
        with patch('src.{{cookiecutter.project_slug}}.utils.seed_manager.GlobalHydra') as mock_hydra:
            mock_hydra.return_value.is_initialized.return_value = True
            mock_hydra.instance.return_value.cfg = mock_config
            
            result_seed = manager.initialize(config=None)
        
        assert result_seed == 789
    
    def test_load_from_hydra_with_direct_seed_parameter(self):
        """Test loading configuration from Hydra direct seed parameter."""
        manager = SeedManager()
        
        # Mock Hydra global configuration with direct seed
        mock_config = OmegaConf.create({"seed": 999})
        
        with patch('src.{{cookiecutter.project_slug}}.utils.seed_manager.GlobalHydra') as mock_hydra:
            mock_hydra.return_value.is_initialized.return_value = True
            mock_hydra.instance.return_value.cfg = mock_config
            
            result_seed = manager.initialize(config=None)
        
        assert result_seed == 999
    
    def test_load_from_hydra_not_initialized(self):
        """Test handling when Hydra is not initialized."""
        manager = SeedManager()
        
        with patch('src.{{cookiecutter.project_slug}}.utils.seed_manager.GlobalHydra') as mock_hydra:
            mock_hydra.return_value.is_initialized.return_value = False
            
            # Should fall back to auto-seed generation
            result_seed = manager.initialize(config=None)
        
        assert isinstance(result_seed, int)
        assert 0 <= result_seed <= 2**32 - 1
    
    def test_load_from_hydra_exception_handling(self):
        """Test graceful handling of Hydra loading exceptions."""
        manager = SeedManager()
        
        with patch('src.{{cookiecutter.project_slug}}.utils.seed_manager.GlobalHydra') as mock_hydra:
            mock_hydra.return_value.is_initialized.side_effect = Exception("Hydra error")
            
            # Should handle exception and proceed with defaults
            result_seed = manager.initialize(config=None)
        
        assert isinstance(result_seed, int)
        assert 0 <= result_seed <= 2**32 - 1


class TestGlobalSeedManagement:
    """Test suite for global seed management functionality."""
    
    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()
    
    def teardown_method(self):
        """Clean up singleton state after each test."""
        SeedManager.reset()
    
    def test_set_global_seed_basic(self):
        """Test basic global seed setting functionality."""
        seed = set_global_seed(42)
        
        assert seed == 42
        assert get_current_seed() == 42
    
    def test_set_global_seed_with_config(self):
        """Test global seed setting with SeedConfig."""
        config = SeedConfig(seed=123, validate_initialization=False)
        seed = set_global_seed(config=config)
        
        assert seed == 123
        assert get_current_seed() == 123
    
    def test_set_global_seed_override_config_seed(self):
        """Test seed parameter overrides config seed value."""
        config = SeedConfig(seed=999)
        seed = set_global_seed(seed=456, config=config)
        
        assert seed == 456
        assert get_current_seed() == 456
    
    def test_set_global_seed_auto_generation(self):
        """Test global seed auto-generation."""
        seed = set_global_seed()
        
        assert isinstance(seed, int)
        assert 0 <= seed <= 2**32 - 1
        assert get_current_seed() == seed
    
    def test_get_current_seed_before_initialization(self):
        """Test get_current_seed returns None before initialization."""
        current_seed = get_current_seed()
        assert current_seed is None
    
    def test_get_numpy_generator(self):
        """Test NumPy generator retrieval."""
        set_global_seed(42)
        generator = get_numpy_generator()
        
        assert isinstance(generator, np.random.Generator)
        
        # Test generator produces deterministic results
        value1 = generator.random()
        
        # Reset and test again
        set_global_seed(42)
        generator2 = get_numpy_generator()
        value2 = generator2.random()
        
        assert value1 == value2


class TestRandomNumberCoordination:
    """Test suite for NumPy and Python random module coordination."""
    
    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()
    
    def teardown_method(self):
        """Clean up singleton state after each test."""
        SeedManager.reset()
    
    def test_numpy_python_random_coordination(self):
        """Test NumPy and Python random modules are coordinated."""
        set_global_seed(42)
        
        # Generate values from both modules
        python_value1 = random.random()
        numpy_value1 = np.random.random()
        numpy_gen_value1 = get_numpy_generator().random()
        
        # Reset seed and generate again
        set_global_seed(42)
        
        python_value2 = random.random()
        numpy_value2 = np.random.random()
        numpy_gen_value2 = get_numpy_generator().random()
        
        # Values should be identical due to seed coordination
        assert python_value1 == python_value2
        assert numpy_value1 == numpy_value2
        assert numpy_gen_value1 == numpy_gen_value2
    
    def test_separate_numpy_python_seeds(self):
        """Test separate NumPy and Python seeds functionality."""
        config = SeedConfig(
            seed=42,
            numpy_seed=123,
            python_seed=456
        )
        set_global_seed(config=config)
        
        # Verify different seeds were used
        manager = get_seed_manager()
        assert manager.current_seed == 42
        
        # Test deterministic behavior with separate seeds
        python_value = random.random()
        numpy_value = np.random.random()
        
        # Reset with same config
        SeedManager.reset()
        set_global_seed(config=config)
        
        python_value2 = random.random()
        numpy_value2 = np.random.random()
        
        assert python_value == python_value2
        assert numpy_value == numpy_value2
    
    def test_numpy_generator_independence(self):
        """Test NumPy generator provides independent random state."""
        set_global_seed(42)
        generator = get_numpy_generator()
        
        # Generate from both legacy and modern NumPy interfaces
        legacy_value = np.random.random()
        generator_value = generator.random()
        
        # Values should be different (independent streams)
        assert legacy_value != generator_value
        
        # But both should be deterministic
        set_global_seed(42)
        generator2 = get_numpy_generator()
        
        legacy_value2 = np.random.random()
        generator_value2 = generator2.random()
        
        assert legacy_value == legacy_value2
        assert generator_value == generator_value2


class TestStatePreservation:
    """Test suite for random state preservation and restoration."""
    
    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()
    
    def teardown_method(self):
        """Clean up singleton state after each test."""
        SeedManager.reset()
    
    def test_state_preservation_enabled(self):
        """Test state preservation when enabled in configuration."""
        config = SeedConfig(seed=42, preserve_state=True)
        set_global_seed(config=config)
        
        manager = get_seed_manager()
        state = manager.get_state()
        
        assert state is not None
        assert 'python_state' in state
        assert 'numpy_legacy_state' in state
        assert 'numpy_generator_state' in state
        assert 'seed' in state
        assert state['seed'] == 42
    
    def test_state_preservation_disabled(self):
        """Test state preservation when disabled in configuration."""
        config = SeedConfig(seed=42, preserve_state=False)
        set_global_seed(config=config)
        
        manager = get_seed_manager()
        state = manager.get_state()
        
        assert state is None
    
    def test_state_restoration(self):
        """Test random state restoration from saved state."""
        config = SeedConfig(seed=42, preserve_state=True)
        set_global_seed(config=config)
        
        manager = get_seed_manager()
        
        # Generate some values to change state
        random.random()
        np.random.random()
        get_numpy_generator().random()
        
        # Save state before generating more values
        saved_state = manager.get_state()
        
        # Generate more values
        value_before = random.random()
        
        # Restore previous state
        manager.restore_state(saved_state)
        
        # Should get same value as before restoration
        value_after = random.random()
        assert value_before == value_after
    
    def test_state_restoration_without_preservation(self):
        """Test state restoration fails when preservation not enabled."""
        config = SeedConfig(seed=42, preserve_state=False)
        set_global_seed(config=config)
        
        manager = get_seed_manager()
        fake_state = {'python_state': None}
        
        with pytest.raises(RuntimeError, match="State preservation not enabled"):
            manager.restore_state(fake_state)
    
    def test_temporary_seed_context(self):
        """Test temporary seed context manager."""
        config = SeedConfig(seed=42, preserve_state=True)
        set_global_seed(config=config)
        
        manager = get_seed_manager()
        original_value = random.random()
        
        # Use temporary seed
        with manager.temporary_seed(999):
            temp_value = random.random()
            assert get_current_seed() == 999
        
        # Should restore original seed
        assert get_current_seed() == 42
        restored_value = random.random()
        
        # Values should be different due to temporary seed usage
        assert temp_value != original_value
        assert temp_value != restored_value
    
    def test_temporary_seed_without_preservation(self):
        """Test temporary seed context requires state preservation."""
        config = SeedConfig(seed=42, preserve_state=False)
        set_global_seed(config=config)
        
        manager = get_seed_manager()
        
        with pytest.raises(RuntimeError, match="Temporary seed requires preserve_state=True"):
            with manager.temporary_seed(999):
                pass


class TestCrossPlatformConsistency:
    """Test suite for cross-platform consistency validation."""
    
    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()
    
    def teardown_method(self):
        """Clean up singleton state after each test."""
        SeedManager.reset()
    
    def test_environment_hash_consistency(self):
        """Test environment hash generation is consistent."""
        set_global_seed(42)
        manager = get_seed_manager()
        
        hash1 = manager.environment_hash
        
        # Reset and initialize again
        SeedManager.reset()
        set_global_seed(42)
        manager2 = get_seed_manager()
        
        hash2 = manager2.environment_hash
        
        # Environment hash should be identical on same platform
        assert hash1 == hash2
        assert len(hash1) == 8
    
    def test_entropy_seed_with_environment_hashing(self):
        """Test entropy-based seed generation with environment hashing."""
        config = SeedConfig(
            seed=None,
            auto_seed=True,
            hash_environment=True
        )
        
        # Generate multiple seeds - should be different due to timing
        seeds = []
        for _ in range(3):
            SeedManager.reset()
            seed = set_global_seed(config=config)
            seeds.append(seed)
            time.sleep(0.001)  # Ensure timestamp difference
        
        # All seeds should be different
        assert len(set(seeds)) == len(seeds)
        
        # All seeds should be in valid range
        for seed in seeds:
            assert 0 <= seed <= 2**32 - 1
    
    def test_entropy_seed_without_environment_hashing(self):
        """Test entropy-based seed generation without environment hashing."""
        config = SeedConfig(
            seed=None,
            auto_seed=True,
            hash_environment=False
        )
        
        seed = set_global_seed(config=config)
        
        assert isinstance(seed, int)
        assert 0 <= seed <= 2**32 - 1
    
    @patch('src.{{cookiecutter.project_slug}}.utils.seed_manager.platform.platform')
    @patch('src.{{cookiecutter.project_slug}}.utils.seed_manager.platform.python_version')
    @patch('src.{{cookiecutter.project_slug}}.utils.seed_manager.np.__version__')
    def test_environment_hash_factors(self, mock_np_version, mock_py_version, mock_platform):
        """Test environment hash includes relevant platform factors."""
        mock_platform.return_value = "Linux-5.4.0"
        mock_py_version.return_value = "3.9.0"
        mock_np_version.__get__ = lambda obj, objtype: "1.21.0"
        
        set_global_seed(42)
        manager = get_seed_manager()
        hash1 = manager.environment_hash
        
        # Change platform and verify hash changes
        SeedManager.reset()
        mock_platform.return_value = "Windows-10"
        
        set_global_seed(42)
        manager2 = get_seed_manager()
        hash2 = manager2.environment_hash
        
        # Hashes should be different due to platform change
        assert hash1 != hash2


class TestPerformanceValidation:
    """Test suite for performance requirements validation."""
    
    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()
    
    def teardown_method(self):
        """Clean up singleton state after each test."""
        SeedManager.reset()
    
    def test_initialization_performance_basic(self):
        """Test basic initialization meets performance requirement."""
        config = SeedConfig(seed=42, validate_initialization=False)
        
        start_time = time.perf_counter()
        set_global_seed(config=config)
        end_time = time.perf_counter()
        
        duration_ms = (end_time - start_time) * 1000
        assert duration_ms < 100, f"Initialization took {duration_ms:.2f}ms > 100ms"
    
    def test_initialization_performance_with_validation(self):
        """Test initialization with validation meets performance requirement."""
        config = SeedConfig(seed=42, validate_initialization=True)
        
        start_time = time.perf_counter()
        set_global_seed(config=config)
        end_time = time.perf_counter()
        
        duration_ms = (end_time - start_time) * 1000
        assert duration_ms < 100, f"Initialization with validation took {duration_ms:.2f}ms > 100ms"
    
    def test_initialization_performance_with_state_preservation(self):
        """Test initialization with state preservation meets performance requirement."""
        config = SeedConfig(
            seed=42,
            validate_initialization=True,
            preserve_state=True,
            log_seed_context=True
        )
        
        start_time = time.perf_counter()
        set_global_seed(config=config)
        end_time = time.perf_counter()
        
        duration_ms = (end_time - start_time) * 1000
        assert duration_ms < 100, f"Full initialization took {duration_ms:.2f}ms > 100ms"
    
    def test_state_operations_performance(self):
        """Test state preservation operations performance."""
        config = SeedConfig(seed=42, preserve_state=True)
        set_global_seed(config=config)
        manager = get_seed_manager()
        
        # Test state capture performance
        start_time = time.perf_counter()
        state = manager.get_state()
        capture_time = (time.perf_counter() - start_time) * 1000
        
        assert capture_time < 10, f"State capture took {capture_time:.2f}ms > 10ms"
        
        # Test state restoration performance
        start_time = time.perf_counter()
        manager.restore_state(state)
        restore_time = (time.perf_counter() - start_time) * 1000
        
        assert restore_time < 10, f"State restoration took {restore_time:.2f}ms > 10ms"


class TestErrorHandling:
    """Test suite for error handling and edge cases."""
    
    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()
    
    def teardown_method(self):
        """Clean up singleton state after each test."""
        SeedManager.reset()
    
    def test_invalid_seed_range_negative(self):
        """Test error handling for negative seed values."""
        manager = SeedManager()
        config = SeedConfig(seed=-1)
        
        with pytest.raises(ValueError, match="Seed must be in range"):
            manager.initialize(config=config)
    
    def test_invalid_seed_range_too_large(self):
        """Test error handling for seed values exceeding 32-bit range."""
        manager = SeedManager()
        config = SeedConfig(seed=2**32)
        
        with pytest.raises(ValueError, match="Seed must be in range"):
            manager.initialize(config=config)
    
    def test_no_seed_auto_disabled(self):
        """Test error when no seed provided and auto-generation disabled."""
        manager = SeedManager()
        config = SeedConfig(seed=None, auto_seed=False)
        
        with pytest.raises(ValueError, match="No seed provided and auto_seed is disabled"):
            manager.initialize(config=config)
    
    def test_initialization_exception_handling(self):
        """Test proper exception handling during initialization."""
        manager = SeedManager()
        
        with patch.object(manager, '_initialize_generators', side_effect=Exception("Generator error")):
            with pytest.raises(RuntimeError, match="Seed manager initialization failed"):
                manager.initialize(config=SeedConfig(seed=42))
    
    def test_state_restoration_invalid_format(self):
        """Test error handling for invalid state format during restoration."""
        config = SeedConfig(seed=42, preserve_state=True)
        set_global_seed(config=config)
        manager = get_seed_manager()
        
        invalid_state = {"invalid_format": True}
        
        with pytest.raises(RuntimeError, match="State restoration failed"):
            manager.restore_state(invalid_state)
    
    def test_state_restoration_corrupted_data(self):
        """Test error handling for corrupted state data."""
        config = SeedConfig(seed=42, preserve_state=True)
        set_global_seed(config=config)
        manager = get_seed_manager()
        
        # Create corrupted state that will cause restoration to fail
        corrupted_state = {
            'python_state': "not_a_valid_state",
            'numpy_legacy_state': None,
            'seed': 42
        }
        
        with pytest.raises(RuntimeError, match="State restoration failed"):
            manager.restore_state(corrupted_state)


class TestExperimentSeedGeneration:
    """Test suite for experiment seed generation functionality."""
    
    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()
    
    def teardown_method(self):
        """Clean up singleton state after each test."""
        SeedManager.reset()
    
    def test_generate_experiment_seeds_basic(self):
        """Test basic experiment seed generation."""
        set_global_seed(42)
        manager = get_seed_manager()
        
        seeds = manager.generate_experiment_seeds(5)
        
        assert len(seeds) == 5
        assert all(isinstance(seed, int) for seed in seeds)
        assert all(0 <= seed <= 2**32 - 1 for seed in seeds)
        
        # Seeds should be deterministic
        seeds2 = manager.generate_experiment_seeds(5)
        assert seeds == seeds2
    
    def test_generate_experiment_seeds_with_base_seed(self):
        """Test experiment seed generation with custom base seed."""
        set_global_seed(42)
        manager = get_seed_manager()
        
        seeds1 = manager.generate_experiment_seeds(3, base_seed=100)
        seeds2 = manager.generate_experiment_seeds(3, base_seed=200)
        
        assert len(seeds1) == 3
        assert len(seeds2) == 3
        assert seeds1 != seeds2  # Different base seeds should produce different sequences
    
    def test_generate_experiment_seeds_deterministic(self):
        """Test experiment seed generation is deterministic."""
        set_global_seed(42)
        manager = get_seed_manager()
        
        seeds1 = manager.generate_experiment_seeds(10, base_seed=123)
        
        # Reset and generate again
        SeedManager.reset()
        set_global_seed(42)
        manager2 = get_seed_manager()
        
        seeds2 = manager2.generate_experiment_seeds(10, base_seed=123)
        
        assert seeds1 == seeds2
    
    def test_generate_experiment_seeds_no_manager_seed(self):
        """Test experiment seed generation error when no manager seed available."""
        manager = SeedManager()
        
        with pytest.raises(RuntimeError, match="No seed available for experiment seed generation"):
            manager.generate_experiment_seeds(5)


class TestReproducibilityValidation:
    """Test suite for reproducibility validation functionality."""
    
    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()
    
    def teardown_method(self):
        """Clean up singleton state after each test."""
        SeedManager.reset()
    
    def test_validate_reproducibility_success(self):
        """Test successful reproducibility validation."""
        config = SeedConfig(seed=42, preserve_state=True)
        set_global_seed(config=config)
        manager = get_seed_manager()
        
        # Generate reference values
        reference_values = {
            'python_random': random.random(),
            'numpy_legacy': np.random.random(),
            'numpy_generator': get_numpy_generator().random()
        }
        
        # Modify state
        random.random()
        np.random.random()
        
        # Validate reproducibility
        result = manager.validate_reproducibility(reference_values)
        
        assert result is True
    
    def test_validate_reproducibility_failure(self):
        """Test reproducibility validation failure detection."""
        config = SeedConfig(seed=42, preserve_state=True)
        set_global_seed(config=config)
        manager = get_seed_manager()
        
        # Use wrong reference values
        wrong_reference = {
            'python_random': 0.12345,
            'numpy_legacy': 0.67890
        }
        
        result = manager.validate_reproducibility(wrong_reference)
        
        assert result is False
    
    def test_validate_reproducibility_tolerance(self):
        """Test reproducibility validation with custom tolerance."""
        config = SeedConfig(seed=42, preserve_state=True)
        set_global_seed(config=config)
        manager = get_seed_manager()
        
        # Generate reference values
        python_val = random.random()
        reference_values = {
            'python_random': python_val + 1e-12  # Tiny difference
        }
        
        # Should pass with default tolerance
        result = manager.validate_reproducibility(reference_values)
        assert result is True
        
        # Should fail with strict tolerance
        result_strict = manager.validate_reproducibility(reference_values, tolerance=1e-15)
        assert result_strict is False
    
    def test_validate_reproducibility_without_initial_state(self):
        """Test reproducibility validation without initial state preservation."""
        config = SeedConfig(seed=42, preserve_state=True)
        set_global_seed(config=config)
        manager = get_seed_manager()
        
        # Clear initial state to test fallback behavior
        manager._initial_state = None
        
        reference_values = {
            'python_random': random.random()
        }
        
        # Should still work by re-initializing with current seed
        result = manager.validate_reproducibility(reference_values)
        assert result is True


class TestDeterministicExecution:
    """Test suite for deterministic experiment execution validation."""
    
    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()
    
    def teardown_method(self):
        """Clean up singleton state after each test."""
        SeedManager.reset()
    
    def test_multiple_runs_identical_results(self):
        """Test multiple runs with identical seeds produce identical results."""
        seed_value = 42
        results = []
        
        for run in range(3):
            SeedManager.reset()
            set_global_seed(seed_value)
            
            # Generate deterministic sequence
            run_results = {
                'python_values': [random.random() for _ in range(5)],
                'numpy_values': [np.random.random() for _ in range(5)],
                'generator_values': [get_numpy_generator().random() for _ in range(5)]
            }
            results.append(run_results)
        
        # All runs should produce identical results
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0]['python_values'], results[i]['python_values'])
            np.testing.assert_array_equal(results[0]['numpy_values'], results[i]['numpy_values'])
            np.testing.assert_array_equal(results[0]['generator_values'], results[i]['generator_values'])
    
    def test_different_seeds_different_results(self):
        """Test different seeds produce different results."""
        results = {}
        
        for seed in [42, 123, 456]:
            SeedManager.reset()
            set_global_seed(seed)
            
            results[seed] = {
                'python_value': random.random(),
                'numpy_value': np.random.random(),
                'generator_value': get_numpy_generator().random()
            }
        
        # All results should be different
        seeds = list(results.keys())
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                seed1, seed2 = seeds[i], seeds[j]
                assert results[seed1]['python_value'] != results[seed2]['python_value']
                assert results[seed1]['numpy_value'] != results[seed2]['numpy_value']
                assert results[seed1]['generator_value'] != results[seed2]['generator_value']
    
    def test_seed_manager_reset_between_experiments(self):
        """Test proper seed manager reset between experiments."""
        # First experiment
        set_global_seed(42)
        first_value = random.random()
        first_seed = get_current_seed()
        
        # Reset and second experiment
        SeedManager.reset()
        set_global_seed(42)
        second_value = random.random()
        second_seed = get_current_seed()
        
        # Should get identical results after reset
        assert first_value == second_value
        assert first_seed == second_seed
    
    def test_complex_simulation_reproducibility(self):
        """Test reproducibility in complex simulation-like scenarios."""
        def mock_simulation_step(agent_positions):
            """Mock simulation step using random numbers."""
            noise = np.random.normal(0, 0.1, agent_positions.shape)
            movement = get_numpy_generator().uniform(-1, 1, agent_positions.shape)
            return agent_positions + noise + movement
        
        seed_value = 42
        num_agents = 10
        num_steps = 20
        
        # Run simulation twice with same seed
        results = []
        for run in range(2):
            SeedManager.reset()
            set_global_seed(seed_value)
            
            positions = np.zeros((num_agents, 2))
            trajectory = [positions.copy()]
            
            for step in range(num_steps):
                positions = mock_simulation_step(positions)
                trajectory.append(positions.copy())
            
            results.append(np.array(trajectory))
        
        # Results should be identical
        np.testing.assert_array_equal(results[0], results[1])


class TestLoggingIntegration:
    """Test suite for logging system integration."""
    
    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()
    
    def teardown_method(self):
        """Clean up singleton state after each test."""
        SeedManager.reset()
    
    @patch('src.{{cookiecutter.project_slug}}.utils.seed_manager.logger')
    def test_logging_context_setup(self, mock_logger):
        """Test logging context setup during initialization."""
        config = SeedConfig(seed=42, log_seed_context=True)
        set_global_seed(config=config)
        
        # Verify logger.configure was called for context binding
        assert mock_logger.configure.called
        
        # Verify info log was generated with seed context
        mock_logger.bind.assert_called_with(module='src.{{cookiecutter.project_slug}}.utils.seed_manager')
        assert mock_logger.bind.return_value.info.called
    
    @patch('src.{{cookiecutter.project_slug}}.utils.seed_manager.logger')
    def test_logging_context_disabled(self, mock_logger):
        """Test logging context setup can be disabled."""
        config = SeedConfig(seed=42, log_seed_context=False)
        set_global_seed(config=config)
        
        # logger.configure should not be called when log_seed_context=False
        # Still gets called for initialization logging, but not for context binding
        mock_logger.bind.assert_called_with(module='src.{{cookiecutter.project_slug}}.utils.seed_manager')
    
    @patch('src.{{cookiecutter.project_slug}}.utils.seed_manager.logger')
    def test_performance_warning_logging(self, mock_logger):
        """Test performance warning logging for slow initialization."""
        config = SeedConfig(seed=42)
        
        # Mock slow initialization
        with patch('time.perf_counter', side_effect=[0, 0.15]):  # 150ms duration
            set_global_seed(config=config)
        
        # Should log performance warning
        warning_calls = [call for call in mock_logger.bind.return_value.warning.call_args_list 
                        if 'initialization exceeded' in str(call)]
        assert len(warning_calls) > 0
    
    @patch('src.{{cookiecutter.project_slug}}.utils.seed_manager.logger')
    def test_error_logging_on_failure(self, mock_logger):
        """Test error logging during initialization failure."""
        manager = SeedManager()
        
        with patch.object(manager, '_initialize_generators', side_effect=Exception("Test error")):
            with pytest.raises(RuntimeError):
                manager.initialize(config=SeedConfig(seed=42))
        
        # Should log error with context
        assert mock_logger.bind.return_value.error.called
        error_call = mock_logger.bind.return_value.error.call_args
        assert "initialization failed" in str(error_call)


class TestIntegrationScenarios:
    """Test suite for integration scenarios with other system components."""
    
    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()
    
    def teardown_method(self):
        """Clean up singleton state after each test."""
        SeedManager.reset()
    
    def test_integration_with_numpy_computations(self):
        """Test seed manager integration with NumPy-based computations."""
        set_global_seed(42)
        
        # Simulate scientific computations
        data = np.random.normal(0, 1, (100, 3))
        processed = np.mean(data, axis=0)
        
        # Reset and repeat
        SeedManager.reset()
        set_global_seed(42)
        
        data2 = np.random.normal(0, 1, (100, 3))
        processed2 = np.mean(data2, axis=0)
        
        # Results should be identical
        np.testing.assert_array_equal(data, data2)
        np.testing.assert_array_equal(processed, processed2)
    
    def test_integration_with_multiple_generators(self):
        """Test seed manager with multiple random number generators."""
        set_global_seed(42)
        
        # Create multiple generators
        gen1 = get_numpy_generator()
        gen2 = np.random.default_rng(42)  # Independent generator
        
        # Generate values
        values1 = [gen1.random() for _ in range(5)]
        values2 = [gen2.random() for _ in range(5)]
        
        # Reset and create new generators
        SeedManager.reset()
        set_global_seed(42)
        
        gen3 = get_numpy_generator()
        gen4 = np.random.default_rng(42)
        
        values3 = [gen3.random() for _ in range(5)]
        values4 = [gen4.random() for _ in range(5)]
        
        # Managed generator should be reproducible
        np.testing.assert_array_equal(values1, values3)
        # Independent generator should also be reproducible
        np.testing.assert_array_equal(values2, values4)
    
    def test_concurrent_access_simulation(self):
        """Test seed manager behavior under simulated concurrent access."""
        import threading
        import time
        
        set_global_seed(42)
        results = {}
        
        def worker(worker_id):
            """Worker function simulating concurrent access."""
            # Each worker gets the same generator (singleton)
            generator = get_numpy_generator()
            time.sleep(0.001 * worker_id)  # Stagger access
            value = generator.random()
            results[worker_id] = value
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All workers should have gotten values
        assert len(results) == 5
        assert all(isinstance(v, float) for v in results.values())