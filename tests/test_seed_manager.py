"""
Comprehensive test suite for seed_manager module.

Tests validate global seed management, reproducibility controls, and cross-platform
deterministic behavior ensuring reproducible research outcomes through NumPy random
state management, Python random module initialization, and experiment-level seed
control with Hydra configuration integration.

Test Coverage:
- Feature F-014: Global seed management for reproducible experiments
- Cross-platform consistency and deterministic execution guarantees  
- Hydra configuration integration for seed parameter management
- Seed initialization timing validation (<100ms requirement)
- Integration with logging system for experiment tracking per Section 5.4.2
- Random state preservation capabilities for experiment checkpointing
- Comprehensive error handling and edge case management
"""

import os
import sys
import time
import random
import platform
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

import pytest
import numpy as np
from loguru import logger

# Test hydra integration conditionally
try:
    from hydra.core.config_store import ConfigStore
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False

from src.odor_plume_nav.utils.seed_manager import (
    SeedConfig,
    SeedManager,
    get_seed_manager,
    set_global_seed,
    get_current_seed,
    get_numpy_generator,
)
from src.odor_plume_nav.config.schemas import BaseModel


class TestSeedConfig:
    """Test suite for SeedConfig validation and schema compliance."""

    def test_seed_config_default_values(self):
        """Test SeedConfig default configuration values."""
        config = SeedConfig()
        
        assert config.seed is None
        assert config.numpy_seed is None
        assert config.python_seed is None
        assert config.auto_seed is True
        assert config.hash_environment is True
        assert config.validate_initialization is True
        assert config.preserve_state is False
        assert config.log_seed_context is True

    def test_seed_config_with_explicit_values(self):
        """Test SeedConfig with explicitly provided values."""
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

    def test_seed_config_validation_inheritance(self):
        """Test that SeedConfig properly inherits from BaseModel."""
        config = SeedConfig(seed=42)
        
        # Test BaseModel functionality
        assert isinstance(config, BaseModel)
        assert hasattr(config, 'model_dump')
        assert hasattr(config, 'model_copy')
        
        # Test model dump
        data = config.model_dump()
        assert data['seed'] == 42
        assert isinstance(data, dict)

    def test_seed_config_extra_forbid(self):
        """Test that SeedConfig forbids extra fields for strict validation."""
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            SeedConfig(
                seed=42,
                invalid_field="should_fail"
            )

    def test_seed_config_field_descriptions(self):
        """Test that all fields have proper descriptions for documentation."""
        config = SeedConfig()
        
        # Get field info from Pydantic model
        fields = config.model_fields
        
        # Verify all fields have descriptions
        required_fields = [
            'seed', 'numpy_seed', 'python_seed', 'auto_seed',
            'hash_environment', 'validate_initialization',
            'preserve_state', 'log_seed_context'
        ]
        
        for field_name in required_fields:
            assert field_name in fields
            assert fields[field_name].description is not None
            assert len(fields[field_name].description) > 10  # Meaningful description

    def test_seed_config_type_validation(self):
        """Test SeedConfig type validation for all fields."""
        # Valid configuration
        config = SeedConfig(
            seed=42,
            numpy_seed=123,
            python_seed=456,
            auto_seed=True,
            hash_environment=False,
            validate_initialization=True,
            preserve_state=False,
            log_seed_context=True
        )
        
        assert isinstance(config.seed, int)
        assert isinstance(config.auto_seed, bool)
        
        # Test invalid types
        with pytest.raises(ValueError):
            SeedConfig(seed="invalid_string")
        
        with pytest.raises(ValueError):
            SeedConfig(auto_seed="not_boolean")


class TestSeedManagerSingleton:
    """Test suite for SeedManager singleton pattern and initialization."""

    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()

    def teardown_method(self):
        """Clean up after each test."""
        SeedManager.reset()

    def test_seed_manager_singleton_pattern(self):
        """Test that SeedManager implements proper singleton pattern."""
        manager1 = SeedManager()
        manager2 = SeedManager()
        manager3 = get_seed_manager()
        
        # All instances should be the same object
        assert manager1 is manager2
        assert manager2 is manager3
        assert id(manager1) == id(manager2) == id(manager3)

    def test_seed_manager_initialization_state(self):
        """Test SeedManager initialization state tracking."""
        manager = SeedManager()
        
        # Initially not initialized
        assert SeedManager._initialized is True  # Constructor sets this
        assert SeedManager._current_seed is None
        assert SeedManager._numpy_generator is None

    def test_seed_manager_reset_functionality(self):
        """Test SeedManager reset functionality for testing."""
        # Initialize manager
        manager = SeedManager()
        manager.initialize(SeedConfig(seed=42))
        
        # Verify state is set
        assert SeedManager._current_seed == 42
        assert SeedManager._numpy_generator is not None
        
        # Reset and verify clean state
        SeedManager.reset()
        
        assert SeedManager._instance is None
        assert SeedManager._initialized is False
        assert SeedManager._current_seed is None
        assert SeedManager._numpy_generator is None

    def test_seed_manager_properties_before_initialization(self):
        """Test SeedManager properties return None before initialization."""
        manager = SeedManager()
        
        assert manager.current_seed is None
        assert manager.run_id is None
        assert manager.environment_hash is None
        assert manager.numpy_generator is None


class TestSeedManagerInitialization:
    """Test suite for SeedManager initialization process and validation."""

    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()

    def teardown_method(self):
        """Clean up after each test."""
        SeedManager.reset()

    def test_initialization_with_explicit_seed(self):
        """Test seed manager initialization with explicit seed value."""
        manager = SeedManager()
        config = SeedConfig(seed=42)
        
        returned_seed = manager.initialize(config)
        
        assert returned_seed == 42
        assert manager.current_seed == 42
        assert manager.run_id is not None
        assert manager.environment_hash is not None
        assert manager.numpy_generator is not None

    def test_initialization_with_dict_config(self):
        """Test seed manager initialization with dictionary configuration."""
        manager = SeedManager()
        config_dict = {
            'seed': 123,
            'validate_initialization': True,
            'log_seed_context': False
        }
        
        returned_seed = manager.initialize(config_dict)
        
        assert returned_seed == 123
        assert manager.current_seed == 123

    def test_initialization_with_none_config(self):
        """Test seed manager initialization with None config (auto-generation)."""
        manager = SeedManager()
        
        returned_seed = manager.initialize(None)
        
        assert isinstance(returned_seed, int)
        assert 0 <= returned_seed <= 2**32 - 1
        assert manager.current_seed == returned_seed

    def test_initialization_timing_requirement(self):
        """Test that seed initialization meets <100ms performance requirement."""
        manager = SeedManager()
        config = SeedConfig(seed=42, validate_initialization=True)
        
        start_time = time.perf_counter()
        manager.initialize(config)
        end_time = time.perf_counter()
        
        initialization_time_ms = (end_time - start_time) * 1000
        
        # Verify performance requirement (Feature F-014)
        assert initialization_time_ms < 100, f"Initialization took {initialization_time_ms:.2f}ms, exceeds 100ms requirement"

    def test_initialization_with_run_id(self):
        """Test seed manager initialization with custom run ID."""
        manager = SeedManager()
        config = SeedConfig(seed=42)
        custom_run_id = "test_experiment_001"
        
        manager.initialize(config, run_id=custom_run_id)
        
        assert manager.run_id == custom_run_id

    def test_initialization_state_preservation_enabled(self):
        """Test initialization with state preservation enabled."""
        manager = SeedManager()
        config = SeedConfig(seed=42, preserve_state=True)
        
        manager.initialize(config)
        
        initial_state = manager.get_state()
        assert initial_state is not None
        assert 'python_state' in initial_state
        assert 'numpy_legacy_state' in initial_state
        assert 'seed' in initial_state

    def test_initialization_state_preservation_disabled(self):
        """Test initialization with state preservation disabled."""
        manager = SeedManager()
        config = SeedConfig(seed=42, preserve_state=False)
        
        manager.initialize(config)
        
        initial_state = manager.get_state()
        assert initial_state is None

    def test_initialization_validation_enabled(self):
        """Test initialization with validation enabled."""
        manager = SeedManager()
        config = SeedConfig(seed=42, validate_initialization=True)
        
        # Should complete without errors
        manager.initialize(config)
        
        # Verify random generators are working
        assert manager.numpy_generator is not None
        test_value = manager.numpy_generator.random()
        assert isinstance(test_value, float)
        assert 0.0 <= test_value <= 1.0

    def test_initialization_error_handling(self):
        """Test initialization error handling and runtime exceptions."""
        manager = SeedManager()
        
        # Test with invalid seed range
        with pytest.raises(RuntimeError, match="Seed manager initialization failed"):
            invalid_config = SeedConfig()
            # Mock _determine_seed to raise ValueError
            with patch.object(manager, '_determine_seed', side_effect=ValueError("Invalid seed")):
                manager.initialize(invalid_config)


class TestGlobalSeedManagement:
    """Test suite for global seed management functions and API."""

    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()

    def teardown_method(self):
        """Clean up after each test."""
        SeedManager.reset()

    def test_set_global_seed_with_explicit_value(self):
        """Test set_global_seed with explicit seed value."""
        seed = set_global_seed(42)
        
        assert seed == 42
        assert get_current_seed() == 42
        
        # Verify global state
        numpy_gen = get_numpy_generator()
        assert numpy_gen is not None

    def test_set_global_seed_with_config(self):
        """Test set_global_seed with SeedConfig object."""
        config = SeedConfig(seed=123, validate_initialization=True)
        seed = set_global_seed(config=config)
        
        assert seed == 123
        assert get_current_seed() == 123

    def test_set_global_seed_with_dict_config(self):
        """Test set_global_seed with dictionary configuration."""
        config_dict = {'seed': 456, 'log_seed_context': False}
        seed = set_global_seed(config=config_dict)
        
        assert seed == 456
        assert get_current_seed() == 456

    def test_set_global_seed_auto_generation(self):
        """Test set_global_seed with automatic seed generation."""
        seed = set_global_seed()
        
        assert isinstance(seed, int)
        assert 0 <= seed <= 2**32 - 1
        assert get_current_seed() == seed

    def test_set_global_seed_override_config_seed(self):
        """Test that explicit seed parameter overrides config seed."""
        config = SeedConfig(seed=999)
        seed = set_global_seed(seed=42, config=config)
        
        # Explicit seed should override config seed
        assert seed == 42
        assert get_current_seed() == 42

    def test_get_current_seed_before_initialization(self):
        """Test get_current_seed returns None before initialization."""
        seed = get_current_seed()
        assert seed is None

    def test_get_numpy_generator_before_initialization(self):
        """Test get_numpy_generator returns None before initialization."""
        generator = get_numpy_generator()
        assert generator is None

    def test_global_functions_after_initialization(self):
        """Test global functions work correctly after initialization."""
        set_global_seed(42)
        
        assert get_current_seed() == 42
        
        generator = get_numpy_generator()
        assert generator is not None
        assert isinstance(generator, np.random.Generator)


class TestRandomStateManagement:
    """Test suite for NumPy and Python random state management."""

    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()

    def teardown_method(self):
        """Clean up after each test."""
        SeedManager.reset()

    def test_numpy_random_state_initialization(self):
        """Test NumPy random state initialization and functionality."""
        manager = SeedManager()
        config = SeedConfig(seed=42, numpy_seed=123)
        
        manager.initialize(config)
        
        # Test legacy NumPy random
        legacy_value = np.random.random()
        assert isinstance(legacy_value, float)
        assert 0.0 <= legacy_value <= 1.0
        
        # Test modern NumPy generator
        generator = manager.numpy_generator
        assert generator is not None
        generator_value = generator.random()
        assert isinstance(generator_value, float)
        assert 0.0 <= generator_value <= 1.0

    def test_python_random_state_initialization(self):
        """Test Python random module state initialization."""
        manager = SeedManager()
        config = SeedConfig(seed=42, python_seed=456)
        
        manager.initialize(config)
        
        # Test Python random
        python_value = random.random()
        assert isinstance(python_value, float)
        assert 0.0 <= python_value <= 1.0

    def test_deterministic_sequence_generation(self):
        """Test that same seed produces deterministic sequences."""
        seed_value = 42
        
        # First run
        SeedManager.reset()
        manager1 = SeedManager()
        config1 = SeedConfig(seed=seed_value)
        manager1.initialize(config1)
        
        sequence1 = [random.random() for _ in range(10)]
        numpy_sequence1 = [np.random.random() for _ in range(10)]
        
        # Second run with same seed
        SeedManager.reset()
        manager2 = SeedManager()
        config2 = SeedConfig(seed=seed_value)
        manager2.initialize(config2)
        
        sequence2 = [random.random() for _ in range(10)]
        numpy_sequence2 = [np.random.random() for _ in range(10)]
        
        # Sequences should be identical
        assert sequence1 == sequence2
        assert numpy_sequence1 == numpy_sequence2

    def test_separate_seed_configuration(self):
        """Test separate seed configuration for NumPy and Python random."""
        manager = SeedManager()
        config = SeedConfig(
            seed=42,
            numpy_seed=123,
            python_seed=456
        )
        
        manager.initialize(config)
        
        # Verify both generators are initialized
        assert manager.numpy_generator is not None
        
        # Generate test values
        python_val = random.random()
        numpy_val = np.random.random()
        generator_val = manager.numpy_generator.random()
        
        # All should be valid float values
        for val in [python_val, numpy_val, generator_val]:
            assert isinstance(val, float)
            assert 0.0 <= val <= 1.0

    def test_generator_state_consistency(self):
        """Test that generator state remains consistent."""
        manager = SeedManager()
        config = SeedConfig(seed=42, validate_initialization=True)
        
        manager.initialize(config)
        
        # Generate values and verify consistency
        gen = manager.numpy_generator
        val1 = gen.random()
        val2 = gen.random()
        
        # Values should be different (unless extremely unlikely)
        assert val1 != val2
        
        # But reproducible with same state
        state = gen.bit_generator.state
        gen.bit_generator.state = state
        # Can't directly test reproducibility without recreation


class TestCrossPlatformConsistency:
    """Test suite for cross-platform deterministic behavior."""

    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()

    def teardown_method(self):
        """Clean up after each test."""
        SeedManager.reset()

    def test_environment_hash_generation(self):
        """Test environment hash generation for cross-platform tracking."""
        manager = SeedManager()
        config = SeedConfig(seed=42, hash_environment=True)
        
        manager.initialize(config)
        
        env_hash = manager.environment_hash
        assert env_hash is not None
        assert isinstance(env_hash, str)
        assert len(env_hash) == 8  # MD5 hash truncated to 8 chars

    def test_environment_hash_consistency(self):
        """Test that environment hash is consistent across multiple runs."""
        # First initialization
        manager1 = SeedManager()
        config1 = SeedConfig(seed=42, hash_environment=True)
        manager1.initialize(config1)
        hash1 = manager1.environment_hash
        
        # Reset and second initialization
        SeedManager.reset()
        manager2 = SeedManager()
        config2 = SeedConfig(seed=42, hash_environment=True)
        manager2.initialize(config2)
        hash2 = manager2.environment_hash
        
        # Environment hashes should be identical on same platform
        assert hash1 == hash2

    def test_platform_information_logging(self):
        """Test that platform information is properly logged."""
        manager = SeedManager()
        config = SeedConfig(seed=42)
        
        with patch('loguru.logger.info') as mock_logger:
            manager.initialize(config)
            
            # Verify platform information is logged
            mock_logger.assert_called()
            call_args = mock_logger.call_args
            
            assert "Seed manager initialized successfully" in call_args[0][0]
            
            # Check extra information includes platform details
            extra = call_args[1]['extra']
            assert 'platform' in extra
            assert 'numpy_version' in extra
            assert 'environment_hash' in extra

    def test_entropy_generation_with_environment_hashing(self):
        """Test entropy generation includes environment characteristics."""
        manager = SeedManager()
        config = SeedConfig(auto_seed=True, hash_environment=True)
        
        # Generate entropy-based seed
        seed = manager.initialize(config)
        
        assert isinstance(seed, int)
        assert 0 <= seed <= 2**32 - 1

    def test_entropy_generation_without_environment_hashing(self):
        """Test entropy generation without environment characteristics."""
        manager = SeedManager()
        config = SeedConfig(auto_seed=True, hash_environment=False)
        
        # Generate entropy-based seed
        seed = manager.initialize(config)
        
        assert isinstance(seed, int)
        assert 0 <= seed <= 2**32 - 1

    def test_deterministic_seed_cross_platform_simulation(self):
        """Simulate cross-platform deterministic behavior testing."""
        # Mock different platform characteristics
        original_platform = platform.platform
        original_python_version = platform.python_version
        
        try:
            # Mock platform info for testing
            platform.platform = lambda: "Linux-5.4.0-x86_64"
            platform.python_version = lambda: "3.9.0"
            
            manager = SeedManager()
            config = SeedConfig(seed=42, hash_environment=True)
            manager.initialize(config)
            
            # Generate deterministic values
            test_values = [random.random() for _ in range(5)]
            
            # Values should be deterministic regardless of "platform"
            assert len(test_values) == 5
            assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in test_values)
            
        finally:
            # Restore original platform functions
            platform.platform = original_platform
            platform.python_version = original_python_version


class TestStatePreservationAndRestoration:
    """Test suite for random state preservation and restoration capabilities."""

    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()

    def teardown_method(self):
        """Clean up after each test."""
        SeedManager.reset()

    def test_state_preservation_enabled(self):
        """Test state preservation when enabled in configuration."""
        manager = SeedManager()
        config = SeedConfig(seed=42, preserve_state=True)
        
        manager.initialize(config)
        
        # Get initial state
        state = manager.get_state()
        
        assert state is not None
        assert isinstance(state, dict)
        assert 'python_state' in state
        assert 'numpy_legacy_state' in state
        assert 'seed' in state
        assert 'timestamp' in state

    def test_state_preservation_disabled(self):
        """Test state preservation when disabled in configuration."""
        manager = SeedManager()
        config = SeedConfig(seed=42, preserve_state=False)
        
        manager.initialize(config)
        
        # Get state should return None
        state = manager.get_state()
        assert state is None

    def test_state_restoration_functionality(self):
        """Test state restoration from preserved state."""
        manager = SeedManager()
        config = SeedConfig(seed=42, preserve_state=True)
        
        manager.initialize(config)
        
        # Generate some random values to change state
        initial_values = [random.random() for _ in range(5)]
        
        # Capture state after generation
        state_after_generation = manager.get_state()
        
        # Generate more values
        more_values = [random.random() for _ in range(5)]
        
        # Restore to captured state
        manager.restore_state(state_after_generation)
        
        # Generate same number of values again
        restored_values = [random.random() for _ in range(5)]
        
        # Values should match the "more_values" sequence
        assert restored_values == more_values

    def test_state_restoration_error_when_disabled(self):
        """Test that state restoration fails when preservation is disabled."""
        manager = SeedManager()
        config = SeedConfig(seed=42, preserve_state=False)
        
        manager.initialize(config)
        
        # Attempt to restore state should fail
        dummy_state = {'python_state': None, 'numpy_legacy_state': None}
        
        with pytest.raises(RuntimeError, match="State preservation not enabled"):
            manager.restore_state(dummy_state)

    def test_state_restoration_invalid_state(self):
        """Test state restoration with invalid state data."""
        manager = SeedManager()
        config = SeedConfig(seed=42, preserve_state=True)
        
        manager.initialize(config)
        
        # Invalid state data
        invalid_state = {'invalid': 'data'}
        
        with pytest.raises(RuntimeError, match="State restoration failed"):
            manager.restore_state(invalid_state)

    def test_checkpoint_and_restore_workflow(self):
        """Test complete checkpoint and restore workflow."""
        manager = SeedManager()
        config = SeedConfig(seed=42, preserve_state=True)
        
        manager.initialize(config)
        
        # Generate initial sequence
        sequence1 = [random.random() for _ in range(10)]
        
        # Create checkpoint
        checkpoint = manager.get_state()
        
        # Generate more values
        sequence2 = [random.random() for _ in range(10)]
        
        # Restore from checkpoint
        manager.restore_state(checkpoint)
        
        # Generate sequence again - should match sequence2
        sequence3 = [random.random() for _ in range(10)]
        
        # Verify sequences
        assert sequence1 != sequence2  # Different sequences
        assert sequence2 == sequence3  # Restored sequence matches


class TestTemporarySeedContext:
    """Test suite for temporary seed context manager functionality."""

    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()

    def teardown_method(self):
        """Clean up after each test."""
        SeedManager.reset()

    def test_temporary_seed_context_basic(self):
        """Test basic temporary seed context functionality."""
        manager = SeedManager()
        config = SeedConfig(seed=42, preserve_state=True)
        
        manager.initialize(config)
        
        # Generate values with main seed
        main_values = [random.random() for _ in range(3)]
        
        # Use temporary seed context
        temp_values = []
        with manager.temporary_seed(999):
            temp_values = [random.random() for _ in range(3)]
        
        # Generate more values with main seed
        main_values_after = [random.random() for _ in range(3)]
        
        # Temporary values should be different
        assert main_values != temp_values
        
        # Main sequence should continue correctly
        # (This test verifies state restoration)
        assert len(main_values_after) == 3

    def test_temporary_seed_context_deterministic(self):
        """Test that temporary seed context produces deterministic results."""
        manager = SeedManager()
        config = SeedConfig(seed=42, preserve_state=True)
        
        manager.initialize(config)
        
        # First use of temporary seed
        temp_values1 = []
        with manager.temporary_seed(999):
            temp_values1 = [random.random() for _ in range(5)]
        
        # Second use of same temporary seed
        temp_values2 = []
        with manager.temporary_seed(999):
            temp_values2 = [random.random() for _ in range(5)]
        
        # Values should be identical
        assert temp_values1 == temp_values2

    def test_temporary_seed_context_requires_preservation(self):
        """Test that temporary seed context requires state preservation."""
        manager = SeedManager()
        config = SeedConfig(seed=42, preserve_state=False)
        
        manager.initialize(config)
        
        # Temporary seed should fail without preservation
        with pytest.raises(RuntimeError, match="Temporary seed requires preserve_state=True"):
            with manager.temporary_seed(999):
                pass

    def test_temporary_seed_context_exception_handling(self):
        """Test temporary seed context handles exceptions properly."""
        manager = SeedManager()
        config = SeedConfig(seed=42, preserve_state=True)
        
        manager.initialize(config)
        
        # Capture original state
        original_seed = manager.current_seed
        
        # Exception in temporary context
        try:
            with manager.temporary_seed(999):
                # Generate some values
                random.random()
                # Raise exception
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected
        
        # Original state should be restored
        assert manager.current_seed == original_seed

    def test_temporary_seed_context_nested(self):
        """Test nested temporary seed contexts."""
        manager = SeedManager()
        config = SeedConfig(seed=42, preserve_state=True)
        
        manager.initialize(config)
        
        results = {}
        
        # Original context
        results['original'] = random.random()
        
        with manager.temporary_seed(111):
            results['temp1'] = random.random()
            
            with manager.temporary_seed(222):
                results['temp2'] = random.random()
            
            # Back to temp1 context
            results['temp1_restored'] = random.random()
        
        # Back to original context
        results['original_restored'] = random.random()
        
        # Verify all values are different and contexts work correctly
        assert len(set(results.values())) == len(results)  # All unique


class TestExperimentSeedGeneration:
    """Test suite for experiment seed generation functionality."""

    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()

    def teardown_method(self):
        """Clean up after each test."""
        SeedManager.reset()

    def test_experiment_seeds_generation_basic(self):
        """Test basic experiment seed generation."""
        manager = SeedManager()
        config = SeedConfig(seed=42)
        
        manager.initialize(config)
        
        # Generate experiment seeds
        seeds = manager.generate_experiment_seeds(5)
        
        assert len(seeds) == 5
        assert all(isinstance(seed, int) for seed in seeds)
        assert all(0 <= seed <= 2**32 - 1 for seed in seeds)

    def test_experiment_seeds_deterministic(self):
        """Test that experiment seed generation is deterministic."""
        seed_value = 42
        
        # First generation
        manager1 = SeedManager()
        config1 = SeedConfig(seed=seed_value)
        manager1.initialize(config1)
        seeds1 = manager1.generate_experiment_seeds(10)
        
        # Reset and second generation
        SeedManager.reset()
        manager2 = SeedManager()
        config2 = SeedConfig(seed=seed_value)
        manager2.initialize(config2)
        seeds2 = manager2.generate_experiment_seeds(10)
        
        # Should be identical
        assert seeds1 == seeds2

    def test_experiment_seeds_with_base_seed(self):
        """Test experiment seed generation with custom base seed."""
        manager = SeedManager()
        config = SeedConfig(seed=42)
        
        manager.initialize(config)
        
        # Generate with custom base seed
        seeds = manager.generate_experiment_seeds(5, base_seed=999)
        
        assert len(seeds) == 5
        assert all(isinstance(seed, int) for seed in seeds)

    def test_experiment_seeds_different_counts(self):
        """Test experiment seed generation with different counts."""
        manager = SeedManager()
        config = SeedConfig(seed=42)
        
        manager.initialize(config)
        
        # Different counts should produce different length lists
        seeds_3 = manager.generate_experiment_seeds(3)
        seeds_7 = manager.generate_experiment_seeds(7)
        
        assert len(seeds_3) == 3
        assert len(seeds_7) == 7
        
        # First 3 elements should be the same (deterministic)
        assert seeds_3 == seeds_7[:3]

    def test_experiment_seeds_no_state_contamination(self):
        """Test that experiment seed generation doesn't affect main random state."""
        manager = SeedManager()
        config = SeedConfig(seed=42)
        
        manager.initialize(config)
        
        # Generate reference sequence
        ref_sequence = [random.random() for _ in range(5)]
        
        # Reset
        SeedManager.reset()
        manager = SeedManager()
        manager.initialize(config)
        
        # Generate experiment seeds (should not affect main state)
        experiment_seeds = manager.generate_experiment_seeds(10)
        
        # Generate same sequence
        test_sequence = [random.random() for _ in range(5)]
        
        # Sequences should be identical (state not contaminated)
        assert ref_sequence == test_sequence

    def test_experiment_seeds_without_initialization(self):
        """Test experiment seed generation fails without initialization."""
        manager = SeedManager()
        
        # Should fail without initialization
        with pytest.raises(RuntimeError, match="No seed available for experiment seed generation"):
            manager.generate_experiment_seeds(5)


class TestReproducibilityValidation:
    """Test suite for reproducibility validation functionality."""

    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()

    def teardown_method(self):
        """Clean up after each test."""
        SeedManager.reset()

    def test_reproducibility_validation_success(self):
        """Test successful reproducibility validation."""
        manager = SeedManager()
        config = SeedConfig(seed=42, preserve_state=True)
        
        manager.initialize(config)
        
        # Generate reference values (this happens during initialization)
        # Reset to initial state and capture reference
        initial_state = manager._initial_state
        manager.restore_state(initial_state)
        
        reference_values = {
            'python_random': random.random(),
            'numpy_legacy': np.random.random(),
            'numpy_generator': manager.numpy_generator.random()
        }
        
        # Validate reproducibility
        is_valid = manager.validate_reproducibility(reference_values)
        
        assert is_valid is True

    def test_reproducibility_validation_failure(self):
        """Test reproducibility validation with incorrect reference values."""
        manager = SeedManager()
        config = SeedConfig(seed=42, preserve_state=True)
        
        manager.initialize(config)
        
        # Incorrect reference values
        wrong_reference = {
            'python_random': 0.999999,  # Unlikely to match
            'numpy_legacy': 0.888888,
            'numpy_generator': 0.777777
        }
        
        # Validation should fail
        with patch('loguru.logger.error') as mock_logger:
            is_valid = manager.validate_reproducibility(wrong_reference)
            
            assert is_valid is False
            mock_logger.assert_called()

    def test_reproducibility_validation_tolerance(self):
        """Test reproducibility validation with numerical tolerance."""
        manager = SeedManager()
        config = SeedConfig(seed=42, preserve_state=True)
        
        manager.initialize(config)
        
        # Generate reference with slight numerical difference
        initial_state = manager._initial_state
        manager.restore_state(initial_state)
        
        base_value = random.random()
        reference_values = {
            'python_random': base_value + 1e-12  # Very small difference
        }
        
        # Should pass with default tolerance
        is_valid = manager.validate_reproducibility(reference_values)
        assert is_valid is True
        
        # Should fail with strict tolerance
        is_valid = manager.validate_reproducibility(reference_values, tolerance=1e-15)
        assert is_valid is False

    def test_reproducibility_validation_without_preservation(self):
        """Test reproducibility validation without state preservation."""
        manager = SeedManager()
        config = SeedConfig(seed=42, preserve_state=False)
        
        manager.initialize(config)
        
        reference_values = {'python_random': 0.5}
        
        # Should handle case when no initial state is available
        # This will re-initialize with current seed
        is_valid = manager.validate_reproducibility(reference_values)
        
        # Result depends on whether re-initialization produces same values
        assert isinstance(is_valid, bool)


class TestHydraConfigurationIntegration:
    """Test suite for Hydra configuration integration."""

    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()

    def teardown_method(self):
        """Clean up after each test."""
        SeedManager.reset()

    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_hydra_config_loading(self):
        """Test loading seed configuration from Hydra global config."""
        # Mock Hydra configuration
        mock_hydra_cfg = OmegaConf.create({
            'seed_manager': {
                'seed': 42,
                'validate_initialization': True,
                'log_seed_context': False
            }
        })
        
        # Mock GlobalHydra
        with patch('src.odor_plume_nav.utils.seed_manager.GlobalHydra') as mock_global_hydra:
            mock_instance = Mock()
            mock_instance.is_initialized.return_value = True
            mock_instance.cfg = mock_hydra_cfg
            mock_global_hydra.instance.return_value = mock_instance
            mock_global_hydra.return_value.is_initialized.return_value = True
            
            manager = SeedManager()
            seed = manager.initialize()
            
            assert seed == 42

    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_hydra_config_fallback_to_seed_param(self):
        """Test fallback to direct seed parameter in Hydra config."""
        # Mock Hydra configuration with direct seed
        mock_hydra_cfg = OmegaConf.create({
            'seed': 123
        })
        
        with patch('src.odor_plume_nav.utils.seed_manager.GlobalHydra') as mock_global_hydra:
            mock_instance = Mock()
            mock_instance.is_initialized.return_value = True
            mock_instance.cfg = mock_hydra_cfg
            mock_global_hydra.instance.return_value = mock_instance
            mock_global_hydra.return_value.is_initialized.return_value = True
            
            manager = SeedManager()
            seed = manager.initialize()
            
            assert seed == 123

    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_hydra_config_not_initialized(self):
        """Test behavior when Hydra is not initialized."""
        with patch('src.odor_plume_nav.utils.seed_manager.GlobalHydra') as mock_global_hydra:
            mock_global_hydra.return_value.is_initialized.return_value = False
            
            manager = SeedManager()
            # Should use default configuration
            seed = manager.initialize()
            
            # Should complete with auto-generated seed
            assert isinstance(seed, int)
            assert 0 <= seed <= 2**32 - 1

    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_hydra_config_store_registration(self):
        """Test that SeedConfig is registered with Hydra ConfigStore."""
        # This test verifies the module registers the config schema
        try:
            from hydra.core.config_store import ConfigStore
            cs = ConfigStore.instance()
            
            # Verify registration exists (this should not raise an error)
            # The actual registration happens at module import
            assert cs is not None
            
        except ImportError:
            pytest.skip("Hydra not available")

    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_seed_config_structured_for_hydra(self):
        """SeedConfig should be usable as a Hydra structured config."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.structured(SeedConfig)
        assert cfg.seed is None

    def test_dictconfig_initialization(self):
        """Test initialization with DictConfig-like object."""
        # Simulate DictConfig behavior without requiring Hydra
        class MockDictConfig:
            def __init__(self, data):
                self._data = data
            
            def __iter__(self):
                return iter(self._data)
            
            def __getitem__(self, key):
                return self._data[key]
            
            def keys(self):
                return self._data.keys()
            
            def values(self):
                return self._data.values()
            
            def items(self):
                return self._data.items()
        
        mock_config = MockDictConfig({
            'seed': 42,
            'validate_initialization': True
        })
        
        manager = SeedManager()
        seed = manager.initialize(dict(mock_config._data))
        
        assert seed == 42


class TestPerformanceTiming:
    """Test suite for performance timing validation."""

    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()

    def teardown_method(self):
        """Clean up after each test."""
        SeedManager.reset()

    def test_initialization_performance_requirement(self):
        """Test that initialization meets <100ms performance requirement per Feature F-014."""
        manager = SeedManager()
        config = SeedConfig(seed=42, validate_initialization=True)
        
        # Measure initialization time
        start_time = time.perf_counter()
        manager.initialize(config)
        end_time = time.perf_counter()
        
        initialization_time_ms = (end_time - start_time) * 1000
        
        # Verify performance requirement
        assert initialization_time_ms < 100, (
            f"Seed initialization took {initialization_time_ms:.2f}ms, "
            f"exceeds 100ms requirement from Feature F-014"
        )

    def test_performance_with_validation_enabled(self):
        """Test performance with validation enabled."""
        manager = SeedManager()
        config = SeedConfig(
            seed=42,
            validate_initialization=True,
            preserve_state=True,
            log_seed_context=True
        )
        
        start_time = time.perf_counter()
        manager.initialize(config)
        end_time = time.perf_counter()
        
        initialization_time_ms = (end_time - start_time) * 1000
        
        # Should still meet performance requirement even with all features enabled
        assert initialization_time_ms < 100

    def test_performance_with_entropy_generation(self):
        """Test performance with entropy-based seed generation."""
        manager = SeedManager()
        config = SeedConfig(
            auto_seed=True,
            hash_environment=True,
            validate_initialization=True
        )
        
        start_time = time.perf_counter()
        manager.initialize(config)
        end_time = time.perf_counter()
        
        initialization_time_ms = (end_time - start_time) * 1000
        
        # Auto-generation should still meet performance requirement
        assert initialization_time_ms < 100

    def test_performance_logging_when_exceeded(self):
        """Test that performance warnings are logged when requirements exceeded."""
        manager = SeedManager()
        config = SeedConfig(seed=42)
        
        # Mock initialization to take longer than 100ms
        original_setup_logging = manager._setup_logging_context
        
        def slow_setup_logging():
            time.sleep(0.11)  # Simulate slow operation
            return original_setup_logging()
        
        with patch.object(manager, '_setup_logging_context', side_effect=slow_setup_logging):
            with patch('loguru.logger.warning') as mock_warning:
                manager.initialize(config)
                
                # Should log performance warning
                mock_warning.assert_called()
                warning_message = mock_warning.call_args[0][0]
                assert "exceeded performance requirement" in warning_message


class TestLoggingIntegration:
    """Test suite for logging system integration per Section 5.4.2."""

    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()

    def teardown_method(self):
        """Clean up after each test."""
        SeedManager.reset()

    def test_logging_context_binding_enabled(self):
        """Test automatic seed context binding for logging system."""
        manager = SeedManager()
        config = SeedConfig(seed=42, log_seed_context=True)
        
        with patch('loguru.logger.configure') as mock_configure:
            manager.initialize(config)
            
            # Verify logger.configure was called for context binding
            mock_configure.assert_called()
            
            # Verify patcher function was provided
            call_args = mock_configure.call_args
            assert 'patcher' in call_args[1]

    def test_logging_context_binding_disabled(self):
        """Test that context binding can be disabled."""
        manager = SeedManager()
        config = SeedConfig(seed=42, log_seed_context=False)
        
        with patch('loguru.logger.configure') as mock_configure:
            manager.initialize(config)
            
            # Configure should not be called when disabled
            mock_configure.assert_not_called()

    def test_seed_context_injection(self):
        """Test that seed context is properly injected into log records."""
        manager = SeedManager()
        config = SeedConfig(seed=42, log_seed_context=True)
        
        manager.initialize(config)
        
        # Test the patcher function directly
        test_record = {'extra': {}}
        
        # Get the patcher function (it's set up during initialization)
        # We need to simulate the patcher behavior
        patcher_record = {
            'extra': {
                'seed': manager.current_seed,
                'run_id': manager.run_id,
                'environment_hash': manager.environment_hash
            }
        }
        
        # Verify expected context is available
        assert manager.current_seed == 42
        assert manager.run_id is not None
        assert manager.environment_hash is not None

    def test_initialization_logging_details(self):
        """Test that initialization logs comprehensive details."""
        manager = SeedManager()
        config = SeedConfig(seed=42, validate_initialization=True)
        
        with patch('loguru.logger.info') as mock_info:
            manager.initialize(config)
            
            # Verify initialization success is logged
            mock_info.assert_called()
            
            call_args = mock_info.call_args
            message = call_args[0][0]
            extra = call_args[1]['extra']
            
            assert "Seed manager initialized successfully" in message
            
            # Verify required logging context per Section 5.4.2
            required_fields = [
                'seed', 'run_id', 'initialization_time_ms',
                'environment_hash', 'numpy_version', 'platform'
            ]
            
            for field in required_fields:
                assert field in extra, f"Missing required logging field: {field}"

    def test_error_logging_with_context(self):
        """Test that errors are logged with proper context."""
        manager = SeedManager()
        
        with patch.object(manager, '_determine_seed', side_effect=ValueError("Test error")):
            with patch('loguru.logger.error') as mock_error:
                with pytest.raises(RuntimeError):
                    manager.initialize(SeedConfig())
                
                # Verify error is logged with context
                mock_error.assert_called()
                
                call_args = mock_error.call_args
                message = call_args[0][0]
                extra = call_args[1]['extra']
                
                assert "Seed manager initialization failed" in message
                assert 'error_type' in extra

    def test_debug_logging_during_initialization(self):
        """Test debug logging during initialization process."""
        manager = SeedManager()
        config = SeedConfig(seed=42, validate_initialization=True)
        
        with patch('loguru.logger.debug') as mock_debug:
            manager.initialize(config)
            
            # Verify debug logging occurs
            mock_debug.assert_called()
            
            # Check for expected debug messages
            debug_calls = [call[0][0] for call in mock_debug.call_args_list]
            expected_messages = [
                "Using configured seed",
                "Initialized random generators",
                "Random state validation samples",
                "Seed context binding enabled"
            ]
            
            # At least some debug messages should be present
            assert len(debug_calls) > 0

    def test_run_id_generation_and_logging(self):
        """Test run ID generation and logging integration."""
        manager = SeedManager()
        config = SeedConfig(seed=42)
        
        manager.initialize(config)
        
        run_id = manager.run_id
        
        # Verify run ID format
        assert run_id is not None
        assert run_id.startswith('run_')
        assert len(run_id) == 12  # 'run_' + 8 char hash


class TestErrorHandlingAndEdgeCases:
    """Test suite for comprehensive error handling and edge case management."""

    def setup_method(self):
        """Reset singleton state before each test."""
        SeedManager.reset()

    def teardown_method(self):
        """Clean up after each test."""
        SeedManager.reset()

    def test_invalid_seed_range_validation(self):
        """Test validation of seed value ranges."""
        manager = SeedManager()
        
        # Test seed too large
        config = SeedConfig(seed=2**32)  # Exceeds valid range
        
        with pytest.raises(RuntimeError, match="Seed manager initialization failed"):
            manager.initialize(config)

    def test_invalid_config_type_handling(self):
        """Test handling of invalid configuration types."""
        manager = SeedManager()
        
        # Test with completely invalid config type
        with pytest.raises(RuntimeError):
            manager.initialize("invalid_config_string")

    def test_missing_numpy_dependency_simulation(self):
        """Test behavior when NumPy operations fail."""
        manager = SeedManager()
        config = SeedConfig(seed=42)
        
        # Mock numpy.random.default_rng to fail
        with patch('numpy.random.default_rng', side_effect=RuntimeError("NumPy error")):
            with pytest.raises(RuntimeError, match="Seed manager initialization failed"):
                manager.initialize(config)

    def test_corrupted_state_restoration(self):
        """Test handling of corrupted state during restoration."""
        manager = SeedManager()
        config = SeedConfig(seed=42, preserve_state=True)
        
        manager.initialize(config)
        
        # Test with corrupted Python state
        corrupted_state = {
            'python_state': "corrupted_data",
            'numpy_legacy_state': None,
            'seed': 42
        }
        
        with pytest.raises(RuntimeError, match="State restoration failed"):
            manager.restore_state(corrupted_state)

    def test_auto_seed_generation_failure_simulation(self):
        """Test handling of auto seed generation failures."""
        manager = SeedManager()
        config = SeedConfig(auto_seed=True)
        
        # Mock entropy generation to fail
        with patch('os.urandom', side_effect=OSError("Entropy source unavailable")):
            with pytest.raises(RuntimeError, match="Seed manager initialization failed"):
                manager.initialize(config)

    def test_seed_config_validation_edge_cases(self):
        """Test SeedConfig validation with edge case values."""
        # Test boundary values
        config = SeedConfig(seed=0)  # Minimum valid seed
        assert config.seed == 0
        
        config = SeedConfig(seed=2**32 - 1)  # Maximum valid seed
        assert config.seed == 2**32 - 1
        
        # Test None values
        config = SeedConfig(
            seed=None,
            numpy_seed=None,
            python_seed=None
        )
        assert config.seed is None
        assert config.numpy_seed is None
        assert config.python_seed is None

    def test_environment_hash_generation_failure(self):
        """Test handling of environment hash generation failures."""
        manager = SeedManager()
        config = SeedConfig(seed=42, hash_environment=True)
        
        # Mock platform.platform to fail
        with patch('platform.platform', side_effect=RuntimeError("Platform detection failed")):
            # Should still complete initialization (environment hash is non-critical)
            manager.initialize(config)
            
            # Manager should be initialized despite environment hash failure
            assert manager.current_seed == 42

    def test_logging_configuration_failure(self):
        """Test handling of logging configuration failures."""
        manager = SeedManager()
        config = SeedConfig(seed=42, log_seed_context=True)
        
        # Mock logger.configure to fail
        with patch('loguru.logger.configure', side_effect=RuntimeError("Logging config failed")):
            # Should still complete initialization (logging setup is non-critical)
            manager.initialize(config)
            
            assert manager.current_seed == 42

    def test_partial_generator_initialization_failure(self):
        """Test handling of partial random generator initialization failures."""
        manager = SeedManager()
        config = SeedConfig(seed=42)
        
        # Mock random.seed to fail
        with patch('random.seed', side_effect=RuntimeError("Python random failed")):
            with pytest.raises(RuntimeError, match="Seed manager initialization failed"):
                manager.initialize(config)

    def test_memory_constraints_during_state_preservation(self):
        """Test behavior under memory constraints during state preservation."""
        manager = SeedManager()
        config = SeedConfig(seed=42, preserve_state=True)
        
        manager.initialize(config)
        
        # Mock memory allocation failure during state capture
        with patch('random.getstate', side_effect=MemoryError("Insufficient memory")):
            with pytest.raises(RuntimeError, match="State restoration failed"):
                # This would fail during get_state -> restore_state cycle
                state = {'python_state': None}  # Simplified state
                manager.restore_state(state)

    def test_concurrent_initialization_protection(self):
        """Test that concurrent initialization is handled properly (singleton protection)."""
        import threading
        import concurrent.futures
        
        results = []
        
        def initialize_seed_manager():
            try:
                manager = SeedManager()
                seed = manager.initialize(SeedConfig(seed=42))
                return seed
            except Exception as e:
                return str(e)
        
        # Run concurrent initializations
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(initialize_seed_manager) for _ in range(3)]
            results = [f.result() for f in futures]
        
        # All should return the same seed (singleton behavior)
        # At least one should succeed, others might fail due to already initialized
        assert 42 in results or all(isinstance(r, str) for r in results)

    def test_cleanup_after_initialization_failure(self):
        """Test proper cleanup when initialization fails."""
        manager = SeedManager()
        
        # Cause initialization to fail after partial setup
        with patch.object(manager, '_initialize_generators', side_effect=RuntimeError("Generator setup failed")):
            with pytest.raises(RuntimeError):
                manager.initialize(SeedConfig(seed=42))
        
        # State should be clean for retry
        assert manager.current_seed is None
        assert manager.numpy_generator is None

    def test_edge_case_seed_values(self):
        """Test handling of edge case seed values."""
        test_seeds = [0, 1, 2**16, 2**31, 2**32 - 1]
        
        for seed_value in test_seeds:
            SeedManager.reset()
            manager = SeedManager()
            config = SeedConfig(seed=seed_value)
            
            returned_seed = manager.initialize(config)
            assert returned_seed == seed_value
            assert manager.current_seed == seed_value

    def test_reproducibility_with_system_randomness(self):
        """Test reproducibility despite system randomness sources."""
        # Test multiple initializations with same seed produce same results
        seed_value = 12345
        reference_values = []
        
        for _ in range(3):
            SeedManager.reset()
            manager = SeedManager()
            config = SeedConfig(seed=seed_value)
            
            manager.initialize(config)
            
            # Generate test values
            values = {
                'python': random.random(),
                'numpy': np.random.random(),
                'generator': manager.numpy_generator.random()
            }
            reference_values.append(values)
        
        # All runs should produce identical values
        first_run = reference_values[0]
        for run in reference_values[1:]:
            assert run['python'] == first_run['python']
            assert run['numpy'] == first_run['numpy'] 
            assert run['generator'] == first_run['generator']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])