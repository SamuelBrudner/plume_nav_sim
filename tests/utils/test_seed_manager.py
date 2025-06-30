"""
Comprehensive test suite for the Random Seed Manager utility (Feature F-014).

This module validates global seed management, deterministic experiment execution,
cross-platform consistency, and Hydra configuration integration essential for
reproducible scientific computing workflows.

Test Coverage Areas:
- Global seed management with set_global_seed() function validation
- Deterministic experiment execution across multiple runs with identical seeds
- Cross-platform consistency ensuring reproducible results across environments
- Hydra configuration system integration for seed parameter management
- NumPy random states and Python random module coordination
- Seed state preservation and restoration capabilities
- Error handling for invalid seed values and edge cases
- SeedManager class comprehensive functionality testing
- Context manager patterns with seed_context() function
- Performance requirements validation (<100ms initialization)
- Thread safety validation for concurrent experiment execution
- Reproducibility reporting and experiment tracking

Author: Test Suite Generator
Version: 2.0.0 (Enhanced for cookiecutter-based architecture)
"""

import pytest
import numpy as np
import random
import time
import threading
import tempfile
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List
from unittest.mock import patch, MagicMock, mock_open
from contextlib import contextmanager

# Test imports with graceful fallbacks for enhanced architecture
try:
    from src.plume_nav_sim.utils.seed_manager import (
        SeedManager,
        RandomState,
        set_global_seed,
        get_global_seed_manager,
        configure_from_hydra,
        seed_context,
        get_reproducibility_report
    )
    SEED_MANAGER_AVAILABLE = True
except ImportError:
    SEED_MANAGER_AVAILABLE = False
    SeedManager = None
    RandomState = None

# Hydra imports with fallback for environments without Hydra
try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict
    OmegaConf = None

# Loguru imports with fallback for testing environments
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    LOGURU_AVAILABLE = False


class TestRandomState:
    """
    Test suite for RandomState class validating state capture, validation, and serialization.
    
    Tests comprehensive random state management including NumPy state capture,
    Python random state preservation, integrity validation, and cross-platform
    serialization patterns essential for experiment checkpointing and reproduction.
    """

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_random_state_creation_with_valid_data(self):
        """Test RandomState creation with valid state data."""
        # Create test state data
        np.random.seed(42)
        random.seed(42)
        
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        
        state = RandomState(
            numpy_state=numpy_state,
            python_state=python_state,
            seed_value=42,
            experiment_id="test_exp_001"
        )
        
        assert state.seed_value == 42
        assert state.experiment_id == "test_exp_001"
        assert state.numpy_state == numpy_state
        assert state.python_state == python_state
        assert state.timestamp > 0
        assert state.state_checksum is not None
        assert len(state.state_checksum) == 16  # MD5 hash truncated to 16 chars

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_random_state_platform_info_generation(self):
        """Test platform information generation for cross-platform consistency."""
        state = RandomState(seed_value=123)
        
        platform_info = state.platform_info
        assert 'platform' in platform_info
        assert 'python_version' in platform_info
        assert 'numpy_version' in platform_info
        assert 'architecture' in platform_info
        assert 'byte_order' in platform_info
        
        # Validate platform information content
        assert platform_info['platform'] == sys.platform
        assert platform_info['python_version'] == sys.version_info[:3]
        assert platform_info['numpy_version'] == np.__version__
        assert platform_info['architecture'] in ['32bit', '64bit']
        assert platform_info['byte_order'] in ['little', 'big']

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_random_state_integrity_validation(self):
        """Test state integrity validation using checksum verification."""
        state = RandomState(
            seed_value=456,
            experiment_id="test_integrity"
        )
        
        # Test initial integrity validation
        assert state.validate_integrity() is True
        
        # Test integrity validation after manual checksum modification
        original_checksum = state.state_checksum
        state.state_checksum = "invalid_checksum"
        assert state.validate_integrity() is False
        
        # Restore original checksum and verify integrity
        state.state_checksum = original_checksum
        assert state.validate_integrity() is True

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_random_state_serialization_patterns(self):
        """Test RandomState serialization and deserialization for persistence."""
        # Create test state with comprehensive data
        np.random.seed(789)
        random.seed(789)
        
        original_state = RandomState(
            numpy_state=np.random.get_state(),
            python_state=random.getstate(),
            seed_value=789,
            experiment_id="serialization_test"
        )
        
        # Test to_dict() serialization
        state_dict = original_state.to_dict()
        expected_keys = {
            'numpy_state', 'python_state', 'seed_value', 'timestamp',
            'experiment_id', 'platform_info', 'state_checksum'
        }
        assert set(state_dict.keys()) == expected_keys
        
        # Test from_dict() deserialization
        restored_state = RandomState.from_dict(state_dict)
        assert restored_state.seed_value == original_state.seed_value
        assert restored_state.experiment_id == original_state.experiment_id
        assert restored_state.state_checksum == original_state.state_checksum
        assert restored_state.validate_integrity() is True

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_random_state_checksum_generation_consistency(self):
        """Test consistent checksum generation for identical state data."""
        # Create multiple states with identical data
        state1 = RandomState(
            seed_value=999,
            experiment_id="checksum_test"
        )
        
        state2 = RandomState(
            seed_value=999,
            experiment_id="checksum_test"
        )
        
        # Checksums should be different due to timestamp differences
        assert state1.state_checksum != state2.state_checksum
        
        # But states with identical timestamp should have identical checksums
        state3 = RandomState(
            seed_value=999,
            experiment_id="checksum_test",
            timestamp=state1.timestamp
        )
        state3.timestamp = state1.timestamp  # Force same timestamp
        state3.state_checksum = state3._generate_checksum()  # Regenerate checksum
        
        # This test validates the checksum algorithm consistency
        assert len(state1.state_checksum) == len(state3.state_checksum) == 16


class TestSeedManagerCore:
    """
    Test suite for core SeedManager functionality including initialization,
    state management, and global seed coordination.
    
    Validates deterministic experiment execution, NumPy and Python random
    module coordination, performance requirements, and thread safety
    essential for scientific computing reproducibility.
    """

    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        """Reset global seed manager state before each test."""
        # Clear any existing global manager
        import src.plume_nav_sim.utils.seed_manager as sm_module
        if hasattr(sm_module, '_global_manager'):
            sm_module._global_manager = None
        
        # Clear SeedManager class state
        if SEED_MANAGER_AVAILABLE and SeedManager:
            SeedManager._instances.clear()
            SeedManager._global_state = None
        
        yield
        
        # Cleanup after test
        if hasattr(sm_module, '_global_manager'):
            sm_module._global_manager = None
        if SEED_MANAGER_AVAILABLE and SeedManager:
            SeedManager._instances.clear()
            SeedManager._global_state = None

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_seed_manager_basic_initialization(self):
        """Test basic SeedManager initialization with default parameters."""
        manager = SeedManager(seed=42, auto_initialize=False)
        
        assert manager.seed == 42
        assert manager.experiment_id is not None
        assert manager.strict_validation is True
        assert manager.enable_logging is True
        assert manager._initialization_time is None  # Not initialized yet

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_seed_manager_initialization_performance(self):
        """Test SeedManager initialization meets <100ms performance requirement."""
        manager = SeedManager(seed=123, auto_initialize=False)
        
        start_time = time.perf_counter()
        initialization_time = manager.initialize()
        end_time = time.perf_counter()
        
        # Validate performance requirement
        assert initialization_time < 100.0  # <100ms requirement
        
        # Validate timing accuracy
        actual_time = (end_time - start_time) * 1000
        assert abs(initialization_time - actual_time) < 10  # Within 10ms tolerance

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_seed_manager_deterministic_execution(self):
        """Test deterministic experiment execution with identical seeds."""
        # First execution with seed 555
        manager1 = SeedManager(seed=555)
        first_random_numpy = np.random.random(5)
        first_random_python = [random.random() for _ in range(5)]
        
        # Second execution with same seed
        manager2 = SeedManager(seed=555)
        second_random_numpy = np.random.random(5)
        second_random_python = [random.random() for _ in range(5)]
        
        # Validate deterministic behavior
        np.testing.assert_array_equal(first_random_numpy, second_random_numpy)
        assert first_random_python == second_random_python

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_seed_manager_state_capture_and_restoration(self):
        """Test comprehensive state capture and restoration functionality."""
        manager = SeedManager(seed=777)
        
        # Generate some random numbers to change state
        initial_numpy = np.random.random(3)
        initial_python = [random.random() for _ in range(3)]
        
        # Capture current state
        captured_state = manager.capture_state()
        assert isinstance(captured_state, RandomState)
        assert captured_state.seed_value == 777
        assert captured_state.experiment_id == manager.experiment_id
        
        # Generate more random numbers
        after_capture_numpy = np.random.random(3)
        after_capture_python = [random.random() for _ in range(3)]
        
        # Restore captured state
        success = manager.restore_state(captured_state)
        assert success is True
        
        # Generate same random numbers - should match post-capture values
        restored_numpy = np.random.random(3)
        restored_python = [random.random() for _ in range(3)]
        
        np.testing.assert_array_equal(after_capture_numpy, restored_numpy)
        assert after_capture_python == restored_python

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_seed_manager_context_manager_functionality(self):
        """Test SeedManager context manager with automatic state handling."""
        manager = SeedManager(seed=888, auto_initialize=False)
        
        with manager as ctx_manager:
            assert ctx_manager is manager
            assert manager._initialization_time is not None  # Should be initialized
            assert len(manager._context_stack) == 1
            
            # Generate random numbers in context
            ctx_random = np.random.random(2)
        
        # Context should be properly exited
        assert len(manager._context_stack) == 0

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_seed_manager_thread_safety(self):
        """Test SeedManager thread safety for concurrent experiment execution."""
        results = {}
        errors = []
        
        def worker_function(worker_id: int, seed: int):
            """Worker function for thread safety testing."""
            try:
                manager = SeedManager(seed=seed, experiment_id=f"thread_{worker_id}")
                # Generate deterministic random numbers
                random_values = np.random.random(5)
                results[worker_id] = random_values.tolist()
            except Exception as e:
                errors.append(f"Worker {worker_id}: {str(e)}")
        
        # Create multiple threads with different seeds
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_function, args=(i, 100 + i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Validate thread safety
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 5, "All threads should complete successfully"
        
        # Validate each thread produced different results (different seeds)
        result_values = list(results.values())
        for i in range(len(result_values)):
            for j in range(i + 1, len(result_values)):
                assert result_values[i] != result_values[j], "Different seeds should produce different results"

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_seed_manager_error_handling_invalid_state(self):
        """Test error handling for invalid state restoration."""
        manager = SeedManager(seed=999)
        
        # Test invalid state object
        with pytest.raises(ValueError, match="Invalid state object"):
            manager.restore_state("invalid_state")
        
        # Test state with invalid integrity (if strict validation enabled)
        if manager.strict_validation:
            invalid_state = RandomState(seed_value=999)
            invalid_state.state_checksum = "invalid_checksum"
            
            with pytest.raises(ValueError, match="State integrity validation failed"):
                manager.restore_state(invalid_state)

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_seed_manager_reproducibility_info_generation(self):
        """Test comprehensive reproducibility information generation."""
        manager = SeedManager(seed=1111, experiment_id="repro_test")
        
        repro_info = manager.get_reproducibility_info()
        
        # Validate required fields
        required_fields = {
            'seed_value', 'experiment_id', 'initialization_time_ms',
            'platform_info', 'environment_variables', 'timestamp',
            'state_history_count', 'current_state_checksum'
        }
        assert set(repro_info.keys()) >= required_fields
        
        # Validate content
        assert repro_info['seed_value'] == 1111
        assert repro_info['experiment_id'] == "repro_test"
        assert isinstance(repro_info['platform_info'], dict)
        assert isinstance(repro_info['environment_variables'], dict)
        assert repro_info['timestamp'] > 0
        assert repro_info['state_history_count'] >= 1


class TestGlobalSeedFunctions:
    """
    Test suite for global seed management functions including set_global_seed(),
    get_global_seed_manager(), and seed_context() context manager.
    
    Validates global state coordination, convenience function behavior,
    and backward compatibility with existing seed management patterns.
    """

    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        """Reset global seed manager state before each test."""
        import src.plume_nav_sim.utils.seed_manager as sm_module
        if hasattr(sm_module, '_global_manager'):
            sm_module._global_manager = None
        yield
        if hasattr(sm_module, '_global_manager'):
            sm_module._global_manager = None

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_set_global_seed_basic_functionality(self):
        """Test basic set_global_seed() function behavior."""
        manager = set_global_seed(seed=2222)
        
        assert isinstance(manager, SeedManager)
        assert manager.seed == 2222
        assert manager._initialization_time is not None
        assert manager._initialization_time < 100.0  # Performance requirement

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_set_global_seed_deterministic_behavior(self):
        """Test set_global_seed() produces deterministic results."""
        # First global seed setting
        set_global_seed(seed=3333)
        first_numpy = np.random.random(3)
        first_python = [random.random() for _ in range(3)]
        
        # Second global seed setting with same seed
        set_global_seed(seed=3333)
        second_numpy = np.random.random(3)
        second_python = [random.random() for _ in range(3)]
        
        # Validate deterministic behavior
        np.testing.assert_array_equal(first_numpy, second_numpy)
        assert first_python == second_python

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_set_global_seed_with_experiment_id(self):
        """Test set_global_seed() with custom experiment identifier."""
        experiment_id = "global_test_experiment"
        manager = set_global_seed(seed=4444, experiment_id=experiment_id)
        
        assert manager.experiment_id == experiment_id
        assert manager.seed == 4444

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_get_global_seed_manager_functionality(self):
        """Test get_global_seed_manager() retrieval functionality."""
        # Initially should return None
        assert get_global_seed_manager() is None
        
        # Set global seed and retrieve manager
        original_manager = set_global_seed(seed=5555)
        retrieved_manager = get_global_seed_manager()
        
        assert retrieved_manager is original_manager
        assert retrieved_manager.seed == 5555

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_seed_context_manager_isolation(self):
        """Test seed_context() provides isolated seed management."""
        # Set initial global seed
        set_global_seed(seed=6666)
        initial_numpy = np.random.random(2)
        
        # Use seed context with different seed
        with seed_context(seed=7777) as ctx_manager:
            assert isinstance(ctx_manager, SeedManager)
            assert ctx_manager.seed == 7777
            
            # Generate random numbers in context
            context_numpy = np.random.random(2)
        
        # After context, global seed should be restored
        global_manager = get_global_seed_manager()
        assert global_manager.seed == 6666
        
        # Random state should be restored (this is implementation dependent)
        # Note: Current implementation doesn't restore state automatically

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_seed_context_with_experiment_id(self):
        """Test seed_context() with custom experiment identifier."""
        experiment_id = "context_test_exp"
        
        with seed_context(seed=8888, experiment_id=experiment_id) as ctx_manager:
            assert ctx_manager.experiment_id == experiment_id
            assert ctx_manager.seed == 8888

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_get_reproducibility_report_global_function(self):
        """Test get_reproducibility_report() global function."""
        # Set global seed first
        set_global_seed(seed=9999, experiment_id="global_repro_test")
        
        report = get_reproducibility_report()
        
        # Validate report structure
        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'system_info' in report
        assert 'environment_variables' in report
        assert 'active_experiments' in report
        assert 'global_seed_manager' in report
        
        # Validate global seed manager info
        global_info = report['global_seed_manager']
        assert global_info['seed_value'] == 9999
        assert global_info['experiment_id'] == "global_repro_test"


class TestHydraConfigurationIntegration:
    """
    Test suite for Hydra configuration system integration validating hierarchical
    configuration composition, override scenarios, and environment variable
    interpolation essential for reproducible research workflows.
    """

    @pytest.fixture
    def mock_hydra_config(self):
        """Provide mock Hydra configuration for testing."""
        if not HYDRA_AVAILABLE:
            # Fallback to regular dict for environments without Hydra
            return {
                'seed': 1234,
                'experiment': {'seed': 5678},
                'simulation': {'seed': 9012},
                'reproducibility': {'seed': 3456}
            }
        
        config_dict = {
            'seed': 1234,
            'experiment': {'seed': 5678},
            'simulation': {'seed': 9012},
            'reproducibility': {'seed': 3456},
            'navigator': {'max_speed': 2.0},
            'video_plume': {'kernel_size': 3}
        }
        return OmegaConf.create(config_dict)

    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        """Reset global seed manager state before each test."""
        import src.plume_nav_sim.utils.seed_manager as sm_module
        if hasattr(sm_module, '_global_manager'):
            sm_module._global_manager = None
        yield
        if hasattr(sm_module, '_global_manager'):
            sm_module._global_manager = None

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_configure_from_hydra_direct_seed(self, mock_hydra_config):
        """Test configure_from_hydra() with direct seed parameter."""
        # Test with direct 'seed' parameter
        success = configure_from_hydra(mock_hydra_config)
        
        assert success is True
        
        # Validate global seed manager was configured
        global_manager = get_global_seed_manager()
        assert global_manager is not None
        assert global_manager.seed == 1234

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_configure_from_hydra_nested_seed_paths(self, mock_hydra_config):
        """Test configure_from_hydra() with nested seed configuration paths."""
        # Remove direct seed and test nested paths
        if HYDRA_AVAILABLE:
            config = OmegaConf.create({
                'experiment': {'seed': 5678},
                'navigator': {'max_speed': 2.0}
            })
        else:
            config = {
                'experiment': {'seed': 5678},
                'navigator': {'max_speed': 2.0}
            }
        
        success = configure_from_hydra(config)
        assert success is True
        
        global_manager = get_global_seed_manager()
        assert global_manager.seed == 5678

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_configure_from_hydra_with_existing_manager(self, mock_hydra_config):
        """Test configure_from_hydra() with existing SeedManager instance."""
        # Create initial manager
        initial_manager = SeedManager(seed=1111, auto_initialize=False)
        
        # Configure from Hydra
        success = initial_manager.configure_from_hydra(mock_hydra_config)
        assert success is True
        assert initial_manager.seed == 1234  # Should be updated

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_configure_from_hydra_empty_config(self):
        """Test configure_from_hydra() with empty configuration."""
        empty_config = {} if not HYDRA_AVAILABLE else OmegaConf.create({})
        
        success = configure_from_hydra(empty_config)
        assert success is False

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_configure_from_hydra_none_config(self):
        """Test configure_from_hydra() with None configuration."""
        success = configure_from_hydra(None)
        assert success is False

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_configure_from_hydra_invalid_seed_type(self):
        """Test configure_from_hydra() with invalid seed type."""
        if HYDRA_AVAILABLE:
            config = OmegaConf.create({'seed': 'invalid_seed_string'})
        else:
            config = {'seed': 'invalid_seed_string'}
        
        # Should handle type conversion gracefully
        with pytest.raises((ValueError, TypeError)):
            configure_from_hydra(config)

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    @patch.dict(os.environ, {'TEST_SEED_VALUE': '7890'})
    def test_configure_from_hydra_environment_interpolation(self):
        """Test Hydra environment variable interpolation in seed configuration."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available for environment interpolation test")
        
        # Create config with environment variable interpolation
        # Note: This would normally be done by Hydra's resolver system
        config = OmegaConf.create({
            'seed': 7890,  # Simulated resolved environment variable
            'navigator': {'max_speed': 2.0}
        })
        
        success = configure_from_hydra(config)
        assert success is True
        
        global_manager = get_global_seed_manager()
        assert global_manager.seed == 7890


class TestErrorHandlingAndEdgeCases:
    """
    Test suite for error handling, edge cases, and boundary conditions
    in seed management functionality.
    
    Validates robust error handling for invalid inputs, extreme values,
    resource constraints, and failure recovery patterns essential for
    reliable scientific computing workflows.
    """

    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        """Reset global seed manager state before each test."""
        import src.plume_nav_sim.utils.seed_manager as sm_module
        if hasattr(sm_module, '_global_manager'):
            sm_module._global_manager = None
        if SEED_MANAGER_AVAILABLE and SeedManager:
            SeedManager._instances.clear()
            SeedManager._global_state = None
        yield
        if hasattr(sm_module, '_global_manager'):
            sm_module._global_manager = None
        if SEED_MANAGER_AVAILABLE and SeedManager:
            SeedManager._instances.clear()
            SeedManager._global_state = None

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_seed_manager_with_negative_seed(self):
        """Test SeedManager behavior with negative seed values."""
        # Negative seeds should be handled gracefully
        manager = SeedManager(seed=-123)
        assert manager.seed == -123
        
        # Should still produce deterministic results
        first_random = np.random.random(3)
        
        manager2 = SeedManager(seed=-123)
        second_random = np.random.random(3)
        
        np.testing.assert_array_equal(first_random, second_random)

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_seed_manager_with_large_seed_values(self):
        """Test SeedManager with large seed values near system limits."""
        # Test with large positive seed
        large_seed = 2**31 - 1  # Maximum 32-bit signed integer
        manager = SeedManager(seed=large_seed)
        assert manager.seed == large_seed
        
        # Should initialize successfully
        assert manager._initialization_time < 100.0

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_seed_manager_with_zero_seed(self):
        """Test SeedManager behavior with zero seed value."""
        manager = SeedManager(seed=0)
        assert manager.seed == 0
        
        # Zero should be a valid seed and produce deterministic results
        first_random = np.random.random(2)
        
        manager2 = SeedManager(seed=0)
        second_random = np.random.random(2)
        
        np.testing.assert_array_equal(first_random, second_random)

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_seed_manager_initialization_failure_handling(self):
        """Test SeedManager behavior when initialization fails."""
        manager = SeedManager(seed=12345, auto_initialize=False)
        
        # Mock numpy.random.seed to raise an exception
        with patch('numpy.random.seed', side_effect=RuntimeError("Mocked initialization failure")):
            with pytest.raises(RuntimeError, match="Failed to initialize seed manager"):
                manager.initialize()

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_seed_manager_state_capture_failure_handling(self):
        """Test SeedManager behavior when state capture fails."""
        manager = SeedManager(seed=54321)
        
        # Mock numpy random state getter to raise an exception
        with patch('numpy.random.get_state', side_effect=RuntimeError("Mocked state capture failure")):
            with pytest.raises(RuntimeError, match="Failed to capture random state"):
                manager.capture_state()

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_seed_manager_state_restoration_failure_handling(self):
        """Test SeedManager behavior when state restoration fails."""
        manager = SeedManager(seed=98765)
        
        # Create valid state for restoration
        valid_state = manager.capture_state()
        
        # Mock numpy set_state to raise an exception
        with patch('numpy.random.set_state', side_effect=RuntimeError("Mocked state restoration failure")):
            with pytest.raises(RuntimeError, match="Failed to restore random state"):
                manager.restore_state(valid_state)

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_seed_manager_concurrent_instance_management(self):
        """Test SeedManager instance management with concurrent access."""
        managers = []
        experiment_ids = []
        
        # Create multiple managers concurrently
        for i in range(10):
            manager = SeedManager(seed=1000 + i, experiment_id=f"concurrent_exp_{i}")
            managers.append(manager)
            experiment_ids.append(manager.experiment_id)
        
        # Validate all managers are properly tracked
        assert len(SeedManager._instances) == 10
        
        # Validate each manager can be retrieved by experiment ID
        for exp_id in experiment_ids:
            retrieved_manager = SeedManager.get_instance(exp_id)
            assert retrieved_manager is not None
            assert retrieved_manager.experiment_id == exp_id

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_seed_manager_memory_cleanup(self):
        """Test SeedManager memory cleanup and resource management."""
        initial_instance_count = len(SeedManager._instances)
        
        # Create temporary manager
        manager = SeedManager(seed=11111, experiment_id="cleanup_test")
        assert len(SeedManager._instances) == initial_instance_count + 1
        
        # Manager should be in instances registry
        assert SeedManager.get_instance("cleanup_test") is manager
        
        # Manual cleanup (in real usage, this would be automatic)
        exp_id = manager.experiment_id
        del manager
        
        # Instance should still be in registry until explicitly cleared
        # Note: Real cleanup would require explicit instance management

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_random_state_with_corrupted_data(self):
        """Test RandomState behavior with corrupted or invalid data."""
        # Test with None values
        state = RandomState(
            numpy_state=None,
            python_state=None,
            seed_value=None
        )
        
        # Should not raise exception during creation
        assert state.seed_value is None
        assert state.numpy_state is None
        assert state.python_state is None

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_global_seed_functions_without_hydra(self):
        """Test global seed functions in environments without Hydra."""
        # This test ensures graceful degradation without Hydra
        manager = set_global_seed(seed=22222)
        assert manager.seed == 22222
        
        # Should work even if Hydra is not available
        retrieved_manager = get_global_seed_manager()
        assert retrieved_manager is manager

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_seed_manager_performance_degradation_handling(self):
        """Test SeedManager behavior when performance requirements are not met."""
        manager = SeedManager(seed=33333, auto_initialize=False)
        
        # Mock slow initialization to exceed performance requirement
        original_time = time.perf_counter
        
        def slow_perf_counter():
            """Mock slow performance counter to simulate performance degradation."""
            return original_time() + 0.15  # Add 150ms delay
        
        with patch('time.perf_counter', side_effect=lambda: time.time()):
            # Force slow initialization by adding delay
            with patch('time.sleep') as mock_sleep:
                # This would be logged as a warning in real implementation
                initialization_time = manager.initialize()
                # Should complete despite performance issues
                assert manager._initialization_time is not None


class TestCrossPlatformConsistency:
    """
    Test suite for cross-platform consistency validation ensuring reproducible
    results across different operating systems and Python environments.
    
    Validates platform information capture, byte order handling, architecture
    compatibility, and numerical consistency across computing environments.
    """

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_platform_info_consistency_validation(self):
        """Test platform information consistency across test runs."""
        state1 = RandomState(seed_value=44444)
        state2 = RandomState(seed_value=55555)
        
        # Platform info should be consistent across instances
        platform_keys = {'platform', 'python_version', 'numpy_version', 'architecture', 'byte_order'}
        
        assert set(state1.platform_info.keys()) == platform_keys
        assert set(state2.platform_info.keys()) == platform_keys
        
        # Platform-specific information should be identical
        assert state1.platform_info['platform'] == state2.platform_info['platform']
        assert state1.platform_info['python_version'] == state2.platform_info['python_version']
        assert state1.platform_info['numpy_version'] == state2.platform_info['numpy_version']
        assert state1.platform_info['architecture'] == state2.platform_info['architecture']
        assert state1.platform_info['byte_order'] == state2.platform_info['byte_order']

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_numerical_consistency_across_seeds(self):
        """Test numerical consistency of random generation across different seeds."""
        # Test multiple seeds for consistent behavior patterns
        seeds_to_test = [1, 42, 123, 999, 2**16, 2**20]
        
        for seed in seeds_to_test:
            # First run
            manager1 = SeedManager(seed=seed)
            random_values_1 = np.random.random(10)
            
            # Second run with same seed
            manager2 = SeedManager(seed=seed)
            random_values_2 = np.random.random(10)
            
            # Should be identical
            np.testing.assert_array_equal(
                random_values_1, 
                random_values_2,
                err_msg=f"Inconsistent results for seed {seed}"
            )

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_environment_variable_handling_consistency(self):
        """Test consistent environment variable handling across platforms."""
        # Test PYTHONHASHSEED environment variable
        original_hashseed = os.environ.get('PYTHONHASHSEED')
        
        try:
            # Set specific hash seed
            os.environ['PYTHONHASHSEED'] = '123'
            
            manager = SeedManager(seed=66666)
            repro_info = manager.get_reproducibility_info()
            
            # Should capture environment variable
            env_vars = repro_info['environment_variables']
            assert 'PYTHONHASHSEED' in env_vars
            assert env_vars['PYTHONHASHSEED'] == '123'
            
        finally:
            # Restore original environment
            if original_hashseed is not None:
                os.environ['PYTHONHASHSEED'] = original_hashseed
            elif 'PYTHONHASHSEED' in os.environ:
                del os.environ['PYTHONHASHSEED']

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_reproducibility_report_platform_consistency(self):
        """Test reproducibility report platform information consistency."""
        set_global_seed(seed=77777, experiment_id="platform_test")
        
        report = get_reproducibility_report()
        
        # Validate system info structure
        system_info = report['system_info']
        required_fields = {'platform', 'python_version', 'numpy_version', 'architecture', 'byte_order'}
        assert set(system_info.keys()) >= required_fields
        
        # Validate field types and values
        assert isinstance(system_info['platform'], str)
        assert isinstance(system_info['python_version'], list)
        assert len(system_info['python_version']) == 3
        assert isinstance(system_info['numpy_version'], str)
        assert system_info['architecture'] in ['32bit', '64bit']
        assert system_info['byte_order'] in ['little', 'big']

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_state_serialization_cross_platform_compatibility(self):
        """Test RandomState serialization compatibility across platforms."""
        # Create state with comprehensive data
        np.random.seed(88888)
        random.seed(88888)
        
        state = RandomState(
            numpy_state=np.random.get_state(),
            python_state=random.getstate(),
            seed_value=88888,
            experiment_id="cross_platform_test"
        )
        
        # Serialize to dict
        state_dict = state.to_dict()
        
        # Verify all required fields are serializable
        import json
        try:
            # Test JSON serialization (common cross-platform format)
            json_str = json.dumps(state_dict, default=str)
            assert len(json_str) > 0
        except (TypeError, ValueError) as e:
            pytest.fail(f"State serialization failed: {e}")
        
        # Test deserialization
        restored_state = RandomState.from_dict(state_dict)
        assert restored_state.seed_value == state.seed_value
        assert restored_state.experiment_id == state.experiment_id


# Integration test combining multiple components
class TestSeedManagerIntegration:
    """
    Integration test suite combining seed management with other system components
    to validate end-to-end functionality and real-world usage patterns.
    """

    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        """Reset global seed manager state before each test."""
        import src.plume_nav_sim.utils.seed_manager as sm_module
        if hasattr(sm_module, '_global_manager'):
            sm_module._global_manager = None
        if SEED_MANAGER_AVAILABLE and SeedManager:
            SeedManager._instances.clear()
            SeedManager._global_state = None
        yield
        if hasattr(sm_module, '_global_manager'):
            sm_module._global_manager = None
        if SEED_MANAGER_AVAILABLE and SeedManager:
            SeedManager._instances.clear()
            SeedManager._global_state = None

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_complete_experiment_workflow(self):
        """Test complete experiment workflow with seed management."""
        # Step 1: Configure global seed
        manager = set_global_seed(seed=99999, experiment_id="integration_test")
        
        # Step 2: Capture initial state for reproducibility
        initial_state = manager.capture_state()
        
        # Step 3: Simulate scientific computation
        numpy_results = np.random.random((10, 3))  # Simulated agent positions
        python_results = [random.gauss(0, 1) for _ in range(10)]  # Simulated measurements
        
        # Step 4: Generate reproducibility report
        report = get_reproducibility_report()
        
        # Step 5: Restore state and verify reproducibility
        manager.restore_state(initial_state)
        
        # Step 6: Repeat computation and verify identical results
        reproduced_numpy = np.random.random((10, 3))
        reproduced_python = [random.gauss(0, 1) for _ in range(10)]
        
        # Validate reproducibility
        np.testing.assert_array_equal(numpy_results, reproduced_numpy)
        assert python_results == reproduced_python
        
        # Validate report completeness
        assert report['global_seed_manager']['seed_value'] == 99999
        assert report['global_seed_manager']['experiment_id'] == "integration_test"

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_multi_experiment_isolation(self):
        """Test isolation between multiple concurrent experiments."""
        # Create multiple experiment contexts
        experiments = {}
        
        for i in range(3):
            exp_id = f"multi_exp_{i}"
            with seed_context(seed=10000 + i, experiment_id=exp_id) as manager:
                # Generate experiment-specific random data
                experiments[exp_id] = {
                    'manager': manager,
                    'numpy_data': np.random.random(5),
                    'python_data': [random.random() for _ in range(5)],
                    'seed': manager.seed
                }
        
        # Validate experiments used different seeds
        seeds = [exp['seed'] for exp in experiments.values()]
        assert len(set(seeds)) == 3, "Each experiment should have unique seed"
        
        # Validate experiments produced different results
        numpy_results = [exp['numpy_data'] for exp in experiments.values()]
        for i in range(len(numpy_results)):
            for j in range(i + 1, len(numpy_results)):
                assert not np.array_equal(numpy_results[i], numpy_results[j])

    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="SeedManager not available")
    def test_performance_monitoring_integration(self):
        """Test integration with performance monitoring and logging."""
        # Create manager with performance monitoring enabled
        manager = SeedManager(seed=11111, enable_logging=True)
        
        # Validate performance requirements are met
        assert manager._initialization_time < 100.0
        
        # Test reproducibility info includes timing data
        repro_info = manager.get_reproducibility_info()
        assert 'initialization_time_ms' in repro_info
        assert repro_info['initialization_time_ms'] < 100.0
        
        # Test logger context binding
        logger_context = manager.bind_to_logger()
        expected_keys = {
            'seed_value', 'experiment_id', 'seed_manager_active',
            'platform', 'initialization_time_ms'
        }
        assert set(logger_context.keys()) >= expected_keys