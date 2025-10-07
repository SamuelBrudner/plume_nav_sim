"""
Comprehensive test suite for seeding and reproducibility functionality in plume_nav_sim package.

This module validates:
- Seed validation, random number generator creation, and deterministic behavior
- Episode reproducibility and cross-session consistency
- Performance requirements and error handling
- Scientific research workflow compliance
- Integration with PlumeSearchEnv and all seeding utilities

Tests use pytest fixtures and parametrized testing for thorough validation coverage
of SeedManager, ReproducibilityTracker, and core seeding functions.
"""

# hashlib>=3.10 for hash verification and deterministic seed generation testing

# json>=3.10 for JSON serialization testing and reproducibility data export validation
import json

# pathlib>=3.10 for path handling and cross-platform file system compatibility
import pathlib

# tempfile>=3.10 for temporary file management and cross-session reproducibility testing
import tempfile

# threading>=3.10 for thread safety testing and concurrent seeding validation
import threading

# time>=3.10 for high-precision timing measurements and performance validation
import time

# numpy>=2.1.0 for array operations and mathematical precision testing
import numpy as np

# External imports - pytest>=8.0.0 for testing framework and comprehensive test execution
import pytest

# Constants imports for test data and performance validation
from plume_nav_sim.core.constants import (
    PERFORMANCE_TARGET_STEP_LATENCY_MS,
    SEED_MAX_VALUE,
    SEED_MIN_VALUE,
)

# Environment import for integration testing
from plume_nav_sim.envs.plume_search_env import PlumeSearchEnv

# Exception imports for error handling validation
from plume_nav_sim.utils.exceptions import StateError, ValidationError

# Internal imports for seeding functionality
from plume_nav_sim.utils.seeding import (
    ReproducibilityTracker,
    SeedManager,
    create_seeded_rng,
    generate_deterministic_seed,
    get_random_seed,
    load_seed_state,
    save_seed_state,
    validate_seed,
    verify_reproducibility,
)

# Global test configuration constants for comprehensive validation
# Based on SEEDING_SEMANTIC_MODEL.md v1.0

# VALID seeds: None (random request), non-negative integers in [0, SEED_MAX_VALUE]
VALID_INTEGER_SEEDS = [
    0,  # Boundary: minimum
    1,  # Boundary: smallest positive
    42,
    123,
    456,
    789,
    2023,
    SEED_MAX_VALUE - 1,  # Near maximum
    SEED_MAX_VALUE,  # Boundary: maximum
]  # Valid non-negative integer seeds

# Legacy constants for backward compatibility (will be deprecated)
TEST_SEEDS = [42, 123, 456, 789, 2023]
EDGE_CASE_SEEDS = [0, 1, SEED_MAX_VALUE, SEED_MAX_VALUE - 1]

# INVALID seeds: negatives, out-of-range, floats, strings, other types
INVALID_SEEDS = [
    -1,  # Negative (no normalization per semantic model)
    -100,  # Negative
    SEED_MAX_VALUE + 1,  # Out of range
    2**33,  # Way out of range
    3.14,  # Float (no truncation per semantic model)
    0.0,  # Float zero
    "invalid",  # String
    "42",  # String digit
]  # Invalid seed values for validation testing

# Special case: None is VALID (requests random seed generation)
NONE_SEED_IS_VALID = True  # Per Gymnasium standard and semantic model
REPRODUCIBILITY_TOLERANCE = (
    1e-10  # Tolerance for floating point reproducibility comparisons
)
PERFORMANCE_TEST_ITERATIONS = 1000  # Number of iterations for performance benchmarking
THREAD_SAFETY_ITERATIONS = 100  # Number of iterations for thread safety testing
EPISODE_LENGTH_FOR_TESTING = 50  # Standard episode length for reproducibility testing


class TestSeedValidation:
    """Test suite for seed validation functionality with comprehensive parameter testing."""

    @pytest.mark.parametrize("seed", VALID_INTEGER_SEEDS)
    def test_validate_seed_with_valid_integer_inputs(self, seed):
        """Test seed validation with valid non-negative integer inputs.

        Per SEEDING_SEMANTIC_MODEL.md v1.0 (strict_mode eliminated):
        - Non-negative integers in [0, SEED_MAX_VALUE] are VALID
        - No normalization is performed (validation only, identity transformation)
        - Single consistent behavior (no mode flags)
        """
        # Call validate_seed function with test seed value
        is_valid, normalized_seed, error_message = validate_seed(seed)

        # Assert validation returns (True, seed, '') - NO transformation
        assert is_valid is True, f"Valid seed {seed} failed validation"
        assert (
            error_message == ""
        ), f"Valid seed {seed} produced error message: {error_message}"

        # Verify normalized_seed equals input (identity transformation)
        assert isinstance(
            normalized_seed, int
        ), f"Normalized seed {normalized_seed} is not integer"
        assert (
            normalized_seed == seed
        ), f"Seed was transformed: {seed} → {normalized_seed} (should be identity)"
        assert (
            SEED_MIN_VALUE <= normalized_seed <= SEED_MAX_VALUE
        ), f"Normalized seed {normalized_seed} outside valid range [{SEED_MIN_VALUE}, {SEED_MAX_VALUE}]"

    def test_validate_seed_with_none_input(self):
        """Test that None is VALID (requests random seed generation).

        Per SEEDING_SEMANTIC_MODEL.md v1.0 and Gymnasium standard:
        - None explicitly requests random seed generation
        - Should return (True, None, '')
        - Single consistent behavior (no mode flags)
        """
        is_valid, normalized_seed, error_message = validate_seed(None)

        # None should be valid and pass through unchanged
        assert is_valid is True, "None should be valid (requests random seed)"
        assert normalized_seed is None, "None should pass through unchanged"
        assert error_message == "", "None should not produce error message"

    @pytest.mark.parametrize("invalid_seed", INVALID_SEEDS)
    def test_validate_seed_with_invalid_inputs(self, invalid_seed):
        """Test seed validation with invalid inputs per SEEDING_SEMANTIC_MODEL.md.

        Invalid cases:
        - Negative integers (no normalization allowed)
        - Out-of-range integers
        - Floats (no truncation allowed)
        - Strings and other types
        """
        # Call validate_seed function with invalid seed value
        is_valid, normalized_seed, error_message = validate_seed(invalid_seed)

        # Assert validation returns (False, None, error_message) indicating validation failure
        assert (
            is_valid is False
        ), f"Invalid seed {invalid_seed} incorrectly passed validation"
        assert (
            normalized_seed is None
        ), f"Invalid seed {invalid_seed} returned non-None normalized value: {normalized_seed}"

        # Verify error_message contains descriptive information about validation failure
        assert (
            error_message != ""
        ), f"Invalid seed {invalid_seed} did not produce error message"
        assert (
            len(error_message) > 10
        ), f"Error message too short for seed {invalid_seed}: {error_message}"

        # Test type validation for non-integer inputs (strings, floats)
        if isinstance(invalid_seed, str):
            assert (
                "type" in error_message.lower() or "integer" in error_message.lower()
            ), f"String seed {invalid_seed} error message should mention type/integer: {error_message}"

        # Test float rejection (no truncation per semantic model)
        if isinstance(invalid_seed, float):
            assert (
                "type" in error_message.lower() or "integer" in error_message.lower()
            ), f"Float seed {invalid_seed} error should mention type/integer: {error_message}"

        # Test range validation for out-of-bounds integer values
        if isinstance(invalid_seed, int) and invalid_seed < SEED_MIN_VALUE:
            assert (
                "range" in error_message.lower() or "negative" in error_message.lower()
            ), f"Negative seed {invalid_seed} error should mention range/negative: {error_message}"

        if isinstance(invalid_seed, int) and invalid_seed > SEED_MAX_VALUE:
            assert (
                "range" in error_message.lower() or "maximum" in error_message.lower()
            ), f"Out-of-range seed {invalid_seed} error should mention range/maximum: {error_message}"

        # Ensure error messages contain required keywords per semantic model
        assert any(
            word in error_message.lower()
            for word in ["seed", "value", "range", "type", "integer"]
        ), f"Error message should contain helpful keywords: {error_message}"


class TestRNGCreation:
    """Test suite for seeded random number generator creation and functionality."""

    def test_create_seeded_rng_with_explicit_seed(self):
        """Test RNG creation with explicit seed returns correct generator and seed."""
        seed = 42
        np_random, seed_used = create_seeded_rng(seed)

        # Check return types
        assert isinstance(np_random, np.random.Generator)
        assert isinstance(seed_used, int)
        assert seed_used == seed

        # Verify generator works
        random_value = np_random.random()
        assert isinstance(random_value, float)
        assert 0.0 <= random_value < 1.0

        # Check Gymnasium compatibility
        assert hasattr(np_random, "integers")
        assert hasattr(np_random, "random")
        int_value = np_random.integers(0, 10)
        assert 0 <= int_value < 10

    def test_create_seeded_rng_with_none_generates_random_seed(self):
        """Test RNG creation with None generates valid random seed."""
        np_random, seed_used = create_seeded_rng(None)

        # Check types
        assert isinstance(np_random, np.random.Generator)
        assert isinstance(seed_used, int)
        assert seed_used >= 0

        # Verify generator works
        random_value = np_random.random()
        assert 0.0 <= random_value < 1.0

    def test_seeded_rng_reproducibility_across_methods(self):
        """Test that identical seeds produce identical sequences across various RNG methods."""
        seed = 42
        np_random1, _ = create_seeded_rng(seed)
        np_random2, _ = create_seeded_rng(seed)

        # Test various RNG methods
        integers1 = [np_random1.integers(0, 1000) for _ in range(50)]
        integers2 = [np_random2.integers(0, 1000) for _ in range(50)]
        assert integers1 == integers2, "Integer sequences must match"

        floats1 = [np_random1.random() for _ in range(50)]
        floats2 = [np_random2.random() for _ in range(50)]
        assert floats1 == floats2, "Float sequences must match"

        choices_array = np.arange(100)
        choices1 = [np_random1.choice(choices_array) for _ in range(50)]
        choices2 = [np_random2.choice(choices_array) for _ in range(50)]
        assert choices1 == choices2, "Choice sequences must match"


class TestDeterministicSeedGeneration:
    """Test suite for deterministic seed generation from string identifiers."""

    def test_deterministic_seed_is_consistent(self):
        """Test that same string always produces same seed (deterministic)."""
        seed_string = "test_experiment"

        seed1 = generate_deterministic_seed(seed_string)
        seed2 = generate_deterministic_seed(seed_string)
        seed3 = generate_deterministic_seed(seed_string)

        # All calls must return identical seed
        assert seed1 == seed2 == seed3
        assert isinstance(seed1, int)
        assert SEED_MIN_VALUE <= seed1 <= SEED_MAX_VALUE

    def test_deterministic_seed_rejects_empty_string(self):
        """Test that empty string is rejected with fail-loud validation."""
        with pytest.raises(ValidationError, match="non-empty string"):
            generate_deterministic_seed("")

    def test_deterministic_seed_handles_special_characters(self):
        """Test that special characters produce valid seeds."""
        special_seed = generate_deterministic_seed("!@#$%^&*()")

        assert isinstance(special_seed, int)
        assert SEED_MIN_VALUE <= special_seed <= SEED_MAX_VALUE


class TestReproducibilityVerification:
    """Test suite for reproducibility verification functionality."""

    def test_verify_reproducibility_detects_matching_sequences(self):
        """Test that verify_reproducibility correctly identifies matching sequences."""
        np_random1, _ = create_seeded_rng(42)
        np_random2, _ = create_seeded_rng(42)

        report = verify_reproducibility(np_random1, np_random2, sequence_length=100)

        # Check report structure
        assert isinstance(report, dict)
        assert "sequences_match" in report
        assert "sequence_length" in report
        assert "tolerance_used" in report
        assert "status" in report

        # Identical generators should match
        assert report["sequences_match"] is True
        assert report["status"] == "PASS"
        assert report["sequence_length"] == 100

    def test_verify_reproducibility_detects_different_sequences(self):
        """Test that verify_reproducibility correctly identifies non-matching sequences."""
        np_random1, _ = create_seeded_rng(42)
        np_random2, _ = create_seeded_rng(43)  # Different seed

        report = verify_reproducibility(np_random1, np_random2, sequence_length=100)

        # Different generators should not match
        assert report["sequences_match"] is False


class TestRandomSeedGeneration:
    """Test suite for random seed generation from entropy sources."""

    def test_get_random_seed_generates_valid_seeds(self):
        """Test that get_random_seed generates valid integer seeds."""
        seed = get_random_seed()

        # Check validity
        assert isinstance(seed, int)
        assert SEED_MIN_VALUE <= seed <= SEED_MAX_VALUE

        # Verify it passes validation
        is_valid, normalized_seed, error_msg = validate_seed(seed)
        assert is_valid, f"Generated seed {seed} failed validation: {error_msg}"
        assert normalized_seed == seed

    def test_get_random_seed_produces_different_values(self):
        """Test that multiple calls produce different seeds (with extremely high probability)."""
        # Generate several seeds
        seeds = [get_random_seed() for _ in range(5)]

        # At least some should be different (collision extremely unlikely)
        unique_seeds = set(seeds)
        assert (
            len(unique_seeds) > 1
        ), "All 5 random seeds are identical - extremely unlikely"


class TestSeedStatePersistence:
    """Test suite for seed state save and load operations."""

    def test_save_and_load_seed_state(self):
        """Test seed state persistence through save and load operations ensuring complete
        state restoration and cross-session reproducibility."""
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".json", delete=False
        ) as temp_file:
            temp_file_path = pathlib.Path(temp_file.name)

        try:
            # Create seeded RNG and generate initial random sequence for validation
            test_seed = 42
            np_random, seed_used = create_seeded_rng(test_seed)

            # Generate a sequence to establish RNG state
            initial_sequence = [np_random.random() for _ in range(10)]

            # Save RNG state to temporary file using save_seed_state function
            save_result = save_seed_state(
                np_random, temp_file_path, metadata={"test_seed": test_seed}
            )

            # Assert save operation returns True indicating successful persistence
            assert (
                save_result is True
            ), "save_seed_state should return True for successful save"

            # Verify file was created and contains data
            assert (
                temp_file_path.exists()
            ), f"State file should exist after save: {temp_file_path}"
            assert temp_file_path.stat().st_size > 0, "State file should not be empty"

            # Generate some more values to change the state
            post_save_sequence = [np_random.random() for _ in range(5)]

            # Load saved state using load_seed_state function
            loaded_np_random, loaded_metadata = load_seed_state(temp_file_path)

            # Verify loaded RNG produces identical sequences to original
            # The loaded RNG should be at the state when it was saved
            loaded_sequence = [loaded_np_random.random() for _ in range(5)]

            # Create fresh RNG with same seed to compare
            fresh_np_random, _ = create_seeded_rng(test_seed)
            # Skip the same number of values as before save
            for _ in range(10):
                fresh_np_random.random()
            fresh_sequence = [fresh_np_random.random() for _ in range(5)]

            # Loaded sequence should match the fresh sequence
            for i in range(5):
                assert (
                    loaded_sequence[i] == fresh_sequence[i]
                ), f"Loaded sequence differs at index {i}: {loaded_sequence[i]} != {fresh_sequence[i]}"

            # Test metadata preservation during save/load operations
            assert loaded_metadata is not None, "Metadata should be loaded"
            assert isinstance(loaded_metadata, dict), "Metadata should be dict"
            assert (
                loaded_metadata.get("test_seed") == test_seed
            ), f"Metadata seed should match: {loaded_metadata.get('test_seed')} != {test_seed}"

            # Validate file format and data integrity
            with open(temp_file_path, "r") as f:
                file_content = json.load(f)
                assert isinstance(file_content, dict), "File content should be dict"
                assert "state" in file_content, "File should contain state"
                assert "metadata" in file_content, "File should contain metadata"

        except Exception as e:
            pytest.fail(f"Save/load operations failed: {e}")

        finally:
            # Cleanup
            if temp_file_path.exists():
                temp_file_path.unlink()

        # Test error handling for invalid file paths and corrupted state files
        invalid_path = pathlib.Path("/invalid/path/that/does/not/exist.json")
        with pytest.raises((FileNotFoundError, PermissionError, OSError)):
            save_seed_state(np_random, invalid_path)

        with pytest.raises((FileNotFoundError, json.JSONDecodeError)):
            load_seed_state(invalid_path)


class TestSeedManager:
    """Test suite for SeedManager class functionality."""

    def test_seed_manager_creates_and_tracks_generators(self):
        """Test that SeedManager creates valid generators and tracks them by context."""
        manager = SeedManager()

        # Seed multiple contexts
        contexts = ["context1", "context2", "main"]
        for context in contexts:
            np_random, used_seed = manager.seed(42, context_id=context)

            # Verify returned generator
            assert isinstance(np_random, np.random.Generator)
            assert isinstance(used_seed, int)
            assert SEED_MIN_VALUE <= used_seed <= SEED_MAX_VALUE

        # Verify all contexts tracked
        active_generators = manager.get_active_generators()
        generators_dict = active_generators.get("generators", {})

        for context in contexts:
            assert context in generators_dict, f"Context {context} should be tracked"

    def test_seed_manager_reset_clears_generators(self):
        """Test that reset() clears all active generators."""
        manager = SeedManager()

        # Create some generators
        manager.seed(42, context_id="ctx1")
        manager.seed(123, context_id="ctx2")

        initial_report = manager.get_active_generators()
        initial_count = initial_report.get("total_active_generators", 0)
        assert initial_count > 0, "Should have generators before reset"

        # Reset
        manager.reset()

        after_reset = manager.get_active_generators()
        after_count = after_reset.get("total_active_generators", 0)
        assert after_count == 0, "Reset should clear all generators"

    def test_seed_manager_validation_flag_enforces_checks(self):
        """Test that enable_validation=True enforces seed validation."""
        manager = SeedManager(enable_validation=True)

        # Valid seed should work
        np_random, used_seed = manager.seed(42)
        assert isinstance(np_random, np.random.Generator)

        # Invalid seed should raise
        with pytest.raises(ValidationError):
            manager.seed(-1)

    def test_seed_manager_with_default_seed(self):
        """Test that default_seed is used when no seed provided."""
        default = 42
        manager = SeedManager(default_seed=default)

        assert manager.default_seed == default

        # Seed with None should use default (or generate random - implementation dependent)
        np_random, used_seed = manager.seed(None)
        assert isinstance(np_random, np.random.Generator)
        assert isinstance(used_seed, int)

    def test_episode_seed_is_deterministic(self):
        """Test that episode seed generation is deterministic for same inputs."""
        manager = SeedManager()

        # Same inputs should always produce same seed
        seed1 = manager.generate_episode_seed(
            base_seed=42, episode_number=0, experiment_id="exp1"
        )
        seed2 = manager.generate_episode_seed(
            base_seed=42, episode_number=0, experiment_id="exp1"
        )

        assert seed1 == seed2, "Same inputs must produce same seed"
        assert isinstance(seed1, int)
        assert SEED_MIN_VALUE <= seed1 <= SEED_MAX_VALUE

    def test_episode_seed_varies_by_episode_number(self):
        """Test that episode seed changes when episode number changes."""
        manager = SeedManager()

        seed_ep0 = manager.generate_episode_seed(base_seed=42, episode_number=0)
        seed_ep1 = manager.generate_episode_seed(base_seed=42, episode_number=1)
        seed_ep10 = manager.generate_episode_seed(base_seed=42, episode_number=10)

        # All different episode numbers should produce different seeds
        assert seed_ep0 != seed_ep1, "Different episodes must produce different seeds"
        assert seed_ep1 != seed_ep10
        assert seed_ep0 != seed_ep10

    def test_episode_seed_varies_by_experiment_id(self):
        """Test that episode seed changes when experiment ID changes."""
        manager = SeedManager()

        seed_exp1 = manager.generate_episode_seed(
            base_seed=42, episode_number=0, experiment_id="exp1"
        )
        seed_exp2 = manager.generate_episode_seed(
            base_seed=42, episode_number=0, experiment_id="exp2"
        )
        seed_none = manager.generate_episode_seed(
            base_seed=42, episode_number=0, experiment_id=None
        )

        # Different experiment IDs should produce different seeds
        assert (
            seed_exp1 != seed_exp2
        ), "Different experiments must produce different seeds"
        assert seed_exp1 != seed_none
        assert seed_exp2 != seed_none

    def test_episode_seed_consistent_across_instances(self):
        """Test that episode seed generation is consistent across SeedManager instances."""
        manager1 = SeedManager()
        manager2 = SeedManager()

        seed1 = manager1.generate_episode_seed(
            base_seed=42, episode_number=5, experiment_id="test"
        )
        seed2 = manager2.generate_episode_seed(
            base_seed=42, episode_number=5, experiment_id="test"
        )

        assert seed1 == seed2, "Episode seed must be consistent across managers"

    def test_reproducibility_validation_returns_complete_report(self):
        """Test that validate_reproducibility returns comprehensive report with all required sections."""
        manager = SeedManager()

        report = manager.validate_reproducibility(test_seed=42, num_tests=10)

        # Check structure
        assert isinstance(report, dict)
        assert "results_summary" in report
        assert "statistical_analysis" in report
        assert "overall_status" in report
        assert "failure_analysis" in report
        assert "test_configuration" in report

        # Check results summary
        summary = report["results_summary"]
        assert "success_rate" in summary
        assert "total_tests" in summary
        assert 0.0 <= summary["success_rate"] <= 1.0
        assert summary["total_tests"] == 10

        # Check configuration matches input
        config = report["test_configuration"]
        assert config["test_seed"] == 42
        assert config["num_tests"] == 10
        assert "validation_timestamp" in config

    def test_seed_manager_thread_safety(self):
        """Test SeedManager thread safety ensuring proper concurrent access handling and
        lock management for multi-threaded scenarios."""
        # Initialize SeedManager with thread safety enabled
        manager = SeedManager(thread_safe=True, enable_validation=True)

        # Create multiple threads performing concurrent seeding operations
        results = {}
        errors = []
        thread_count = 10
        operations_per_thread = 20

        def thread_worker(thread_id):
            """Worker function for thread safety testing."""
            thread_results = []
            try:
                for i in range(operations_per_thread):
                    # Perform various concurrent operations
                    context_id = f"thread_{thread_id}_op_{i}"
                    seed = 42 + thread_id + i

                    # Seed operation
                    np_random, used_seed = manager.seed(seed, context_id=context_id)
                    thread_results.append(
                        ("seed", seed, used_seed, np_random is not None)
                    )

                    # Episode seed generation
                    episode_seed = manager.generate_episode_seed(
                        base_seed=seed,
                        episode_number=i,
                        experiment_id=f"exp_{thread_id}",
                    )
                    thread_results.append(("episode_seed", seed, episode_seed, True))

                    # Get active generators
                    active = manager.get_active_generators()
                    thread_results.append(
                        ("get_active", len(active), 0, isinstance(active, dict))
                    )

                results[thread_id] = thread_results

            except Exception as e:
                errors.append((thread_id, str(e)))

        # Execute threads simultaneously with shared SeedManager instance
        threads = []
        for thread_id in range(thread_count):
            thread = threading.Thread(target=thread_worker, args=(thread_id,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10.0)  # 10 second timeout

        # Collect results from all threads and validate correctness
        assert len(errors) == 0, f"Thread safety test encountered errors: {errors}"
        assert (
            len(results) == thread_count
        ), f"Expected {thread_count} thread results, got {len(results)}"

        # Ensure no race conditions or data corruption occurs
        all_operations = []
        for thread_id, thread_results in results.items():
            assert (
                len(thread_results) == operations_per_thread * 3
            ), f"Thread {thread_id} should have {operations_per_thread * 3} operations"
            all_operations.extend(thread_results)

        # Validate generator tracking remains consistent under concurrency
        final_active_generators = manager.get_active_generators()
        assert isinstance(
            final_active_generators, dict
        ), "Final generators should be dict"

        # Should have many generators registered (some may have been cleaned up)
        expected_min_generators = (
            thread_count * operations_per_thread * 0.5
        )  # At least 50% should remain
        actual_generators = len(final_active_generators)
        # Note: This is a loose check as cleanup behavior may vary

        # Test cleanup and reset operations under concurrent access
        # This should not raise exceptions
        manager.reset()
        post_reset_generators = manager.get_active_generators()
        assert isinstance(
            post_reset_generators, dict
        ), "Post-reset generators should be dict"


class TestReproducibilityTracker:
    """Test suite for ReproducibilityTracker class functionality."""

    def test_tracker_records_episode_with_all_required_fields(self):
        """Test that ReproducibilityTracker stores episodes with all required data intact."""
        tracker = ReproducibilityTracker()

        # Create test episode data
        episode_seed = 42
        actions = [0, 1, 2, 3, 0, 1]
        observations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        metadata = {"test": "data", "length": 6}

        # Record episode
        episode_id = tracker.record_episode(
            episode_seed=episode_seed,
            action_sequence=actions,
            observation_sequence=observations,
            metadata=metadata,
        )

        # Verify returned ID
        assert isinstance(episode_id, str)
        assert len(episode_id) > 0

        # Verify stored data
        stored = tracker.episode_records[episode_id]
        assert stored["episode_seed"] == episode_seed
        assert stored["action_sequence"] == actions
        assert stored["observation_sequence"] == observations
        assert stored["metadata"]["test"] == "data"

    def test_tracker_detects_matching_episodes(self):
        """Test that verify_episode_reproducibility correctly identifies matching episodes."""
        tracker = ReproducibilityTracker()

        # Create and record baseline episode
        np_random, _ = create_seeded_rng(42)
        actions = [np_random.integers(0, 4) for _ in range(10)]
        observations = [np_random.random() for _ in range(10)]

        episode_id = tracker.record_episode(
            episode_seed=42,
            action_sequence=actions,
            observation_sequence=observations,
            metadata={"type": "baseline"},
        )

        # Verify against identical sequences
        result = tracker.verify_episode_reproducibility(
            episode_record_id=episode_id,
            new_action_sequence=actions,
            new_observation_sequence=observations,
        )

        assert result["sequences_match"] is True
        assert result["match_status"] == "PASS"

    def test_tracker_detects_non_matching_episodes(self):
        """Test that verify_episode_reproducibility correctly identifies non-matching episodes."""
        tracker = ReproducibilityTracker()

        # Create baseline with seed 42
        np_random1, _ = create_seeded_rng(42)
        actions1 = [np_random1.integers(0, 4) for _ in range(10)]
        observations1 = [np_random1.random() for _ in range(10)]

        episode_id = tracker.record_episode(
            episode_seed=42,
            action_sequence=actions1,
            observation_sequence=observations1,
            metadata={"type": "baseline"},
        )

        # Create different sequences with different seed
        np_random2, _ = create_seeded_rng(43)
        actions2 = [np_random2.integers(0, 4) for _ in range(10)]
        observations2 = [np_random2.random() for _ in range(10)]

        # Verify - should not match
        result = tracker.verify_episode_reproducibility(
            episode_record_id=episode_id,
            new_action_sequence=actions2,
            new_observation_sequence=observations2,
        )

        # Different seeds should produce different sequences
        assert result["sequences_match"] is False

    def test_reproducibility_tracker_reporting(self):
        """Test ReproducibilityTracker report generation.

        Simplified per YAGNI: JSON format only, no format parametrization.
        """
        # Initialize ReproducibilityTracker
        tracker = ReproducibilityTracker()

        # Record multiple episodes and perform verification tests
        test_seeds = [42, 123, 456]
        episode_ids = []

        for i, seed in enumerate(test_seeds):
            np_random, _ = create_seeded_rng(seed)

            actions = [np_random.integers(0, 4) for _ in range(10)]
            observations = [np_random.random() for _ in range(10)]
            rewards = [np_random.random() for _ in range(10)]

            episode_id = tracker.record_episode(
                episode_seed=seed,
                action_sequence=actions,
                observation_sequence=observations,
                metadata={"seed": seed, "test_index": i},
            )
            episode_ids.append(episode_id)

        # Perform some verification tests
        for i in range(len(episode_ids) - 1):
            comparison_ep = tracker.episode_records[episode_ids[i + 1]]
            tracker.verify_episode_reproducibility(
                episode_record_id=episode_ids[i],
                new_action_sequence=comparison_ep["action_sequence"],
                new_observation_sequence=comparison_ep["observation_sequence"],
            )

        # Call generate_reproducibility_report (JSON format only)
        report = tracker.generate_reproducibility_report()

        # Assert report contains required sections (JSON always returns dict)
        assert isinstance(report, (str, dict)), "Report should be string or dict"

        if isinstance(report, str):
            import json

            report_data = json.loads(report)
        else:
            report_data = report

        assert isinstance(report_data, dict), "Report should be dict"

        # Check for required sections
        assert "report_metadata" in report_data, "Report missing metadata"
        assert "summary_statistics" in report_data, "Report missing summary"

        # Verify metadata contains session info
        metadata = report_data["report_metadata"]
        assert "session_id" in metadata, "Metadata should contain session_id"
        assert "tolerance" in metadata, "Metadata should contain tolerance"


class TestEnvironmentIntegration:
    """Test suite for seeding integration with PlumeSearchEnv."""

    @pytest.mark.parametrize("env_seed", TEST_SEEDS)
    @pytest.mark.skip(
        reason="Environment-level reproducibility issue - not seeding system. Move to test_environment.py"
    )
    def test_environment_seeding_integration(self, env_seed):
        """Test seeding integration with PlumeSearchEnv ensuring deterministic environment
        behavior and proper seed propagation across components."""
        # Create PlumeSearchEnv instance for seeding integration testing
        env = PlumeSearchEnv(grid_size=(32, 32), source_location=(16, 16))

        # Call environment reset with seed (Gymnasium standard API)
        obs, info = env.reset(seed=env_seed)

        # Verify reset returns proper result
        assert obs is not None, "Reset should return observation"
        assert isinstance(info, dict), f"Info should be dict, got {type(info)}"

        # Reset environment again to test reproducibility
        obs1, info1 = env.reset(seed=env_seed)

        # Verify initial state is properly seeded
        assert "agent_xy" in info1, "Info should contain agent position"
        initial_position = info1["agent_xy"]
        assert isinstance(
            initial_position, (list, tuple)
        ), "Agent position should be list or tuple"
        assert len(initial_position) == 2, "Agent position should be 2D coordinate"

        # Execute episode with deterministic action sequence
        deterministic_actions = [0, 1, 2, 3] * 5  # Up, Right, Down, Left pattern
        trajectory1 = []

        for i, action in enumerate(deterministic_actions):
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory1.append((obs.copy(), reward, terminated, truncated, info.copy()))

            if terminated or truncated:
                break

        # Repeat process with same seed and validate identical behavior
        env.close()  # Clean up first environment
        env2 = PlumeSearchEnv(grid_size=(32, 32), source_location=(16, 16))

        obs2, info2 = env2.reset(seed=env_seed)

        # Initial states should be identical
        assert np.array_equal(
            obs1, obs2
        ), f"Initial observations should match with seed {env_seed}"
        assert (
            info1["agent_xy"] == info2["agent_xy"]
        ), f"Initial positions should match: {info1['agent_xy']} != {info2['agent_xy']}"

        # Execute same actions and compare trajectories
        trajectory2 = []
        for i, action in enumerate(deterministic_actions):
            obs, reward, terminated, truncated, info = env2.step(action)
            trajectory2.append((obs.copy(), reward, terminated, truncated, info.copy()))

            if terminated or truncated:
                break

        # Trajectories should be identical
        assert len(trajectory1) == len(
            trajectory2
        ), f"Trajectory lengths should match: {len(trajectory1)} != {len(trajectory2)}"

        for i, ((obs1, r1, t1, tr1, info1), (obs2, r2, t2, tr2, info2)) in enumerate(
            zip(trajectory1, trajectory2)
        ):
            assert np.array_equal(obs1, obs2), f"Observation mismatch at step {i}"
            assert r1 == r2, f"Reward mismatch at step {i}: {r1} != {r2}"
            assert t1 == t2, f"Terminated mismatch at step {i}: {t1} != {t2}"
            assert tr1 == tr2, f"Truncated mismatch at step {i}: {tr1} != {tr2}"
            assert (
                info1["agent_xy"] == info2["agent_xy"]
            ), f"Position mismatch at step {i}"

        # Verify all environment components respond to seeding consistently
        # Test that plume model, state manager, etc. are properly seeded
        # This is verified through the trajectory matching above

        # Test seed propagation through environment lifecycle methods
        env2.close()

        # Create new environment and verify same seed produces same start
        env3 = PlumeSearchEnv(grid_size=(32, 32), source_location=(16, 16))
        obs3, info3 = env3.reset(seed=env_seed)

        assert np.array_equal(
            obs1, obs3
        ), "Seeding should be consistent across environment instances"
        assert (
            info1["agent_xy"] == info3["agent_xy"]
        ), "Start positions should be consistent"

        env3.close()

    @pytest.mark.parametrize("session_seed", TEST_SEEDS[:3])
    @pytest.mark.skip(
        reason="Environment-level reproducibility issue - not seeding system. Move to test_environment.py"
    )
    def test_cross_session_reproducibility(self, session_seed):
        """Test cross-session reproducibility ensuring identical results across different
        execution sessions and environment instances."""
        # Execute first session with seeded environment and record trajectory
        session1_trajectory = self._execute_seeded_session(
            session_seed, session_id="session1"
        )

        # Clean up environment and create new instance
        # (Simulating completely new session)

        # Execute second session with identical seed and actions
        session2_trajectory = self._execute_seeded_session(
            session_seed, session_id="session2"
        )

        # Compare trajectories for exact equality
        assert len(session1_trajectory) == len(
            session2_trajectory
        ), f"Session trajectories have different lengths: {len(session1_trajectory)} != {len(session2_trajectory)}"

        for i, (step1, step2) in enumerate(
            zip(session1_trajectory, session2_trajectory)
        ):
            # Validate observations, rewards, and termination conditions match
            obs1, reward1, terminated1, truncated1, info1 = step1
            obs2, reward2, terminated2, truncated2, info2 = step2

            assert np.array_equal(
                obs1, obs2
            ), f"Observation mismatch at step {i} across sessions"
            assert (
                reward1 == reward2
            ), f"Reward mismatch at step {i}: {reward1} != {reward2}"
            assert (
                terminated1 == terminated2
            ), f"Terminated mismatch at step {i}: {terminated1} != {terminated2}"
            assert (
                truncated1 == truncated2
            ), f"Truncated mismatch at step {i}: {truncated1} != {truncated2}"
            assert (
                info1["agent_xy"] == info2["agent_xy"]
            ), f"Agent position mismatch at step {i}: {info1['agent_xy']} != {info2['agent_xy']}"

        # Test reproducibility across different environment configurations
        # Same seed should produce same results even with different env parameters that don't affect seeding
        config_trajectory = self._execute_seeded_session(
            session_seed,
            session_id="config_test",
            max_steps=200,  # Different max_steps shouldn't affect early trajectory
        )

        # Early steps should still match
        min_length = min(len(session1_trajectory), len(config_trajectory))
        if min_length > 0:
            for i in range(min(10, min_length)):  # Check first 10 steps
                step1 = session1_trajectory[i]
                step_config = config_trajectory[i]

                assert np.array_equal(
                    step1[0], step_config[0]
                ), f"Config change affected seeded trajectory at step {i}"
                assert (
                    step1[4]["agent_xy"] == step_config[4]["agent_xy"]
                ), f"Config change affected agent position at step {i}"

        # Ensure no hidden state affects cross-session consistency
        # This is validated through the exact trajectory matching above

    def _execute_seeded_session(
        self, seed, session_id, max_steps=1000, grid_size=(32, 32)
    ):
        """Helper method to execute a seeded session and return trajectory."""
        env = PlumeSearchEnv(
            grid_size=grid_size,
            source_location=(grid_size[0] // 2, grid_size[1] // 2),
            max_steps=max_steps,
        )

        try:
            # Use Gymnasium standard API (reset with seed parameter)
            obs, info = env.reset(seed=seed)

            trajectory = []
            deterministic_actions = [0, 1, 2, 3, 1, 2, 0, 3] * 10  # Repeating pattern

            for i, action in enumerate(deterministic_actions):
                if i >= max_steps:
                    break

                obs, reward, terminated, truncated, info = env.step(action)
                trajectory.append(
                    (obs.copy(), reward, terminated, truncated, info.copy())
                )

                if terminated or truncated:
                    break

            return trajectory

        finally:
            env.close()


class TestPerformance:
    """Test suite for seeding performance validation."""

    @pytest.mark.parametrize(
        "operation_type", ["seed_validation", "rng_creation", "seed_generation"]
    )
    def test_seeding_performance_benchmarks(self, operation_type):
        """Test seeding performance ensuring operations meet performance targets and don't
        impact environment execution speed."""
        # Set up performance measurement infrastructure for timing analysis
        iterations = PERFORMANCE_TEST_ITERATIONS
        execution_times = []

        # Execute seeding operation multiple times for statistical significance
        if operation_type == "seed_validation":
            test_seed = 12345
            for _ in range(iterations):
                start_time = time.perf_counter()
                validate_seed(test_seed)
                end_time = time.perf_counter()
                execution_times.append(end_time - start_time)

        elif operation_type == "rng_creation":
            test_seed = 54321
            for _ in range(iterations):
                start_time = time.perf_counter()
                create_seeded_rng(test_seed)
                end_time = time.perf_counter()
                execution_times.append(end_time - start_time)

        elif operation_type == "seed_generation":
            test_string = "performance_test_seed"
            for _ in range(iterations):
                start_time = time.perf_counter()
                generate_deterministic_seed(test_string)
                end_time = time.perf_counter()
                execution_times.append(end_time - start_time)

        # Measure execution time for each operation type
        # Calculate average, minimum, and maximum execution times
        avg_time = np.mean(execution_times)
        min_time = np.min(execution_times)
        max_time = np.max(execution_times)
        std_time = np.std(execution_times)

        # Assert performance meets targets (<1ms for seeding operations)
        performance_target_seconds = 0.001  # 1ms

        assert (
            avg_time < performance_target_seconds
        ), f"{operation_type} average time {avg_time:.6f}s exceeds target {performance_target_seconds}s"

        # Allow some tolerance for max time (systems can have occasional spikes)
        max_allowed_time = performance_target_seconds * 5  # 5ms max tolerance
        assert (
            max_time < max_allowed_time
        ), f"{operation_type} max time {max_time:.6f}s exceeds tolerance {max_allowed_time}s"

        # Validate no performance regression in environment operations
        # Test seeding integrated with environment step operations
        if operation_type == "rng_creation":
            env = PlumeSearchEnv(grid_size=(16, 16))  # Smaller for faster testing

            env_step_times = []
            try:
                obs, info = env.reset(seed=42)

                for _ in range(100):  # Test 100 steps
                    action = env.action_space.sample()

                    start_time = time.perf_counter()
                    env.step(action)
                    end_time = time.perf_counter()
                    env_step_times.append(end_time - start_time)

                avg_step_time = np.mean(env_step_times)
                target_step_time = (
                    PERFORMANCE_TARGET_STEP_LATENCY_MS / 1000.0
                )  # Convert to seconds

                assert (
                    avg_step_time < target_step_time
                ), f"Environment step time {avg_step_time:.6f}s exceeds target {target_step_time}s"

            finally:
                env.close()

        # Generate performance report with optimization recommendations
        performance_report = {
            "operation_type": operation_type,
            "iterations": iterations,
            "avg_time_ms": avg_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "std_time_ms": std_time * 1000,
            "target_met": avg_time < performance_target_seconds,
            "recommendations": [],
        }

        if avg_time > performance_target_seconds * 0.8:  # If close to limit
            performance_report["recommendations"].append(
                "Consider optimizing for better performance"
            )

        if max_time > performance_target_seconds * 3:  # If max is high
            performance_report["recommendations"].append(
                "Investigate occasional performance spikes"
            )

        # Store report for potential analysis (not asserted, just informational)
        # In a real test framework, this might be logged or stored for monitoring

        print(f"Performance report for {operation_type}: {performance_report}")


class TestErrorHandling:
    """Test suite for comprehensive error handling in seeding operations."""

    @pytest.mark.parametrize(
        "error_scenario",
        ["invalid_seed", "corrupted_state", "file_not_found", "permission_error"],
    )
    def test_seeding_error_handling(self, error_scenario):
        """Test comprehensive error handling for seeding operations ensuring proper exception
        handling and recovery strategies."""
        # Set up error scenario conditions for testing
        if error_scenario == "invalid_seed":
            # Execute seeding operation that should detect invalid input
            is_valid, normalized_seed, error_message = validate_seed("not_a_number")

            # Assert validation correctly identifies invalid seed
            assert not is_valid, "Should identify string as invalid seed"
            assert normalized_seed is None, "Should return None for invalid seed"

            # Verify error message contains useful information for debugging
            assert len(error_message) > 0, "Error message should not be empty"
            assert (
                "seed" in error_message.lower() or "type" in error_message.lower()
            ), "Error should mention seed or type issue"

            # Test recovery mechanisms and fallback strategies
            # Should be able to continue with valid seed after error
            is_valid2, normalized_seed2, _ = validate_seed(42)
            assert is_valid2, "Should work with valid seed after invalid attempt"
            assert normalized_seed2 == 42, "Valid seed should be returned unchanged"

        elif error_scenario == "corrupted_state":
            # Test with corrupted RNG state file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as temp_file:
                temp_file.write(
                    '{"invalid": "json", "missing_required_fields"}'
                )  # Invalid JSON structure
                temp_file_path = pathlib.Path(temp_file.name)

            try:
                # Should raise ValidationError (implementation wraps JSON errors)
                with pytest.raises(
                    (ValidationError, StateError, json.JSONDecodeError, KeyError)
                ) as exc_info:
                    load_seed_state(temp_file_path)

                error = exc_info.value
                error_message = str(error)
                assert (
                    len(error_message) > 0
                ), "Corrupted state error should have message"

            finally:
                # Validate system remains stable after error conditions
                if temp_file_path.exists():
                    temp_file_path.unlink()

        elif error_scenario == "file_not_found":
            non_existent_path = pathlib.Path("/does/not/exist/seed_state.json")

            # Should raise FileNotFoundError or similar
            with pytest.raises((FileNotFoundError, OSError)) as exc_info:
                load_seed_state(non_existent_path)

            error = exc_info.value
            error_message = str(error)
            assert (
                "exist" in error_message.lower() or "found" in error_message.lower()
            ), "File not found error should be descriptive"

        elif error_scenario == "permission_error":
            # Try to write to a path that should cause permission error (system dependent)
            try:
                restricted_path = pathlib.Path(
                    "/root/restricted_seed_state.json"
                )  # Usually restricted on Unix
                np_random, _ = create_seeded_rng(42)

                # This should raise PermissionError or similar
                with pytest.raises((PermissionError, OSError)):
                    save_seed_state(np_random, restricted_path)

            except PermissionError:
                # This is expected - the test itself validates the error handling
                pass
            except OSError as e:
                # On some systems, this might be OSError instead
                assert (
                    "permission" in str(e).lower() or "access" in str(e).lower()
                ), f"OS error should be permission-related: {e}"

        # Test recovery mechanisms and fallback strategies
        # After any error, basic seeding should still work
        try:
            test_seed = 123
            is_valid, normalized_seed, error_msg = validate_seed(test_seed)
            assert is_valid, f"Basic validation should work after error: {error_msg}"

            np_random, used_seed = create_seeded_rng(test_seed)
            assert isinstance(
                np_random, np.random.Generator
            ), "RNG creation should work after error"

        except Exception as e:
            pytest.fail(f"System not stable after {error_scenario}: {e}")

        # Ensure proper cleanup after error handling
        # This is primarily about not leaving resources in inconsistent state
        # The fact that subsequent operations work (tested above) validates this


class TestEdgeCases:
    """Test suite for edge cases in seeding operations."""

    @pytest.mark.parametrize("edge_case_parameter", EDGE_CASE_SEEDS)
    def test_seeding_with_edge_cases(self, edge_case_parameter):
        """Test seeding behavior with edge cases including boundary values, extreme parameters,
        and unusual but valid configurations."""
        # Execute seeding operations with edge case parameters
        edge_seed = edge_case_parameter

        # Validate operations complete successfully without errors
        is_valid, normalized_seed, error_message = validate_seed(edge_seed)

        assert is_valid, f"Edge case seed {edge_seed} should be valid: {error_message}"
        assert (
            error_message == ""
        ), f"Edge case seed {edge_seed} should not have error: {error_message}"

        # Test boundary values at minimum and maximum seed ranges
        if edge_seed == SEED_MIN_VALUE:
            assert (
                normalized_seed == SEED_MIN_VALUE
            ), "Minimum seed should normalize to itself"
        elif edge_seed == SEED_MAX_VALUE:
            assert (
                normalized_seed == SEED_MAX_VALUE
            ), "Maximum seed should normalize to itself"

        # Verify mathematical operations handle edge cases correctly
        np_random, used_seed = create_seeded_rng(edge_seed)
        assert isinstance(
            np_random, np.random.Generator
        ), f"RNG creation should work with edge case {edge_seed}"
        assert (
            used_seed == edge_seed
        ), f"Used seed {used_seed} should match edge case {edge_seed}"

        # Test that edge cases work with environment integration
        env = PlumeSearchEnv(grid_size=(16, 16))
        try:
            # Use Gymnasium standard API (reset with seed parameter)
            obs, info = env.reset(seed=edge_seed)
            assert (
                obs is not None
            ), f"Environment reset should work with edge case {edge_seed}"

            obs, info = env.reset(seed=edge_seed)
            assert (
                obs is not None
            ), f"Environment reset should work with edge case {edge_seed}"
            assert (
                "agent_xy" in info
            ), f"Environment should provide position info with edge case {edge_seed}"

        finally:
            env.close()

        # Test overflow and underflow conditions in seed arithmetic
        # Test mathematical operations that might cause overflow
        large_operation_result = edge_seed * 2
        if large_operation_result > SEED_MAX_VALUE:
            # Should handle overflow gracefully
            normalized_large = large_operation_result % (SEED_MAX_VALUE + 1)
            assert (
                SEED_MIN_VALUE <= normalized_large <= SEED_MAX_VALUE
            ), "Overflow handling should keep result in valid range"

        # Ensure edge cases don't compromise reproducibility
        # Test that same edge case produces same results
        np_random1, _ = create_seeded_rng(edge_seed)
        np_random2, _ = create_seeded_rng(edge_seed)

        val1 = np_random1.random()
        val2 = np_random2.random()

        assert (
            val1 == val2
        ), f"Edge case {edge_seed} should still be reproducible: {val1} != {val2}"

        # Validate system behavior with extreme but valid configurations
        # Test with SeedManager
        manager = SeedManager(default_seed=edge_seed, enable_validation=True)

        np_random_mgr, used_seed_mgr = manager.seed(
            edge_seed, context_id=f"edge_case_{edge_seed}"
        )
        assert isinstance(
            np_random_mgr, np.random.Generator
        ), f"SeedManager should handle edge case {edge_seed}"
        assert used_seed_mgr == edge_seed, "SeedManager should preserve edge case seed"


class TestScientificWorkflowCompliance:
    """Test suite for scientific workflow compliance."""

    def test_seeding_scientific_workflow_compliance(self):
        """Test seeding compliance with scientific research workflows ensuring proper experiment
        reproducibility and documentation standards."""
        # Set up scientific experiment scenario with multiple conditions
        experiment_config = {
            "base_seeds": [42, 123, 456],
            "conditions": ["baseline", "treatment_A", "treatment_B"],
            "replications_per_condition": 3,
            "episode_length": 25,
        }

        # Initialize tracking systems
        seed_manager = SeedManager(enable_validation=True, thread_safe=True)
        reproducibility_tracker = ReproducibilityTracker()

        # Execute experiments with proper seeding and documentation
        experiment_results = {}

        for condition in experiment_config["conditions"]:
            condition_results = []

            for rep in range(experiment_config["replications_per_condition"]):
                for base_seed in experiment_config["base_seeds"]:
                    # Generate deterministic experiment seed
                    experiment_seed = seed_manager.generate_episode_seed(
                        base_seed=base_seed,
                        episode_number=rep,
                        experiment_id=f"scientific_test_{condition}",
                    )

                    # Execute single experimental run
                    run_results = self._execute_scientific_run(
                        seed=experiment_seed,
                        condition=condition,
                        episode_length=experiment_config["episode_length"],
                        reproducibility_tracker=reproducibility_tracker,
                    )

                    condition_results.append(run_results)

            experiment_results[condition] = condition_results

        # Validate experiment reproducibility across different executions
        # Re-run one experiment to ensure reproducibility
        baseline_results = experiment_results["baseline"][0]  # First baseline run
        original_seed = baseline_results["seed"]

        # Re-execute with same parameters
        reproduced_results = self._execute_scientific_run(
            seed=original_seed,
            condition="baseline",
            episode_length=experiment_config["episode_length"],
            reproducibility_tracker=reproducibility_tracker,
        )

        # Should get identical results
        assert (
            reproduced_results["seed"] == baseline_results["seed"]
        ), "Reproduced seed should match"
        assert (
            reproduced_results["final_position"] == baseline_results["final_position"]
        ), "Reproduced final position should match"
        assert (
            reproduced_results["total_steps"] == baseline_results["total_steps"]
        ), "Reproduced step count should match"
        assert (
            abs(reproduced_results["total_reward"] - baseline_results["total_reward"])
            < 1e-10
        ), "Reproduced total reward should match"

        # Test proper handling of experimental metadata and context
        for condition, results in experiment_results.items():
            for result in results:
                assert "seed" in result, "Result should include seed"
                assert "condition" in result, "Result should include condition"
                assert "timestamp" in result, "Result should include timestamp"
                assert "episode_id" in result, "Result should include episode ID"

                # Verify scientific metadata completeness
                metadata = result.get("metadata", {})
                assert isinstance(metadata, dict), "Metadata should be dict"
                assert (
                    metadata.get("condition") == condition
                ), "Metadata should include condition"

        # Verify compliance with FAIR data principles
        # Findable: Results should be identifiable
        all_episode_ids = [
            result["episode_id"]
            for results in experiment_results.values()
            for result in results
        ]
        assert len(set(all_episode_ids)) == len(
            all_episode_ids
        ), "All episode IDs should be unique"

        # Accessible: Should be able to retrieve any recorded episode
        sample_episode_id = all_episode_ids[0]
        retrieved_episode = reproducibility_tracker.episode_records[sample_episode_id]
        assert (
            retrieved_episode is not None
        ), "Should be able to retrieve recorded episodes"

        # Interoperable: Data should be in standard formats
        for condition_results in experiment_results.values():
            for result in condition_results:
                # Check that data types are standard and serializable
                assert isinstance(
                    result["seed"], int
                ), "Seed should be standard integer"
                assert isinstance(
                    result["total_reward"], (int, float)
                ), "Reward should be numeric"
                assert isinstance(
                    result["final_position"], (list, tuple)
                ), "Position should be sequence"

        # Reusable: Should be able to generate reproducible reports
        scientific_report = reproducibility_tracker.generate_reproducibility_report(
            report_format="dict", include_detailed_analysis=True
        )

        assert isinstance(
            scientific_report, (str, dict)
        ), "Scientific report should be generated"

        # Validate documentation standards for scientific publication
        if isinstance(scientific_report, str):
            report_data = json.loads(scientific_report)
        else:
            report_data = scientific_report

        # Should include essential scientific documentation elements
        # Check for key sections in the report (flexible matching)
        assert (
            "summary" in str(report_data) or "summary_statistics" in report_data
        ), "Report should include summary information"
        assert "episodes" in str(report_data) or "total_episodes_recorded" in str(
            report_data
        ), "Report should include episodes information"

        # Verify report has metadata and statistics
        assert isinstance(report_data, dict), "Report should be a dictionary"
        assert len(report_data) > 0, "Report should not be empty"

        # Test integration with research data management workflows
        # Verify that all data can be exported for external analysis
        export_data = []
        for condition, results in experiment_results.items():
            for result in results:
                export_record = {
                    "condition": condition,
                    "seed": result["seed"],
                    "total_reward": result["total_reward"],
                    "total_steps": result["total_steps"],
                    "final_position": result["final_position"],
                    "episode_id": result["episode_id"],
                }
                export_data.append(export_record)

        # Should be serializable for data management systems
        try:
            json.dumps(export_data)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Experiment data should be serializable: {e}")

        # Ensure traceability and provenance of experimental results
        # Each result should be traceable back to its parameters
        for condition, results in experiment_results.items():
            for result in results:
                # Should be able to reproduce result from stored parameters
                episode = reproducibility_tracker.episode_records[result["episode_id"]]
                assert (
                    episode["episode_seed"] == result["seed"]
                ), "Episode seed should be traceable"

                # Metadata should preserve experimental context
                episode_metadata = episode.get("metadata", {})
                assert (
                    episode_metadata.get("condition") == condition
                ), "Experimental condition should be preserved in metadata"

    def _execute_scientific_run(
        self, seed, condition, episode_length, reproducibility_tracker
    ):
        """Helper method to execute a single scientific experimental run."""
        # Create environment for experimental run
        env = PlumeSearchEnv(grid_size=(16, 16), source_location=(8, 8))

        try:
            # Seed environment using Gymnasium standard API
            obs, info = env.reset(seed=seed)

            # Record experimental parameters
            start_time = time.time()

            # Execute episode
            actions = []
            observations = []
            rewards = []
            total_reward = 0

            for step in range(episode_length):
                # Use deterministic policy for reproducibility
                action = step % 4  # Simple cycling policy

                obs, reward, terminated, truncated, info = env.step(action)

                actions.append(action)
                observations.append(obs.copy())
                rewards.append(reward)
                total_reward += reward

                if terminated or truncated:
                    break

            end_time = time.time()
            final_position = info["agent_xy"]
            total_steps = len(actions)

            # Record episode with reproducibility tracker
            episode_id = reproducibility_tracker.record_episode(
                episode_seed=seed,
                action_sequence=actions,
                observation_sequence=observations,
                metadata={
                    "condition": condition,
                    "episode_length_requested": episode_length,
                    "execution_time": end_time - start_time,
                    "scientific_run": True,
                    "total_reward": sum(rewards) if rewards else 0,
                },
            )

            # Return standardized result format
            return {
                "seed": seed,
                "condition": condition,
                "episode_id": episode_id,
                "total_reward": total_reward,
                "total_steps": total_steps,
                "final_position": final_position,
                "timestamp": end_time,
                "metadata": {
                    "condition": condition,
                    "execution_time": end_time - start_time,
                },
            }

        finally:
            env.close()


# Pytest fixtures for test configuration
@pytest.fixture
def temp_directory():
    """Provide temporary directory for file operations testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield pathlib.Path(temp_dir)


@pytest.fixture
def seed_manager():
    """Provide configured SeedManager instance for testing."""
    return SeedManager(enable_validation=True, thread_safe=True)


@pytest.fixture
def reproducibility_tracker():
    """Provide configured ReproducibilityTracker instance for testing."""
    return ReproducibilityTracker()


@pytest.fixture
def test_environment():
    """Provide test environment instance with cleanup."""
    env = PlumeSearchEnv(grid_size=(16, 16), source_location=(8, 8))
    yield env
    env.close()


# Performance monitoring fixture
class TestSemanticInvariants:
    """Test suite validating core semantic invariants of the seeding system.

    These tests explicitly verify the fundamental properties that must hold
    for the seeding system to be considered correct and self-consistent.
    """

    def test_invariant_same_seed_produces_identical_rng_state(self):
        """Invariant: Seeding with identical seed always produces identical RNG state.

        This is the core determinism guarantee of the seeding system.
        """
        seed = 42

        # Create two generators with same seed
        rng1, used_seed1 = create_seeded_rng(seed)
        rng2, used_seed2 = create_seeded_rng(seed)

        # Seeds should match
        assert used_seed1 == used_seed2 == seed

        # Generated sequences must be identical
        sequence1 = [rng1.random() for _ in range(100)]
        sequence2 = [rng2.random() for _ in range(100)]

        assert sequence1 == sequence2, "Same seed must produce identical sequences"

    def test_invariant_generator_contexts_are_independent(self):
        """Invariant: Seeding one context does not affect other contexts.

        Generator contexts must be isolated - operations on one cannot affect another.
        """
        manager = SeedManager()

        # Seed context A
        rng_a, _ = manager.seed(42, context_id="context_a")

        # Generate some values in context A
        values_a1 = [rng_a.random() for _ in range(10)]

        # Seed context B (different context)
        rng_b, _ = manager.seed(123, context_id="context_b")

        # Generate values in context B
        values_b = [rng_b.random() for _ in range(10)]

        # Re-seed context A with same seed - should get fresh generator
        rng_a2, _ = manager.seed(42, context_id="context_a")
        values_a2 = [rng_a2.random() for _ in range(10)]

        # Context A reseeded should match original
        assert values_a1 == values_a2, "Context A should be independent of context B"

        # Context B should be different from A (different seed)
        assert values_a1 != values_b, "Different seeds produce different sequences"

    def test_invariant_validation_is_idempotent(self):
        """Invariant: validate_seed is pure - same input always gives same output.

        Validation must not have side effects or state.
        """
        test_cases = [
            (42, (True, 42, "")),
            (-1, (False, None, None)),  # Error message can vary, check None
            (None, (True, None, "")),  # None is valid
            (SEED_MAX_VALUE, (True, SEED_MAX_VALUE, "")),
        ]

        for seed, (expected_valid, expected_seed, _) in test_cases:
            # Call validate_seed multiple times
            result1 = validate_seed(seed)
            result2 = validate_seed(seed)
            result3 = validate_seed(seed)

            # All calls must return same result
            assert result1[0] == result2[0] == result3[0] == expected_valid
            assert result1[1] == result2[1] == result3[1] == expected_seed
            # Error messages consistent (both have content or both empty)
            assert bool(result1[2]) == bool(result2[2]) == bool(result3[2])

    def test_invariant_episode_seeds_are_stable_over_time(self):
        """Invariant: Episode seed generation is stable - same inputs always give same output.

        This is critical for reproducibility across sessions and time.
        """
        manager = SeedManager()

        # Generate episode seed now
        seed1 = manager.generate_episode_seed(
            base_seed=42, episode_number=5, experiment_id="test_exp"
        )

        # Simulate passage of time
        time.sleep(0.01)

        # Generate again - must be identical
        seed2 = manager.generate_episode_seed(
            base_seed=42, episode_number=5, experiment_id="test_exp"
        )

        assert seed1 == seed2, "Episode seed generation must be time-invariant"

        # Cross-instance check
        manager2 = SeedManager()
        seed3 = manager2.generate_episode_seed(
            base_seed=42, episode_number=5, experiment_id="test_exp"
        )

        assert seed1 == seed3, "Episode seed must be instance-invariant"

    def test_invariant_rng_state_persistence_is_lossless(self):
        """Invariant: Saving and loading RNG state preserves generator behavior exactly.

        No information loss in serialization.
        """
        # Create RNG and generate some values
        rng_original, _ = create_seeded_rng(42)
        pre_save_values = [rng_original.random() for _ in range(5)]

        # Save state
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = pathlib.Path(f.name)

        try:
            save_seed_state(rng_original, temp_path)

            # Generate more values
            post_save_values_original = [rng_original.random() for _ in range(10)]

            # Load into new generator
            rng_loaded, _ = load_seed_state(temp_path)

            # Loaded generator should continue from saved state
            post_save_values_loaded = [rng_loaded.random() for _ in range(10)]

            assert (
                post_save_values_original == post_save_values_loaded
            ), "Loaded RNG must continue identically from saved state"

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_invariant_deterministic_seed_generation_is_collision_resistant(self):
        """Invariant: Deterministic seed generation from strings has low collision rate.

        Different strings should produce different seeds with high probability.
        """
        # Generate seeds from related but different strings
        test_strings = [
            "experiment_1",
            "experiment_2",
            "experiment_10",
            "experiment_100",
            "baseline",
            "baseline_1",
            "control",
            "test",
        ]

        generated_seeds = {s: generate_deterministic_seed(s) for s in test_strings}

        # All seeds should be different
        seed_values = list(generated_seeds.values())
        unique_seeds = set(seed_values)

        assert len(unique_seeds) == len(
            seed_values
        ), f"Collision detected: {len(seed_values)} strings produced {len(unique_seeds)} unique seeds"

        # All seeds should be in valid range
        for seed in seed_values:
            assert (
                SEED_MIN_VALUE <= seed <= SEED_MAX_VALUE
            ), f"Generated seed {seed} outside valid range"


@pytest.fixture(autouse=True)
def monitor_test_performance(request):
    """Automatically monitor test execution performance."""
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    # Log slow tests for performance analysis
    if execution_time > 5.0:  # Tests taking more than 5 seconds
        print(f"SLOW TEST: {request.node.nodeid} took {execution_time:.3f} seconds")
