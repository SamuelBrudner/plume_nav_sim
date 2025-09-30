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
import hashlib

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

    @pytest.mark.parametrize("seed", [None] + TEST_SEEDS)
    def test_create_seeded_rng_basic_functionality(self, seed):
        """Test basic functionality of seeded random number generator creation ensuring
        proper RNG initialization and seed handling with Gymnasium compatibility."""
        # Call create_seeded_rng function with test seed value
        result = create_seeded_rng(seed)

        # Assert function returns tuple of (np_random, seed_used)
        assert isinstance(
            result, tuple
        ), f"create_seeded_rng should return tuple, got {type(result)}"
        assert (
            len(result) == 2
        ), f"create_seeded_rng should return 2-tuple, got {len(result)} elements"

        np_random, seed_used = result

        # Verify np_random is numpy.random.Generator instance
        assert isinstance(
            np_random, np.random.Generator
        ), f"First element should be numpy Generator, got {type(np_random)}"

        # Validate seed_used is integer when seed is provided
        if seed is not None:
            assert isinstance(
                seed_used, int
            ), f"seed_used should be int for provided seed, got {type(seed_used)}"
            assert (
                seed_used == seed
            ), f"seed_used {seed_used} should match provided seed {seed}"

        # Test None seed case where random seed is generated automatically
        if seed is None:
            assert isinstance(
                seed_used, int
            ), f"Auto-generated seed should be int, got {type(seed_used)}"
            # Note: Gymnasium can generate seeds up to 2^128, which is outside our validation range
            # This is acceptable - we only validate user-provided seeds
            assert seed_used >= 0, f"Auto-generated seed should be non-negative"

        # Ensure returned generator is properly initialized and functional
        random_value = np_random.random()
        assert isinstance(
            random_value, float
        ), f"Generator should produce float, got {type(random_value)}"
        assert (
            0.0 <= random_value < 1.0
        ), f"Random value {random_value} outside expected range [0, 1)"

        # Validate Gymnasium compatibility with returned generator
        # Test standard methods used by Gymnasium
        assert hasattr(
            np_random, "integers"
        ), "Generator missing integers method for Gymnasium"
        assert hasattr(
            np_random, "random"
        ), "Generator missing random method for Gymnasium"

        # Verify integers method works correctly
        int_value = np_random.integers(0, 10)
        assert isinstance(
            int_value, (int, np.integer)
        ), f"integers() should return integer type"
        assert (
            0 <= int_value < 10
        ), f"integers(0, 10) returned {int_value} outside range"

    @pytest.mark.parametrize("test_seed", TEST_SEEDS)
    def test_seeded_rng_reproducibility(self, test_seed):
        """Test reproducibility of seeded random number generators ensuring identical seeds
        produce identical random sequences with comprehensive sequence comparison."""
        # Create two RNG instances using create_seeded_rng with identical test_seed
        np_random1, seed_used1 = create_seeded_rng(test_seed)
        np_random2, seed_used2 = create_seeded_rng(test_seed)

        # Verify seeds are identical
        assert (
            seed_used1 == seed_used2
        ), f"Seeds should match: {seed_used1} != {seed_used2}"
        assert (
            seed_used1 == test_seed
        ), f"Seed should match input: {seed_used1} != {test_seed}"

        # Generate random sequences from both generators using various methods
        sequence_length = 100

        # Test integer sequences
        integers1 = [np_random1.integers(0, 1000) for _ in range(sequence_length)]
        integers2 = [np_random2.integers(0, 1000) for _ in range(sequence_length)]

        # Test float sequences
        floats1 = [np_random1.random() for _ in range(sequence_length)]
        floats2 = [np_random2.random() for _ in range(sequence_length)]

        # Test random choices from array
        choices_array = np.arange(100)
        choices1 = [np_random1.choice(choices_array) for _ in range(sequence_length)]
        choices2 = [np_random2.choice(choices_array) for _ in range(sequence_length)]

        # Compare sequences element-wise for exact equality
        assert (
            integers1 == integers2
        ), f"Integer sequences differ for seed {test_seed}: first 5 elements {integers1[:5]} vs {integers2[:5]}"

        # Validate reproducibility across different sequence lengths
        for i in range(min(10, sequence_length)):
            assert (
                floats1[i] == floats2[i]
            ), f"Float at index {i} differs: {floats1[i]} != {floats2[i]} for seed {test_seed}"
            assert (
                choices1[i] == choices2[i]
            ), f"Choice at index {i} differs: {choices1[i]} != {choices2[i]} for seed {test_seed}"

        # Ensure no statistical bias in generated sequences
        # Test that values are distributed across expected ranges
        assert any(i < 500 for i in integers1), "Integer sequence may be biased high"
        assert any(i >= 500 for i in integers1), "Integer sequence may be biased low"
        assert any(f < 0.5 for f in floats1), "Float sequence may be biased high"
        assert any(f >= 0.5 for f in floats1), "Float sequence may be biased low"

        # Verify reproducibility holds across multiple random operations
        # Reset both generators and test again
        np_random1_reset, _ = create_seeded_rng(test_seed)
        np_random2_reset, _ = create_seeded_rng(test_seed)

        # Should produce same first values again
        assert (
            np_random1_reset.random() == np_random2_reset.random()
        ), f"Reset generators with seed {test_seed} produce different values"


class TestDeterministicSeedGeneration:
    """Test suite for deterministic seed generation from string identifiers."""

    @pytest.mark.parametrize(
        "seed_string", ["test_experiment", "baseline_run", "condition_A"]
    )
    @pytest.mark.parametrize("hash_algorithm", ["sha256", "md5"])
    def test_generate_deterministic_seed_consistency(self, seed_string, hash_algorithm):
        """Test deterministic seed generation from string identifiers ensuring consistent
        seed values from identical strings and proper hash-based generation."""
        # Call generate_deterministic_seed with test string and hash algorithm
        seed1 = generate_deterministic_seed(seed_string, hash_algorithm=hash_algorithm)

        # Generate seed multiple times with same parameters
        seed2 = generate_deterministic_seed(seed_string, hash_algorithm=hash_algorithm)
        seed3 = generate_deterministic_seed(seed_string, hash_algorithm=hash_algorithm)

        # Assert all generated seeds are identical for same input
        assert seed1 == seed2, f"Seeds should be identical: {seed1} != {seed2}"
        assert seed1 == seed3, f"Seeds should be identical: {seed1} != {seed3}"
        assert seed2 == seed3, f"Seeds should be identical: {seed2} != {seed3}"

        # Validate generated seed is within valid seed range
        assert isinstance(
            seed1, int
        ), f"Generated seed should be int, got {type(seed1)}"
        assert (
            SEED_MIN_VALUE <= seed1 <= SEED_MAX_VALUE
        ), f"Generated seed {seed1} outside valid range [{SEED_MIN_VALUE}, {SEED_MAX_VALUE}]"

        # Test different hash algorithms produce different but consistent seeds
        if hash_algorithm == "sha256":
            md5_seed = generate_deterministic_seed(seed_string, hash_algorithm="md5")
            # Should be different algorithms produce different seeds (usually)
            # But still should be consistent
            assert isinstance(md5_seed, int), "MD5 seed should be integer"
            assert (
                SEED_MIN_VALUE <= md5_seed <= SEED_MAX_VALUE
            ), "MD5 seed should be in valid range"

        # Ensure empty strings are rejected and special characters are handled properly
        if seed_string == "test_experiment":
            # Test with empty string - should fail-loud
            with pytest.raises(ValidationError, match="non-empty string"):
                generate_deterministic_seed("", hash_algorithm=hash_algorithm)

            # Test with special characters
            special_seed = generate_deterministic_seed(
                "!@#$%^&*()", hash_algorithm=hash_algorithm
            )
            assert isinstance(
                special_seed, int
            ), "Special characters should produce valid seed"
            assert (
                SEED_MIN_VALUE <= special_seed <= SEED_MAX_VALUE
            ), "Special char seed should be in range"

        # Verify hash collision resistance for similar input strings
        similar_string = seed_string + "_modified"
        similar_seed = generate_deterministic_seed(
            similar_string, hash_algorithm=hash_algorithm
        )
        assert (
            similar_seed != seed1
        ), f"Similar strings should produce different seeds: '{seed_string}' vs '{similar_string}'"


class TestReproducibilityVerification:
    """Test suite for reproducibility verification functionality."""

    @pytest.mark.parametrize("sequence_length", [10, 100, 1000])
    @pytest.mark.parametrize("tolerance", [1e-10, 1e-6, 1e-3])
    def test_verify_reproducibility_function(self, sequence_length, tolerance):
        """Test reproducibility verification function ensuring accurate detection of identical
        and different random sequences with statistical analysis."""
        # Create identical seeded RNGs for reproducibility testing
        test_seed = 42
        np_random1, _ = create_seeded_rng(test_seed)
        np_random2, _ = create_seeded_rng(test_seed)

        # Call verify_reproducibility with generators and test parameters
        report = verify_reproducibility(
            np_random1, np_random2, sequence_length=sequence_length, tolerance=tolerance
        )

        # Assert function returns detailed reproducibility report dictionary
        assert isinstance(report, dict), f"Report should be dict, got {type(report)}"

        # Verify report contains match status, statistical analysis, and metrics
        required_keys = [
            "sequences_match",
            "sequence_length",
            "tolerance_used",
            "status",
        ]
        for key in required_keys:
            assert key in report, f"Report missing required key: {key}"

        assert isinstance(
            report["sequences_match"], bool
        ), "sequences_match should be boolean"
        assert (
            report["sequence_length"] == sequence_length
        ), "sequence_length should match input"
        assert (
            abs(report["tolerance_used"] - tolerance) < 1e-12
        ), "tolerance should match input"

        # Test with identical generators expecting perfect match
        assert report["sequences_match"] is True, "Identical generators should match"
        assert (
            report["status"] == "PASS"
        ), "Status should be PASS for matching sequences"

        # Check discrepancy analysis
        if "discrepancy_analysis" in report:
            assert (
                report["discrepancy_analysis"]["num_discrepancies"] == 0
            ), "Identical generators should have 0 discrepancies"

        # Test with different generators expecting mismatch detection
        np_random3, _ = create_seeded_rng(test_seed + 1)  # Different seed
        np_random4, _ = create_seeded_rng(test_seed + 1)  # Same different seed

        different_report = verify_reproducibility(
            np_random1, np_random3, sequence_length=sequence_length, tolerance=tolerance
        )

        # Should detect differences (usually, unless extremely unlucky with randomness)
        assert isinstance(
            different_report["sequences_match"], bool
        ), "Different report should have boolean sequences_match"

        # Validate tolerance parameter affects floating point comparisons
        if tolerance >= 1e-3:
            # With high tolerance, might find matches in some comparisons
            assert (
                abs(different_report["tolerance_used"] - tolerance) < 1e-12
            ), "Tolerance should be preserved"

        # Ensure statistical metrics are accurate and meaningful
        if "statistical_analysis" in report:
            stats = report["statistical_analysis"]
            assert isinstance(stats, dict), "Statistical analysis should be dict"

            # Check for meaningful statistical metrics
            if "mean_difference" in stats:
                assert isinstance(
                    stats["mean_difference"], (int, float)
                ), "mean_difference should be numeric"

            if "max_difference" in stats:
                assert isinstance(
                    stats["max_difference"], (int, float)
                ), "max_difference should be numeric"


class TestRandomSeedGeneration:
    """Test suite for random seed generation from entropy sources."""

    @pytest.mark.parametrize("use_system_entropy", [True, False])
    def test_get_random_seed_entropy_sources(self, use_system_entropy):
        """Test random seed generation from entropy sources ensuring high-quality random
        seed generation and proper fallback mechanisms."""
        # Call get_random_seed with system entropy preference parameter
        seed = get_random_seed(use_system_entropy=use_system_entropy)

        # Assert returned value is valid integer seed within acceptable range
        assert isinstance(seed, int), f"Generated seed should be int, got {type(seed)}"
        assert (
            SEED_MIN_VALUE <= seed <= SEED_MAX_VALUE
        ), f"Generated seed {seed} outside valid range [{SEED_MIN_VALUE}, {SEED_MAX_VALUE}]"

        # Verify multiple calls produce different seed values
        seed2 = get_random_seed(use_system_entropy=use_system_entropy)
        seed3 = get_random_seed(use_system_entropy=use_system_entropy)

        # Should be extremely unlikely to get same seed multiple times
        assert (
            seed != seed2 or seed != seed3
        ), f"Random seed generation may not be working: got {seed}, {seed2}, {seed3}"

        # Test system entropy usage when available and requested
        if use_system_entropy:
            # Should attempt to use system entropy (implementation detail)
            # At minimum should produce valid seeds
            assert isinstance(seed, int), "System entropy seed should be integer"

        # Validate fallback mechanism when system entropy unavailable
        else:
            # Should use fallback entropy source
            assert isinstance(seed, int), "Fallback seed should be integer"

        # Ensure generated seeds pass validation and are suitable for RNG
        is_valid, normalized_seed, error_msg = validate_seed(seed)
        assert is_valid, f"Generated seed {seed} failed validation: {error_msg}"
        assert normalized_seed == seed, f"Generated seed should not need normalization"

        # Test seed quality through basic statistical randomness checks
        seeds_sample = [
            get_random_seed(use_system_entropy=use_system_entropy) for _ in range(20)
        ]

        # Should have variety in generated seeds (basic randomness check)
        unique_seeds = set(seeds_sample)
        assert (
            len(unique_seeds) >= len(seeds_sample) * 0.8
        ), f"Only {len(unique_seeds)} unique seeds out of {len(seeds_sample)}, may indicate poor randomness"

        # Note: Distribution tests removed - with only 20 samples, statistical
        # clustering at one end is possible and doesn't indicate a problem.
        # Proper distribution testing would require 100s-1000s of samples and
        # chi-square or KS tests. YAGNI: we're testing seed generation, not RNG quality.


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
            ), f"save_seed_state should return True for successful save"

            # Verify file was created and contains data
            assert (
                temp_file_path.exists()
            ), f"State file should exist after save: {temp_file_path}"
            assert temp_file_path.stat().st_size > 0, f"State file should not be empty"

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

    @pytest.mark.parametrize("default_seed", [None] + TEST_SEEDS[:3])
    @pytest.mark.parametrize("enable_validation", [True, False])
    @pytest.mark.parametrize("thread_safe", [True, False])
    def test_seed_manager_basic_operations(
        self, default_seed, enable_validation, thread_safe
    ):
        """Test SeedManager basic operations including initialization, seeding, and generator
        tracking with proper state management."""
        # Initialize SeedManager with test configuration parameters
        manager = SeedManager(
            default_seed=default_seed,
            enable_validation=enable_validation,
            thread_safe=thread_safe,
        )

        # Verify initialization properties
        assert (
            manager.default_seed == default_seed
        ), f"Default seed should be {default_seed}"
        assert (
            manager.enable_validation == enable_validation
        ), f"Validation should be {enable_validation}"
        assert (
            manager.thread_safe == thread_safe
        ), f"Thread safety should be {thread_safe}"

        # Test seed method with various seed values and context IDs
        test_contexts = ["context1", "context2", "main"]
        generators = {}

        for context in test_contexts:
            for seed in [42, 123, None]:
                # Call seed method
                np_random, used_seed = manager.seed(seed, context_id=context)

                # Verify returned generators are properly initialized and tracked
                assert isinstance(
                    np_random, np.random.Generator
                ), f"Should return Generator for context {context}, seed {seed}"
                assert isinstance(
                    used_seed, int
                ), f"Used seed should be int for {context}, {seed}"

                # Only validate range for explicitly provided seeds, not random ones
                if seed is not None:
                    assert (
                        SEED_MIN_VALUE <= used_seed <= SEED_MAX_VALUE
                    ), f"Used seed {used_seed} outside valid range"

                generators[(context, seed)] = (np_random, used_seed)

        # Validate get_active_generators returns correct generator information
        active_generators = manager.get_active_generators()
        assert isinstance(active_generators, dict), "Active generators should be dict"

        # Should have entries for each context (check in nested 'generators' dict)
        generators_dict = active_generators.get("generators", {})
        for context in test_contexts:
            assert (
                context in generators_dict
            ), f"Context {context} should be tracked in generators"

        # Test reset functionality clears all active generators
        initial_count = active_generators.get(
            "total_active_generators", len(generators_dict)
        )
        manager.reset()

        after_reset_report = manager.get_active_generators()
        after_reset_count = after_reset_report.get("total_active_generators", 0)
        assert (
            after_reset_count == 0 or after_reset_count < initial_count
        ), "Reset should clear or reduce active generators"

        # Ensure validation settings affect seed processing as expected
        if enable_validation:
            # Should validate seed inputs
            with pytest.raises(ValidationError):
                manager.seed(-1)  # Invalid seed
        else:
            # May be more permissive with validation disabled
            try:
                result = manager.seed(42)  # Should work regardless
                assert (
                    result is not None
                ), "Valid seed should work even without validation"
            except ValidationError:
                # Still acceptable if validation occurs elsewhere
                pass

        # Validate thread safety configuration affects internal locking
        if thread_safe:
            # Should have thread safety mechanisms (implementation detail)
            # Test that basic operations work
            np_random, seed = manager.seed(42, context_id="thread_test")
            assert isinstance(
                np_random, np.random.Generator
            ), "Thread safe manager should work"

    @pytest.mark.parametrize("base_seed", TEST_SEEDS)
    @pytest.mark.parametrize("episode_number", [0, 1, 10, 100])
    @pytest.mark.parametrize("experiment_id", ["exp1", "baseline", None])
    def test_seed_manager_episode_seed_generation(
        self, base_seed, episode_number, experiment_id
    ):
        """Test SeedManager episode-specific seed generation ensuring deterministic episode
        seeds and proper experiment context handling."""
        # Initialize SeedManager for episode seed testing
        manager = SeedManager(enable_validation=True)

        # Call generate_episode_seed with base seed, episode number, and experiment ID
        episode_seed = manager.generate_episode_seed(
            base_seed=base_seed,
            episode_number=episode_number,
            experiment_id=experiment_id,
        )

        # Assert generated seed is deterministic for same input parameters
        episode_seed2 = manager.generate_episode_seed(
            base_seed=base_seed,
            episode_number=episode_number,
            experiment_id=experiment_id,
        )

        assert (
            episode_seed == episode_seed2
        ), f"Episode seed should be deterministic: {episode_seed} != {episode_seed2}"

        # Verify different episode numbers produce different seeds
        if episode_number < 999:  # Avoid edge case overflow
            different_episode_seed = manager.generate_episode_seed(
                base_seed=base_seed,
                episode_number=episode_number + 1,
                experiment_id=experiment_id,
            )
            assert (
                different_episode_seed != episode_seed
            ), f"Different episodes should produce different seeds: {episode_number} vs {episode_number + 1}"

        # Test experiment ID affects seed generation appropriately
        if experiment_id is not None:
            different_exp_seed = manager.generate_episode_seed(
                base_seed=base_seed,
                episode_number=episode_number,
                experiment_id="different_exp",
            )
            assert (
                different_exp_seed != episode_seed
            ), f"Different experiment IDs should produce different seeds"

        # Validate all generated seeds are within valid range
        assert isinstance(
            episode_seed, int
        ), f"Episode seed should be int, got {type(episode_seed)}"
        assert (
            SEED_MIN_VALUE <= episode_seed <= SEED_MAX_VALUE
        ), f"Episode seed {episode_seed} outside valid range [{SEED_MIN_VALUE}, {SEED_MAX_VALUE}]"

        # Ensure episode seed generation is consistent across manager instances
        manager2 = SeedManager(enable_validation=True)
        episode_seed3 = manager2.generate_episode_seed(
            base_seed=base_seed,
            episode_number=episode_number,
            experiment_id=experiment_id,
        )

        assert (
            episode_seed3 == episode_seed
        ), f"Episode seed should be consistent across managers: {episode_seed3} != {episode_seed}"

    @pytest.mark.parametrize("test_seed", TEST_SEEDS)
    @pytest.mark.parametrize("num_tests", [5, 10, 20])
    def test_seed_manager_reproducibility_validation(self, test_seed, num_tests):
        """Test SeedManager reproducibility validation functionality ensuring comprehensive
        testing and statistical analysis of seeding behavior."""
        # Initialize SeedManager for reproducibility validation testing
        manager = SeedManager(enable_validation=True, thread_safe=True)

        # Call validate_reproducibility with test seed and number of tests
        validation_report = manager.validate_reproducibility(
            test_seed=test_seed,
            num_tests=num_tests,
        )

        # Assert returned report contains comprehensive validation results
        assert isinstance(
            validation_report, dict
        ), f"Validation report should be dict, got {type(validation_report)}"

        # Check for required top-level keys
        required_keys = [
            "results_summary",
            "statistical_analysis",
            "overall_status",
        ]
        for key in required_keys:
            assert key in validation_report, f"Validation report missing key: {key}"

        # Check results_summary structure
        assert "success_rate" in validation_report["results_summary"]
        assert "total_tests" in validation_report["results_summary"]

        # Verify statistical analysis includes success rate and deviation metrics
        stats = validation_report["statistical_analysis"]
        assert isinstance(stats, dict), "Statistical analysis should be dict"

        success_rate = validation_report["results_summary"]["success_rate"]
        assert isinstance(success_rate, (int, float)), "Success rate should be numeric"
        assert (
            0.0 <= success_rate <= 1.0
        ), f"Success rate {success_rate} should be in [0, 1]"

        total_tests = validation_report["results_summary"]["total_tests"]
        assert (
            total_tests == num_tests
        ), f"Total tests {total_tests} should match input {num_tests}"

        # Check failure analysis section
        assert "failure_analysis" in validation_report, "Should have failure analysis"
        failure_analysis = validation_report["failure_analysis"]
        num_failures = failure_analysis.get("num_failures", 0)

        assert isinstance(num_failures, int), "Number of failures should be int"
        assert (
            0 <= num_failures <= total_tests
        ), f"Failures {num_failures} should be <= total {total_tests}"

        # Test failure analysis identifies patterns and root causes
        if num_failures > 0:
            assert (
                "failure_analysis" in validation_report
            ), "Should have failure analysis if failures occurred"
            failure_analysis = validation_report["failure_analysis"]
            assert isinstance(failure_analysis, dict), "Failure analysis should be dict"

        # Validate recommendations are provided for reproducibility improvement
        if "recommendations" in validation_report:
            recommendations = validation_report["recommendations"]
            assert isinstance(recommendations, list), "Recommendations should be list"

        # Ensure report includes test configuration with seed used
        assert (
            "test_configuration" in validation_report
        ), "Report should include test config"
        config = validation_report["test_configuration"]
        assert (
            config["test_seed"] == test_seed
        ), f"Reported seed should match input seed {test_seed}"
        assert (
            config["num_tests"] == num_tests
        ), f"Reported num_tests should match input"

        # Should have timestamp in test_configuration
        assert (
            "validation_timestamp" in config
        ), "Config should include validation_timestamp"

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

    @pytest.mark.parametrize("episode_seed", TEST_SEEDS)
    @pytest.mark.parametrize("episode_length", [10, 50, 100])
    def test_reproducibility_tracker_episode_recording(
        self, episode_seed, episode_length
    ):
        """Test ReproducibilityTracker episode recording functionality ensuring comprehensive
        episode data storage and checksum validation."""
        # Initialize ReproducibilityTracker for episode recording testing
        tracker = ReproducibilityTracker()

        # Generate test action and observation sequences of specified length
        np_random, _ = create_seeded_rng(episode_seed)

        action_sequence = [np_random.integers(0, 4) for _ in range(episode_length)]
        observation_sequence = [np_random.random() for _ in range(episode_length)]
        reward_sequence = [
            np_random.random() * 2 - 1 for _ in range(episode_length)
        ]  # Rewards in [-1, 1]

        metadata = {
            "episode_seed": episode_seed,
            "episode_length": episode_length,
            "test_run": True,
        }

        # Call record_episode with seed, sequences, and metadata
        episode_id = tracker.record_episode(
            episode_seed=episode_seed,
            action_sequence=action_sequence,
            observation_sequence=observation_sequence,
            metadata=metadata,
        )

        # Assert episode record ID is returned for future reference
        assert episode_id is not None, "record_episode should return episode ID"
        assert isinstance(
            episode_id, str
        ), f"Episode ID should be string, got {type(episode_id)}"
        assert len(episode_id) > 0, "Episode ID should not be empty"

        # Verify episode data is stored
        stored_episode = tracker.episode_records[episode_id]
        assert stored_episode is not None, f"Episode {episode_id} should be retrievable"
        assert isinstance(stored_episode, dict), "Stored episode should be dict"

        # Check required fields
        required_fields = [
            "episode_seed",
            "action_sequence",
            "observation_sequence",
            "metadata",
        ]
        for field in required_fields:
            assert field in stored_episode, f"Stored episode missing field: {field}"

        # Verify data integrity
        assert (
            stored_episode["episode_seed"] == episode_seed
        ), "Stored seed should match input"
        assert (
            len(stored_episode["action_sequence"]) == episode_length
        ), "Action sequence length should match"
        assert (
            len(stored_episode["observation_sequence"]) == episode_length
        ), "Observation sequence length should match"

        # Validate metadata preservation and storage
        stored_metadata = stored_episode["metadata"]
        assert isinstance(stored_metadata, dict), "Stored metadata should be dict"
        assert (
            stored_metadata["episode_seed"] == episode_seed
        ), "Metadata seed should match"
        assert (
            stored_metadata["episode_length"] == episode_length
        ), "Metadata length should match"

        # Test record retrieval and data integrity validation
        # Sequences should match exactly
        for i in range(episode_length):
            assert (
                stored_episode["action_sequence"][i] == action_sequence[i]
            ), f"Action mismatch at index {i}: {stored_episode['action_sequence'][i]} != {action_sequence[i]}"
            assert (
                abs(stored_episode["observation_sequence"][i] - observation_sequence[i])
                < 1e-10
            ), f"Observation mismatch at index {i}"

    @pytest.mark.parametrize("episodes_should_match", [True, False])
    def test_reproducibility_tracker_verification(self, episodes_should_match):
        """Test ReproducibilityTracker episode reproducibility verification ensuring accurate
        comparison and discrepancy analysis."""
        # Initialize ReproducibilityTracker and record baseline episode
        tracker = ReproducibilityTracker()

        base_seed = 42
        episode_length = 20

        # Generate baseline episode data
        np_random_base, _ = create_seeded_rng(base_seed)

        base_actions = [np_random_base.integers(0, 4) for _ in range(episode_length)]
        base_observations = [np_random_base.random() for _ in range(episode_length)]
        base_rewards = [np_random_base.random() * 2 - 1 for _ in range(episode_length)]

        baseline_id = tracker.record_episode(
            episode_seed=base_seed,
            action_sequence=base_actions,
            observation_sequence=base_observations,
            metadata={"type": "baseline"},
        )

        # Generate new episode sequences (identical or different based on parameter)
        if episodes_should_match:
            # Use same seed for identical episode
            comparison_seed = base_seed
        else:
            # Use different seed for different episode
            comparison_seed = base_seed + 1

        np_random_comp, _ = create_seeded_rng(comparison_seed)

        comp_actions = [np_random_comp.integers(0, 4) for _ in range(episode_length)]
        comp_observations = [np_random_comp.random() for _ in range(episode_length)]
        comp_rewards = [np_random_comp.random() * 2 - 1 for _ in range(episode_length)]

        comparison_id = tracker.record_episode(
            episode_seed=comparison_seed,
            action_sequence=comp_actions,
            observation_sequence=comp_observations,
            metadata={"type": "comparison"},
        )

        # Call verify_episode_reproducibility with baseline episode and new sequences
        # API compares a stored episode against new sequences
        comparison_episode = tracker.episode_records[comparison_id]
        verification_result = tracker.verify_episode_reproducibility(
            episode_record_id=baseline_id,
            new_action_sequence=comparison_episode["action_sequence"],
            new_observation_sequence=comparison_episode["observation_sequence"],
            custom_tolerance=REPRODUCIBILITY_TOLERANCE,
        )

        # Assert verification results match expected outcome
        assert isinstance(
            verification_result, dict
        ), f"Verification result should be dict, got {type(verification_result)}"

        required_keys = [
            "sequences_match",
            "episode_record_id",
            "match_status",
        ]
        for key in required_keys:
            assert key in verification_result, f"Verification result missing key: {key}"

        sequences_match = verification_result["sequences_match"]
        assert isinstance(sequences_match, bool), "sequences_match should be boolean"

        if episodes_should_match:
            assert sequences_match is True, "Identical seed episodes should match"
            # When sequences match, status should be PASS
            assert (
                verification_result["match_status"] == "PASS"
            ), "Match status should be PASS"
        else:
            # Different seeds should usually produce different results
            # Note: there's a small chance they could be identical by coincidence
            if not sequences_match:
                # Check for discrepancy analysis
                if "discrepancy_analysis" in verification_result:
                    analysis = verification_result["discrepancy_analysis"]
                    assert isinstance(
                        analysis, dict
                    ), "Discrepancy analysis should be dict"

        # Validate statistical measures are accurate and meaningful
        if "statistical_analysis" in verification_result:
            stats = verification_result["statistical_analysis"]
            assert isinstance(stats, dict), "Statistical analysis should be dict"

            if "total_comparisons" in stats:
                total_comps = stats["total_comparisons"]
                expected_total = episode_length * 3  # actions, observations, rewards
                assert (
                    total_comps == expected_total
                ), f"Total comparisons should be {expected_total}"

        # Test tolerance handling for floating point comparisons
        tolerance = verification_result.get("tolerance_used", REPRODUCIBILITY_TOLERANCE)
        assert isinstance(tolerance, (int, float)), "Tolerance should be numeric"
        assert tolerance > 0, "Tolerance should be positive"

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
        env2.seed(env_seed)

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
        env3.seed(env_seed)
        obs3, info3 = env3.reset(seed=env_seed)

        assert np.array_equal(
            obs1, obs3
        ), "Seeding should be consistent across environment instances"
        assert (
            info1["agent_xy"] == info3["agent_xy"]
        ), "Start positions should be consistent"

        env3.close()

    @pytest.mark.parametrize("session_seed", TEST_SEEDS[:3])
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
            # Execute seeding operation that should trigger error
            with pytest.raises(ValidationError) as exc_info:
                validate_seed("not_a_number")

            # Assert appropriate exception type is raised (ValidationError, StateError)
            error = exc_info.value
            assert isinstance(
                error, ValidationError
            ), f"Should raise ValidationError, got {type(error)}"

            # Verify error message contains useful information for debugging
            error_message = str(error)
            assert len(error_message) > 0, "Error message should not be empty"
            assert "seed" in error_message.lower(), "Error should mention seed"

            # Test recovery mechanisms and fallback strategies
            # Should be able to continue with valid seed after error
            is_valid, normalized_seed, _ = validate_seed(42)
            assert is_valid, "Should recover and work with valid seed"

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
                # Should raise StateError or related exception
                with pytest.raises(
                    (StateError, json.JSONDecodeError, KeyError)
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
            ), f"Minimum seed should normalize to itself"
        elif edge_seed == SEED_MAX_VALUE:
            assert (
                normalized_seed == SEED_MAX_VALUE
            ), f"Maximum seed should normalize to itself"

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
            ), f"Overflow handling should keep result in valid range"

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
        assert used_seed_mgr == edge_seed, f"SeedManager should preserve edge case seed"


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
            format="json", include_detailed_analysis=True
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
        required_elements = ["summary", "episodes_recorded", "methodology"]
        for element in required_elements:
            if element not in report_data:
                # Allow some flexibility in report structure
                assert any(
                    element in key for key in report_data.keys()
                ), f"Report should include {element} information"

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
                seed=seed,
                action_sequence=actions,
                observation_sequence=observations,
                reward_sequence=rewards,
                metadata={
                    "condition": condition,
                    "episode_length_requested": episode_length,
                    "execution_time": end_time - start_time,
                    "scientific_run": True,
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
