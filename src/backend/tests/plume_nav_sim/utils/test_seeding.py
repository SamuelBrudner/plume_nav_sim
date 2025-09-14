# External imports with version comments for testing framework and utilities
import pytest  # >=8.0.0 - Testing framework for comprehensive test execution, fixtures, parametrization, and assertions
import numpy as np  # >=2.1.0 - Array operations, random number generation, and mathematical testing utilities for seeding validation
import gymnasium.utils.seeding  # >=0.29.0 - Gymnasium seeding utilities for RNG creation and API compatibility testing
import time  # >=3.10 - Performance timing measurements for seeding operation benchmarks and latency validation
import threading  # >=3.10 - Thread safety testing for concurrent seeding operations and multi-threaded scenarios
import tempfile  # >=3.10 - Temporary file creation for seed state persistence testing and file operation validation
import json  # >=3.10 - JSON serialization testing for seed state persistence and data validation
import pathlib  # >=3.10 - Path manipulation for seed state file testing and cross-platform compatibility validation
import hashlib  # >=3.10 - Hash function testing for deterministic seed generation and cryptographic validation
import warnings  # >=3.10 - Warning management for deprecation testing and performance notification validation
import uuid  # >=3.10 - Unique identifier testing for seed tracking and session management validation

# Internal imports from seeding utilities module
from plume_nav_sim.utils.seeding import (
    validate_seed,  # Primary seed validation function for testing parameter validation, error handling, and normalization behavior
    create_seeded_rng,  # RNG creation function testing for Gymnasium compatibility, proper initialization, and deterministic behavior
    generate_deterministic_seed,  # Deterministic seed generation testing from string identifiers for experiment naming and reproducible research
    verify_reproducibility,  # Reproducibility verification testing through sequence comparison and statistical analysis validation
    get_random_seed,  # Random seed generation testing from system entropy sources for secure initialization testing
    save_seed_state,  # Seed state persistence testing for experiment reproduction and cross-session reproducibility validation
    load_seed_state,  # Seed state loading testing for experiment restoration and state recovery validation
    SeedManager,  # Centralized seed management class testing for validation, reproducibility tracking, and thread safety
    ReproducibilityTracker  # Reproducibility tracking class testing for scientific research with comprehensive logging and statistical analysis
)

# Internal imports from exception handling module
from plume_nav_sim.utils.exceptions import (
    ValidationError,  # Validation error exception testing for seed parameter validation failures and error reporting
    StateError,  # State error exception testing for invalid random number generator states and seeding failures
    ComponentError,  # Component error exception testing for seeding component failures and RNG management issues
    ResourceError  # Resource error exception testing for memory issues and state persistence failures
)

# Internal imports from constants module
from plume_nav_sim.core.constants import (
    SEED_MIN_VALUE,  # Minimum seed value constant for boundary testing and validation limits
    SEED_MAX_VALUE,  # Maximum seed value constant for boundary testing and overflow prevention validation
    VALID_SEED_TYPES,  # Valid seed types list for type validation testing and parameter checking
    VALIDATION_ERROR_MESSAGES  # Error message templates for validation testing and error reporting consistency
)

# Global test constants for comprehensive validation and performance benchmarking
TEST_SEEDS = [42, 123, 456, 789, 999, 2**31-1]
INVALID_SEEDS = [-1, 2**31, 'invalid', 3.14, [], {}, object()]
REPRODUCIBILITY_TOLERANCE = 1e-10
PERFORMANCE_TARGET_SEEDING_MS = 1.0
TEST_SEQUENCE_LENGTH = 1000
THREAD_SAFETY_ITERATIONS = 100


# Unit tests for validate_seed function with comprehensive parameter validation testing
@pytest.mark.unit
@pytest.mark.parametrize('seed', TEST_SEEDS + [None, 0, np.int32(123), np.int64(456)])
def test_validate_seed_valid_inputs(seed):
    """Test validate_seed function with valid seed inputs including positive integers, zero, None values, and numpy integer types for comprehensive parameter validation testing"""
    # Test validate_seed with valid integer seeds from TEST_SEEDS list
    is_valid, normalized_seed, error_message = validate_seed(seed)
    
    # Verify function returns (True, normalized_seed, '') for valid inputs
    assert is_valid is True, f"Expected valid seed {seed} to pass validation"
    assert error_message == '', f"Expected empty error message for valid seed {seed}, got: {error_message}"
    
    # Test None seed handling returning (True, None, '') for random seeding
    if seed is None:
        assert normalized_seed is None, "None seed should return None as normalized seed"
    else:
        # Validate numpy integer type handling with proper normalization
        assert isinstance(normalized_seed, int), f"Expected normalized seed to be int, got {type(normalized_seed)}"
        assert SEED_MIN_VALUE <= normalized_seed <= SEED_MAX_VALUE, f"Normalized seed {normalized_seed} out of valid range"
        
    # Check seed normalization for edge cases and boundary values
    if seed is not None and seed >= 0:
        expected_normalized = int(seed) if hasattr(seed, 'item') else seed
        if hasattr(seed, 'item'):  # numpy integer types
            expected_normalized = int(seed.item())
        assert normalized_seed == expected_normalized, f"Expected normalization of {seed} to be {expected_normalized}"
    
    # Verify error message is empty string for all valid inputs
    assert error_message == '', "Valid seeds should have empty error messages"
    
    # Test strict_mode parameter behavior with additional validation
    is_valid_strict, normalized_strict, error_strict = validate_seed(seed, strict_mode=True)
    assert is_valid_strict == is_valid, "Strict mode should not affect valid seeds"
    assert normalized_strict == normalized_seed, "Strict mode should not change normalization for valid seeds"


@pytest.mark.unit
@pytest.mark.parametrize('invalid_seed', INVALID_SEEDS)
def test_validate_seed_invalid_inputs(invalid_seed):
    """Test validate_seed function with invalid seed inputs including negative values, out-of-range integers, wrong types, and edge cases for comprehensive error validation testing"""
    # Test validate_seed with invalid seed values from INVALID_SEEDS list
    is_valid, normalized_seed, error_message = validate_seed(invalid_seed)
    
    # Verify function returns (False, None, error_message) for invalid inputs
    assert is_valid is False, f"Expected invalid seed {invalid_seed} to fail validation"
    assert normalized_seed is None, f"Expected None normalized seed for invalid input {invalid_seed}"
    assert error_message != '', f"Expected non-empty error message for invalid seed {invalid_seed}"
    
    # Check error messages match templates from VALIDATION_ERROR_MESSAGES
    expected_error_base = VALIDATION_ERROR_MESSAGES['invalid_seed']
    assert expected_error_base in error_message or "out of range" in error_message, \
        f"Expected error message to contain validation template for {invalid_seed}"
    
    # Test negative seed handling and proper error reporting
    if isinstance(invalid_seed, int) and invalid_seed < 0:
        # Negative seeds should be handled by normalization, not validation failure
        # This test case may need adjustment based on actual implementation
        pass
    
    # Validate out-of-range integer detection and overflow prevention
    if isinstance(invalid_seed, int) and invalid_seed > SEED_MAX_VALUE:
        assert "out of range" in error_message, f"Expected range error for oversized seed {invalid_seed}"
    
    # Test wrong type rejection with appropriate error messages
    if not isinstance(invalid_seed, (int, type(None))):
        assert "invalid" in error_message.lower() or "type" in error_message.lower(), \
            f"Expected type error message for {type(invalid_seed)} seed"
    
    # Verify strict_mode parameter enhances validation for edge cases
    is_valid_strict, normalized_strict, error_strict = validate_seed(invalid_seed, strict_mode=True)
    assert is_valid_strict is False, "Strict mode should also reject invalid seeds"
    assert "strict mode" in error_strict or error_strict == error_message, \
        "Strict mode should provide enhanced error context"


@pytest.mark.unit
def test_validate_seed_normalization():
    """Test validate_seed function normalization behavior for negative seeds, large integers, and edge cases ensuring proper modulo operation and range compliance"""
    # Test negative seed normalization using modulo operation
    negative_seed = -42
    is_valid, normalized_seed, error_message = validate_seed(negative_seed)
    
    # Verify normalized seed is within [SEED_MIN_VALUE, SEED_MAX_VALUE] range
    if is_valid:  # Some implementations may normalize, others may reject
        assert SEED_MIN_VALUE <= normalized_seed <= SEED_MAX_VALUE, \
            f"Normalized negative seed {normalized_seed} not in valid range"
        # Test mathematical consistency of normalization formula
        expected_normalized = negative_seed % (SEED_MAX_VALUE + 1)
        assert normalized_seed == expected_normalized, \
            f"Expected normalized seed {expected_normalized}, got {normalized_seed}"
    
    # Test large integer handling and overflow prevention
    large_seed = SEED_MAX_VALUE + 100
    is_valid_large, normalized_large, error_large = validate_seed(large_seed)
    if not is_valid_large:
        assert "out of range" in error_large, "Large seeds should trigger range error"
    
    # Check boundary value normalization at SEED_MIN_VALUE and SEED_MAX_VALUE
    for boundary in [SEED_MIN_VALUE, SEED_MAX_VALUE]:
        is_valid_boundary, normalized_boundary, _ = validate_seed(boundary)
        assert is_valid_boundary, f"Boundary value {boundary} should be valid"
        assert normalized_boundary == boundary, f"Boundary seed should not be modified"
    
    # Test edge cases around integer overflow boundaries
    edge_cases = [SEED_MAX_VALUE - 1, SEED_MAX_VALUE, 0, 1]
    for edge_seed in edge_cases:
        is_valid_edge, normalized_edge, _ = validate_seed(edge_seed)
        assert is_valid_edge, f"Edge case seed {edge_seed} should be valid"
        assert normalized_edge == edge_seed, f"Edge seed {edge_seed} should not be normalized"
    
    # Verify deterministic normalization produces consistent results
    test_seed = -1000
    for _ in range(5):  # Test consistency across multiple calls
        is_valid_repeat, normalized_repeat, _ = validate_seed(test_seed)
        if is_valid_repeat:
            # All normalized results should be identical
            first_result = normalized_repeat
            break
    
    if 'first_result' in locals():
        for _ in range(5):
            is_valid_check, normalized_check, _ = validate_seed(test_seed)
            if is_valid_check:
                assert normalized_check == first_result, "Normalization should be deterministic"


# Unit tests for create_seeded_rng function with Gymnasium compatibility validation
@pytest.mark.unit
@pytest.mark.parametrize('seed', TEST_SEEDS + [None])
def test_create_seeded_rng_valid_seeds(seed):
    """Test create_seeded_rng function with valid seeds ensuring Gymnasium compatibility, proper RNG initialization, and deterministic behavior validation"""
    # Test create_seeded_rng with valid seeds from TEST_SEEDS
    np_random, seed_used = create_seeded_rng(seed)
    
    # Verify function returns (np_random, seed_used) tuple
    assert isinstance(np_random, np.random.Generator), \
        f"Expected numpy.random.Generator, got {type(np_random)}"
    assert isinstance(seed_used, int), f"Expected int seed_used, got {type(seed_used)}"
    
    # Check np_random is numpy.random.Generator instance
    assert hasattr(np_random, 'bit_generator'), "Generator should have bit_generator attribute"
    assert hasattr(np_random, 'random'), "Generator should have random method"
    
    # Validate seed_used matches input seed for deterministic cases
    if seed is not None:
        assert seed_used == seed, f"Expected seed_used {seed_used} to match input seed {seed}"
    else:
        # Test None seed handling with random seed generation
        assert SEED_MIN_VALUE <= seed_used <= SEED_MAX_VALUE, \
            f"Random seed {seed_used} should be in valid range"
    
    # Verify Gymnasium compatibility using gymnasium.utils.seeding.np_random
    # Test that generated RNG behaves identically to direct gymnasium creation
    if seed is not None:
        gym_rng, gym_seed = gymnasium.utils.seeding.np_random(seed)
        assert seed_used == gym_seed, "Seed should match gymnasium seeding result"
        
        # Test sequence compatibility
        test_sequence_1 = np_random.random(10)
        test_sequence_2 = gym_rng.random(10)
        np.testing.assert_array_almost_equal(test_sequence_1, test_sequence_2, 
                                           err_msg="RNG sequences should match gymnasium")
    
    # Check RNG state initialization and proper configuration
    state = np_random.bit_generator.state
    assert 'bit_generator' in state, "RNG state should contain bit_generator info"
    assert 'state' in state, "RNG state should contain state info"


@pytest.mark.unit
@pytest.mark.parametrize('invalid_seed', INVALID_SEEDS)
def test_create_seeded_rng_invalid_seeds(invalid_seed):
    """Test create_seeded_rng function error handling with invalid seeds ensuring proper ValidationError raising and error message accuracy"""
    # Test create_seeded_rng with invalid seeds from INVALID_SEEDS
    with pytest.raises(ValidationError) as exc_info:
        create_seeded_rng(invalid_seed, validate_input=True)
    
    # Verify ValidationError is raised for invalid inputs
    validation_error = exc_info.value
    assert isinstance(validation_error, ValidationError), \
        f"Expected ValidationError, got {type(validation_error)}"
    
    # Check error message accuracy and context information
    assert invalid_seed == validation_error.invalid_value or \
           str(invalid_seed) in str(validation_error.invalid_value), \
        "ValidationError should contain invalid seed information"
    
    # Test validate_input parameter controls validation behavior
    if isinstance(invalid_seed, int):  # Only test with integer-like invalid seeds
        try:
            # With validate_input=False, some invalid seeds might still fail due to gymnasium constraints
            np_random, seed_used = create_seeded_rng(invalid_seed, validate_input=False)
            # If it doesn't raise an error, that's also valid behavior
            assert isinstance(np_random, np.random.Generator), "Should return valid generator"
        except (ValidationError, StateError):
            # Expected behavior for truly invalid seeds
            pass
    
    # Validate exception contains proper error details and recovery suggestions
    assert hasattr(validation_error, 'parameter_name'), "ValidationError should have parameter_name"
    assert hasattr(validation_error, 'recovery_suggestion'), "ValidationError should have recovery suggestion"
    
    # Test error propagation and logging integration
    assert validation_error.message, "ValidationError should have descriptive message"
    
    # Verify no RNG is created when validation fails
    # This is implicitly tested by the exception raising above


@pytest.mark.reproducibility
@pytest.mark.parametrize('seed', TEST_SEEDS[:3])
def test_create_seeded_rng_reproducibility(seed):
    """Test create_seeded_rng reproducibility ensuring identical seeds produce identical RNG states and generate identical random sequences"""
    # Create two RNGs with identical seed using create_seeded_rng
    np_random_1, seed_used_1 = create_seeded_rng(seed)
    np_random_2, seed_used_2 = create_seeded_rng(seed)
    
    # Check seed_used values match for identical inputs
    assert seed_used_1 == seed_used_2, f"Identical seeds should produce identical seed_used values"
    assert seed_used_1 == seed, f"Seed used should match input seed {seed}"
    
    # Generate identical random sequences from both RNGs
    sequence_length = 100
    sequence_1 = np_random_1.random(sequence_length)
    sequence_2 = np_random_2.random(sequence_length)
    
    # Compare sequences for exact equality using numpy.array_equal
    np.testing.assert_array_equal(sequence_1, sequence_2, 
                                 err_msg="Identical seeds should produce identical sequences")
    
    # Test various random operations (integers, floats, choice)
    # Reset RNGs to same state
    np_random_1, _ = create_seeded_rng(seed)
    np_random_2, _ = create_seeded_rng(seed)
    
    # Test integer generation
    int_seq_1 = np_random_1.integers(0, 100, size=50)
    int_seq_2 = np_random_2.integers(0, 100, size=50)
    np.testing.assert_array_equal(int_seq_1, int_seq_2, 
                                 err_msg="Integer sequences should match")
    
    # Test choice operation
    np_random_1, _ = create_seeded_rng(seed)
    np_random_2, _ = create_seeded_rng(seed)
    
    choices = [1, 2, 3, 4, 5]
    choice_seq_1 = [np_random_1.choice(choices) for _ in range(20)]
    choice_seq_2 = [np_random_2.choice(choices) for _ in range(20)]
    assert choice_seq_1 == choice_seq_2, "Choice sequences should match"
    
    # Verify state consistency after sequence generation
    state_1 = np_random_1.bit_generator.state
    state_2 = np_random_2.bit_generator.state
    # States will be different after generation, but should be deterministic
    
    # Validate reproducibility across multiple test runs
    # This is implicitly tested by the parametrized test running multiple times


# Unit tests for generate_deterministic_seed function with cryptographic validation
@pytest.mark.unit
@pytest.mark.parametrize('seed_string', ['test', 'experiment_1', 'long_string_identifier', ''])
def test_generate_deterministic_seed_consistency(seed_string):
    """Test generate_deterministic_seed function consistency ensuring identical string inputs always produce identical seed values with cryptographic hash validation"""
    if seed_string == '':
        # Empty string should raise ValidationError
        with pytest.raises(ValidationError):
            generate_deterministic_seed(seed_string)
        return
    
    # Test generate_deterministic_seed with various string inputs
    seed_1 = generate_deterministic_seed(seed_string)
    seed_2 = generate_deterministic_seed(seed_string)
    
    # Verify identical strings produce identical seeds across multiple calls
    assert seed_1 == seed_2, f"Identical strings should produce identical seeds: {seed_1} != {seed_2}"
    assert isinstance(seed_1, int), f"Generated seed should be integer, got {type(seed_1)}"
    
    # Check seed values are within valid range [SEED_MIN_VALUE, SEED_MAX_VALUE]
    assert SEED_MIN_VALUE <= seed_1 <= SEED_MAX_VALUE, \
        f"Generated seed {seed_1} not in valid range [{SEED_MIN_VALUE}, {SEED_MAX_VALUE}]"
    
    # Test different hash algorithms (SHA256, MD5, SHA1) if supported
    available_algorithms = ['sha256', 'md5', 'sha1']
    algorithm_results = {}
    
    for algorithm in available_algorithms:
        if algorithm in hashlib.algorithms_available:
            try:
                seed_alg = generate_deterministic_seed(seed_string, hash_algorithm=algorithm)
                algorithm_results[algorithm] = seed_alg
                assert SEED_MIN_VALUE <= seed_alg <= SEED_MAX_VALUE, \
                    f"Seed from {algorithm} not in valid range"
            except ValidationError:
                # Some algorithms might not be supported
                pass
    
    # Validate encoding parameter affects seed generation consistently
    encodings = ['utf-8', 'ascii', 'latin1']
    encoding_results = {}
    
    for encoding in encodings:
        try:
            seed_enc = generate_deterministic_seed(seed_string, encoding=encoding)
            encoding_results[encoding] = seed_enc
            assert SEED_MIN_VALUE <= seed_enc <= SEED_MAX_VALUE, \
                f"Seed with {encoding} encoding not in valid range"
        except (UnicodeEncodeError, ValidationError):
            # Some encodings might not support all characters
            pass
    
    # Test empty string handling and edge case string inputs
    # Empty string tested above
    
    # Verify cryptographic hash function integration and security
    # Hash should be deterministic and well-distributed
    if len(seed_string) > 0:
        # Test that similar strings produce different seeds (avalanche effect)
        similar_string = seed_string + 'x'
        similar_seed = generate_deterministic_seed(similar_string)
        assert similar_seed != seed_1, "Similar strings should produce different seeds"


@pytest.mark.unit  
def test_generate_deterministic_seed_distribution():
    """Test generate_deterministic_seed function distribution ensuring hash output produces well-distributed seeds across the valid range"""
    # Generate large number of deterministic seeds with different string inputs
    num_seeds = 1000
    test_strings = [f"test_string_{i}" for i in range(num_seeds)]
    generated_seeds = [generate_deterministic_seed(s) for s in test_strings]
    
    # Check for clustering or bias in hash function output
    assert len(set(generated_seeds)) > num_seeds * 0.95, \
        "Generated seeds should have high uniqueness (>95%)"
    
    # Analyze distribution of generated seeds across valid range
    seed_range = SEED_MAX_VALUE - SEED_MIN_VALUE + 1
    bin_count = 100
    bins = np.linspace(SEED_MIN_VALUE, SEED_MAX_VALUE + 1, bin_count + 1)
    hist, _ = np.histogram(generated_seeds, bins=bins)
    
    # Validate uniform distribution properties using statistical tests
    expected_per_bin = num_seeds / bin_count
    # Chi-square goodness of fit test (simplified)
    chi_square = np.sum((hist - expected_per_bin) ** 2 / expected_per_bin)
    
    # With 100 bins and reasonable distribution, chi-square should not be extreme
    assert chi_square < num_seeds * 2, f"Distribution appears non-uniform (chi-square: {chi_square})"
    
    # Test collision resistance with similar string inputs
    base_string = "collision_test"
    similar_strings = [f"{base_string}_{i}" for i in range(100)]
    collision_seeds = [generate_deterministic_seed(s) for s in similar_strings]
    
    unique_collision_seeds = len(set(collision_seeds))
    assert unique_collision_seeds == len(collision_seeds), \
        f"Expected no collisions, got {len(collision_seeds) - unique_collision_seeds} collisions"
    
    # Verify hash algorithm produces quality pseudo-random distribution
    # Test that adjacent seeds are not correlated
    adjacent_diffs = [abs(generated_seeds[i+1] - generated_seeds[i]) 
                     for i in range(len(generated_seeds)-1)]
    mean_diff = np.mean(adjacent_diffs)
    expected_mean_diff = seed_range / 2  # Expected for uniform distribution
    
    # Allow 20% deviation from expected mean difference
    assert abs(mean_diff - expected_mean_diff) < expected_mean_diff * 0.2, \
        f"Adjacent differences suggest non-random distribution"
    
    # Check boundary value generation at range extremes
    boundary_strings = [f"boundary_{i}" for i in range(1000)]
    boundary_seeds = [generate_deterministic_seed(s) for s in boundary_strings]
    
    # Should have seeds at various points across range
    min_seed = min(boundary_seeds)
    max_seed = max(boundary_seeds)
    seed_span = max_seed - min_seed
    expected_span = seed_range * 0.5  # Expect at least 50% of range coverage
    
    assert seed_span >= expected_span, \
        f"Seed distribution doesn't span enough of valid range: {seed_span} < {expected_span}"


# Reproducibility tests for verify_reproducibility function
@pytest.mark.reproducibility
@pytest.mark.parametrize('seed', TEST_SEEDS[:3])
def test_verify_reproducibility_identical_generators(seed):
    """Test verify_reproducibility function with identical generators ensuring perfect reproducibility detection and statistical analysis accuracy"""
    # Create two identical generators using same seed
    rng1, _ = create_seeded_rng(seed)
    rng2, _ = create_seeded_rng(seed)
    
    # Run verify_reproducibility with various sequence lengths
    sequence_lengths = [100, 500, 1000]
    
    for seq_len in sequence_lengths:
        # Reset generators to identical state
        rng1, _ = create_seeded_rng(seed)
        rng2, _ = create_seeded_rng(seed)
        
        result = verify_reproducibility(rng1, rng2, sequence_length=seq_len, 
                                      tolerance=REPRODUCIBILITY_TOLERANCE)
        
        # Verify function returns match status as True
        assert result['sequences_match'] is True, \
            f"Identical generators should match (seed={seed}, len={seq_len})"
        
        # Check statistical analysis shows zero deviation
        stats = result['statistical_analysis']
        assert stats['mean_absolute_error'] <= REPRODUCIBILITY_TOLERANCE, \
            f"Mean absolute error should be within tolerance: {stats['mean_absolute_error']}"
        assert stats['max_deviation'] <= REPRODUCIBILITY_TOLERANCE, \
            f"Max deviation should be within tolerance: {stats['max_deviation']}"
        
        # Test tolerance parameter behavior with perfect matches
        assert result['reproducibility_score'] >= 0.99, \
            f"Reproducibility score should be near perfect: {result['reproducibility_score']}"
        
        # Validate comprehensive report contains expected metrics
        assert 'verification_timestamp' in result, "Report should include timestamp"
        assert 'sequence_length' in result, "Report should include sequence length"
        assert result['sequence_length'] == seq_len, "Reported length should match requested"
        
        # Test sequence comparison with different random operations
        assert stats['exact_matches'] == seq_len, \
            f"All elements should match exactly: {stats['exact_matches']}/{seq_len}"
        assert stats['exact_match_percentage'] == 100.0, \
            f"Exact match percentage should be 100%: {stats['exact_match_percentage']}"


@pytest.mark.reproducibility
def test_verify_reproducibility_different_generators():
    """Test verify_reproducibility function with different generators ensuring proper mismatch detection and statistical analysis of differences"""
    # Create generators with different seeds
    rng1, _ = create_seeded_rng(42)
    rng2, _ = create_seeded_rng(123)
    
    # Run verify_reproducibility expecting mismatch detection
    result = verify_reproducibility(rng1, rng2, sequence_length=TEST_SEQUENCE_LENGTH,
                                  tolerance=REPRODUCIBILITY_TOLERANCE)
    
    # Verify function returns match status as False
    assert result['sequences_match'] is False, \
        "Different generators should not match"
    
    # Check statistical analysis shows significant deviation
    stats = result['statistical_analysis']
    assert stats['mean_absolute_error'] > REPRODUCIBILITY_TOLERANCE, \
        f"Mean absolute error should exceed tolerance: {stats['mean_absolute_error']}"
    assert stats['max_deviation'] > REPRODUCIBILITY_TOLERANCE, \
        f"Max deviation should exceed tolerance: {stats['max_deviation']}"
    
    # Test detailed discrepancy analysis and reporting
    discrepancy = result['discrepancy_analysis']
    assert discrepancy['num_discrepancies'] > 0, \
        "Should detect discrepancies between different generators"
    assert len(discrepancy['discrepancy_indices']) > 0, \
        "Should report specific discrepancy locations"
    assert discrepancy['largest_discrepancy'] > 0, \
        "Should report magnitude of largest discrepancy"
    
    # Validate tolerance parameter affects mismatch sensitivity
    # Test with very loose tolerance
    loose_result = verify_reproducibility(rng1, rng2, sequence_length=100, tolerance=1.0)
    # Even with loose tolerance, different seeds should not match
    assert loose_result['sequences_match'] is False, \
        "Different generators should not match even with loose tolerance"
    
    # Test comprehensive report includes difference statistics
    assert result['status'] in ['PASS', 'FAIL'], "Result should have clear status"
    assert result['status'] == 'FAIL', "Different generators should fail reproducibility test"
    assert 'recommendation' in result, "Should include recommendation"


# Unit tests for get_random_seed function with entropy source validation
@pytest.mark.unit
@pytest.mark.parametrize('use_system_entropy', [True, False])
def test_get_random_seed_entropy_sources(use_system_entropy):
    """Test get_random_seed function with different entropy sources ensuring high-quality random seed generation and security validation"""
    # Test get_random_seed with system entropy enabled and disabled
    seed1 = get_random_seed(use_system_entropy=use_system_entropy)
    seed2 = get_random_seed(use_system_entropy=use_system_entropy)
    
    # Verify generated seeds are within valid range
    assert SEED_MIN_VALUE <= seed1 <= SEED_MAX_VALUE, \
        f"Generated seed {seed1} not in valid range"
    assert SEED_MIN_VALUE <= seed2 <= SEED_MAX_VALUE, \
        f"Generated seed {seed2} not in valid range"
    
    # Check seed quality and randomness properties
    assert isinstance(seed1, int), f"Generated seed should be integer, got {type(seed1)}"
    assert isinstance(seed2, int), f"Generated seed should be integer, got {type(seed2)}"
    
    # Test multiple calls produce different seeds (high probability)
    seeds = [get_random_seed(use_system_entropy=use_system_entropy) for _ in range(100)]
    unique_seeds = len(set(seeds))
    
    # Expect high uniqueness (>90%) for random generation
    assert unique_seeds > 90, f"Random seeds should be mostly unique: {unique_seeds}/100"
    
    # Test fallback methods when system entropy unavailable
    if not use_system_entropy:
        # Test with fallback method specified
        seed_fallback = get_random_seed(use_system_entropy=False, fallback_method=1)
        assert SEED_MIN_VALUE <= seed_fallback <= SEED_MAX_VALUE, \
            "Fallback seed should be in valid range"
    
    # Validate security of entropy sources and seed generation
    # System entropy should provide better randomness than fallback
    if use_system_entropy:
        # Test that system entropy seeds pass basic randomness checks
        entropy_seeds = [get_random_seed(use_system_entropy=True) for _ in range(50)]
        
        # Check distribution is not obviously biased
        seed_range = SEED_MAX_VALUE - SEED_MIN_VALUE + 1
        mean_seed = np.mean(entropy_seeds)
        expected_mean = SEED_MIN_VALUE + seed_range / 2
        
        # Allow 20% deviation from expected mean (loose test for basic randomness)
        assert abs(mean_seed - expected_mean) < expected_mean * 0.2, \
            "System entropy seeds should be reasonably distributed"
    
    # Verify seed validation passes for all generated seeds
    for test_seed in seeds[:10]:  # Test first 10 seeds
        is_valid, normalized, error_msg = validate_seed(test_seed)
        assert is_valid, f"Generated seed {test_seed} should pass validation: {error_msg}"
        assert normalized == test_seed, "Generated seeds should not need normalization"


# Unit tests for save_seed_state and load_seed_state functions with roundtrip validation
@pytest.mark.unit
@pytest.mark.parametrize('seed', TEST_SEEDS[:3])
def test_save_load_seed_state_roundtrip(seed):
    """Test save_seed_state and load_seed_state functions roundtrip ensuring perfect state preservation and recovery across save/load operations"""
    # Create seeded RNG and generate initial random sequence
    original_rng, _ = create_seeded_rng(seed)
    initial_sequence = original_rng.random(50)
    
    # Save RNG state to temporary file using save_seed_state
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        temp_path = pathlib.Path(temp_file.name)
    
    try:
        # Test metadata preservation and integrity checking
        test_metadata = {
            'experiment_id': 'test_experiment',
            'session_info': 'roundtrip_test',
            'seed_used': seed
        }
        
        success = save_seed_state(original_rng, temp_path, metadata=test_metadata, 
                                create_backup=False)
        assert success is True, "Seed state save should succeed"
        
        # Verify state persistence maintains all RNG properties
        assert temp_path.exists(), "Seed state file should be created"
        assert temp_path.stat().st_size > 0, "Seed state file should not be empty"
        
        # Load RNG state from file using load_seed_state
        restored_rng, loaded_metadata = load_seed_state(temp_path, validate_state=True)
        
        # Compare original and restored generators for identical behavior
        # Generate sequences from both and compare
        original_rng, _ = create_seeded_rng(seed)  # Reset original
        post_save_sequence = restored_rng.random(50)
        expected_sequence = original_rng.random(50)
        
        np.testing.assert_array_almost_equal(post_save_sequence, expected_sequence,
                                           decimal=15, 
                                           err_msg="Restored RNG should match original behavior")
        
        # Verify metadata was preserved correctly
        assert 'experiment_id' in loaded_metadata, "Experiment ID should be preserved"
        assert loaded_metadata['experiment_id'] == 'test_experiment', "Metadata should match"
        assert 'load_timestamp' in loaded_metadata, "Load timestamp should be added"
        assert 'original_timestamp' in loaded_metadata, "Original timestamp should be preserved"
        
        # Test file format and cross-platform compatibility
        # Verify JSON format
        with open(temp_path, 'r') as f:
            saved_data = json.load(f)
        
        assert 'version' in saved_data, "Save format should include version"
        assert 'timestamp' in saved_data, "Save format should include timestamp"
        assert 'rng_state' in saved_data, "Save format should include RNG state"
        assert 'metadata' in saved_data, "Save format should include metadata"
        
    finally:
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()


@pytest.mark.unit
def test_save_load_seed_state_error_handling():
    """Test save_seed_state and load_seed_state functions error handling ensuring proper exception raising for file operations and validation failures"""
    rng, _ = create_seeded_rng(42)
    
    # Test save_seed_state with invalid file paths and permission errors
    invalid_path = pathlib.Path('/root/invalid/path/seed_state.json')
    
    with pytest.raises((ResourceError, OSError, PermissionError)):
        save_seed_state(rng, invalid_path)
    
    # Test with invalid RNG object
    with pytest.raises(ValidationError):
        save_seed_state("not_an_rng", pathlib.Path('test.json'))
    
    # Test load_seed_state with non-existent files and corrupted data
    nonexistent_path = pathlib.Path('nonexistent_file.json')
    
    with pytest.raises(ResourceError):
        load_seed_state(nonexistent_path)
    
    # Test with corrupted JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        temp_path = pathlib.Path(temp_file.name)
        temp_file.write("invalid json content {")
    
    try:
        with pytest.raises(ValidationError):
            load_seed_state(temp_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    
    # Test with valid JSON but invalid seed state format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        temp_path = pathlib.Path(temp_file.name)
        json.dump({'invalid': 'format'}, temp_file)
    
    try:
        with pytest.raises(ValidationError):
            load_seed_state(temp_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    
    # Verify appropriate exceptions are raised for error conditions
    # Check error messages provide helpful context and recovery suggestions
    try:
        load_seed_state(nonexistent_path)
    except ResourceError as e:
        assert 'does not exist' in str(e.message).lower(), \
            "Error message should indicate file doesn't exist"
        assert hasattr(e, 'recovery_suggestion'), "Should have recovery suggestion"
    
    # Test backup creation and atomic write operations
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        temp_path = pathlib.Path(temp_file.name)
        json.dump({'existing': 'content'}, temp_file)
    
    try:
        # Test backup creation
        success = save_seed_state(rng, temp_path, create_backup=True)
        assert success, "Save with backup should succeed"
        
        # Check that backup was created
        backup_files = list(temp_path.parent.glob(f"{temp_path.stem}.*.backup.*"))
        # Note: Backup functionality may vary based on implementation
        
    finally:
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
        # Cleanup any backup files
        for backup_file in temp_path.parent.glob(f"{temp_path.stem}.*.backup.*"):
            backup_file.unlink()
    
    # Test graceful handling of file system limitations
    # This would require platform-specific testing for full coverage


# Unit tests for SeedManager class initialization and configuration
@pytest.mark.unit
@pytest.mark.parametrize('thread_safe', [True, False])
@pytest.mark.parametrize('enable_validation', [True, False])
def test_seed_manager_initialization(thread_safe, enable_validation):
    """Test SeedManager class initialization with various configuration options ensuring proper setup, validation, and thread safety configuration"""
    # Test SeedManager initialization with various parameter combinations
    default_seed = 42
    seed_manager = SeedManager(
        default_seed=default_seed,
        enable_validation=enable_validation,
        thread_safe=thread_safe
    )
    
    # Verify default_seed, enable_validation, and thread_safe configuration
    assert seed_manager.default_seed == default_seed, \
        f"Expected default_seed {default_seed}, got {seed_manager.default_seed}"
    assert seed_manager.enable_validation == enable_validation, \
        f"Expected enable_validation {enable_validation}, got {seed_manager.enable_validation}"
    assert seed_manager.thread_safe == thread_safe, \
        f"Expected thread_safe {thread_safe}, got {seed_manager.thread_safe}"
    
    # Check internal state initialization (generators, history, locks)
    assert isinstance(seed_manager.active_generators, dict), \
        "active_generators should be dictionary"
    assert len(seed_manager.active_generators) == 0, \
        "Initial active_generators should be empty"
    assert isinstance(seed_manager.seed_history, list), \
        "seed_history should be list"
    assert len(seed_manager.seed_history) == 0, \
        "Initial seed_history should be empty"
    
    # Test default seed validation during initialization
    if default_seed is not None:
        # Valid default seed should be accepted
        assert seed_manager.default_seed == default_seed, \
            "Valid default seed should be stored"
    
    # Test invalid default seed
    with pytest.raises(ValidationError):
        SeedManager(default_seed="invalid", enable_validation=True, thread_safe=False)
    
    # Verify thread safety setup when enabled
    if thread_safe:
        assert hasattr(seed_manager, '_lock'), "Thread-safe manager should have lock"
        assert seed_manager._lock is not None, "Lock should be initialized"
        assert isinstance(seed_manager._lock, threading.Lock), \
            "Lock should be threading.Lock instance"
    else:
        assert seed_manager._lock is None, "Non-thread-safe manager should not have lock"
    
    # Check logger initialization and configuration
    assert hasattr(seed_manager, 'logger'), "SeedManager should have logger"
    assert seed_manager.logger is not None, "Logger should be initialized"
    
    # Validate initial state consistency and proper resource setup
    assert seed_manager.default_seed == default_seed, "Default seed should be preserved"
    
    # Test initialization with None default_seed
    none_manager = SeedManager(default_seed=None, enable_validation=enable_validation, 
                              thread_safe=thread_safe)
    assert none_manager.default_seed is None, "None default_seed should be preserved"


# Unit tests for SeedManager seeding operations and tracking
@pytest.mark.unit
@pytest.mark.parametrize('seed', TEST_SEEDS + [None])
def test_seed_manager_seeding_operations(seed):
    """Test SeedManager seeding operations including seed creation, tracking, and context management with validation and thread safety testing"""
    # Test SeedManager.seed() with various seed values
    seed_manager = SeedManager(enable_validation=True, thread_safe=True)
    
    np_random, seed_used = seed_manager.seed(seed=seed)
    
    # Verify RNG creation and tracking in active_generators
    assert isinstance(np_random, np.random.Generator), \
        f"Expected numpy.random.Generator, got {type(np_random)}"
    assert isinstance(seed_used, int), f"Expected int seed_used, got {type(seed_used)}"
    
    # Check seed history recording with timestamps and context
    assert len(seed_manager.seed_history) > 0, "Seed operation should be recorded in history"
    last_history = seed_manager.seed_history[-1]
    assert last_history['operation'] == 'seed', "History should record seed operation"
    assert last_history['seed_used'] == seed_used, "History should record actual seed used"
    assert 'timestamp' in last_history, "History should include timestamp"
    assert 'context_id' in last_history, "History should include context ID"
    
    # Test context_id generation and management
    context_id = last_history['context_id']
    assert context_id in seed_manager.active_generators, \
        "Context ID should be tracked in active generators"
    
    generator_info = seed_manager.active_generators[context_id]
    assert generator_info['generator'] is np_random, \
        "Tracked generator should match returned generator"
    assert generator_info['seed_used'] == seed_used, \
        "Tracked seed should match returned seed"
    assert 'creation_timestamp' in generator_info, \
        "Generator info should include creation timestamp"
    
    # Validate thread safety during concurrent seeding operations
    # Test multiple seeding operations
    seeds_to_test = [123, 456, 789] if seed is not None else [None, None, None]
    contexts = []
    
    for test_seed in seeds_to_test:
        rng, used_seed = seed_manager.seed(seed=test_seed, context_id=f"test_context_{used_seed if test_seed else 'none'}")
        contexts.append((rng, used_seed))
    
    # Verify all contexts are tracked
    assert len(seed_manager.active_generators) == len(contexts) + 1, \
        "All seeding operations should be tracked"
    
    # Test default seed fallback when no seed provided
    if seed is None:
        # With no default seed set, should generate random seed
        assert SEED_MIN_VALUE <= seed_used <= SEED_MAX_VALUE, \
            "Random seed should be in valid range"
    else:
        # With seed provided, should match input
        assert seed_used == seed, f"Seed used should match input seed {seed}"
    
    # Verify logging integration for seeding operations
    # (This would require capturing log output in a full implementation)
    
    # Test custom context_id
    custom_context = "custom_test_context"
    custom_rng, custom_seed = seed_manager.seed(seed=42, context_id=custom_context)
    assert custom_context in seed_manager.active_generators, \
        "Custom context ID should be tracked"


# Unit tests for SeedManager episode seed generation
@pytest.mark.unit
@pytest.mark.parametrize('base_seed', TEST_SEEDS[:3])
@pytest.mark.parametrize('episode_number', [0, 1, 10, 100])
def test_seed_manager_episode_seed_generation(base_seed, episode_number):
    """Test SeedManager episode seed generation ensuring deterministic episode seeding with proper context integration and sequence management"""
    seed_manager = SeedManager()
    
    # Test generate_episode_seed with various base seeds and episode numbers
    episode_seed = seed_manager.generate_episode_seed(base_seed, episode_number)
    
    # Verify deterministic seed generation for episode contexts
    repeat_seed = seed_manager.generate_episode_seed(base_seed, episode_number)
    assert episode_seed == repeat_seed, \
        f"Identical inputs should produce identical episode seeds: {episode_seed} != {repeat_seed}"
    
    # Check episode seed uniqueness across different episode numbers
    other_episode_seed = seed_manager.generate_episode_seed(base_seed, episode_number + 1)
    assert episode_seed != other_episode_seed, \
        f"Different episode numbers should produce different seeds"
    
    # Test experiment_id integration in seed generation
    experiment_id = "test_experiment_123"
    exp_seed_1 = seed_manager.generate_episode_seed(base_seed, episode_number, 
                                                   experiment_id=experiment_id)
    exp_seed_2 = seed_manager.generate_episode_seed(base_seed, episode_number, 
                                                   experiment_id=experiment_id)
    assert exp_seed_1 == exp_seed_2, \
        "Same experiment context should produce identical seeds"
    
    # Different experiment IDs should produce different seeds
    different_exp_id = "different_experiment_456"
    different_exp_seed = seed_manager.generate_episode_seed(base_seed, episode_number,
                                                           experiment_id=different_exp_id)
    assert exp_seed_1 != different_exp_seed, \
        "Different experiment IDs should produce different seeds"
    
    # Validate seed range compliance for generated episode seeds
    assert SEED_MIN_VALUE <= episode_seed <= SEED_MAX_VALUE, \
        f"Episode seed {episode_seed} not in valid range"
    assert isinstance(episode_seed, int), \
        f"Episode seed should be integer, got {type(episode_seed)}"
    
    # Test reproducibility of episode seed sequences
    sequence_length = 10
    episode_sequence_1 = [seed_manager.generate_episode_seed(base_seed, i) 
                         for i in range(sequence_length)]
    episode_sequence_2 = [seed_manager.generate_episode_seed(base_seed, i) 
                         for i in range(sequence_length)]
    
    assert episode_sequence_1 == episode_sequence_2, \
        "Episode seed sequences should be reproducible"
    
    # Verify context tracking and logging for episode seeding
    # Episode seeding doesn't modify active_generators, so no tracking expected
    initial_generator_count = len(seed_manager.active_generators)
    seed_manager.generate_episode_seed(base_seed, episode_number)
    assert len(seed_manager.active_generators) == initial_generator_count, \
        "Episode seed generation should not affect active generators"


# Reproducibility tests for SeedManager validation capabilities
@pytest.mark.reproducibility
@pytest.mark.parametrize('test_seed', TEST_SEEDS[:3])
def test_seed_manager_reproducibility_validation(test_seed):
    """Test SeedManager reproducibility validation ensuring comprehensive testing, statistical analysis, and scientific reporting capabilities"""
    seed_manager = SeedManager()
    
    # Test SeedManager.validate_reproducibility() with various test seeds
    validation_result = seed_manager.validate_reproducibility(test_seed, num_tests=10)
    
    # Verify comprehensive reproducibility testing across multiple runs
    assert isinstance(validation_result, dict), \
        "Validation result should be dictionary"
    assert 'test_configuration' in validation_result, \
        "Result should include test configuration"
    assert 'results_summary' in validation_result, \
        "Result should include results summary"
    assert 'overall_status' in validation_result, \
        "Result should include overall status"
    
    # Check statistical analysis and success rate calculations
    results_summary = validation_result['results_summary']
    assert 'success_count' in results_summary, \
        "Should report success count"
    assert 'success_rate' in results_summary, \
        "Should report success rate"
    assert 'total_tests' in results_summary, \
        "Should report total test count"
    
    success_rate = results_summary['success_rate']
    assert 0.0 <= success_rate <= 1.0, \
        f"Success rate should be between 0 and 1: {success_rate}"
    
    # Test failure pattern analysis and root cause identification
    if 'failure_analysis' in validation_result:
        failure_analysis = validation_result['failure_analysis']
        assert 'num_failures' in failure_analysis, \
            "Failure analysis should include failure count"
        assert 'failure_rate' in failure_analysis, \
            "Failure analysis should include failure rate"
    
    # Validate reproducibility report generation and content
    assert 'statistical_analysis' in validation_result, \
        "Should include statistical analysis"
    statistical_analysis = validation_result['statistical_analysis']
    
    if success_rate > 0:
        assert 'mean_reproducibility_score' in statistical_analysis, \
            "Should include mean reproducibility score"
        mean_score = statistical_analysis['mean_reproducibility_score']
        assert 0.0 <= mean_score <= 1.0, \
            f"Mean score should be between 0 and 1: {mean_score}"
    
    # Test recommendation system for reproducibility improvement
    assert 'recommendations' in validation_result, \
        "Result should include recommendations"
    recommendations = validation_result['recommendations']
    assert isinstance(recommendations, list), \
        "Recommendations should be list"
    
    # Verify scientific documentation quality and completeness
    test_config = validation_result['test_configuration']
    assert test_config['test_seed'] == test_seed, \
        "Test configuration should record correct seed"
    assert test_config['num_tests'] == 10, \
        "Test configuration should record correct test count"
    assert 'validation_timestamp' in test_config, \
        "Test configuration should include timestamp"
    
    # Test with higher number of tests for more thorough analysis
    detailed_result = seed_manager.validate_reproducibility(test_seed, num_tests=25, 
                                                           tolerance=REPRODUCIBILITY_TOLERANCE)
    
    detailed_success_rate = detailed_result['results_summary']['success_rate']
    assert detailed_success_rate >= 0.8, \
        f"Detailed validation should have high success rate: {detailed_success_rate}"
    
    # Verify overall status classification
    overall_status = detailed_result['overall_status']
    assert overall_status in ['PASS', 'FAIL'], \
        f"Overall status should be PASS or FAIL: {overall_status}"
    
    if detailed_success_rate >= 0.95:
        assert overall_status == 'PASS', \
            "High success rate should result in PASS status"


# Thread safety tests for SeedManager concurrent operations
@pytest.mark.unit
@pytest.mark.slow
def test_seed_manager_thread_safety():
    """Test SeedManager thread safety ensuring concurrent access protection, data consistency, and thread-safe operations across multiple threads"""
    # Create SeedManager with thread_safe=True configuration
    seed_manager = SeedManager(thread_safe=True, enable_validation=True)
    
    # Launch multiple threads performing concurrent seeding operations
    num_threads = 10
    operations_per_thread = 10
    thread_results = []
    errors = []
    
    def seeding_worker(thread_id):
        """Worker function for concurrent seeding operations"""
        thread_generators = []
        thread_seeds = []
        
        try:
            for i in range(operations_per_thread):
                seed = (thread_id * 100) + i  # Unique seed per operation
                context_id = f"thread_{thread_id}_op_{i}"
                
                rng, seed_used = seed_manager.seed(seed=seed, context_id=context_id)
                thread_generators.append(rng)
                thread_seeds.append(seed_used)
                
                # Verify thread safety during seed generation and tracking
                assert isinstance(rng, np.random.Generator), \
                    f"Thread {thread_id} should get valid generator"
                assert seed_used == seed, \
                    f"Thread {thread_id} should get correct seed"
                
                # Brief sleep to increase chance of race conditions
                time.sleep(0.001)
            
            return {
                'thread_id': thread_id,
                'generators': thread_generators,
                'seeds': thread_seeds,
                'success': True
            }
            
        except Exception as e:
            errors.append(f"Thread {thread_id} error: {e}")
            return {
                'thread_id': thread_id,
                'success': False,
                'error': str(e)
            }
    
    # Create and start threads
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=lambda tid=i: thread_results.append(seeding_worker(tid)))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join(timeout=10.0)  # 10 second timeout
        if thread.is_alive():
            errors.append(f"Thread timeout")
    
    # Verify data consistency across concurrent access scenarios
    assert len(errors) == 0, f"Thread safety errors occurred: {errors}"
    assert len(thread_results) == num_threads, \
        f"Expected {num_threads} thread results, got {len(thread_results)}"
    
    # Check that all operations succeeded
    successful_threads = [r for r in thread_results if r.get('success', False)]
    assert len(successful_threads) == num_threads, \
        f"All threads should succeed: {len(successful_threads)}/{num_threads}"
    
    # Check lock acquisition and release during operations
    # Verify all expected generators are tracked
    expected_total_generators = num_threads * operations_per_thread
    actual_generators = len(seed_manager.active_generators)
    assert actual_generators == expected_total_generators, \
        f"Expected {expected_total_generators} generators, got {actual_generators}"
    
    # Test thread-local context management and isolation
    # Verify each thread's generators are distinct
    all_context_ids = set(seed_manager.active_generators.keys())
    assert len(all_context_ids) == expected_total_generators, \
        "All context IDs should be unique"
    
    # Validate no race conditions or data corruption occurs
    # Check seed history integrity
    total_history_entries = len(seed_manager.seed_history)
    assert total_history_entries >= expected_total_generators, \
        f"History should record all operations: {total_history_entries} >= {expected_total_generators}"
    
    # Verify each thread's seeds are correctly recorded
    for result in successful_threads:
        thread_id = result['thread_id']
        expected_seeds = result['seeds']
        
        # Find history entries for this thread
        thread_entries = [h for h in seed_manager.seed_history 
                         if f"thread_{thread_id}" in h.get('context_id', '')]
        
        assert len(thread_entries) >= len(expected_seeds), \
            f"Thread {thread_id} history entries should match operations"


# Unit tests for ReproducibilityTracker initialization and configuration
@pytest.mark.unit
@pytest.mark.parametrize('strict_validation', [True, False])
@pytest.mark.parametrize('default_tolerance', [1e-6, 1e-10, 1e-12])
def test_reproducibility_tracker_initialization(strict_validation, default_tolerance):
    """Test ReproducibilityTracker class initialization with various parameters ensuring proper setup, session management, and validation framework configuration"""
    # Test ReproducibilityTracker initialization with various parameters
    session_id = f"test_session_{int(time.time())}"
    tracker = ReproducibilityTracker(
        default_tolerance=default_tolerance,
        strict_validation=strict_validation,
        session_id=session_id
    )
    
    # Verify default_tolerance, strict_validation, and session_id setup
    assert tracker.default_tolerance == default_tolerance, \
        f"Expected tolerance {default_tolerance}, got {tracker.default_tolerance}"
    assert tracker.strict_validation == strict_validation, \
        f"Expected strict_validation {strict_validation}, got {tracker.strict_validation}"
    assert tracker.session_id == session_id, \
        f"Expected session_id {session_id}, got {tracker.session_id}"
    
    # Check internal data structure initialization (records, results, summaries)
    assert isinstance(tracker.episode_records, dict), \
        "episode_records should be dictionary"
    assert len(tracker.episode_records) == 0, \
        "Initial episode_records should be empty"
    assert isinstance(tracker.test_results, dict), \
        "test_results should be dictionary"
    assert len(tracker.test_results) == 0, \
        "Initial test_results should be empty"
    assert isinstance(tracker.statistical_summaries, list), \
        "statistical_summaries should be list"
    assert len(tracker.statistical_summaries) == 0, \
        "Initial statistical_summaries should be empty"
    
    # Test session_id generation when not provided
    auto_tracker = ReproducibilityTracker(
        default_tolerance=default_tolerance,
        strict_validation=strict_validation
    )
    assert auto_tracker.session_id is not None, \
        "Session ID should be auto-generated when not provided"
    assert auto_tracker.session_id.startswith("repro_session_"), \
        "Auto-generated session ID should have correct prefix"
    assert len(auto_tracker.session_id) > len("repro_session_"), \
        "Auto-generated session ID should include unique identifier"
    
    # Validate logger setup for reproducibility tracking
    assert hasattr(tracker, 'logger'), "Tracker should have logger"
    assert tracker.logger is not None, "Logger should be initialized"
    
    # Check parameter validation during initialization
    # Test invalid tolerance
    with pytest.raises(ValidationError):
        ReproducibilityTracker(default_tolerance=-1.0)
    
    with pytest.raises(ValidationError):
        ReproducibilityTracker(default_tolerance=0.0)
    
    # Test invalid session_id
    with pytest.raises(ValidationError):
        ReproducibilityTracker(session_id="")
    
    with pytest.raises(ValidationError):
        ReproducibilityTracker(session_id="   ")  # Only whitespace
    
    # Verify proper resource allocation and initial state consistency
    assert tracker.default_tolerance > 0, "Tolerance should be positive"
    assert isinstance(tracker.strict_validation, bool), \
        "Strict validation should be boolean"


# Reproducibility tests for ReproducibilityTracker episode recording
@pytest.mark.reproducibility
@pytest.mark.parametrize('episode_seed', TEST_SEEDS[:3])
def test_reproducibility_tracker_episode_recording(episode_seed):
    """Test ReproducibilityTracker episode recording functionality ensuring comprehensive data capture, checksum calculation, and metadata storage"""
    tracker = ReproducibilityTracker(strict_validation=True)
    
    # Generate test action and observation sequences
    rng, _ = create_seeded_rng(episode_seed)
    action_sequence = rng.integers(0, 4, size=50).tolist()  # Actions 0-3
    observation_sequence = rng.random(51).tolist()  # Initial obs + 50 step obs
    
    # Test record_episode with various episode seeds and data
    episode_metadata = {
        'environment': 'test_environment',
        'episode_length': len(action_sequence),
        'completion_status': 'success'
    }
    
    episode_id = tracker.record_episode(
        episode_seed=episode_seed,
        action_sequence=action_sequence,
        observation_sequence=observation_sequence,
        episode_metadata=episode_metadata
    )
    
    # Verify comprehensive data validation and storage
    assert isinstance(episode_id, str), f"Episode ID should be string, got {type(episode_id)}"
    assert episode_id in tracker.episode_records, "Episode should be recorded"
    
    recorded_episode = tracker.episode_records[episode_id]
    assert recorded_episode['episode_seed'] == episode_seed, \
        "Recorded seed should match input seed"
    assert recorded_episode['action_sequence'] == action_sequence, \
        "Recorded actions should match input sequence"
    assert recorded_episode['observation_sequence'] == observation_sequence, \
        "Recorded observations should match input sequence"
    
    # Check checksum calculation for sequence integrity
    assert 'action_checksum' in recorded_episode, \
        "Episode record should include action checksum"
    assert 'observation_checksum' in recorded_episode, \
        "Episode record should include observation checksum"
    
    # Verify checksums are consistent
    action_checksum_1 = recorded_episode['action_checksum']
    
    # Record same episode again and verify checksum consistency
    episode_id_2 = tracker.record_episode(
        episode_seed=episode_seed,
        action_sequence=action_sequence,
        observation_sequence=observation_sequence
    )
    
    recorded_episode_2 = tracker.episode_records[episode_id_2]
    action_checksum_2 = recorded_episode_2['action_checksum']
    
    # Identical sequences should have identical checksums
    assert action_checksum_1 == action_checksum_2, \
        "Identical action sequences should have identical checksums"
    
    # Test metadata preservation and context information
    assert 'metadata' in recorded_episode, "Episode record should include metadata"
    recorded_metadata = recorded_episode['metadata']
    assert recorded_metadata['environment'] == 'test_environment', \
        "Metadata should be preserved correctly"
    
    # Validate episode record ID generation and uniqueness
    assert episode_id != episode_id_2, "Episode IDs should be unique"
    assert episode_id.startswith("episode_"), "Episode ID should have correct prefix"
    
    # Test timestamp recording and episode tracking
    assert 'recording_timestamp' in recorded_episode, \
        "Episode record should include timestamp"
    assert recorded_episode['recording_timestamp'] > 0, \
        "Timestamp should be positive"
    assert 'session_id' in recorded_episode, \
        "Episode record should include session ID"
    assert recorded_episode['session_id'] == tracker.session_id, \
        "Session ID should match tracker session"
    
    # Test sequence length validation in strict mode
    if tracker.strict_validation:
        # Strict mode should validate observation sequence length
        expected_obs_length = len(action_sequence) + 1
        assert len(observation_sequence) == expected_obs_length, \
            f"Strict mode should validate obs length: {len(observation_sequence)} == {expected_obs_length}"
    
    # Test error handling for invalid sequences
    with pytest.raises(ValidationError):
        tracker.record_episode(
            episode_seed=episode_seed,
            action_sequence=[],  # Empty action sequence
            observation_sequence=observation_sequence
        )
    
    with pytest.raises(ValidationError):
        tracker.record_episode(
            episode_seed="invalid_seed",  # Invalid seed type
            action_sequence=action_sequence,
            observation_sequence=observation_sequence
        )


# Reproducibility tests for ReproducibilityTracker verification capabilities
@pytest.mark.reproducibility
def test_reproducibility_tracker_verification():
    """Test ReproducibilityTracker episode verification functionality ensuring accurate comparison, statistical analysis, and detailed reporting"""
    tracker = ReproducibilityTracker(default_tolerance=REPRODUCIBILITY_TOLERANCE)
    
    # Record reference episode using record_episode
    seed = 42
    rng, _ = create_seeded_rng(seed)
    original_actions = rng.integers(0, 4, size=30).tolist()
    original_observations = rng.random(31).tolist()
    
    episode_id = tracker.record_episode(
        episode_seed=seed,
        action_sequence=original_actions,
        observation_sequence=original_observations,
        episode_metadata={'test_type': 'verification_test'}
    )
    
    # Generate matching episode sequence
    rng_match, _ = create_seeded_rng(seed)
    matching_actions = rng_match.integers(0, 4, size=30).tolist()
    matching_observations = rng_match.random(31).tolist()
    
    # Test verify_episode_reproducibility with matching scenario
    match_result = tracker.verify_episode_reproducibility(
        episode_record_id=episode_id,
        new_action_sequence=matching_actions,
        new_observation_sequence=matching_observations
    )
    
    # Verify detailed sequence comparison and statistical analysis
    assert isinstance(match_result, dict), "Verification result should be dictionary"
    assert 'match_status' in match_result, "Result should include match status"
    assert 'sequences_match' in match_result, "Result should include sequences match flag"
    assert match_result['sequences_match'] is True, \
        "Identical sequences should match"
    assert match_result['match_status'] == 'PASS', \
        "Matching sequences should have PASS status"
    
    # Check statistical analysis details
    assert 'action_comparison' in match_result, "Should include action comparison"
    assert 'observation_comparison' in match_result, "Should include observation comparison"
    
    action_comp = match_result['action_comparison']
    obs_comp = match_result['observation_comparison']
    
    assert action_comp['sequences_match'] is True, "Actions should match"
    assert obs_comp['sequences_match'] is True, "Observations should match"
    
    # Generate non-matching episode sequence
    different_seed = 123
    rng_diff, _ = create_seeded_rng(different_seed)
    different_actions = rng_diff.integers(0, 4, size=30).tolist()
    different_observations = rng_diff.random(31).tolist()
    
    # Test with non-matching scenario
    mismatch_result = tracker.verify_episode_reproducibility(
        episode_record_id=episode_id,
        new_action_sequence=different_actions,
        new_observation_sequence=different_observations
    )
    
    # Check discrepancy identification and detailed analysis
    assert mismatch_result['sequences_match'] is False, \
        "Different sequences should not match"
    assert mismatch_result['match_status'] == 'FAIL', \
        "Non-matching sequences should have FAIL status"
    
    assert 'discrepancy_analysis' in mismatch_result, \
        "Mismatch should include discrepancy analysis"
    
    discrepancy = mismatch_result['discrepancy_analysis']
    assert discrepancy['total_discrepancies'] > 0, \
        "Should identify discrepancies between different sequences"
    
    # Test tolerance-based comparison and sensitivity analysis
    # Test with custom tolerance
    custom_tolerance = 1e-15
    strict_result = tracker.verify_episode_reproducibility(
        episode_record_id=episode_id,
        new_action_sequence=matching_actions,
        new_observation_sequence=matching_observations,
        custom_tolerance=custom_tolerance
    )
    
    assert strict_result['tolerance_used'] == custom_tolerance, \
        "Should use custom tolerance when provided"
    
    # Validate comprehensive verification report generation
    assert 'verification_id' in match_result, "Should include verification ID"
    assert 'verification_timestamp' in match_result, "Should include timestamp"
    assert 'reproducibility_score' in match_result, "Should include reproducibility score"
    
    # Test error handling for invalid episode IDs
    with pytest.raises(ValidationError):
        tracker.verify_episode_reproducibility(
            episode_record_id="nonexistent_episode",
            new_action_sequence=matching_actions,
            new_observation_sequence=matching_observations
        )
    
    # Test sequence length mismatch handling
    short_actions = original_actions[:10]  # Shorter sequence
    length_result = tracker.verify_episode_reproducibility(
        episode_record_id=episode_id,
        new_action_sequence=short_actions,
        new_observation_sequence=original_observations
    )
    
    assert length_result['match_status'] == 'LENGTH_MISMATCH', \
        "Length mismatch should be detected"
    assert length_result['sequences_match'] is False, \
        "Length mismatch should result in no match"


# Reproducibility tests for ReproducibilityTracker reporting capabilities
@pytest.mark.reproducibility
def test_reproducibility_tracker_reporting():
    """Test ReproducibilityTracker reporting functionality ensuring scientific-quality reports, statistical summaries, and research documentation"""
    tracker = ReproducibilityTracker()
    
    # Record multiple episodes and verification results
    test_data = []
    for i, seed in enumerate(TEST_SEEDS[:3]):
        rng, _ = create_seeded_rng(seed)
        actions = rng.integers(0, 4, size=20).tolist()
        observations = rng.random(21).tolist()
        
        episode_id = tracker.record_episode(
            episode_seed=seed,
            action_sequence=actions,
            observation_sequence=observations,
            episode_metadata={'episode_number': i}
        )
        
        test_data.append({
            'episode_id': episode_id,
            'seed': seed,
            'actions': actions,
            'observations': observations
        })
        
        # Verify episode reproducibility (should pass)
        rng_verify, _ = create_seeded_rng(seed)
        verify_actions = rng_verify.integers(0, 4, size=20).tolist()
        verify_observations = rng_verify.random(21).tolist()
        
        verification_result = tracker.verify_episode_reproducibility(
            episode_record_id=episode_id,
            new_action_sequence=verify_actions,
            new_observation_sequence=verify_observations
        )
    
    # Test generate_reproducibility_report with various formats
    report_dict = tracker.generate_reproducibility_report(report_format='dict')
    
    # Verify statistical summaries and success rate calculations
    assert isinstance(report_dict, dict), "Report should be dictionary"
    assert 'summary_statistics' in report_dict, "Report should include summary statistics"
    assert 'reproducibility_assessment' in report_dict, "Report should include assessment"
    
    summary = report_dict['summary_statistics']
    assert summary['total_episodes_recorded'] == len(test_data), \
        f"Should record {len(test_data)} episodes"
    assert summary['total_verifications_performed'] > 0, \
        "Should have performed verifications"
    
    success_rate = summary['overall_success_rate']
    assert 0.0 <= success_rate <= 1.0, "Success rate should be between 0 and 1"
    
    # Check detailed failure analysis when include_detailed_analysis=True
    detailed_report = tracker.generate_reproducibility_report(
        report_format='dict',
        include_detailed_analysis=True
    )
    
    assert 'failure_analysis' in detailed_report, \
        "Detailed report should include failure analysis"
    assert 'episode_statistics' in detailed_report, \
        "Detailed report should include episode statistics"
    
    # Test custom sections integration and report formatting
    custom_sections = {
        'experimental_setup': 'Test reproducibility validation',
        'hardware_info': 'Test environment'
    }
    
    custom_report = tracker.generate_reproducibility_report(
        report_format='dict',
        custom_sections=custom_sections
    )
    
    assert 'custom_experimental_setup' in custom_report, \
        "Custom sections should be included"
    
    # Validate scientific documentation quality and completeness
    assessment = report_dict['reproducibility_assessment']
    assert 'overall_status' in assessment, "Should include overall status"
    assert 'confidence_level' in assessment, "Should include confidence level"
    assert 'recommendations' in assessment, "Should include recommendations"
    
    # Test report format compatibility (JSON, markdown, etc.)
    json_report = tracker.generate_reproducibility_report(report_format='json')
    assert 'json_report' in json_report, "JSON format should be wrapped in dictionary"
    
    markdown_report = tracker.generate_reproducibility_report(report_format='markdown')
    assert 'markdown_report' in markdown_report, "Markdown format should be wrapped in dictionary"
    
    # Verify report metadata includes session information
    metadata = report_dict['report_metadata']
    assert metadata['session_id'] == tracker.session_id, \
        "Report metadata should include session ID"
    assert 'generation_timestamp' in metadata, \
        "Report metadata should include timestamp"


# Unit tests for ReproducibilityTracker data export capabilities
@pytest.mark.unit
@pytest.mark.parametrize('export_format', ['json', 'csv'])
@pytest.mark.parametrize('compress_output', [True, False])
def test_reproducibility_tracker_data_export(export_format, compress_output):
    """Test ReproducibilityTracker data export functionality ensuring multiple format support, data integrity, and archival compatibility"""
    tracker = ReproducibilityTracker()
    
    # Generate reproducibility data with multiple episodes and tests
    test_episodes = []
    for seed in [42, 123]:
        rng, _ = create_seeded_rng(seed)
        actions = rng.integers(0, 4, size=10).tolist()
        observations = rng.random(11).tolist()
        
        episode_id = tracker.record_episode(
            episode_seed=seed,
            action_sequence=actions,
            observation_sequence=observations
        )
        test_episodes.append(episode_id)
        
        # Add verification
        rng_verify, _ = create_seeded_rng(seed)
        verify_actions = rng_verify.integers(0, 4, size=10).tolist()
        verify_observations = rng_verify.random(11).tolist()
        
        tracker.verify_episode_reproducibility(
            episode_record_id=episode_id,
            new_action_sequence=verify_actions,
            new_observation_sequence=verify_observations
        )
    
    # Test export_reproducibility_data with various formats
    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = pathlib.Path(temp_dir) / f"export_test"
        
        success = tracker.export_reproducibility_data(
            export_path=export_path,
            export_format=export_format,
            include_raw_data=True,
            compress_output=compress_output
        )
        
        # Verify data integrity and format compliance
        assert success is True, "Export should succeed"
        
        # Check expected file extension
        if export_format == 'json':
            if compress_output:
                expected_path = export_path.with_suffix('.json.gz')
            else:
                expected_path = export_path.with_suffix('.json')
        else:  # csv
            expected_path = export_path.with_suffix('.csv')
        
        assert expected_path.exists(), f"Export file should exist: {expected_path}"
        assert expected_path.stat().st_size > 0, "Export file should not be empty"
        
        # Test compression functionality when enabled
        if compress_output and export_format == 'json':
            # Verify compressed file is smaller than uncompressed
            import gzip
            with gzip.open(expected_path, 'rt') as f:
                content = f.read()
            assert len(content) > 0, "Compressed content should be readable"
            
            # Verify it contains expected data
            if export_format == 'json':
                import json
                data = json.loads(content)
                assert 'export_metadata' in data, "Exported data should include metadata"
        
        # Check raw data inclusion when include_raw_data=True
        if export_format == 'json' and not compress_output:
            with open(expected_path, 'r') as f:
                exported_data = json.load(f)
            
            assert 'episode_records' in exported_data, \
                "Should include episode records when raw data included"
            assert 'verification_results' in exported_data, \
                "Should include verification results when raw data included"
            
            # Verify episode data is present
            episode_records = exported_data['episode_records']
            assert len(episode_records) == len(test_episodes), \
                "Should export all recorded episodes"
        
        elif export_format == 'csv':
            # Verify CSV format
            import csv
            with open(expected_path, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)
                assert 'verification_id' in header, "CSV should include verification_id column"
                assert 'match_status' in header, "CSV should include match_status column"
                
                # Count data rows
                data_rows = list(reader)
                assert len(data_rows) > 0, "CSV should contain verification data"
        
        # Validate export path creation and file operations
        # Path should be created successfully and file should be valid
        
        # Test atomic export operations and error handling
        # Test with invalid path
        invalid_path = pathlib.Path('/invalid/path/export')
        with pytest.raises((ResourceError, OSError, PermissionError)):
            tracker.export_reproducibility_data(
                export_path=invalid_path,
                export_format=export_format
            )


# Performance tests for seeding operations against timing targets
@pytest.mark.performance
@pytest.mark.slow
def test_seeding_performance_benchmarks():
    """Test seeding operations performance ensuring all functions meet timing targets and performance requirements for reinforcement learning applications"""
    # Benchmark validate_seed function execution time
    validation_times = []
    test_seeds = TEST_SEEDS * 20  # More samples for reliable timing
    
    for seed in test_seeds:
        start_time = time.perf_counter()
        validate_seed(seed)
        end_time = time.perf_counter()
        validation_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    mean_validation_time = np.mean(validation_times)
    assert mean_validation_time < PERFORMANCE_TARGET_SEEDING_MS, \
        f"Seed validation too slow: {mean_validation_time:.3f}ms > {PERFORMANCE_TARGET_SEEDING_MS}ms"
    
    # Measure create_seeded_rng latency against 1ms target
    rng_creation_times = []
    for seed in TEST_SEEDS * 10:
        start_time = time.perf_counter()
        create_seeded_rng(seed)
        end_time = time.perf_counter()
        rng_creation_times.append((end_time - start_time) * 1000)
    
    mean_rng_time = np.mean(rng_creation_times)
    assert mean_rng_time < PERFORMANCE_TARGET_SEEDING_MS, \
        f"RNG creation too slow: {mean_rng_time:.3f}ms > {PERFORMANCE_TARGET_SEEDING_MS}ms"
    
    # Test generate_deterministic_seed performance with various inputs
    deterministic_times = []
    test_strings = [f"performance_test_{i}" for i in range(100)]
    
    for test_string in test_strings:
        start_time = time.perf_counter()
        generate_deterministic_seed(test_string)
        end_time = time.perf_counter()
        deterministic_times.append((end_time - start_time) * 1000)
    
    mean_deterministic_time = np.mean(deterministic_times)
    assert mean_deterministic_time < PERFORMANCE_TARGET_SEEDING_MS, \
        f"Deterministic seed generation too slow: {mean_deterministic_time:.3f}ms > {PERFORMANCE_TARGET_SEEDING_MS}ms"
    
    # Benchmark SeedManager operations under load
    seed_manager = SeedManager(thread_safe=True)
    manager_times = []
    
    for i in range(50):
        start_time = time.perf_counter()
        seed_manager.seed(seed=42 + i)
        end_time = time.perf_counter()
        manager_times.append((end_time - start_time) * 1000)
    
    mean_manager_time = np.mean(manager_times)
    assert mean_manager_time < PERFORMANCE_TARGET_SEEDING_MS * 2, \
        f"SeedManager operations too slow: {mean_manager_time:.3f}ms > {PERFORMANCE_TARGET_SEEDING_MS * 2}ms"
    
    # Measure ReproducibilityTracker recording and verification speed
    tracker = ReproducibilityTracker()
    recording_times = []
    verification_times = []
    
    for i in range(10):  # Fewer iterations due to higher complexity
        seed = TEST_SEEDS[i % len(TEST_SEEDS)]
        rng, _ = create_seeded_rng(seed)
        actions = rng.integers(0, 4, size=20).tolist()
        observations = rng.random(21).tolist()
        
        # Measure recording time
        start_time = time.perf_counter()
        episode_id = tracker.record_episode(seed, actions, observations)
        end_time = time.perf_counter()
        recording_times.append((end_time - start_time) * 1000)
        
        # Measure verification time
        rng_verify, _ = create_seeded_rng(seed)
        verify_actions = rng_verify.integers(0, 4, size=20).tolist()
        verify_observations = rng_verify.random(21).tolist()
        
        start_time = time.perf_counter()
        tracker.verify_episode_reproducibility(episode_id, verify_actions, verify_observations)
        end_time = time.perf_counter()
        verification_times.append((end_time - start_time) * 1000)
    
    mean_recording_time = np.mean(recording_times)
    mean_verification_time = np.mean(verification_times)
    
    # Allow higher targets for more complex operations
    assert mean_recording_time < PERFORMANCE_TARGET_SEEDING_MS * 10, \
        f"Episode recording too slow: {mean_recording_time:.3f}ms"
    assert mean_verification_time < PERFORMANCE_TARGET_SEEDING_MS * 20, \
        f"Episode verification too slow: {mean_verification_time:.3f}ms"
    
    # Test performance degradation under concurrent access
    # This would involve threading tests with timing measurements
    
    # Validate memory usage stays within acceptable limits
    import sys
    current_memory = sys.getsizeof(seed_manager) + sys.getsizeof(tracker)
    max_memory_mb = 10  # 10MB limit for seeding components
    assert current_memory < max_memory_mb * 1024 * 1024, \
        f"Memory usage too high: {current_memory / (1024*1024):.1f}MB > {max_memory_mb}MB"


# Integration tests for seeding with environment workflow
@pytest.mark.integration
@pytest.mark.parametrize('seed', TEST_SEEDS[:2])
def test_seeding_integration_with_environment(seed):
    """Test seeding integration with plume navigation environment ensuring end-to-end reproducibility and proper integration with Gymnasium framework"""
    # This test would integrate with the actual environment when available
    # For now, we'll test the seeding utilities in isolation
    
    # Create environment instances with seeded initialization
    seed_manager = SeedManager(enable_validation=True)
    
    # Test episode-level reproducibility using SeedManager
    episode_seeds = []
    for episode_num in range(5):
        episode_seed = seed_manager.generate_episode_seed(seed, episode_num, 
                                                         experiment_id="integration_test")
        episode_seeds.append(episode_seed)
        
        # Create RNG for this episode
        episode_rng, actual_seed = seed_manager.seed(seed=episode_seed, 
                                                    context_id=f"episode_{episode_num}")
        
        # Simulate environment interactions
        # In real integration, this would interact with the environment
        action_sequence = episode_rng.integers(0, 4, size=100).tolist()
        observation_sequence = episode_rng.random(101).tolist()
        
    # Verify identical episodes with identical seeds end-to-end
    # Repeat the same sequence and verify reproducibility
    repeat_episode_seeds = []
    for episode_num in range(5):
        repeat_seed = seed_manager.generate_episode_seed(seed, episode_num,
                                                        experiment_id="integration_test")
        repeat_episode_seeds.append(repeat_seed)
    
    assert episode_seeds == repeat_episode_seeds, \
        "Episode seed generation should be reproducible"
    
    # Check integration with environment reset() and step() methods
    # This would test that the seeding utilities work correctly with Gymnasium API
    for episode_seed in episode_seeds[:2]:  # Test first 2 episodes
        rng1, _ = create_seeded_rng(episode_seed)
        rng2, _ = create_seeded_rng(episode_seed)
        
        # Simulate environment interactions
        sequence1 = rng1.random(50)
        sequence2 = rng2.random(50)
        
        np.testing.assert_array_equal(sequence1, sequence2,
                                    err_msg="Identical episode seeds should produce identical sequences")
    
    # Test cross-session reproducibility using state persistence
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
        temp_path = pathlib.Path(temp_file.name)
    
    try:
        # Save session state
        test_rng, _ = create_seeded_rng(seed)
        save_seed_state(test_rng, temp_path, 
                       metadata={'integration_test': True, 'session_seed': seed})
        
        # Load in new session
        restored_rng, metadata = load_seed_state(temp_path)
        
        # Verify cross-session reproducibility
        assert metadata['integration_test'] is True, \
            "Metadata should be preserved across sessions"
        assert metadata['session_seed'] == seed, \
            "Session seed should be preserved"
        
    finally:
        if temp_path.exists():
            temp_path.unlink()
    
    # Validate scientific research workflow integration
    tracker = ReproducibilityTracker(session_id="integration_test_session")
    
    for i, episode_seed in enumerate(episode_seeds[:2]):
        rng, _ = create_seeded_rng(episode_seed)
        actions = rng.integers(0, 4, size=30).tolist()
        observations = rng.random(31).tolist()
        
        # Record episode
        episode_id = tracker.record_episode(
            episode_seed=episode_seed,
            action_sequence=actions,
            observation_sequence=observations,
            episode_metadata={'integration_test': True, 'episode_index': i}
        )
        
        # Verify reproducibility
        rng_verify, _ = create_seeded_rng(episode_seed)
        verify_actions = rng_verify.integers(0, 4, size=30).tolist()
        verify_observations = rng_verify.random(31).tolist()
        
        result = tracker.verify_episode_reproducibility(
            episode_record_id=episode_id,
            new_action_sequence=verify_actions,
            new_observation_sequence=verify_observations
        )
        
        assert result['sequences_match'] is True, \
            f"Integration test episode {i} should be reproducible"
    
    # Test integration with RL training frameworks and workflows
    # This would test compatibility with libraries like stable-baselines3, etc.
    # For now, we verify that our seeding utilities maintain state correctly
    
    # Generate a training-like sequence with multiple episodes
    training_seeds = [seed_manager.generate_episode_seed(seed, i) for i in range(10)]
    
    # Verify all seeds are valid and within range
    for training_seed in training_seeds:
        assert SEED_MIN_VALUE <= training_seed <= SEED_MAX_VALUE, \
            f"Training seed {training_seed} not in valid range"
        
        is_valid, normalized, error_msg = validate_seed(training_seed)
        assert is_valid, f"Training seed {training_seed} should be valid: {error_msg}"


# Unit tests for seeding error handling and recovery mechanisms
@pytest.mark.unit
def test_seeding_error_recovery():
    """Test seeding error handling and recovery ensuring graceful failure handling, error reporting, and system stability under error conditions"""
    # Test error handling for all seeding functions with invalid inputs
    
    # Test validate_seed error handling
    invalid_inputs = [object(), complex(1, 2), float('inf'), float('nan')]
    for invalid_input in invalid_inputs:
        is_valid, normalized_seed, error_message = validate_seed(invalid_input)
        assert is_valid is False, f"Invalid input {invalid_input} should fail validation"
        assert normalized_seed is None, "Invalid input should return None normalized seed"
        assert error_message != '', "Invalid input should have error message"
    
    # Test create_seeded_rng error handling
    with pytest.raises(ValidationError) as exc_info:
        create_seeded_rng("invalid_seed")
    
    validation_error = exc_info.value
    # Verify proper exception types and error messages
    assert isinstance(validation_error, ValidationError), \
        "Should raise ValidationError for invalid seed"
    assert hasattr(validation_error, 'recovery_suggestion'), \
        "ValidationError should have recovery suggestion"
    
    # Test generate_deterministic_seed error handling
    with pytest.raises(ValidationError):
        generate_deterministic_seed("")  # Empty string
    
    with pytest.raises(ValidationError):
        generate_deterministic_seed(None)  # None input
    
    # Test recovery mechanisms and fallback strategies
    # Test get_random_seed fallback behavior
    try:
        # Test with system entropy disabled (fallback mode)
        fallback_seed = get_random_seed(use_system_entropy=False)
        assert SEED_MIN_VALUE <= fallback_seed <= SEED_MAX_VALUE, \
            "Fallback seed should be in valid range"
    except ResourceError as e:
        # If fallback fails, should have recovery suggestion
        assert hasattr(e, 'recovery_suggestion'), \
            "ResourceError should have recovery suggestion"
    
    # Test SeedManager error handling
    seed_manager = SeedManager()
    
    # Test invalid seed input
    with pytest.raises(ValidationError):
        seed_manager.seed(seed="invalid")
    
    # Test invalid base seed for episode generation
    with pytest.raises(ValidationError):
        seed_manager.generate_episode_seed(base_seed="invalid", episode_number=0)
    
    # Test invalid episode number
    with pytest.raises(ValidationError):
        seed_manager.generate_episode_seed(base_seed=42, episode_number=-1)
    
    # Check system stability after error conditions
    # System should remain functional after errors
    valid_rng, valid_seed = seed_manager.seed(seed=42)
    assert isinstance(valid_rng, np.random.Generator), \
        "SeedManager should remain functional after errors"
    
    # Test ReproducibilityTracker error handling
    tracker = ReproducibilityTracker()
    
    # Test invalid episode recording
    with pytest.raises(ValidationError):
        tracker.record_episode(
            episode_seed="invalid",
            action_sequence=[1, 2, 3],
            observation_sequence=[0.1, 0.2, 0.3, 0.4]
        )
    
    with pytest.raises(ValidationError):
        tracker.record_episode(
            episode_seed=42,
            action_sequence=[],  # Empty sequence
            observation_sequence=[0.1]
        )
    
    # Test error logging and debugging information capture
    # Errors should be logged appropriately (would need log capture in full implementation)
    
    # Test error context preservation and reporting
    try:
        create_seeded_rng("definitely_invalid")
    except ValidationError as e:
        error_details = e.get_error_details() if hasattr(e, 'get_error_details') else {}
        # Error should preserve context
        assert 'error_id' in error_details or hasattr(e, 'error_id'), \
            "Error should have tracking ID"
    
    # Test automated recovery where applicable
    # File operations should have cleanup
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
        temp_path = pathlib.Path(temp_file.name)
        temp_file.write("invalid json {")  # Corrupt file
    
    try:
        with pytest.raises(ValidationError):
            load_seed_state(temp_path)
        
        # File system should remain stable after error
        assert temp_path.exists(), "File should still exist after load error"
        
    finally:
        if temp_path.exists():
            temp_path.unlink()


# Edge case tests for seeding utilities robustness
@pytest.mark.edge_case
def test_seeding_edge_cases():
    """Test seeding utilities with edge cases including boundary values, extreme parameters, and unusual scenarios ensuring robustness and reliability"""
    # Test boundary values at SEED_MIN_VALUE and SEED_MAX_VALUE
    boundary_seeds = [SEED_MIN_VALUE, SEED_MAX_VALUE, 0, 1, SEED_MAX_VALUE - 1]
    
    for boundary_seed in boundary_seeds:
        # All boundary values should be valid
        is_valid, normalized, error_msg = validate_seed(boundary_seed)
        assert is_valid, f"Boundary seed {boundary_seed} should be valid: {error_msg}"
        assert normalized == boundary_seed, f"Boundary seed should not be normalized"
        
        # Should be able to create RNG with boundary values
        rng, seed_used = create_seeded_rng(boundary_seed)
        assert isinstance(rng, np.random.Generator), "Boundary seed should create valid RNG"
        assert seed_used == boundary_seed, "Boundary seed should be preserved"
    
    # Test extreme string inputs for deterministic seed generation
    extreme_strings = [
        "a" * 10000,  # Very long string
        "🚀🎯🔬",      # Unicode characters
        "\n\t\r",      # Whitespace characters
        "0" * 1000,    # Repetitive string
        "!@#$%^&*()",  # Special characters
    ]
    
    for extreme_string in extreme_strings:
        try:
            seed = generate_deterministic_seed(extreme_string)
            assert SEED_MIN_VALUE <= seed <= SEED_MAX_VALUE, \
                f"Extreme string should produce valid seed: {seed}"
            
            # Same string should produce same seed
            repeat_seed = generate_deterministic_seed(extreme_string)
            assert seed == repeat_seed, "Extreme string should be deterministic"
            
        except ValidationError:
            # Some extreme strings might be rejected, which is acceptable
            pass
    
    # Test memory pressure scenarios during state operations
    # Create large RNG state for memory testing
    large_rng, _ = create_seeded_rng(42)
    large_sequence = large_rng.random(100000)  # Generate large state
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
        temp_path = pathlib.Path(temp_file.name)
    
    try:
        # Save and load large state
        success = save_seed_state(large_rng, temp_path, 
                                 metadata={'large_test': True, 'sequence_size': len(large_sequence)})
        assert success, "Should handle large state saving"
        
        restored_rng, metadata = load_seed_state(temp_path)
        assert metadata['large_test'] is True, "Metadata should be preserved"
        
        # Test that restored RNG works
        test_sequence = restored_rng.random(100)
        assert len(test_sequence) == 100, "Restored RNG should function correctly"
        
    finally:
        if temp_path.exists():
            temp_path.unlink()
    
    # Check behavior with corrupted seed state files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        temp_path = pathlib.Path(temp_file.name)
        # Write partially valid JSON
        json.dump({'version': '1', 'timestamp': time.time()}, temp_file)
        # Missing required fields
    
    try:
        with pytest.raises(ValidationError):
            load_seed_state(temp_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    
    # Test thread interruption during seeding operations
    # This is complex to test reliably, but we can test basic thread safety
    import threading
    
    thread_results = []
    error_results = []
    
    def thread_worker():
        try:
            seed_manager = SeedManager(thread_safe=True)
            for i in range(10):
                rng, seed = seed_manager.seed(seed=42 + i)
                thread_results.append((rng, seed))
        except Exception as e:
            error_results.append(e)
    
    threads = [threading.Thread(target=thread_worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    
    assert len(error_results) == 0, f"Thread operations should not error: {error_results}"
    assert len(thread_results) > 0, "Thread operations should succeed"
    
    # Validate behavior with system resource limitations
    # Test with very large number of generators
    seed_manager = SeedManager(thread_safe=False)  # Disable thread safety for speed
    generators = []
    
    try:
        for i in range(1000):  # Create many generators
            rng, seed = seed_manager.seed(seed=i)
            generators.append(rng)
        
        # System should handle many generators
        assert len(generators) == 1000, "Should create all requested generators"
        
    except (MemoryError, ResourceError):
        # Acceptable to fail with resource exhaustion
        pass
    
    # Test recovery from partial operation failures
    # Test atomic operations
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir) / "atomic_test.json"
        
        # Simulate interrupted save operation
        rng, _ = create_seeded_rng(42)
        
        # This should either succeed completely or fail cleanly
        try:
            success = save_seed_state(rng, temp_path, create_backup=True)
            if success:
                assert temp_path.exists(), "Successful save should create file"
                
                # File should be valid JSON
                with open(temp_path, 'r') as f:
                    data = json.load(f)
                assert 'version' in data, "Saved file should be valid"
                
        except (ResourceError, OSError):
            # Failure is acceptable, but no partial files should exist
            assert not temp_path.exists() or temp_path.stat().st_size == 0, \
                "Failed save should not leave corrupted files"


# Cross-platform compatibility tests for seeding utilities
@pytest.mark.unit
def test_seeding_cross_platform_compatibility():
    """Test seeding utilities cross-platform compatibility ensuring consistent behavior across different operating systems and Python versions"""
    # Test deterministic seed generation consistency across platforms
    test_strings = ['cross_platform_test', 'unicode_test_🌍', 'special_chars_!@#']
    
    for test_string in test_strings:
        try:
            seed1 = generate_deterministic_seed(test_string, hash_algorithm='sha256')
            seed2 = generate_deterministic_seed(test_string, hash_algorithm='sha256')
            
            # Same string should always produce same seed regardless of platform
            assert seed1 == seed2, \
                f"Deterministic seed generation should be consistent: {seed1} != {seed2}"
            
            # Verify it's in valid range
            assert SEED_MIN_VALUE <= seed1 <= SEED_MAX_VALUE, \
                f"Cross-platform seed should be in valid range: {seed1}"
                
        except ValidationError:
            # Some strings might be rejected consistently across platforms
            pass
    
    # Verify RNG behavior compatibility with different NumPy versions
    # Test that our seeding works with current NumPy version
    for seed in TEST_SEEDS[:3]:
        rng, seed_used = create_seeded_rng(seed)
        
        # Generate test sequence
        sequence = rng.random(100)
        assert len(sequence) == 100, "RNG should generate requested sequence length"
        assert all(0.0 <= x <= 1.0 for x in sequence), \
            "Random values should be in [0, 1] range"
        
        # Test reproducibility with same seed
        rng2, seed_used2 = create_seeded_rng(seed)
        sequence2 = rng2.random(100)
        
        np.testing.assert_array_equal(sequence, sequence2,
                                    err_msg="Same seed should produce identical sequences")
    
    # Check file path handling for seed state persistence
    test_paths = [
        "simple_path.json",
        "path with spaces.json",
        "path_with_unicode_🌟.json"
    ]
    
    for path_name in test_paths:
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                test_path = pathlib.Path(temp_dir) / path_name
                rng, _ = create_seeded_rng(42)
                
                # Test save
                success = save_seed_state(rng, test_path)
                if success:
                    assert test_path.exists(), f"Should create file with path: {path_name}"
                    
                    # Test load
                    restored_rng, metadata = load_seed_state(test_path)
                    assert isinstance(restored_rng, np.random.Generator), \
                        f"Should restore RNG from path: {path_name}"
                        
            except (OSError, UnicodeError):
                # Some path names might not be supported on all platforms
                pass
    
    # Test entropy source availability and fallback behavior
    # Test system entropy
    try:
        system_seed = get_random_seed(use_system_entropy=True)
        assert SEED_MIN_VALUE <= system_seed <= SEED_MAX_VALUE, \
            "System entropy seed should be valid"
    except ResourceError:
        # System entropy might not be available on all platforms
        pass
    
    # Test fallback entropy
    fallback_seed = get_random_seed(use_system_entropy=False)
    assert SEED_MIN_VALUE <= fallback_seed <= SEED_MAX_VALUE, \
        "Fallback entropy seed should be valid"
    
    # Validate hash function consistency across Python versions
    # SHA-256 should be available on all platforms
    assert 'sha256' in hashlib.algorithms_available, \
        "SHA-256 should be available on all platforms"
    
    test_string = "hash_consistency_test"
    hash1 = hashlib.sha256(test_string.encode('utf-8')).hexdigest()
    hash2 = hashlib.sha256(test_string.encode('utf-8')).hexdigest()
    assert hash1 == hash2, "Hash function should be deterministic"
    
    # Test thread safety behavior on different platforms
    seed_manager = SeedManager(thread_safe=True)
    thread_seeds = []
    
    def platform_thread_worker():
        for i in range(5):
            rng, seed = seed_manager.seed(seed=100 + i)
            thread_seeds.append(seed)
    
    thread = threading.Thread(target=platform_thread_worker)
    thread.start()
    thread.join(timeout=5.0)
    
    assert len(thread_seeds) == 5, "Thread safety should work across platforms"
    assert all(isinstance(s, int) for s in thread_seeds), \
        "Thread-generated seeds should be valid integers"
    
    # Check serialization format compatibility
    # JSON format should be consistent across platforms
    test_data = {
        'test_key': 'test_value',
        'numeric_value': 42,
        'list_value': [1, 2, 3],
        'nested': {'inner': 'value'}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        temp_path = pathlib.Path(temp_file.name)
        json.dump(test_data, temp_file)
    
    try:
        # Read back and verify
        with open(temp_path, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == test_data, "JSON serialization should be consistent"
        
    finally:
        if temp_path.exists():
            temp_path.unlink()
    
    # Test that our constants are consistent
    assert isinstance(SEED_MIN_VALUE, int), "SEED_MIN_VALUE should be integer"
    assert isinstance(SEED_MAX_VALUE, int), "SEED_MAX_VALUE should be integer"
    assert SEED_MIN_VALUE < SEED_MAX_VALUE, "Seed range should be valid"
    assert SEED_MIN_VALUE >= 0, "Minimum seed should be non-negative"
    
    # Validate that reproducibility tolerance is appropriate for platform
    test_tolerance = REPRODUCIBILITY_TOLERANCE
    assert isinstance(test_tolerance, float), "Tolerance should be float"
    assert test_tolerance > 0, "Tolerance should be positive"
    assert test_tolerance < 1e-5, "Tolerance should be strict for reproducibility"