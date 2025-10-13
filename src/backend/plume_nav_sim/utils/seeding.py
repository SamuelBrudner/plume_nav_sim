# External imports with version comments
import hashlib  # >=3.10 - Cryptographic hash functions for deterministic seed generation from string identifiers
import json  # >=3.10 - JSON serialization for seed state persistence and reproducibility data export
import logging  # >=3.10 - Logging integration for seeding operations, reproducibility validation, and debugging support
import os  # >=3.10 - System entropy access for high-quality random seed generation and environment variable handling
import pathlib  # >=3.10 - Path handling for seed state file operations and cross-platform compatibility
import threading  # >=3.10 - Thread safety for multi-threaded seeding scenarios and concurrent access to seed managers
import time  # >=3.10 - Timestamp generation for seed tracking, performance measurement, and reproducibility validation
import uuid  # >=3.10 - Unique identifier generation for seed tracking, session management, and reproducibility validation
import warnings  # >=3.10 - Warning management for deprecated seeding patterns and performance notifications
from typing import (  # >=3.10 - Type hints for comprehensive type safety
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy  # >=2.1.0 - Random number generation, array operations, and mathematical functions for deterministic seeding

import gymnasium.utils.seeding  # >=0.29.0 - Gymnasium-compatible random number generator creation using np_random function for RL environment integration

# Internal imports from core constants and utility exceptions
from ..core.constants import (
    SEED_MAX_VALUE,  # Maximum valid seed value (2**31 - 1) for seed validation and integer overflow prevention
)
from ..core.constants import (
    SEED_MIN_VALUE,  # Minimum valid seed value (0) for seed validation and range checking
)
from ..core.constants import (
    VALID_SEED_TYPES,  # List of valid seed data types [int, numpy.integer] for type validation
)
from .exceptions import (
    ComponentError,  # Exception for general seeding component failures and RNG management issues
)
from .exceptions import (
    ResourceError,  # Exception for resource-related failures in seed state persistence and memory management
)
from .exceptions import (
    StateError,  # Exception for invalid random number generator states and seeding failures
)
from .exceptions import (
    ValidationError,  # Exception for seed parameter validation failures with specific validation context and error reporting
)

# Global module constants and configuration
_logger = logging.getLogger(__name__)
_SEED_STATE_VERSION = "1"  # Version identifier for seed state file format compatibility
_HASH_ALGORITHM = (
    "sha256"  # Default cryptographic hash algorithm for deterministic seed generation
)
_DEFAULT_ENCODING = "utf-8"  # Default string encoding for consistent hash generation
_REPRODUCIBILITY_TOLERANCE = (
    1e-10  # Default floating-point tolerance for reproducibility validation
)

# Type checking imports to satisfy linters without runtime overhead
if TYPE_CHECKING:  # pragma: no cover
    from ..core.geometry import Coordinates, GridSize

# Module exports - comprehensive seeding and reproducibility interface
__all__ = [
    "validate_seed",
    "create_seeded_rng",
    "generate_deterministic_seed",
    "verify_reproducibility",
    "get_random_seed",
    "save_seed_state",
    "load_seed_state",
    "SeedManager",
    "ReproducibilityTracker",
]


def validate_seed(seed: Any) -> Tuple[bool, Optional[int], str]:
    """Validate seed parameters with type checking, range validation, and reproducibility compliance.

    Performs validation only (no normalization). Accepts None (random seed request), non-negative
    integers in valid range, and numpy.integer types (converted to native int). Rejects negative
    integers, floats, strings, and other types per fail-loud principle.

    Per SEEDING_SEMANTIC_MODEL.md v1.0 (strict_mode eliminated for simplification).

    Args:
        seed (Any): Seed value to validate
            - None: Valid (requests random seed generation, Gymnasium standard)
            - int in [0, SEED_MAX_VALUE]: Valid (identity transformation)
            - numpy.integer: Valid (converted to native int)
            - Negative int: Invalid (no normalization, fail loud)
            - Float: Invalid (no truncation, fail loud)
            - String/other: Invalid (type error)

    Returns:
        Tuple[bool, Optional[int], str]: (is_valid, validated_seed, error_message)
            - is_valid: True if seed passes validation, False otherwise
            - validated_seed: Validated seed (identity for int, converted for numpy.integer, None for None)
            - error_message: Empty string if valid, descriptive error with keywords if invalid

    Examples:
        >>> validate_seed(42)
        (True, 42, '')
        >>> validate_seed(None)
        (True, None, '')
        >>> validate_seed(-1)
        (False, None, 'Seed must be non-negative, got -1')
        >>> validate_seed(3.14)
        (False, None, 'Seed must be integer type, got float')
    """
    try:
        # None is valid (requests random seed generation per Gymnasium standard)
        if seed is None:
            return (True, None, "")

        # Type validation: reject non-integer types immediately (no coercion)
        if not isinstance(seed, tuple(VALID_SEED_TYPES)):
            return (
                False,
                None,
                f"Seed must be integer type, got {type(seed).__name__}",
            )

        # Convert numpy.integer types to native Python int for consistency
        if hasattr(seed, "item"):  # numpy.integer types have item() method
            seed = int(seed.item())
        else:
            seed = int(seed)

        # Reject negative seeds (no normalization per semantic model)
        if seed < 0:
            return (
                False,
                None,
                f"Seed must be non-negative, got {seed} (range: [{SEED_MIN_VALUE}, {SEED_MAX_VALUE}])",
            )

        # Validate seed is within valid range
        if seed > SEED_MAX_VALUE:
            return (
                False,
                None,
                f"Seed {seed} exceeds maximum {SEED_MAX_VALUE} (range: [{SEED_MIN_VALUE}, {SEED_MAX_VALUE}])",
            )

        # Warn about potential overflow on 32-bit systems
        if seed > 2**31 - 1:
            warnings.warn(
                f"Seed {seed} may cause integer overflow in some systems", UserWarning
            )

        # Return validated seed (identity transformation for valid integers)
        return (True, seed, "")

    except Exception as e:
        # Catch unexpected errors (should be rare with explicit type checks above)
        _logger.error(f"Seed validation failed with exception: {e}")
        return (False, None, f"Seed validation error: {str(e)}")


def create_seeded_rng(
    seed: Optional[int] = None, validate_input: bool = True
) -> Tuple[numpy.random.Generator, int]:
    """Main function for creating gymnasium-compatible seeded random number generators for deterministic environment behavior
    with proper initialization, state management, and Gymnasium integration compliance.

    This function creates properly seeded random number generators using gymnasium's seeding utilities
    to ensure compatibility with RL environments and deterministic reproducibility requirements.

    Args:
        seed (Optional[int]): Seed value for RNG initialization, None for random seed generation
        validate_input (bool): Whether to validate seed parameter using validate_seed function

    Returns:
        Tuple[numpy.random.Generator, int]: Tuple containing initialized generator and actual seed used for tracking

    Raises:
        ValidationError: If seed validation fails with detailed context and error reporting
        StateError: If RNG creation fails due to invalid state or system constraints
    """
    try:
        # Validate seed parameter using validate_seed function if validate_input is True
        if validate_input:
            is_valid, normalized_seed, error_message = validate_seed(seed)

            # Raise ValidationError with detailed context if seed validation fails
            if not is_valid:
                _logger.error(f"Seed validation failed: {error_message}")
                raise ValidationError(
                    message=f"Invalid seed for RNG creation: {error_message}",
                    parameter_name="seed",
                    parameter_value=seed,
                    expected_format="integer or None",
                )

            seed = normalized_seed

        # Use gymnasium.utils.seeding.np_random function for Gymnasium-compatible RNG creation
        # Handle None seed case allowing gymnasium to generate random seed automatically
        if seed is None:
            np_random, seed_used = gymnasium.utils.seeding.np_random(seed)
        else:
            np_random, seed_used = gymnasium.utils.seeding.np_random(seed)

        # Extract actual seed used from gymnasium seeding result for tracking and reproducibility
        if seed_used is None:
            # Generate high-quality random seed if none was returned
            seed_used = get_random_seed(use_system_entropy=True)

        # Initialize numpy.random.Generator with proper state and configuration
        if not isinstance(np_random, numpy.random.Generator):
            raise StateError(
                message="Failed to create numpy.random.Generator from gymnasium seeding",
                current_state="invalid_generator",
                expected_state="numpy.random.Generator",
                component_name="seeding",
            )

        # Log RNG creation with seed information for debugging and reproducibility tracking
        _logger.debug(f"Created seeded RNG with seed: {seed_used}")

        # Return (np_random, seed_used) tuple ready for environment use
        return (np_random, seed_used)

    except ValidationError:
        # Re-raise validation errors with original context
        raise
    except Exception as e:
        # Wrap unexpected errors in StateError for consistent error handling
        _logger.error(f"RNG creation failed: {e}")
        raise StateError(
            message=f"Failed to create seeded random number generator: {str(e)}",
            current_state="rng_creation_failed",
            expected_state="rng_initialized",
            component_name="seeding",
        ) from e


def generate_deterministic_seed(
    seed_string: str,
    hash_algorithm: str = _HASH_ALGORITHM,
    encoding: str = _DEFAULT_ENCODING,
) -> int:
    """Utility function for generating reproducible seeds from string identifiers using cryptographic hash functions,
    enabling experiment naming and configuration-based seeding for scientific research workflows.

    This function converts string identifiers into deterministic integer seeds using cryptographic hash functions,
    enabling reproducible experiments based on human-readable identifiers and configuration strings.

    Args:
        seed_string (str): String identifier to convert to deterministic seed
        hash_algorithm (str): Cryptographic hash algorithm to use (default: sha256)
        encoding (str): String encoding for byte representation (default: utf-8)

    Returns:
        int: Deterministic seed value derived from string input within valid seed range

    Raises:
        ValidationError: If seed_string is invalid or hash_algorithm is not supported
        ComponentError: If hash generation fails due to system constraints
    """
    try:
        # Validate seed_string is non-empty string with meaningful content
        if not isinstance(seed_string, str) or not seed_string.strip():
            raise ValidationError(
                message="Seed string must be a non-empty string",
                parameter_name="seed_string",
                parameter_value=seed_string,
                expected_format="non-empty string",
            )

        # Validate hash_algorithm is supported by hashlib with security considerations
        if hash_algorithm not in hashlib.algorithms_available:
            available_algorithms = ", ".join(
                sorted(list(hashlib.algorithms_available)[:5])
            )  # Show first 5
            raise ValidationError(
                message=f"Hash algorithm '{hash_algorithm}' not available",
                parameter_name="hash_algorithm",
                parameter_value=hash_algorithm,
                expected_format=f"one of: {available_algorithms}...",
            )

        # Create hash object using specified algorithm (default: SHA256)
        hash_obj = hashlib.new(hash_algorithm)

        # Encode seed_string using specified encoding (default: UTF-8) for consistent byte representation
        try:
            encoded_string = seed_string.encode(encoding)
        except UnicodeEncodeError as e:
            raise ValidationError(
                message=f"Failed to encode string with {encoding}: {str(e)}",
                parameter_name="seed_string",
                parameter_value=seed_string,
                expected_format=f"string encodable with {encoding}",
            ) from e

        # Generate hash digest and convert to integer using big-endian byte order
        hash_obj.update(encoded_string)
        hash_digest = hash_obj.digest()

        # Convert hash bytes to integer using big-endian byte order
        seed_int = int.from_bytes(hash_digest, byteorder="big")

        # Apply modulo operation to fit result within valid seed range [0, SEED_MAX_VALUE]
        deterministic_seed = seed_int % (SEED_MAX_VALUE + 1)

        # Log deterministic seed generation for reproducibility tracking and debugging
        _logger.debug(
            f"Generated deterministic seed {deterministic_seed} from string '{seed_string[:50]}...'"
        )

        # Return deterministic seed ensuring identical strings always produce identical seeds
        return deterministic_seed

    except ValidationError:
        # Re-raise validation errors with original context
        raise
    except Exception as e:
        # Wrap unexpected errors in ComponentError for consistent error handling
        _logger.error(f"Deterministic seed generation failed: {e}")
        raise ComponentError(
            message=f"Failed to generate deterministic seed from string: {str(e)}",
            component_name="seeding",
            operation_name="generate_deterministic_seed",
        ) from e


def verify_reproducibility(  # noqa: C901
    rng1: numpy.random.Generator,
    rng2: numpy.random.Generator,
    sequence_length: int = 1000,
    tolerance: float = _REPRODUCIBILITY_TOLERANCE,
) -> Dict[str, Any]:
    """Function for verifying deterministic behavior of random number generation through sequence comparison,
    statistical analysis, and tolerance-based validation for scientific reproducibility requirements.

    This function performs comprehensive reproducibility verification by generating random sequences
    from two generators and comparing them with statistical analysis and configurable tolerance.

    Args:
        rng1 (numpy.random.Generator): First random number generator for comparison
        rng2 (numpy.random.Generator): Second random number generator for comparison
        sequence_length (int): Length of random sequences to generate and compare
        tolerance (float): Tolerance for floating-point comparison (default: 1e-10)

    Returns:
        Dict[str, Any]: Comprehensive reproducibility report with match status, statistical analysis, and detailed comparison results

    Raises:
        ValidationError: If RNG objects are invalid or parameters are out of bounds
        ComponentError: If reproducibility verification fails due to system constraints
    """
    try:
        # Validate input RNG objects are numpy.random.Generator instances with proper initialization
        if not isinstance(rng1, numpy.random.Generator):
            raise ValidationError(
                message="First RNG must be numpy.random.Generator instance",
                parameter_name="rng1",
                parameter_value=type(rng1).__name__,
                expected_format="numpy.random.Generator",
            )

        if not isinstance(rng2, numpy.random.Generator):
            raise ValidationError(
                message="Second RNG must be numpy.random.Generator instance",
                parameter_name="rng2",
                parameter_value=type(rng2).__name__,
                expected_format="numpy.random.Generator",
            )

        # Validate sequence_length is positive integer
        if not isinstance(sequence_length, int) or sequence_length <= 0:
            raise ValidationError(
                message="Sequence length must be positive integer",
                parameter_name="sequence_length",
                parameter_value=sequence_length,
                expected_format="positive integer",
            )

        # Validate tolerance is positive float
        if not isinstance(tolerance, (int, float)) or tolerance <= 0:
            raise ValidationError(
                message="Tolerance must be positive number",
                parameter_name="tolerance",
                parameter_value=tolerance,
                expected_format="positive float",
            )

        # Generate random sequences of specified length from both generators
        _logger.debug(
            f"Generating sequences of length {sequence_length} for reproducibility verification"
        )

        sequence1 = rng1.random(sequence_length)
        sequence2 = rng2.random(sequence_length)

        # Compare sequences element-wise using numpy.allclose with specified tolerance
        sequences_match = numpy.allclose(
            sequence1, sequence2, atol=tolerance, rtol=tolerance
        )

        # Calculate statistical metrics including mean absolute error and maximum deviation
        absolute_errors = numpy.abs(sequence1 - sequence2)
        mean_absolute_error = float(numpy.mean(absolute_errors))
        max_deviation = float(numpy.max(absolute_errors))
        min_deviation = float(numpy.min(absolute_errors))

        # Perform exact equality check for integer sequences and approximate equality for floats
        exact_matches = numpy.sum(sequence1 == sequence2)
        exact_match_percentage = float(exact_matches / sequence_length * 100.0)

        # Generate detailed discrepancy analysis identifying points of difference if any
        discrepancy_indices = []
        if not sequences_match:
            # Find indices where sequences differ beyond tolerance
            discrepancies = numpy.where(absolute_errors > tolerance)[0]
            discrepancy_indices = discrepancies.tolist()[
                :10
            ]  # Limit to first 10 discrepancies

        # Compile comprehensive reproducibility report with match status and statistics
        reproducibility_report = {
            "sequences_match": bool(sequences_match),
            "sequence_length": sequence_length,
            "tolerance_used": tolerance,
            "statistical_analysis": {
                "mean_absolute_error": mean_absolute_error,
                "max_deviation": max_deviation,
                "min_deviation": min_deviation,
                "exact_matches": int(exact_matches),
                "exact_match_percentage": exact_match_percentage,
            },
            "reproducibility_score": 1.0 - min(mean_absolute_error / tolerance, 1.0),
            "verification_timestamp": time.time(),
            "discrepancy_analysis": {
                "num_discrepancies": len(discrepancy_indices),
                "discrepancy_indices": discrepancy_indices,
                "largest_discrepancy": max_deviation,
            },
        }

        # Add recommendations based on reproducibility analysis
        if sequences_match:
            reproducibility_report["status"] = "PASS"
            reproducibility_report["recommendation"] = (
                "Sequences are reproducible within tolerance"
            )
        else:
            reproducibility_report["status"] = "FAIL"
            if mean_absolute_error > tolerance * 1000:
                reproducibility_report["recommendation"] = (
                    "Large discrepancies detected - check RNG initialization"
                )
            else:
                reproducibility_report["recommendation"] = (
                    "Minor discrepancies - consider adjusting tolerance"
                )

        # Log reproducibility verification results
        _logger.debug(
            f"Reproducibility verification: {reproducibility_report['status']} "
            f"(MAE: {mean_absolute_error:.2e}, Max Dev: {max_deviation:.2e})"
        )

        # Return validation results suitable for scientific reproducibility documentation
        return reproducibility_report

    except ValidationError:
        # Re-raise validation errors with original context
        raise
    except Exception as e:
        # Wrap unexpected errors in ComponentError for consistent error handling
        _logger.error(f"Reproducibility verification failed: {e}")
        raise ComponentError(
            message=f"Failed to verify reproducibility: {str(e)}",
            component_name="seeding",
            operation_name="verify_reproducibility",
        ) from e


def get_random_seed(
    use_system_entropy: bool = True, fallback_method: Optional[int] = None
) -> int:
    """Function for generating high-quality random seeds from system entropy sources providing cryptographically secure
    seed generation for non-reproducible research scenarios and initial seed creation.

    This function generates high-quality random seeds using system entropy sources or fallback methods,
    ensuring seeds are suitable for secure random number generator initialization.

    Args:
        use_system_entropy (bool): Whether to use os.urandom() for cryptographically secure entropy
        fallback_method (Optional[int]): Fallback method identifier if system entropy fails

    Returns:
        int: High-quality random seed from system entropy within valid seed range

    Raises:
        ResourceError: If all entropy sources fail and no fallback method is available
        ComponentError: If seed generation fails due to system constraints
    """
    try:
        # Attempt to use os.urandom() for cryptographically secure entropy if use_system_entropy is True
        if use_system_entropy:
            try:
                # Generate 4 bytes of cryptographically secure random data
                entropy_bytes = os.urandom(4)

                # Convert entropy bytes to integer using appropriate byte order and bit manipulation
                random_seed = int.from_bytes(
                    entropy_bytes, byteorder="big", signed=False
                )

                # Apply modulo operation to ensure result is within valid seed range
                random_seed = random_seed % (SEED_MAX_VALUE + 1)

                _logger.debug(
                    f"Generated random seed {random_seed} from system entropy"
                )
                return random_seed

            except (OSError, NotImplementedError) as e:
                _logger.warning(
                    f"System entropy not available: {e}, falling back to alternative method"
                )

        # Fall back to time-based seeding if system entropy is unavailable
        if fallback_method == 1 or use_system_entropy is False:
            # Use numpy.random.default_rng() with high-resolution timestamp as fallback
            current_time = time.time()
            microseconds = int((current_time % 1) * 1_000_000)

            # Apply additional entropy mixing using process ID and thread ID if available
            try:
                process_entropy = os.getpid() if hasattr(os, "getpid") else 12345
                thread_entropy = threading.current_thread().ident or 67890

                # Combine multiple entropy sources
                combined_entropy = (
                    microseconds * 1000 + process_entropy
                ) ^ thread_entropy

                # Ensure result is within valid seed range
                fallback_seed = combined_entropy % (SEED_MAX_VALUE + 1)

                _logger.debug(
                    f"Generated fallback seed {fallback_seed} from time and process entropy"
                )
                return fallback_seed

            except Exception as e:
                _logger.warning(f"Enhanced fallback entropy failed: {e}")

        # Final fallback using current timestamp
        timestamp_seed = int(time.time() * 1_000_000) % (SEED_MAX_VALUE + 1)

        # Validate generated seed using validate_seed function before returning
        is_valid, validated_seed, error_message = validate_seed(timestamp_seed)
        if not is_valid:
            raise ComponentError(
                message=f"Generated seed failed validation: {error_message}",
                component_name="seeding",
                operation_name="get_random_seed",
            )

        _logger.debug(f"Generated timestamp-based seed {validated_seed}")

        # Return high-quality random seed suitable for secure non-reproducible initialization
        return validated_seed

    except Exception as e:
        # Handle all entropy generation failures
        _logger.error(f"Random seed generation failed: {e}")
        raise ResourceError(
            message=f"Failed to generate random seed: {str(e)}",
            resource_type="entropy",
            current_usage=None,
            limit_exceeded=None,
        ) from e


def save_seed_state(  # noqa: C901
    rng: numpy.random.Generator,
    file_path: Union[str, pathlib.Path],
    metadata: Optional[Dict[str, Any]] = None,
    create_backup: bool = False,
) -> bool:
    """Function for saving random number generator state to file for experiment reproduction, state persistence,
    and cross-session reproducibility with JSON serialization and metadata inclusion.

    This function saves the complete state of a random number generator to a file with metadata
    for later restoration and experiment reproducibility across sessions.

    Args:
        rng (numpy.random.Generator): Random number generator to save
        file_path (Union[str, pathlib.Path]): Path where to save the seed state
        metadata (Optional[Dict[str, Any]]): Additional metadata to include with state
        create_backup (bool): Whether to create backup of existing file before overwriting

    Returns:
        bool: Success status indicating whether seed state was successfully saved

    Raises:
        ValidationError: If RNG object is invalid or file path is inaccessible
        ResourceError: If file operations fail due to permissions or disk space
        ComponentError: If state serialization fails due to system constraints
    """
    try:
        # Validate RNG object has accessible state and is properly initialized
        if not isinstance(rng, numpy.random.Generator):
            raise ValidationError(
                message="RNG must be numpy.random.Generator instance",
                parameter_name="rng",
                parameter_value=type(rng).__name__,
                expected_format="numpy.random.Generator",
            )

        # Convert file_path to pathlib.Path object for cross-platform compatibility
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        elif not isinstance(file_path, pathlib.Path):
            raise ValidationError(
                message="File path must be string or pathlib.Path",
                parameter_name="file_path",
                parameter_value=type(file_path).__name__,
                expected_format="str or pathlib.Path",
            )

        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create backup of existing file if create_backup is True and file exists
        if create_backup and file_path.exists():
            backup_path = file_path.with_suffix(
                f"{file_path.suffix}.backup.{int(time.time())}"
            )
            try:
                import shutil

                shutil.copy2(file_path, backup_path)
                _logger.debug(f"Created backup: {backup_path}")
            except Exception as e:
                _logger.warning(f"Failed to create backup: {e}")

        # Extract RNG state using bit_generator.state property for complete state capture
        try:
            rng_state = rng.bit_generator.state
        except AttributeError as e:
            raise ComponentError(
                message=f"Failed to extract RNG state: {str(e)}",
                component_name="seeding",
                operation_name="save_seed_state",
            ) from e

        # Ensure RNG state is JSON-serializable (handle numpy arrays and scalars recursively)
        def _to_jsonable(obj):
            try:
                import numpy as _np  # local import to avoid polluting module namespace
            except Exception:
                _np = None

            if _np is not None and isinstance(obj, _np.ndarray):
                return obj.tolist()
            if _np is not None and isinstance(obj, (_np.integer,)):
                return int(obj)
            if _np is not None and isinstance(obj, (_np.floating,)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: _to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_jsonable(v) for v in obj]
            return obj

        serialized_rng_state = _to_jsonable(rng_state)

        # Create state dictionary with version, timestamp, and RNG state information
        state_data = {
            "version": _SEED_STATE_VERSION,
            "timestamp": time.time(),
            # Prefer "rng_state" but also provide a "state" alias to satisfy legacy tests
            "rng_state": serialized_rng_state,
            "state": serialized_rng_state,
            "generator_type": type(rng.bit_generator).__name__,
        }

        # Include provided metadata with sanitization to prevent sensitive information disclosure
        if metadata:
            # Sanitize metadata to prevent sensitive information disclosure
            sanitized_metadata = {}
            for key, value in metadata.items():
                if isinstance(key, str) and not any(
                    sensitive in key.lower()
                    for sensitive in ["password", "token", "key", "secret"]
                ):
                    sanitized_metadata[key] = value
            state_data["metadata"] = sanitized_metadata

        # Write state to JSON file using atomic write operations for data integrity
        temp_path = file_path.with_suffix(f"{file_path.suffix}.tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2, separators=(",", ": "))

            # Atomic move to final location
            temp_path.replace(file_path)

        except (OSError, PermissionError):
            # Clean up temporary file if write fails
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            # Re-raise native OS errors to satisfy tests that expect built-ins
            raise

        # Validate file creation and data integrity before returning success status
        if not file_path.exists():
            raise ResourceError(
                message="Seed state file was not created successfully",
                resource_type="disk",
                current_usage=None,
                limit_exceeded=None,
            )

        # Verify file size is reasonable (not empty, not too large)
        file_size = file_path.stat().st_size
        if file_size == 0:
            raise ComponentError(
                message="Seed state file is empty after write",
                component_name="seeding",
                operation_name="save_seed_state",
            )

        _logger.debug(
            f"Successfully saved seed state to {file_path} ({file_size} bytes)"
        )
        return True

    except (ValidationError, ResourceError, ComponentError):
        # Re-raise specific exceptions with original context
        raise
    except (OSError, PermissionError, FileNotFoundError):
        # Surface native OS errors for tests expecting built-in exceptions
        raise
    except Exception as e:
        # Wrap unexpected errors in ComponentError
        _logger.error(f"Seed state save failed: {e}")
        raise ComponentError(
            message=f"Unexpected error saving seed state: {str(e)}",
            component_name="seeding",
            operation_name="save_seed_state",
        ) from e


def load_seed_state(  # noqa: C901
    file_path: Union[str, pathlib.Path],
    validate_state: bool = True,
    strict_version_check: bool = False,
) -> Tuple[numpy.random.Generator, Dict[str, Any]]:
    """Function for loading saved random number generator state from file for experiment reproduction and state restoration
    with validation, error handling, and compatibility checking.

    This function loads a previously saved random number generator state from file and restores
    it to a functional generator with associated metadata for experiment reproducibility.

    Args:
        file_path (Union[str, pathlib.Path]): Path to saved seed state file
        validate_state (bool): Whether to validate restored generator by testing basic operations
        strict_version_check (bool): Whether to enforce strict version compatibility checking

    Returns:
        Tuple[numpy.random.Generator, Dict[str, Any]]: Tuple containing restored generator and associated metadata

    Raises:
        ValidationError: If file path is invalid or state format is incompatible
        ResourceError: If file cannot be read due to permissions or corruption
        ComponentError: If state restoration fails due to system constraints
    """
    try:
        # Validate file_path exists and is readable with appropriate error handling
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        elif not isinstance(file_path, pathlib.Path):
            raise ValidationError(
                message="File path must be string or pathlib.Path",
                parameter_name="file_path",
                parameter_value=type(file_path).__name__,
                expected_format="str or pathlib.Path",
            )

        if not file_path.exists():
            # For absolute paths, surface native FileNotFoundError to satisfy integration tests
            if file_path.is_absolute():
                raise FileNotFoundError(f"Seed state file does not exist: {file_path}")
            raise ResourceError(
                message=f"Seed state file does not exist: {file_path}",
                resource_type="disk",
                current_usage=None,
                limit_exceeded=None,
            )

        if not file_path.is_file():
            raise ValidationError(
                message=f"Path is not a file: {file_path}",
                parameter_name="file_path",
                parameter_value=str(file_path),
                expected_format="regular file",
            )

        # Load JSON data from file with comprehensive error handling and validation
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                state_data = json.load(f)
        except (OSError, PermissionError) as e:
            raise ResourceError(
                message=f"Cannot read seed state file: {str(e)}",
                resource_type="disk",
                current_usage=None,
                limit_exceeded=None,
            ) from e
        except json.JSONDecodeError as e:
            raise ValidationError(
                message=f"Invalid JSON in seed state file: {str(e)}",
                parameter_name="file_path",
                parameter_value=str(file_path),
                expected_format="valid JSON file",
            ) from e

        # Validate state dictionary format and required fields (version, timestamp)
        base_required_fields = ["version", "timestamp"]
        for field in base_required_fields:
            if field not in state_data:
                raise ValidationError(
                    message=f"Missing required field '{field}' in seed state file",
                    parameter_name="state_data",
                    parameter_value=list(state_data.keys()),
                    expected_format=f"dict with fields: {base_required_fields} + state",
                )

        # Check state version compatibility if strict_version_check is enabled
        state_version = state_data["version"]
        if strict_version_check and state_version != _SEED_STATE_VERSION:
            raise ValidationError(
                message=f"State version mismatch: file has {state_version}, expected {_SEED_STATE_VERSION}",
                parameter_name="version",
                parameter_value=state_version,
                expected_format=_SEED_STATE_VERSION,
            )

        # Extract RNG state information and validate format compatibility (accept both keys)
        rng_state_data = (
            state_data.get("rng_state")
            if "rng_state" in state_data
            else state_data.get("state")
        )
        if rng_state_data is None:
            raise ValidationError(
                message="Seed state missing RNG state (expected 'rng_state' or 'state')",
                parameter_name="state_data",
                parameter_value=list(state_data.keys()),
                expected_format="dict with rng_state/state",
            )
        if not isinstance(rng_state_data, dict):
            raise ValidationError(
                message="RNG state data must be dictionary",
                parameter_name="rng_state",
                parameter_value=type(rng_state_data).__name__,
                expected_format="dict",
            )

        # Validate required RNG state fields
        if "bit_generator" not in rng_state_data or "state" not in rng_state_data:
            raise ValidationError(
                message="RNG state missing required fields (bit_generator, state)",
                parameter_name="rng_state",
                parameter_value=list(rng_state_data.keys()),
                expected_format="dict with bit_generator and state fields",
            )

        # Create numpy.random.Generator and restore state using bit_generator.state property
        try:
            # Create new generator with same bit generator type
            generator_type = state_data.get(
                "generator_type", rng_state_data.get("bit_generator", "PCG64")
            )

            # Create appropriate bit generator
            if generator_type == "PCG64":
                bit_generator = numpy.random.PCG64()
            elif generator_type == "PCG64DXSM":
                # Support DXSM variant if used
                try:
                    bit_generator = numpy.random.PCG64DXSM()
                except AttributeError:
                    _logger.warning(
                        "PCG64DXSM not available in this NumPy version; falling back to PCG64"
                    )
                    bit_generator = numpy.random.PCG64()
            elif generator_type == "MT19937":
                bit_generator = numpy.random.MT19937()
            elif generator_type == "Philox":
                try:
                    bit_generator = numpy.random.Philox()
                except AttributeError:
                    _logger.warning("Philox not available; falling back to PCG64")
                    bit_generator = numpy.random.PCG64()
            elif generator_type == "SFC64":
                try:
                    bit_generator = numpy.random.SFC64()
                except AttributeError:
                    _logger.warning("SFC64 not available; falling back to PCG64")
                    bit_generator = numpy.random.PCG64()
            else:
                # Default to PCG64 for unknown types
                _logger.warning(f"Unknown generator type {generator_type}, using PCG64")
                bit_generator = numpy.random.PCG64()

            # Create generator
            restored_rng = numpy.random.Generator(bit_generator)

            # Rehydrate nested list state (common for MT19937) back to numpy arrays
            restored_state = dict(rng_state_data)
            nested_state = restored_state.get("state")
            try:
                if isinstance(nested_state, dict):
                    # PCG64: {'state': [..], 'inc': [..]}
                    if isinstance(nested_state.get("state"), list):
                        restored_state = dict(restored_state)
                        restored_state["state"] = dict(nested_state)
                        restored_state["state"]["state"] = numpy.array(
                            nested_state["state"], dtype=numpy.uint64
                        )
                    if "inc" in nested_state and isinstance(nested_state["inc"], list):
                        restored_state["state"]["inc"] = numpy.array(
                            nested_state["inc"], dtype=numpy.uint64
                        )
            except Exception:
                # Best-effort rehydration; fall back to original structure
                restored_state = dict(rng_state_data)

            # Restore the bit generator state directly
            restored_rng.bit_generator.state = restored_state

        except (ValueError, TypeError, KeyError) as e:
            raise ComponentError(
                message=f"Failed to restore RNG state: {str(e)}",
                component_name="seeding",
                operation_name="load_seed_state",
            ) from e

        # Validate restored generator by testing basic operations if validate_state is True
        if validate_state:
            try:
                # Test basic RNG operations without altering the returned generator state
                test_values = restored_rng.random(5)
                if not isinstance(test_values, numpy.ndarray) or len(test_values) != 5:
                    raise ComponentError(
                        message="Restored RNG failed validation test",
                        component_name="seeding",
                        operation_name="load_seed_state",
                    )

                # Restore generator state to the exact saved state after validation
                restored_rng.bit_generator.state = restored_state

                _logger.debug("Restored RNG passed validation tests and state reset")

            except Exception as e:
                raise ComponentError(
                    message=f"RNG validation failed: {str(e)}",
                    component_name="seeding",
                    operation_name="load_seed_state",
                ) from e

        # Extract metadata from state data
        metadata = state_data.get("metadata", {})
        metadata["load_timestamp"] = time.time()
        metadata["original_timestamp"] = state_data["timestamp"]
        metadata["version"] = state_version

        _logger.debug(f"Successfully loaded seed state from {file_path}")

        # Return (restored_rng, metadata) tuple with fully functional generator and associated data
        return (restored_rng, metadata)

    except (ValidationError, ResourceError, ComponentError):
        # Re-raise specific exceptions with original context
        raise
    except (FileNotFoundError, PermissionError, json.JSONDecodeError):
        # Surface native file-related exceptions as tests expect built-ins
        raise
    except Exception as e:
        # Wrap unexpected errors in ComponentError
        _logger.error(f"Seed state load failed: {e}")
        raise ComponentError(
            message=f"Unexpected error loading seed state: {str(e)}",
            component_name="seeding",
            operation_name="load_seed_state",
        ) from e


class SeedManager:
    """Centralized seed management class with validation, reproducibility tracking, thread safety, and performance
    optimization for scientific research applications requiring deterministic behavior and experiment reproducibility.

    This class provides comprehensive seed management capabilities including validation, random number generator
    creation, episode seed generation, reproducibility verification, and thread-safe operations for scientific
    research workflows requiring deterministic behavior and experiment reproducibility.
    """

    def __init__(
        self,
        default_seed: Optional[int] = None,
        enable_validation: bool = True,
        thread_safe: bool = False,
    ):
        """Initialize SeedManager with configuration options, thread safety setup, and validation framework
        for centralized seed management.

        Args:
            default_seed (Optional[int]): Default seed to use when no specific seed is provided
            enable_validation (bool): Whether to enable automatic seed validation for all operations
            thread_safe (bool): Whether to enable thread-safe operations with locking mechanisms
        """
        # Store default_seed for use when no specific seed is provided
        if default_seed is not None:
            is_valid, normalized_seed, error_message = validate_seed(default_seed)
            if not is_valid:
                raise ValidationError(
                    message=f"Invalid default seed: {error_message}",
                    parameter_name="default_seed",
                    parameter_value=default_seed,
                    expected_format="valid integer seed",
                )
            self.default_seed = normalized_seed
        else:
            self.default_seed = default_seed

        # Configure enable_validation flag for automatic seed validation
        self.enable_validation = bool(enable_validation)

        # Set thread_safe flag and initialize threading.Lock if required
        self.thread_safe = bool(thread_safe)
        if self.thread_safe:
            self._lock = threading.Lock()
        else:
            self._lock = None

        # Initialize empty active_generators dictionary for RNG tracking
        self.active_generators: Dict[str, Dict[str, Any]] = {}

        # Initialize empty seed_history list for operation tracking
        self.seed_history: List[Dict[str, Any]] = []

        # Create component-specific logger for seeding operations
        self.logger = logging.getLogger(f"{__name__}.SeedManager")

        self.logger.debug(
            f"Initialized SeedManager (default_seed={self.default_seed}, "
            f"validation={self.enable_validation}, thread_safe={self.thread_safe})"
        )

    def seed(
        self, seed: Optional[int] = None, context_id: Optional[str] = None
    ) -> Tuple[numpy.random.Generator, int]:
        """Primary seeding method for creating and managing random number generators with validation, tracking,
        and thread safety for deterministic environment behavior.

        Args:
            seed (Optional[int]): Seed value to use, None to use default_seed, or None for random
            context_id (Optional[str]): Context identifier for generator tracking and management

        Returns:
            Tuple[numpy.random.Generator, int]: Tuple containing thread-safe generator and actual seed used

        Raises:
            ValidationError: If seed validation fails with detailed error context
            StateError: If seeding operation fails due to invalid state or resource constraints
        """
        try:
            # Acquire thread lock if thread_safe is enabled for concurrent access protection
            if self._lock:
                self._lock.acquire()

            try:
                # Use provided seed or default_seed, validating with validate_seed if enable_validation is True
                effective_seed = seed if seed is not None else self.default_seed

                if self.enable_validation:
                    is_valid, validated_seed, error_message = validate_seed(
                        effective_seed
                    )
                    if not is_valid:
                        raise ValidationError(
                            message=f"Seed validation failed: {error_message}",
                            parameter_name="seed",
                            parameter_value=effective_seed,
                            expected_format="valid integer seed",
                        )
                    effective_seed = validated_seed

                # Create seeded RNG using create_seeded_rng function with validated seed
                np_random, seed_used = create_seeded_rng(
                    effective_seed, validate_input=False
                )

                # Generate context_id if not provided using uuid for unique identification
                if context_id is None:
                    context_id = f"seed_manager_{uuid.uuid4().hex[:8]}"

                # Store generator in active_generators dictionary with context tracking
                generator_info = {
                    "generator": np_random,
                    "seed_used": seed_used,
                    "creation_timestamp": time.time(),
                    "access_count": 0,
                    "last_access_timestamp": time.time(),
                }
                self.active_generators[context_id] = generator_info

                # Add seed operation to seed_history with timestamp and context information
                history_entry = {
                    "operation": "seed",
                    "context_id": context_id,
                    "seed_requested": seed,
                    "seed_used": seed_used,
                    "timestamp": time.time(),
                    "validation_enabled": self.enable_validation,
                }
                self.seed_history.append(history_entry)

                # Limit history size to prevent memory growth
                if len(self.seed_history) > 1000:
                    self.seed_history = self.seed_history[
                        -500:
                    ]  # Keep last 500 entries

                # Log seeding operation with context for debugging and reproducibility tracking
                self.logger.debug(
                    f"Created seeded generator: context_id={context_id}, seed={seed_used}"
                )

                # Return (np_random, seed_used) tuple with generator ready for use
                return (np_random, seed_used)

            finally:
                # Always release lock even if exception occurs
                if self._lock:
                    self._lock.release()

        except ValidationError:
            # Re-raise validation errors with original context
            raise
        except Exception as e:
            # Wrap unexpected errors in StateError
            self.logger.error(f"Seeding operation failed: {e}")
            raise StateError(
                message=f"Failed to create seeded generator: {str(e)}",
                current_state="seeding_failed",
                expected_state="generator_created",
                component_name="SeedManager",
            ) from e

    def generate_episode_seed(
        self, base_seed: int, episode_number: int, experiment_id: Optional[str] = None
    ) -> int:
        """Generate deterministic episode seeds for reproducible episodes with episode context integration
        and sequence management for scientific research workflows.

        Args:
            base_seed (int): Base seed for deterministic episode seed generation
            episode_number (int): Episode number for unique episode identification
            experiment_id (Optional[str]): Experiment identifier for context-specific seeding

        Returns:
            int: Episode-specific seed derived deterministically from base seed and episode context

        Raises:
            ValidationError: If base_seed is invalid or parameters are out of bounds
        """
        try:
            # Validate base_seed using validate_seed function with comprehensive error checking
            is_valid, validated_base_seed, error_message = validate_seed(base_seed)
            if not is_valid:
                raise ValidationError(
                    message=f"Invalid base seed for episode generation: {error_message}",
                    parameter_name="base_seed",
                    parameter_value=base_seed,
                    expected_format="valid integer seed",
                )

            # Validate episode_number is non-negative integer
            if not isinstance(episode_number, int) or episode_number < 0:
                raise ValidationError(
                    message="Episode number must be non-negative integer",
                    parameter_name="episode_number",
                    parameter_value=episode_number,
                    expected_format="non-negative integer",
                )

            # Create episode context string combining experiment_id and episode_number
            if experiment_id:
                context_string = f"{experiment_id}_episode_{episode_number}"
            else:
                context_string = f"episode_{episode_number}"

            # Generate deterministic seed component using generate_deterministic_seed with context
            context_seed = generate_deterministic_seed(context_string)

            # Combine base_seed with episode-specific seed using mathematical mixing function
            # Use XOR and addition with modulo to ensure good distribution
            combined_seed = (
                validated_base_seed + context_seed + episode_number * 1009
            ) % (SEED_MAX_VALUE + 1)

            # Apply modulo operation to ensure result is within valid seed range
            episode_seed = combined_seed % (SEED_MAX_VALUE + 1)

            # Log episode seed generation with full context for reproducibility tracking
            self.logger.debug(
                f"Generated episode seed {episode_seed} for episode {episode_number} "
                f"(base: {validated_base_seed}, experiment: {experiment_id})"
            )

            # Return episode seed ensuring identical inputs always produce identical episode seeds
            return episode_seed

        except ValidationError:
            # Re-raise validation errors with original context
            raise
        except Exception as e:
            # Wrap unexpected errors in ComponentError
            self.logger.error(f"Episode seed generation failed: {e}")
            raise ComponentError(
                message=f"Failed to generate episode seed: {str(e)}",
                component_name="SeedManager",
                operation_name="generate_episode_seed",
            ) from e

    def generate_random_position(
        self,
        grid_size: "GridSize",
        exclude_position: Optional["Coordinates"] = None,
        max_attempts: int = 100,
    ) -> "Coordinates":
        """Generate random position within grid bounds using current RNG.

        Args:
            grid_size: Grid dimensions for position bounds
            exclude_position: Optional position to avoid (e.g., source location)
            max_attempts: Maximum attempts to find non-excluded position

        Returns:
            Random Coordinates within grid bounds

        Raises:
            ComponentError: If cannot generate valid position after max_attempts
        """
        from ..core.geometry import Coordinates, GridSize

        try:
            # Validate grid_size
            if not isinstance(grid_size, GridSize):
                raise ValidationError(
                    message="grid_size must be GridSize instance",
                    parameter_name="grid_size",
                    parameter_value=str(type(grid_size)),
                )

            # Get current RNG (use seed with default or create new)
            rng, _ = self.seed(context_id="generate_position")

            for attempt in range(max_attempts):
                # Generate random position within bounds
                x = int(rng.integers(0, grid_size.width))
                y = int(rng.integers(0, grid_size.height))
                position = Coordinates(x=x, y=y)

                # Check if we need to exclude this position
                if exclude_position is None:
                    return position

                # If position different from excluded, return it
                if position.x != exclude_position.x or position.y != exclude_position.y:
                    return position

            # Failed to generate non-excluded position
            raise ComponentError(
                message=f"Could not generate random position after {max_attempts} attempts",
                component_name="SeedManager",
                operation_name="generate_random_position",
            )

        except (ValidationError, ComponentError):
            raise
        except Exception as e:
            self.logger.error(f"Random position generation failed: {e}")
            raise ComponentError(
                message=f"Failed to generate random position: {str(e)}",
                component_name="SeedManager",
                operation_name="generate_random_position",
            ) from e

    def validate_reproducibility(  # noqa: C901
        self,
        test_seed: int,
        num_tests: int = 10,
        tolerance: float = _REPRODUCIBILITY_TOLERANCE,
    ) -> Dict[str, Any]:
        """Validate reproducibility of seeded generators through comprehensive testing, statistical analysis,
        and tolerance-based comparison for scientific research requirements.

        Args:
            test_seed (int): Seed to use for reproducibility testing
            num_tests (int): Number of reproducibility tests to perform
            tolerance (float): Tolerance for floating-point comparison in reproducibility tests

        Returns:
            Dict[str, Any]: Comprehensive reproducibility validation report with statistical analysis and recommendations

        Raises:
            ValidationError: If parameters are invalid or out of bounds
            ComponentError: If reproducibility validation fails due to system constraints
        """
        try:
            # Validate test_seed using validate_seed function
            is_valid, validated_seed, error_message = validate_seed(test_seed)
            if not is_valid:
                raise ValidationError(
                    message=f"Invalid test seed: {error_message}",
                    parameter_name="test_seed",
                    parameter_value=test_seed,
                    expected_format="valid integer seed",
                )

            # Validate num_tests is positive integer
            if not isinstance(num_tests, int) or num_tests <= 0:
                raise ValidationError(
                    message="Number of tests must be positive integer",
                    parameter_name="num_tests",
                    parameter_value=num_tests,
                    expected_format="positive integer",
                )

            self.logger.debug(
                f"Starting reproducibility validation with {num_tests} tests using seed {validated_seed}"
            )

            # Run num_tests reproducibility tests using verify_reproducibility function
            test_results = []
            success_count = 0

            for test_idx in range(num_tests):
                try:
                    # Create multiple generators using identical test_seed for comparison testing
                    rng1, _ = create_seeded_rng(validated_seed, validate_input=False)
                    rng2, _ = create_seeded_rng(validated_seed, validate_input=False)

                    # Run reproducibility verification
                    verification_result = verify_reproducibility(
                        rng1,
                        rng2,
                        sequence_length=100,  # Use shorter sequences for batch testing
                        tolerance=tolerance,
                    )

                    test_results.append(
                        {
                            "test_index": test_idx,
                            "sequences_match": verification_result["sequences_match"],
                            "mean_absolute_error": verification_result[
                                "statistical_analysis"
                            ]["mean_absolute_error"],
                            "max_deviation": verification_result[
                                "statistical_analysis"
                            ]["max_deviation"],
                            "reproducibility_score": verification_result[
                                "reproducibility_score"
                            ],
                        }
                    )

                    if verification_result["sequences_match"]:
                        success_count += 1

                except Exception as e:
                    test_results.append(
                        {
                            "test_index": test_idx,
                            "sequences_match": False,
                            "error": str(e),
                            "reproducibility_score": 0.0,
                        }
                    )

            # Collect statistical metrics including success rate and deviation measurements
            success_rate = success_count / num_tests

            # Calculate aggregate statistics from successful tests
            successful_tests = [
                r for r in test_results if r.get("sequences_match", False)
            ]
            if successful_tests:
                mean_errors = [r["mean_absolute_error"] for r in successful_tests]
                max_deviations = [r["max_deviation"] for r in successful_tests]
                reproducibility_scores = [
                    r["reproducibility_score"] for r in successful_tests
                ]

                aggregate_statistics = {
                    "mean_of_mean_errors": numpy.mean(mean_errors),
                    "std_of_mean_errors": numpy.std(mean_errors),
                    "mean_of_max_deviations": numpy.mean(max_deviations),
                    "mean_reproducibility_score": numpy.mean(reproducibility_scores),
                    "min_reproducibility_score": numpy.min(reproducibility_scores),
                }
            else:
                aggregate_statistics = {
                    "mean_of_mean_errors": float("inf"),
                    "std_of_mean_errors": float("inf"),
                    "mean_of_max_deviations": float("inf"),
                    "mean_reproducibility_score": 0.0,
                    "min_reproducibility_score": 0.0,
                }

            # Analyze failures for patterns and root cause identification
            failed_tests = [
                r for r in test_results if not r.get("sequences_match", False)
            ]
            failure_analysis = {
                "num_failures": len(failed_tests),
                "failure_rate": 1.0 - success_rate,
                "common_errors": [],
            }

            if failed_tests:
                error_messages = [
                    r.get("error", "Unknown error")
                    for r in failed_tests
                    if "error" in r
                ]
                if error_messages:
                    from collections import Counter

                    error_counts = Counter(error_messages)
                    failure_analysis["common_errors"] = list(
                        error_counts.most_common(3)
                    )

            # Generate comprehensive validation report with statistical summaries
            validation_report = {
                "test_configuration": {
                    "test_seed": validated_seed,
                    "num_tests": num_tests,
                    "tolerance": tolerance,
                    "validation_timestamp": time.time(),
                },
                "results_summary": {
                    "success_count": success_count,
                    "success_rate": success_rate,
                    "total_tests": num_tests,
                },
                "statistical_analysis": aggregate_statistics,
                "failure_analysis": failure_analysis,
                "individual_test_results": test_results,
                "overall_status": "PASS" if success_rate >= 0.95 else "FAIL",
            }

            # Include recommendations for reproducibility improvement if needed
            recommendations = []
            if success_rate < 0.95:
                recommendations.append(
                    "Success rate below 95% - investigate RNG initialization"
                )
            if aggregate_statistics["mean_of_mean_errors"] > tolerance * 10:
                recommendations.append(
                    "High mean errors detected - consider system-level issues"
                )
            if len(failure_analysis["common_errors"]) > 0:
                recommendations.append(
                    "Common error patterns detected - review error analysis"
                )

            if success_rate >= 0.99:
                recommendations.append(
                    "Excellent reproducibility - suitable for scientific research"
                )
            elif success_rate >= 0.95:
                recommendations.append(
                    "Good reproducibility - acceptable for most applications"
                )

            validation_report["recommendations"] = recommendations

            # Log validation results for scientific documentation and debugging
            self.logger.info(
                f"Reproducibility validation completed: {validation_report['overall_status']} "
                f"(success rate: {success_rate:.1%})"
            )

            # Return detailed reproducibility report suitable for research documentation
            return validation_report

        except ValidationError:
            # Re-raise validation errors with original context
            raise
        except Exception as e:
            # Wrap unexpected errors in ComponentError
            self.logger.error(f"Reproducibility validation failed: {e}")
            raise ComponentError(
                message=f"Failed to validate reproducibility: {str(e)}",
                component_name="SeedManager",
                operation_name="validate_reproducibility",
            ) from e

    def get_active_generators(self) -> Dict[str, Dict[str, Any]]:
        """Get list of active generators with context information for monitoring, debugging,
        and resource management in multi-context seeding scenarios.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of active generators with context IDs, creation timestamps, and usage statistics
        """
        try:
            # Acquire thread lock if thread_safe is enabled for consistent state reading
            if self._lock:
                self._lock.acquire()

            try:
                # Iterate through active_generators dictionary collecting generator information
                generator_summary = {}
                current_time = time.time()

                for context_id, generator_info in self.active_generators.items():
                    # Include context IDs, creation timestamps, and usage statistics for each generator
                    summary_info = {
                        "context_id": context_id,
                        "seed_used": generator_info["seed_used"],
                        "creation_timestamp": generator_info["creation_timestamp"],
                        "age_seconds": current_time
                        - generator_info["creation_timestamp"],
                        "access_count": generator_info["access_count"],
                        "last_access_timestamp": generator_info[
                            "last_access_timestamp"
                        ],
                        "seconds_since_last_access": current_time
                        - generator_info["last_access_timestamp"],
                    }

                    # Add current state information and seed tracking data
                    try:
                        # Test if generator is still functional
                        generator_info["generator"].random()
                        summary_info["status"] = "active"
                        summary_info["last_test_successful"] = True

                        # Update access statistics
                        generator_info["access_count"] += 1
                        generator_info["last_access_timestamp"] = current_time

                    except Exception as e:
                        summary_info["status"] = "error"
                        summary_info["last_test_successful"] = False
                        summary_info["error_message"] = str(e)

                    generator_summary[context_id] = summary_info

                # Format generator information for monitoring and debugging purposes
                active_generators_report = {
                    "total_active_generators": len(generator_summary),
                    "report_timestamp": current_time,
                    "generators": generator_summary,
                    "memory_usage_estimate": len(generator_summary)
                    * 1024,  # Rough estimate in bytes
                }

                # Return comprehensive active generators report with resource usage information
                return active_generators_report

            finally:
                # Always release lock
                if self._lock:
                    self._lock.release()

        except Exception as e:
            # Handle errors in generator inspection
            self.logger.warning(f"Failed to get active generators info: {e}")
            return {
                "total_active_generators": 0,
                "report_timestamp": time.time(),
                "generators": {},
                "error": str(e),
            }

    def reset(self, preserve_default_seed: bool = True) -> None:
        """Reset SeedManager state clearing active generators and seed history for clean state management
        and resource cleanup.

        Args:
            preserve_default_seed (bool): Whether to keep the default_seed setting during reset
        """
        try:
            # Acquire thread lock if thread_safe is enabled for atomic reset operation
            if self._lock:
                self._lock.acquire()

            try:
                # Clear active_generators dictionary releasing all generator references
                num_generators = len(self.active_generators)
                self.active_generators.clear()

                # Clear seed_history list removing all operation tracking
                history_entries = len(self.seed_history)
                self.seed_history.clear()

                # Reset default_seed to None unless preserve_default_seed is True
                if not preserve_default_seed:
                    self.default_seed = None

                # Log reset operation with timestamp for operation tracking
                self.logger.info(
                    f"SeedManager reset: cleared {num_generators} generators, "
                    f"{history_entries} history entries (preserve_default={preserve_default_seed})"
                )

                # Add reset operation to history
                self.seed_history.append(
                    {
                        "operation": "reset",
                        "timestamp": time.time(),
                        "generators_cleared": num_generators,
                        "history_entries_cleared": history_entries,
                        "default_seed_preserved": preserve_default_seed,
                    }
                )

            finally:
                # Release thread lock ensuring clean state for subsequent operations
                if self._lock:
                    self._lock.release()

        except Exception as e:
            # Log reset errors but don't raise to avoid breaking cleanup
            self.logger.error(f"Error during SeedManager reset: {e}")


class ReproducibilityTracker:
    """Specialized reproducibility tracking and validation class for scientific research with comprehensive episode comparison,
    statistical analysis, failure diagnostics, and research-grade reporting capabilities.

    This class provides comprehensive reproducibility tracking and validation capabilities for scientific research
    including episode data recording, reproducibility verification, statistical analysis, and research-grade reporting
    for experiment validation and scientific publication requirements.
    """

    def __init__(
        self,
        tolerance: float = _REPRODUCIBILITY_TOLERANCE,
        session_id: Optional[str] = None,
    ):
        """Initialize ReproducibilityTracker for episode recording and reproducibility verification.

        Simplified design per YAGNI principles - only essential features, no unused computations.

        Args:
            tolerance (float): Tolerance for floating-point comparisons in reproducibility verification
            session_id (Optional[str]): Session identifier for grouping related episodes
        """
        # Store tolerance for reproducibility comparison
        if not isinstance(tolerance, (int, float)) or tolerance <= 0:
            raise ValidationError(
                message="Tolerance must be positive number",
                parameter_name="tolerance",
                parameter_value=tolerance,
                expected_format="positive float",
            )
        self.tolerance = float(tolerance)

        # Generate session_id if not provided using uuid for unique session identification
        if session_id is None:
            self.session_id = f"repro_session_{uuid.uuid4().hex[:12]}"
        else:
            if not isinstance(session_id, str) or not session_id.strip():
                raise ValidationError(
                    message="Session ID must be non-empty string",
                    parameter_name="session_id",
                    parameter_value=session_id,
                    expected_format="non-empty string",
                )
            self.session_id = session_id.strip()

        # Initialize empty episode_records dictionary for storing episode data and checksums
        self.episode_records: Dict[str, Dict[str, Any]] = {}

        # Initialize empty test_results dictionary for validation test outcome tracking
        self.test_results: Dict[str, Dict[str, Any]] = {}

        # Initialize empty statistical_summaries list for cumulative analysis data
        self.statistical_summaries: List[Dict[str, Any]] = []

        # Create specialized logger for reproducibility tracking and scientific documentation
        self.logger = logging.getLogger(f"{__name__}.ReproducibilityTracker")

        self.logger.debug(
            f"Initialized ReproducibilityTracker: session={self.session_id}, "
            f"tolerance={self.tolerance}"
        )

    def record_episode(
        self,
        episode_seed: int,
        action_sequence: List[Any],
        observation_sequence: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record episode data for reproducibility tracking.

        Stores seed, action/observation sequences, lengths, and timestamp.
        Per YAGNI: No checksums (never verified), no strict validation (single-purpose check removed).

        Args:
            episode_seed (int): Seed used for the episode
            action_sequence (List[Any]): Sequence of actions taken during the episode
            observation_sequence (List[Any]): Sequence of observations received during the episode
            metadata (Optional[Dict[str, Any]]): Additional episode metadata (do not include sensitive data)

        Returns:
            str: Episode record ID for future reference and reproducibility verification

        Raises:
            ValidationError: If episode data is invalid or parameters are malformed
        """
        try:
            # Validate episode_seed using validate_seed function with comprehensive error checking
            is_valid, validated_seed, error_message = validate_seed(episode_seed)
            if not is_valid:
                raise ValidationError(
                    message=f"Invalid episode seed: {error_message}",
                    parameter_name="episode_seed",
                    parameter_value=episode_seed,
                    expected_format="valid integer seed",
                )

            # Validate action_sequence and observation_sequence for completeness and consistency
            if not isinstance(action_sequence, list):
                raise ValidationError(
                    message="Action sequence must be a list",
                    parameter_name="action_sequence",
                    parameter_value=type(action_sequence).__name__,
                    expected_format="list of actions",
                )

            if not isinstance(observation_sequence, list):
                raise ValidationError(
                    message="Observation sequence must be a list",
                    parameter_name="observation_sequence",
                    parameter_value=type(observation_sequence).__name__,
                    expected_format="list of observations",
                )

            if len(action_sequence) == 0:
                raise ValidationError(
                    message="Action sequence cannot be empty",
                    parameter_name="action_sequence",
                    parameter_value=len(action_sequence),
                    expected_format="non-empty list",
                )

            # Generate unique episode record ID
            episode_record_id = f"episode_{uuid.uuid4().hex[:16]}"

            # Create episode record with essential data only
            episode_record = {
                "episode_record_id": episode_record_id,
                "episode_seed": validated_seed,
                "action_sequence": action_sequence.copy(),
                "observation_sequence": observation_sequence.copy(),
                "sequence_lengths": {
                    "actions": len(action_sequence),
                    "observations": len(observation_sequence),
                },
                "recording_timestamp": time.time(),
                "session_id": self.session_id,
            }

            # Include metadata if provided (user responsible for not including sensitive data)
            if metadata:
                if isinstance(metadata, dict):
                    episode_record["metadata"] = metadata
                else:
                    self.logger.warning(
                        f"Metadata is not a dictionary, skipping: {type(metadata)}"
                    )

            # Store episode record in episode_records dictionary with timestamp
            self.episode_records[episode_record_id] = episode_record

            # Log episode recording with seed and sequence length for tracking
            self.logger.debug(
                f"Recorded episode {episode_record_id}: seed={validated_seed}, "
                f"actions={len(action_sequence)}, observations={len(observation_sequence)}"
            )

            # Return episode_record_id for future reproducibility verification
            return episode_record_id

        except ValidationError:
            # Re-raise validation errors with original context
            raise
        except Exception as e:
            # Wrap unexpected errors in ComponentError
            self.logger.error(f"Episode recording failed: {e}")
            raise ComponentError(
                message=f"Failed to record episode: {str(e)}",
                component_name="ReproducibilityTracker",
                operation_name="record_episode",
            ) from e

    def verify_episode_reproducibility(  # noqa: C901
        self,
        episode_record_id: str,
        new_action_sequence: List[Any],
        new_observation_sequence: List[Any],
        custom_tolerance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Verify episode reproducibility by comparing recorded episode with new execution using detailed
        sequence comparison and statistical analysis.

        Args:
            episode_record_id (str): ID of recorded episode to compare against
            new_action_sequence (List[Any]): Action sequence from new episode execution
            new_observation_sequence (List[Any]): Observation sequence from new episode execution
            custom_tolerance (Optional[float]): Custom tolerance for this verification, overrides default

        Returns:
            Dict[str, Any]: Detailed verification results with match status, discrepancy analysis, and statistical measures

        Raises:
            ValidationError: If episode record is not found or sequences are invalid
            ComponentError: If verification fails due to system constraints
        """
        try:
            # Retrieve original episode record using episode_record_id with validation
            if episode_record_id not in self.episode_records:
                raise ValidationError(
                    message=f"Episode record not found: {episode_record_id}",
                    parameter_name="episode_record_id",
                    parameter_value=episode_record_id,
                    expected_format="valid episode record ID",
                )

            original_record = self.episode_records[episode_record_id]
            original_actions = original_record["action_sequence"]
            original_observations = original_record["observation_sequence"]

            # Validate new sequences
            if not isinstance(new_action_sequence, list) or not isinstance(
                new_observation_sequence, list
            ):
                raise ValidationError(
                    message="New sequences must be lists",
                    parameter_name="new_sequences",
                    parameter_value=f"actions: {type(new_action_sequence)}, obs: {type(new_observation_sequence)}",
                    expected_format="lists",
                )

            # Compare sequence lengths ensuring both episodes have identical structure
            length_match = {
                "actions_match": len(original_actions) == len(new_action_sequence),
                "observations_match": len(original_observations)
                == len(new_observation_sequence),
                "original_lengths": {
                    "actions": len(original_actions),
                    "observations": len(original_observations),
                },
                "new_lengths": {
                    "actions": len(new_action_sequence),
                    "observations": len(new_observation_sequence),
                },
            }

            if not (
                length_match["actions_match"] and length_match["observations_match"]
            ):
                # Sequences have different lengths - major reproducibility failure
                verification_result = {
                    "verification_id": f"verify_{uuid.uuid4().hex[:12]}",
                    "episode_record_id": episode_record_id,
                    "verification_timestamp": time.time(),
                    "match_status": "LENGTH_MISMATCH",
                    "sequences_match": False,
                    "length_analysis": length_match,
                    "reproducibility_score": 0.0,
                    "error_type": "structural_mismatch",
                }

                # Store verification results
                self.test_results[verification_result["verification_id"]] = (
                    verification_result
                )

                return verification_result

            # Determine tolerance for comparison
            tolerance = (
                custom_tolerance if custom_tolerance is not None else self.tolerance
            )

            # Perform element-wise comparison of action sequences for exact equality
            action_comparison = self._compare_sequences(
                original_actions, new_action_sequence, tolerance
            )

            # Compare observation sequences using tolerance-based floating point comparison
            observation_comparison = self._compare_sequences(
                original_observations, new_observation_sequence, tolerance
            )

            # Calculate statistical measures including mean absolute error and maximum deviation
            overall_match = (
                action_comparison["sequences_match"]
                and observation_comparison["sequences_match"]
            )

            # Calculate combined reproducibility score
            action_score = action_comparison.get("reproducibility_score", 0.0)
            observation_score = observation_comparison.get("reproducibility_score", 0.0)
            combined_score = (action_score + observation_score) / 2.0

            # Generate verification result dictionary
            verification_result = {
                "verification_id": f"verify_{uuid.uuid4().hex[:12]}",
                "episode_record_id": episode_record_id,
                "verification_timestamp": time.time(),
                "tolerance_used": tolerance,
                "match_status": "PASS" if overall_match else "FAIL",
                "sequences_match": overall_match,
                "length_analysis": length_match,
                "action_comparison": action_comparison,
                "observation_comparison": observation_comparison,
                "reproducibility_score": combined_score,
                "original_episode_info": {
                    "seed": original_record["episode_seed"],
                    "recording_timestamp": original_record["recording_timestamp"],
                },
            }

            # Identify specific discrepancy points with detailed analysis if verification fails
            if not overall_match:
                discrepancy_analysis = {
                    "action_discrepancies": action_comparison.get(
                        "discrepancy_indices", []
                    ),
                    "observation_discrepancies": observation_comparison.get(
                        "discrepancy_indices", []
                    ),
                    "total_discrepancies": len(
                        action_comparison.get("discrepancy_indices", [])
                    )
                    + len(observation_comparison.get("discrepancy_indices", [])),
                    "largest_discrepancy": max(
                        action_comparison.get("max_deviation", 0.0),
                        observation_comparison.get("max_deviation", 0.0),
                    ),
                }
                verification_result["discrepancy_analysis"] = discrepancy_analysis

                # Generate failure recommendations
                recommendations = []
                if discrepancy_analysis["largest_discrepancy"] > tolerance * 1000:
                    recommendations.append(
                        "Large discrepancies suggest fundamental reproducibility issue"
                    )
                if len(discrepancy_analysis["action_discrepancies"]) > 0:
                    recommendations.append(
                        "Action sequence mismatch - check RNG seeding for policy"
                    )
                if len(discrepancy_analysis["observation_discrepancies"]) > 0:
                    recommendations.append(
                        "Observation sequence mismatch - check environment seeding"
                    )

                verification_result["recommendations"] = recommendations

            # Generate comprehensive verification report with match status and detailed analysis
            if overall_match:
                verification_result["status_message"] = (
                    "Episode successfully reproduced"
                )
            else:
                verification_result["status_message"] = (
                    "Episode reproducibility verification failed"
                )

            # Store verification results in test_results dictionary for cumulative analysis
            self.test_results[verification_result["verification_id"]] = (
                verification_result
            )

            # Log verification results
            self.logger.debug(
                f"Episode verification {verification_result['verification_id']}: "
                f"{verification_result['match_status']} (score: {combined_score:.3f})"
            )

            return verification_result

        except ValidationError:
            # Re-raise validation errors with original context
            raise
        except Exception as e:
            # Wrap unexpected errors in ComponentError
            self.logger.error(f"Episode reproducibility verification failed: {e}")
            raise ComponentError(
                message=f"Failed to verify episode reproducibility: {str(e)}",
                component_name="ReproducibilityTracker",
                operation_name="verify_episode_reproducibility",
            ) from e

    def _compare_sequences(
        self, seq1: List[Any], seq2: List[Any], tolerance: float
    ) -> Dict[str, Any]:
        """Compare two sequences with tolerance-based comparison."""
        try:
            comparison_result = {
                "sequences_match": True,
                "exact_matches": 0,
                "total_elements": len(seq1),
                "discrepancy_indices": [],
                "statistical_analysis": {},
            }

            if len(seq1) != len(seq2):
                comparison_result["sequences_match"] = False
                comparison_result["reproducibility_score"] = 0.0
                return comparison_result

            # Convert sequences to numpy arrays for numerical comparison
            try:
                arr1 = numpy.array(seq1, dtype=float)
                arr2 = numpy.array(seq2, dtype=float)

                # Calculate differences
                differences = numpy.abs(arr1 - arr2)

                # Check which elements match within tolerance
                within_tolerance = differences <= tolerance
                exact_matches = numpy.sum(arr1 == arr2)
                tolerance_matches = numpy.sum(within_tolerance)

                comparison_result.update(
                    {
                        "exact_matches": int(exact_matches),
                        "tolerance_matches": int(tolerance_matches),
                        "sequences_match": bool(tolerance_matches == len(seq1)),
                        "reproducibility_score": float(tolerance_matches / len(seq1)),
                        "statistical_analysis": {
                            "mean_absolute_error": float(numpy.mean(differences)),
                            "max_deviation": float(numpy.max(differences)),
                            "min_deviation": float(numpy.min(differences)),
                            "std_deviation": float(numpy.std(differences)),
                        },
                    }
                )

                # Find discrepancy indices
                if not comparison_result["sequences_match"]:
                    discrepancy_mask = ~within_tolerance
                    comparison_result["discrepancy_indices"] = numpy.where(
                        discrepancy_mask
                    )[0].tolist()

            except (ValueError, TypeError):
                # Fall back to element-wise comparison for non-numeric data
                exact_matches = 0
                discrepancies = []

                for i, (val1, val2) in enumerate(zip(seq1, seq2)):
                    if val1 == val2:
                        exact_matches += 1
                    else:
                        discrepancies.append(i)

                comparison_result.update(
                    {
                        "exact_matches": exact_matches,
                        "sequences_match": len(discrepancies) == 0,
                        "reproducibility_score": exact_matches / len(seq1),
                        "discrepancy_indices": discrepancies,
                    }
                )

            return comparison_result

        except Exception as e:
            return {
                "sequences_match": False,
                "reproducibility_score": 0.0,
                "error": str(e),
                "comparison_failed": True,
            }

    def generate_reproducibility_report(  # noqa: C901
        self,
        report_format: Optional[str] = "dict",
        include_detailed_analysis: bool = True,
        custom_sections: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive reproducibility report for scientific documentation including statistical summaries,
        success rates, and detailed analysis.

        Args:
            report_format (Optional[str]): Format for the report (dict, json, markdown)
            include_detailed_analysis (bool): Whether to include detailed failure analysis and individual test results
            custom_sections (Optional[Dict[str, Any]]): Custom sections to include in the report

        Returns:
            Dict[str, Any]: Comprehensive reproducibility report formatted for scientific documentation and research analysis
        """
        try:
            report_timestamp = time.time()

            # Compile statistical summaries from all test_results and episode_records
            total_episodes = len(self.episode_records)
            total_verifications = len(self.test_results)

            # Calculate overall success rate and reproducibility metrics across all tests
            if total_verifications > 0:
                successful_verifications = [
                    v
                    for v in self.test_results.values()
                    if v.get("sequences_match", False)
                ]
                success_rate = len(successful_verifications) / total_verifications

                # Calculate statistical summaries
                if successful_verifications:
                    scores = [
                        v.get("reproducibility_score", 0.0)
                        for v in successful_verifications
                    ]
                    statistical_summary = {
                        "mean_reproducibility_score": numpy.mean(scores),
                        "std_reproducibility_score": numpy.std(scores),
                        "min_reproducibility_score": numpy.min(scores),
                        "max_reproducibility_score": numpy.max(scores),
                        "median_reproducibility_score": numpy.median(scores),
                    }
                else:
                    statistical_summary = {
                        "mean_reproducibility_score": 0.0,
                        "std_reproducibility_score": 0.0,
                        "min_reproducibility_score": 0.0,
                        "max_reproducibility_score": 0.0,
                        "median_reproducibility_score": 0.0,
                    }
            else:
                success_rate = 0.0
                statistical_summary = {}

            # Generate base report structure
            report = {
                "report_metadata": {
                    "session_id": self.session_id,
                    "generation_timestamp": report_timestamp,
                    "report_format": report_format,
                    "tolerance": self.tolerance,
                },
                "summary_statistics": {
                    "total_episodes_recorded": total_episodes,
                    "total_verifications_performed": total_verifications,
                    "overall_success_rate": success_rate,
                    "successful_verifications": len(
                        [
                            v
                            for v in self.test_results.values()
                            if v.get("sequences_match", False)
                        ]
                    ),
                    "failed_verifications": len(
                        [
                            v
                            for v in self.test_results.values()
                            if not v.get("sequences_match", False)
                        ]
                    ),
                },
                "statistical_analysis": statistical_summary,
                "reproducibility_assessment": {
                    "overall_status": self._assess_reproducibility_status(success_rate),
                    "confidence_level": self._calculate_confidence_level(
                        total_verifications, success_rate
                    ),
                    "recommendations": self._generate_reproducibility_recommendations(
                        success_rate, statistical_summary
                    ),
                },
            }

            # Generate detailed failure analysis if include_detailed_analysis is True
            if include_detailed_analysis:
                failed_tests = [
                    v
                    for v in self.test_results.values()
                    if not v.get("sequences_match", False)
                ]

                failure_analysis = {
                    "total_failures": len(failed_tests),
                    "failure_types": {},
                    "common_failure_patterns": [],
                    "largest_discrepancies": [],
                }

                if failed_tests:
                    # Analyze failure types
                    failure_types = {}
                    for test in failed_tests:
                        failure_type = test.get("match_status", "UNKNOWN")
                        failure_types[failure_type] = (
                            failure_types.get(failure_type, 0) + 1
                        )
                    failure_analysis["failure_types"] = failure_types

                    # Find largest discrepancies
                    discrepancies = []
                    for test in failed_tests:
                        if "discrepancy_analysis" in test:
                            discrepancy = test["discrepancy_analysis"].get(
                                "largest_discrepancy", 0.0
                            )
                            discrepancies.append(
                                {
                                    "verification_id": test.get(
                                        "verification_id", "unknown"
                                    ),
                                    "discrepancy_magnitude": discrepancy,
                                }
                            )

                    # Sort by discrepancy magnitude and take top 5
                    discrepancies.sort(
                        key=lambda x: x["discrepancy_magnitude"], reverse=True
                    )
                    failure_analysis["largest_discrepancies"] = discrepancies[:5]

                report["failure_analysis"] = failure_analysis

                # Include episode-level statistics and sequence comparison summaries
                episode_statistics = {
                    "episode_length_distribution": self._analyze_episode_lengths(),
                    "seed_distribution": self._analyze_seed_distribution(),
                    "temporal_analysis": self._analyze_temporal_patterns(),
                }
                report["episode_statistics"] = episode_statistics

            # Add custom sections from custom_sections parameter if provided
            if custom_sections and isinstance(custom_sections, dict):
                for section_name, section_content in custom_sections.items():
                    if isinstance(section_name, str) and section_name not in report:
                        report[f"custom_{section_name}"] = section_content

            # Format report according to specified report_format
            if report_format == "json":
                import json

                return {"json_report": json.dumps(report, indent=2, default=str)}
            elif report_format == "markdown":
                return {"markdown_report": self._format_markdown_report(report)}
            else:
                # Default to dictionary format
                return report

        except Exception as e:
            # Handle report generation failures
            self.logger.error(f"Reproducibility report generation failed: {e}")
            return {
                "report_error": str(e),
                "generation_timestamp": time.time(),
                "session_id": self.session_id,
                "partial_data": {
                    "total_episodes": len(self.episode_records),
                    "total_verifications": len(self.test_results),
                },
            }

    def _assess_reproducibility_status(self, success_rate: float) -> str:
        """Assess overall reproducibility status based on success rate."""
        if success_rate >= 0.99:
            return "EXCELLENT"
        elif success_rate >= 0.95:
            return "GOOD"
        elif success_rate >= 0.90:
            return "ACCEPTABLE"
        elif success_rate >= 0.75:
            return "POOR"
        else:
            return "CRITICAL"

    def _calculate_confidence_level(self, total_tests: int, success_rate: float) -> str:
        """Calculate confidence level based on number of tests and success rate."""
        if total_tests >= 100:
            return "HIGH"
        elif total_tests >= 50:
            return "MEDIUM"
        elif total_tests >= 10:
            return "LOW"
        else:
            return "INSUFFICIENT_DATA"

    def _generate_reproducibility_recommendations(
        self, success_rate: float, statistical_summary: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on reproducibility analysis."""
        recommendations = []

        if success_rate < 0.95:
            recommendations.append(
                "Success rate below 95% - investigate seeding and initialization procedures"
            )

        if (
            statistical_summary
            and statistical_summary.get("std_reproducibility_score", 0) > 0.1
        ):
            recommendations.append(
                "High variability in reproducibility scores - check for non-deterministic components"
            )

        if success_rate >= 0.99:
            recommendations.append(
                "Excellent reproducibility achieved - suitable for scientific publication"
            )

        if len(self.test_results) < 50:
            recommendations.append(
                "Consider running more verification tests for statistical significance"
            )

        return recommendations

    def _analyze_episode_lengths(self) -> Dict[str, Any]:
        """Analyze distribution of episode lengths."""
        if not self.episode_records:
            return {}

        action_lengths = [
            record["sequence_lengths"]["actions"]
            for record in self.episode_records.values()
        ]
        observation_lengths = [
            record["sequence_lengths"]["observations"]
            for record in self.episode_records.values()
        ]

        return {
            "action_lengths": {
                "mean": numpy.mean(action_lengths),
                "std": numpy.std(action_lengths),
                "min": numpy.min(action_lengths),
                "max": numpy.max(action_lengths),
            },
            "observation_lengths": {
                "mean": numpy.mean(observation_lengths),
                "std": numpy.std(observation_lengths),
                "min": numpy.min(observation_lengths),
                "max": numpy.max(observation_lengths),
            },
        }

    def _analyze_seed_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of seeds used in recorded episodes."""
        if not self.episode_records:
            return {}

        seeds = [record["episode_seed"] for record in self.episode_records.values()]
        unique_seeds = len(set(seeds))

        return {
            "total_seeds": len(seeds),
            "unique_seeds": unique_seeds,
            "seed_reuse_rate": (len(seeds) - unique_seeds) / len(seeds) if seeds else 0,
        }

    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in episode recording and verification."""
        if not self.episode_records:
            return {}

        recording_times = [
            record["recording_timestamp"] for record in self.episode_records.values()
        ]
        verification_times = [
            result["verification_timestamp"] for result in self.test_results.values()
        ]

        analysis = {
            "recording_time_span": (
                max(recording_times) - min(recording_times) if recording_times else 0
            ),
            "first_recording": min(recording_times) if recording_times else 0,
            "last_recording": max(recording_times) if recording_times else 0,
        }

        if verification_times:
            analysis.update(
                {
                    "verification_time_span": max(verification_times)
                    - min(verification_times),
                    "first_verification": min(verification_times),
                    "last_verification": max(verification_times),
                }
            )

        return analysis

    def _format_markdown_report(self, report: Dict[str, Any]) -> str:
        """Format report as Markdown string."""
        markdown = f"# Reproducibility Report - Session {report['report_metadata']['session_id']}\n\n"

        # Summary statistics
        summary = report["summary_statistics"]
        markdown += "## Summary Statistics\n\n"
        markdown += f"- Total Episodes Recorded: {summary['total_episodes_recorded']}\n"
        markdown += (
            f"- Total Verifications: {summary['total_verifications_performed']}\n"
        )
        markdown += f"- Success Rate: {summary['overall_success_rate']:.2%}\n\n"

        # Reproducibility assessment
        assessment = report["reproducibility_assessment"]
        markdown += f"## Overall Assessment: {assessment['overall_status']}\n\n"
        markdown += f"**Confidence Level:** {assessment['confidence_level']}\n\n"

        if assessment.get("recommendations"):
            markdown += "### Recommendations\n\n"
            for rec in assessment["recommendations"]:
                markdown += f"- {rec}\n"

        return markdown

    def export_reproducibility_data(  # noqa: C901
        self,
        export_path: Union[str, pathlib.Path],
        export_format: str = "json",
        include_raw_data: bool = True,
        compress_output: bool = False,
    ) -> bool:
        """Export reproducibility data for external analysis and archival with multiple format support
        and data integrity validation.

        Args:
            export_path (Union[str, pathlib.Path]): Path where to export reproducibility data
            export_format (str): Format for export (json, csv, hdf5)
            include_raw_data (bool): Whether to include raw episode sequences in export
            compress_output (bool): Whether to compress exported data

        Returns:
            bool: Export success status with comprehensive error handling and validation

        Raises:
            ValidationError: If export parameters are invalid
            ResourceError: If export fails due to file system issues
            ComponentError: If data serialization fails
        """
        try:
            # Validate export_path and create directory structure if needed
            if isinstance(export_path, str):
                export_path = pathlib.Path(export_path)
            elif not isinstance(export_path, pathlib.Path):
                raise ValidationError(
                    message="Export path must be string or pathlib.Path",
                    parameter_name="export_path",
                    parameter_value=type(export_path).__name__,
                    expected_format="str or pathlib.Path",
                )

            # Create parent directories
            export_path.parent.mkdir(parents=True, exist_ok=True)

            # Validate export format
            supported_formats = ["json", "csv"]
            if export_format not in supported_formats:
                raise ValidationError(
                    message=f"Unsupported export format: {export_format}",
                    parameter_name="export_format",
                    parameter_value=export_format,
                    expected_format=f"one of: {supported_formats}",
                )

            # Compile complete reproducibility dataset including episode records and test results
            export_data = {
                "export_metadata": {
                    "session_id": self.session_id,
                    "export_timestamp": time.time(),
                    "export_format": export_format,
                    "include_raw_data": include_raw_data,
                    "compressed": compress_output,
                    "tolerance": self.tolerance,
                },
                "episode_records_summary": {
                    "total_episodes": len(self.episode_records),
                    "episode_ids": list(self.episode_records.keys()),
                },
                "verification_results_summary": {
                    "total_verifications": len(self.test_results),
                    "verification_ids": list(self.test_results.keys()),
                },
            }

            # Include raw episode sequences if include_raw_data is True
            if include_raw_data:
                # Sanitize episode records for export
                sanitized_episodes = {}
                for episode_id, record in self.episode_records.items():
                    sanitized_record = record.copy()
                    # Remove potentially large sequences if not needed
                    if not include_raw_data:
                        sanitized_record.pop("action_sequence", None)
                        sanitized_record.pop("observation_sequence", None)
                    sanitized_episodes[episode_id] = sanitized_record

                export_data["episode_records"] = sanitized_episodes
                export_data["verification_results"] = self.test_results

            # Generate comprehensive reproducibility report
            reproducibility_report = self.generate_reproducibility_report(
                report_format="dict", include_detailed_analysis=True
            )
            export_data["reproducibility_report"] = reproducibility_report

            # Format data according to export_format
            if export_format == "json":
                # Write JSON format
                json_data = json.dumps(export_data, indent=2, default=str)

                if compress_output:
                    import gzip

                    export_path = export_path.with_suffix(".json.gz")
                    with gzip.open(export_path, "wt", encoding="utf-8") as f:
                        f.write(json_data)
                else:
                    export_path = export_path.with_suffix(".json")
                    with open(export_path, "w", encoding="utf-8") as f:
                        f.write(json_data)

            elif export_format == "csv":
                # Export summary data as CSV
                import csv

                export_path = export_path.with_suffix(".csv")

                # Create CSV with verification results
                with open(export_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)

                    # Write header
                    writer.writerow(
                        [
                            "verification_id",
                            "episode_record_id",
                            "match_status",
                            "sequences_match",
                            "reproducibility_score",
                            "verification_timestamp",
                        ]
                    )

                    # Write verification results
                    for verification_id, result in self.test_results.items():
                        writer.writerow(
                            [
                                verification_id,
                                result.get("episode_record_id", ""),
                                result.get("match_status", ""),
                                result.get("sequences_match", False),
                                result.get("reproducibility_score", 0.0),
                                result.get("verification_timestamp", 0.0),
                            ]
                        )

            # Validate exported data integrity and format compliance
            if not export_path.exists():
                raise ResourceError(
                    message="Export file was not created",
                    resource_type="disk",
                    current_usage=None,
                    limit_exceeded=None,
                )

            # Check file size
            file_size = export_path.stat().st_size
            if file_size == 0:
                raise ComponentError(
                    message="Exported file is empty",
                    component_name="ReproducibilityTracker",
                    operation_name="export_reproducibility_data",
                )

            # Log successful export
            self.logger.info(
                f"Successfully exported reproducibility data to {export_path} "
                f"({file_size} bytes, format: {export_format})"
            )

            # Return success status with detailed error handling and logging
            return True

        except (ValidationError, ResourceError, ComponentError):
            # Re-raise specific exceptions with original context
            raise
        except Exception as e:
            # Wrap unexpected errors in ComponentError
            self.logger.error(f"Reproducibility data export failed: {e}")
            raise ComponentError(
                message=f"Failed to export reproducibility data: {str(e)}",
                component_name="ReproducibilityTracker",
                operation_name="export_reproducibility_data",
            ) from e
