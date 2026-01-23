"""Seeding utilities and reproducibility helpers."""

import hashlib
import json
import logging
import os
import pathlib
import threading
import time
import uuid
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy

try:
    import gymnasium.utils.seeding  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - minimal stub for trimmed environments
    import sys
    import types

    def _install_minimal_gymnasium_seeding_stub() -> None:
        """Install a minimal gymnasium.utils.seeding stub."""

        gym_module = sys.modules.get("gymnasium")
        if gym_module is None:
            gym_module = types.ModuleType("gymnasium")
            gym_module.__path__ = []  # type: ignore[attr-defined]
            sys.modules["gymnasium"] = gym_module

        utils_module = sys.modules.get("gymnasium.utils")
        if utils_module is None:
            utils_module = types.ModuleType("gymnasium.utils")
            sys.modules["gymnasium.utils"] = utils_module
            gym_module.utils = utils_module  # type: ignore[attr-defined]

        seeding_module = types.ModuleType("gymnasium.utils.seeding")

        def np_random(seed=None):  # type: ignore[override]
            rng = numpy.random.default_rng(seed)
            used_seed = 0 if seed is None else int(seed)
            return rng, used_seed

        seeding_module.np_random = np_random  # type: ignore[attr-defined]
        sys.modules["gymnasium.utils.seeding"] = seeding_module
        utils_module.seeding = seeding_module  # type: ignore[attr-defined]

    _install_minimal_gymnasium_seeding_stub()

from ..constants import SEED_MAX_VALUE, SEED_MIN_VALUE, VALID_SEED_TYPES
from .exceptions import ComponentError, ResourceError, StateError, ValidationError

_logger = logging.getLogger(__name__)
_SEED_STATE_VERSION = "1"
_HASH_ALGORITHM = "sha256"
_DEFAULT_ENCODING = "utf-8"
_REPRODUCIBILITY_TOLERANCE = 1e-10

if TYPE_CHECKING:  # pragma: no cover
    from ..core.geometry import Coordinates, GridSize

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
    """Validate seed parameters with type checking, range validation, and reproducibility compliance."""
    try:
        if seed is None:
            return (True, None, "")

        if not isinstance(seed, tuple(VALID_SEED_TYPES)):
            return (
                False,
                None,
                f"Seed must be integer type, got {type(seed).__name__}",
            )

        if hasattr(seed, "item"):
            seed = int(seed.item())
        else:
            seed = int(seed)

        if seed < 0:
            return (
                False,
                None,
                f"Seed must be non-negative, got {seed} (range: [{SEED_MIN_VALUE}, {SEED_MAX_VALUE}])",
            )

        if seed > SEED_MAX_VALUE:
            return (
                False,
                None,
                f"Seed {seed} exceeds maximum {SEED_MAX_VALUE} (range: [{SEED_MIN_VALUE}, {SEED_MAX_VALUE}])",
            )

        if seed > 2**31 - 1:
            warnings.warn(
                f"Seed {seed} may cause integer overflow in some systems",
                UserWarning,
                stacklevel=2,
            )

        return (True, seed, "")

    except Exception as e:
        _logger.error(f"Seed validation failed with exception: {e}")
        return (False, None, f"Seed validation error: {str(e)}")


def create_seeded_rng(
    seed: Optional[int] = None, validate_input: bool = True
) -> Tuple[numpy.random.Generator, int]:
    """Main function for creating gymnasium-compatible seeded random number generators for deterministic environment behavior with proper initialization, state management, and Gymnasium integration compliance."""
    try:
        if validate_input:
            is_valid, normalized_seed, error_message = validate_seed(seed)

            if not is_valid:
                _logger.error(f"Seed validation failed: {error_message}")
                raise ValidationError(
                    message=f"Invalid seed for RNG creation: {error_message}",
                    parameter_name="seed",
                    parameter_value=seed,
                    expected_format="integer or None",
                )

            seed = normalized_seed

        if seed is None:
            np_random, seed_used = gymnasium.utils.seeding.np_random(seed)
        else:
            np_random, seed_used = gymnasium.utils.seeding.np_random(seed)

        if seed_used is None:
            seed_used = get_random_seed(use_system_entropy=True)

        if not isinstance(np_random, numpy.random.Generator):
            raise StateError(
                message="Failed to create numpy.random.Generator from gymnasium seeding",
                current_state="invalid_generator",
                expected_state="numpy.random.Generator",
                component_name="seeding",
            )

        _logger.debug(f"Created seeded RNG with seed: {seed_used}")

        return (np_random, seed_used)

    except ValidationError:
        raise
    except Exception as e:
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
    """Utility function for generating reproducible seeds from string identifiers using cryptographic hash functions, enabling experiment naming and configuration-based seeding for scientific research workflows."""
    try:
        if not isinstance(seed_string, str) or not seed_string.strip():
            raise ValidationError(
                message="Seed string must be a non-empty string",
                parameter_name="seed_string",
                parameter_value=seed_string,
                expected_format="non-empty string",
            )

        if hash_algorithm not in hashlib.algorithms_available:
            available_algorithms = ", ".join(
                sorted(list(hashlib.algorithms_available)[:5])
            )
            raise ValidationError(
                message=f"Hash algorithm '{hash_algorithm}' not available",
                parameter_name="hash_algorithm",
                parameter_value=hash_algorithm,
                expected_format=f"one of: {available_algorithms}...",
            )

        hash_obj = hashlib.new(hash_algorithm)

        try:
            encoded_string = seed_string.encode(encoding)
        except UnicodeEncodeError as e:
            raise ValidationError(
                message=f"Failed to encode string with {encoding}: {str(e)}",
                parameter_name="seed_string",
                parameter_value=seed_string,
                expected_format=f"string encodable with {encoding}",
            ) from e

        hash_obj.update(encoded_string)
        hash_digest = hash_obj.digest()

        seed_int = int.from_bytes(hash_digest, byteorder="big")

        deterministic_seed = seed_int % (SEED_MAX_VALUE + 1)

        _logger.debug(
            f"Generated deterministic seed {deterministic_seed} from string '{seed_string[:50]}...'"
        )

        return deterministic_seed

    except ValidationError:
        raise
    except Exception as e:
        _logger.error(f"Deterministic seed generation failed: {e}")
        raise ComponentError(
            message=f"Failed to generate deterministic seed from string: {str(e)}",
            component_name="seeding",
            operation_name="generate_deterministic_seed",
        ) from e


def verify_reproducibility(
    rng1: numpy.random.Generator,
    rng2: numpy.random.Generator,
    sequence_length: int = 1000,
    tolerance: float = _REPRODUCIBILITY_TOLERANCE,
) -> Dict[str, Any]:  # noqa: C901
    """Function for verifying deterministic behavior of random number generation through sequence comparison, statistical analysis, and tolerance-based validation for scientific reproducibility requirements."""
    try:
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

        if not isinstance(sequence_length, int) or sequence_length <= 0:
            raise ValidationError(
                message="Sequence length must be positive integer",
                parameter_name="sequence_length",
                parameter_value=sequence_length,
                expected_format="positive integer",
            )

        if not isinstance(tolerance, (int, float)) or tolerance <= 0:
            raise ValidationError(
                message="Tolerance must be positive number",
                parameter_name="tolerance",
                parameter_value=tolerance,
                expected_format="positive float",
            )

        _logger.debug(
            f"Generating sequences of length {sequence_length} for reproducibility verification"
        )

        sequence1 = rng1.random(sequence_length)
        sequence2 = rng2.random(sequence_length)

        sequences_match = numpy.allclose(
            sequence1, sequence2, atol=tolerance, rtol=tolerance
        )

        absolute_errors = numpy.abs(sequence1 - sequence2)
        mean_absolute_error = float(numpy.mean(absolute_errors))
        max_deviation = float(numpy.max(absolute_errors))
        min_deviation = float(numpy.min(absolute_errors))

        exact_matches = numpy.sum(sequence1 == sequence2)
        exact_match_percentage = float(exact_matches / sequence_length * 100.0)

        discrepancy_indices = []
        if not sequences_match:
            discrepancies = numpy.where(absolute_errors > tolerance)[0]
            discrepancy_indices = discrepancies.tolist()[:10]

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

        _logger.debug(
            f"Reproducibility verification: {reproducibility_report['status']} "
            f"(MAE: {mean_absolute_error:.2e}, Max Dev: {max_deviation:.2e})"
        )

        return reproducibility_report

    except ValidationError:
        raise
    except Exception as e:
        _logger.error(f"Reproducibility verification failed: {e}")
        raise ComponentError(
            message=f"Failed to verify reproducibility: {str(e)}",
            component_name="seeding",
            operation_name="verify_reproducibility",
        ) from e


def get_random_seed(
    use_system_entropy: bool = True, fallback_method: Optional[int] = None
) -> int:
    """Function for generating high-quality random seeds from system entropy sources providing cryptographically secure seed generation for non-reproducible research scenarios and initial seed creation."""
    try:
        if use_system_entropy:
            try:
                entropy_bytes = os.urandom(4)

                random_seed = int.from_bytes(
                    entropy_bytes, byteorder="big", signed=False
                )

                random_seed = random_seed % (SEED_MAX_VALUE + 1)

                _logger.debug(
                    f"Generated random seed {random_seed} from system entropy"
                )
                return random_seed

            except (OSError, NotImplementedError) as e:
                _logger.warning(
                    f"System entropy not available: {e}, falling back to alternative method"
                )

        if fallback_method == 1 or use_system_entropy is False:
            current_time = time.time()
            microseconds = int((current_time % 1) * 1_000_000)

            try:
                process_entropy = os.getpid() if hasattr(os, "getpid") else 12345
                thread_entropy = threading.current_thread().ident or 67890

                combined_entropy = (
                    microseconds * 1000 + process_entropy
                ) ^ thread_entropy

                fallback_seed = combined_entropy % (SEED_MAX_VALUE + 1)

                _logger.debug(
                    f"Generated fallback seed {fallback_seed} from time and process entropy"
                )
                return fallback_seed

            except Exception as e:
                _logger.warning(f"Enhanced fallback entropy failed: {e}")

        timestamp_seed = int(time.time() * 1_000_000) % (SEED_MAX_VALUE + 1)

        is_valid, validated_seed, error_message = validate_seed(timestamp_seed)
        if not is_valid:
            raise ComponentError(
                message=f"Generated seed failed validation: {error_message}",
                component_name="seeding",
                operation_name="get_random_seed",
            )

        _logger.debug(f"Generated timestamp-based seed {validated_seed}")

        return validated_seed

    except Exception as e:
        _logger.error(f"Random seed generation failed: {e}")
        raise ResourceError(
            message=f"Failed to generate random seed: {str(e)}",
            resource_type="entropy",
            current_usage=None,
            limit_exceeded=None,
        ) from e


def save_seed_state(
    rng: numpy.random.Generator,
    file_path: Union[str, pathlib.Path],
    metadata: Optional[Dict[str, Any]] = None,
    create_backup: bool = False,
) -> bool:  # noqa: C901
    """Function for saving random number generator state to file for experiment reproduction, state persistence, and cross-session reproducibility with JSON serialization and metadata inclusion."""
    try:
        if not isinstance(rng, numpy.random.Generator):
            raise ValidationError(
                message="RNG must be numpy.random.Generator instance",
                parameter_name="rng",
                parameter_value=type(rng).__name__,
                expected_format="numpy.random.Generator",
            )

        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        elif not isinstance(file_path, pathlib.Path):
            raise ValidationError(
                message="File path must be string or pathlib.Path",
                parameter_name="file_path",
                parameter_value=type(file_path).__name__,
                expected_format="str or pathlib.Path",
            )

        file_path.parent.mkdir(parents=True, exist_ok=True)

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

        try:
            rng_state = rng.bit_generator.state
        except AttributeError as e:
            raise ComponentError(
                message=f"Failed to extract RNG state: {str(e)}",
                component_name="seeding",
                operation_name="save_seed_state",
            ) from e

        def _to_jsonable(obj):
            try:
                import numpy as _np
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

        state_data = {
            "version": _SEED_STATE_VERSION,
            "timestamp": time.time(),
            "rng_state": serialized_rng_state,
            "state": serialized_rng_state,
            "generator_type": type(rng.bit_generator).__name__,
        }

        if metadata:
            sanitized_metadata = {}
            for key, value in metadata.items():
                if isinstance(key, str) and not any(
                    sensitive in key.lower()
                    for sensitive in ["password", "token", "key", "secret"]
                ):
                    sanitized_metadata[key] = value
            state_data["metadata"] = sanitized_metadata

        temp_path = file_path.with_suffix(f"{file_path.suffix}.tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2, separators=(",", ": "))

            temp_path.replace(file_path)

        except OSError:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise

        if not file_path.exists():
            raise ResourceError(
                message="Seed state file was not created successfully",
                resource_type="disk",
                current_usage=None,
                limit_exceeded=None,
            )

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
        raise
    except OSError:
        raise
    except Exception as e:
        _logger.error(f"Seed state save failed: {e}")
        raise ComponentError(
            message=f"Unexpected error saving seed state: {str(e)}",
            component_name="seeding",
            operation_name="save_seed_state",
        ) from e


def load_seed_state(
    file_path: Union[str, pathlib.Path],
    validate_state: bool = True,
    strict_version_check: bool = False,
) -> Tuple[numpy.random.Generator, Dict[str, Any]]:  # noqa: C901
    """Function for loading saved random number generator state from file for experiment reproduction and state restoration with validation, error handling, and compatibility checking."""
    try:
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

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                state_data = json.load(f)
        except OSError as e:
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

        base_required_fields = ["version", "timestamp"]
        for field in base_required_fields:
            if field not in state_data:
                raise ValidationError(
                    message=f"Missing required field '{field}' in seed state file",
                    parameter_name="state_data",
                    parameter_value=list(state_data.keys()),
                    expected_format=f"dict with fields: {base_required_fields} + state",
                )

        state_version = state_data["version"]
        if strict_version_check and state_version != _SEED_STATE_VERSION:
            raise ValidationError(
                message=f"State version mismatch: file has {state_version}, expected {_SEED_STATE_VERSION}",
                parameter_name="version",
                parameter_value=state_version,
                expected_format=_SEED_STATE_VERSION,
            )

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

        if "bit_generator" not in rng_state_data or "state" not in rng_state_data:
            raise ValidationError(
                message="RNG state missing required fields (bit_generator, state)",
                parameter_name="rng_state",
                parameter_value=list(rng_state_data.keys()),
                expected_format="dict with bit_generator and state fields",
            )

        try:
            generator_type = state_data.get(
                "generator_type", rng_state_data.get("bit_generator", "PCG64")
            )

            if generator_type == "PCG64":
                bit_generator = numpy.random.PCG64()
            elif generator_type == "PCG64DXSM":
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
                _logger.warning(f"Unknown generator type {generator_type}, using PCG64")
                bit_generator = numpy.random.PCG64()

            restored_rng = numpy.random.Generator(bit_generator)

            restored_state = dict(rng_state_data)
            nested_state = restored_state.get("state")
            try:
                if isinstance(nested_state, dict):
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
                restored_state = dict(rng_state_data)

            restored_rng.bit_generator.state = restored_state

        except (ValueError, TypeError, KeyError) as e:
            raise ComponentError(
                message=f"Failed to restore RNG state: {str(e)}",
                component_name="seeding",
                operation_name="load_seed_state",
            ) from e

        if validate_state:
            try:
                test_values = restored_rng.random(5)
                if not isinstance(test_values, numpy.ndarray) or len(test_values) != 5:
                    raise ComponentError(
                        message="Restored RNG failed validation test",
                        component_name="seeding",
                        operation_name="load_seed_state",
                    )

                restored_rng.bit_generator.state = restored_state

                _logger.debug("Restored RNG passed validation tests and state reset")

            except Exception as e:
                raise ComponentError(
                    message=f"RNG validation failed: {str(e)}",
                    component_name="seeding",
                    operation_name="load_seed_state",
                ) from e

        metadata = state_data.get("metadata", {})
        metadata["load_timestamp"] = time.time()
        metadata["original_timestamp"] = state_data["timestamp"]
        metadata["version"] = state_version

        _logger.debug(f"Successfully loaded seed state from {file_path}")

        return (restored_rng, metadata)

    except (ValidationError, ResourceError, ComponentError):
        raise
    except (FileNotFoundError, PermissionError, json.JSONDecodeError):
        raise
    except Exception as e:
        _logger.error(f"Seed state load failed: {e}")
        raise ComponentError(
            message=f"Unexpected error loading seed state: {str(e)}",
            component_name="seeding",
            operation_name="load_seed_state",
        ) from e


class SeedManager:
    """Centralized seed management class with validation, reproducibility tracking, thread safety, and performance optimization for scientific research applications requiring deterministic behavior and experiment reproducibility."""

    def __init__(
        self,
        default_seed: Optional[int] = None,
        enable_validation: bool = True,
        thread_safe: bool = False,
    ):
        """Initialize SeedManager with configuration options, thread safety setup, and validation framework for centralized seed management."""
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

        self.enable_validation = bool(enable_validation)

        self.thread_safe = bool(thread_safe)
        self._lock: Optional[threading.RLock]
        if self.thread_safe:
            self._lock = threading.RLock()
        else:
            self._lock = None

        self.active_generators: Dict[str, Dict[str, Any]] = {}

        self._episode_rng_context: Optional[str] = None

        self.seed_history: List[Dict[str, Any]] = []

        self.logger = logging.getLogger(f"{__name__}.SeedManager")

        self.logger.debug(
            f"Initialized SeedManager (default_seed={self.default_seed}, "
            f"validation={self.enable_validation}, thread_safe={self.thread_safe})"
        )

    def seed(
        self, seed: Optional[int] = None, context_id: Optional[str] = None
    ) -> Tuple[numpy.random.Generator, int]:  # noqa: C901
        """Primary seeding method for creating and managing random number generators with validation, tracking, and thread safety for deterministic environment behavior."""
        try:
            if self._lock:
                self._lock.acquire()

            try:
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

                np_random, seed_used = create_seeded_rng(
                    effective_seed, validate_input=False
                )

                if context_id is None:
                    context_id = f"seed_manager_{uuid.uuid4().hex[:8]}"

                generator_info = {
                    "generator": np_random,
                    "seed_used": seed_used,
                    "creation_timestamp": time.time(),
                    "access_count": 0,
                    "last_access_timestamp": time.time(),
                }
                self.active_generators[context_id] = generator_info

                if context_id.startswith("episode_reset_"):
                    self._episode_rng_context = context_id

                history_entry = {
                    "operation": "seed",
                    "context_id": context_id,
                    "seed_requested": seed,
                    "seed_used": seed_used,
                    "timestamp": time.time(),
                    "validation_enabled": self.enable_validation,
                }
                self.seed_history.append(history_entry)

                if len(self.seed_history) > 1000:
                    self.seed_history = self.seed_history[-500:]

                self.logger.debug(
                    f"Created seeded generator: context_id={context_id}, seed={seed_used}"
                )

                return (np_random, seed_used)

            finally:
                if self._lock:
                    self._lock.release()

        except ValidationError:
            raise
        except Exception as e:
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
        """Generate deterministic episode seeds for reproducible episodes with episode context integration and sequence management for scientific research workflows."""
        try:
            is_valid, validated_base_seed, error_message = validate_seed(base_seed)
            if not is_valid:
                raise ValidationError(
                    message=f"Invalid base seed for episode generation: {error_message}",
                    parameter_name="base_seed",
                    parameter_value=base_seed,
                    expected_format="valid integer seed",
                )

            if not isinstance(episode_number, int) or episode_number < 0:
                raise ValidationError(
                    message="Episode number must be non-negative integer",
                    parameter_name="episode_number",
                    parameter_value=episode_number,
                    expected_format="non-negative integer",
                )

            if experiment_id:
                context_string = f"{experiment_id}_episode_{episode_number}"
            else:
                context_string = f"episode_{episode_number}"

            context_seed = generate_deterministic_seed(context_string)

            combined_seed = (
                validated_base_seed + context_seed + episode_number * 1009
            ) % (SEED_MAX_VALUE + 1)

            episode_seed = combined_seed % (SEED_MAX_VALUE + 1)

            self.logger.debug(
                f"Generated episode seed {episode_seed} for episode {episode_number} "
                f"(base: {validated_base_seed}, experiment: {experiment_id})"
            )

            return episode_seed

        except ValidationError:
            raise
        except Exception as e:
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
    ) -> "Coordinates":  # noqa: C901
        """Generate random position within grid bounds using current RNG."""
        from ..core.geometry import Coordinates, GridSize

        try:
            if not isinstance(grid_size, GridSize):
                raise ValidationError(
                    message="grid_size must be GridSize instance",
                    parameter_name="grid_size",
                    parameter_value=str(type(grid_size)),
                )

            if self._lock:
                self._lock.acquire()

            try:
                generator_info = None
                context_id = None

                if (
                    self._episode_rng_context
                    and self._episode_rng_context in self.active_generators
                ):
                    context_id = self._episode_rng_context
                    generator_info = self.active_generators[context_id]

                if generator_info is None:
                    if self.seed_history:
                        last_entry = self.seed_history[-1]
                        seed_value = last_entry.get("seed_used")
                        if seed_value is None:
                            raise StateError(
                                message="Seed history missing seed_used entry",
                                current_state="generate_random_position",
                                expected_state="seed_recorded",
                            )
                    else:
                        if self.default_seed is None:
                            raise StateError(
                                message="No seed history or default seed available",
                                current_state="generate_random_position",
                                expected_state="seed_initialized",
                            )
                        seed_value = self.default_seed

                    rng, seed_used = create_seeded_rng(seed_value, validate_input=False)
                    context_id = f"generate_position_{uuid.uuid4().hex[:8]}"
                    generator_info = {
                        "generator": rng,
                        "seed_used": seed_used,
                        "creation_timestamp": time.time(),
                        "access_count": 0,
                        "last_access_timestamp": time.time(),
                    }
                    self.active_generators[context_id] = generator_info
                else:
                    rng = generator_info["generator"]

                generator_info["access_count"] += 1
                generator_info["last_access_timestamp"] = time.time()

                for _attempt in range(max_attempts):
                    x = int(rng.integers(0, grid_size.width))
                    y = int(rng.integers(0, grid_size.height))
                    position = Coordinates(x=x, y=y)

                    if exclude_position is None:
                        return position

                    if (
                        position.x != exclude_position.x
                        or position.y != exclude_position.y
                    ):
                        return position

            finally:
                if self._lock:
                    self._lock.release()

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

    def validate_reproducibility(
        self,
        test_seed: int,
        num_tests: int = 10,
        tolerance: float = _REPRODUCIBILITY_TOLERANCE,
    ) -> Dict[str, Any]:  # noqa: C901
        """Validate reproducibility of seeded generators through comprehensive testing, statistical analysis, and tolerance-based comparison for scientific research requirements."""
        try:
            is_valid, validated_seed, error_message = validate_seed(test_seed)
            if not is_valid:
                raise ValidationError(
                    message=f"Invalid test seed: {error_message}",
                    parameter_name="test_seed",
                    parameter_value=test_seed,
                    expected_format="valid integer seed",
                )

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

            test_results = []
            success_count = 0

            for test_idx in range(num_tests):
                try:
                    rng1, _ = create_seeded_rng(validated_seed, validate_input=False)
                    rng2, _ = create_seeded_rng(validated_seed, validate_input=False)

                    verification_result = verify_reproducibility(
                        rng1,
                        rng2,
                        sequence_length=100,
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

            success_rate = success_count / num_tests

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

            self.logger.info(
                f"Reproducibility validation completed: {validation_report['overall_status']} "
                f"(success rate: {success_rate:.1%})"
            )

            return validation_report

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Reproducibility validation failed: {e}")
            raise ComponentError(
                message=f"Failed to validate reproducibility: {str(e)}",
                component_name="SeedManager",
                operation_name="validate_reproducibility",
            ) from e

    def get_active_generators(self) -> Dict[str, Dict[str, Any]]:
        """Get list of active generators with context information for monitoring, debugging, and resource management in multi-context seeding scenarios."""
        try:
            if self._lock:
                self._lock.acquire()

            try:
                generator_summary = {}
                current_time = time.time()

                for context_id, generator_info in self.active_generators.items():
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

                    try:
                        generator_info["generator"].random()
                        summary_info["status"] = "active"
                        summary_info["last_test_successful"] = True

                        generator_info["access_count"] += 1
                        generator_info["last_access_timestamp"] = current_time

                    except Exception as e:
                        summary_info["status"] = "error"
                        summary_info["last_test_successful"] = False
                        summary_info["error_message"] = str(e)

                    generator_summary[context_id] = summary_info

                active_generators_report = {
                    "total_active_generators": len(generator_summary),
                    "report_timestamp": current_time,
                    "generators": generator_summary,
                    "memory_usage_estimate": len(generator_summary) * 1024,
                }

                return active_generators_report

            finally:
                if self._lock:
                    self._lock.release()

        except Exception as e:
            self.logger.warning(f"Failed to get active generators info: {e}")
            return {
                "total_active_generators": 0,
                "report_timestamp": time.time(),
                "generators": {},
                "error": str(e),
            }

    def reset(self, preserve_default_seed: bool = True) -> None:
        """Reset SeedManager state clearing active generators and seed history for clean state management and resource cleanup."""
        try:
            if self._lock:
                self._lock.acquire()

            try:
                num_generators = len(self.active_generators)
                self.active_generators.clear()

                history_entries = len(self.seed_history)
                self.seed_history.clear()

                if not preserve_default_seed:
                    self.default_seed = None

                self.logger.info(
                    f"SeedManager reset: cleared {num_generators} generators, "
                    f"{history_entries} history entries (preserve_default={preserve_default_seed})"
                )

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
                if self._lock:
                    self._lock.release()

        except Exception as e:
            self.logger.error(f"Error during SeedManager reset: {e}")


class ReproducibilityTracker:
    """Specialized reproducibility tracking and validation class for scientific research with comprehensive episode comparison, statistical analysis, failure diagnostics, and research-grade reporting capabilities."""

    def __init__(
        self,
        tolerance: float = _REPRODUCIBILITY_TOLERANCE,
        session_id: Optional[str] = None,
    ):
        """Initialize ReproducibilityTracker for episode recording and reproducibility verification."""
        if not isinstance(tolerance, (int, float)) or tolerance <= 0:
            raise ValidationError(
                message="Tolerance must be positive number",
                parameter_name="tolerance",
                parameter_value=tolerance,
                expected_format="positive float",
            )
        self.tolerance = float(tolerance)

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

        self.episode_records: Dict[str, Dict[str, Any]] = {}

        self.test_results: Dict[str, Dict[str, Any]] = {}

        self.statistical_summaries: List[Dict[str, Any]] = []

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
        """Record episode data for reproducibility tracking."""
        try:
            is_valid, validated_seed, error_message = validate_seed(episode_seed)
            if not is_valid:
                raise ValidationError(
                    message=f"Invalid episode seed: {error_message}",
                    parameter_name="episode_seed",
                    parameter_value=episode_seed,
                    expected_format="valid integer seed",
                )

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

            episode_record_id = f"episode_{uuid.uuid4().hex[:16]}"

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

            if metadata:
                if isinstance(metadata, dict):
                    episode_record["metadata"] = metadata
                else:
                    self.logger.warning(
                        f"Metadata is not a dictionary, skipping: {type(metadata)}"
                    )

            self.episode_records[episode_record_id] = episode_record

            self.logger.debug(
                f"Recorded episode {episode_record_id}: seed={validated_seed}, "
                f"actions={len(action_sequence)}, observations={len(observation_sequence)}"
            )

            return episode_record_id

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Episode recording failed: {e}")
            raise ComponentError(
                message=f"Failed to record episode: {str(e)}",
                component_name="ReproducibilityTracker",
                operation_name="record_episode",
            ) from e

    def verify_episode_reproducibility(
        self,
        episode_record_id: str,
        new_action_sequence: List[Any],
        new_observation_sequence: List[Any],
        custom_tolerance: Optional[float] = None,
    ) -> Dict[str, Any]:  # noqa: C901
        """Verify episode reproducibility by comparing recorded episode with new execution using detailed sequence comparison and statistical analysis."""
        try:
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

            if not isinstance(new_action_sequence, list) or not isinstance(
                new_observation_sequence, list
            ):
                raise ValidationError(
                    message="New sequences must be lists",
                    parameter_name="new_sequences",
                    parameter_value=f"actions: {type(new_action_sequence)}, obs: {type(new_observation_sequence)}",
                    expected_format="lists",
                )

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

                self.test_results[verification_result["verification_id"]] = (
                    verification_result
                )

                return verification_result

            tolerance = (
                custom_tolerance if custom_tolerance is not None else self.tolerance
            )

            action_comparison = self._compare_sequences(
                original_actions, new_action_sequence, tolerance
            )

            observation_comparison = self._compare_sequences(
                original_observations, new_observation_sequence, tolerance
            )

            overall_match = (
                action_comparison["sequences_match"]
                and observation_comparison["sequences_match"]
            )

            action_score = action_comparison.get("reproducibility_score", 0.0)
            observation_score = observation_comparison.get("reproducibility_score", 0.0)
            combined_score = (action_score + observation_score) / 2.0

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

            if overall_match:
                verification_result["status_message"] = (
                    "Episode successfully reproduced"
                )
            else:
                verification_result["status_message"] = (
                    "Episode reproducibility verification failed"
                )

            self.test_results[verification_result["verification_id"]] = (
                verification_result
            )

            self.logger.debug(
                f"Episode verification {verification_result['verification_id']}: "
                f"{verification_result['match_status']} (score: {combined_score:.3f})"
            )

            return verification_result

        except ValidationError:
            raise
        except Exception as e:
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

            try:
                arr1 = numpy.array(seq1, dtype=float)
                arr2 = numpy.array(seq2, dtype=float)

                differences = numpy.abs(arr1 - arr2)

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

                if not comparison_result["sequences_match"]:
                    discrepancy_mask = ~within_tolerance
                    comparison_result["discrepancy_indices"] = numpy.where(
                        discrepancy_mask
                    )[0].tolist()

            except (ValueError, TypeError):
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

    def generate_reproducibility_report(
        self,
        report_format: Optional[str] = "dict",
        include_detailed_analysis: bool = True,
        custom_sections: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:  # noqa: C901
        """Generate comprehensive reproducibility report for scientific documentation including statistical summaries, success rates, and detailed analysis."""
        try:
            report_timestamp = time.time()

            total_episodes = len(self.episode_records)
            total_verifications = len(self.test_results)

            if total_verifications > 0:
                successful_verifications = [
                    v
                    for v in self.test_results.values()
                    if v.get("sequences_match", False)
                ]
                success_rate = len(successful_verifications) / total_verifications

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
                    failure_types = {}
                    for test in failed_tests:
                        failure_type = test.get("match_status", "UNKNOWN")
                        failure_types[failure_type] = (
                            failure_types.get(failure_type, 0) + 1
                        )
                    failure_analysis["failure_types"] = failure_types

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

                    discrepancies.sort(
                        key=lambda x: x["discrepancy_magnitude"], reverse=True
                    )
                    failure_analysis["largest_discrepancies"] = discrepancies[:5]

                report["failure_analysis"] = failure_analysis

                episode_statistics = {
                    "episode_length_distribution": self._analyze_episode_lengths(),
                    "seed_distribution": self._analyze_seed_distribution(),
                    "temporal_analysis": self._analyze_temporal_patterns(),
                }
                report["episode_statistics"] = episode_statistics

            if custom_sections and isinstance(custom_sections, dict):
                for section_name, section_content in custom_sections.items():
                    if isinstance(section_name, str) and section_name not in report:
                        report[f"custom_{section_name}"] = section_content

            if report_format == "json":
                import json

                return {"json_report": json.dumps(report, indent=2, default=str)}
            elif report_format == "markdown":
                return {"markdown_report": self._format_markdown_report(report)}
            else:
                return report

        except Exception as e:
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

        summary = report["summary_statistics"]
        markdown += "## Summary Statistics\n\n"
        markdown += f"- Total Episodes Recorded: {summary['total_episodes_recorded']}\n"
        markdown += (
            f"- Total Verifications: {summary['total_verifications_performed']}\n"
        )
        markdown += f"- Success Rate: {summary['overall_success_rate']:.2%}\n\n"

        assessment = report["reproducibility_assessment"]
        markdown += f"## Overall Assessment: {assessment['overall_status']}\n\n"
        markdown += f"**Confidence Level:** {assessment['confidence_level']}\n\n"

        if assessment.get("recommendations"):
            markdown += "### Recommendations\n\n"
            for rec in assessment["recommendations"]:
                markdown += f"- {rec}\n"

        return markdown

    def export_reproducibility_data(
        self,
        export_path: Union[str, pathlib.Path],
        export_format: str = "json",
        include_raw_data: bool = True,
        compress_output: bool = False,
    ) -> bool:  # noqa: C901
        """Export reproducibility data for external analysis and archival with multiple format support and data integrity validation."""
        try:
            if isinstance(export_path, str):
                export_path = pathlib.Path(export_path)
            elif not isinstance(export_path, pathlib.Path):
                raise ValidationError(
                    message="Export path must be string or pathlib.Path",
                    parameter_name="export_path",
                    parameter_value=type(export_path).__name__,
                    expected_format="str or pathlib.Path",
                )

            export_path.parent.mkdir(parents=True, exist_ok=True)

            supported_formats = ["json", "csv"]
            if export_format not in supported_formats:
                raise ValidationError(
                    message=f"Unsupported export format: {export_format}",
                    parameter_name="export_format",
                    parameter_value=export_format,
                    expected_format=f"one of: {supported_formats}",
                )

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

            if include_raw_data:
                sanitized_episodes = {}
                for episode_id, record in self.episode_records.items():
                    sanitized_record = record.copy()
                    if not include_raw_data:
                        sanitized_record.pop("action_sequence", None)
                        sanitized_record.pop("observation_sequence", None)
                    sanitized_episodes[episode_id] = sanitized_record

                export_data["episode_records"] = sanitized_episodes
                export_data["verification_results"] = self.test_results

            reproducibility_report = self.generate_reproducibility_report(
                report_format="dict", include_detailed_analysis=True
            )
            export_data["reproducibility_report"] = reproducibility_report

            if export_format == "json":
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
                import csv

                export_path = export_path.with_suffix(".csv")

                with open(export_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)

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

            if not export_path.exists():
                raise ResourceError(
                    message="Export file was not created",
                    resource_type="disk",
                    current_usage=None,
                    limit_exceeded=None,
                )

            file_size = export_path.stat().st_size
            if file_size == 0:
                raise ComponentError(
                    message="Exported file is empty",
                    component_name="ReproducibilityTracker",
                    operation_name="export_reproducibility_data",
                )

            self.logger.info(
                f"Successfully exported reproducibility data to {export_path} "
                f"({file_size} bytes, format: {export_format})"
            )

            return True

        except (ValidationError, ResourceError, ComponentError):
            raise
        except Exception as e:
            self.logger.error(f"Reproducibility data export failed: {e}")
            raise ComponentError(
                message=f"Failed to export reproducibility data: {str(e)}",
                component_name="ReproducibilityTracker",
                operation_name="export_reproducibility_data",
            ) from e
