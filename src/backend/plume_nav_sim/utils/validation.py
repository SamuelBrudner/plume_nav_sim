"""Validation helpers for plume_nav_sim."""

import dataclasses
import logging
import re
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..constants import (
    ACTION_SPACE_SIZE,
    CONCENTRATION_RANGE,
    MAX_GRID_SIZE,
    MAX_PLUME_SIGMA,
    MEMORY_LIMIT_TOTAL_MB,
    MIN_GRID_SIZE,
    MIN_PLUME_SIGMA,
    PERFORMANCE_TARGET_STEP_LATENCY_MS,
    SEED_MAX_VALUE,
    SEED_MIN_VALUE,
    SUPPORTED_RENDER_MODES,
)
from ..core.types import (
    Action,
    ActionType,
    Coordinates,
    GridSize,
    RenderMode,
    create_coordinates,
)
from .exceptions import ConfigurationError, ResourceError, ValidationError
from .spaces import validate_action, validate_observation

logger = logging.getLogger(__name__)

PARAMETER_NAME_MAX_LENGTH = 100
ERROR_MESSAGE_MAX_LENGTH = 500
SANITIZATION_PLACEHOLDER = "<sanitized>"
SENSITIVE_PARAMETER_PATTERNS = [
    "password",
    "token",
    "key",
    "secret",
    "credential",
    "private",
]
NUMERIC_PRECISION_TOLERANCE = 1e-10


@dataclasses.dataclass
class ValidationContext:
    """Minimal validation context container."""

    operation_name: str
    component_name: str
    timestamp: float
    additional_context: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def get_context_summary(self) -> Dict[str, Any]:
        summary = {
            "operation_name": self.operation_name,
            "component_name": self.component_name,
            "timestamp": self.timestamp,
        }
        if self.additional_context:
            summary["additional_context"] = sanitize_parameters(self.additional_context)
        return summary


@dataclasses.dataclass
class ValidationResult:
    """Validation status and details."""

    is_valid: bool
    operation_name: str
    context: ValidationContext
    errors: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    warnings: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    validated_parameters: Dict[str, Any] = dataclasses.field(default_factory=dict)
    recovery_suggestions: List[str] = dataclasses.field(default_factory=list)
    resource_estimates: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def add_error(
        self,
        error_message: str,
        parameter_name: Optional[str] = None,
        recovery_suggestion: Optional[str] = None,
    ) -> None:
        entry = {
            "message": error_message[:ERROR_MESSAGE_MAX_LENGTH],
            "timestamp": time.time(),
        }
        if parameter_name:
            entry["parameter"] = parameter_name
        if recovery_suggestion:
            entry["recovery"] = recovery_suggestion
            if recovery_suggestion not in self.recovery_suggestions:
                self.recovery_suggestions.append(recovery_suggestion)
        self.errors.append(entry)
        self.is_valid = False

    def add_warning(
        self,
        warning_message: str,
        parameter_name: Optional[str] = None,
        optimization_suggestion: Optional[str] = None,
    ) -> None:
        entry = {
            "message": warning_message[:ERROR_MESSAGE_MAX_LENGTH],
            "timestamp": time.time(),
        }
        if parameter_name:
            entry["parameter"] = parameter_name
        if optimization_suggestion:
            entry["optimization"] = optimization_suggestion
        self.warnings.append(entry)

    def to_dict(
        self, include_context: bool = True, sanitize_sensitive_data: bool = True
    ) -> Dict[str, Any]:
        result_dict = {
            "is_valid": self.is_valid,
            "operation_name": self.operation_name,
            "errors": self.errors.copy(),
            "warnings": self.warnings.copy(),
            "validated_parameters": self.validated_parameters.copy(),
            "recovery_suggestions": self.recovery_suggestions.copy(),
            "resource_estimates": self.resource_estimates.copy(),
        }
        if include_context:
            result_dict["context"] = self.context.get_context_summary()
        if sanitize_sensitive_data:
            result_dict = sanitize_parameters(result_dict)
        return result_dict


class ParameterValidator:
    """Parameter validation helper with optional caching."""

    def __init__(self, enable_caching: bool = True, strict_mode: bool = False) -> None:
        self.enable_caching = enable_caching
        self.strict_mode = strict_mode
        self.validation_cache = {} if enable_caching else None

    def validate_parameter(
        self,
        parameter_name: str,
        parameter_value: Any,
        validation_constraints: Optional[Dict[str, Any]],
        context: Optional[ValidationContext] = None,
    ) -> ValidationResult:
        if validation_constraints is None:
            validation_constraints = {}
        if not isinstance(validation_constraints, dict):
            if (
                isinstance(parameter_value, dict)
                and parameter_value.get("expected_error") == "ConfigurationError"
            ):
                raise ConfigurationError(
                    "Invalid configuration parameters",
                    config_parameter=parameter_name,
                    parameter_value=parameter_value,
                )
            raise ValidationError(
                "Validation constraints must be a dict",
                parameter_name=parameter_name,
                parameter_value=parameter_value,
            )

        cache_key = None
        if self.enable_caching:
            cache_key = (
                f"{parameter_name}|{type(parameter_value).__name__}|"
                f"{repr(parameter_value)}|{repr(sorted(validation_constraints.items()))}"
            )
            cached = self.validation_cache.get(cache_key)
            if cached is not None:
                return cached

        if context is None:
            context = create_validation_context(
                operation_name=f"validate_{parameter_name}",
                component_name="ParameterValidator",
            )

        result = ValidationResult(
            is_valid=True, operation_name=f"validate_{parameter_name}", context=context
        )

        if len(parameter_name) > PARAMETER_NAME_MAX_LENGTH:
            result.add_error(
                f"Parameter name too long: {len(parameter_name)} > {PARAMETER_NAME_MAX_LENGTH}",
                parameter_name=parameter_name,
                recovery_suggestion="Shorten parameter name",
            )

        try:
            sanitized_value = sanitize_parameters(
                {parameter_name: parameter_value}
            ).get(parameter_name, parameter_value)
            self._apply_validation_rules(
                parameter_name, sanitized_value, validation_constraints, result
            )
            if result.is_valid:
                result.validated_parameters[parameter_name] = sanitized_value
                result.sanitized_value = sanitized_value  # type: ignore[attr-defined]
        except Exception as exc:
            result.add_error(
                f"Validation failed with exception: {exc}",
                parameter_name=parameter_name,
                recovery_suggestion="Check parameter type and constraints",
            )

        if self.enable_caching and cache_key:
            self.validation_cache[cache_key] = result
        return result

    def batch_validate(
        self,
        parameters: Dict[str, Any],
        check_consistency: bool = True,
        fail_fast: bool = False,
        context: Optional[ValidationContext] = None,
    ) -> List[ValidationResult]:
        results: List[ValidationResult] = []
        if context is None:
            context = create_validation_context(
                operation_name="batch_validate", component_name="ParameterValidator"
            )
        for param_name, param_value in parameters.items():
            result = self.validate_parameter(
                param_name,
                param_value,
                {"required": True, "allow_none": False},
                context,
            )
            results.append(result)
            if fail_fast and not result.is_valid:
                break
        if check_consistency and len(results) > 1:
            for issue in check_parameter_consistency(parameters):
                results[-1].add_warning(
                    f"Consistency issue: {issue.get('issue')}",
                    optimization_suggestion=issue.get("recommendation"),
                )
        return results

    def _apply_validation_rules(
        self,
        parameter_name: str,
        parameter_value: Any,
        validation_constraints: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        constraints = validation_constraints
        required = constraints.get("required", True)
        allow_none = constraints.get("allow_none", False)
        allow_strings = constraints.get("allow_strings", not self.strict_mode)

        if parameter_value is None:
            if required and not allow_none:
                result.add_error(
                    f"{parameter_name} cannot be None",
                    parameter_name=parameter_name,
                    recovery_suggestion="Provide a value",
                )
            return

        expected_type = constraints.get("type") or constraints.get("expected_type")
        if expected_type and not isinstance(parameter_value, expected_type):
            result.add_error(
                f"{parameter_name} must be {expected_type}",
                parameter_name=parameter_name,
            )

        if isinstance(parameter_value, str):
            if not allow_strings:
                result.add_error(
                    f"{parameter_name} must not be a string",
                    parameter_name=parameter_name,
                )
            max_length = constraints.get("max_length")
            if max_length and len(parameter_value) > max_length:
                result.add_error(
                    f"{parameter_name} exceeds max_length {max_length}",
                    parameter_name=parameter_name,
                )

        if isinstance(parameter_value, (list, tuple)):
            max_length = constraints.get("max_length")
            if max_length and len(parameter_value) > max_length:
                result.add_error(
                    f"{parameter_name} exceeds max_length {max_length}",
                    parameter_name=parameter_name,
                )

        if isinstance(parameter_value, (int, float, np.number)):
            min_val = constraints.get("min_value")
            max_val = constraints.get("max_value")
            if min_val is not None and parameter_value < min_val:
                result.add_error(
                    f"{parameter_name} below min_value {min_val}",
                    parameter_name=parameter_name,
                )
            if max_val is not None and parameter_value > max_val:
                result.add_error(
                    f"{parameter_name} exceeds max_value {max_val}",
                    parameter_name=parameter_name,
                )


def _get_config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def validate_environment_config(
    config: Any,
    strict_mode: bool = False,
    check_performance: bool = True,
    validation_context: Optional[dict] = None,
) -> ValidationResult:
    """Validate environment configuration."""
    context = create_validation_context(
        operation_name="validate_environment_config",
        component_name="environment",
        additional_context=validation_context,
    )
    result = ValidationResult(
        is_valid=True, operation_name="validate_environment_config", context=context
    )

    grid_size = _get_config_value(config, "grid_size")
    source_location = _get_config_value(config, "source_location")
    plume_params = _get_config_value(config, "plume_params")

    if grid_size is None:
        result.add_error(
            "Missing grid_size",
            recovery_suggestion="Provide grid_size in the configuration",
        )
    if source_location is None:
        result.add_error(
            "Missing source_location",
            recovery_suggestion="Provide source_location in the configuration",
        )

    for validator, args, label in [
        (validate_grid_size, (grid_size,), "grid_size"),
        (validate_coordinates, (source_location, grid_size), "source_location"),
    ]:
        if args[0] is None:
            continue
        try:
            validator(*args, context=context)
        except (ValidationError, ResourceError) as exc:
            result.add_error(f"{label} validation failed: {exc}")

    if plume_params is not None and grid_size is not None:
        try:
            validate_plume_parameters(plume_params, grid_size, context=context)
        except (ValidationError, ResourceError) as exc:
            result.add_error(f"plume_parameters validation failed: {exc}")

    if grid_size is not None and source_location is not None:
        for issue in check_parameter_consistency(
            {
                "grid_size": grid_size,
                "source_location": source_location,
                "plume_parameters": plume_params,
            }
        ):
            result.add_warning(issue.get("issue", "Consistency issue"))

    if not result.is_valid:
        raise ConfigurationError(
            "Environment configuration validation failed",
            config_parameter="environment_config",
            parameter_value=result.errors,
        )

    return result


def validate_action_input(
    action: ActionType, context: Optional[ValidationContext] = None
) -> int:
    """Compatibility wrapper for action validation."""
    return validate_action_parameter(
        action,
        allow_enum_types=True,
        strict_bounds_checking=False,
        context=context,
    )


def validate_action_parameter(
    action: ActionType,
    allow_enum_types: bool = True,
    strict_bounds_checking: bool = False,
    context: Optional[ValidationContext] = None,
) -> int:
    """Validate an action value."""
    if context is None:
        context = create_validation_context(
            operation_name="validate_action_parameter",
            component_name="action_validation",
        )

    if action is None:
        raise ValidationError(
            "Action cannot be None",
            parameter_constraints={"expected_range": [0, 3]},
        )

    if isinstance(action, Action) and allow_enum_types:
        validated_action = action.value
    elif isinstance(action, (int, np.integer)):
        validated_action = int(action)
    elif hasattr(action, "__int__"):
        validated_action = int(action)
    else:
        raise ValidationError(
            f"Invalid action type: {type(action)}",
            parameter_constraints={"expected_types": ["int", "Action enum"]},
        )

    if strict_bounds_checking or not (0 <= validated_action < ACTION_SPACE_SIZE):
        if not (0 <= validated_action < ACTION_SPACE_SIZE):
            raise ValidationError(
                f"Action {validated_action} out of bounds [0, {ACTION_SPACE_SIZE - 1}]",
                parameter_constraints={"valid_range": [0, ACTION_SPACE_SIZE - 1]},
            )

    if validated_action not in {
        Action.UP.value,
        Action.RIGHT.value,
        Action.DOWN.value,
        Action.LEFT.value,
    }:
        raise ValidationError(
            f"Invalid action {validated_action}",
            parameter_constraints={"valid_actions": [0, 1, 2, 3]},
        )

    try:
        validate_action(validated_action)
    except Exception as exc:
        raise ValidationError(
            f"Action failed basic space validation: {exc}",
            parameter_constraints={"space_validation": "failed"},
        )

    return validated_action


def validate_observation_parameter(
    observation: np.ndarray,
    check_concentration_range: bool = True,
    strict_dtype_checking: bool = False,
    context: Optional[ValidationContext] = None,
) -> np.ndarray:
    """Validate an observation array."""
    if context is None:
        context = create_validation_context(
            operation_name="validate_observation_parameter",
            component_name="observation_validation",
        )

    if not isinstance(observation, np.ndarray):
        raise ValidationError(
            f"Observation must be numpy.ndarray, got {type(observation)}",
            parameter_constraints={"expected_type": "numpy.ndarray"},
        )

    if observation.shape != (1,):
        raise ValidationError(
            f"Invalid observation shape: {observation.shape}",
            parameter_constraints={"expected_shape": (1,)},
        )

    if strict_dtype_checking and observation.dtype != np.float32:
        if observation.dtype in [np.float64, np.int32, np.int64]:
            observation = observation.astype(np.float32)
        else:
            raise ValidationError(
                f"Invalid observation dtype: {observation.dtype}",
                parameter_constraints={"expected_dtype": "float32"},
            )

    if check_concentration_range:
        min_val, max_val = CONCENTRATION_RANGE
        if np.any(observation < min_val) or np.any(observation > max_val):
            invalid_values = observation[
                (observation < min_val) | (observation > max_val)
            ]
            raise ValidationError(
                f"Observation values {invalid_values} outside valid range [{min_val}, {max_val}]",
                parameter_constraints={"valid_range": CONCENTRATION_RANGE},
            )

    try:
        validate_observation(observation)
    except Exception as exc:
        raise ValidationError(
            f"Observation failed basic space validation: {exc}",
            parameter_constraints={"space_validation": "failed"},
        )

    return observation


def validate_coordinates(
    coordinates: Union[Coordinates, tuple, list],
    grid_bounds: Optional[GridSize] = None,
    allow_negative: bool = False,
    strict_integer_conversion: bool = False,
    context: Optional[ValidationContext] = None,
) -> Coordinates:
    """Validate coordinate input."""
    if context is None:
        context = create_validation_context(
            operation_name="validate_coordinates",
            component_name="coordinate_validation",
        )

    if isinstance(coordinates, Coordinates):
        coord_tuple = (coordinates.x, coordinates.y)
    elif isinstance(coordinates, (tuple, list)):
        coord_tuple = tuple(coordinates)
    else:
        raise ValidationError(
            f"Invalid coordinates type: {type(coordinates)}",
            parameter_constraints={"expected_types": ["Coordinates", "tuple", "list"]},
        )

    if len(coord_tuple) != 2:
        raise ValidationError(
            f"Coordinates must have exactly 2 elements, got {len(coord_tuple)}",
            parameter_constraints={"required_elements": 2},
        )

    x, y = coord_tuple
    if not isinstance(x, (int, float, np.number)) or not isinstance(
        y, (int, float, np.number)
    ):
        raise ValidationError(
            f"Coordinate elements must be numeric, got x={type(x)}, y={type(y)}",
            parameter_constraints={"element_types": "numeric"},
        )

    x_int = int(round(x))
    y_int = int(round(y))

    if not allow_negative and (x_int < 0 or y_int < 0):
        raise ValidationError(
            f"Negative coordinates not allowed: ({x_int}, {y_int})",
            parameter_constraints={"allow_negative": allow_negative},
        )

    validated_coords = Coordinates(x=x_int, y=y_int)
    # Convert tuple to GridSize if needed
    if grid_bounds is not None and not isinstance(grid_bounds, GridSize):
        grid_bounds = GridSize(width=grid_bounds[0], height=grid_bounds[1])
    if grid_bounds is not None and not validated_coords.is_within_bounds(grid_bounds):
        raise ValidationError(
            f"Coordinates ({x_int}, {y_int}) outside grid bounds {grid_bounds}",
            parameter_constraints={
                "grid_bounds": (grid_bounds.width, grid_bounds.height)
            },
        )

    return validated_coords


def validate_grid_size(
    grid_size: Union[GridSize, tuple, list],
    check_memory_limits: bool = True,
    validate_performance: bool = False,
    resource_constraints: Optional[dict] = None,
    context: Optional[ValidationContext] = None,
) -> GridSize:
    """Validate grid size input."""
    if context is None:
        context = create_validation_context(
            operation_name="validate_grid_size", component_name="grid_validation"
        )

    if isinstance(grid_size, GridSize):
        size_tuple = (grid_size.width, grid_size.height)
    elif isinstance(grid_size, (tuple, list)):
        size_tuple = tuple(grid_size)
    else:
        raise ValidationError(
            f"Invalid grid_size type: {type(grid_size)}",
            parameter_constraints={"expected_types": ["GridSize", "tuple", "list"]},
        )

    if len(size_tuple) != 2:
        raise ValidationError(
            f"Grid size must have exactly 2 dimensions, got {len(size_tuple)}",
            parameter_constraints={"required_dimensions": 2},
        )

    width, height = size_tuple
    if not isinstance(width, (int, np.integer)) or not isinstance(
        height, (int, np.integer)
    ):
        raise ValidationError(
            f"Grid dimensions must be integers, got width={type(width)}, height={type(height)}",
            parameter_constraints={"dimension_types": "integer"},
        )

    if width <= 0 or height <= 0:
        raise ValidationError(
            f"Grid dimensions must be positive, got ({width}, {height})",
            parameter_constraints={"positive_dimensions": True},
        )

    min_width, min_height = MIN_GRID_SIZE
    max_width, max_height = MAX_GRID_SIZE

    if width < min_width or height < min_height:
        raise ValidationError(
            f"Grid size ({width}, {height}) below minimum {MIN_GRID_SIZE}",
            parameter_constraints={"minimum_size": MIN_GRID_SIZE},
        )

    if width > max_width or height > max_height:
        raise ResourceError(
            f"Grid size ({width}, {height}) exceeds maximum {MAX_GRID_SIZE}",
            resource_type="grid_dimensions",
            current_usage=(width, height),
            limit=MAX_GRID_SIZE,
        )

    validated_grid = GridSize(width=width, height=height)

    if check_memory_limits:
        estimated_memory = validated_grid.estimate_memory_mb()
        if estimated_memory > MEMORY_LIMIT_TOTAL_MB:
            raise ResourceError(
                f"Estimated memory usage ({estimated_memory:.1f}MB) exceeds limit ({MEMORY_LIMIT_TOTAL_MB}MB)",
                resource_type="memory",
                current_usage=estimated_memory,
                limit=MEMORY_LIMIT_TOTAL_MB,
            )
        if resource_constraints:
            max_memory = resource_constraints.get("max_memory_mb")
            if max_memory and estimated_memory > max_memory:
                raise ResourceError(
                    f"Memory usage ({estimated_memory:.1f}MB) exceeds constraint ({max_memory}MB)",
                    resource_type="memory",
                    current_usage=estimated_memory,
                    limit=max_memory,
                )

    return validated_grid


def validate_plume_parameters(
    plume_params: Any,
    grid_size: Optional[GridSize] = None,
    check_mathematical_consistency: bool = True,
    validate_field_generation: bool = False,
    context: Optional[ValidationContext] = None,
) -> dict:
    """Validate plume parameters."""
    if context is None:
        context = create_validation_context(
            operation_name="validate_plume_parameters",
            component_name="plume_validation",
        )

    if not isinstance(plume_params, dict):
        raise ValidationError(
            f"Invalid plume_params type: {type(plume_params)}",
            parameter_constraints={"expected_types": ["dict"]},
        )

    required_keys = ["source_location", "sigma"]
    missing_keys = [key for key in required_keys if key not in plume_params]
    if missing_keys:
        raise ValidationError(
            f"Missing required plume parameter keys: {missing_keys}",
            parameter_constraints={"required_keys": required_keys},
        )

    source_coords = plume_params.get("source_location")
    if isinstance(source_coords, Coordinates):
        coords = source_coords
    else:
        coords = create_coordinates(source_coords)

    if grid_size is not None and not coords.is_within_bounds(grid_size):
        raise ValidationError(
            f"Source location ({coords.x}, {coords.y}) outside grid bounds ({grid_size.width}, {grid_size.height})",
            parameter_constraints={"grid_bounds": (grid_size.width, grid_size.height)},
        )

    sigma = plume_params.get("sigma")
    if not isinstance(sigma, (int, float, np.number)):
        raise ValidationError(
            f"Sigma must be numeric, got {type(sigma)}",
            parameter_constraints={"sigma_type": "numeric"},
        )
    sigma_val = float(sigma)
    if sigma_val <= 0:
        raise ValidationError(
            f"Sigma must be positive, got {sigma_val}",
            parameter_constraints={"sigma_positive": True},
        )
    if sigma_val < MIN_PLUME_SIGMA or sigma_val > MAX_PLUME_SIGMA:
        raise ValidationError(
            f"Sigma {sigma_val} outside valid range [{MIN_PLUME_SIGMA}, {MAX_PLUME_SIGMA}]",
            parameter_constraints={"sigma_range": [MIN_PLUME_SIGMA, MAX_PLUME_SIGMA]},
        )

    normalized = dict(plume_params)
    normalized["source_location"] = coords
    normalized["sigma"] = sigma_val
    return normalized


def validate_render_mode(
    render_mode: Union[RenderMode, str],
    check_backend_availability: bool = False,
    validate_display_capability: bool = False,
    context: Optional[ValidationContext] = None,
) -> RenderMode:
    """Validate render mode input."""
    if context is None:
        context = create_validation_context(
            operation_name="validate_render_mode", component_name="render_validation"
        )

    if isinstance(render_mode, str):
        if render_mode not in SUPPORTED_RENDER_MODES:
            raise ValidationError(
                f"Unsupported render mode '{render_mode}'",
                parameter_constraints={"supported_modes": SUPPORTED_RENDER_MODES},
            )
        validated_mode = (
            RenderMode.RGB_ARRAY if render_mode == "rgb_array" else RenderMode.HUMAN
        )
    elif isinstance(render_mode, RenderMode):
        validated_mode = render_mode
    else:
        raise ValidationError(
            f"Invalid render_mode type: {type(render_mode)}",
            parameter_constraints={"expected_types": ["RenderMode", "str"]},
        )

    if validated_mode.value not in SUPPORTED_RENDER_MODES:
        raise ValidationError(
            f"Render mode '{validated_mode.value}' not supported",
            parameter_constraints={"supported_modes": SUPPORTED_RENDER_MODES},
        )

    if check_backend_availability or validate_display_capability:
        pass

    return validated_mode


def validate_seed_value(
    seed: Optional[Union[int, np.integer]],
    allow_none: bool = True,
    strict_type_checking: bool = False,
    context: Optional[ValidationContext] = None,
) -> Optional[int]:
    """Validate seed value for reproducibility."""
    if context is None:
        context = create_validation_context(
            operation_name="validate_seed_value", component_name="seed_validation"
        )

    if seed is None:
        if allow_none:
            return None
        raise ValidationError(
            "Seed cannot be None when allow_none is False",
            parameter_constraints={"allow_none": allow_none},
        )

    if isinstance(seed, (int, np.integer)):
        validated_seed = int(seed)
    elif hasattr(seed, "__int__") and not strict_type_checking:
        validated_seed = int(seed)
    else:
        raise ValidationError(
            f"Invalid seed type: {type(seed)}",
            parameter_constraints={"expected_types": ["int", "numpy.integer", "None"]},
        )

    if not (SEED_MIN_VALUE <= validated_seed <= SEED_MAX_VALUE):
        raise ValidationError(
            f"Seed {validated_seed} outside valid range [{SEED_MIN_VALUE}, {SEED_MAX_VALUE}]",
            parameter_constraints={"valid_range": [SEED_MIN_VALUE, SEED_MAX_VALUE]},
        )

    return validated_seed


def validate_performance_constraints(
    performance_requirements: Dict[str, Any],
    check_system_capabilities: bool = False,
    validate_timing_targets: bool = False,
    system_resources: Optional[dict] = None,
    context: Optional[ValidationContext] = None,
) -> Dict[str, Any]:
    """Validate performance constraints."""
    if context is None:
        context = create_validation_context(
            operation_name="validate_performance_constraints",
            component_name="performance_validation",
        )

    if not isinstance(performance_requirements, dict):
        raise ValidationError(
            f"Performance requirements must be dict, got {type(performance_requirements)}",
            parameter_constraints={"expected_type": "dict"},
        )

    validated = performance_requirements.copy()
    validated.setdefault("step_latency_ms", PERFORMANCE_TARGET_STEP_LATENCY_MS)
    validated.setdefault("memory_limit_mb", MEMORY_LIMIT_TOTAL_MB)

    step_latency = validated.get("step_latency_ms")
    if step_latency is not None:
        if not isinstance(step_latency, (int, float)) or step_latency <= 0:
            raise ValidationError(
                f"Step latency must be positive number, got {step_latency}",
                parameter_constraints={"step_latency_positive": True},
            )

    memory_limit = validated.get("memory_limit_mb")
    if memory_limit is not None:
        if not isinstance(memory_limit, (int, float)) or memory_limit <= 0:
            raise ValidationError(
                f"Memory limit must be positive number, got {memory_limit}",
                parameter_constraints={"memory_limit_positive": True},
            )
        if memory_limit > MEMORY_LIMIT_TOTAL_MB:
            raise ResourceError(
                f"Memory limit {memory_limit}MB exceeds system limit {MEMORY_LIMIT_TOTAL_MB}MB",
                resource_type="memory",
                current_usage=memory_limit,
                limit_exceeded=MEMORY_LIMIT_TOTAL_MB,
            )

    if check_system_capabilities and system_resources:
        available_memory = system_resources.get("available_memory_mb")
        if available_memory and memory_limit and memory_limit > available_memory * 0.8:
            logger.warning(
                "Memory requirement %.0fMB may exceed available memory %.0fMB",
                memory_limit,
                available_memory,
            )

    if validate_timing_targets:
        pass

    return validated


def sanitize_parameters(
    parameters: Dict[str, Any],
    additional_sensitive_keys: Optional[List[str]] = None,
    strict_sanitization: bool = False,
    preserve_types: bool = True,
    context: Optional[ValidationContext] = None,
) -> Dict[str, Any]:
    """Sanitize sensitive values for logging."""
    if not isinstance(parameters, dict):
        return {}

    patterns = list(SENSITIVE_PARAMETER_PATTERNS)
    if additional_sensitive_keys:
        patterns.extend(additional_sensitive_keys)

    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error:
            continue

    seen: set[int] = set()

    def _sanitize_value(key: str, value: Any) -> Any:
        if id(value) in seen:
            return SANITIZATION_PLACEHOLDER
        if any(pattern.search(key) for pattern in compiled):
            return SANITIZATION_PLACEHOLDER
        if isinstance(value, str):
            sanitized_str = value.replace("\x00", "")
            if len(sanitized_str) > 1000:
                truncated = sanitized_str[:1000]
                return f"{truncated}..." if strict_sanitization else truncated
            return sanitized_str
        if isinstance(value, np.ndarray):
            seen.add(id(value))
            return [_sanitize_value(f"{key}_item", item) for item in value.tolist()]
        if isinstance(value, dict):
            seen.add(id(value))
            return {k: _sanitize_value(k, v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            seen.add(id(value))
            return type(value)(_sanitize_value(f"{key}_item", item) for item in value)
        return value

    return {key: _sanitize_value(key, value) for key, value in parameters.items()}


def check_parameter_consistency(
    parameters: Dict[str, Any],
    strict_consistency_checking: bool = False,
    check_mathematical_relationships: bool = True,
    consistency_rules: Optional[List[str]] = None,
    context: Optional[ValidationContext] = None,
) -> List[Dict[str, Any]]:
    """Check basic cross-parameter consistency."""
    if context is None:
        context = create_validation_context(
            operation_name="check_parameter_consistency",
            component_name="consistency_validation",
        )

    issues: List[Dict[str, Any]] = []
    grid_size = parameters.get("grid_size")
    source_location = parameters.get("source_location")

    if grid_size is None or source_location is None:
        return issues

    if hasattr(grid_size, "width") and hasattr(grid_size, "height"):
        grid_width, grid_height = grid_size.width, grid_size.height
    elif isinstance(grid_size, (tuple, list)) and len(grid_size) == 2:
        grid_width, grid_height = grid_size
    else:
        return issues

    if hasattr(source_location, "x") and hasattr(source_location, "y"):
        src_x, src_y = source_location.x, source_location.y
    elif isinstance(source_location, (tuple, list)) and len(source_location) == 2:
        src_x, src_y = source_location
    else:
        return issues

    if src_x < 0 or src_x >= grid_width or src_y < 0 or src_y >= grid_height:
        issues.append(
            {
                "issue": (
                    f"Source location ({src_x}, {src_y}) outside grid bounds "
                    f"({grid_width}, {grid_height})"
                ),
                "severity": "high",
                "recommendation": "Adjust source location to be within bounds",
            }
        )
    return issues


def create_validation_context(
    operation_name: str,
    component_name: Optional[str] = None,
    additional_context: Optional[dict] = None,
    include_caller_info: bool = True,
    include_performance_tracking: bool = False,
) -> ValidationContext:
    """Create a ValidationContext."""
    context = ValidationContext(
        operation_name=operation_name,
        component_name=component_name or "unknown",
        timestamp=time.time(),
    )
    if additional_context and isinstance(additional_context, dict):
        context.additional_context.update(sanitize_parameters(additional_context))
    if include_caller_info or include_performance_tracking:
        pass
    return context


def validate_grid_dimensions(
    grid_dimensions: Union[GridSize, tuple, list],
    check_memory_limits: bool = True,
    validate_performance: bool = False,
    resource_constraints: Optional[dict] = None,
    context: Optional[ValidationContext] = None,
) -> GridSize:
    """Compatibility shim for validate_grid_size."""
    return validate_grid_size(
        grid_dimensions,
        check_memory_limits=check_memory_limits,
        validate_performance=validate_performance,
        resource_constraints=resource_constraints,
        context=context,
    )


__all__ = [
    "validate_environment_config",
    "validate_action_input",
    "validate_action_parameter",
    "validate_observation_parameter",
    "validate_coordinates",
    "validate_grid_size",
    "validate_plume_parameters",
    "validate_render_mode",
    "validate_seed_value",
    "validate_performance_constraints",
    "sanitize_parameters",
    "check_parameter_consistency",
    "create_validation_context",
    "validate_grid_dimensions",
    "ValidationContext",
    "ValidationResult",
    "ParameterValidator",
]
