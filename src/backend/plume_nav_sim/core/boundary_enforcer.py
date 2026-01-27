import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np

from ..utils.exceptions import ValidationError
from ..utils.logging import ComponentType, get_component_logger, monitor_performance

from ..utils.validation import validate_action_parameter, validate_coordinates

from .constants import (
    BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS,
    BOUNDARY_VALIDATION_CACHE_SIZE,
    MOVEMENT_VECTORS,
)

from .enums import Action
from .geometry import Coordinates, GridSize
from .types import ActionType, CoordinateType, create_coordinates, create_grid_size

DEFAULT_CLAMPING_ENABLED = True  # Default clamping behavior for position enforcement
POSITION_BOUNDS_TOLERANCE = (
    0  # Tolerance for position bounds checking (strict by default)
)

__all__ = [
    "BoundaryEnforcer",
    "BoundaryEnforcementResult",
    "MovementConstraint",
    "validate_movement_bounds",
    "enforce_position_bounds",
    "is_position_within_bounds",
    "clamp_coordinates_to_bounds",
    "calculate_bounded_movement",
]


@dataclass(frozen=True)
class BoundaryEnforcementResult:
    # Core boundary enforcement result fields
    original_position: Coordinates  # Original position before boundary enforcement
    final_position: Coordinates  # Final position after boundary constraint application
    position_modified: bool  # Flag indicating if position was modified by enforcement
    boundary_hit: bool  # Flag indicating if boundary constraint was activated

    # Additional analysis and performance metrics fields
    constraint_direction: Tuple[int, int] = field(
        init=False
    )  # Direction of constraint application
    enforcement_time: float = field(
        default=0.0, init=False
    )  # Time taken for enforcement operation
    constraint_reason: str = field(
        default="", init=False
    )  # Detailed reason for constraint application

    def __post_init__(self):
        """Post-initialization processing to calculate derived constraint analysis fields."""
        # Calculate constraint_direction from position difference vector
        dx = self.final_position.x - self.original_position.x
        dy = self.final_position.y - self.original_position.y

        # Use object.__setattr__ for frozen dataclass field modification
        object.__setattr__(self, "constraint_direction", (dx, dy))

        # Set default constraint_reason based on boundary hit and position modification
        if self.boundary_hit and self.position_modified:
            if dx != 0 and dy != 0:
                object.__setattr__(
                    self, "constraint_reason", "Corner boundary constraint applied"
                )
            elif dx != 0:
                object.__setattr__(
                    self, "constraint_reason", "X-axis boundary constraint applied"
                )
            elif dy != 0:
                object.__setattr__(
                    self, "constraint_reason", "Y-axis boundary constraint applied"
                )
            else:
                object.__setattr__(
                    self, "constraint_reason", "Position maintained within bounds"
                )
        else:
            object.__setattr__(
                self, "constraint_reason", "No boundary constraint required"
            )

    def get_constraint_analysis(self) -> Dict[str, Any]:
        # Calculate position difference vector between original and final positions
        position_delta = (
            self.final_position.x - self.original_position.x,
            self.final_position.y - self.original_position.y,
        )

        # Determine constraint direction from position modification pattern
        constraint_type = "none"
        if self.position_modified:
            if position_delta[0] != 0 and position_delta[1] != 0:
                constraint_type = "corner"
            elif position_delta[0] != 0:
                constraint_type = "x_boundary"
            elif position_delta[1] != 0:
                constraint_type = "y_boundary"
            else:
                constraint_type = "maintained"

        # Analyze boundary hit type and constraint characteristics
        constraint_magnitude = max(abs(position_delta[0]), abs(position_delta[1]))

        # Calculate constraint strength as measure of enforcement intensity
        constraint_strength = "none"
        if self.boundary_hit:
            if constraint_magnitude == 0:
                constraint_strength = "preventive"  # Movement prevented
            else:
                constraint_strength = "corrective"  # Position corrected

        return {
            "position_delta": position_delta,
            "constraint_direction": self.constraint_direction,
            "constraint_type": constraint_type,
            "constraint_magnitude": constraint_magnitude,
            "constraint_strength": constraint_strength,
            "boundary_hit": self.boundary_hit,
            "position_modified": self.position_modified,
            "constraint_reason": self.constraint_reason,
            "enforcement_time_ms": self.enforcement_time,
            "original_coordinates": (
                self.original_position.x,
                self.original_position.y,
            ),
            "final_coordinates": (self.final_position.x, self.final_position.y),
        }

    def was_movement_constrained(self) -> bool:
        # Return position_modified flag indicating constraint application
        return self.position_modified

    def get_final_position(self) -> Coordinates:
        # Return final_position coordinates for external use
        return self.final_position

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_position": (
                self.original_position.x,
                self.original_position.y,
            ),
            "final_position": (self.final_position.x, self.final_position.y),
            "position_modified": self.position_modified,
            "boundary_hit": self.boundary_hit,
            "constraint_direction": self.constraint_direction,
            "enforcement_time_ms": self.enforcement_time,
            "constraint_reason": self.constraint_reason,
            "constraint_analysis": self.get_constraint_analysis(),
        }


@dataclass
class MovementConstraint:
    # Core constraint configuration fields
    enable_clamping: bool = (
        DEFAULT_CLAMPING_ENABLED  # Enable position clamping to bounds
    )
    strict_validation: bool = True  # Enable strict boundary validation
    log_boundary_violations: bool = False  # Enable boundary violation logging

    # Advanced constraint configuration fields with defaults
    tolerance: float = field(
        default=POSITION_BOUNDS_TOLERANCE
    )  # Bounds checking tolerance
    custom_boundaries: Dict[str, Any] = field(
        default_factory=dict
    )  # Custom boundary rules
    performance_monitoring: bool = field(default=True)  # Enable performance monitoring

    def validate_configuration(self) -> bool:
        # Validate tolerance is non-negative for bounds checking
        if self.tolerance < 0:
            raise ValidationError(
                "Movement constraint tolerance must be non-negative",
                parameter_name="tolerance",
                parameter_value=self.tolerance,
                expected_format="float >= 0",
            )

        # Check custom_boundaries dictionary structure if not empty
        if self.custom_boundaries:
            if not isinstance(self.custom_boundaries, dict):
                raise ValidationError(
                    "Custom boundaries must be a dictionary",
                    parameter_name="custom_boundaries",
                    parameter_value=type(self.custom_boundaries).__name__,
                    expected_format="dict",
                )

            # Validate custom boundary entries have required structure
            for key, _value in self.custom_boundaries.items():
                if not isinstance(key, str):
                    raise ValidationError(
                        f"Custom boundary key must be string, got {type(key).__name__}",
                        parameter_name=f"custom_boundaries[{key}]",
                    )

        # Return True if all constraint configuration validations pass
        return True

    def clone(self, overrides: Optional[Dict[str, Any]] = None) -> "MovementConstraint":
        # Create deep copy of current constraint configuration
        new_config = MovementConstraint(
            enable_clamping=self.enable_clamping,
            strict_validation=self.strict_validation,
            log_boundary_violations=self.log_boundary_violations,
            tolerance=self.tolerance,
            custom_boundaries=self.custom_boundaries.copy(),
            performance_monitoring=self.performance_monitoring,
        )

        # Apply parameter overrides if provided with validation
        if overrides:
            for key, value in overrides.items():
                if hasattr(new_config, key):
                    setattr(new_config, key, value)
                else:
                    raise ValidationError(
                        f"Invalid override parameter: {key}",
                        parameter_name="overrides",
                        parameter_value=key,
                    )

        # Validate new configuration if overrides were applied
        if overrides:
            new_config.validate_configuration()

        # Return new MovementConstraint instance
        return new_config


class BoundaryEnforcer:
    def __init__(
        self,
        grid_size: GridSize,
        constraint_config: Optional[MovementConstraint] = None,
        enable_caching: bool = True,
    ) -> None:
        try:
            self.grid_size = create_grid_size(grid_size)
            validate_coordinates((0, 0), self.grid_size)  # Validate grid configuration
        except ValidationError as exc:
            raise ValidationError(
                f"Invalid grid_size for boundary enforcer: {exc}",
                parameter_name="grid_size",
                parameter_value=str(grid_size),
            ) from exc

        # Set up constraint_config or create default MovementConstraint if None provided
        if constraint_config is None:
            self.constraint_config = MovementConstraint()
        else:
            # Validate constraint configuration
            constraint_config.validate_configuration()
            self.constraint_config = constraint_config

        # Store enable_caching flag for performance optimization of repeated validations
        self.enable_caching = enable_caching

        # Initialize component logger using get_component_logger for boundary enforcement operations
        self.logger = get_component_logger(
            component_name="boundary_enforcer",
            component_type=ComponentType.BOUNDARY_ENFORCER,
            enable_performance_tracking=True,
        )

        # Initialize validation_cache dictionary for position validation results
        self.validation_cache: Dict[str, Any] = {}

        # Initialize performance metrics tracking dictionary for boundary operations
        self.performance_metrics: Dict[str, List[float]] = {
            "validation_times": [],
            "enforcement_times": [],
            "movement_validations": [],
        }
        self.cache_metrics: Dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }

        # Reset enforcement_count and boundary_hits for enforcement statistics tracking
        self.enforcement_count = 0
        self.boundary_hits = 0

    @monitor_performance("position_validation", 0.05, False)
    def validate_position(  # noqa: C901
        self,
        position: CoordinateType,
        *,
        raise_on_invalid: bool = False,
        context_info: Optional[str] = None,
    ) -> bool:
        """Validate that `position` lies within the configured grid bounds."""
        start_time = time.perf_counter()

        try:
            coords = create_coordinates(position)

            cache_key, cached = self._get_cached_position_validation(coords)
            if cached is not None:
                return cached

            is_valid = coords.is_within_bounds(self.grid_size)
            is_valid = self._apply_strict_position_validation(coords, is_valid)

            self._update_position_validation_cache(cache_key, is_valid)
            self._record_position_validation_time(start_time)

            if not is_valid and raise_on_invalid:
                self._raise_position_out_of_bounds(coords, context_info)

            if context_info and not is_valid:
                self.logger.debug(
                    f"Position validation failed: {context_info}, position={coords}"
                )

            return is_valid

        except ValidationError as exc:
            if raise_on_invalid:
                raise
            self.logger.debug(f"Validation error for position {position}: {exc}")
            return False
        except Exception as exc:
            self.logger.error(f"Unexpected error during position validation: {exc}")
            if raise_on_invalid:
                raise ValidationError(
                    f"Position validation failed due to internal error: {exc}",
                    parameter_name="position",
                    parameter_value=str(position),
                ) from exc
            return False

    def _get_cached_position_validation(
        self, coords: Coordinates
    ) -> Tuple[Optional[str], Optional[bool]]:
        """Return cache key and cached result (if any) for a position validation."""
        if not self.enable_caching:
            return None, None

        cache_key = (
            f"validate_{coords.x}_{coords.y}_"
            f"{self.grid_size.width}_{self.grid_size.height}"
        )
        cached = self.validation_cache.get(cache_key)
        if cached is not None:
            self.cache_metrics["hits"] += 1
            return cache_key, cached

        self.cache_metrics["misses"] += 1
        return cache_key, None

    def _apply_strict_position_validation(
        self, coords: Coordinates, is_valid: bool
    ) -> bool:
        """Apply strict edge tolerance rules when strict_validation is enabled."""
        if not (self.constraint_config.strict_validation and is_valid):
            return is_valid

        tol = max(self.constraint_config.tolerance, 0.0)
        near_edge = (
            coords.x < tol
            or coords.x >= self.grid_size.width - tol
            or coords.y < tol
            or coords.y >= self.grid_size.height - tol
        )
        return not near_edge

    def _update_position_validation_cache(
        self, cache_key: Optional[str], is_valid: bool
    ) -> None:
        """Update the position validation cache and eviction metrics."""
        if not (self.enable_caching and cache_key is not None):
            return

        self.validation_cache[cache_key] = is_valid
        if len(self.validation_cache) > BOUNDARY_VALIDATION_CACHE_SIZE:
            first_key = next(iter(self.validation_cache))
            self.validation_cache.pop(first_key, None)
            self.cache_metrics["evictions"] += 1

    def _record_position_validation_time(self, start_time: float) -> None:
        """Record the elapsed validation time in milliseconds."""
        self.performance_metrics["validation_times"].append(
            (time.perf_counter() - start_time) * 1000
        )

    def _raise_position_out_of_bounds(
        self, coords: Coordinates, context_info: Optional[str]
    ) -> None:
        """Raise a ValidationError for an out-of-bounds position."""
        message = f"Position {coords} outside grid bounds {self.grid_size}"
        if context_info:
            message += f" (Context: {context_info})"
        raise ValidationError(
            message,
            parameter_name="position",
            parameter_value=(coords.x, coords.y),
            expected_format=(
                f"coordinates within bounds (0, 0) to "
                f"({self.grid_size.width - 1}, {self.grid_size.height - 1})"
            ),
        )

    @monitor_performance("movement_validation", 0.05, False)
    def is_movement_valid(
        self, current_position: CoordinateType, action: ActionType
    ) -> bool:
        """Fast boolean validation for proposed movement."""
        if not isinstance(current_position, Coordinates):
            current_position = create_coordinates(current_position)

        try:
            validated_action = self._normalize_action(action)
        except ValidationError:
            return False

        vector = MOVEMENT_VECTORS.get(validated_action.value)
        if vector is None:
            return False

        new_x = current_position.x + vector[0]
        new_y = current_position.y + vector[1]
        if 0 <= new_x < self.grid_size.width and 0 <= new_y < self.grid_size.height:
            self.performance_metrics["movement_validations"].append(1)
            return True

        try:
            result = validate_movement_bounds(
                current_position=current_position,
                action=validated_action,
                grid_bounds=self.grid_size,
                strict_validation=False,
            )
            if result:
                self.performance_metrics["movement_validations"].append(1)
            return result
        except Exception as exc:
            self.logger.warning(f"Movement validation error: {exc}")
            return False

    @monitor_performance("boundary_enforcement", 0.1, True)
    def enforce_movement_bounds(
        self, current_position: CoordinateType, action: ActionType
    ) -> BoundaryEnforcementResult:
        start_time = time.perf_counter()

        try:
            coords = (
                current_position
                if isinstance(current_position, Coordinates)
                else create_coordinates(current_position)
            )

            normalized_action = self._normalize_action(action)
            if hasattr(normalized_action, "to_vector"):
                movement_vector = normalized_action.to_vector()
            else:
                movement_vector = MOVEMENT_VECTORS.get(normalized_action.value)

            new_x = coords.x + movement_vector[0]
            new_y = coords.y + movement_vector[1]

            initial_in_bounds = coords.is_within_bounds(self.grid_size)

            in_bounds = (
                0 <= new_x < self.grid_size.width and 0 <= new_y < self.grid_size.height
            )

            if in_bounds:
                final_position = (
                    coords if movement_vector == (0, 0) else Coordinates(new_x, new_y)
                )
                boundary_hit = not initial_in_bounds
            else:
                proposed_position = Coordinates(new_x, new_y)
                if self.constraint_config.enable_clamping:
                    final_position = clamp_coordinates_to_bounds(
                        proposed_position, self.grid_size
                    )
                else:
                    final_position = coords
                boundary_hit = True

            result = BoundaryEnforcementResult(
                original_position=coords,
                final_position=final_position,
                position_modified=(boundary_hit and final_position != coords),
                boundary_hit=boundary_hit,
            )

            enforcement_time = (time.perf_counter() - start_time) * 1000
            self.performance_metrics["enforcement_times"].append(enforcement_time)
            object.__setattr__(result, "enforcement_time", enforcement_time)

            self.enforcement_count += 1
            if boundary_hit:
                self.boundary_hits += 1

            return result

        except ValidationError as exc:
            self.logger.error(f"Movement enforcement validation failed: {exc}")
            raise
        except Exception as exc:
            self.logger.error(f"Movement enforcement error: {exc}")
            raise

    def get_valid_moves(self, current_position: CoordinateType) -> List[Action]:
        # Convert current_position to Coordinates for consistent processing
        coords = create_coordinates(current_position)

        # Initialize empty list for valid actions collection
        valid_actions = []

        # Iterate through all possible actions from Action enum
        for action in Action:
            try:
                # For each action, use is_movement_valid to check validity
                if self.is_movement_valid(coords, action):
                    # Add action to valid actions list if movement would be legal
                    valid_actions.append(action)
            except Exception as e:
                # Skip actions that cause validation errors
                self.logger.debug(f"Action {action} validation failed: {e}")
                continue

        # Return complete list of valid actions from current position for action space analysis
        return valid_actions

    def update_grid_size(self, new_grid_size: GridSize) -> None:
        # Validate new_grid_size has positive dimensions using validate_coordinates
        try:
            validate_coordinates((0, 0), new_grid_size)
        except Exception as e:
            raise ValidationError(
                f"Invalid new grid_size: {e}",
                parameter_name="new_grid_size",
                parameter_value=str(new_grid_size),
            ) from e

        # Log grid size update with old and new dimensions for debugging
        old_grid = self.grid_size
        self.logger.info(f"Updating grid size from {old_grid} to {new_grid_size}")

        # Clear validation_cache to prevent stale boundary results
        if self.enable_caching:
            cache_entries_cleared = len(self.validation_cache)
            self.validation_cache.clear()
            self.logger.debug(f"Cleared {cache_entries_cleared} cache entries")

        # Update grid_size property with new dimensions
        self.grid_size = new_grid_size

        # Reset performance metrics and enforcement statistics for new grid configuration
        self.performance_metrics = {
            "validation_times": [],
            "enforcement_times": [],
            "cache_hits": [],
            "cache_misses": [],
        }
        self.enforcement_count = 0
        self.boundary_hits = 0

    def get_boundary_statistics(self) -> Dict[str, Any]:
        # Compile total enforcement operations performed from enforcement_count
        total_operations = self.enforcement_count

        # Calculate boundary hit rate from boundary_hits and total operations
        hit_rate = (
            (self.boundary_hits / total_operations) if total_operations > 0 else 0.0
        )

        # Base statistics dictionary
        stats = {
            "total_enforcements": total_operations,
            "boundary_hits": self.boundary_hits,
            "hit_rate": hit_rate,
            "grid_size": {
                "width": self.grid_size.width,
                "height": self.grid_size.height,
            },
            "configuration": {
                "clamping_enabled": self.constraint_config.enable_clamping,
                "strict_validation": self.constraint_config.strict_validation,
                "caching_enabled": self.enable_caching,
                "tolerance": self.constraint_config.tolerance,
            },
        }

        # Include cache hit/miss ratios if caching is enabled
        if self.enable_caching:
            cache_hits = self.cache_metrics["hits"]
            cache_misses = self.cache_metrics["misses"]
            total_cache_attempts = cache_hits + cache_misses

            stats["cache_statistics"] = {
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "hit_ratio": (
                    cache_hits / total_cache_attempts
                    if total_cache_attempts > 0
                    else 0.0
                ),
                "cache_entries": len(self.validation_cache),
                "cache_capacity": BOUNDARY_VALIDATION_CACHE_SIZE,
                "evictions": self.cache_metrics["evictions"],
            }

        # Add performance metrics including average enforcement time
        if self.performance_metrics["enforcement_times"]:
            enforcement_times = self.performance_metrics["enforcement_times"]
            stats["performance_metrics"] = {
                "average_enforcement_time_ms": np.mean(enforcement_times),
                "min_enforcement_time_ms": np.min(enforcement_times),
                "max_enforcement_time_ms": np.max(enforcement_times),
                "total_enforcement_time_ms": np.sum(enforcement_times),
            }

            # Include performance target compliance
            target_time = BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS
            compliant_operations = sum(t <= target_time for t in enforcement_times)
            stats["performance_metrics"]["target_compliance_rate"] = (
                compliant_operations / len(enforcement_times)
            )

        # Include validation success rate and error frequency analysis
        if self.performance_metrics["validation_times"]:
            validation_times = self.performance_metrics["validation_times"]
            stats["validation_metrics"] = {
                "total_validations": len(validation_times),
                "average_validation_time_ms": np.mean(validation_times),
                "validation_efficiency": len(validation_times)
                / max(total_operations, 1),
            }

        return stats

    def clear_cache(self, force_cleanup: bool = False) -> int:
        if not self.enable_caching:
            return 0

        entries_cleared = len(self.validation_cache)
        self.validation_cache.clear()

        if force_cleanup:
            self.cache_metrics = {"hits": 0, "misses": 0, "evictions": 0}

        # Log cache clearing operation with entry count
        self.logger.debug(f"Cache cleared: {entries_cleared} entries removed")

        # Return number of entries that were cleared from cache
        return entries_cleared

    def validate_constraint_configuration(self, strict_mode: bool = False) -> bool:
        try:
            self.constraint_config.validate_configuration()

            # Check grid_size compatibility with constraint configuration
            if (
                self.constraint_config.tolerance
                >= min(self.grid_size.width, self.grid_size.height) / 2
            ):
                if strict_mode:
                    raise ValidationError(
                        "Constraint tolerance too large relative to grid size",
                        parameter_name="tolerance",
                        parameter_value=self.constraint_config.tolerance,
                    )
                else:
                    self.logger.warning(
                        f"Large tolerance ({self.constraint_config.tolerance}) relative to grid size {self.grid_size}"
                    )

            # Apply strict validation rules if strict_mode enabled
            if strict_mode and self.constraint_config.custom_boundaries:
                for (
                    boundary_name,
                    boundary_config,
                ) in self.constraint_config.custom_boundaries.items():
                    if not isinstance(boundary_config, (dict, bool, float, int)):
                        raise ValidationError(
                            f"Invalid custom boundary configuration for {boundary_name}",
                            parameter_name="custom_boundaries",
                            parameter_value=type(boundary_config).__name__,
                        )

            # Log successful configuration validation
            self.logger.debug("Constraint configuration validation passed")

            # Return True if all configuration validations pass successfully
            return True

        except ValidationError:
            # Re-raise ValidationError with configuration context
            raise
        except Exception as e:
            # Handle unexpected validation errors
            raise ValidationError(
                f"Constraint configuration validation failed: {e}",
                parameter_name="constraint_config",
            ) from e

    # Standalone utility functions for performance-critical boundary validation operations

    def _normalize_action(self, action: ActionType) -> Action:
        """Normalize incoming action to an `Action` enum, raising `ValidationError` if invalid."""
        if isinstance(action, Action):
            return action

        if not isinstance(action, (int, np.integer)):
            raise ValidationError(
                f"Invalid action type: {type(action)}. Expected int or Action enum"
            )

        validated = validate_action_parameter(int(action), allow_enum_types=False)
        try:
            return Action(validated)
        except ValueError as exc:
            raise ValidationError(f"Invalid action value: {validated}") from exc


def validate_movement_bounds(
    current_position: CoordinateType,
    action: ActionType,
    grid_bounds: GridSize,
    strict_validation: bool = True,
) -> bool:
    try:
        # Convert current_position to Coordinates using create_coordinates factory for type consistency
        coords = create_coordinates(current_position)

        # Validate action parameter using validate_action_parameter with type checking
        if strict_validation:
            validate_action_parameter(action)

        if hasattr(action, "to_vector"):
            movement_vector = action.to_vector()
        else:
            movement_vector = MOVEMENT_VECTORS.get(int(action), (0, 0))

        # Calculate proposed destination position by adding movement delta to current position
        new_x = coords.x + movement_vector[0]
        new_y = coords.y + movement_vector[1]

        # Check if proposed position is within grid bounds using bounds checking
        return (
            new_x >= 0
            and new_x < grid_bounds.width
            and new_y >= 0
            and new_y < grid_bounds.height
        )
    except Exception:
        # Return False for any validation errors to maintain performance
        return False


def enforce_position_bounds(
    position: CoordinateType,
    grid_bounds: GridSize,
    clamp_to_bounds: bool = True,
    log_boundary_hits: bool = False,
) -> Tuple[Coordinates, bool]:
    # Convert position to Coordinates using create_coordinates with validation
    coords = create_coordinates(position)

    # Check if position is already within grid bounds using position.is_within_bounds()
    if coords.is_within_bounds(grid_bounds):
        # Return original position unchanged if already within bounds
        return coords, False

    # Position is outside bounds - determine action based on clamp_to_bounds
    if not clamp_to_bounds:
        # Return original position with modification flag if clamping disabled
        return coords, True  # Position modified flag indicates bounds violation

    # Apply coordinate clamping if clamp_to_bounds enabled and position exceeds bounds
    # Clamp x-coordinate to valid range [0, grid_bounds.width - 1]
    clamped_x = max(0, min(coords.x, grid_bounds.width - 1))

    # Clamp y-coordinate to valid range [0, grid_bounds.height - 1]
    clamped_y = max(0, min(coords.y, grid_bounds.height - 1))

    # Create new Coordinates object with constrained values
    constrained_position = Coordinates(clamped_x, clamped_y)

    # Log boundary hit if log_boundary_hits enabled and position was modified
    if log_boundary_hits:
        logger = logging.getLogger("plume_nav_sim.boundary_enforcer")
        logger.info(
            f"Position clamped: {coords} -> {constrained_position} "
            f"(bounds: {grid_bounds})"
        )

    # Return tuple with final position and modification status for caller analysis
    return constrained_position, True


def is_position_within_bounds(position: CoordinateType, grid_bounds: GridSize) -> bool:
    try:
        # Convert position to Coordinates using fast conversion without full validation
        if isinstance(position, Coordinates):
            coords = position
        else:
            coords = create_coordinates(position)

        # Check x-coordinate is within range [0, grid_bounds.width - 1] using integer comparison
        if coords.x < 0 or coords.x >= grid_bounds.width:
            return False

        # Check y-coordinate is within range [0, grid_bounds.height - 1] using integer comparison
        return coords.y >= 0 and coords.y < grid_bounds.height
    except Exception:
        # Return False for any conversion or validation errors
        return False


def clamp_coordinates_to_bounds(
    coordinates: CoordinateType, grid_bounds: GridSize
) -> Coordinates:
    # Convert coordinates to Coordinates object for consistent processing
    coords = create_coordinates(coordinates)

    # Clamp x-coordinate using max(0, min(coordinates.x, grid_bounds.width - 1))
    clamped_x = max(0, min(coords.x, grid_bounds.width - 1))

    # Clamp y-coordinate using max(0, min(coordinates.y, grid_bounds.height - 1))
    clamped_y = max(0, min(coords.y, grid_bounds.height - 1))

    return Coordinates(clamped_x, clamped_y)


def calculate_bounded_movement(
    current_position: CoordinateType,
    action: ActionType,
    grid_bounds: GridSize,
    allow_boundary_clamping: bool = True,
) -> Tuple[Coordinates, bool, bool]:
    try:
        # Convert current_position to Coordinates and validate action parameter
        coords = create_coordinates(current_position)

        # Calculate movement delta from action using Action.to_vector() method
        if hasattr(action, "to_vector"):
            movement_vector = action.to_vector()
        else:
            # Convert integer action to movement vector
            action_int = int(action)
            movement_vector = MOVEMENT_VECTORS.get(action_int, (0, 0))

        # Determine proposed destination position by adding movement delta
        proposed_x = coords.x + movement_vector[0]
        proposed_y = coords.y + movement_vector[1]
        proposed_position = Coordinates(proposed_x, proposed_y)

        # Check if proposed position is within grid bounds using boundary validation
        is_within_bounds = (
            0 <= proposed_x < grid_bounds.width and 0 <= proposed_y < grid_bounds.height
        )

        if is_within_bounds:
            # If movement is valid, return proposed position with movement_successful=True
            return proposed_position, True, False
        # Movement exceeds bounds
        boundary_hit = True

        if allow_boundary_clamping:
            # If movement exceeds bounds and allow_boundary_clamping is True, clamp position to bounds
            final_position = clamp_coordinates_to_bounds(proposed_position, grid_bounds)
            movement_successful = True  # Movement succeeded with clamping
        else:
            # If movement exceeds bounds and clamping disabled, keep current position
            final_position = coords
            movement_successful = False  # Movement failed

        # Return tuple with final position, movement success status, and boundary hit indication
        return final_position, movement_successful, boundary_hit

    except Exception:
        # Handle errors by returning safe defaults
        return create_coordinates(current_position), False, True
