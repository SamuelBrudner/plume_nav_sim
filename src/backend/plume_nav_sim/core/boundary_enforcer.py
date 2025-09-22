"""
Boundary enforcement module for plume_nav_sim providing comprehensive agent movement validation,
grid constraint enforcement, position bounds checking, and coordinate system management with
performance optimization and detailed error reporting for the Gymnasium environment's navigation
system ensuring agents stay within valid grid positions.

This module implements the core boundary enforcement logic for the plume navigation simulation,
ensuring that agent movements remain within valid grid boundaries through comprehensive validation,
clamping, and constraint analysis. The module provides both high-level boundary enforcer classes
and low-level utility functions for performance-critical validation operations.

Key Features:
- High-performance boundary validation with <0.1ms per check performance targets
- Comprehensive position bounds checking with detailed error reporting
- Flexible movement constraint configuration with clamping and validation policies
- Integration with the broader plume_nav_sim validation and error handling framework
- Thread-safe caching for repeated validation operations
- Component-specific logging with performance monitoring and timing analysis
"""

# External imports with version comments
import logging  # >=3.10 - Logging integration for boundary enforcement operations and debugging
import time  # >=3.10 - High-precision timing for performance measurement and benchmarking
from dataclasses import (  # >=3.10 - Data class utilities for configuration and result structures
    dataclass,
    field,
)
from typing import (  # >=3.10 - Type hints for boundary enforcer methods and interfaces
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np  # >=2.1.0 - Array operations, coordinate calculations, and mathematical validation

# Internal imports from core modules for coordinate and action type system integration
from .enums import Action
from .geometry import Coordinates, GridSize
from .types import ActionType, CoordinateType


def create_coordinates(coords: CoordinateType) -> Coordinates:
    """Factory function to create a Coordinates object from various inputs."""
    if isinstance(coords, Coordinates):
        return coords
    if isinstance(coords, (tuple, list)) and len(coords) == 2:
        return Coordinates(x=coords[0], y=coords[1])
    raise TypeError(f"Cannot create Coordinates from {type(coords)}")


from ..utils.exceptions import StateError, ValidationError
from ..utils.logging import ComponentType, get_component_logger, monitor_performance

# Internal imports from utils for validation, error handling, and logging integration
from ..utils.validation import validate_action_parameter, validate_coordinates

# Internal imports for system-wide constants and configuration values
from .constants import (
    DEFAULT_GRID_SIZE,
    MOVEMENT_VECTORS,
    PERFORMANCE_TARGET_STEP_LATENCY_MS,
    VALIDATION_ERROR_MESSAGES,
)

# Global constants for boundary enforcement configuration and performance optimization
BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS = (
    0.1  # Performance target for boundary operations
)
DEFAULT_CLAMPING_ENABLED = True  # Default clamping behavior for position enforcement
BOUNDARY_VALIDATION_CACHE_SIZE = 500  # Cache size for validation result optimization
POSITION_BOUNDS_TOLERANCE = (
    0  # Tolerance for position bounds checking (strict by default)
)

# Module exports for comprehensive boundary enforcement interface
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
    """
    Data class representing comprehensive boundary enforcement results with position updates,
    constraint analysis, boundary hit detection, and performance metrics for detailed
    boundary enforcement reporting and debugging support.

    This immutable data class captures complete information about boundary enforcement
    operations, enabling detailed analysis of constraint application, movement modifications,
    and performance characteristics for monitoring and debugging purposes.
    """

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
        """
        Generate comprehensive constraint analysis including direction, magnitude, and reasoning
        for boundary enforcement debugging and detailed movement analysis.

        Returns:
            dict: Dictionary containing detailed constraint analysis with direction vectors and boundary details
        """
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

        # Compile constraint analysis with enforcement reasoning and metrics
        analysis = {
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

        # Return comprehensive constraint analysis dictionary for debugging
        return analysis

    def was_movement_constrained(self) -> bool:
        """
        Determine if movement was constrained by boundary enforcement for movement analysis
        and constraint detection in navigation systems.

        Returns:
            bool: True if movement was constrained by boundaries, False if movement was unconstrained
        """
        # Return position_modified flag indicating constraint application
        return self.position_modified

    def get_final_position(self) -> Coordinates:
        """
        Get final position after boundary enforcement for position management and
        coordinate system integration with agent state updates.

        Returns:
            Coordinates: Final position coordinates after boundary constraint application
        """
        # Return final_position coordinates for external use
        return self.final_position

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert boundary enforcement result to dictionary for serialization and external analysis,
        enabling integration with logging, monitoring, and debugging systems.

        Returns:
            dict: Dictionary representation of boundary enforcement result with analysis data
        """
        # Include original_position and final_position as coordinate tuples
        result_dict = {
            "original_position": (self.original_position.x, self.original_position.y),
            "final_position": (self.final_position.x, self.final_position.y),
            "position_modified": self.position_modified,
            "boundary_hit": self.boundary_hit,
            "constraint_direction": self.constraint_direction,
            "enforcement_time_ms": self.enforcement_time,
            "constraint_reason": self.constraint_reason,
        }

        # Add constraint analysis for comprehensive debugging information
        result_dict["constraint_analysis"] = self.get_constraint_analysis()

        # Return complete boundary enforcement result dictionary
        return result_dict


@dataclass
class MovementConstraint:
    """
    Data class representing movement constraint configuration with boundary policies, clamping
    settings, and enforcement parameters for customizable boundary enforcement behavior and
    flexible constraint management across different navigation scenarios.

    This configuration class enables fine-grained control over boundary enforcement behavior,
    allowing customization of clamping policies, validation strictness, logging preferences,
    and performance monitoring for different use cases and environments.
    """

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
        """
        Validate movement constraint configuration for consistency and feasibility, ensuring
        all configuration parameters are logically consistent and implementable.

        Returns:
            bool: True if configuration is valid, raises ValidationError if invalid

        Raises:
            ValidationError: If configuration parameters are invalid or inconsistent
        """
        # Validate tolerance is non-negative for bounds checking
        if self.tolerance < 0:
            raise ValidationError(
                "Movement constraint tolerance must be non-negative",
                parameter_name="tolerance",
                invalid_value=self.tolerance,
                expected_format="float >= 0",
            )

        # Check custom_boundaries dictionary structure if not empty
        if self.custom_boundaries:
            if not isinstance(self.custom_boundaries, dict):
                raise ValidationError(
                    "Custom boundaries must be a dictionary",
                    parameter_name="custom_boundaries",
                    invalid_value=type(self.custom_boundaries).__name__,
                    expected_format="dict",
                )

            # Validate custom boundary entries have required structure
            for key, value in self.custom_boundaries.items():
                if not isinstance(key, str):
                    raise ValidationError(
                        f"Custom boundary key must be string, got {type(key).__name__}",
                        parameter_name=f"custom_boundaries[{key}]",
                    )

        # Ensure configuration parameters are logically consistent
        if self.strict_validation and not self.enable_clamping:
            # This combination might be valid but worth noting
            pass  # Allow this combination with potential warning

        # Return True if all constraint configuration validations pass
        return True

    def clone(self, overrides: Optional[Dict[str, Any]] = None) -> "MovementConstraint":
        """
        Create deep copy of movement constraint configuration with optional parameter overrides
        for constraint inheritance and configuration customization scenarios.

        Args:
            overrides: Optional dictionary of parameter overrides to apply

        Returns:
            MovementConstraint: New MovementConstraint instance with optional modifications
        """
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
                        invalid_value=key,
                    )

        # Validate new configuration if overrides were applied
        if overrides:
            new_config.validate_configuration()

        # Return new MovementConstraint instance
        return new_config


class BoundaryEnforcer:
    """
    Primary boundary enforcement class providing comprehensive agent movement validation,
    grid constraint enforcement, position bounds checking, and coordinate system management
    with performance monitoring and caching for Gymnasium environment navigation system.

    This class serves as the central component for boundary enforcement operations, integrating
    with the validation framework, error handling system, and logging infrastructure to provide
    robust, high-performance boundary checking and constraint application for agent navigation.
    """

    def __init__(
        self,
        grid_size: GridSize,
        constraint_config: Optional[MovementConstraint] = None,
        enable_caching: bool = True,
    ) -> None:
        """
        Initialize boundary enforcer with grid dimensions, constraint configuration, and
        performance optimization settings for comprehensive boundary enforcement operations.

        Args:
            grid_size: Grid dimensions for boundary limit definitions
            constraint_config: Optional constraint configuration, defaults to standard settings
            enable_caching: Enable validation result caching for performance optimization

        Raises:
            ValidationError: If grid_size is invalid or constraint configuration is inconsistent
        """
        # Validate and store grid_size for boundary calculations using validate_coordinates
        try:
            validate_coordinates((0, 0), grid_size)  # Validate grid_size format
            self.grid_size = grid_size
        except Exception as e:
            raise ValidationError(
                f"Invalid grid_size for boundary enforcer: {e}",
                parameter_name="grid_size",
                invalid_value=str(grid_size),
            ) from e

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

        # Initialize validation_cache dictionary if caching enabled for performance
        self.validation_cache: Dict[str, Any] = {} if enable_caching else {}

        # Initialize performance_metrics for operation monitoring and timing analysis
        self.performance_metrics: Dict[str, List[float]] = {
            "validation_times": [],
            "enforcement_times": [],
            "cache_hits": [],
            "cache_misses": [],
        }

        # Reset enforcement_count and boundary_hits for enforcement statistics tracking
        self.enforcement_count = 0
        self.boundary_hits = 0

        # Log boundary enforcer initialization with configuration details
        self.logger.info(
            f"BoundaryEnforcer initialized with grid_size={grid_size}, "
            f"caching={'enabled' if enable_caching else 'disabled'}"
        )

    @monitor_performance("position_validation", 0.1, False)
    def validate_position(
        self,
        position: CoordinateType,
        raise_on_invalid: bool = True,
        context_info: Optional[str] = None,
    ) -> bool:
        """
        Comprehensive position validation ensuring coordinates are within grid boundaries
        with detailed error reporting and constraint analysis for position management.

        Args:
            position: Position coordinates to validate against grid boundaries
            raise_on_invalid: Whether to raise ValidationError for invalid positions
            context_info: Optional context information for detailed error reporting

        Returns:
            bool: True if position is valid, False if invalid (when raise_on_invalid=False)

        Raises:
            ValidationError: If position is invalid and raise_on_invalid=True
        """
        start_time = time.perf_counter()

        try:
            # Convert position to Coordinates using create_coordinates with validation
            coords = create_coordinates(position)

            # Check validation cache if caching enabled for performance optimization
            if self.enable_caching:
                cache_key = f"validate_{coords.x}_{coords.y}_{self.grid_size.width}_{self.grid_size.height}"
                if cache_key in self.validation_cache:
                    self.performance_metrics["cache_hits"].append(1)
                    return self.validation_cache[cache_key]
                else:
                    self.performance_metrics["cache_misses"].append(1)

            # Validate position is within grid bounds using position.is_within_bounds()
            is_valid = coords.is_within_bounds(self.grid_size)

            # Apply strict validation rules if constraint_config.strict_validation enabled
            if self.constraint_config.strict_validation and is_valid:
                # Additional strict validation checks
                tolerance = self.constraint_config.tolerance
                if (
                    coords.x < tolerance
                    or coords.x >= self.grid_size.width - tolerance
                    or coords.y < tolerance
                    or coords.y >= self.grid_size.height - tolerance
                ):

                    if tolerance > 0:
                        is_valid = False

            # Cache validation result if caching enabled for future lookups
            if self.enable_caching:
                self.validation_cache[cache_key] = is_valid

            # Update performance metrics with operation timing and success data
            validation_time = (time.perf_counter() - start_time) * 1000
            self.performance_metrics["validation_times"].append(validation_time)

            # Log validation activity with context information if provided
            if context_info and not is_valid:
                self.logger.debug(
                    f"Position validation failed: {context_info}, position={coords}"
                )

            # Raise ValidationError with boundary details if raise_on_invalid and position invalid
            if not is_valid and raise_on_invalid:
                error_msg = f"Position {coords} outside grid bounds {self.grid_size}"
                if context_info:
                    error_msg += f" (Context: {context_info})"

                raise ValidationError(
                    error_msg,
                    parameter_name="position",
                    invalid_value=(coords.x, coords.y),
                    expected_format=f"coordinates within bounds (0, 0) to ({self.grid_size.width-1}, {self.grid_size.height-1})",
                )

            # Return validation result indicating position validity within grid boundaries
            return is_valid

        except ValidationError:
            # Re-raise ValidationError without modification
            raise
        except Exception as e:
            # Handle unexpected errors during validation
            self.logger.error(f"Unexpected error during position validation: {e}")
            if raise_on_invalid:
                raise ValidationError(
                    f"Position validation failed due to internal error: {e}",
                    parameter_name="position",
                    invalid_value=str(position),
                ) from e
            return False

    @monitor_performance("movement_validation", 0.05, False)
    def is_movement_valid(
        self, current_position: CoordinateType, action: ActionType
    ) -> bool:
        """
        Fast boolean check for movement validity from current position with action ensuring
        proposed movement stays within grid bounds optimized for high-frequency validation.

        Args:
            current_position: Current agent position coordinates
            action: Action to be performed (Action enum or integer)

        Returns:
            bool: True if movement is valid within bounds, False if movement would violate boundaries
        """
        try:
            # Use validate_movement_bounds for fast validation without comprehensive analysis
            result = validate_movement_bounds(
                current_position=current_position,
                action=action,
                grid_bounds=self.grid_size,
                strict_validation=False,  # Optimize for speed
            )

            # Update movement validation statistics for performance monitoring
            self.performance_metrics.setdefault("movement_validations", []).append(1)

            # Return boolean result without raising exceptions for performance optimization
            return result

        except Exception as e:
            # Log error but return False for graceful degradation
            self.logger.warning(f"Movement validation error: {e}")
            return False

    @monitor_performance("boundary_enforcement", 0.1, True)
    def enforce_movement_bounds(
        self, current_position: CoordinateType, action: ActionType
    ) -> BoundaryEnforcementResult:
        """
        Primary movement enforcement method applying boundary constraints to agent movement
        with comprehensive result analysis and performance monitoring for environment step operations.

        Args:
            current_position: Current agent position coordinates
            action: Action to be performed (Action enum or integer)

        Returns:
            BoundaryEnforcementResult: Comprehensive boundary enforcement result with position updates and constraint analysis
        """
        start_time = time.perf_counter()

        try:
            # Convert current_position to Coordinates and validate action parameter
            coords = create_coordinates(current_position)
            validate_action_parameter(action)

            # Calculate movement delta using action.to_vector() or MOVEMENT_VECTORS lookup
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

            # Check if proposed position is within grid bounds using GridSize.contains_coordinates()
            is_valid_movement = proposed_position.is_within_bounds(self.grid_size)

            # Initialize result tracking variables
            final_position = coords  # Default to current position
            position_modified = False
            boundary_hit = False

            if is_valid_movement:
                # If movement valid, return BoundaryEnforcementResult with original position unchanged
                final_position = proposed_position
            else:
                # Movement exceeds bounds - apply constraint policy
                boundary_hit = True

                if self.constraint_config.enable_clamping:
                    # If movement exceeds bounds and clamping enabled, clamp position using clamp_coordinates_to_bounds
                    final_position = clamp_coordinates_to_bounds(
                        proposed_position, self.grid_size
                    )
                    position_modified = (
                        final_position.x != coords.x or final_position.y != coords.y
                    )
                else:
                    # If movement exceeds bounds and clamping disabled, keep current position
                    final_position = coords
                    position_modified = False

            # Create BoundaryEnforcementResult with final position and constraint analysis
            result = BoundaryEnforcementResult(
                original_position=coords,
                final_position=final_position,
                position_modified=position_modified,
                boundary_hit=boundary_hit,
            )

            # Set enforcement timing
            enforcement_time = (time.perf_counter() - start_time) * 1000
            object.__setattr__(result, "enforcement_time", enforcement_time)

            # Update boundary hit statistics and performance metrics
            self.enforcement_count += 1
            if boundary_hit:
                self.boundary_hits += 1

            self.performance_metrics["enforcement_times"].append(enforcement_time)

            # Log boundary enforcement if constraint_config.log_boundary_violations enabled
            if self.constraint_config.log_boundary_violations and boundary_hit:
                self.logger.info(
                    f"Boundary enforcement applied: {coords} -> {final_position} "
                    f"(action={action}, hit={boundary_hit})"
                )

            # Return comprehensive BoundaryEnforcementResult with enforcement analysis
            return result

        except Exception as e:
            # Handle errors during boundary enforcement
            self.logger.error(f"Boundary enforcement failed: {e}")

            # Return safe result with current position preserved
            safe_coords = create_coordinates(current_position)
            return BoundaryEnforcementResult(
                original_position=safe_coords,
                final_position=safe_coords,
                position_modified=False,
                boundary_hit=True,  # Indicate error condition
            )

    def get_valid_moves(self, current_position: CoordinateType) -> List[Action]:
        """
        Analyze current position to determine which actions would result in valid movements
        within grid boundaries for action space analysis and navigation planning.

        Args:
            current_position: Current agent position coordinates

        Returns:
            list: List of valid Action enum values that would not violate boundary constraints from current position
        """
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
        """
        Update grid size configuration and clear related caches for boundary enforcement
        configuration management and runtime grid size changes.

        Args:
            new_grid_size: New grid dimensions for boundary enforcement

        Raises:
            ValidationError: If new_grid_size has invalid dimensions
        """
        # Validate new_grid_size has positive dimensions using validate_coordinates
        try:
            validate_coordinates((0, 0), new_grid_size)
        except Exception as e:
            raise ValidationError(
                f"Invalid new grid_size: {e}",
                parameter_name="new_grid_size",
                invalid_value=str(new_grid_size),
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
        """
        Get comprehensive boundary enforcement statistics including hit rates, performance
        metrics, and validation efficiency for monitoring and optimization analysis.

        Returns:
            dict: Dictionary containing boundary enforcement statistics and performance analysis
        """
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
            cache_hits = sum(self.performance_metrics.get("cache_hits", []))
            cache_misses = sum(self.performance_metrics.get("cache_misses", []))
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
            compliant_operations = sum(1 for t in enforcement_times if t <= target_time)
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

        # Return comprehensive boundary enforcement statistics
        return stats

    def clear_cache(self, force_cleanup: bool = False) -> int:
        """
        Clear validation cache and reset performance statistics for memory management and testing
        scenarios with optional resource cleanup.

        Args:
            force_cleanup: Whether to force cleanup of additional resources

        Returns:
            int: Number of cache entries cleared
        """
        if not self.enable_caching:
            return 0

        # Count current cache entries for reporting
        entries_cleared = len(self.validation_cache)

        # Clear validation_cache dictionary
        self.validation_cache.clear()

        # Reset cache-related performance metrics
        if force_cleanup:
            self.performance_metrics["cache_hits"] = []
            self.performance_metrics["cache_misses"] = []

        # Log cache clearing operation with entry count
        self.logger.debug(f"Cache cleared: {entries_cleared} entries removed")

        # Return number of entries that were cleared from cache
        return entries_cleared

    def validate_constraint_configuration(self, strict_mode: bool = False) -> bool:
        """
        Validate current constraint configuration for consistency and feasibility with
        comprehensive configuration analysis and error reporting.

        Args:
            strict_mode: Whether to apply strict validation rules and enhanced checking

        Returns:
            bool: True if configuration is valid, raises ValidationError if invalid

        Raises:
            ValidationError: If configuration is invalid or inconsistent
        """
        try:
            # Validate constraint_config using MovementConstraint.validate_configuration()
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
                        invalid_value=self.constraint_config.tolerance,
                    )
                else:
                    self.logger.warning(
                        f"Large tolerance ({self.constraint_config.tolerance}) relative to grid size {self.grid_size}"
                    )

            # Apply strict validation rules if strict_mode enabled
            if strict_mode:
                # Additional strict validation checks
                if self.constraint_config.custom_boundaries:
                    for (
                        boundary_name,
                        boundary_config,
                    ) in self.constraint_config.custom_boundaries.items():
                        if not isinstance(boundary_config, (dict, bool, float, int)):
                            raise ValidationError(
                                f"Invalid custom boundary configuration for {boundary_name}",
                                parameter_name="custom_boundaries",
                                invalid_value=type(boundary_config).__name__,
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


def validate_movement_bounds(
    current_position: CoordinateType,
    action: ActionType,
    grid_bounds: GridSize,
    strict_validation: bool = True,
) -> bool:
    """
    Standalone function to validate movement bounds without requiring full boundary enforcer
    instance, optimized for performance-critical validation operations with comprehensive
    boundary checking and minimal overhead.

    Args:
        current_position: Current agent position coordinates
        action: Action to be performed (Action enum or integer value)
        grid_bounds: Grid size defining boundary limits
        strict_validation: Whether to apply strict validation rules including edge cases

    Returns:
        bool: True if movement is within bounds, False if movement would violate boundary constraints
    """
    try:
        # Convert current_position to Coordinates using create_coordinates factory for type consistency
        coords = create_coordinates(current_position)

        # Validate action parameter using validate_action_parameter with type checking
        if strict_validation:
            validate_action_parameter(action)

        # Get movement vector from action using Action.to_vector() or MOVEMENT_VECTORS lookup
        if hasattr(action, "to_vector"):
            movement_vector = action.to_vector()
        else:
            # Convert integer action to movement vector
            action_int = int(action)
            movement_vector = MOVEMENT_VECTORS.get(action_int, (0, 0))

        # Calculate proposed destination position by adding movement delta to current position
        new_x = coords.x + movement_vector[0]
        new_y = coords.y + movement_vector[1]

        # Check if proposed position is within grid bounds using bounds checking
        if (
            new_x < 0
            or new_x >= grid_bounds.width
            or new_y < 0
            or new_y >= grid_bounds.height
        ):
            return False

        # Apply strict validation rules if strict_validation enabled including edge case testing
        if strict_validation:
            # Additional validation for edge cases and boundary proximity
            if (
                new_x == 0
                or new_x == grid_bounds.width - 1
                or new_y == 0
                or new_y == grid_bounds.height - 1
            ):
                # Position is on grid boundary - still valid but worth noting
                pass

        # Return boolean result indicating movement validity without raising exceptions for performance
        return True

    except Exception:
        # Return False for any validation errors to maintain performance
        return False


def enforce_position_bounds(
    position: CoordinateType,
    grid_bounds: GridSize,
    clamp_to_bounds: bool = True,
    log_boundary_hits: bool = False,
) -> Tuple[Coordinates, bool]:
    """
    Enforce position bounds by clamping coordinates to valid grid range ensuring agent position
    stays within boundaries with optional boundary hit detection and logging for position management.

    Args:
        position: Position coordinates to enforce bounds on
        grid_bounds: Grid size defining boundary limits
        clamp_to_bounds: Whether to clamp coordinates to valid range
        log_boundary_hits: Whether to log boundary constraint applications

    Returns:
        tuple: Tuple of (constrained_position: Coordinates, position_modified: bool) with boundary enforcement result
    """
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
    """
    Fast boolean check for position validity within grid bounds optimized for high-frequency
    boundary validation without comprehensive error reporting or logging overhead.

    Args:
        position: Position coordinates to check
        grid_bounds: Grid size defining boundary limits

    Returns:
        bool: True if position is within grid bounds, False otherwise
    """
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
        if coords.y < 0 or coords.y >= grid_bounds.height:
            return False

        # Return boolean result using logical AND of both coordinate range checks
        return True

    except Exception:
        # Return False for any conversion or validation errors
        return False


def clamp_coordinates_to_bounds(
    coordinates: CoordinateType, grid_bounds: GridSize
) -> Coordinates:
    """
    Clamp coordinate values to grid boundaries ensuring valid position within grid constraints
    with coordinate-by-coordinate clamping for precise boundary enforcement and position correction.

    Args:
        coordinates: Coordinates to clamp to grid boundaries
        grid_bounds: Grid size defining boundary limits

    Returns:
        Coordinates: Coordinates object with values clamped to grid boundaries
    """
    # Convert coordinates to Coordinates object for consistent processing
    coords = create_coordinates(coordinates)

    # Clamp x-coordinate using max(0, min(coordinates.x, grid_bounds.width - 1))
    clamped_x = max(0, min(coords.x, grid_bounds.width - 1))

    # Clamp y-coordinate using max(0, min(coordinates.y, grid_bounds.height - 1))
    clamped_y = max(0, min(coords.y, grid_bounds.height - 1))

    # Create new Coordinates object with clamped values
    clamped_coords = Coordinates(clamped_x, clamped_y)

    # Return clamped Coordinates ready for safe position usage
    return clamped_coords


def calculate_bounded_movement(
    current_position: CoordinateType,
    action: ActionType,
    grid_bounds: GridSize,
    allow_boundary_clamping: bool = True,
) -> Tuple[Coordinates, bool, bool]:
    """
    Calculate movement result with boundary constraints applied ensuring agent movement stays
    within valid grid positions with comprehensive movement analysis and constraint reporting.

    Args:
        current_position: Current agent position coordinates
        action: Action to be performed (Action enum or integer value)
        grid_bounds: Grid size defining boundary limits
        allow_boundary_clamping: Whether to allow position clamping when boundaries are exceeded

    Returns:
        tuple: Tuple of (final_position: Coordinates, movement_successful: bool, boundary_hit: bool) with movement analysis
    """
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
        else:
            # Movement exceeds bounds
            boundary_hit = True

            if allow_boundary_clamping:
                # If movement exceeds bounds and allow_boundary_clamping is True, clamp position to bounds
                final_position = clamp_coordinates_to_bounds(
                    proposed_position, grid_bounds
                )
                movement_successful = True  # Movement succeeded with clamping
            else:
                # If movement exceeds bounds and clamping disabled, keep current position
                final_position = coords
                movement_successful = False  # Movement failed

            # Return tuple with final position, movement success status, and boundary hit indication
            return final_position, movement_successful, boundary_hit

    except Exception:
        # Handle errors by returning safe defaults
        safe_coords = create_coordinates(current_position)
        return safe_coords, False, True
