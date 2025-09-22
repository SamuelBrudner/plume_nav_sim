"""
Core action processing module for plume_nav_sim providing comprehensive action validation,
movement calculation, boundary-aware action processing, and coordinate system integration
for the Gymnasium-compatible reinforcement learning environment with performance optimization,
error handling, and state transition management.

This module implements:
- ActionProcessor: Primary action processing class with comprehensive validation and boundary enforcement
- ActionProcessingResult: Detailed results with movement outcome and performance metrics
- ActionProcessingConfig: Configuration for processor behavior and performance tuning
- Standalone utility functions for performance-critical scenarios
- Caching and performance monitoring for <0.1ms action processing targets
- Integration with boundary enforcement and coordinate system validation
"""

import dataclasses  # >=3.10
import functools  # standard library
import logging  # standard library
import time  # standard library
from typing import Any, Dict, List, Optional, Tuple, Union  # >=3.10

import numpy as np  # >=2.1.0

# Internal imports for core data structures and type system
from .enums import Action
from .geometry import Coordinates, GridSize
from .state import AgentState
from .types import ActionType, MovementVector


def validate_action(action: ActionType) -> bool:
    """Placeholder for the real validation function."""
    return True


def get_movement_vector(action: Action) -> MovementVector:
    """Placeholder for the real movement vector function."""
    return MOVEMENT_VECTORS[action.value]


# Exception handling system for error reporting and recovery
from ..utils.exceptions import ComponentError, StateError, ValidationError

# Logging and performance monitoring utilities
from ..utils.logging import ComponentType, get_component_logger, monitor_performance

# Validation utilities for comprehensive parameter checking
from ..utils.validation import (
    create_validation_context,
    validate_action_parameter,
    validate_coordinates,
)

# Boundary enforcement integration for movement validation
from .boundary_enforcer import BoundaryEnforcementResult, BoundaryEnforcer

# System constants for action processing and performance targets
from .constants import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_SPACE_SIZE,
    ACTION_UP,
    MOVEMENT_VECTORS,
    PERFORMANCE_TARGET_STEP_LATENCY_MS,
    VALIDATION_ERROR_MESSAGES,
)

# Global configuration constants for action processing optimization
ACTION_PROCESSING_PERFORMANCE_TARGET_MS = 0.1
ACTION_VALIDATION_CACHE_SIZE = 100
DEFAULT_ENABLE_BOUNDARY_ENFORCEMENT = True
MOVEMENT_CALCULATION_PRECISION = 1e-10

# Module exports for public API
__all__ = [
    "ActionProcessor",
    "ActionProcessingResult",
    "ActionProcessingConfig",
    "process_action",
    "validate_action_bounds",
    "calculate_movement_delta",
    "is_valid_action_for_position",
]


@dataclasses.dataclass(frozen=True)
class ActionProcessingResult:
    """
    Comprehensive data class for action processing results containing movement outcome,
    boundary enforcement details, performance metrics, and validation status for detailed
    analysis and environment integration.

    Attributes:
        action: The processed action (Action enum or integer)
        original_position: Agent position before processing
        final_position: Agent position after processing and boundary enforcement
        action_valid: Whether the action was valid according to validation rules
        position_changed: Whether the agent position was modified during processing
        boundary_hit: Whether boundary constraints affected the movement
        movement_delta: The coordinate delta vector (dx, dy) applied
        processing_time_ms: Time taken to process the action in milliseconds
        validation_error: Optional validation error message if action was invalid
    """

    # Required parameters for action processing result creation
    action: ActionType
    original_position: Coordinates
    final_position: Coordinates
    action_valid: bool

    # Computed properties initialized in __post_init__
    position_changed: bool = dataclasses.field(init=False)
    boundary_hit: bool = dataclasses.field(default=False)
    movement_delta: MovementVector = dataclasses.field(default=(0, 0))
    processing_time_ms: float = dataclasses.field(default=0.0)
    validation_error: Optional[str] = dataclasses.field(default=None)

    def __post_init__(self):
        """Initialize computed properties after dataclass creation."""
        # Calculate position_changed by comparing original and final positions
        object.__setattr__(
            self, "position_changed", self.original_position != self.final_position
        )

    def was_movement_successful(self) -> bool:
        """
        Determine if movement was successful based on action validity and position change.

        Returns:
            True if action was valid and resulted in successful movement
        """
        return self.action_valid and self.position_changed

    def get_movement_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive movement summary for analysis and debugging.

        Returns:
            Dictionary containing movement details, boundary status, and performance metrics
        """
        return {
            "action": (
                int(self.action) if isinstance(self.action, Action) else self.action
            ),
            "original_position": {
                "x": self.original_position.x,
                "y": self.original_position.y,
            },
            "final_position": {"x": self.final_position.x, "y": self.final_position.y},
            "movement_successful": self.was_movement_successful(),
            "position_changed": self.position_changed,
            "boundary_hit": self.boundary_hit,
            "movement_delta": self.movement_delta,
            "action_valid": self.action_valid,
            "processing_time_ms": self.processing_time_ms,
            "validation_error": self.validation_error,
        }

    def to_dict(self, include_performance_data: bool = True) -> Dict[str, Any]:
        """
        Convert action processing result to dictionary for serialization and external analysis.

        Args:
            include_performance_data: Whether to include timing and performance metrics

        Returns:
            Dictionary representation of action processing result
        """
        result = {
            "action": (
                int(self.action) if isinstance(self.action, Action) else self.action
            ),
            "original_position": (self.original_position.x, self.original_position.y),
            "final_position": (self.final_position.x, self.final_position.y),
            "action_valid": self.action_valid,
            "position_changed": self.position_changed,
            "boundary_hit": self.boundary_hit,
            "movement_delta": self.movement_delta,
        }

        if include_performance_data:
            result["processing_time_ms"] = self.processing_time_ms

        if self.validation_error:
            result["validation_error"] = self.validation_error

        return result


@dataclasses.dataclass
class ActionProcessingConfig:
    """
    Configuration data class for action processor behavior including validation settings,
    boundary enforcement policies, performance monitoring, and error handling options.

    Attributes:
        enable_validation: Enable comprehensive action validation
        enforce_boundaries: Enable boundary constraint enforcement during movement
        enable_performance_monitoring: Enable timing and performance tracking
        strict_validation: Enable strict validation with comprehensive checking
        cache_validation_results: Enable caching of validation results for performance
        performance_target_ms: Target processing time in milliseconds
        log_action_processing: Enable detailed logging of action processing operations
    """

    # Required configuration parameters
    enable_validation: bool = True
    enforce_boundaries: bool = DEFAULT_ENABLE_BOUNDARY_ENFORCEMENT
    enable_performance_monitoring: bool = True

    # Optional configuration with defaults
    strict_validation: bool = False
    cache_validation_results: bool = True
    performance_target_ms: float = ACTION_PROCESSING_PERFORMANCE_TARGET_MS
    log_action_processing: bool = False

    def validate_configuration(self) -> bool:
        """
        Validate action processing configuration for consistency and feasibility.

        Returns:
            True if configuration is valid

        Raises:
            ValidationError: If configuration is invalid or inconsistent
        """
        # Validate performance_target_ms is positive value
        if self.performance_target_ms <= 0:
            raise ValidationError(
                "performance_target_ms must be positive value",
                parameter_constraints={"min_value": 0.001, "max_value": 1000.0},
            )

        # Check configuration parameter compatibility
        if self.strict_validation and not self.enable_validation:
            raise ValidationError(
                "strict_validation requires enable_validation to be True",
                recovery_suggestion="Set enable_validation=True or strict_validation=False",
            )

        # Ensure reasonable performance targets
        if self.performance_target_ms < 0.01:  # Less than 10 microseconds
            raise ValidationError(
                "performance_target_ms too aggressive, may cause performance issues",
                parameter_constraints={"recommended_min": 0.01},
            )

        return True

    def clone(
        self, overrides: Optional[Dict[str, Any]] = None
    ) -> "ActionProcessingConfig":
        """
        Create deep copy of action processing configuration with optional parameter overrides.

        Args:
            overrides: Optional dictionary of parameters to override

        Returns:
            New configuration instance with optional modifications
        """
        # Create copy of current configuration parameters
        config_data = dataclasses.asdict(self)

        # Apply parameter overrides if provided
        if overrides:
            config_data.update(overrides)

        # Create new ActionProcessingConfig with copied/modified values
        new_config = ActionProcessingConfig(**config_data)

        # Validate new configuration if overrides were applied
        if overrides:
            new_config.validate_configuration()

        return new_config


class ActionProcessor:
    """
    Primary action processing class providing comprehensive action validation, movement
    calculation, boundary enforcement integration, and performance monitoring for Gymnasium
    environment step operations with caching and error handling.

    This class serves as the central component for processing agent actions with:
    - Comprehensive action validation with customizable strictness
    - Movement calculation with high-precision coordinate arithmetic
    - Boundary enforcement integration for grid constraint compliance
    - Performance monitoring with sub-millisecond timing targets
    - Caching for validation results and movement calculations
    - Extensive error handling with recovery suggestions
    """

    def __init__(
        self,
        grid_size: GridSize,
        config: Optional[ActionProcessingConfig] = None,
        boundary_enforcer: Optional[BoundaryEnforcer] = None,
    ):
        """
        Initialize action processor with grid size, configuration, and optional boundary enforcer.

        Args:
            grid_size: Grid dimensions for boundary validation and movement calculations
            config: Action processing configuration, defaults to standard settings
            boundary_enforcer: Optional boundary enforcer, created if None and boundaries enabled

        Raises:
            ValidationError: If grid_size is invalid or configuration is inconsistent
            ComponentError: If initialization fails
        """
        try:
            # Validate and store grid_size for boundary constraint validation
            if not isinstance(grid_size, GridSize):
                raise ValidationError("grid_size must be GridSize instance")

            if grid_size.width <= 0 or grid_size.height <= 0:
                raise ValidationError("grid_size dimensions must be positive")

            self.grid_size = grid_size

            # Set config to provided value or create default ActionProcessingConfig
            self.config = config if config is not None else ActionProcessingConfig()
            self.config.validate_configuration()

            # Initialize boundary_enforcer if needed
            if boundary_enforcer is not None:
                self.boundary_enforcer = boundary_enforcer
            elif self.config.enforce_boundaries:
                self.boundary_enforcer = BoundaryEnforcer(grid_size)
            else:
                self.boundary_enforcer = None

            # Initialize component logger for action processing operations
            self.logger = get_component_logger(
                component_name="action_processor",
                component_type=ComponentType.UTILS,
                enable_performance_tracking=self.config.enable_performance_monitoring,
            )

            # Initialize validation cache if caching enabled
            if self.config.cache_validation_results:
                self.validation_cache: Dict[str, bool] = {}
            else:
                self.validation_cache = {}

            # Initialize performance metrics for operation timing and analysis
            self.performance_metrics: Dict[str, List[float]] = {
                "action_processing": [],
                "validation_time": [],
                "movement_calculation": [],
                "boundary_enforcement": [],
            }

            # Initialize statistics counters
            self.actions_processed = 0
            self.validation_errors = 0

            self.logger.info(
                f"ActionProcessor initialized for grid {grid_size.width}x{grid_size.height}",
                extra={
                    "grid_size": {"width": grid_size.width, "height": grid_size.height},
                    "config": dataclasses.asdict(self.config),
                    "boundary_enforcement_enabled": self.boundary_enforcer is not None,
                },
            )

        except Exception as e:
            raise ComponentError(
                f"Failed to initialize ActionProcessor: {e}",
                component_name="action_processor",
                operation_name="__init__",
                underlying_error=e,
            ) from e

    @monitor_performance(
        "action_processing", ACTION_PROCESSING_PERFORMANCE_TARGET_MS, True
    )
    def process_action(
        self, action: ActionType, current_position: Coordinates
    ) -> ActionProcessingResult:
        """
        Primary action processing method with comprehensive validation, movement calculation,
        boundary enforcement, and performance monitoring for environment step integration.

        Args:
            action: Action to process (Action enum or integer 0-3)
            current_position: Current agent position for movement calculation

        Returns:
            Comprehensive action processing result with movement outcome and analysis

        Raises:
            ValidationError: If action or position parameters are invalid
            StateError: If action processing fails due to invalid state
        """
        start_time = time.perf_counter()
        validation_error_msg = None

        try:
            # Validate action parameter if validation enabled
            if self.config.enable_validation:
                is_valid = self.validate_action(action, raise_on_invalid=False)
                if not is_valid:
                    validation_error_msg = f"Invalid action: {action}"
                    if self.config.strict_validation:
                        raise ValidationError(validation_error_msg)
            else:
                is_valid = True

            # Calculate movement delta with error handling
            try:
                movement_delta = calculate_movement_delta(
                    action, validate_action=self.config.enable_validation
                )
            except Exception as e:
                is_valid = False
                movement_delta = (0, 0)
                validation_error_msg = f"Movement calculation failed: {e}"
                if self.config.strict_validation:
                    raise StateError(f"Failed to calculate movement delta: {e}") from e

            # Apply boundary enforcement if enabled
            boundary_hit = False
            if self.boundary_enforcer and self.config.enforce_boundaries:
                try:
                    final_position, movement_delta, boundary_hit, position_changed = (
                        self.calculate_movement_outcome(
                            action, current_position, apply_boundary_enforcement=True
                        )
                    )
                except Exception as e:
                    final_position = current_position
                    boundary_hit = True
                    is_valid = False
                    validation_error_msg = f"Boundary enforcement failed: {e}"
                    self.logger.warning(f"Boundary enforcement error: {e}")
            else:
                # Calculate final position without boundary enforcement
                try:
                    final_position = current_position.move(movement_delta)
                except Exception as e:
                    final_position = current_position
                    is_valid = False
                    validation_error_msg = f"Position calculation failed: {e}"

            # Calculate processing time
            processing_time = (time.perf_counter() - start_time) * 1000.0

            # Create comprehensive ActionProcessingResult
            result = ActionProcessingResult(
                action=action,
                original_position=current_position,
                final_position=final_position,
                action_valid=is_valid,
            )

            # Set computed fields using object.__setattr__ for frozen dataclass
            object.__setattr__(result, "boundary_hit", boundary_hit)
            object.__setattr__(result, "movement_delta", movement_delta)
            object.__setattr__(result, "processing_time_ms", processing_time)
            object.__setattr__(result, "validation_error", validation_error_msg)

            # Update performance metrics and statistics
            self.performance_metrics["action_processing"].append(processing_time)
            self.actions_processed += 1
            if not is_valid:
                self.validation_errors += 1

            # Log performance warning if target exceeded
            if processing_time > self.config.performance_target_ms:
                self.logger.warning(
                    f"Action processing exceeded target: {processing_time:.3f}ms > {self.config.performance_target_ms}ms"
                )

            # Log processing activity if enabled
            if self.config.log_action_processing:
                self.logger.debug(
                    f"Processed action {action}: {current_position} → {final_position} "
                    f"({'valid' if is_valid else 'invalid'}, {processing_time:.3f}ms)"
                )

            return result

        except Exception as e:
            # Update error statistics
            self.validation_errors += 1

            # Create error result
            error_result = ActionProcessingResult(
                action=action,
                original_position=current_position,
                final_position=current_position,
                action_valid=False,
            )
            object.__setattr__(error_result, "validation_error", str(e))
            object.__setattr__(
                error_result,
                "processing_time_ms",
                (time.perf_counter() - start_time) * 1000.0,
            )

            # Log error with context
            self.logger.error(f"Action processing failed: {e}", exception=e)

            if isinstance(e, (ValidationError, StateError)):
                raise
            else:
                raise ComponentError(
                    f"Unexpected error in action processing: {e}",
                    component_name="action_processor",
                    operation_name="process_action",
                    underlying_error=e,
                ) from e

    @functools.lru_cache(maxsize=ACTION_VALIDATION_CACHE_SIZE)
    def validate_action(
        self, action: ActionType, raise_on_invalid: bool = True
    ) -> bool:
        """
        Comprehensive action validation with type checking, bounds validation, and Action
        enum compatibility for runtime action processing.

        Args:
            action: Action to validate (Action enum or integer)
            raise_on_invalid: Whether to raise exception if action is invalid

        Returns:
            True if action is valid, False if invalid (when raise_on_invalid=False)

        Raises:
            ValidationError: If action is invalid and raise_on_invalid=True
        """
        try:
            # Use validate_action_parameter with validation context
            context = create_validation_context(
                operation_name="validate_action", component_name="action_processor"
            )

            # Perform comprehensive validation
            if self.config.strict_validation:
                # Strict mode: comprehensive type and bounds checking
                is_valid = validate_action_parameter(
                    action, context=context, strict_validation=True
                )
            else:
                # Standard mode: basic bounds validation
                is_valid = validate_action_bounds(action, strict_validation=False)

            if not is_valid and raise_on_invalid:
                constraints = {
                    "valid_range": f"0 to {ACTION_SPACE_SIZE - 1}",
                    "valid_types": ["Action enum", "int"],
                }
                error = ValidationError(f"Invalid action: {action}")
                error.set_parameter_constraints(constraints)
                raise error

            return is_valid

        except ValidationError:
            if raise_on_invalid:
                raise
            return False
        except Exception as e:
            self.logger.error(f"Action validation failed: {e}", exception=e)
            if raise_on_invalid:
                raise ValidationError(f"Action validation error: {e}") from e
            return False

    def get_valid_actions(self, current_position: Coordinates) -> List[Action]:
        """
        Analyze current position to determine which actions would result in valid movements
        within grid boundaries for action space filtering.

        Args:
            current_position: Current agent position for movement analysis

        Returns:
            List of valid Action enum values that would not violate boundary constraints
        """
        valid_actions = []

        try:
            # Iterate through all possible actions
            for action in [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]:
                try:
                    # Check if action would result in valid movement
                    if is_valid_action_for_position(
                        action, current_position, self.grid_size
                    ):
                        valid_actions.append(action)
                except Exception as e:
                    self.logger.debug(f"Error checking action {action} validity: {e}")
                    continue

            # Use boundary_enforcer for enhanced analysis if available
            if self.boundary_enforcer:
                try:
                    boundary_valid_actions = self.boundary_enforcer.get_valid_moves(
                        current_position
                    )
                    # Intersect with our calculated valid actions for consistency
                    if boundary_valid_actions:
                        action_ints = [int(action) for action in valid_actions]
                        filtered_actions = [
                            Action(action_int)
                            for action_int in boundary_valid_actions
                            if action_int in action_ints
                        ]
                        valid_actions = filtered_actions
                except Exception as e:
                    self.logger.warning(f"Boundary enforcer analysis failed: {e}")

            return valid_actions

        except Exception as e:
            self.logger.error(f"Failed to get valid actions: {e}", exception=e)
            # Return all actions as fallback
            return [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]

    def calculate_movement_outcome(
        self,
        action: ActionType,
        current_position: Coordinates,
        apply_boundary_enforcement: bool = True,
    ) -> Tuple[Coordinates, MovementVector, bool, bool]:
        """
        Calculate comprehensive movement outcome including final position, boundary effects,
        and movement analysis for detailed action processing.

        Args:
            action: Action to process for movement calculation
            current_position: Starting position for movement
            apply_boundary_enforcement: Whether to apply boundary constraints

        Returns:
            Tuple of (final_position, movement_delta, boundary_hit, position_changed)
        """
        try:
            # Calculate movement delta
            movement_delta = calculate_movement_delta(action, validate_action=False)

            # Compute proposed destination
            proposed_position = current_position.move(movement_delta)

            boundary_hit = False
            final_position = proposed_position

            # Apply boundary enforcement if enabled
            if apply_boundary_enforcement and self.boundary_enforcer:
                enforcement_result = self.boundary_enforcer.enforce_movement_bounds(
                    current_position, action
                )

                final_position = enforcement_result.final_position
                boundary_hit = enforcement_result.boundary_hit

                # Update movement delta if position was modified
                if enforcement_result.position_modified:
                    actual_delta = (
                        final_position.x - current_position.x,
                        final_position.y - current_position.y,
                    )
                    movement_delta = actual_delta

            # Calculate position_changed
            position_changed = final_position != current_position

            return final_position, movement_delta, boundary_hit, position_changed

        except Exception as e:
            self.logger.error(f"Movement calculation failed: {e}", exception=e)
            # Return safe defaults
            return current_position, (0, 0), True, False

    def get_processing_statistics(
        self,
        include_cache_analysis: bool = True,
        include_performance_trends: bool = True,
    ) -> Dict[str, Any]:
        """
        Get comprehensive action processing statistics including performance metrics,
        cache efficiency, and error analysis for monitoring.

        Args:
            include_cache_analysis: Whether to include cache hit/miss statistics
            include_performance_trends: Whether to include performance trend analysis

        Returns:
            Processing statistics with performance analysis, cache metrics, and error rates
        """
        try:
            # Calculate basic statistics
            success_rate = (self.actions_processed - self.validation_errors) / max(
                1, self.actions_processed
            )

            statistics = {
                "actions_processed": self.actions_processed,
                "validation_errors": self.validation_errors,
                "success_rate": success_rate,
                "error_rate": self.validation_errors / max(1, self.actions_processed),
            }

            # Add performance analysis
            if (
                self.performance_metrics["action_processing"]
                and include_performance_trends
            ):
                processing_times = self.performance_metrics["action_processing"]
                statistics["performance_analysis"] = {
                    "average_processing_time_ms": np.mean(processing_times),
                    "median_processing_time_ms": np.median(processing_times),
                    "max_processing_time_ms": np.max(processing_times),
                    "min_processing_time_ms": np.min(processing_times),
                    "target_compliance_rate": sum(
                        1
                        for t in processing_times
                        if t <= self.config.performance_target_ms
                    )
                    / len(processing_times),
                }

            # Add cache analysis
            if include_cache_analysis and hasattr(self, "validation_cache"):
                cache_size = len(self.validation_cache)
                statistics["cache_analysis"] = {
                    "cache_size": cache_size,
                    "cache_enabled": self.config.cache_validation_results,
                    "estimated_hit_rate": min(
                        0.8, cache_size / max(1, self.actions_processed)
                    ),
                }

            # Add configuration information
            statistics["configuration"] = {
                "validation_enabled": self.config.enable_validation,
                "boundary_enforcement_enabled": self.config.enforce_boundaries,
                "performance_monitoring_enabled": self.config.enable_performance_monitoring,
                "performance_target_ms": self.config.performance_target_ms,
            }

            return statistics

        except Exception as e:
            self.logger.error(f"Failed to generate statistics: {e}", exception=e)
            return {"error": str(e), "actions_processed": self.actions_processed}

    def clear_cache(self) -> Dict[str, Any]:
        """
        Clear validation cache and reset performance statistics for memory management and testing.

        Returns:
            Cache clearing report with statistics and memory usage analysis
        """
        try:
            # Calculate current cache size
            initial_cache_size = (
                len(self.validation_cache) if self.validation_cache else 0
            )
            initial_metrics_count = sum(
                len(metrics) for metrics in self.performance_metrics.values()
            )

            # Clear validation cache
            if hasattr(self, "validation_cache"):
                self.validation_cache.clear()

            # Reset performance statistics
            for metric_name in self.performance_metrics:
                self.performance_metrics[metric_name] = []

            # Clear boundary enforcer cache if available
            if self.boundary_enforcer and hasattr(
                self.boundary_enforcer, "clear_cache"
            ):
                try:
                    self.boundary_enforcer.clear_cache()
                except Exception as e:
                    self.logger.warning(f"Failed to clear boundary enforcer cache: {e}")

            # Generate cache clearing report
            report = {
                "cache_cleared": True,
                "initial_cache_size": initial_cache_size,
                "initial_metrics_count": initial_metrics_count,
                "current_cache_size": len(self.validation_cache),
                "current_metrics_count": sum(
                    len(metrics) for metrics in self.performance_metrics.values()
                ),
                "memory_freed_estimate_kb": (initial_cache_size + initial_metrics_count)
                * 0.1,
            }

            self.logger.info(
                f"Cache cleared: {initial_cache_size} entries, {initial_metrics_count} metrics"
            )

            return report

        except Exception as e:
            self.logger.error(f"Cache clearing failed: {e}", exception=e)
            return {"cache_cleared": False, "error": str(e)}

    def update_configuration(self, new_config: ActionProcessingConfig) -> None:
        """
        Update action processor configuration and apply changes to processing behavior.

        Args:
            new_config: New configuration to apply

        Raises:
            ValidationError: If new configuration is invalid
        """
        try:
            # Validate new configuration
            new_config.validate_configuration()

            # Check if significant changes require cache clearing
            cache_settings_changed = (
                new_config.cache_validation_results
                != self.config.cache_validation_results
                or new_config.strict_validation != self.config.strict_validation
            )

            # Update configuration
            old_config = dataclasses.asdict(self.config)
            self.config = new_config

            # Clear cache if settings changed
            if cache_settings_changed:
                self.clear_cache()
                self.logger.info("Cache cleared due to configuration changes")

            # Update boundary enforcement if needed
            if new_config.enforce_boundaries != old_config.get("enforce_boundaries"):
                if new_config.enforce_boundaries and not self.boundary_enforcer:
                    self.boundary_enforcer = BoundaryEnforcer(self.grid_size)
                elif not new_config.enforce_boundaries:
                    self.boundary_enforcer = None

            self.logger.info(
                "Configuration updated",
                extra={
                    "old_config": old_config,
                    "new_config": dataclasses.asdict(new_config),
                    "cache_cleared": cache_settings_changed,
                },
            )

        except Exception as e:
            self.logger.error(f"Configuration update failed: {e}", exception=e)
            raise ValidationError(f"Failed to update configuration: {e}") from e


# Standalone utility functions for performance-critical scenarios


def process_action(
    action: ActionType,
    current_position: Coordinates,
    grid_bounds: GridSize,
    enforce_boundaries: bool = True,
    validate_action: bool = True,
) -> Tuple[Coordinates, bool, bool]:
    """
    Standalone action processing function for performance-critical scenarios requiring
    minimal overhead with comprehensive validation, movement calculation, and boundary
    enforcement integration.

    Args:
        action: Action to process (Action enum or integer)
        current_position: Current agent position
        grid_bounds: Grid size for boundary validation
        enforce_boundaries: Whether to enforce grid boundaries
        validate_action: Whether to validate action parameter

    Returns:
        Tuple of (final_position, action_valid, boundary_hit) with comprehensive movement analysis

    Raises:
        ValidationError: If critical parameters are invalid
    """
    try:
        # Validate action if requested
        action_valid = True
        if validate_action:
            action_valid = validate_action_bounds(action, strict_validation=False)

        # Calculate movement delta
        movement_delta = calculate_movement_delta(action, validate_action=False)

        # Calculate proposed destination
        proposed_position = current_position.move(movement_delta)

        boundary_hit = False
        final_position = proposed_position

        # Apply boundary enforcement if enabled
        if enforce_boundaries:
            # Simple boundary clamping
            clamped_x = max(0, min(grid_bounds.width - 1, proposed_position.x))
            clamped_y = max(0, min(grid_bounds.height - 1, proposed_position.y))

            final_position = Coordinates(clamped_x, clamped_y)
            boundary_hit = final_position != proposed_position

        return final_position, action_valid, boundary_hit

    except Exception as e:
        # Return safe defaults on error
        return current_position, False, True


def validate_action_bounds(action: ActionType, strict_validation: bool = False) -> bool:
    """
    Fast action bounds validation for discrete action space compliance ensuring action
    is within cardinal direction range with performance optimization.

    Args:
        action: Action to validate
        strict_validation: Whether to apply strict type checking

    Returns:
        True if action is within bounds [0, 3] for cardinal directions, False otherwise
    """
    try:
        # Handle Action enum
        if isinstance(action, Action):
            action_int = int(action)
        elif isinstance(action, int):
            action_int = action
        else:
            if strict_validation:
                return False
            # Try to convert to int
            try:
                action_int = int(action)
            except (TypeError, ValueError):
                return False

        # Check bounds
        return 0 <= action_int < ACTION_SPACE_SIZE

    except Exception:
        return False


def calculate_movement_delta(
    action: ActionType, validate_action: bool = False
) -> MovementVector:
    """
    Calculate coordinate delta vector from action for movement calculations with
    mathematical precision and error handling.

    Args:
        action: Action for movement calculation
        validate_action: Whether to validate action parameter

    Returns:
        Movement delta tuple (dx, dy) for coordinate arithmetic and position updates

    Raises:
        ValidationError: If action is invalid and validation is enabled
    """
    try:
        # Validate action if requested
        if validate_action and not validate_action_bounds(action):
            raise ValidationError(f"Invalid action for movement calculation: {action}")

        # Convert Action enum to integer
        if isinstance(action, Action):
            action_int = int(action)
        else:
            action_int = int(action)

        # Lookup movement vector from MOVEMENT_VECTORS
        if action_int in MOVEMENT_VECTORS:
            return MOVEMENT_VECTORS[action_int]
        else:
            # Fallback for out-of-bounds actions
            return (0, 0)

    except Exception as e:
        if validate_action:
            raise ValidationError(f"Movement delta calculation failed: {e}") from e
        return (0, 0)


def is_valid_action_for_position(
    action: ActionType, current_position: Coordinates, grid_bounds: GridSize
) -> bool:
    """
    Check if action would result in valid movement from current position considering
    boundary constraints and grid limitations.

    Args:
        action: Action to check for validity
        current_position: Current agent position
        grid_bounds: Grid boundaries for validation

    Returns:
        True if action would result in valid movement within grid bounds, False otherwise
    """
    try:
        # Validate action bounds first
        if not validate_action_bounds(action):
            return False

        # Calculate movement delta
        movement_delta = calculate_movement_delta(action, validate_action=False)

        # Calculate proposed position
        new_x = current_position.x + movement_delta[0]
        new_y = current_position.y + movement_delta[1]

        # Check if proposed position is within grid bounds
        return 0 <= new_x < grid_bounds.width and 0 <= new_y < grid_bounds.height

    except Exception:
        # Return False for any errors to be safe
        return False
