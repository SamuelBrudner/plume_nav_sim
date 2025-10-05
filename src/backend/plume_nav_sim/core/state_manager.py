"""
Central state management module for plume_nav_sim coordinating agent state, episode lifecycle,
and cross-component state synchronization with comprehensive validation, performance monitoring,
and reproducibility support for Gymnasium-compatible reinforcement learning environment implementation.

This module provides the core StateManager class and related utilities for managing all aspects
of environment state including agent position tracking, episode lifecycle coordination, component
integration, and state validation with performance optimization targeting <1ms step latency.
"""

import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from ..utils.exceptions import ComponentError, StateError, ValidationError
from ..utils.logging import get_component_logger, monitor_performance

# Internal imports - utilities and infrastructure
from ..utils.seeding import SeedManager
from ..utils.validation import (
    ValidationContext,
    ValidationResult,
    validate_action_input,
    validate_coordinates,
)

# Internal imports - component coordination
from .boundary_enforcer import BoundaryEnforcer

# Internal imports - system constants
from .constants import (
    DEFAULT_GOAL_RADIUS,
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_STEPS,
    DEFAULT_PLUME_SIGMA,
    DEFAULT_SOURCE_LOCATION,
    MOVEMENT_VECTORS,
    PERFORMANCE_TARGET_STEP_LATENCY_MS,
)

# Internal imports - core types and data structures (single source of truth)
from .enums import Action
from .geometry import Coordinates, GridSize
from .models import PlumeModel
from .snapshots import StateSnapshot
from .state import AgentState, EpisodeState
from .types import (
    PerformanceMetrics,
    create_agent_state,
    create_coordinates,
    create_episode_state,
)

# Module-level constants for state management configuration
STATE_MANAGER_VERSION = "1.0.0"
DEFAULT_PERFORMANCE_MONITORING = True
DEFAULT_STATE_VALIDATION = True
DEFAULT_HISTORY_TRACKING = False
STATE_SNAPSHOT_CACHE_SIZE = 50
COMPONENT_COORDINATION_TIMEOUT = 1.0


@dataclass
class StateManagerConfig:
    """
    Configuration data class for state manager containing all parameters needed for state management
    including grid dimensions, episode settings, component coordination options, and performance
    monitoring configuration with comprehensive validation and serialization support.
    """

    grid_size: GridSize
    source_location: Coordinates
    max_steps: int
    goal_radius: float
    enable_performance_monitoring: bool = field(default=DEFAULT_PERFORMANCE_MONITORING)
    enable_state_validation: bool = field(default=DEFAULT_STATE_VALIDATION)
    enable_history_tracking: bool = field(default=DEFAULT_HISTORY_TRACKING)
    enable_boundary_enforcement: bool = field(default=True)
    snapshot_cache_size: int = field(default=STATE_SNAPSHOT_CACHE_SIZE)
    component_coordination_timeout: float = field(
        default=COMPONENT_COORDINATION_TIMEOUT
    )
    custom_parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize state manager configuration with validation, default values, and parameter consistency checking."""
        # Validate grid_size has positive width and height dimensions within system performance limits
        if not isinstance(self.grid_size, GridSize):
            raise ValidationError(
                message="grid_size must be a GridSize instance",
                parameter_name="grid_size",
                parameter_value=type(self.grid_size).__name__,
                expected_format="GridSize dataclass",
            )

        if self.grid_size.width <= 0 or self.grid_size.height <= 0:
            raise ValidationError(
                message="Grid dimensions must be positive integers",
                parameter_name="grid_size",
                parameter_value=f"({self.grid_size.width}, {self.grid_size.height})",
                expected_format="positive integers",
            )

        # Validate source_location coordinates are within grid bounds using coordinate validation utilities
        if not isinstance(self.source_location, Coordinates):
            raise ValidationError(
                message="source_location must be a Coordinates instance",
                parameter_name="source_location",
                parameter_value=type(self.source_location).__name__,
                expected_format="Coordinates dataclass",
            )

        if not self.grid_size.contains_coordinates(self.source_location):
            raise ValidationError(
                message="Source location must be within grid bounds",
                parameter_name="source_location",
                parameter_value=f"({self.source_location.x}, {self.source_location.y})",
                expected_format=f"coordinates within (0,0) to ({self.grid_size.width-1},{self.grid_size.height-1})",
            )

        # Check max_steps is positive integer within reasonable episode duration and performance constraints
        if not isinstance(self.max_steps, int) or self.max_steps <= 0:
            raise ValidationError(
                message="max_steps must be a positive integer",
                parameter_name="max_steps",
                parameter_value=str(self.max_steps),
                expected_format="positive integer",
            )

        # Validate goal_radius is non-negative float enabling reachable goal conditions within grid bounds
        if not isinstance(self.goal_radius, (int, float)) or self.goal_radius < 0:
            raise ValidationError(
                message="goal_radius must be a non-negative number",
                parameter_name="goal_radius",
                parameter_value=str(self.goal_radius),
                expected_format="non-negative float",
            )

        # Validate cache size is reasonable
        if self.snapshot_cache_size < 0 or self.snapshot_cache_size > 1000:
            raise ValidationError(
                message="snapshot_cache_size must be between 0 and 1000",
                parameter_name="snapshot_cache_size",
                parameter_value=str(self.snapshot_cache_size),
                expected_format="integer between 0 and 1000",
            )

    def validate(
        self,
        strict_mode: bool = False,
        validation_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Comprehensive validation of all configuration parameters with cross-parameter consistency checking,
        mathematical feasibility analysis, and component compatibility verification.
        """
        try:
            # Validate grid_size dimensions and estimate memory requirements using system resource constraints
            total_cells = self.grid_size.width * self.grid_size.height
            estimated_memory_mb = (total_cells * 8) / (
                1024 * 1024
            )  # 8 bytes per float64 cell

            if estimated_memory_mb > 500:  # 500MB limit for grid storage
                raise ValidationError(
                    message=f"Grid size too large, estimated memory usage: {estimated_memory_mb:.1f}MB",
                    parameter_name="grid_size",
                    parameter_value=f"{self.grid_size.width}x{self.grid_size.height}",
                    expected_format="dimensions resulting in <500MB memory usage",
                )

            # Cross-validate all parameters for component integration requirements and dependency satisfaction
            max_distance = self.source_location.distance_to(
                create_coordinates(
                    (self.grid_size.width - 1, self.grid_size.height - 1)
                )
            )

            if self.goal_radius > max_distance:
                raise ValidationError(
                    message="Goal radius larger than maximum possible distance in grid",
                    parameter_name="goal_radius",
                    parameter_value=str(self.goal_radius),
                    expected_format=f"<={max_distance:.2f} (maximum grid distance)",
                )

            # Apply strict validation rules if strict_mode enabled
            if strict_mode:
                if self.max_steps > 10000:
                    raise ValidationError(
                        message="Maximum steps too high for strict validation",
                        parameter_name="max_steps",
                        parameter_value=str(self.max_steps),
                        expected_format="<=10000 in strict mode",
                    )

            return True

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                message=f"Configuration validation failed: {str(e)}",
                parameter_name="configuration",
                parameter_value="various",
                expected_format="valid configuration parameters",
            ) from e

    def estimate_resources(
        self,
        include_component_overhead: bool = True,
        include_monitoring_costs: bool = True,
    ) -> Dict[str, Any]:
        """
        Estimates computational and memory resources required for state management operations
        including component coordination overhead and performance monitoring costs.
        """
        # Calculate base state management memory requirements
        base_memory_mb = 10  # Base StateManager overhead

        # Grid-related memory
        grid_cells = self.grid_size.width * self.grid_size.height
        grid_memory_mb = (grid_cells * 4) / (1024 * 1024)  # 4 bytes per float32

        # History tracking memory if enabled
        history_memory_mb = 0
        if self.enable_history_tracking:
            avg_episode_length = min(self.max_steps, 1000)  # Estimate
            history_memory_mb = (avg_episode_length * 100) / (
                1024 * 1024
            )  # 100 bytes per step

        # Snapshot cache memory
        snapshot_memory_mb = (
            self.snapshot_cache_size * 1
        ) / 1024  # 1KB per snapshot estimate

        # Component coordination overhead
        component_overhead_mb = 0
        if include_component_overhead:
            component_overhead_mb = 5  # Estimated overhead for boundary enforcer, etc.

        # Performance monitoring overhead
        monitoring_overhead_mb = 0
        if include_monitoring_costs and self.enable_performance_monitoring:
            monitoring_overhead_mb = 2  # Performance metrics storage

        total_memory_mb = (
            base_memory_mb
            + grid_memory_mb
            + history_memory_mb
            + snapshot_memory_mb
            + component_overhead_mb
            + monitoring_overhead_mb
        )

        return {
            "memory_usage_mb": {
                "base_state_manager": base_memory_mb,
                "grid_storage": grid_memory_mb,
                "history_tracking": history_memory_mb,
                "snapshot_cache": snapshot_memory_mb,
                "component_overhead": component_overhead_mb,
                "monitoring_overhead": monitoring_overhead_mb,
                "total_estimated": total_memory_mb,
            },
            "computational_estimates": {
                "step_latency_target_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS,
                "reset_latency_estimate_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS * 10,
                "validation_overhead_percent": 5 if self.enable_state_validation else 0,
            },
            "recommendations": {
                "suitable_for_real_time": total_memory_mb < 100,
                "optimization_suggestions": self._generate_optimization_suggestions(
                    total_memory_mb
                ),
            },
        }

    def _generate_optimization_suggestions(self, total_memory_mb: float) -> List[str]:
        """Generate optimization suggestions based on resource estimates."""
        suggestions = []

        if total_memory_mb > 100:
            suggestions.append("Consider reducing grid size for better performance")

        if self.enable_history_tracking and self.max_steps > 5000:
            suggestions.append(
                "History tracking with long episodes may impact memory usage"
            )

        if self.snapshot_cache_size > 20:
            suggestions.append(
                "Large snapshot cache may not be necessary for most applications"
            )

        return suggestions

    def clone(
        self,
        overrides: Optional[Dict[str, Any]] = None,
        preserve_custom_parameters: bool = True,
    ) -> "StateManagerConfig":
        """
        Creates deep copy of state manager configuration with optional parameter overrides
        for testing, experimentation, and configuration variants.
        """
        # Create deep copy of current configuration
        cloned_config = copy.deepcopy(self)

        # Apply parameter overrides if provided
        if overrides:
            for key, value in overrides.items():
                if hasattr(cloned_config, key):
                    setattr(cloned_config, key, value)
                else:
                    raise ValidationError(
                        message=f"Invalid override parameter: {key}",
                        parameter_name="overrides",
                        parameter_value=key,
                        expected_format="valid StateManagerConfig attribute",
                    )

        # Handle custom parameters
        if not preserve_custom_parameters:
            cloned_config.custom_parameters = {}

        # Validate cloned configuration
        cloned_config.validate()

        return cloned_config

    def to_dict(
        self,
        include_resource_estimates: bool = False,
        include_validation_results: bool = False,
    ) -> Dict[str, Any]:
        """
        Converts state manager configuration to dictionary format for serialization,
        storage, and external system integration.
        """
        config_dict = {
            "grid_size": (self.grid_size.width, self.grid_size.height),
            "source_location": (self.source_location.x, self.source_location.y),
            "max_steps": self.max_steps,
            "goal_radius": self.goal_radius,
            "enable_performance_monitoring": self.enable_performance_monitoring,
            "enable_state_validation": self.enable_state_validation,
            "enable_history_tracking": self.enable_history_tracking,
            "enable_boundary_enforcement": self.enable_boundary_enforcement,
            "snapshot_cache_size": self.snapshot_cache_size,
            "component_coordination_timeout": self.component_coordination_timeout,
            "custom_parameters": copy.deepcopy(self.custom_parameters),
        }

        if include_resource_estimates:
            config_dict["resource_estimates"] = self.estimate_resources()

        if include_validation_results:
            try:
                self.validate(strict_mode=True)
                config_dict["validation_results"] = {"is_valid": True, "errors": []}
            except ValidationError as e:
                config_dict["validation_results"] = {
                    "is_valid": False,
                    "errors": [str(e)],
                }

        return config_dict


@dataclass
class StateValidationResult:
    """
    Data class containing comprehensive state validation results including consistency analysis,
    error detection, component synchronization status, and recovery recommendations for state
    management debugging and quality assurance.
    """

    is_valid: bool
    validation_details: Dict[str, Any]
    validation_timestamp: float
    validation_errors: List[Dict[str, Any]] = field(default_factory=list)
    validation_warnings: List[Dict[str, Any]] = field(default_factory=list)
    component_status: Dict[str, Any] = field(default_factory=dict)
    consistency_analysis: Dict[str, Any] = field(default_factory=dict)
    recovery_recommendations: List[str] = field(default_factory=list)

    def add_error(
        self,
        error_message: str,
        error_category: str,
        recovery_suggestion: Optional[str] = None,
    ) -> None:
        """
        Adds validation error with severity classification and recovery suggestions
        for comprehensive error tracking and resolution guidance.
        """
        if not error_message.strip():
            raise ValidationError(
                message="Error message cannot be empty",
                parameter_name="error_message",
                parameter_value="empty string",
                expected_format="non-empty descriptive string",
            )

        error_entry = {
            "message": error_message,
            "category": error_category,
            "timestamp": time.time(),
            "severity": "critical",
        }

        if recovery_suggestion:
            error_entry["recovery_suggestion"] = recovery_suggestion

        self.validation_errors.append(error_entry)
        self.is_valid = False

        # Update validation details with error context
        if "error_summary" not in self.validation_details:
            self.validation_details["error_summary"] = {
                "total_errors": 0,
                "categories": {},
            }

        self.validation_details["error_summary"]["total_errors"] += 1
        category_count = self.validation_details["error_summary"]["categories"].get(
            error_category, 0
        )
        self.validation_details["error_summary"]["categories"][error_category] = (
            category_count + 1
        )

    def add_warning(
        self,
        warning_message: str,
        warning_category: str,
        optimization_suggestion: Optional[str] = None,
    ) -> None:
        """
        Adds validation warning for non-critical issues with optimization suggestions
        and performance impact analysis.
        """
        warning_entry = {
            "message": warning_message,
            "category": warning_category,
            "timestamp": time.time(),
            "severity": "warning",
        }

        if optimization_suggestion:
            warning_entry["optimization_suggestion"] = optimization_suggestion

        self.validation_warnings.append(warning_entry)

        # Update validation details with warning context
        if "warning_summary" not in self.validation_details:
            self.validation_details["warning_summary"] = {
                "total_warnings": 0,
                "categories": {},
            }

        self.validation_details["warning_summary"]["total_warnings"] += 1
        category_count = self.validation_details["warning_summary"]["categories"].get(
            warning_category, 0
        )
        self.validation_details["warning_summary"]["categories"][warning_category] = (
            category_count + 1
        )

    def get_summary(self, include_detailed_analysis: bool = False) -> Dict[str, Any]:
        """
        Generates comprehensive validation summary including error counts, warning analysis,
        component status, and actionable recommendations for quality assessment.
        """
        # Count errors and warnings by category
        error_categories = {}
        for error in self.validation_errors:
            category = error.get("category", "unknown")
            error_categories[category] = error_categories.get(category, 0) + 1

        warning_categories = {}
        for warning in self.validation_warnings:
            category = warning.get("category", "unknown")
            warning_categories[category] = warning_categories.get(category, 0) + 1

        summary = {
            "overall_status": "VALID" if self.is_valid else "INVALID",
            "validation_timestamp": self.validation_timestamp,
            "error_summary": {
                "total_errors": len(self.validation_errors),
                "error_categories": error_categories,
                "critical_errors": len(
                    [
                        e
                        for e in self.validation_errors
                        if e.get("severity") == "critical"
                    ]
                ),
            },
            "warning_summary": {
                "total_warnings": len(self.validation_warnings),
                "warning_categories": warning_categories,
            },
            "component_status_summary": {
                "total_components": len(self.component_status),
                "healthy_components": len(
                    [
                        s
                        for s in self.component_status.values()
                        if s.get("status") == "healthy"
                    ]
                ),
                "components_with_issues": len(
                    [
                        s
                        for s in self.component_status.values()
                        if s.get("status") != "healthy"
                    ]
                ),
            },
            "recommendations": self.recovery_recommendations,
        }

        if include_detailed_analysis:
            summary["detailed_findings"] = {
                "validation_errors": self.validation_errors,
                "validation_warnings": self.validation_warnings,
                "component_details": self.component_status,
                "consistency_analysis": self.consistency_analysis,
                "validation_details": self.validation_details,
            }

        return summary


class StateSynchronizer:
    """
    Utility class for coordinating state synchronization between state manager and other core
    components ensuring consistent data flow, dependency coordination, and component integration
    with performance optimization and error handling.
    """

    def __init__(
        self,
        state_manager: "StateManager",
        boundary_enforcer: Optional[BoundaryEnforcer] = None,
    ):
        """Initialize state synchronizer with component references and coordination mechanisms."""
        # Store component references for centralized coordination
        self.state_manager = state_manager
        self.boundary_enforcer = boundary_enforcer

        # Initialize component registry for dynamic component registration
        self.component_registry: Dict[str, Any] = {}

        # Initialize synchronization locks for thread-safe component coordination
        self.synchronization_locks: Dict[str, threading.Lock] = {}

        # Create component logger for synchronization operation tracking
        self.logger = get_component_logger("StateSynchronizer")

        # Initialize synchronization metrics for performance monitoring
        self.synchronization_metrics: Dict[str, Any] = {
            "total_synchronizations": 0,
            "failed_synchronizations": 0,
            "average_sync_time_ms": 0.0,
            "last_sync_timestamp": 0.0,
        }

    def register_component(
        self, component_name: str, component_instance: Any, sync_methods: List[str]
    ) -> None:
        """
        Registers component for state synchronization with dependency tracking and
        coordination setup for consistent cross-component state management.
        """
        # Validate component_name is unique and follows naming conventions
        if not isinstance(component_name, str) or not component_name.strip():
            raise ValidationError(
                message="Component name must be a non-empty string",
                parameter_name="component_name",
                parameter_value=str(component_name),
                expected_format="non-empty string",
            )

        if component_name in self.component_registry:
            raise ComponentError(
                message=f"Component '{component_name}' is already registered",
                component_name="StateSynchronizer",
                operation_name="register_component",
            )

        # Validate component_instance has required interface methods
        for method_name in sync_methods:
            if not hasattr(component_instance, method_name):
                raise ValidationError(
                    message=f"Component missing required method: {method_name}",
                    parameter_name="component_instance",
                    parameter_value=f"missing {method_name}",
                    expected_format=f"object with methods: {sync_methods}",
                )

        # Store component in registry with sync methods
        self.component_registry[component_name] = {
            "instance": component_instance,
            "sync_methods": sync_methods,
            "registration_timestamp": time.time(),
            "last_sync_timestamp": 0.0,
            "sync_count": 0,
            "sync_errors": 0,
        }

        # Create synchronization lock for thread safety
        self.synchronization_locks[component_name] = threading.Lock()

        # Log component registration
        self.logger.info(
            f"Registered component '{component_name}' with sync methods: {sync_methods}"
        )

    def synchronize_state(
        self,
        strict_synchronization: bool = False,
        component_subset: Optional[List[str]] = None,
    ) -> StateValidationResult:
        """
        Performs comprehensive state synchronization across all registered components
        with consistency validation and performance monitoring.
        """
        sync_start_time = time.time()
        validation_result = StateValidationResult(
            is_valid=True, validation_details={}, validation_timestamp=sync_start_time
        )

        try:
            # Determine synchronization scope
            components_to_sync = component_subset or list(
                self.component_registry.keys()
            )

            # Track synchronization progress
            sync_progress = {
                "total_components": len(components_to_sync),
                "synchronized_components": 0,
                "failed_components": [],
            }

            # Synchronize each component
            for component_name in components_to_sync:
                if component_name not in self.component_registry:
                    validation_result.add_warning(
                        f"Component '{component_name}' not found in registry",
                        "missing_component",
                        "Ensure component is properly registered",
                    )
                    continue

                component_info = self.component_registry[component_name]

                # Acquire synchronization lock for thread safety
                with self.synchronization_locks[component_name]:
                    try:
                        # Perform component synchronization
                        component_instance = component_info["instance"]
                        sync_methods = component_info["sync_methods"]

                        # Execute synchronization methods
                        for method_name in sync_methods:
                            if hasattr(component_instance, method_name):
                                method = getattr(component_instance, method_name)
                                method(self.state_manager)

                        # Update component sync statistics
                        component_info["last_sync_timestamp"] = time.time()
                        component_info["sync_count"] += 1
                        sync_progress["synchronized_components"] += 1

                        # Record component status in validation result
                        validation_result.component_status[component_name] = {
                            "status": "synchronized",
                            "last_sync": component_info["last_sync_timestamp"],
                            "sync_count": component_info["sync_count"],
                        }

                    except Exception as e:
                        # Handle component synchronization failure
                        component_info["sync_errors"] += 1
                        sync_progress["failed_components"].append(component_name)

                        validation_result.add_error(
                            f"Component '{component_name}' synchronization failed: {str(e)}",
                            "sync_failure",
                            f"Check component '{component_name}' state and methods",
                        )

                        validation_result.component_status[component_name] = {
                            "status": "sync_failed",
                            "error": str(e),
                            "sync_errors": component_info["sync_errors"],
                        }

                        self.logger.error(
                            f"Component '{component_name}' sync failed: {e}"
                        )

            # Update global synchronization metrics
            sync_duration_ms = (time.time() - sync_start_time) * 1000
            self.synchronization_metrics["total_synchronizations"] += 1
            self.synchronization_metrics["last_sync_timestamp"] = time.time()

            if len(sync_progress["failed_components"]) > 0:
                self.synchronization_metrics["failed_synchronizations"] += 1
                validation_result.is_valid = False

            # Update average sync time (rolling average)
            current_avg = self.synchronization_metrics["average_sync_time_ms"]
            total_syncs = self.synchronization_metrics["total_synchronizations"]
            new_avg = (
                (current_avg * (total_syncs - 1)) + sync_duration_ms
            ) / total_syncs
            self.synchronization_metrics["average_sync_time_ms"] = new_avg

            # Store synchronization details
            validation_result.validation_details["synchronization"] = {
                "duration_ms": sync_duration_ms,
                "components_processed": sync_progress["total_components"],
                "successful_syncs": sync_progress["synchronized_components"],
                "failed_syncs": len(sync_progress["failed_components"]),
                "strict_mode": strict_synchronization,
            }

            # Generate recovery recommendations if needed
            if not validation_result.is_valid:
                validation_result.recovery_recommendations.extend(
                    [
                        "Check failed component error logs for specific issues",
                        "Verify component interfaces match expected synchronization methods",
                        "Consider re-registering failed components",
                    ]
                )

            # Log synchronization completion
            success_rate = (
                sync_progress["synchronized_components"]
                / sync_progress["total_components"]
                * 100
            )
            self.logger.info(
                f"State synchronization completed: {success_rate:.1f}% success rate "
                f"({sync_duration_ms:.2f}ms)"
            )

            return validation_result

        except Exception as e:
            # Handle unexpected synchronization errors
            validation_result.is_valid = False
            validation_result.add_error(
                f"State synchronization failed: {str(e)}",
                "synchronization_error",
                "Review synchronization process and component states",
            )

            self.logger.error(f"State synchronization failed: {e}")
            return validation_result


class StateManager:
    """
    Central state management class coordinating agent state, episode lifecycle, component
    synchronization, and performance monitoring for Gymnasium-compatible reinforcement learning
    environment with comprehensive validation, reproducibility support, and cross-component integration.
    """

    def __init__(
        self,
        config: StateManagerConfig,
        seed_manager: Optional[SeedManager] = None,
        boundary_enforcer: Optional[BoundaryEnforcer] = None,
    ):
        """
        Initialize state manager with configuration validation, component coordination,
        performance monitoring setup, and comprehensive state management infrastructure.
        """
        # Validate configuration using comprehensive parameter checking
        if not isinstance(config, StateManagerConfig):
            raise ValidationError(
                message="config must be a StateManagerConfig instance",
                parameter_name="config",
                parameter_value=type(config).__name__,
                expected_format="StateManagerConfig dataclass",
            )

        config.validate()
        self.config = config

        # Initialize or store provided components
        self.seed_manager = seed_manager if seed_manager is not None else SeedManager()

        if boundary_enforcer is not None:
            self.boundary_enforcer = boundary_enforcer
        else:
            # Create boundary enforcer with current grid configuration
            self.boundary_enforcer = BoundaryEnforcer(grid_size=config.grid_size)

        # Initialize state tracking variables
        self.current_agent_state: Optional[AgentState] = None
        self.current_episode_state: Optional[EpisodeState] = None
        self.episode_active: bool = False
        self.episode_count: int = 0

        # Initialize state history and caching
        self.state_history: List[Dict[str, Any]] = []
        self.state_snapshots: Dict[str, StateSnapshot] = {}

        # Initialize performance monitoring
        self.performance_metrics = PerformanceMetrics()

        # Initialize component synchronization
        self.synchronizer = StateSynchronizer(self, self.boundary_enforcer)

        # Create component logger for state management operations
        self.logger = get_component_logger("StateManager")

        # Initialize thread safety
        self.state_lock = threading.RLock()

        # Initialize component cache for performance optimization
        self.component_cache: Dict[str, Any] = {}

        # Create a minimal plume model for snapshots using current config
        try:
            self.plume_model = PlumeModel(
                source_location=self.config.source_location,
                sigma=DEFAULT_PLUME_SIGMA,
                grid_compatibility=self.config.grid_size,
            )
        except Exception:
            # Fallback: do not block initialization if plume model cannot be created
            self.plume_model = PlumeModel(
                source_location=self.config.source_location,
                sigma=max(DEFAULT_PLUME_SIGMA, 1.0),
                grid_compatibility=self.config.grid_size,
            )

        # Log successful initialization
        self.logger.info(
            f"StateManager initialized with grid size {config.grid_size.width}x{config.grid_size.height}"
        )

    @monitor_performance("episode_reset", PERFORMANCE_TARGET_STEP_LATENCY_MS * 10, True)
    def reset_episode(
        self,
        seed: Optional[int] = None,
        episode_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Resets state manager for new episode with agent placement, episode initialization,
        component coordination, and performance monitoring for Gymnasium reset() method support.
        """
        with self.state_lock:
            try:
                reset_start_time = time.time()

                # Validate and apply seed for reproducible episode generation
                if seed is not None:
                    self.seed_manager.seed(seed)
                    self.logger.debug(f"Episode reset with seed: {seed}")
                else:
                    self.logger.debug("Episode reset without seed (random)")

                # Generate unique episode ID for tracking
                episode_id = f"episode_{int(time.time() * 1000)}_{self.episode_count}"

                # Clear previous episode state
                self.current_agent_state = None
                self.current_episode_state = None
                if self.config.enable_history_tracking:
                    self.state_history.clear()
                self.component_cache.clear()

                # Generate random agent start position excluding source location
                max_attempts = 100
                for attempt in range(max_attempts):
                    start_position = self.seed_manager.generate_random_position(
                        self.config.grid_size,
                        exclude_position=self.config.source_location,
                    )

                    # Validate agent start position using boundary enforcer
                    if self.boundary_enforcer.validate_position(
                        start_position, raise_on_invalid=False
                    ):
                        break
                else:
                    raise StateError(
                        message=f"Could not generate valid start position after {max_attempts} attempts",
                        current_state="episode_reset",
                        expected_state="valid_start_position_generated",
                    )

                # Create new AgentState using factory with initial position
                self.current_agent_state = create_agent_state(
                    position=start_position, step_count=0, total_reward=0.0
                )

                # Create new EpisodeState using factory
                self.current_episode_state = create_episode_state(
                    agent_state=self.current_agent_state,
                    terminated=False,
                    truncated=False,
                )

                # Initialize episode tracking
                self.episode_active = True
                self.episode_count += 1

                # Create initial state snapshot for reproducibility
                if self.config.enable_history_tracking:
                    initial_snapshot = self.create_state_snapshot(
                        snapshot_name=f"{episode_id}_initial", validate_consistency=True
                    )

                    # Add to history
                    self.state_history.append(
                        {
                            "step_number": 0,
                            "snapshot": initial_snapshot,
                            "timestamp": time.time(),
                            "event_type": "episode_start",
                        }
                    )

                # Record episode reset timing in performance metrics
                reset_duration_ms = (time.time() - reset_start_time) * 1000
                self.performance_metrics.record_timing(
                    "episode_reset", reset_duration_ms
                )

                # Generate episode initialization info
                episode_init_info = {
                    "episode_id": episode_id,
                    "agent_start_position": (start_position.x, start_position.y),
                    "source_location": (
                        self.config.source_location.x,
                        self.config.source_location.y,
                    ),
                    "max_steps": self.config.max_steps,
                    "goal_radius": self.config.goal_radius,
                    "seed": seed,
                    "reset_duration_ms": reset_duration_ms,
                    "episode_count": self.episode_count,
                }

                self.logger.info(
                    f"Episode {self.episode_count} reset completed in {reset_duration_ms:.2f}ms"
                )
                return episode_init_info

            except (ValidationError, StateError, ComponentError):
                # Re-raise specific exceptions with context
                raise
            except Exception as e:
                # Wrap unexpected errors in StateError
                self.logger.error(f"Episode reset failed: {e}")
                raise StateError(
                    message=f"Episode reset failed: {str(e)}",
                    current_state="episode_reset",
                    expected_state="reset_complete",
                    component_name="StateManager",
                ) from e

    @monitor_performance("process_step", PERFORMANCE_TARGET_STEP_LATENCY_MS, False)
    def process_step(
        self, action: Action, action_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Processes environment step with action validation, agent state update, component coordination,
        and performance monitoring for Gymnasium step() method implementation.
        """
        with self.state_lock:
            try:
                step_start_time = time.time()

                # Validate episode is active and states are initialized
                if not self.episode_active:
                    raise StateError(
                        message="Cannot process step: episode is not active",
                        current_state="inactive",
                        expected_state="active",
                        component_name="StateManager",
                    )

                if (
                    self.current_agent_state is None
                    or self.current_episode_state is None
                ):
                    raise StateError(
                        message="Cannot process step: agent or episode state not initialized",
                        current_state="uninitialized",
                        expected_state="initialized",
                        component_name="StateManager",
                    )

                # Validate action parameter using action space compliance
                validate_action_input(action)

                # Calculate new agent position using movement vectors
                if action not in MOVEMENT_VECTORS:
                    raise ValidationError(
                        message=f"Invalid action: {action}",
                        parameter_name="action",
                        parameter_value=str(action),
                        expected_format="valid Action enum value",
                    )

                movement_delta = MOVEMENT_VECTORS[action]
                current_pos = self.current_agent_state.position
                new_position = create_coordinates(
                    (
                        current_pos.x + movement_delta[0],
                        current_pos.y + movement_delta[1],
                    )
                )

                # Validate new position using boundary enforcer
                if self.boundary_enforcer.validate_position(
                    new_position, raise_on_invalid=False
                ):
                    # Update agent position with successful movement
                    self.current_agent_state.update_position(new_position)
                else:
                    # Log attempted boundary violation but continue with current position
                    self.logger.debug(f"Action {action} blocked by boundary enforcer")

                # Increment step count for episode tracking
                self.current_agent_state.increment_step()

                # Create state snapshot if history tracking enabled
                if self.config.enable_history_tracking:
                    step_snapshot = self.create_state_snapshot(
                        snapshot_name=f"step_{self.current_agent_state.step_count}",
                        validate_consistency=False,
                    )

                    self.state_history.append(
                        {
                            "step_number": self.current_agent_state.step_count,
                            "action": action,
                            "position": (
                                self.current_agent_state.position.x,
                                self.current_agent_state.position.y,
                            ),
                            "snapshot": step_snapshot,
                            "timestamp": time.time(),
                            "event_type": "step_processed",
                        }
                    )

                # Record step processing timing
                step_duration_ms = (time.time() - step_start_time) * 1000
                self.performance_metrics.record_timing("process_step", step_duration_ms)

                # Generate step processing result
                step_result = {
                    "step_number": self.current_agent_state.step_count,
                    "agent_position": (
                        self.current_agent_state.position.x,
                        self.current_agent_state.position.y,
                    ),
                    "action_processed": action,
                    "boundary_valid": self.boundary_enforcer.validate_position(
                        self.current_agent_state.position, raise_on_invalid=False
                    ),
                    "step_duration_ms": step_duration_ms,
                    "episode_active": self.episode_active,
                }

                # Add context information if provided
                if action_context:
                    step_result["context"] = action_context

                return step_result

            except (ValidationError, StateError, ComponentError):
                # Re-raise specific exceptions
                raise
            except Exception as e:
                # Wrap unexpected errors
                self.logger.error(f"Step processing failed: {e}")
                raise StateError(
                    message=f"Step processing failed: {str(e)}",
                    current_state="step_processing",
                    expected_state="step_complete",
                    component_name="StateManager",
                ) from e

    def check_episode_termination(
        self,
        source_location: Coordinates,
        goal_radius: float,
        termination_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Checks episode termination and truncation conditions with goal detection, step limit validation,
        and comprehensive termination analysis for Gymnasium episode completion.
        """
        if (
            not self.episode_active
            or self.current_agent_state is None
            or self.current_episode_state is None
        ):
            return {
                "terminated": False,
                "truncated": False,
                "reason": "episode_not_active",
                "analysis": {
                    "error": "Episode not in valid state for termination checking"
                },
            }

        # Calculate distance from agent to source
        current_position = self.current_agent_state.position
        distance_to_source = current_position.distance_to(source_location)

        # Check goal achievement
        goal_achieved = distance_to_source <= goal_radius

        # Check step limit truncation
        step_limit_reached = (
            self.current_agent_state.step_count >= self.config.max_steps
        )

        # Determine termination status
        terminated = goal_achieved
        truncated = step_limit_reached and not goal_achieved

        # Update episode state if terminated or truncated
        if terminated or truncated:
            self.current_episode_state.set_termination(
                terminated=terminated,
                truncated=truncated,
                reason="goal_achieved" if terminated else "step_limit",
            )
            self.episode_active = False

        # Create comprehensive termination analysis
        termination_analysis = {
            "terminated": terminated,
            "truncated": truncated,
            "goal_achieved": goal_achieved,
            "distance_to_source": distance_to_source,
            "goal_radius": goal_radius,
            "step_count": self.current_agent_state.step_count,
            "max_steps": self.config.max_steps,
            "episode_duration": (
                self.current_episode_state.get_episode_duration()
                if self.current_episode_state
                else 0
            ),
            "termination_reason": None,
        }

        # Determine specific termination reason
        if terminated:
            termination_analysis["termination_reason"] = "goal_reached"
        elif truncated:
            termination_analysis["termination_reason"] = "step_limit_exceeded"
        else:
            termination_analysis["termination_reason"] = "episode_continues"

        # Include context if provided
        if termination_context:
            termination_analysis["context"] = termination_context

        # Log termination events
        if terminated:
            self.logger.info(
                f"Episode terminated: goal achieved at step {self.current_agent_state.step_count}"
            )
        elif truncated:
            self.logger.info(
                f"Episode truncated: step limit {self.config.max_steps} reached"
            )

        return termination_analysis

    def get_current_state(
        self,
        include_performance_data: bool = False,
        include_component_details: bool = False,
    ) -> Dict[str, Any]:
        """
        Returns current environment state including agent position, episode status, component states,
        and optional performance metrics for external access and coordination.
        """
        with self.state_lock:
            if self.current_agent_state is None or self.current_episode_state is None:
                return {
                    "state_available": False,
                    "error": "No active episode state",
                    "episode_active": self.episode_active,
                    "episode_count": self.episode_count,
                }

            # Extract current state information
            current_state = {
                "state_available": True,
                "episode_active": self.episode_active,
                "episode_count": self.episode_count,
                "agent_state": {
                    "position": (
                        self.current_agent_state.position.x,
                        self.current_agent_state.position.y,
                    ),
                    "step_count": self.current_agent_state.step_count,
                    "total_reward": self.current_agent_state.total_reward,
                },
                "episode_state": {
                    "terminated": self.current_episode_state.terminated,
                    "truncated": self.current_episode_state.truncated,
                    "is_done": self.current_episode_state.is_done(),
                    "episode_duration": self.current_episode_state.get_episode_duration(),
                },
                "configuration": {
                    "grid_size": (
                        self.config.grid_size.width,
                        self.config.grid_size.height,
                    ),
                    "source_location": (
                        self.config.source_location.x,
                        self.config.source_location.y,
                    ),
                    "max_steps": self.config.max_steps,
                    "goal_radius": self.config.goal_radius,
                },
                "state_timestamp": time.time(),
            }

            # Add performance data if requested
            if include_performance_data:
                current_state["performance_metrics"] = (
                    self.performance_metrics.get_performance_summary()
                )

            # Add component details if requested
            if include_component_details:
                current_state["component_details"] = {
                    "boundary_enforcer": {
                        "grid_size": (
                            self.boundary_enforcer.grid_size.width,
                            self.boundary_enforcer.grid_size.height,
                        ),
                        "position_valid": self.boundary_enforcer.validate_position(
                            self.current_agent_state.position, raise_on_invalid=False
                        ),
                    },
                    "seed_manager": {
                        "current_seed": getattr(self.seed_manager, "seed_value", None)
                    },
                    "synchronizer": {
                        "registered_components": len(
                            self.synchronizer.component_registry
                        ),
                        "sync_metrics": self.synchronizer.synchronization_metrics,
                    },
                }

            # Add history information if enabled
            if self.config.enable_history_tracking:
                current_state["history"] = {
                    "total_snapshots": len(self.state_history),
                    "cached_snapshots": len(self.state_snapshots),
                }

            return current_state

    def create_state_snapshot(
        self, snapshot_name: Optional[str] = None, validate_consistency: bool = False
    ) -> StateSnapshot:
        """
        Creates immutable state snapshot with timestamp, validation, and caching for debugging,
        reproducibility, and state history management.
        """
        if self.current_agent_state is None or self.current_episode_state is None:
            raise StateError(
                message="Cannot create snapshot: no active episode state",
                current_state="no_active_episode",
                expected_state="active_episode",
                component_name="StateManager",
            )

        # Create deep copies for immutability
        agent_state_copy = copy.deepcopy(self.current_agent_state)
        episode_state_copy = copy.deepcopy(self.current_episode_state)

        # Generate timestamp for snapshot
        snapshot_timestamp = time.time()

        # Create StateSnapshot with copied states
        snapshot = StateSnapshot(
            agent_state=agent_state_copy,
            episode_state=episode_state_copy,
            timestamp=snapshot_timestamp,
            plume_model=self.plume_model,
        )

        # Validate snapshot consistency if requested
        if validate_consistency:
            consistency_result = snapshot.validate_consistency()
            if not consistency_result["is_consistent"]:
                self.logger.warning(
                    f"Snapshot consistency validation failed: {consistency_result}"
                )

        # Generate snapshot name if not provided
        if snapshot_name is None:
            snapshot_name = f"snapshot_{int(snapshot_timestamp * 1000)}"

        # Cache snapshot with size limit management
        self.state_snapshots[snapshot_name] = snapshot

        # Manage cache size by removing oldest snapshots
        if len(self.state_snapshots) > self.config.snapshot_cache_size:
            oldest_key = min(
                self.state_snapshots.keys(),
                key=lambda k: self.state_snapshots[k].timestamp,
            )
            del self.state_snapshots[oldest_key]

        # Log snapshot creation
        self.logger.debug(
            f"Created state snapshot '{snapshot_name}' with timestamp {snapshot_timestamp}"
        )

        return snapshot

    def validate_state_consistency(
        self, strict_validation: bool = False, validate_component_sync: bool = False
    ) -> StateValidationResult:
        """
        Performs comprehensive state consistency validation across agent state, episode state,
        and component coordination with detailed error analysis and recovery recommendations.
        """
        validation_timestamp = time.time()
        validation_result = StateValidationResult(
            is_valid=True,
            validation_details={},
            validation_timestamp=validation_timestamp,
        )

        try:
            # Validate current states are available
            if self.current_agent_state is None:
                validation_result.add_error(
                    "Current agent state is None",
                    "missing_state",
                    "Ensure episode has been reset and initialized",
                )

            if self.current_episode_state is None:
                validation_result.add_error(
                    "Current episode state is None",
                    "missing_state",
                    "Ensure episode has been reset and initialized",
                )

            # If states are available, perform detailed validation
            if self.current_agent_state and self.current_episode_state:
                # Validate agent position is within grid bounds
                if not self.config.grid_size.contains_coordinates(
                    self.current_agent_state.position
                ):
                    validation_result.add_error(
                        f"Agent position {self.current_agent_state.position} is outside grid bounds",
                        "boundary_violation",
                        "Reset agent position or adjust grid size",
                    )

                # Validate step count consistency
                if self.current_agent_state.step_count < 0:
                    validation_result.add_error(
                        f"Negative step count: {self.current_agent_state.step_count}",
                        "invalid_step_count",
                        "Reset step count to valid value",
                    )

                # Validate reward accumulation
                if not isinstance(self.current_agent_state.total_reward, (int, float)):
                    validation_result.add_error(
                        f"Invalid reward type: {type(self.current_agent_state.total_reward)}",
                        "invalid_reward",
                        "Ensure reward is numeric value",
                    )

                # Validate episode state consistency
                if (
                    self.current_episode_state.terminated
                    and self.current_episode_state.truncated
                ):
                    validation_result.add_warning(
                        "Episode marked as both terminated and truncated",
                        "state_ambiguity",
                        "Review episode termination logic",
                    )

                # Check boundary enforcer integration
                boundary_valid = self.boundary_enforcer.validate_position(
                    self.current_agent_state.position, raise_on_invalid=False
                )
                if not boundary_valid:
                    validation_result.add_error(
                        "Agent position fails boundary validation",
                        "boundary_inconsistency",
                        "Synchronize state manager with boundary enforcer",
                    )

                # Strict validation checks
                if strict_validation:
                    if self.current_agent_state.step_count > self.config.max_steps:
                        validation_result.add_error(
                            f"Step count {self.current_agent_state.step_count} exceeds max_steps {self.config.max_steps}",
                            "step_limit_violation",
                            "Check episode truncation logic",
                        )

            # Component synchronization validation if requested
            if validate_component_sync:
                sync_result = self.synchronizer.synchronize_state(
                    strict_synchronization=strict_validation
                )
                if not sync_result.is_valid:
                    validation_result.add_warning(
                        "Component synchronization validation failed",
                        "sync_failure",
                        "Review component registration and synchronization methods",
                    )
                    validation_result.component_status.update(
                        sync_result.component_status
                    )

            # Record validation details
            validation_result.validation_details["validation_type"] = (
                "comprehensive" if strict_validation else "standard"
            )
            validation_result.validation_details["components_checked"] = (
                validate_component_sync
            )
            validation_result.validation_details["episode_active"] = self.episode_active
            validation_result.validation_details["episode_count"] = self.episode_count

            # Generate recovery recommendations
            if validation_result.validation_errors:
                validation_result.recovery_recommendations.extend(
                    [
                        "Review episode initialization and reset procedures",
                        "Validate component integration and state synchronization",
                        "Check boundary enforcement and movement logic",
                    ]
                )

            # Log validation completion
            error_count = len(validation_result.validation_errors)
            warning_count = len(validation_result.validation_warnings)
            self.logger.info(
                f"State consistency validation completed: {error_count} errors, {warning_count} warnings"
            )

            return validation_result

        except Exception as e:
            # Handle validation process errors
            validation_result.is_valid = False
            validation_result.add_error(
                f"State validation process failed: {str(e)}",
                "validation_error",
                "Review validation logic and state accessibility",
            )

            self.logger.error(f"State validation failed: {e}")
            return validation_result

    def get_performance_metrics(
        self,
        include_component_breakdown: bool = False,
        include_optimization_recommendations: bool = False,
    ) -> Dict[str, Any]:
        """
        Returns comprehensive performance metrics for state management operations including timing analysis,
        component coordination efficiency, and optimization recommendations.
        """
        # Get base performance metrics
        base_metrics = self.performance_metrics.get_performance_summary()

        # Compile episode-level statistics
        episode_metrics = {
            "total_episodes": self.episode_count,
            "current_episode_active": self.episode_active,
            "current_step_count": (
                self.current_agent_state.step_count if self.current_agent_state else 0
            ),
            "average_episode_length": 0.0,
            "step_processing_efficiency": 0.0,
        }

        # Calculate efficiency metrics if performance data available
        if hasattr(self.performance_metrics, "timing_data"):
            step_timings = self.performance_metrics.get_timing_data().get(
                "process_step", []
            )
            if step_timings:
                avg_step_time = sum(step_timings) / len(step_timings)
                episode_metrics["average_step_time_ms"] = avg_step_time
                episode_metrics["step_processing_efficiency"] = (
                    PERFORMANCE_TARGET_STEP_LATENCY_MS / avg_step_time
                )

        # Combine metrics
        performance_data = {
            "collection_timestamp": time.time(),
            "base_metrics": base_metrics,
            "episode_metrics": episode_metrics,
            "memory_usage": {
                "active_snapshots": len(self.state_snapshots),
                "history_entries": len(self.state_history),
                "component_cache_size": len(self.component_cache),
            },
        }

        # Add component breakdown if requested
        if include_component_breakdown:
            performance_data["component_breakdown"] = {
                "synchronizer_metrics": self.synchronizer.synchronization_metrics,
                "boundary_enforcer_calls": getattr(
                    self.boundary_enforcer, "validation_count", 0
                ),
                "seed_manager_operations": getattr(
                    self.seed_manager, "operation_count", 0
                ),
            }

        # Add optimization recommendations if requested
        if include_optimization_recommendations:
            recommendations = []

            # Analyze step timing
            if (
                episode_metrics.get("average_step_time_ms", 0)
                > PERFORMANCE_TARGET_STEP_LATENCY_MS
            ):
                recommendations.append(
                    "Step processing time exceeds target - consider optimization"
                )

            # Analyze memory usage
            if len(self.state_snapshots) > 30:
                recommendations.append(
                    "High snapshot count - consider reducing cache size"
                )

            # Analyze synchronization efficiency
            sync_metrics = self.synchronizer.synchronization_metrics
            if sync_metrics["failed_synchronizations"] > 0:
                recommendations.append("Component synchronization failures detected")

            performance_data["optimization_recommendations"] = recommendations

        return performance_data

    def cleanup(
        self, preserve_configuration: bool = True, clear_performance_data: bool = False
    ) -> None:
        """
        Performs comprehensive cleanup of state manager including component coordination cleanup,
        cache clearing, and resource release for environment shutdown or reset.
        """
        with self.state_lock:
            try:
                self.logger.info("Starting StateManager cleanup")

                # Set episode as inactive
                self.episode_active = False

                # Clear current states
                self.current_agent_state = None
                self.current_episode_state = None

                # Clear history and snapshots
                self.state_history.clear()
                self.state_snapshots.clear()

                # Clear component cache
                self.component_cache.clear()

                # Clear performance data if requested
                if clear_performance_data:
                    if hasattr(self.performance_metrics, "clear"):
                        self.performance_metrics.clear()

                # Reset tracking variables (preserve episode_count for continuity)
                if not preserve_configuration:
                    self.episode_count = 0

                # Component cleanup
                if hasattr(self.synchronizer, "component_registry"):
                    # Clear synchronization locks
                    self.synchronizer.synchronization_locks.clear()

                    # Clear component registry
                    self.synchronizer.component_registry.clear()

                self.logger.info(
                    f"StateManager cleanup completed (configuration preserved: {preserve_configuration})"
                )

            except Exception as e:
                self.logger.error(f"StateManager cleanup failed: {e}")
                raise ComponentError(
                    message=f"StateManager cleanup failed: {str(e)}",
                    component_name="StateManager",
                    operation_name="cleanup",
                ) from e


def create_state_manager(
    config: Optional[StateManagerConfig] = None,
    seed_manager: Optional[SeedManager] = None,
    boundary_enforcer: Optional[BoundaryEnforcer] = None,
    enable_performance_monitoring: bool = DEFAULT_PERFORMANCE_MONITORING,
    enable_state_validation: bool = DEFAULT_STATE_VALIDATION,
) -> StateManager:
    """
    Factory function to create properly configured StateManager with component coordination,
    validation setup, and performance monitoring for reinforcement learning environment state management.
    """
    try:
        # Create default StateManagerConfig if none provided
        if config is None:
            default_grid_size = GridSize(
                width=DEFAULT_GRID_SIZE[0], height=DEFAULT_GRID_SIZE[1]
            )
            default_source_location = create_coordinates(DEFAULT_SOURCE_LOCATION)

            config = StateManagerConfig(
                grid_size=default_grid_size,
                source_location=default_source_location,
                max_steps=DEFAULT_MAX_STEPS,
                goal_radius=DEFAULT_GOAL_RADIUS,
                enable_performance_monitoring=enable_performance_monitoring,
                enable_state_validation=enable_state_validation,
            )

        # Validate configuration using comprehensive checking
        config.validate()

        # Initialize SeedManager if not provided
        if seed_manager is None:
            seed_manager = SeedManager()

        # Create BoundaryEnforcer if not provided
        if boundary_enforcer is None:
            boundary_enforcer = BoundaryEnforcer(grid_size=config.grid_size)

        # Initialize StateManager with validated configuration and components
        state_manager = StateManager(
            config=config,
            seed_manager=seed_manager,
            boundary_enforcer=boundary_enforcer,
        )

        # Validate complete setup
        validation_result = state_manager.validate_state_consistency(
            strict_validation=enable_state_validation
        )

        if not validation_result.is_valid and enable_state_validation:
            raise ComponentError(
                message="StateManager initialization validation failed",
                component_name="StateManagerFactory",
                operation_name="create_state_manager",
            )

        logger = get_component_logger("StateManagerFactory")
        logger.info("StateManager created successfully with component coordination")

        return state_manager

    except (ValidationError, ComponentError):
        # Re-raise specific exceptions
        raise
    except Exception as e:
        # Wrap unexpected errors
        raise ComponentError(
            message=f"StateManager creation failed: {str(e)}",
            component_name="StateManagerFactory",
            operation_name="create_state_manager",
        ) from e


def validate_state_manager_config(
    config: StateManagerConfig,
    strict_validation: bool = False,
    validation_context: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Comprehensive validation of state manager configuration parameters ensuring mathematical consistency,
    component compatibility, and performance feasibility for state management operations.
    """
    try:
        validation_report = {
            "is_valid": True,
            "validation_timestamp": time.time(),
            "strict_mode": strict_validation,
            "findings": [],
            "warnings": [],
            "recommendations": [],
            "resource_analysis": {},
        }

        # Validate grid size dimensions
        if config.grid_size.width <= 0 or config.grid_size.height <= 0:
            validation_report["is_valid"] = False
            validation_report["findings"].append(
                {
                    "category": "grid_dimensions",
                    "severity": "critical",
                    "message": "Grid dimensions must be positive integers",
                    "current_value": f"({config.grid_size.width}, {config.grid_size.height})",
                }
            )

        # Check memory implications
        total_cells = config.grid_size.width * config.grid_size.height
        estimated_memory_mb = (total_cells * 8) / (1024 * 1024)

        if estimated_memory_mb > 1000:  # 1GB limit
            validation_report["is_valid"] = False
            validation_report["findings"].append(
                {
                    "category": "memory_usage",
                    "severity": "critical",
                    "message": f"Estimated memory usage too high: {estimated_memory_mb:.1f}MB",
                    "recommendation": "Reduce grid size for acceptable memory usage",
                }
            )
        elif estimated_memory_mb > 500:  # 500MB warning
            validation_report["warnings"].append(
                {
                    "category": "memory_usage",
                    "message": f"High memory usage estimated: {estimated_memory_mb:.1f}MB",
                    "recommendation": "Consider grid size optimization",
                }
            )

        # Validate source location
        if not config.grid_size.contains_coordinates(config.source_location):
            validation_report["is_valid"] = False
            validation_report["findings"].append(
                {
                    "category": "source_location",
                    "severity": "critical",
                    "message": "Source location outside grid bounds",
                    "current_value": f"({config.source_location.x}, {config.source_location.y})",
                    "grid_bounds": f"(0,0) to ({config.grid_size.width-1}, {config.grid_size.height-1})",
                }
            )

        # Validate max_steps
        if config.max_steps <= 0:
            validation_report["is_valid"] = False
            validation_report["findings"].append(
                {
                    "category": "episode_limits",
                    "severity": "critical",
                    "message": "max_steps must be positive",
                    "current_value": config.max_steps,
                }
            )
        elif strict_validation and config.max_steps > 10000:
            validation_report["warnings"].append(
                {
                    "category": "episode_limits",
                    "message": "Very high max_steps value may impact performance",
                    "current_value": config.max_steps,
                    "recommendation": "Consider reducing max_steps for better performance",
                }
            )

        # Validate goal_radius
        if config.goal_radius < 0:
            validation_report["is_valid"] = False
            validation_report["findings"].append(
                {
                    "category": "goal_parameters",
                    "severity": "critical",
                    "message": "goal_radius cannot be negative",
                    "current_value": config.goal_radius,
                }
            )

        # Cross-validate parameters for mathematical consistency
        max_possible_distance = np.sqrt(
            (config.grid_size.width - 1) ** 2 + (config.grid_size.height - 1) ** 2
        )

        if config.goal_radius > max_possible_distance:
            validation_report["warnings"].append(
                {
                    "category": "mathematical_consistency",
                    "message": "Goal radius larger than maximum grid distance",
                    "current_value": config.goal_radius,
                    "max_distance": max_possible_distance,
                    "recommendation": "Reduce goal_radius or increase grid size",
                }
            )

        # Generate resource analysis
        resource_estimates = config.estimate_resources()
        validation_report["resource_analysis"] = resource_estimates

        # Generate recommendations
        if validation_report["is_valid"]:
            validation_report["recommendations"].extend(
                [
                    "Configuration is valid and ready for use",
                    "Monitor performance metrics during operation",
                ]
            )

            if estimated_memory_mb > 100:
                validation_report["recommendations"].append(
                    "Consider performance testing with current grid size"
                )
        else:
            validation_report["recommendations"].extend(
                [
                    "Fix critical validation errors before proceeding",
                    "Review configuration parameters for consistency",
                    "Consider using smaller grid dimensions for testing",
                ]
            )

        return validation_report["is_valid"], validation_report

    except Exception as e:
        # Handle validation process errors
        error_report = {
            "is_valid": False,
            "validation_timestamp": time.time(),
            "validation_error": str(e),
            "recommendations": [
                "Fix validation process errors before configuration use"
            ],
        }
        return False, error_report


# Module exports for external use
__all__ = [
    "StateManager",
    "StateManagerConfig",
    "StateValidationResult",
    "StateSynchronizer",
    "create_state_manager",
    "validate_state_manager_config",
]
