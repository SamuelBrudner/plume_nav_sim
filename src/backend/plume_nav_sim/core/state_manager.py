import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .._compat import (
    ComponentError,
    SeedManager,
    StateError,
    ValidationError,
    validate_action_input,
)
from ..constants import (
    DEFAULT_GOAL_RADIUS,
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_STEPS,
    DEFAULT_PLUME_SIGMA,
    DEFAULT_SOURCE_LOCATION,
    MOVEMENT_VECTORS,
    PERFORMANCE_TARGET_STEP_LATENCY_MS,
)
from ..logging import get_component_logger
from .boundary_enforcer import BoundaryEnforcer
from .types import Action, AgentState, Coordinates, GridSize, create_coordinates

DEFAULT_PERFORMANCE_MONITORING = True
DEFAULT_STATE_VALIDATION = True
DEFAULT_HISTORY_TRACKING = False
STATE_SNAPSHOT_CACHE_SIZE = 50
COMPONENT_COORDINATION_TIMEOUT = 1.0


@dataclass
class StateManagerConfig:
    """Configuration for StateManager."""

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

    def __post_init__(self) -> None:
        self._validate_core()

    def _validate_core(self) -> None:
        if not isinstance(self.grid_size, GridSize):
            raise ValidationError(
                "grid_size must be a GridSize instance",
                parameter_name="grid_size",
                parameter_value=type(self.grid_size).__name__,
            )
        if self.grid_size.width <= 0 or self.grid_size.height <= 0:
            raise ValidationError(
                "Grid dimensions must be positive",
                parameter_name="grid_size",
                parameter_value=f"({self.grid_size.width}, {self.grid_size.height})",
            )
        if not isinstance(self.source_location, Coordinates):
            raise ValidationError(
                "source_location must be Coordinates",
                parameter_name="source_location",
                parameter_value=type(self.source_location).__name__,
            )
        if not self.grid_size.contains(self.source_location):
            raise ValidationError(
                "Source location must be within grid bounds",
                parameter_name="source_location",
                parameter_value=f"({self.source_location.x}, {self.source_location.y})",
            )
        if not isinstance(self.max_steps, int) or self.max_steps <= 0:
            raise ValidationError(
                "max_steps must be a positive integer",
                parameter_name="max_steps",
                parameter_value=str(self.max_steps),
            )
        if not isinstance(self.goal_radius, (int, float)) or self.goal_radius < 0:
            raise ValidationError(
                "goal_radius must be non-negative",
                parameter_name="goal_radius",
                parameter_value=str(self.goal_radius),
            )
        if self.snapshot_cache_size < 0 or self.snapshot_cache_size > 1000:
            raise ValidationError(
                "snapshot_cache_size must be between 0 and 1000",
                parameter_name="snapshot_cache_size",
                parameter_value=str(self.snapshot_cache_size),
            )
        if (
            not isinstance(self.component_coordination_timeout, (int, float))
            or self.component_coordination_timeout <= 0
        ):
            raise ValidationError(
                "component_coordination_timeout must be positive",
                parameter_name="component_coordination_timeout",
                parameter_value=str(self.component_coordination_timeout),
            )
        if not isinstance(self.custom_parameters, dict):
            self.custom_parameters = {}

    def validate(
        self,
        strict_mode: bool = False,
        validation_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Validate configuration values and basic cross-checks."""
        self._validate_core()
        max_distance = self.source_location.distance_to(
            create_coordinates((self.grid_size.width - 1, self.grid_size.height - 1))
        )
        if self.goal_radius > max_distance:
            raise ValidationError(
                "goal_radius exceeds maximum possible distance",
                parameter_name="goal_radius",
                parameter_value=str(self.goal_radius),
            )
        if strict_mode and self.max_steps > 10000:
            raise ValidationError(
                "max_steps too large for strict validation",
                parameter_name="max_steps",
                parameter_value=str(self.max_steps),
            )
        return True

    def estimate_resources(
        self,
        include_component_overhead: bool = True,
        include_monitoring_costs: bool = True,
    ) -> Dict[str, Any]:
        """Estimate memory use for state tracking."""
        base_memory_mb = 10.0
        grid_cells = self.grid_size.width * self.grid_size.height
        grid_memory_mb = (grid_cells * 4) / (1024 * 1024)
        history_memory_mb = 0.0
        if self.enable_history_tracking:
            avg_episode_length = min(self.max_steps, 1000)
            history_memory_mb = (avg_episode_length * 100) / (1024 * 1024)
        snapshot_memory_mb = (self.snapshot_cache_size * 1) / 1024
        component_overhead_mb = 5.0 if include_component_overhead else 0.0
        monitoring_overhead_mb = (
            2.0
            if include_monitoring_costs and self.enable_performance_monitoring
            else 0.0
        )
        total_memory_mb = (
            base_memory_mb
            + grid_memory_mb
            + history_memory_mb
            + snapshot_memory_mb
            + component_overhead_mb
            + monitoring_overhead_mb
        )
        return {
            "memory_mb": total_memory_mb,
            "breakdown_mb": {
                "base_state_manager": base_memory_mb,
                "grid_storage": grid_memory_mb,
                "history_tracking": history_memory_mb,
                "snapshot_cache": snapshot_memory_mb,
                "component_overhead": component_overhead_mb,
                "monitoring_overhead": monitoring_overhead_mb,
            },
            "step_latency_target_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS,
        }

    def clone(
        self,
        overrides: Optional[Dict[str, Any]] = None,
        preserve_custom_parameters: bool = True,
    ) -> "StateManagerConfig":
        """Clone configuration with optional overrides."""
        cloned_config = copy.deepcopy(self)
        if overrides:
            for key, value in overrides.items():
                if not hasattr(cloned_config, key):
                    raise ValidationError(
                        f"Invalid override parameter: {key}",
                        parameter_name="overrides",
                        parameter_value=key,
                    )
                setattr(cloned_config, key, value)
        if not preserve_custom_parameters:
            cloned_config.custom_parameters = {}
        if cloned_config.enable_state_validation:
            cloned_config.validate()
        return cloned_config

    def to_dict(
        self,
        include_resource_estimates: bool = False,
        include_validation_results: bool = False,
    ) -> Dict[str, Any]:
        """Serialize configuration to a dictionary."""
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
            except ValidationError as exc:
                config_dict["validation_results"] = {
                    "is_valid": False,
                    "errors": [str(exc)],
                }
        return config_dict


@dataclass
class StateValidationResult:
    """State validation output container."""

    is_valid: bool
    validation_details: Dict[str, Any]
    validation_timestamp: float
    validation_errors: List[Dict[str, Any]] = field(default_factory=list)
    validation_warnings: List[Dict[str, Any]] = field(default_factory=list)
    component_status: Dict[str, Any] = field(default_factory=dict)
    recovery_recommendations: List[str] = field(default_factory=list)

    def add_error(
        self,
        error_message: str,
        error_category: str,
        recovery_suggestion: Optional[str] = None,
    ) -> None:
        entry = {
            "message": error_message,
            "category": error_category,
            "timestamp": time.time(),
        }
        if recovery_suggestion:
            entry["recovery_suggestion"] = recovery_suggestion
        self.validation_errors.append(entry)
        self.is_valid = False

    def add_warning(
        self,
        warning_message: str,
        warning_category: str,
        optimization_suggestion: Optional[str] = None,
    ) -> None:
        entry = {
            "message": warning_message,
            "category": warning_category,
            "timestamp": time.time(),
        }
        if optimization_suggestion:
            entry["optimization_suggestion"] = optimization_suggestion
        self.validation_warnings.append(entry)

    def get_summary(self, include_details: bool = False) -> Dict[str, Any]:
        """Summarize validation results."""
        summary = {
            "overall_status": "VALID" if self.is_valid else "INVALID",
            "validation_timestamp": self.validation_timestamp,
            "error_count": len(self.validation_errors),
            "warning_count": len(self.validation_warnings),
        }
        if include_details:
            summary.update(
                {
                    "validation_errors": self.validation_errors,
                    "validation_warnings": self.validation_warnings,
                    "component_status": self.component_status,
                    "validation_details": self.validation_details,
                }
            )
        return summary


class StateSynchronizer:
    """Coordinate state synchronization across components."""

    def __init__(
        self,
        state_manager: "StateManager",
        boundary_enforcer: Optional[BoundaryEnforcer] = None,
    ) -> None:
        self.state_manager = state_manager
        self.boundary_enforcer = boundary_enforcer
        self.component_registry: Dict[str, Any] = {}
        self.synchronization_locks: Dict[str, threading.Lock] = {}
        self.logger = get_component_logger("state_manager")
        self.synchronization_metrics: Dict[str, Any] = {
            "total_synchronizations": 0,
            "failed_synchronizations": 0,
            "average_sync_time_ms": 0.0,
            "last_sync_timestamp": 0.0,
        }

    def register_component(
        self, component_name: str, component_instance: Any, sync_methods: List[str]
    ) -> None:
        if not isinstance(component_name, str) or not component_name.strip():
            raise ValidationError(
                "Component name must be a non-empty string",
                parameter_name="component_name",
                parameter_value=str(component_name),
            )
        if component_name in self.component_registry:
            raise ComponentError(
                message=f"Component '{component_name}' is already registered",
                component_name="StateSynchronizer",
                operation_name="register_component",
            )
        for method_name in sync_methods:
            if not hasattr(component_instance, method_name):
                raise ValidationError(
                    f"Component missing required method: {method_name}",
                    parameter_name="component_instance",
                    parameter_value=f"missing {method_name}",
                )
        self.component_registry[component_name] = {
            "instance": component_instance,
            "sync_methods": sync_methods,
            "last_sync_timestamp": 0.0,
            "sync_count": 0,
            "sync_errors": 0,
        }
        self.synchronization_locks[component_name] = threading.Lock()

    def synchronize_state(
        self,
        strict_synchronization: bool = False,
        component_subset: Optional[List[str]] = None,
    ) -> StateValidationResult:
        """Synchronize registered components."""
        sync_start_time = time.time()
        validation_result = StateValidationResult(
            is_valid=True, validation_details={}, validation_timestamp=sync_start_time
        )
        components_to_sync = component_subset or list(self.component_registry.keys())
        for component_name in components_to_sync:
            if component_name not in self.component_registry:
                validation_result.add_warning(
                    f"Component '{component_name}' not found",
                    "missing_component",
                    "Ensure component is registered",
                )
                continue
            component_info = self.component_registry[component_name]
            with self.synchronization_locks[component_name]:
                try:
                    component_instance = component_info["instance"]
                    for method_name in component_info["sync_methods"]:
                        getattr(component_instance, method_name)(self.state_manager)
                    component_info["last_sync_timestamp"] = time.time()
                    component_info["sync_count"] += 1
                    validation_result.component_status[component_name] = {
                        "status": "synchronized",
                        "last_sync": component_info["last_sync_timestamp"],
                        "sync_count": component_info["sync_count"],
                    }
                except Exception as exc:
                    component_info["sync_errors"] += 1
                    validation_result.add_error(
                        f"Component '{component_name}' synchronization failed: {exc}",
                        "sync_failure",
                        f"Check component '{component_name}' state",
                    )
                    validation_result.component_status[component_name] = {
                        "status": "sync_failed",
                        "error": str(exc),
                        "sync_errors": component_info["sync_errors"],
                    }
                    self.logger.error(
                        f"Component '{component_name}' sync failed: {exc}"
                    )
        sync_duration_ms = (time.time() - sync_start_time) * 1000
        self.synchronization_metrics["total_synchronizations"] += 1
        self.synchronization_metrics["last_sync_timestamp"] = time.time()
        if validation_result.validation_errors:
            self.synchronization_metrics["failed_synchronizations"] += 1
            validation_result.is_valid = False
        total_syncs = self.synchronization_metrics["total_synchronizations"]
        current_avg = self.synchronization_metrics["average_sync_time_ms"]
        self.synchronization_metrics["average_sync_time_ms"] = (
            (current_avg * (total_syncs - 1)) + sync_duration_ms
        ) / total_syncs
        validation_result.validation_details["synchronization"] = {
            "duration_ms": sync_duration_ms,
            "components_processed": len(components_to_sync),
            "strict_mode": strict_synchronization,
        }
        return validation_result


class StateManager:
    """Manage agent and episode state."""

    def __init__(
        self,
        config: StateManagerConfig,
        seed_manager: Optional[SeedManager] = None,
        boundary_enforcer: Optional[BoundaryEnforcer] = None,
    ) -> None:
        if not isinstance(config, StateManagerConfig):
            raise ValidationError(
                "config must be a StateManagerConfig instance",
                parameter_name="config",
                parameter_value=type(config).__name__,
            )
        if config.enable_state_validation:
            config.validate()
        self.config = config
        self.seed_manager = seed_manager if seed_manager is not None else SeedManager()
        if boundary_enforcer is not None:
            self.boundary_enforcer = boundary_enforcer
        else:
            self.boundary_enforcer = BoundaryEnforcer(grid_size=config.grid_size)
        self.current_agent_state: Optional[AgentState] = None
        self.current_episode_state: Optional[Dict[str, Any]] = None
        self.episode_active: bool = False
        self.episode_count: int = 0
        self.state_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {}
        self.synchronizer = StateSynchronizer(self, self.boundary_enforcer)
        self.logger = get_component_logger("state_manager")
        self.state_lock = threading.RLock()
        self.component_cache: Dict[str, Any] = {}
        self.plume_params = {
            "source_location": self.config.source_location,
            "sigma": float(DEFAULT_PLUME_SIGMA),
        }

    def reset_episode(
        self,
        seed: Optional[int] = None,
        episode_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Reset episode state and return metadata."""
        with self.state_lock:
            try:
                reset_start_time = time.time()
                if seed is not None:
                    self.seed_manager.seed(seed)
                episode_id = f"episode_{int(time.time() * 1000)}_{self.episode_count}"
                self.current_agent_state = None
                self.current_episode_state = None
                if self.config.enable_history_tracking:
                    self.state_history.clear()
                self.component_cache.clear()
                start_position = None
                for _ in range(100):
                    candidate = self.seed_manager.generate_random_position(
                        self.config.grid_size,
                        exclude_position=self.config.source_location,
                    )
                    if self.boundary_enforcer.validate_position(
                        candidate, raise_on_invalid=False
                    ):
                        start_position = candidate
                        break
                if start_position is None:
                    raise StateError(
                        message="Could not generate valid start position",
                        current_state="episode_reset",
                        expected_state="valid_start_position_generated",
                    )
                self.current_agent_state = AgentState(
                    position=start_position, step_count=0, total_reward=0.0
                )
                self.current_episode_state = {
                    "episode_id": episode_id,
                    "terminated": False,
                    "truncated": False,
                    "start_time": time.time(),
                    "end_time": None,
                    "termination_reason": None,
                }
                self.episode_active = True
                self.episode_count += 1
                if self.config.enable_history_tracking:
                    self.state_history.append(
                        {
                            "step_number": 0,
                            "timestamp": time.time(),
                            "event_type": "episode_start",
                            "agent_position": start_position.to_tuple(),
                        }
                    )
                reset_duration_ms = (time.time() - reset_start_time) * 1000
                self.performance_metrics.setdefault("episode_reset", []).append(
                    reset_duration_ms
                )
                return {
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
            except (ValidationError, StateError, ComponentError):
                raise
            except Exception as exc:
                raise StateError(
                    message=f"Episode reset failed: {exc}",
                    current_state="episode_reset",
                    expected_state="reset_complete",
                    component_name="StateManager",
                ) from exc

    def process_step(
        self, action: Action, action_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process one step of movement."""
        with self.state_lock:
            try:
                step_start_time = time.time()
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
                        message="Cannot process step: state not initialized",
                        current_state="uninitialized",
                        expected_state="initialized",
                        component_name="StateManager",
                    )
                action_id = validate_action_input(action)
                movement_delta = MOVEMENT_VECTORS[action_id]
                current_pos = self.current_agent_state.position
                new_position = create_coordinates(
                    (
                        current_pos.x + movement_delta[0],
                        current_pos.y + movement_delta[1],
                    )
                )
                if self.boundary_enforcer.validate_position(
                    new_position, raise_on_invalid=False
                ):
                    self.current_agent_state.update_position(new_position)
                self.current_agent_state.increment_step()
                if self.config.enable_history_tracking:
                    self.state_history.append(
                        {
                            "step_number": self.current_agent_state.step_count,
                            "action": action,
                            "position": (
                                self.current_agent_state.position.x,
                                self.current_agent_state.position.y,
                            ),
                            "timestamp": time.time(),
                            "event_type": "step_processed",
                        }
                    )
                step_duration_ms = (time.time() - step_start_time) * 1000
                self.performance_metrics.setdefault("process_step", []).append(
                    step_duration_ms
                )
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
                if action_context:
                    step_result["context"] = action_context
                return step_result
            except (ValidationError, StateError, ComponentError):
                raise
            except Exception as exc:
                raise StateError(
                    message=f"Step processing failed: {exc}",
                    current_state="step_processing",
                    expected_state="step_complete",
                    component_name="StateManager",
                ) from exc

    def initialize_episode(self, episode_seed: Optional[int] = None) -> Dict[str, Any]:
        """Compatibility alias for reset_episode."""
        return self.reset_episode(seed=episode_seed)

    def update_agent_state(self, action: int, step_reward: float) -> Dict[str, Any]:
        """Compatibility wrapper for apply_step."""
        return self.apply_step(action=action, reward=step_reward)

    def apply_step(self, action: int, reward: float) -> Dict[str, Any]:
        """Apply action and record external reward."""
        step_info = self.process_step(action)  # type: ignore[arg-type]
        if self.current_agent_state is None:
            raise StateError(
                message="Cannot update reward: no active agent state",
                current_state="no_active_episode",
                expected_state="active_episode",
                component_name="StateManager",
            )
        self.current_agent_state.add_reward(float(reward))
        return step_info

    def check_episode_termination(
        self,
        source_location: Coordinates,
        goal_radius: float,
        termination_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Check goal reach and step limit termination."""
        if (
            not self.episode_active
            or self.current_agent_state is None
            or self.current_episode_state is None
        ):
            return {
                "terminated": False,
                "truncated": False,
                "reason": "episode_not_active",
                "analysis": {"error": "Episode not in valid state"},
            }
        current_position = self.current_agent_state.position
        distance_to_source = current_position.distance_to(source_location)
        goal_achieved = distance_to_source <= goal_radius
        step_limit_reached = (
            self.current_agent_state.step_count >= self.config.max_steps
        )
        terminated = goal_achieved
        truncated = step_limit_reached and not goal_achieved
        if terminated or truncated:
            self.current_episode_state["terminated"] = terminated
            self.current_episode_state["truncated"] = truncated
            self.current_episode_state["termination_reason"] = (
                "goal_achieved" if terminated else "step_limit"
            )
            self.current_episode_state["end_time"] = time.time()
            self.episode_active = False
        analysis = {
            "terminated": terminated,
            "truncated": truncated,
            "goal_achieved": goal_achieved,
            "distance_to_source": distance_to_source,
            "goal_radius": goal_radius,
            "step_count": self.current_agent_state.step_count,
            "max_steps": self.config.max_steps,
            "termination_reason": (
                "goal_reached"
                if terminated
                else "step_limit_exceeded" if truncated else "episode_continues"
            ),
        }
        if termination_context:
            analysis["context"] = termination_context
        return {
            "terminated": terminated,
            "truncated": truncated,
            "reason": analysis["termination_reason"],
            "analysis": analysis,
        }

    def get_current_state(
        self,
        include_performance_data: bool = False,
        include_component_details: bool = False,
    ) -> Dict[str, Any]:
        """Return current state snapshot for external consumers."""
        with self.state_lock:
            if self.current_agent_state is None or self.current_episode_state is None:
                return {
                    "state_available": False,
                    "error": "No active episode state",
                    "episode_active": self.episode_active,
                    "episode_count": self.episode_count,
                }
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
                    "terminated": bool(self.current_episode_state.get("terminated")),
                    "truncated": bool(self.current_episode_state.get("truncated")),
                    "is_done": bool(self.current_episode_state.get("terminated"))
                    or bool(self.current_episode_state.get("truncated")),
                    "episode_duration": (
                        (self.current_episode_state.get("end_time") or time.time())
                        - self.current_episode_state.get("start_time", time.time())
                    ),
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
            if include_performance_data:
                current_state["performance_metrics"] = self._performance_summary()
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
            if self.config.enable_history_tracking:
                current_state["history"] = {"total_snapshots": len(self.state_history)}
            return current_state

    def validate_state_consistency(
        self, strict_validation: bool = False, validate_component_sync: bool = False
    ) -> StateValidationResult:
        """Validate agent/episode state consistency."""
        validation_timestamp = time.time()
        validation_result = StateValidationResult(
            is_valid=True,
            validation_details={},
            validation_timestamp=validation_timestamp,
        )
        if self.current_agent_state is None:
            validation_result.add_error(
                "Current agent state is None",
                "missing_state",
                "Ensure episode has been reset",
            )
        if self.current_episode_state is None:
            validation_result.add_error(
                "Current episode state is None",
                "missing_state",
                "Ensure episode has been reset",
            )
        if self.current_agent_state and self.current_episode_state:
            if not self.config.grid_size.contains(self.current_agent_state.position):
                validation_result.add_error(
                    f"Agent position {self.current_agent_state.position} is outside grid",
                    "boundary_violation",
                    "Reset agent position or adjust grid size",
                )
            if self.current_agent_state.step_count < 0:
                validation_result.add_error(
                    f"Negative step count: {self.current_agent_state.step_count}",
                    "invalid_step_count",
                    "Reset step count to valid value",
                )
            if self.current_episode_state.get(
                "terminated"
            ) and self.current_episode_state.get("truncated"):
                validation_result.add_warning(
                    "Episode marked as both terminated and truncated",
                    "state_ambiguity",
                    "Review episode termination logic",
                )
            boundary_valid = self.boundary_enforcer.validate_position(
                self.current_agent_state.position, raise_on_invalid=False
            )
            if not boundary_valid:
                validation_result.add_error(
                    "Agent position fails boundary validation",
                    "boundary_inconsistency",
                    "Synchronize state manager with boundary enforcer",
                )
            if strict_validation:
                if self.current_agent_state.step_count > self.config.max_steps:
                    validation_result.add_error(
                        "Step count exceeds max_steps",
                        "step_limit_violation",
                        "Check episode truncation logic",
                    )
        if validate_component_sync:
            sync_result = self.synchronizer.synchronize_state(
                strict_synchronization=strict_validation
            )
            if not sync_result.is_valid:
                validation_result.add_warning(
                    "Component synchronization validation failed",
                    "sync_failure",
                    "Review component registration",
                )
                validation_result.component_status.update(sync_result.component_status)
        validation_result.validation_details.update(
            {
                "validation_type": "strict" if strict_validation else "standard",
                "components_checked": validate_component_sync,
                "episode_active": self.episode_active,
                "episode_count": self.episode_count,
            }
        )
        if validation_result.validation_errors:
            validation_result.recovery_recommendations.extend(
                [
                    "Review episode initialization and reset procedures",
                    "Validate component integration and state synchronization",
                ]
            )
        return validation_result

    def _performance_summary(self) -> Dict[str, Any]:
        timings = {
            key: list(values) for key, values in self.performance_metrics.items()
        }
        step_timings = timings.get("process_step", [])
        total_step_time = sum(step_timings)
        avg_step_time = total_step_time / len(step_timings) if step_timings else 0.0
        return {
            "total_step_time_ms": total_step_time,
            "average_step_time_ms": avg_step_time,
            "timings": timings,
        }

    def get_performance_metrics(
        self,
        include_component_breakdown: bool = False,
        include_optimization_recommendations: bool = False,
    ) -> Dict[str, Any]:
        """Return performance metrics for state manager."""
        base_metrics = self._performance_summary()
        episode_metrics = {
            "total_episodes": self.episode_count,
            "current_episode_active": self.episode_active,
            "current_step_count": (
                self.current_agent_state.step_count if self.current_agent_state else 0
            ),
        }
        if base_metrics.get("average_step_time_ms") is not None:
            episode_metrics["average_step_time_ms"] = base_metrics[
                "average_step_time_ms"
            ]
        performance_data = {
            "collection_timestamp": time.time(),
            "base_metrics": base_metrics,
            "episode_metrics": episode_metrics,
            "memory_usage": {
                "history_entries": len(self.state_history),
                "component_cache_size": len(self.component_cache),
            },
        }
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
        if include_optimization_recommendations:
            recommendations = []
            avg_step = episode_metrics.get("average_step_time_ms")
            if avg_step and avg_step > PERFORMANCE_TARGET_STEP_LATENCY_MS:
                recommendations.append(
                    "Step processing time exceeds target - consider optimization"
                )
            if self.synchronizer.synchronization_metrics["failed_synchronizations"]:
                recommendations.append("Component synchronization failures detected")
            performance_data["optimization_recommendations"] = recommendations
        return performance_data

    def cleanup(
        self, preserve_configuration: bool = True, clear_performance_data: bool = False
    ) -> None:
        """Clear cached state and release resources."""
        with self.state_lock:
            try:
                self.episode_active = False
                self.current_agent_state = None
                self.current_episode_state = None
                self.state_history.clear()
                self.component_cache.clear()
                if clear_performance_data:
                    self.performance_metrics.clear()
                if not preserve_configuration:
                    self.episode_count = 0
                if hasattr(self.synchronizer, "component_registry"):
                    self.synchronizer.synchronization_locks.clear()
                    self.synchronizer.component_registry.clear()
            except Exception as exc:
                raise ComponentError(
                    message=f"StateManager cleanup failed: {exc}",
                    component_name="StateManager",
                    operation_name="cleanup",
                ) from exc


def create_state_manager(
    config: Optional[StateManagerConfig] = None,
    seed_manager: Optional[SeedManager] = None,
    boundary_enforcer: Optional[BoundaryEnforcer] = None,
    enable_performance_monitoring: bool = DEFAULT_PERFORMANCE_MONITORING,
    enable_state_validation: bool = DEFAULT_STATE_VALIDATION,
) -> StateManager:
    """Factory for StateManager creation."""
    try:
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
        if seed_manager is None:
            seed_manager = SeedManager()
        if boundary_enforcer is None:
            boundary_enforcer = BoundaryEnforcer(grid_size=config.grid_size)
        state_manager = StateManager(
            config=config,
            seed_manager=seed_manager,
            boundary_enforcer=boundary_enforcer,
        )
        if enable_state_validation:
            validation_result = state_manager.validate_state_consistency(
                strict_validation=True
            )
            if not validation_result.is_valid:
                raise ComponentError(
                    message="StateManager initialization validation failed",
                    component_name="StateManagerFactory",
                    operation_name="create_state_manager",
                )
        return state_manager
    except (ValidationError, ComponentError):
        raise
    except Exception as exc:
        raise ComponentError(
            message=f"StateManager creation failed: {exc}",
            component_name="StateManagerFactory",
            operation_name="create_state_manager",
        ) from exc


def validate_state_manager_config(
    config: StateManagerConfig,
    strict_validation: bool = False,
    validation_context: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate a StateManagerConfig instance."""
    report = {
        "is_valid": True,
        "validation_timestamp": time.time(),
        "strict_mode": strict_validation,
        "errors": [],
        "warnings": [],
        "resource_analysis": {},
        "context": validation_context or {},
    }
    try:
        config.validate(
            strict_mode=strict_validation, validation_context=validation_context
        )
    except ValidationError as exc:
        report["is_valid"] = False
        report["errors"].append(str(exc))
    try:
        report["resource_analysis"] = config.estimate_resources()
    except Exception as exc:
        report["warnings"].append(f"Resource estimation failed: {exc}")
    return report["is_valid"], report


__all__ = [
    "StateManager",
    "StateManagerConfig",
    "StateValidationResult",
    "StateSynchronizer",
    "create_state_manager",
    "validate_state_manager_config",
]
