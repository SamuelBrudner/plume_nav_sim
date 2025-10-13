"""
Central episode management orchestrator for plume_nav_sim coordinating complete episode lifecycle
including initialization, step processing, termination, and cleanup with comprehensive component
integration, performance monitoring, and Gymnasium API compliance for reinforcement learning
environment implementation.

This module provides enterprise-grade episode management with mathematical precision, performance
optimization targeting <1ms step latency, comprehensive component coordination, and robust error
handling for production-ready reinforcement learning environments.
"""

import copy  # >=3.10 - Deep copying for state snapshots and configuration cloning with immutability preservation
import time  # >=3.10 - High-precision timing for episode performance measurement and state timestamping
import uuid  # >=3.10 - Unique episode identifier generation for tracking and reproducibility
from dataclasses import (  # >=3.10 - Data class utilities for episode configuration and result data structures
    dataclass,
    field,
)
from typing import (  # >=3.10 - Type hints for episode manager methods, component interfaces, and return value specifications
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    cast,
)

# Standard library imports with version comments
import numpy as np  # >=2.1.0 - Array operations, mathematical calculations, and observation array management for episode processing

try:  # pragma: no cover - numpy<1.20 compatibility
    from numpy.typing import NDArray
except ImportError:  # pragma: no cover
    NDArray = np.ndarray  # type: ignore[assignment]
from typing_extensions import NotRequired, TypedDict

from ..utils.exceptions import ComponentError, StateError, ValidationError
from ..utils.logging import get_component_logger, monitor_performance

# Internal utility imports - infrastructure and coordination
from ..utils.seeding import SeedManager
from .action_processor import ActionProcessingResult, ActionProcessor
from .constants import (
    DEFAULT_GOAL_RADIUS,
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_STEPS,
    PERFORMANCE_TARGET_STEP_LATENCY_MS,
)
from .reward_calculator import RewardCalculator, RewardResult
from .snapshots import StateSnapshot

# Internal core imports - component coordination
from .state_manager import StateManager, StateManagerConfig

# Internal core imports - data types and system constants
from .types import (
    Action,
    AgentState,
    Coordinates,
    EnvironmentConfig,
    EpisodeState,
    GridSize,
    PerformanceMetrics,
    create_episode_state,
    create_step_info,
)

# Module-level constants for episode management configuration
EPISODE_MANAGER_VERSION = "1.0.0"
DEFAULT_ENABLE_PERFORMANCE_MONITORING = True
DEFAULT_ENABLE_STATE_VALIDATION = True
DEFAULT_ENABLE_COMPONENT_INTEGRATION = True
EPISODE_PROCESSING_CACHE_SIZE = 100
COMPONENT_COORDINATION_TIMEOUT = 5.0

# Module exports for external use
__all__ = [
    "EpisodeManager",
    "EpisodeManagerConfig",
    "EpisodeResult",
    "EpisodeStatistics",
    "create_episode_manager",
    "validate_episode_config",
]


class FindingDict(TypedDict):
    category: str
    severity: str
    message: str


class WarningDict(TypedDict, total=False):
    category: str
    message: str
    recommendation: NotRequired[str]


class ParameterAnalysisDict(TypedDict, total=False):
    resource_estimates: Dict[str, object]
    grid_size: Tuple[int, int]
    max_steps: int
    goal_radius: float
    performance_monitoring_enabled: bool
    state_validation_enabled: bool
    component_integration_enabled: bool


class ValidationReport(TypedDict, total=False):
    is_valid: bool
    validation_timestamp: float
    strict_mode: bool
    findings: List[FindingDict]
    warnings: List[WarningDict]
    recommendations: List[str]
    parameter_analysis: ParameterAnalysisDict
    validation_error: NotRequired[str]
    error_details: NotRequired[Dict[str, str]]


class ConsistencyReport(TypedDict, total=False):
    is_consistent: bool
    validation_timestamp: float
    validation_details: Dict[str, object]
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]
    error: NotRequired[str]


@dataclass
class EpisodeManagerConfig:
    """
    Configuration data class for episode manager containing environment parameters, component
    settings, performance options, and integration policies with comprehensive validation and
    serialization support for complete episode orchestration setup.
    """

    # Required configuration parameters
    env_config: EnvironmentConfig

    # Optional configuration with defaults
    enable_performance_monitoring: bool = True
    enable_state_validation: bool = True
    enable_component_integration: bool = field(
        default=DEFAULT_ENABLE_COMPONENT_INTEGRATION
    )
    enable_reproducibility_validation: bool = field(default=True)
    # Backward-compat aliases expected by some tests
    enable_reproducibility_tracking: bool = field(default=True)
    episode_timeout_ms: float = field(default=30_000.0)
    component_coordination_timeout: float = field(
        default=COMPONENT_COORDINATION_TIMEOUT
    )
    episode_cache_size: int = field(default=EPISODE_PROCESSING_CACHE_SIZE)
    component_configs: Dict[str, object] = field(default_factory=dict)
    custom_parameters: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Initialize episode manager configuration with environment parameters, component settings,
        and performance options for comprehensive episode orchestration.
        """
        # Validate env_config using EnvironmentConfig.validate() for comprehensive environment parameter checking
        if not isinstance(self.env_config, EnvironmentConfig):
            raise ValidationError(
                message="env_config must be EnvironmentConfig instance",
                parameter_name="env_config",
                parameter_value=type(self.env_config).__name__,
                expected_format="EnvironmentConfig dataclass",
            )

        self.env_config.validate()

        # Set enable_performance_monitoring for episode-level timing and resource usage monitoring
        if not isinstance(self.enable_performance_monitoring, bool):
            self.enable_performance_monitoring = DEFAULT_ENABLE_PERFORMANCE_MONITORING

        # Set enable_state_validation for cross-component state consistency checking and error detection
        if not isinstance(self.enable_state_validation, bool):
            self.enable_state_validation = DEFAULT_ENABLE_STATE_VALIDATION

        # Enable component integration by default for seamless cross-component coordination
        if not isinstance(self.enable_component_integration, bool):
            self.enable_component_integration = DEFAULT_ENABLE_COMPONENT_INTEGRATION

        # Enable reproducibility validation by default for deterministic episode behavior verification
        if not isinstance(self.enable_reproducibility_validation, bool):
            self.enable_reproducibility_validation = True

        # Keep tracking alias in sync
        if not isinstance(self.enable_reproducibility_tracking, bool):
            self.enable_reproducibility_tracking = (
                self.enable_reproducibility_validation
            )
        else:
            # Mirror to validation flag when explicitly provided
            self.enable_reproducibility_validation = (
                self.enable_reproducibility_tracking
            )

        # Set component_coordination_timeout to COMPONENT_COORDINATION_TIMEOUT for dependency management
        if (
            not isinstance(self.component_coordination_timeout, (int, float))
            or self.component_coordination_timeout <= 0
        ):
            self.component_coordination_timeout = COMPONENT_COORDINATION_TIMEOUT

        # Initialize episode_cache_size to EPISODE_PROCESSING_CACHE_SIZE for performance optimization
        if not isinstance(self.episode_cache_size, int) or self.episode_cache_size < 0:
            self.episode_cache_size = EPISODE_PROCESSING_CACHE_SIZE

        # Normalize episode timeout value (milliseconds, positive)
        if (
            not isinstance(self.episode_timeout_ms, (int, float))
            or self.episode_timeout_ms <= 0
        ):
            self.episode_timeout_ms = 30_000.0

        # Initialize empty component_configs dictionary for component-specific configuration storage
        if not isinstance(self.component_configs, dict):
            self.component_configs = {}

        # Initialize empty custom_parameters dictionary for extensibility and episode-specific configuration
        if not isinstance(self.custom_parameters, dict):
            self.custom_parameters = {}

        # Cross-validate all parameters for episode management feasibility and component compatibility
        self._validate_parameter_compatibility()

    def _validate_parameter_compatibility(self):
        """Validate cross-parameter compatibility for episode management feasibility."""
        # Validate timeout settings
        if self.component_coordination_timeout > 60.0:
            raise ValidationError(
                message="component_coordination_timeout too high, may cause performance issues",
                parameter_name="component_coordination_timeout",
                parameter_value=str(self.component_coordination_timeout),
                expected_format="<=60.0 seconds",
            )

        # Validate cache size is reasonable
        if self.episode_cache_size > 1000:
            raise ValidationError(
                message="episode_cache_size too high, may cause memory issues",
                parameter_name="episode_cache_size",
                parameter_value=str(self.episode_cache_size),
                expected_format="<=1000 entries",
            )

    def validate(
        self,
        strict_mode: bool = False,
        validation_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Comprehensive validation of all episode manager configuration parameters with cross-component
        consistency checking and performance analysis.
        """
        try:
            # Validate env_config using EnvironmentConfig.validate() with comprehensive parameter checking
            self.env_config.validate()

            # Check component configuration compatibility between component_configs and environment requirements
            if self.component_configs:
                for component_name, component_config in self.component_configs.items():
                    if not isinstance(component_name, str):
                        raise ValidationError(
                            message=f"Component name must be string: {component_name}",
                            parameter_name="component_configs",
                            parameter_value=str(type(component_name)),
                            expected_format="string component names",
                        )

            # Validate performance monitoring and state validation configuration consistency
            if strict_mode:
                if (
                    not self.enable_state_validation
                    and self.enable_reproducibility_validation
                ):
                    raise ValidationError(
                        message="reproducibility validation requires state validation",
                        parameter_name="validation_configuration",
                        parameter_value="enable_state_validation=False with enable_reproducibility_validation=True",
                        expected_format="consistent validation settings",
                    )

            # Ensure component integration settings are compatible with system architecture requirements
            if self.enable_component_integration and not self.enable_state_validation:
                raise ValidationError(
                    message="component integration requires state validation",
                    parameter_name="integration_configuration",
                    parameter_value="enable_component_integration=True with enable_state_validation=False",
                    expected_format="consistent component settings",
                )

            # Check custom_parameters for valid keys, value types, and episode management compatibility
            if self.custom_parameters:
                for param_name, param_value in self.custom_parameters.items():
                    if not isinstance(param_name, str):
                        raise ValidationError(
                            message=f"Custom parameter name must be string: {param_name}",
                            parameter_name="custom_parameters",
                            parameter_value=str(type(param_name)),
                            expected_format="string parameter names",
                        )

            # Apply strict validation rules if strict_mode enabled including enhanced precision checking
            if strict_mode:
                # Additional validation for production deployment
                max_steps = self.env_config.max_steps
                if max_steps > 50000:
                    raise ValidationError(
                        message="max_steps too high for strict validation",
                        parameter_name="env_config.max_steps",
                        parameter_value=str(max_steps),
                        expected_format="<=50000 in strict mode",
                    )

            # Cross-validate all parameters for episode orchestration feasibility and resource constraints
            self._validate_parameter_compatibility()

            # Validate timeout and cache settings for reasonable performance and memory usage
            # Already handled in _validate_parameter_compatibility

            # Check reproducibility validation configuration for deterministic behavior requirements
            # Already validated above

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

    def derive_component_configs(
        self, validate_derived_configs: bool = True
    ) -> Dict[str, Any]:
        """
        Derive individual component configurations from episode manager configuration ensuring
        parameter consistency and integration compatibility.
        """
        try:
            # Create StateManagerConfig from env_config parameters with grid size, source location, and episode limits
            state_manager_config = StateManagerConfig(
                grid_size=cast(GridSize, self.env_config.grid_size),
                source_location=cast(Coordinates, self.env_config.source_location),
                max_steps=self.env_config.max_steps,
                goal_radius=self.env_config.goal_radius,
                enable_performance_monitoring=self.enable_performance_monitoring,
                enable_state_validation=self.enable_state_validation,
            )

            # Generate RewardCalculatorConfig using goal radius and reward structure from environment configuration
            from .reward_calculator import RewardCalculatorConfig

            reward_calculator_config = RewardCalculatorConfig(
                goal_radius=self.env_config.goal_radius,
                reward_goal_reached=1.0,  # Standard reward for goal achievement
                reward_default=0.0,  # Sparse reward structure
                enable_performance_monitoring=self.enable_performance_monitoring,
            )

            # Create ActionProcessingConfig with boundary enforcement and validation settings
            from .action_processor import ActionProcessingConfig

            action_processor_config = ActionProcessingConfig(
                enable_validation=self.enable_state_validation,
                enforce_boundaries=True,
                enable_performance_monitoring=self.enable_performance_monitoring,
                strict_validation=self.enable_state_validation,
            )

            # Include performance monitoring settings in all component configurations
            # Internal configs (objects) for use by EpisodeManager
            component_configs = {
                "StateManager": state_manager_config,
                "RewardCalculator": reward_calculator_config,
                "ActionProcessor": action_processor_config,
            }

            # Public, test-friendly view (dicts) for validation in tests
            public_configs: Dict[str, Any] = {
                "state_manager": {
                    "grid_size": state_manager_config.grid_size,
                    "source_location": state_manager_config.source_location,
                    "max_steps": state_manager_config.max_steps,
                    "goal_radius": state_manager_config.goal_radius,
                    "enabled": True,
                },
                "reward_calculator": {
                    "goal_radius": reward_calculator_config.goal_radius,
                    "reward_goal_reached": reward_calculator_config.reward_goal_reached,
                    "reward_default": reward_calculator_config.reward_default,
                    "enabled": True,
                },
                "action_processor": {
                    "enable_validation": action_processor_config.enable_validation,
                    "enforce_boundaries": action_processor_config.enforce_boundaries,
                    "strict_validation": action_processor_config.strict_validation,
                    "enabled": True,
                },
            }

            # Add custom component parameters from component_configs if specified
            for component_name, custom_config in self.component_configs.items():
                if component_name in component_configs:
                    # Merge custom configuration with derived configuration
                    if hasattr(component_configs[component_name], "__dict__"):
                        for key, value in custom_config.items():
                            if hasattr(component_configs[component_name], key):
                                setattr(component_configs[component_name], key, value)

            # Validate derived configurations if validate_derived_configs enabled using component validation
            if validate_derived_configs:
                try:
                    state_manager_config.validate()
                    reward_calculator_config.validate()
                    action_processor_config.validate_configuration()
                except Exception as e:
                    raise ValidationError(
                        message=f"Derived component configuration validation failed: {str(e)}",
                        parameter_name="derived_configs",
                        parameter_value="component configurations",
                        expected_format="valid component configurations",
                    ) from e

            # Ensure cross-component parameter consistency and integration compatibility
            # This is implicitly ensured by deriving from the same env_config

            # Return combined mapping: lowercase dicts for tests + internal object configs
            public_configs.update(component_configs)
            return public_configs

        except Exception as e:
            raise ComponentError(
                message=f"Failed to derive component configurations: {str(e)}",
                component_name="EpisodeManagerConfig",
                operation_name="derive_component_configs",
            ) from e

    def estimate_episode_resources(
        self,
        expected_episode_length: int = 1000,
        include_component_overhead: bool = True,
    ) -> Dict[str, Any]:
        """
        Estimate computational and memory resources required for episode processing including
        all component overhead and coordination costs.
        """
        try:
            # Estimate base episode management overhead including coordination and state tracking
            base_overhead_mb = 5.0  # Base EpisodeManager memory

            # Calculate component resource requirements using env_config parameters
            grid_cells = (
                self.env_config.grid_size.width * self.env_config.grid_size.height
            )
            grid_memory_mb = (grid_cells * 4) / (1024 * 1024)  # 4 bytes per float32

            # Project episode processing time based on expected_episode_length and performance targets
            target_step_latency = (
                PERFORMANCE_TARGET_STEP_LATENCY_MS / 1000
            )  # Convert to seconds
            estimated_episode_duration_s = expected_episode_length * target_step_latency

            # Include component coordination overhead if include_component_overhead is True
            component_overhead_mb = 0
            if include_component_overhead:
                component_overhead_mb = (
                    10.0  # StateManager, RewardCalculator, ActionProcessor
                )

            # Estimate memory usage for state management, caching, and component coordination
            cache_memory_mb = (
                self.episode_cache_size * 1
            ) / 1024  # 1KB per cached item estimate

            # Calculate performance monitoring overhead if enable_performance_monitoring is True
            monitoring_overhead_mb = 0
            if self.enable_performance_monitoring:
                monitoring_overhead_mb = 2.0  # PerformanceMetrics storage

            total_memory_mb = (
                base_overhead_mb
                + grid_memory_mb
                + component_overhead_mb
                + cache_memory_mb
                + monitoring_overhead_mb
            )

            # Generate resource optimization recommendations based on configuration and usage patterns
            recommendations = []
            if total_memory_mb > 100:
                recommendations.append(
                    "Consider reducing grid size for better memory efficiency"
                )
            if expected_episode_length > 10000:
                recommendations.append(
                    "Long episodes may impact performance - consider episode length optimization"
                )
            if self.episode_cache_size > 50:
                recommendations.append(
                    "Large cache size may not provide significant benefits"
                )

            # Return comprehensive resource estimation with capacity planning and optimization guidance
            return {
                "memory_usage_mb": {
                    "base_episode_manager": base_overhead_mb,
                    "grid_storage": grid_memory_mb,
                    "component_overhead": component_overhead_mb,
                    "cache_storage": cache_memory_mb,
                    "monitoring_overhead": monitoring_overhead_mb,
                    "total_estimated": total_memory_mb,
                },
                "performance_estimates": {
                    "expected_episode_duration_s": estimated_episode_duration_s,
                    "target_step_latency_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS,
                    "estimated_steps_per_second": 1000
                    / PERFORMANCE_TARGET_STEP_LATENCY_MS,
                },
                "optimization_recommendations": recommendations,
                "resource_feasibility": {
                    "memory_reasonable": total_memory_mb < 500,
                    "performance_achievable": estimated_episode_duration_s < 60,
                    "configuration_optimal": len(recommendations) == 0,
                },
            }

        except Exception as e:
            return {
                "error": f"Resource estimation failed: {str(e)}",
                "memory_usage_mb": {"total_estimated": 0},
                "performance_estimates": {},
                "optimization_recommendations": [
                    "Fix configuration errors before resource analysis"
                ],
            }

    def clone(
        self,
        overrides: Optional[Dict[str, Any]] = None,
        preserve_component_configs: bool = True,
    ) -> "EpisodeManagerConfig":
        """
        Create deep copy of episode manager configuration with optional parameter overrides
        for testing and experimentation.
        """
        try:
            # Create deep copy of current configuration including env_config and all component settings
            cloned_env_config = copy.deepcopy(self.env_config)
            cloned_component_configs = (
                self.component_configs.copy() if preserve_component_configs else {}
            )
            cloned_custom_parameters = self.custom_parameters.copy()

            # Create new configuration with cloned parameters
            cloned_config = EpisodeManagerConfig(
                env_config=cloned_env_config,
                enable_performance_monitoring=self.enable_performance_monitoring,
                enable_state_validation=self.enable_state_validation,
                enable_component_integration=self.enable_component_integration,
                enable_reproducibility_validation=self.enable_reproducibility_validation,
                component_coordination_timeout=self.component_coordination_timeout,
                episode_cache_size=self.episode_cache_size,
                component_configs=cloned_component_configs,
                custom_parameters=cloned_custom_parameters,
            )

            # Apply parameter overrides if provided with type validation and compatibility checking
            if overrides:
                for key, value in overrides.items():
                    if hasattr(cloned_config, key):
                        setattr(cloned_config, key, value)
                    else:
                        # Add to custom_parameters if not a standard parameter
                        cloned_config.custom_parameters[key] = value

            # Preserve component_configs if preserve_component_configs is True, otherwise reset to defaults
            # Already handled above

            # Update component integration and monitoring settings with override values if specified
            # Already handled above

            # Validate cloned configuration with new parameters using comprehensive validate method
            cloned_config.validate()

            # Ensure override parameters maintain cross-component compatibility and system integration
            # Validation handles this

            # Return new EpisodeManagerConfig instance with overrides applied and validation completed
            return cloned_config

        except Exception as e:
            raise ComponentError(
                message=f"Failed to clone episode manager configuration: {str(e)}",
                component_name="EpisodeManagerConfig",
                operation_name="clone",
            ) from e


@dataclass
class EpisodeResult:
    """
    Comprehensive data class containing complete episode execution results including final states,
    performance metrics, component statistics, and reproducibility information for analysis and debugging.
    """

    # Required parameters for episode result creation
    episode_id: str
    terminated: bool
    truncated: bool
    total_steps: int

    # Optional result information with defaults
    total_reward: float = field(default=0.0)
    episode_duration_ms: float = field(default=0.0)
    final_agent_position: Optional[Coordinates] = field(default=None)
    final_distance_to_goal: Optional[float] = field(default=None)
    performance_metrics: Dict[str, object] = field(default_factory=dict)
    component_statistics: Dict[str, object] = field(default_factory=dict)
    state_snapshots: List[StateSnapshot] = field(default_factory=list)
    termination_reason: Optional[str] = field(default=None)

    def __post_init__(self):
        """
        Initialize episode result with basic episode information and completion status.
        """
        # Store episode_id for unique episode identification and tracking
        if not self.episode_id or not isinstance(self.episode_id, str):
            raise ValidationError(
                message="episode_id must be non-empty string",
                parameter_name="episode_id",
                parameter_value=str(self.episode_id),
                expected_format="non-empty string identifier",
            )

        # Set terminated and truncated flags for episode completion analysis
        # Clamp invalid numeric values to safe defaults
        if self.total_steps < 0:
            self.total_steps = 0
        if self.episode_duration_ms < 0:
            self.episode_duration_ms = 0.0

        # Initialize final_agent_position to None for later population with episode end state
        # Already handled by field default

        # Set final_distance_to_goal to None for goal achievement analysis
        # Already handled by field default

        # Initialize empty performance_metrics dictionary for timing and resource analysis
        # Ensure containers are dict/list instances
        if not isinstance(self.performance_metrics, dict):
            self.performance_metrics = {}
        if not isinstance(self.component_statistics, dict):
            self.component_statistics = {}
        if not isinstance(self.state_snapshots, list):
            self.state_snapshots = []

        # Set termination_reason to None for later population with completion analysis
        # Already handled by field default

    def set_final_state(
        self,
        final_agent_state: AgentState,
        final_distance: float | None = None,
        termination_reason: str | None = None,
        *,
        # Backward-compat alias used by some tests
        distance_to_goal: float | None = None,
    ) -> None:
        """
        Set final episode state information for comprehensive episode completion analysis.
        """
        try:
            # Extract final_agent_position from final_agent_state.position
            if hasattr(final_agent_state, "position") and isinstance(
                final_agent_state.position, Coordinates
            ):
                self.final_agent_position = final_agent_state.position

            # Set total_reward from final_agent_state.total_reward for episode reward analysis
            if hasattr(final_agent_state, "total_reward"):
                self.total_reward = float(final_agent_state.total_reward)

            # Store final_distance_to_goal for goal achievement and performance evaluation
            if distance_to_goal is None:
                candidate = final_distance
            else:
                candidate = distance_to_goal

            if isinstance(candidate, (int, float)) and candidate >= 0:
                self.final_distance_to_goal = float(candidate)

            # Set termination_reason for detailed episode completion analysis
            if isinstance(termination_reason, str) and termination_reason.strip():
                self.termination_reason = termination_reason

            # Update episode result with comprehensive final state information for analysis
            # All updates completed above

        except Exception as e:
            raise ComponentError(
                message=f"Failed to set final episode state: {str(e)}",
                component_name="EpisodeResult",
                operation_name="set_final_state",
            ) from e

    # Backward-compat: tests expect `duration_ms`; keep internal field name
    # `episode_duration_ms` but expose a read-only property alias.
    @property
    def duration_ms(self) -> float:
        return float(self.episode_duration_ms)

    def add_performance_metrics(
        self, episode_metrics: Dict[str, Any], component_metrics: Dict[str, Any]
    ) -> None:
        """
        Add performance metrics from episode execution including timing analysis and resource usage.
        """
        try:
            # Merge episode_metrics into performance_metrics with timing and resource data
            if isinstance(episode_metrics, dict):
                self.performance_metrics.update(episode_metrics)
                # Normalize step timing list if available from metrics summary
                timings = episode_metrics.get("timings")
                if isinstance(timings, dict):
                    step_series = timings.get("episode_step")
                    if isinstance(step_series, list):
                        self.performance_metrics["step_timing"] = list(step_series)

            # Update component_statistics with component_metrics for detailed component analysis
            if isinstance(component_metrics, dict):
                self.component_statistics.update(component_metrics)

            # Calculate performance ratios and efficiency metrics from collected data
            if (
                "total_step_time_ms" in self.performance_metrics
                and self.total_steps > 0
            ):
                avg_step_time = (
                    self.performance_metrics["total_step_time_ms"] / self.total_steps
                )
                self.performance_metrics["average_step_time_ms"] = avg_step_time
                self.performance_metrics["step_efficiency"] = (
                    PERFORMANCE_TARGET_STEP_LATENCY_MS / avg_step_time
                )

            # Generate performance summary and optimization recommendations
            if "average_step_time_ms" in self.performance_metrics:
                if (
                    self.performance_metrics["average_step_time_ms"]
                    > PERFORMANCE_TARGET_STEP_LATENCY_MS
                ):
                    self.performance_metrics["performance_warning"] = (
                        "Step latency exceeded target"
                    )

        except Exception:
            # Log error but don't raise to avoid disrupting episode completion
            pass

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return collected performance metrics and component statistics.

        The structure intentionally mirrors what tests assert:
        - top-level episode metrics (e.g., total_duration_ms, average_step_time_ms ...)
        - nested component metrics under 'component_metrics'
        """
        data = dict(self.performance_metrics)
        # Keep component stats separate to avoid key collisions
        data["component_metrics"] = dict(self.component_statistics)
        return data

    def get_summary(
        self, *, include_performance: bool = False, include_components: bool = False
    ) -> Dict[str, Any]:
        """Return a human-friendly summary dictionary for reporting/tests."""
        summary: Dict[str, Any] = {
            "episode_id": self.episode_id,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "total_steps": self.total_steps,
            "total_reward": self.total_reward,
            "final_position": (
                self.final_agent_position.to_tuple()
                if isinstance(self.final_agent_position, Coordinates)
                else None
            ),
            "final_distance_to_goal": self.final_distance_to_goal,
        }

        if include_performance:
            summary["performance"] = self.get_performance_metrics()
        if include_components:
            summary["components"] = dict(self.component_statistics)
        return summary

    def add_state_snapshot(self, snapshot: StateSnapshot) -> None:
        """
        Add state snapshot to episode result for replay and debugging support.
        """
        try:
            # Validate snapshot is proper StateSnapshot with timestamp and state information
            if not isinstance(snapshot, StateSnapshot):
                raise ValidationError(
                    message="snapshot must be StateSnapshot instance",
                    parameter_name="snapshot",
                    parameter_value=type(snapshot).__name__,
                    expected_format="StateSnapshot dataclass",
                )

            # Append snapshot to state_snapshots list for episode replay capability
            self.state_snapshots.append(snapshot)

            # Manage state_snapshots list size to prevent excessive memory usage
            if len(self.state_snapshots) > 1000:  # Reasonable limit
                # Remove oldest snapshots to maintain memory bounds
                self.state_snapshots = self.state_snapshots[
                    -500:
                ]  # Keep most recent 500

        except Exception:
            # Log error but don't raise to avoid disrupting episode execution
            pass

    @property
    def duration_ms(self) -> float:
        """Compatibility alias for episode duration in milliseconds."""
        return float(self.episode_duration_ms)

    def get_summary(
        self,
        include_performance_analysis: bool = False,
        include_component_details: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive episode summary for analysis, logging, and reporting.
        """
        try:
            # Compile basic episode information including ID, completion status, and step count
            summary = {
                "episode_id": self.episode_id,
                "completion_status": {
                    "terminated": self.terminated,
                    "truncated": self.truncated,
                    "total_steps": self.total_steps,
                    "episode_duration_ms": self.episode_duration_ms,
                },
                "reward_analysis": {
                    "total_reward": self.total_reward,
                    "average_reward_per_step": self.total_reward
                    / max(1, self.total_steps),
                },
            }

            # Include final position, distance to goal, and total reward for outcome analysis
            if self.final_agent_position:
                summary["final_state"] = {
                    "agent_position": (
                        self.final_agent_position.x,
                        self.final_agent_position.y,
                    ),
                    "distance_to_goal": self.final_distance_to_goal,
                    "termination_reason": self.termination_reason,
                }

            # Add performance analysis with timing metrics if include_performance_analysis is True
            if include_performance_analysis and self.performance_metrics:
                summary["performance_analysis"] = self.performance_metrics.copy()

            # Include detailed component statistics if include_component_details is True
            if include_component_details and self.component_statistics:
                summary["component_details"] = self.component_statistics.copy()

            # Generate episode success analysis and goal achievement assessment
            summary["success_analysis"] = {
                "goal_achieved": self.terminated,
                "episode_completed_naturally": self.terminated or self.truncated,
                "efficiency_score": self._calculate_efficiency_score(),
            }

            # Return comprehensive episode summary for monitoring and research analysis
            return summary

        except Exception as e:
            # Return basic summary on error
            return {
                "episode_id": self.episode_id,
                "error": f"Summary generation failed: {str(e)}",
                "basic_stats": {
                    "terminated": self.terminated,
                    "truncated": self.truncated,
                    "total_steps": self.total_steps,
                    "total_reward": self.total_reward,
                },
            }

    def _calculate_efficiency_score(self) -> float:
        """Calculate episode efficiency score based on goal achievement and step count."""
        try:
            if self.terminated:  # Goal achieved
                # Lower step count = higher efficiency
                return max(
                    0.0, 1.0 - (self.total_steps / 10000)
                )  # Normalize by reasonable max steps
            else:
                # Episode failed or truncated
                return 0.0
        except Exception:
            return 0.0


@dataclass
class EpisodeStatistics:
    """
    Data class for collecting and analyzing episode-level statistics including performance trends,
    success rates, component efficiency, and system optimization recommendations for multi-episode analysis.
    """

    # Required parameter
    statistics_id: str

    # Statistics tracking with defaults
    episodes_completed: int = field(default=0)
    episodes_terminated: int = field(default=0)
    episodes_truncated: int = field(default=0)
    average_episode_length: float = field(default=0.0)
    average_episode_duration_ms: float = field(default=0.0)
    average_total_reward: float = field(default=0.0)
    performance_trends: Dict[str, object] = field(default_factory=dict)
    component_efficiency: Dict[str, object] = field(default_factory=dict)
    episode_results: List[EpisodeResult] = field(default_factory=list)

    def __post_init__(self):
        """
        Initialize episode statistics collection with unique identifier for tracking.
        """
        # Store statistics_id for unique statistics collection identification
        if not self.statistics_id or not isinstance(self.statistics_id, str):
            self.statistics_id = f"stats_{int(time.time() * 1000)}"

        # Initialize episode counters to 0 for completion, termination, and truncation tracking
        # Already handled by field defaults

        # Set average statistics to 0.0 for incremental calculation as episodes complete
        # Already handled by field defaults

        # Initialize empty performance_trends dictionary for timing and resource analysis
        if not isinstance(self.performance_trends, dict):
            self.performance_trends = {}

        # Initialize empty component_efficiency dictionary for component performance tracking
        if not isinstance(self.component_efficiency, dict):
            self.component_efficiency = {}

        # Initialize empty episode_results list for detailed episode result storage
        if not isinstance(self.episode_results, list):
            self.episode_results = []

    def add_episode_result(self, episode_result: EpisodeResult) -> None:
        """
        Add completed episode result to statistics collection with trend analysis.
        """
        try:
            # Append episode_result to episode_results list for detailed tracking
            if isinstance(episode_result, EpisodeResult):
                self.episode_results.append(episode_result)
            else:
                raise ValidationError(
                    message="episode_result must be EpisodeResult instance",
                    parameter_name="episode_result",
                    parameter_value=type(episode_result).__name__,
                    expected_format="EpisodeResult dataclass",
                )

            # Increment episodes_completed counter for total episode tracking
            self.episodes_completed += 1

            # Update termination/truncation counters based on episode outcome
            if episode_result.terminated:
                self.episodes_terminated += 1
            if episode_result.truncated:
                self.episodes_truncated += 1

            # Recalculate running averages for episode length, duration, and reward
            self._update_running_averages(episode_result)

            # Update performance_trends with episode timing and efficiency data
            self._update_performance_trends(episode_result)

            # Update component_efficiency with component-specific performance metrics
            self._update_component_efficiency(episode_result)

        except Exception as e:
            raise ComponentError(
                message=f"Failed to add episode result: {str(e)}",
                component_name="EpisodeStatistics",
                operation_name="add_episode_result",
            ) from e

    def _update_running_averages(self, episode_result: EpisodeResult) -> None:
        """Update running averages with new episode data."""
        n = self.episodes_completed
        if n <= 1:
            self.average_episode_length = float(episode_result.total_steps)
            self.average_episode_duration_ms = episode_result.episode_duration_ms
            self.average_total_reward = episode_result.total_reward
        else:
            # Incremental average calculation
            self.average_episode_length = (
                (self.average_episode_length * (n - 1)) + episode_result.total_steps
            ) / n
            self.average_episode_duration_ms = (
                (self.average_episode_duration_ms * (n - 1))
                + episode_result.episode_duration_ms
            ) / n
            self.average_total_reward = (
                (self.average_total_reward * (n - 1)) + episode_result.total_reward
            ) / n

    def _update_performance_trends(self, episode_result: EpisodeResult) -> None:
        """Update performance trends with episode data."""
        try:
            if not self.performance_trends.get("step_times"):
                self.performance_trends["step_times"] = []
            if not self.performance_trends.get("episode_durations"):
                self.performance_trends["episode_durations"] = []

            # Add latest performance data
            if episode_result.total_steps > 0:
                avg_step_time = (
                    episode_result.episode_duration_ms / episode_result.total_steps
                )
                self.performance_trends["step_times"].append(avg_step_time)

            self.performance_trends["episode_durations"].append(
                episode_result.episode_duration_ms
            )

            # Maintain reasonable list sizes
            for key in ["step_times", "episode_durations"]:
                if len(self.performance_trends[key]) > 100:
                    self.performance_trends[key] = self.performance_trends[key][-50:]
        except Exception:
            pass  # Don't fail statistics on performance trend errors

    def _update_component_efficiency(self, episode_result: EpisodeResult) -> None:
        """Update component efficiency metrics."""
        try:
            if episode_result.component_statistics:
                for component, stats in episode_result.component_statistics.items():
                    if component not in self.component_efficiency:
                        self.component_efficiency[component] = {
                            "calls": 0,
                            "total_time": 0.0,
                        }

                    if isinstance(stats, dict):
                        self.component_efficiency[component]["calls"] += stats.get(
                            "calls", 1
                        )
                        self.component_efficiency[component]["total_time"] += stats.get(
                            "time", 0.0
                        )
        except Exception:
            pass  # Don't fail statistics on component efficiency errors

    def calculate_success_rate(self) -> float:
        """
        Calculate episode success rate based on goal achievement and completion status.
        """
        try:
            # Calculate success rate as episodes_terminated / episodes_completed ratio
            if self.episodes_completed == 0:
                return 0.0

            # Handle division by zero case when no episodes completed
            success_rate = self.episodes_terminated / self.episodes_completed

            # Return success percentage for goal achievement analysis
            return success_rate

        except Exception:
            # Return 0.0 on any calculation errors
            return 0.0

    def get_performance_summary(
        self,
        include_trend_analysis: bool = True,
        include_optimization_recommendations: bool = True,
    ) -> Dict[str, object]:
        """
        Generate comprehensive performance summary with trends and optimization recommendations.
        """
        try:
            # Compile basic performance statistics including averages and success rates
            summary: Dict[str, object] = {
                "episode_counts": {
                    "total_completed": self.episodes_completed,
                    "terminated": self.episodes_terminated,
                    "truncated": self.episodes_truncated,
                    "success_rate": self.calculate_success_rate(),
                },
                "average_metrics": {
                    "episode_length": self.average_episode_length,
                    "episode_duration_ms": self.average_episode_duration_ms,
                    "total_reward": self.average_total_reward,
                },
                "performance_indicators": {
                    "average_step_latency_ms": self.average_episode_duration_ms
                    / max(1, self.average_episode_length),
                    "episodes_per_second": 1000
                    / max(1, self.average_episode_duration_ms),
                },
            }

            # Include performance trend analysis if include_trend_analysis is True
            if include_trend_analysis and self.performance_trends:
                trend_analysis = {}

                if (
                    "step_times" in self.performance_trends
                    and self.performance_trends["step_times"]
                ):
                    step_times = self.performance_trends["step_times"]
                    trend_analysis["step_time_trends"] = {
                        "recent_average_ms": (
                            sum(step_times[-10:]) / len(step_times[-10:])
                            if step_times
                            else 0
                        ),
                        "performance_improving": len(step_times) > 1
                        and step_times[-1] < step_times[0],
                        "target_compliance_rate": sum(
                            1
                            for t in step_times
                            if t <= PERFORMANCE_TARGET_STEP_LATENCY_MS
                        )
                        / len(step_times),
                    }

                summary["trend_analysis"] = trend_analysis

            # Add component efficiency analysis and bottleneck identification
            if self.component_efficiency:
                efficiency_analysis = {}
                for component, metrics in self.component_efficiency.items():
                    if metrics["calls"] > 0:
                        avg_time = metrics["total_time"] / metrics["calls"]
                        efficiency_analysis[component] = {
                            "average_call_time_ms": avg_time,
                            "total_calls": metrics["calls"],
                            "performance_rating": (
                                "good" if avg_time < 1.0 else "needs_optimization"
                            ),
                        }

                summary["component_efficiency"] = efficiency_analysis

            # Generate optimization recommendations if include_optimization_recommendations is True
            if include_optimization_recommendations:
                recommendations = []

                # Performance-based recommendations
                avg_step_time = summary["performance_indicators"][
                    "average_step_latency_ms"
                ]
                if avg_step_time > PERFORMANCE_TARGET_STEP_LATENCY_MS:
                    recommendations.append(
                        f"Step latency ({avg_step_time:.2f}ms) exceeds target ({PERFORMANCE_TARGET_STEP_LATENCY_MS}ms)"
                    )

                # Success rate recommendations
                success_rate = summary["episode_counts"]["success_rate"]
                if success_rate < 0.1:
                    recommendations.append(
                        "Low success rate suggests difficulty tuning needed"
                    )
                elif success_rate > 0.9:
                    recommendations.append(
                        "High success rate suggests environment may be too easy"
                    )

                # Episode length recommendations
                if self.average_episode_length < 10:
                    recommendations.append(
                        "Very short episodes may indicate premature termination"
                    )
                elif self.average_episode_length > 5000:
                    recommendations.append(
                        "Long episodes may impact training efficiency"
                    )

                summary["optimization_recommendations"] = recommendations

            # Return comprehensive performance summary for system monitoring and improvement
            return summary

        except Exception as e:
            # Return basic summary on error
            return {
                "error": f"Performance summary generation failed: {str(e)}",
                "basic_stats": {
                    "episodes_completed": self.episodes_completed,
                    "success_rate": self.calculate_success_rate(),
                },
            }


class EpisodeManager:
    """
    Central episode management orchestrator coordinating complete episode lifecycle including
    initialization, step processing, termination, and cleanup with comprehensive component
    integration, performance monitoring, and Gymnasium API compliance for reinforcement learning
    environment implementation.
    """

    def __init__(
        self, config: EpisodeManagerConfig, seed_manager: Optional[SeedManager] = None
    ):
        """
        Initialize episode manager with configuration validation, component coordination setup,
        and performance monitoring for comprehensive episode orchestration.
        """
        try:
            # Validate configuration using config.validate() with comprehensive parameter checking
            if not isinstance(config, EpisodeManagerConfig):
                raise ValidationError(
                    message="config must be EpisodeManagerConfig instance",
                    parameter_name="config",
                    parameter_value=type(config).__name__,
                    expected_format="EpisodeManagerConfig dataclass",
                )

            config.validate()

            # Store configuration for episode manager parameter access and component coordination
            self.config = config

            # Initialize or store provided seed_manager for reproducible episode generation
            self.seed_manager = (
                seed_manager if seed_manager is not None else SeedManager()
            )

            # Derive component configurations for state manager, reward calculator, action processor
            component_configs = self.config.derive_component_configs()

            # Create StateManager with derived StateManagerConfig for centralized state coordination
            self.state_manager = StateManager(
                config=component_configs["StateManager"], seed_manager=self.seed_manager
            )

            # Initialize RewardCalculator with derived RewardCalculatorConfig for goal-based rewards
            # RewardCalculator expects the lightweight PerformanceMetrics from core.types
            self.reward_calculator = RewardCalculator(
                config=component_configs["RewardCalculator"],
                performance_metrics=(
                    PerformanceMetrics()
                    if self.config.enable_performance_monitoring
                    else None
                ),
            )

            # Create ActionProcessor with derived ActionProcessingConfig for movement processing
            self.action_processor = ActionProcessor(
                grid_size=cast(GridSize, self.config.env_config.grid_size),
                config=component_configs["ActionProcessor"],
            )

            # Set current_episode_state to None for initial inactive state
            self.current_episode_state: Optional[EpisodeState] = None

            # Create PerformanceMetrics instance for episode-level timing and resource monitoring
            self.performance_metrics = PerformanceMetrics()

            # Initialize component logger for episode management operations and debugging
            self.logger = get_component_logger("EpisodeManager")

            # Set episode_active to False and episode_count to 0 for initial tracking state
            self.episode_active: bool = False
            self.episode_count: int = 0

            # Initialize component_cache dictionary for cross-component data optimization
            self.component_cache: Dict[str, object] = {}

            # Create EpisodeStatistics instance for multi-episode analysis and trend tracking
            self.episode_statistics = EpisodeStatistics(
                statistics_id=f"episode_manager_{int(time.time() * 1000)}"
            )

            # Validate complete episode manager initialization and component integration consistency
            self._validate_component_integration()

            self.logger.info(
                f"EpisodeManager initialized successfully with {len(component_configs)} components"
            )

        except Exception as e:
            raise ComponentError(
                message=f"EpisodeManager initialization failed: {str(e)}",
                component_name="EpisodeManager",
                operation_name="__init__",
            ) from e

    def _validate_component_integration(self) -> None:
        """Validate that all components are properly integrated and compatible."""
        try:
            # Check that all components are initialized
            if not self.state_manager:
                raise ComponentError(
                    "StateManager not initialized",
                    "EpisodeManager",
                    "component_validation",
                )
            if not self.reward_calculator:
                raise ComponentError(
                    "RewardCalculator not initialized",
                    "EpisodeManager",
                    "component_validation",
                )
            if not self.action_processor:
                raise ComponentError(
                    "ActionProcessor not initialized",
                    "EpisodeManager",
                    "component_validation",
                )

            # Validate component configuration compatibility
            state_config = self.state_manager.config

            # Check grid size consistency between components
            if (
                state_config.grid_size.width != self.action_processor.grid_size.width
                or state_config.grid_size.height
                != self.action_processor.grid_size.height
            ):
                raise ComponentError(
                    "Grid size mismatch between StateManager and ActionProcessor",
                    "EpisodeManager",
                    "component_validation",
                )

            # Check goal radius consistency
            if state_config.goal_radius != self.reward_calculator.config.goal_radius:
                raise ComponentError(
                    "Goal radius mismatch between StateManager and RewardCalculator",
                    "EpisodeManager",
                    "component_validation",
                )

        except Exception as e:
            if isinstance(e, ComponentError):
                raise
            raise ComponentError(
                f"Component integration validation failed: {str(e)}",
                "EpisodeManager",
                "component_validation",
            ) from e

    @monitor_performance("episode_reset", PERFORMANCE_TARGET_STEP_LATENCY_MS * 10, True)
    def reset_episode(
        self,
        seed: Optional[int] = None,
        episode_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray[np.float32], Dict[str, object]]:
        """
        Initialize new episode with agent placement, component coordination, and state setup for
        Gymnasium reset() method implementation with comprehensive reproducibility support.
        """
        try:
            # Start performance timing for episode reset operation with comprehensive monitoring
            reset_start_time = time.time()

            # Validate and apply seed using seed_manager for deterministic episode generation
            if seed is not None:
                self.seed_manager.seed(seed)
                self.logger.debug(f"Episode reset with seed: {seed}")
            else:
                self.logger.debug("Episode reset without seed (random)")

            # Clear previous episode state and component caches for clean episode initialization
            self.current_episode_state = None
            self.component_cache.clear()

            # Generate unique episode ID using uuid for tracking and reproducibility analysis
            episode_id = f"episode_{uuid.uuid4().hex[:8]}_{self.episode_count + 1}"

            # Coordinate episode reset across all components using state_manager.reset_episode()
            state_reset_info: object = self.state_manager.reset_episode(
                seed, episode_options
            )

            # Initialize agent start position excluding source location through component coordination
            # Already handled by state_manager.reset_episode()

            # Ensure we have an AgentState available
            agent_state = self.state_manager.current_agent_state
            if agent_state is None:
                raise StateError(
                    message="Cannot reset episode: missing agent state",
                    current_state="episode_reset",
                    expected_state="reset_initialization",
                )

            # Build a simple observation field matching grid shape (height, width)
            grid = self.config.env_config.grid_size
            observation = np.zeros((grid.height, grid.width), dtype=np.float32)

            # Create initial step info dictionary using create_step_info with agent state and metadata
            # Legacy-compatible info fields expected by tests
            source_location = cast(Coordinates, self.config.env_config.source_location)
            agent_xy = (agent_state.position.x, agent_state.position.y)
            distance_to_source = float(
                agent_state.position.distance_to(source_location)
            )

            info = create_step_info(
                agent_state=agent_state,
                additional_info={
                    "episode_id": episode_id,
                    "reset_info": state_reset_info,
                    "seed": seed,
                    "episode_options": episode_options,
                    "agent_xy": agent_xy,
                    "distance_to_source": distance_to_source,
                },
            )

            # Set episode_active to True and increment episode_count for state tracking
            self.episode_active = True
            self.episode_count += 1

            # Create new episode state for tracking
            self.current_episode_state = create_episode_state(
                agent_state=agent_state,
                terminated=False,
                truncated=False,
                episode_id=episode_id,
            )

            # Create initial state snapshot for reproducibility and debugging support
            if self.config.enable_state_validation:
                initial_snapshot = self.state_manager.create_state_snapshot(
                    snapshot_name=f"{episode_id}_initial", validate_consistency=True
                )

                # Store snapshot in component cache
                self.component_cache["initial_snapshot"] = initial_snapshot

            # Record episode reset timing in performance_metrics for optimization analysis
            reset_duration_ms = (time.time() - reset_start_time) * 1000
            self.performance_metrics.record_timing("episode_reset", reset_duration_ms)

            # Log episode reset completion with performance context and episode identification
            self.logger.info(
                f"Episode {self.episode_count} reset completed in {reset_duration_ms:.2f}ms"
            )

            # Return tuple of (observation, info) for Gymnasium API compliance and environment coordination
            return observation, info

        except Exception as e:
            self.logger.error(f"Episode reset failed: {e}")
            raise StateError(
                message=f"Episode reset failed: {str(e)}",
                current_state="episode_reset",
                expected_state="reset_initialization",
            ) from e

    @monitor_performance("episode_step", PERFORMANCE_TARGET_STEP_LATENCY_MS, False)
    def process_step(
        self, action: Action
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, object]]:
        """
        Process single environment step with action validation, component coordination, and
        comprehensive state updates for Gymnasium step() method implementation.
        """
        try:
            # Start performance timing for step processing operation with latency monitoring
            step_start_time = time.time()

            # Validate episode is active and current_episode_state is properly initialized
            if not self.episode_active:
                raise StateError(
                    message="Cannot process step: episode is not active",
                    current_state="inactive",
                    expected_state="process_step",
                )

            if self.current_episode_state is None:
                raise StateError(
                    message="Cannot process step: episode state not initialized",
                    current_state="uninitialized",
                    expected_state="process_step",
                )

            # Guard against None agent state for mypy strict typing
            agent_state = self.state_manager.current_agent_state
            if agent_state is None:
                raise StateError(
                    message="Cannot process step: missing agent state",
                    current_state="uninitialized",
                    expected_state="process_step",
                )

            # Process action using action_processor.process_action() with movement validation
            action_result: ActionProcessingResult = (
                self.action_processor.process_action(
                    action=action,
                    current_position=agent_state.position,
                )
            )

            # Update agent position through state_manager.process_step() with action result
            state_step_result: Dict[str, object] = self.state_manager.process_step(
                action, action_context={"action_result": action_result}
            )

            # Build observation field with grid shape
            grid = self.config.env_config.grid_size
            observation = np.zeros((grid.height, grid.width), dtype=np.float32)

            # Calculate reward using reward_calculator.calculate_reward() with goal detection
            current_position = agent_state.position
            # Use normalized source_location from EnvironmentConfig
            source_location = cast(Coordinates, self.config.env_config.source_location)

            reward_result: RewardResult = self.reward_calculator.calculate_reward(
                agent_position=current_position, source_location=source_location
            )

            # Check episode termination using reward_calculator.check_termination() with step limits
            termination_result = self.reward_calculator.check_termination(
                agent_state=agent_state,
                source_location=source_location,
                max_steps=self.config.env_config.max_steps,
            )

            # Update agent reward through reward_calculator.update_agent_reward() with state coordination
            self.reward_calculator.update_agent_reward(
                agent_state=agent_state,
                reward_result=reward_result,
            )

            # Update episode state termination flags if episode completed with detailed reason analysis
            terminated = termination_result.terminated
            truncated = termination_result.truncated

            if terminated or truncated:
                self.current_episode_state.set_termination(
                    terminated=terminated,
                    truncated=truncated,
                    reason=termination_result.termination_reason,
                )
                self.episode_active = False

            # Create comprehensive step info dictionary with agent state, performance, and debug information
            # Legacy-compatible extras
            source_location = cast(Coordinates, self.config.env_config.source_location)
            agent_xy = (agent_state.position.x, agent_state.position.y)
            distance_to_source = float(
                agent_state.position.distance_to(source_location)
            )

            info = create_step_info(
                agent_state=agent_state,
                additional_info={
                    "action_result": action_result.to_dict(),
                    "reward_result": reward_result.to_dict(),
                    "termination_result": termination_result.get_summary(),
                    "state_step_result": state_step_result,
                    "agent_xy": agent_xy,
                    "distance_to_source": distance_to_source,
                },
            )

            # Create state snapshot if debugging enabled for episode replay and analysis
            if self.config.enable_state_validation:
                step_snapshot = self.state_manager.create_state_snapshot(
                    snapshot_name=f"step_{agent_state.step_count}",
                    validate_consistency=False,
                )

                # Cache recent snapshots
                self.component_cache[f"step_{agent_state.step_count}"] = step_snapshot

            # Update component_cache with step processing results for performance optimization
            self.component_cache["last_step_info"] = {
                "step_number": agent_state.step_count,
                "reward": reward_result.reward,
                "terminated": terminated,
                "truncated": truncated,
            }

            # Record step processing timing in performance_metrics for latency analysis and optimization
            step_duration_ms = (time.time() - step_start_time) * 1000
            self.performance_metrics.record_timing("episode_step", step_duration_ms)

            # Log step processing completion with action, outcome, and performance context
            if step_duration_ms > PERFORMANCE_TARGET_STEP_LATENCY_MS:
                self.logger.warning(
                    f"Step {agent_state.step_count} "
                    f"exceeded target latency: {step_duration_ms:.2f}ms"
                )

            # Return 5-tuple (observation, reward, terminated, truncated, info) for Gymnasium API compliance
            return observation, reward_result.reward, terminated, truncated, info

        except Exception as e:
            self.logger.error(f"Episode step processing failed: {e}")
            raise StateError(
                message=f"Episode step processing failed: {str(e)}",
                current_state="step_processing",
                expected_state="process_action",
            ) from e

    def finalize_episode(
        self, collect_detailed_statistics: bool = True
    ) -> EpisodeResult:
        """
        Complete episode processing with final state analysis, statistics collection, and
        component cleanup for episode conclusion.
        """
        try:
            # Validate episode is complete with proper termination or truncation status
            if self.current_episode_state is None:
                raise StateError(
                    message="Cannot finalize episode: no episode state available",
                    current_state="no_episode",
                    expected_state="finalize_episode",
                )

            if not (
                self.current_episode_state.terminated
                or self.current_episode_state.truncated
            ):
                self.logger.warning(
                    "Finalizing episode that is not terminated or truncated"
                )

            # Get final state from state_manager with complete agent and episode information
            self.state_manager.get_current_state(
                include_performance_data=True,
                include_component_details=collect_detailed_statistics,
            )

            # Calculate final performance metrics; convert seconds -> milliseconds
            episode_duration_ms = (
                self.current_episode_state.get_episode_duration() * 1000.0
            )

            # Collect component statistics from state_manager, reward_calculator, and action_processor
            component_stats = {}
            if collect_detailed_statistics:
                component_stats["state_manager"] = (
                    self.state_manager.get_performance_metrics()
                )
                component_stats["reward_calculator"] = (
                    self.reward_calculator.get_reward_statistics()
                )
                component_stats["action_processor"] = (
                    self.action_processor.get_processing_statistics()
                )

            # Create comprehensive EpisodeResult with final states and performance analysis
            # Guard and snapshot current episode/agent states for typing
            episode_state = self.current_episode_state
            agent_state = self.state_manager.current_agent_state
            if agent_state is None:
                raise StateError(
                    message="Cannot finalize episode: missing agent state",
                    current_state="finalize",
                    expected_state="finalize_episode",
                )

            episode_result = EpisodeResult(
                episode_id=episode_state.episode_id,
                terminated=episode_state.terminated,
                truncated=episode_state.truncated,
                total_steps=agent_state.step_count,
                total_reward=agent_state.total_reward,
                episode_duration_ms=episode_duration_ms,
            )

            # Set final state information
            final_distance = None
            if agent_state:
                source_location = cast(
                    Coordinates, self.config.env_config.source_location
                )
                final_distance = agent_state.position.distance_to(source_location)

            episode_result.set_final_state(
                final_agent_state=self.state_manager.current_agent_state,
                final_distance=final_distance or 0.0,
                termination_reason=getattr(
                    self.current_episode_state, "termination_reason", "unknown"
                ),
            )

            # Add performance metrics
            performance_metrics = self.performance_metrics.get_performance_summary()
            episode_result.add_performance_metrics(performance_metrics, component_stats)

            # Add episode result to episode_statistics for multi-episode trend analysis
            self.episode_statistics.add_episode_result(episode_result)

            # Create final state snapshot for episode completion analysis and debugging
            if self.config.enable_state_validation:
                final_snapshot = self.state_manager.create_state_snapshot(
                    snapshot_name=f"{episode_result.episode_id}_final",
                    validate_consistency=True,
                )
                episode_result.add_state_snapshot(final_snapshot)

            # Set episode_active to False and update current_episode_state for cleanup
            self.episode_active = False

            # Log episode completion with comprehensive statistics and performance summary
            self.logger.info(
                f"Episode {episode_result.episode_id} finalized: "
                f"{'TERMINATED' if episode_result.terminated else 'TRUNCATED'} "
                f"after {episode_result.total_steps} steps "
                f"({episode_duration_ms:.2f}ms, reward={episode_result.total_reward})"
            )

            # Return EpisodeResult for analysis, storage, and research evaluation
            return episode_result

        except Exception as e:
            self.logger.error(f"Episode finalization failed: {e}")
            raise ComponentError(
                message=f"Episode finalization failed: {str(e)}",
                component_name="EpisodeManager",
                operation_name="finalize_episode",
            ) from e

    def get_episode_statistics(self):
        """Return the aggregated EpisodeStatistics object used by the manager."""
        return self.episode_statistics

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return EpisodeManager-level performance metrics in a simple dict."""
        metrics = self.performance_metrics
        data: Dict[str, Any] = {
            "average_step_time_ms": metrics.average_step_time_ms(),
            "total_processing_time_ms": sum(metrics.step_durations_ms),
        }
        resets = metrics.other_timings_ms.get("episode_reset")
        if resets:
            data["reset_time_ms"] = resets[-1]
        else:
            data["reset_time_ms"] = 0.0
        return data

    def get_current_state(
        self,
        include_performance_data: bool = False,
        include_component_details: bool = False,
    ) -> Dict[str, object]:
        """
        Get comprehensive current episode state including agent position, episode status,
        component states, and performance metrics for external monitoring.
        """
        try:
            # Validate current episode is active with proper state initialization
            if not self.episode_active or self.current_episode_state is None:
                return {
                    "episode_active": False,
                    "episode_count": self.episode_count,
                    "error": "No active episode",
                    "timestamp": time.time(),
                }

            # Prefer returning attribute-accessible structure used by tests
            from types import SimpleNamespace

            agent_state_obj = self.state_manager.current_agent_state
            episode_state_obj = self.current_episode_state

            current_state = SimpleNamespace(
                episode_active=self.episode_active,
                episode_count=self.episode_count,
                episode_id=episode_state_obj.episode_id,
                agent_state=agent_state_obj,
                episode_state=episode_state_obj,
                timestamp=time.time(),
            )

            # Add performance data from performance_metrics if include_performance_data is True
            if include_performance_data:
                current_state.performance_data = (
                    self.performance_metrics.get_performance_summary()
                )

            # Include detailed component states if include_component_details is True
            if include_component_details:
                current_state.component_details = {
                    "reward_calculator": self.reward_calculator.get_reward_statistics(),
                    "action_processor": self.action_processor.get_processing_statistics(),
                    "component_cache_size": len(self.component_cache),
                }

            # Add episode identification and tracking information for external coordination
            current_state.tracking_info = {
                "episode_id": self.current_episode_state.episode_id,
                "episode_count": self.episode_count,
                "step_count": agent_state_obj.step_count if agent_state_obj else 0,
            }

            # Return comprehensive current state dictionary for monitoring and analysis
            return current_state

        except Exception as e:
            self.logger.error(f"Failed to get current state: {e}")
            return {
                "episode_active": self.episode_active,
                "episode_count": self.episode_count,
                "error": f"State retrieval failed: {str(e)}",
                "timestamp": time.time(),
            }

    def validate_episode_consistency(self, strict_validation: bool = False) -> bool:
        """
        Perform comprehensive episode consistency validation across all components with detailed
        error analysis and recovery recommendations.
        """
        try:
            validation_timestamp = time.time()
            validation_result: ConsistencyReport = {
                "is_consistent": True,
                "validation_timestamp": validation_timestamp,
                "validation_details": {},
                "errors": [],
                "warnings": [],
                "recommendations": [],
            }

            # Validate current episode state consistency across all components
            if not self.episode_active and self.current_episode_state is not None:
                validation_result["warnings"].append(
                    "Episode state exists but episode not active"
                )

            # Check agent state synchronization between state_manager and reward_calculator
            agent_state = getattr(self.state_manager, "current_agent_state", None)
            if self.current_episode_state is not None and agent_state is not None:
                state_position = agent_state.position
                # Validate position is within grid bounds
                gs = cast(GridSize, self.config.env_config.grid_size)
                if not gs.contains_coordinates(state_position):
                    validation_result["is_consistent"] = False
                    validation_result["errors"].append(
                        f"Agent position {state_position} outside grid bounds"
                    )

            # Check episode termination logic consistency across reward calculator and state manager
            if (
                self.current_episode_state is not None
                and (
                    self.current_episode_state.terminated
                    or self.current_episode_state.truncated
                )
                and self.episode_active
            ):
                validation_result["warnings"].append(
                    "Episode terminated/truncated but still marked as active"
                )

            # Apply strict validation rules if strict_validation enabled with enhanced checking
            if strict_validation:
                try:
                    state_validation = self.state_manager.validate_state_consistency(
                        strict_validation=True
                    )
                    if not getattr(state_validation, "is_valid", True):
                        validation_result["is_consistent"] = False
                        errors = getattr(state_validation, "validation_errors", [])
                        if isinstance(errors, list):
                            for entry in errors:
                                if isinstance(entry, dict):
                                    msg = entry.get("message")
                                    if isinstance(msg, str):
                                        validation_result["errors"].append(msg)
                except Exception as e:
                    validation_result["warnings"].append(
                        f"State manager validation failed: {str(e)}"
                    )

            # Generate detailed error analysis and recovery recommendations for inconsistencies
            if validation_result["errors"]:
                validation_result["recommendations"].extend(
                    [
                        "Reset episode to clear inconsistent state",
                        "Check component initialization and integration",
                        "Validate configuration parameters",
                    ]
                )

            return bool(validation_result["is_consistent"])
        except Exception:
            return False

    def cleanup(
        self, preserve_statistics: bool = True, clear_performance_data: bool = False
    ) -> None:
        """
        Perform comprehensive cleanup of episode manager and all components with resource release
        and final statistics collection.
        """
        try:
            self.logger.info("Starting EpisodeManager cleanup")

            # Finalize current episode if active with proper completion analysis
            if self.episode_active and self.current_episode_state:
                try:
                    self.finalize_episode(
                        collect_detailed_statistics=preserve_statistics
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to finalize active episode during cleanup: {e}"
                    )

            # Cleanup state_manager with component coordination and resource release
            if self.state_manager:
                try:
                    self.state_manager.cleanup(
                        preserve_configuration=preserve_statistics,
                        clear_performance_data=clear_performance_data,
                    )
                except Exception as e:
                    self.logger.error(f"StateManager cleanup failed: {e}")

            # Clear component caches and performance optimization data structures
            self.component_cache.clear()

            # Preserve episode_statistics if preserve_statistics is True for analysis continuity
            if not preserve_statistics:
                self.episode_statistics = EpisodeStatistics(
                    statistics_id=f"episode_manager_{int(time.time() * 1000)}"
                )

            # Clear performance metrics data if clear_performance_data is True
            if clear_performance_data and hasattr(self.performance_metrics, "clear"):
                self.performance_metrics.clear()

            # Reset episode tracking counters and state flags for clean shutdown
            self.episode_active = False
            self.current_episode_state = None

            if not preserve_statistics:
                self.episode_count = 0

            # Clear component-specific caches
            if hasattr(self.action_processor, "clear_cache"):
                self.action_processor.clear_cache()
            if hasattr(self.reward_calculator, "clear_cache"):
                self.reward_calculator.clear_cache()

            # Log cleanup completion with final statistics summary and resource release information
            self.logger.info(
                f"EpisodeManager cleanup completed (statistics preserved: {preserve_statistics}, "
                f"performance data cleared: {clear_performance_data})"
            )

        except Exception as e:
            self.logger.error(f"EpisodeManager cleanup failed: {e}")
            raise ComponentError(
                message=f"EpisodeManager cleanup failed: {str(e)}",
                component_name="EpisodeManager",
                operation_name="cleanup",
            ) from e


def create_episode_manager(
    config: Optional[EpisodeManagerConfig] = None,
    seed_manager: Optional[SeedManager] = None,
    enable_performance_monitoring: bool = DEFAULT_ENABLE_PERFORMANCE_MONITORING,
    enable_component_validation: bool = DEFAULT_ENABLE_STATE_VALIDATION,
) -> EpisodeManager:
    """
    Factory function to create properly configured EpisodeManager with comprehensive component
    coordination, validation setup, and performance monitoring for reinforcement learning
    environment episode orchestration.
    """
    try:
        # Create default EpisodeManagerConfig if none provided using system constants and parameter validation
        if config is None:
            # Create default EnvironmentConfig
            default_grid_size = GridSize(
                width=DEFAULT_GRID_SIZE[0], height=DEFAULT_GRID_SIZE[1]
            )

            default_env_config = EnvironmentConfig(
                grid_size=default_grid_size,
                max_steps=DEFAULT_MAX_STEPS,
                goal_radius=DEFAULT_GOAL_RADIUS,
            )

            config = EpisodeManagerConfig(
                env_config=default_env_config,
                enable_performance_monitoring=enable_performance_monitoring,
                enable_state_validation=enable_component_validation,
            )

        # Validate configuration using config.validate() with comprehensive parameter and consistency checking
        config.validate()

        # Initialize or create SeedManager if not provided with proper seeding configuration for reproducibility
        if seed_manager is None:
            seed_manager = SeedManager()

        # Initialize EpisodeManager with validated configuration and all component dependencies
        episode_manager = EpisodeManager(config=config, seed_manager=seed_manager)

        # Enable performance monitoring if requested with timing collection and analysis setup
        # Already handled by config

        # Enable component validation if requested with consistency checking and error detection
        # Already handled by config

        # Establish cross-component coordination and dependency injection for seamless integration
        # Already handled by EpisodeManager initialization

        # Validate complete episode manager setup and component integration consistency
        validation_result = episode_manager.validate_episode_consistency(
            strict_validation=enable_component_validation
        )

        if not validation_result["is_consistent"] and enable_component_validation:
            raise ComponentError(
                message="Episode manager setup validation failed",
                component_name="EpisodeManagerFactory",
                operation_name="create_episode_manager",
            )

        logger = get_component_logger("EpisodeManagerFactory")
        logger.info("EpisodeManager created successfully with component coordination")

        # Return fully configured EpisodeManager ready for Gymnasium environment episode orchestration
        return episode_manager

    except Exception as e:
        raise ComponentError(
            message=f"EpisodeManager creation failed: {str(e)}",
            component_name="EpisodeManagerFactory",
            operation_name="create_episode_manager",
        ) from e


def validate_episode_config(
    config: EpisodeManagerConfig,
    strict_validation: bool = False,
    validation_context: Optional[Dict[str, object]] = None,
) -> Tuple[bool, ValidationReport]:
    """
    Comprehensive validation of episode manager configuration parameters ensuring mathematical
    consistency, component compatibility, and performance feasibility for episode management operations.
    """
    try:
        validation_report: ValidationReport = {
            "is_valid": True,
            "validation_timestamp": time.time(),
            "strict_mode": strict_validation,
            "findings": [],
            "warnings": [],
            "recommendations": [],
            "parameter_analysis": {},
        }

        # Validate EnvironmentConfig parameters including grid size, source location, and episode limits
        try:
            config.env_config.validate()
            validation_report["findings"].append(
                {
                    "category": "environment_config",
                    "severity": "info",
                    "message": "Environment configuration is valid",
                }
            )
        except ValidationError as e:
            validation_report["is_valid"] = False
            validation_report["findings"].append(
                {
                    "category": "environment_config",
                    "severity": "error",
                    "message": f"Environment configuration invalid: {str(e)}",
                }
            )

        # Check component configuration compatibility including state manager, reward calculator, and action processor
        try:
            component_configs = config.derive_component_configs(
                validate_derived_configs=True
            )
            validation_report["findings"].append(
                {
                    "category": "component_compatibility",
                    "severity": "info",
                    "message": f"Successfully derived {len(component_configs)} component configurations",
                }
            )
        except Exception as e:
            validation_report["is_valid"] = False
            validation_report["findings"].append(
                {
                    "category": "component_compatibility",
                    "severity": "error",
                    "message": f"Component configuration derivation failed: {str(e)}",
                }
            )

        # Validate performance requirements and computational feasibility for episode processing targets
        try:
            resource_estimates = config.estimate_episode_resources(
                expected_episode_length=config.env_config.max_steps,
                include_component_overhead=True,
            )

            total_memory_mb = resource_estimates["memory_usage_mb"]["total_estimated"]
            if total_memory_mb > 1000:  # 1GB limit
                validation_report["warnings"].append(
                    {
                        "category": "resource_usage",
                        "message": f"High memory usage estimated: {total_memory_mb:.1f}MB",
                        "recommendation": "Consider reducing grid size or episode cache size",
                    }
                )

            validation_report["parameter_analysis"][
                "resource_estimates"
            ] = resource_estimates

        except Exception as e:
            validation_report["warnings"].append(
                {
                    "category": "resource_analysis",
                    "message": f"Resource estimation failed: {str(e)}",
                }
            )

        # Cross-validate all parameters for mathematical consistency and component integration requirements
        grid_size = cast(GridSize, config.env_config.grid_size)
        max_steps = config.env_config.max_steps

        # Check reasonable parameter ranges
        if max_steps > 100000:
            validation_report["warnings"].append(
                {
                    "category": "parameter_ranges",
                    "message": f"Very high max_steps ({max_steps}) may impact performance",
                    "recommendation": "Consider reducing max_steps for better performance",
                }
            )

        if grid_size.width * grid_size.height > 500000:  # 500k cells
            validation_report["warnings"].append(
                {
                    "category": "parameter_ranges",
                    "message": f"Large grid size ({grid_size.width}x{grid_size.height}) may impact performance",
                    "recommendation": "Consider smaller grid dimensions",
                }
            )

        # Apply strict validation rules if strict_validation enabled including additional precision and boundary checking
        if strict_validation:
            # Additional validation for production deployment
            if config.component_coordination_timeout < 1.0:
                validation_report["warnings"].append(
                    {
                        "category": "strict_validation",
                        "message": "Short coordination timeout may cause integration issues",
                        "recommendation": "Increase component_coordination_timeout for stability",
                    }
                )

            if not config.enable_performance_monitoring:
                validation_report["warnings"].append(
                    {
                        "category": "strict_validation",
                        "message": "Performance monitoring disabled in strict mode",
                        "recommendation": "Enable performance monitoring for production use",
                    }
                )

        # Check resource implications including memory usage, computation time, and optimization opportunities
        # Already handled above

        # Validate seeding configuration for reproducibility requirements and deterministic behavior
        if (
            config.enable_reproducibility_validation
            and not config.enable_state_validation
        ):
            validation_report["warnings"].append(
                {
                    "category": "reproducibility",
                    "message": "Reproducibility validation requires state validation",
                    "recommendation": "Enable state validation for reproducibility guarantees",
                }
            )

        # Generate detailed validation report with findings, warnings, optimization recommendations, and error recovery suggestions
        if validation_report["is_valid"]:
            validation_report["recommendations"].extend(
                [
                    "Configuration is valid and ready for use",
                    "Monitor performance metrics during episode execution",
                    "Consider enabling all validation options for development",
                ]
            )
        else:
            validation_report["recommendations"].extend(
                [
                    "Fix critical validation errors before proceeding",
                    "Review component configuration parameters",
                    "Validate environment configuration independently",
                ]
            )

        # Include component-specific validation results and integration feasibility analysis
        validation_report["parameter_analysis"].update(
            {
                "grid_size": (grid_size.width, grid_size.height),
                "max_steps": max_steps,
                "goal_radius": config.env_config.goal_radius,
                "performance_monitoring_enabled": config.enable_performance_monitoring,
                "state_validation_enabled": config.enable_state_validation,
                "component_integration_enabled": config.enable_component_integration,
            }
        )

        # Return comprehensive validation result with actionable feedback and configuration improvement suggestions
        return validation_report["is_valid"], validation_report

    except Exception as e:
        # Handle validation process errors
        error_report: ValidationReport = {
            "is_valid": False,
            "validation_timestamp": time.time(),
            "strict_mode": strict_validation,
            "findings": [],
            "warnings": [
                {
                    "category": "validation_process",
                    "message": "Validation process failed",
                    "recommendation": "Fix validation process errors before configuration use",
                }
            ],
            "recommendations": [
                "Fix validation process errors before configuration use"
            ],
            "parameter_analysis": {},
            "validation_error": str(e),
            "error_details": {
                "exception_type": type(e).__name__,
                "exception_message": str(e),
            },
        }
        return False, error_report
