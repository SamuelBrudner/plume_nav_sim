"""Episode lifecycle management: init, step, terminate, reset."""

import contextlib
import copy
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

try:  # pragma: no cover - numpy<1.20 compatibility
    from numpy.typing import NDArray
except ImportError:  # pragma: no cover
    NDArray = np.ndarray  # type: ignore[assignment]
from ..constants import (
    DEFAULT_GOAL_RADIUS,
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_STEPS,
    PERFORMANCE_TARGET_STEP_LATENCY_MS,
)
from ..envs.config_types import EnvironmentConfig
from ..utils.exceptions import (
    ComponentError,
    ResourceError,
    StateError,
    ValidationError,
)
from ..utils.logging import get_component_logger, monitor_performance
from ..utils.seeding import SeedManager
from .action_processor import ActionProcessingResult, ActionProcessor
from .reward_calculator import RewardCalculator, RewardResult
from .state_manager import StateManager, StateManagerConfig
from .types import Action, AgentState, Coordinates, GridSize


class ProcessStepResult(tuple):
    __slots__ = ()

    def __new__(
        cls,
        observation: NDArray[np.float32],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, object],
    ) -> "ProcessStepResult":
        return super().__new__(cls, (observation, reward, terminated, truncated, info))

    def __getitem__(self, index):
        if isinstance(index, int):
            if index == 3:
                return super().__getitem__(4)
            if index == -2:
                return super().__getitem__(3)
        return super().__getitem__(index)


def _build_step_info(
    agent_state: AgentState, additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "agent_xy": (agent_state.position.x, agent_state.position.y),
        "agent_position": (agent_state.position.x, agent_state.position.y),
        "step_count": agent_state.step_count,
        "total_reward": agent_state.total_reward,
        "goal_reached": agent_state.goal_reached,
    }
    if additional_info:
        info.update(additional_info)
    return info


def _create_episode_state(
    episode_id: str, terminated: bool, truncated: bool
) -> Dict[str, Any]:
    return {
        "episode_id": episode_id,
        "terminated": terminated,
        "truncated": truncated,
        "start_time": time.time(),
        "end_time": None,
        "termination_reason": None,
    }


__all__ = [
    "EpisodeManager",
    "EpisodeManagerConfig",
    "EpisodeResult",
    "EpisodeStatistics",
    "create_episode_manager",
    "validate_episode_config",
]

ValidationReport = Dict[str, Any]
ConsistencyReport = Dict[str, Any]


@dataclass
class EpisodeManagerConfig:
    env_config: EnvironmentConfig
    enable_performance_monitoring: bool = True
    enable_state_validation: bool = True
    enable_component_integration: bool = field(default=True)
    enable_reproducibility_validation: bool = field(default=True)
    enable_reproducibility_tracking: bool = field(default=True)
    episode_timeout_ms: float = field(default=30_000.0)
    component_coordination_timeout: float = field(default=5.0)
    episode_cache_size: int = field(default=100)
    component_configs: Dict[str, object] = field(default_factory=dict)
    custom_parameters: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:  # noqa: C901
        if not isinstance(self.env_config, EnvironmentConfig):
            raise ValidationError(
                message="env_config must be EnvironmentConfig instance",
                parameter_name="env_config",
                parameter_value=type(self.env_config).__name__,
                expected_format="EnvironmentConfig dataclass",
            )
        self.env_config.validate()
        if not isinstance(self.enable_performance_monitoring, bool):
            self.enable_performance_monitoring = True
        if not isinstance(self.enable_state_validation, bool):
            self.enable_state_validation = True
        if not isinstance(self.enable_component_integration, bool):
            self.enable_component_integration = True
        if not isinstance(self.enable_reproducibility_validation, bool):
            self.enable_reproducibility_validation = True
        if not isinstance(self.enable_reproducibility_tracking, bool):
            self.enable_reproducibility_tracking = (
                self.enable_reproducibility_validation
            )
        else:
            self.enable_reproducibility_validation = (
                self.enable_reproducibility_tracking
            )
        if self.enable_component_integration and not self.enable_state_validation:
            self.enable_component_integration = False
        if (
            not isinstance(self.component_coordination_timeout, (int, float))
            or self.component_coordination_timeout <= 0
        ):
            self.component_coordination_timeout = 5.0
        if not isinstance(self.episode_cache_size, int) or self.episode_cache_size < 0:
            self.episode_cache_size = 100
        if (
            not isinstance(self.episode_timeout_ms, (int, float))
            or self.episode_timeout_ms <= 0
        ):
            self.episode_timeout_ms = 30_000.0
        if not isinstance(self.component_configs, dict):
            self.component_configs = {}
        if not isinstance(self.custom_parameters, dict):
            self.custom_parameters = {}
        self._validate_parameter_compatibility()

    def _validate_parameter_compatibility(self):
        if self.component_coordination_timeout > 60.0:
            raise ValidationError(
                message="component_coordination_timeout too high, may cause performance issues",
                parameter_name="component_coordination_timeout",
                parameter_value=str(self.component_coordination_timeout),
                expected_format="<=60.0 seconds",
            )
        if self.episode_cache_size > 1000:
            raise ValidationError(
                message="episode_cache_size too high, may cause memory issues",
                parameter_name="episode_cache_size",
                parameter_value=str(self.episode_cache_size),
                expected_format="<=1000 entries",
            )

    def validate(  # noqa: C901
        self,
        strict_mode: bool = False,
        validation_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            self.env_config.validate()
            if self.component_configs:
                for component_name, _component_config in self.component_configs.items():
                    if not isinstance(component_name, str):
                        raise ValidationError(
                            message=f"Component name must be string: {component_name}",
                            parameter_name="component_configs",
                            parameter_value=str(type(component_name)),
                            expected_format="string component names",
                        )
            if strict_mode and (
                not self.enable_state_validation
                and self.enable_reproducibility_validation
            ):
                raise ValidationError(
                    message="reproducibility validation requires state validation",
                    parameter_name="validation_configuration",
                    parameter_value="enable_state_validation=False with enable_reproducibility_validation=True",
                    expected_format="consistent validation settings",
                )
            if self.enable_component_integration and not self.enable_state_validation:
                raise ValidationError(
                    message="component integration requires state validation",
                    parameter_name="integration_configuration",
                    parameter_value="enable_component_integration=True with enable_state_validation=False",
                    expected_format="consistent component settings",
                )
            if self.custom_parameters:
                for param_name, _param_value in self.custom_parameters.items():
                    if not isinstance(param_name, str):
                        raise ValidationError(
                            message=f"Custom parameter name must be string: {param_name}",
                            parameter_name="custom_parameters",
                            parameter_value=str(type(param_name)),
                            expected_format="string parameter names",
                        )
            if strict_mode:
                max_steps = self.env_config.max_steps
                if max_steps > 50000:
                    raise ValidationError(
                        message="max_steps too high for strict validation",
                        parameter_name="env_config.max_steps",
                        parameter_value=str(max_steps),
                        expected_format="<=50000 in strict mode",
                    )
            self._validate_parameter_compatibility()
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

    def derive_component_configs(  # noqa: C901
        self, validate_derived_configs: bool = True
    ) -> Dict[str, Any]:
        try:
            state_manager_config = StateManagerConfig(
                grid_size=cast(GridSize, self.env_config.grid_size),
                source_location=cast(Coordinates, self.env_config.source_location),
                max_steps=self.env_config.max_steps,
                goal_radius=self.env_config.goal_radius,
                enable_performance_monitoring=self.enable_performance_monitoring,
                enable_state_validation=self.enable_state_validation,
            )
            from .reward_calculator import RewardCalculatorConfig

            reward_calculator_config = RewardCalculatorConfig(
                goal_radius=self.env_config.goal_radius,
                reward_goal_reached=1.0,  # Standard reward for goal achievement
                reward_default=0.0,  # Sparse reward structure
                enable_performance_monitoring=self.enable_performance_monitoring,
            )
            from .action_processor import ActionProcessingConfig

            action_processor_config = ActionProcessingConfig(
                enable_validation=self.enable_state_validation,
                enforce_boundaries=True,
                enable_performance_monitoring=self.enable_performance_monitoring,
                strict_validation=self.enable_state_validation,
            )
            component_configs: Dict[str, Any] = {
                "state_manager": state_manager_config,
                "reward_calculator": reward_calculator_config,
                "action_processor": action_processor_config,
                "StateManager": state_manager_config,
                "RewardCalculator": reward_calculator_config,
                "ActionProcessor": action_processor_config,
            }
            for component_name, custom_config in self.component_configs.items():
                if not isinstance(custom_config, dict):
                    raise ValidationError(
                        message=(
                            "Custom component configuration overrides must be provided as "
                            "dictionaries of field names to values"
                        ),
                        parameter_name="component_configs",
                        parameter_value=type(custom_config).__name__,
                        expected_format=(
                            "dict mapping component names to dicts of dataclass field overrides"
                        ),
                    )
                target_config = component_configs.get(
                    component_name
                ) or component_configs.get(str(component_name).lower())
                if target_config is None:
                    raise ValidationError(
                        message=(
                            f"Unknown component configuration override: {component_name}"
                        ),
                        parameter_name="component_configs",
                        parameter_value=str(component_name),
                        expected_format=(
                            "known component names such as 'state_manager', "
                            "'reward_calculator', or 'action_processor'"
                        ),
                    )
                if not hasattr(target_config, "__dataclass_fields__"):
                    raise ValidationError(
                        message=(
                            f"Target configuration for component '{component_name}' is not a "
                            "dataclass-based config"
                        ),
                        parameter_name="component_configs",
                        parameter_value=type(target_config).__name__,
                        expected_format="dataclass-based component configuration object",
                    )
                for key, value in custom_config.items():
                    if hasattr(target_config, key):
                        setattr(target_config, key, value)
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
            return component_configs
        except ValidationError:
            raise
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
        try:
            grid = self.env_config.grid_size
            grid_cells = grid.width * grid.height
            grid_memory_mb = (grid_cells * 4) / (1024 * 1024)
            cache_memory_mb = self.episode_cache_size / 1024
            component_overhead_mb = 10.0 if include_component_overhead else 0.0
            monitoring_overhead_mb = 2.0 if self.enable_performance_monitoring else 0.0
            total_memory_mb = (
                5.0
                + grid_memory_mb
                + component_overhead_mb
                + cache_memory_mb
                + monitoring_overhead_mb
            )
            duration_s = expected_episode_length * (
                PERFORMANCE_TARGET_STEP_LATENCY_MS / 1000
            )
            return {
                "memory_usage_mb": {"total_estimated": total_memory_mb},
                "performance_estimates": {
                    "expected_episode_duration_s": duration_s,
                    "target_step_latency_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS,
                },
            }
        except Exception as e:
            return {"error": f"Resource estimation failed: {str(e)}"}

    def clone(
        self,
        overrides: Optional[Dict[str, Any]] = None,
        preserve_component_configs: bool = True,
    ) -> "EpisodeManagerConfig":
        try:
            cloned_env_config = copy.deepcopy(self.env_config)
            cloned_component_configs = (
                self.component_configs.copy() if preserve_component_configs else {}
            )
            cloned_custom_parameters = self.custom_parameters.copy()
            cloned_config = EpisodeManagerConfig(
                env_config=cloned_env_config,
                enable_performance_monitoring=self.enable_performance_monitoring,
                enable_state_validation=self.enable_state_validation,
                enable_component_integration=self.enable_component_integration,
                enable_reproducibility_validation=self.enable_reproducibility_validation,
                enable_reproducibility_tracking=self.enable_reproducibility_tracking,
                component_coordination_timeout=self.component_coordination_timeout,
                episode_cache_size=self.episode_cache_size,
                component_configs=cloned_component_configs,
                custom_parameters=cloned_custom_parameters,
            )
            if overrides:
                env_overrides: Dict[str, Any] = {}
                env_field_names = set(
                    getattr(self.env_config, "__dataclass_fields__", {}).keys()
                )
                for key, value in overrides.items():
                    if key in env_field_names:
                        env_overrides[key] = value
                    elif hasattr(cloned_config, key):
                        setattr(cloned_config, key, value)
                    else:
                        cloned_config.custom_parameters[key] = value
                if env_overrides:
                    cloned_config.env_config = replace(
                        cloned_config.env_config, **env_overrides
                    )
            cloned_config.validate()
            return cloned_config
        except ValidationError:
            raise
        except Exception as e:
            raise ComponentError(
                message=f"Failed to clone episode manager configuration: {str(e)}",
                component_name="EpisodeManagerConfig",
                operation_name="clone",
            ) from e


@dataclass
class EpisodeResult:
    episode_id: str
    terminated: bool
    truncated: bool
    total_steps: int
    total_reward: float = field(default=0.0)
    episode_duration_ms: float = field(default=0.0)
    final_agent_position: Optional[Coordinates] = field(default=None)
    final_distance_to_goal: Optional[float] = field(default=None)
    performance_metrics: Dict[str, object] = field(default_factory=dict)
    component_statistics: Dict[str, object] = field(default_factory=dict)
    state_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    termination_reason: Optional[str] = field(default=None)

    def __post_init__(self):
        if not self.episode_id or not isinstance(self.episode_id, str):
            raise ValidationError(
                message="episode_id must be non-empty string",
                parameter_name="episode_id",
                parameter_value=str(self.episode_id),
                expected_format="non-empty string identifier",
            )
        self.total_steps = max(self.total_steps, 0)
        self.episode_duration_ms = max(self.episode_duration_ms, 0.0)
        if not isinstance(self.performance_metrics, dict):
            self.performance_metrics = {}
        if not isinstance(self.component_statistics, dict):
            self.component_statistics = {}
        if not isinstance(self.state_snapshots, list):
            self.state_snapshots = []

    def set_final_state(
        self,
        final_agent_state: AgentState,
        final_distance: float | None = None,
        termination_reason: str | None = None,
        *,
        distance_to_goal: float | None = None,
    ) -> None:
        try:
            if hasattr(final_agent_state, "position") and isinstance(
                final_agent_state.position, Coordinates
            ):
                self.final_agent_position = final_agent_state.position
            if hasattr(final_agent_state, "total_reward"):
                self.total_reward = float(final_agent_state.total_reward)
            candidate = final_distance if distance_to_goal is None else distance_to_goal
            if isinstance(candidate, (int, float)) and candidate >= 0:
                self.final_distance_to_goal = float(candidate)
            if isinstance(termination_reason, str) and termination_reason.strip():
                self.termination_reason = termination_reason
        except Exception as e:
            raise ComponentError(
                message=f"Failed to set final episode state: {str(e)}",
                component_name="EpisodeResult",
                operation_name="set_final_state",
            ) from e

    @property
    def duration_ms(self) -> float:
        return float(self.episode_duration_ms)

    @duration_ms.setter
    def duration_ms(self, value: float) -> None:
        self.episode_duration_ms = float(max(0.0, value))

    def add_performance_metrics(
        self, episode_metrics: Dict[str, Any], component_metrics: Dict[str, Any]
    ) -> None:
        with contextlib.suppress(Exception):
            if isinstance(episode_metrics, dict):
                self.performance_metrics.update(episode_metrics)
                timings = episode_metrics.get("timings")
                if isinstance(timings, dict):
                    step_series = timings.get("episode_step")
                    if isinstance(step_series, list):
                        self.performance_metrics["step_timing"] = list(step_series)
            if isinstance(component_metrics, dict):
                self.component_statistics.update(component_metrics)
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
            if "average_step_time_ms" in self.performance_metrics and (
                self.performance_metrics["average_step_time_ms"]
                > PERFORMANCE_TARGET_STEP_LATENCY_MS
            ):
                self.performance_metrics["performance_warning"] = (
                    "Step latency exceeded target"
                )

    def get_performance_metrics(self) -> Dict[str, Any]:
        data = dict(self.performance_metrics)
        data["component_metrics"] = dict(self.component_statistics)
        return data

    def get_summary(
        self,
        *,
        include_performance: bool = False,
        include_components: bool = False,
        include_performance_analysis: Optional[bool] = None,
        include_component_details: Optional[bool] = None,
    ) -> Dict[str, Any]:
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
        summary["goal_reached"] = bool(
            self.termination_reason == "goal_reached" or self.terminated
        )
        summary["success"] = self.terminated and not self.truncated
        if include_performance or include_performance_analysis:
            summary["performance"] = self.get_performance_metrics()
        if include_components or include_component_details:
            summary["components"] = dict(self.component_statistics)
        if self.termination_reason:
            summary["termination_reason"] = self.termination_reason
        summary["success_analysis"] = {
            "goal_achieved": self.terminated,
            "episode_completed_naturally": self.terminated or self.truncated,
            "total_reward": self.total_reward,
        }
        return summary

    def add_state_snapshot(self, snapshot: Dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            if not isinstance(snapshot, dict):
                raise ValidationError(
                    message="snapshot must be dict",
                    parameter_name="snapshot",
                    parameter_value=type(snapshot).__name__,
                    expected_format="dict snapshot",
                )
            self.state_snapshots.append(snapshot)
            if len(self.state_snapshots) > 1000:
                self.state_snapshots = self.state_snapshots[-500:]


@dataclass
class EpisodeStatistics:
    statistics_id: Optional[str] = None
    session_id: Optional[str] = None
    episodes_completed: int = field(default=0)
    episodes_terminated: int = field(default=0)
    episodes_truncated: int = field(default=0)
    successful_episodes: int = field(default=0)
    truncated_episodes: int = field(default=0)
    average_episode_length: float = field(default=0.0)
    average_episode_duration_ms: float = field(default=0.0)
    average_total_reward: float = field(default=0.0)
    performance_trends: Dict[str, object] = field(default_factory=dict)
    component_efficiency: Dict[str, object] = field(default_factory=dict)
    episode_results: List[EpisodeResult] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.session_id, str) and not self.statistics_id:
            self.statistics_id = f"stats_{self.session_id}"
        if not self.statistics_id or not isinstance(self.statistics_id, str):
            self.statistics_id = f"stats_{int(time.time() * 1000)}"
        if not isinstance(self.performance_trends, dict):
            self.performance_trends = {}
        if not isinstance(self.component_efficiency, dict):
            self.component_efficiency = {}
        if not isinstance(self.episode_results, list):
            self.episode_results = []

    def add_episode_result(self, episode_result: EpisodeResult) -> None:
        try:
            if not isinstance(episode_result, EpisodeResult):
                raise ValidationError(
                    message="episode_result must be EpisodeResult instance",
                    parameter_name="episode_result",
                    parameter_value=type(episode_result).__name__,
                    expected_format="EpisodeResult dataclass",
                )
            self.episode_results.append(episode_result)
            self.episodes_completed += 1
            if episode_result.terminated:
                self.episodes_terminated += 1
                self.successful_episodes += 1
            if episode_result.truncated:
                self.episodes_truncated += 1
                self.truncated_episodes += 1
            self._update_running_averages(episode_result)
            self._update_performance_trends(episode_result)
            self._update_component_efficiency(episode_result)
        except Exception as e:
            raise ComponentError(
                message=f"Failed to add episode result: {str(e)}",
                component_name="EpisodeStatistics",
                operation_name="add_episode_result",
            ) from e

    def _update_running_averages(self, episode_result: EpisodeResult) -> None:
        n = self.episodes_completed
        if n <= 1:
            self.average_episode_length = float(episode_result.total_steps)
            self.average_episode_duration_ms = episode_result.episode_duration_ms
            self.average_total_reward = episode_result.total_reward
        else:
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
        with contextlib.suppress(Exception):
            step_times = self.performance_trends.setdefault("step_times", [])
            episode_durations = self.performance_trends.setdefault(
                "episode_durations", []
            )
            if episode_result.total_steps > 0:
                step_times.append(
                    episode_result.episode_duration_ms / episode_result.total_steps
                )
            episode_durations.append(episode_result.episode_duration_ms)
            if len(step_times) > 100:
                self.performance_trends["step_times"] = step_times[-50:]
            if len(episode_durations) > 100:
                self.performance_trends["episode_durations"] = episode_durations[-50:]

    def _update_component_efficiency(self, episode_result: EpisodeResult) -> None:
        with contextlib.suppress(Exception):
            if episode_result.component_statistics:
                for component, stats in episode_result.component_statistics.items():
                    entry = self.component_efficiency.setdefault(
                        component, {"calls": 0, "total_time": 0.0}
                    )
                    if isinstance(stats, dict):
                        entry["calls"] += stats.get("calls", 1)
                        entry["total_time"] += stats.get("time", 0.0)

    def calculate_success_rate(self) -> float:
        return (
            self.episodes_terminated / self.episodes_completed
            if self.episodes_completed
            else 0.0
        )

    def get_performance_summary(  # noqa: C901
        self,
        include_trend_analysis: bool = True,
        include_optimization_recommendations: bool = True,
        include_trends: Optional[bool] = None,
    ) -> Dict[str, object]:
        try:
            if include_trends is not None:
                include_trend_analysis = include_trends
            average_step_latency = self.average_episode_duration_ms / max(
                1, self.average_episode_length
            )
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
                    "average_step_latency_ms": average_step_latency,
                    "episodes_per_second": 1000
                    / max(1, self.average_episode_duration_ms),
                },
            }
            if include_trend_analysis and self.performance_trends:
                summary["trend_analysis"] = dict(self.performance_trends)
            if self.component_efficiency:
                summary["component_efficiency"] = dict(self.component_efficiency)
            if include_optimization_recommendations:
                recommendations = []
                if average_step_latency > PERFORMANCE_TARGET_STEP_LATENCY_MS:
                    recommendations.append(
                        f"Step latency ({average_step_latency:.2f}ms) exceeds target ({PERFORMANCE_TARGET_STEP_LATENCY_MS}ms)"
                    )
                if self.calculate_success_rate() < 0.1:
                    recommendations.append(
                        "Low success rate suggests difficulty tuning needed"
                    )
                summary["optimization_recommendations"] = recommendations
            return summary
        except Exception as e:
            return {
                "error": f"Performance summary generation failed: {str(e)}",
                "basic_stats": {
                    "episodes_completed": self.episodes_completed,
                    "success_rate": self.calculate_success_rate(),
                },
            }

    def get_optimization_recommendations(self) -> List[str]:
        summary = self.get_performance_summary(
            include_trend_analysis=True,
            include_optimization_recommendations=True,
        )
        recommendations = summary.get("optimization_recommendations")
        return list(recommendations) if isinstance(recommendations, list) else []


class EpisodeManager:
    def __init__(
        self, config: EpisodeManagerConfig, seed_manager: Optional[SeedManager] = None
    ):
        try:
            if not isinstance(config, EpisodeManagerConfig):
                raise ValidationError(
                    message="config must be EpisodeManagerConfig instance",
                    parameter_name="config",
                    parameter_value=type(config).__name__,
                    expected_format="EpisodeManagerConfig dataclass",
                )
            config.validate()
            self.config = config
            self.seed_manager = (
                seed_manager if seed_manager is not None else SeedManager()
            )
            component_configs = self.config.derive_component_configs()
            state_manager_config = component_configs.get(
                "state_manager"
            ) or component_configs.get("StateManager")
            reward_calculator_config = component_configs.get(
                "reward_calculator"
            ) or component_configs.get("RewardCalculator")
            action_processor_config = component_configs.get(
                "action_processor"
            ) or component_configs.get("ActionProcessor")
            if (
                not state_manager_config
                or not reward_calculator_config
                or not action_processor_config
            ):
                raise ComponentError(
                    message="Derived component configurations missing",
                    component_name="EpisodeManagerConfig",
                    operation_name="derive_component_configs",
                )
            self.state_manager = StateManager(
                config=state_manager_config, seed_manager=self.seed_manager
            )
            self.reward_calculator = RewardCalculator(config=reward_calculator_config)
            self.action_processor = ActionProcessor(
                grid_size=cast(GridSize, self.config.env_config.grid_size),
                config=action_processor_config,
            )
            self.current_episode_state: Optional[Dict[str, Any]] = None
            self.performance_metrics: Dict[str, List[float]] = {}
            self.logger = get_component_logger("episode_manager")
            self.episode_active: bool = False
            self.episode_count: int = 0
            self.component_cache: Dict[str, object] = {}
            self.episode_statistics = EpisodeStatistics(
                statistics_id=f"episode_manager_{int(time.time() * 1000)}"
            )
            self._validate_component_integration()
            self.logger.info(
                f"EpisodeManager initialized successfully with {len(component_configs)} components"
            )
        except (ValidationError, StateError, ComponentError, ResourceError):
            raise
        except Exception as e:
            raise ComponentError(
                message=f"EpisodeManager initialization failed: {str(e)}",
                component_name="EpisodeManager",
                operation_name="__init__",
                underlying_error=e,
            ) from e

    def _record_timing(self, metric_name: str, duration_ms: float) -> None:
        if not isinstance(duration_ms, (int, float)):
            return
        self.performance_metrics.setdefault(metric_name, []).append(float(duration_ms))

    def _performance_summary(self) -> Dict[str, Any]:
        timings = {
            key: list(values) for key, values in self.performance_metrics.items()
        }
        step_timings = timings.get("episode_step", [])
        total_step_time = sum(step_timings)
        avg_step_time = total_step_time / len(step_timings) if step_timings else 0.0
        return {
            "total_step_time_ms": total_step_time,
            "average_step_time_ms": avg_step_time,
            "timings": timings,
        }

    def _validate_component_integration(self) -> None:
        try:
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
            state_config = self.state_manager.config
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
        try:
            reset_start_time = time.time()
            if seed is not None:
                self.seed_manager.seed(seed)
                self.logger.debug(f"Episode reset with seed: {seed}")
            else:
                self.logger.debug("Episode reset without seed (random)")
            self.current_episode_state = None
            self.component_cache.clear()
            episode_id = f"episode_{self.episode_count + 1:08d}"
            state_reset_info: object = self.state_manager.reset_episode(
                seed, episode_options
            )
            agent_state = self.state_manager.current_agent_state
            if agent_state is None:
                raise StateError(
                    message="Cannot reset episode: missing agent state",
                    current_state="episode_reset",
                    expected_state="reset_initialization",
                )
            grid = self.config.env_config.grid_size
            observation = np.zeros((grid.height, grid.width), dtype=np.float32)
            source_location = cast(Coordinates, self.config.env_config.source_location)
            distance_to_source = float(
                agent_state.position.distance_to(source_location)
            )
            info = _build_step_info(
                agent_state,
                additional_info={
                    "episode_id": episode_id,
                    "reset_info": state_reset_info,
                    "seed": seed,
                    "episode_options": episode_options,
                    "distance_to_source": distance_to_source,
                },
            )
            self.episode_active = True
            self.episode_count += 1
            self.current_episode_state = _create_episode_state(
                episode_id=episode_id, terminated=False, truncated=False
            )
            reset_duration_ms = (time.time() - reset_start_time) * 1000
            self._record_timing("episode_reset", reset_duration_ms)
            self.logger.debug(
                f"Episode {self.episode_count} reset completed in {reset_duration_ms:.2f}ms"
            )
            return observation, info
        except Exception as e:
            self.logger.error(f"Episode reset failed: {e}")
            raise StateError(
                message=f"Episode reset failed: {str(e)}",
                current_state="episode_reset",
                expected_state="reset_initialization",
            ) from e

    @monitor_performance("episode_step", PERFORMANCE_TARGET_STEP_LATENCY_MS, False)
    def process_step(self, action: Action) -> ProcessStepResult:
        try:
            step_start_time = time.time()
            if not self.episode_active:
                raise StateError(
                    message="Cannot process step: episode is not active. Call reset_episode() before stepping.",
                    current_state="inactive",
                    expected_state="process_step",
                )
            if self.current_episode_state is None:
                raise StateError(
                    message="Cannot process step: episode state not initialized",
                    current_state="uninitialized",
                    expected_state="process_step",
                )
            agent_state = self.state_manager.current_agent_state
            if agent_state is None:
                raise StateError(
                    message="Cannot process step: missing agent state",
                    current_state="uninitialized",
                    expected_state="process_step",
                )
            action_result: ActionProcessingResult = (
                self.action_processor.process_action(
                    action=action, current_position=agent_state.position
                )
            )
            state_step_result: Dict[str, object] = self.state_manager.process_step(
                action, action_context={"action_result": action_result}
            )
            grid = self.config.env_config.grid_size
            observation = np.zeros((grid.height, grid.width), dtype=np.float32)
            source_location = cast(Coordinates, self.config.env_config.source_location)
            reward_result: RewardResult = self.reward_calculator.calculate_reward(
                agent_position=agent_state.position, source_location=source_location
            )
            termination_result = self.reward_calculator.check_termination(
                agent_state=agent_state,
                source_location=source_location,
                max_steps=self.config.env_config.max_steps,
            )
            self.reward_calculator.update_agent_reward(
                agent_state=agent_state,
                reward_result=reward_result,
            )
            terminated = termination_result.terminated
            truncated = termination_result.truncated
            if terminated or truncated:
                if self.current_episode_state is not None:
                    self.current_episode_state["terminated"] = terminated
                    self.current_episode_state["truncated"] = truncated
                    self.current_episode_state["termination_reason"] = (
                        termination_result.termination_reason
                    )
                    self.current_episode_state["end_time"] = time.time()
                self.episode_active = False
            if self.current_episode_state is not None:
                self.current_episode_state["step_count"] = agent_state.step_count
            distance_to_source = float(
                agent_state.position.distance_to(source_location)
            )
            info = _build_step_info(
                agent_state,
                additional_info={
                    "action_result": action_result.to_dict(),
                    "reward_result": reward_result.to_dict(),
                    "termination_result": termination_result.get_summary(),
                    "state_step_result": state_step_result,
                    "distance_to_source": distance_to_source,
                },
            )
            self.component_cache["last_step_info"] = {
                "step_number": agent_state.step_count,
                "reward": reward_result.reward,
                "terminated": terminated,
                "truncated": truncated,
            }
            step_duration_ms = (time.time() - step_start_time) * 1000
            self._record_timing("episode_step", step_duration_ms)
            if step_duration_ms > (PERFORMANCE_TARGET_STEP_LATENCY_MS * 5.0):
                self.logger.warning(
                    f"Step {agent_state.step_count} exceeded latency: {step_duration_ms:.2f}ms"
                )
            return ProcessStepResult(
                observation, reward_result.reward, terminated, truncated, info
            )
        except (ComponentError, ResourceError, ValidationError, StateError):
            raise
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
        try:
            if self.current_episode_state is None:
                raise StateError(
                    message="Cannot finalize episode: no episode state available",
                    current_state="no_episode",
                    expected_state="finalize_episode",
                )
            if not (
                bool(self.current_episode_state.get("terminated"))
                or bool(self.current_episode_state.get("truncated"))
            ):
                self.logger.warning(
                    "Finalizing episode that is not terminated or truncated"
                )
            self.state_manager.get_current_state(
                include_performance_data=True,
                include_component_details=collect_detailed_statistics,
            )
            start_time = self.current_episode_state.get("start_time") or time.time()
            end_time = self.current_episode_state.get("end_time") or time.time()
            episode_duration_ms = (end_time - start_time) * 1000.0
            component_stats = (
                {
                    "state_manager": self.state_manager.get_performance_metrics(),
                    "reward_calculator": self.reward_calculator.get_reward_statistics(),
                    "action_processor": self.action_processor.get_processing_statistics(),
                }
                if collect_detailed_statistics
                else {}
            )
            episode_state = self.current_episode_state
            agent_state = self.state_manager.current_agent_state
            if agent_state is None:
                raise StateError(
                    message="Cannot finalize episode: missing agent state",
                    current_state="finalize",
                    expected_state="finalize_episode",
                )
            episode_result = EpisodeResult(
                episode_id=str(episode_state.get("episode_id", "unknown")),
                terminated=bool(episode_state.get("terminated")),
                truncated=bool(episode_state.get("truncated")),
                total_steps=agent_state.step_count,
                total_reward=agent_state.total_reward,
                episode_duration_ms=episode_duration_ms,
            )
            source_location = cast(Coordinates, self.config.env_config.source_location)
            final_distance = agent_state.position.distance_to(source_location)
            episode_result.set_final_state(
                final_agent_state=agent_state,
                final_distance=final_distance,
                termination_reason=(
                    self.current_episode_state.get("termination_reason") or "unknown"
                ),
            )
            performance_metrics = self._performance_summary()
            episode_result.add_performance_metrics(performance_metrics, component_stats)
            self.episode_statistics.add_episode_result(episode_result)
            self.episode_active = False
            self.logger.info(
                f"Episode {episode_result.episode_id} finalized: "
                f"{'TERMINATED' if episode_result.terminated else 'TRUNCATED'} "
                f"after {episode_result.total_steps} steps "
                f"({episode_duration_ms:.2f}ms, reward={episode_result.total_reward})"
            )
            return episode_result
        except (StateError, ComponentError, ValidationError, ResourceError):
            raise
        except Exception as e:
            self.logger.error(f"Episode finalization failed: {e}")
            raise ComponentError(
                message=f"Episode finalization failed: {str(e)}",
                component_name="EpisodeManager",
                operation_name="finalize_episode",
            ) from e

    def get_episode_statistics(self):
        return self.episode_statistics

    def get_performance_metrics(self) -> Dict[str, Any]:
        summary = self._performance_summary()
        resets = self.performance_metrics.get("episode_reset") or []
        return {
            "average_step_time_ms": summary.get("average_step_time_ms", 0.0),
            "total_processing_time_ms": summary.get("total_step_time_ms", 0.0),
            "reset_time_ms": resets[-1] if resets else 0.0,
        }

    def get_current_state(
        self,
        include_performance_data: bool = False,
        include_component_details: bool = False,
    ) -> Dict[str, object]:
        try:
            if not self.episode_active or self.current_episode_state is None:
                raise StateError(
                    message="Cannot get current state: episode is not active",
                    current_state="inactive",
                    expected_state="get_current_state",
                )
            from types import SimpleNamespace

            agent_state = self.state_manager.current_agent_state
            episode_state = dict(self.current_episode_state)
            start_time = episode_state.get("start_time", time.time())
            end_time = episode_state.get("end_time") or time.time()
            episode_state.setdefault(
                "step_count", agent_state.step_count if agent_state else 0
            )
            episode_state.setdefault("episode_duration", end_time - start_time)
            current_state = SimpleNamespace(
                episode_active=self.episode_active,
                episode_count=self.episode_count,
                episode_id=episode_state.get("episode_id"),
                agent_state=agent_state,
                episode_state=episode_state,
                timestamp=time.time(),
            )
            if include_performance_data:
                current_state.performance_data = (
                    self.performance_metrics.get_performance_summary()
                )
            if include_component_details:
                current_state.component_details = {
                    "reward_calculator": self.reward_calculator.get_reward_statistics(),
                    "action_processor": self.action_processor.get_processing_statistics(),
                    "component_cache_size": len(self.component_cache),
                }
            current_state.tracking_info = {
                "episode_id": episode_state.get("episode_id"),
                "episode_count": self.episode_count,
                "step_count": agent_state.step_count if agent_state else 0,
            }
            return current_state
        except StateError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to get current state: {e}")
            return {
                "episode_active": self.episode_active,
                "episode_count": self.episode_count,
                "error": f"State retrieval failed: {str(e)}",
                "timestamp": time.time(),
            }

    def validate_episode_consistency(  # noqa: C901
        self,
        strict_validation: bool = False,
        *,
        strict: Optional[bool] = None,
    ) -> bool:
        try:
            if strict is not None:
                strict_validation = bool(strict)
            validation_result: ConsistencyReport = {
                "is_consistent": True,
                "errors": [],
                "warnings": [],
            }
            if not self.episode_active and self.current_episode_state is not None:
                validation_result["warnings"].append(
                    "Episode state exists but episode not active"
                )
            agent_state = getattr(self.state_manager, "current_agent_state", None)
            if self.current_episode_state is not None and agent_state is not None:
                state_position = agent_state.position
                gs = cast(GridSize, self.config.env_config.grid_size)
                if not gs.contains(state_position):
                    validation_result["is_consistent"] = False
                    validation_result["errors"].append(
                        f"Agent position {state_position} outside grid bounds"
                    )
            if (
                self.current_episode_state is not None
                and (
                    bool(self.current_episode_state.get("terminated"))
                    or bool(self.current_episode_state.get("truncated"))
                )
                and self.episode_active
            ):
                validation_result["warnings"].append(
                    "Episode terminated/truncated but still marked as active"
                )
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
            self._last_consistency_report = validation_result
            return bool(validation_result["is_consistent"])
        except Exception:
            return False

    def cleanup(  # noqa: C901
        self, preserve_statistics: bool = True, clear_performance_data: bool = False
    ) -> None:
        try:
            self.logger.info("Starting EpisodeManager cleanup")
            if self.episode_active and self.current_episode_state:
                try:
                    self.finalize_episode(
                        collect_detailed_statistics=preserve_statistics
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to finalize active episode during cleanup: {e}"
                    )
            if self.state_manager:
                try:
                    self.state_manager.cleanup(
                        preserve_configuration=preserve_statistics,
                        clear_performance_data=clear_performance_data,
                    )
                except Exception as e:
                    self.logger.error(f"StateManager cleanup failed: {e}")
            self.component_cache.clear()
            if not preserve_statistics:
                self.episode_statistics = EpisodeStatistics(
                    statistics_id=f"episode_manager_{int(time.time() * 1000)}"
                )
            if clear_performance_data:
                self.performance_metrics.clear()
            self.episode_active = False
            self.current_episode_state = None
            if not preserve_statistics:
                self.episode_count = 0
            if hasattr(self.action_processor, "clear_cache"):
                self.action_processor.clear_cache()
            if hasattr(self.reward_calculator, "clear_cache"):
                self.reward_calculator.clear_cache()
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
    enable_performance_monitoring: Optional[bool] = None,
    enable_component_validation: Optional[bool] = None,
) -> EpisodeManager:
    try:
        if config is None:
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
                enable_performance_monitoring=(
                    enable_performance_monitoring
                    if enable_performance_monitoring is not None
                    else True
                ),
                enable_state_validation=(
                    enable_component_validation
                    if enable_component_validation is not None
                    else True
                ),
                enable_component_integration=(
                    enable_component_validation
                    if enable_component_validation is not None
                    else True
                ),
            )
        else:
            if enable_performance_monitoring is not None:
                config.enable_performance_monitoring = enable_performance_monitoring
            if enable_component_validation is not None:
                config.enable_state_validation = enable_component_validation
                if not enable_component_validation:
                    config.enable_component_integration = False
        config.validate()
        if seed_manager is None:
            seed_manager = SeedManager()
        episode_manager = EpisodeManager(config=config, seed_manager=seed_manager)
        validation_result = episode_manager.validate_episode_consistency(
            strict_validation=enable_component_validation
        )
        if (
            isinstance(validation_result, dict)
            and not validation_result.get("is_consistent", True)
            and enable_component_validation
        ):
            raise ComponentError(
                message="Episode manager setup validation failed",
                component_name="EpisodeManagerFactory",
                operation_name="create_episode_manager",
            )
        logger = get_component_logger("episode_manager")
        logger.info("EpisodeManager created successfully with component coordination")
        return episode_manager
    except ValidationError:
        raise
    except Exception as e:
        raise ComponentError(
            message=f"EpisodeManager creation failed: {str(e)}",
            component_name="EpisodeManagerFactory",
            operation_name="create_episode_manager",
        ) from e


def validate_episode_config(  # noqa: C901
    config: EpisodeManagerConfig,
    strict_validation: bool = False,
    validation_context: Optional[Dict[str, object]] = None,
) -> Tuple[bool, ValidationReport]:
    report: ValidationReport = {
        "is_valid": True,
        "validation_status": "valid",
        "validation_timestamp": time.time(),
        "strict_mode": strict_validation,
        "warnings": [],
        "errors": [],
        "parameter_analysis": {},
    }
    try:
        config.validate(strict_mode=strict_validation)
    except ValidationError as e:
        report["is_valid"] = False
        report["validation_status"] = "invalid"
        report["errors"].append(str(e))
        return False, report
    except Exception as e:
        report["is_valid"] = False
        report["validation_status"] = "error"
        report["errors"].append(str(e))
        return False, report
    grid_size = cast(GridSize, config.env_config.grid_size)
    max_steps = config.env_config.max_steps
    if max_steps > 100000:
        report["warnings"].append(
            f"Very high max_steps ({max_steps}) may impact performance"
        )
    if grid_size.width * grid_size.height > 500000:
        report["warnings"].append(
            f"Large grid size ({grid_size.width}x{grid_size.height}) may impact performance"
        )
    if config.enable_reproducibility_validation and not config.enable_state_validation:
        report["warnings"].append(
            "Reproducibility validation requires state validation"
        )
    report["parameter_analysis"] = {
        "grid_size": (grid_size.width, grid_size.height),
        "max_steps": max_steps,
        "goal_radius": config.env_config.goal_radius,
        "performance_monitoring_enabled": config.enable_performance_monitoring,
        "state_validation_enabled": config.enable_state_validation,
        "component_integration_enabled": config.enable_component_integration,
    }
    return True, report
