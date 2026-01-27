import copy
import math
import time
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

# Standard library imports with version comments
import numpy as np

from ..utils.exceptions import ComponentError, ValidationError
from ..utils.logging import get_component_logger, monitor_performance

# Internal utility imports for validation, exceptions, and logging
from ..utils.validation import validate_coordinates

# Internal core imports for constants, types, and mathematical operations
from .constants import (
    DEFAULT_GOAL_RADIUS,
    DISTANCE_PRECISION,
    PERFORMANCE_TARGET_STEP_LATENCY_MS,
    REWARD_DEFAULT,
    REWARD_GOAL_REACHED,
)
from .geometry import Coordinates, calculate_euclidean_distance
from .state import AgentState

REWARD_CALCULATOR_VERSION = "1.0.0"
DEFAULT_DISTANCE_CALCULATION_METHOD = "euclidean"
REWARD_CALCULATION_CACHE_SIZE = 1000
DISTANCE_CALCULATION_PRECISION = 1e-12

__all__ = [
    "RewardCalculator",
    "RewardCalculatorConfig",
    "RewardResult",
    "TerminationResult",
    "create_reward_calculator",
    "validate_reward_config",
]


@dataclass
class RewardCalculatorConfig:
    # Primary reward calculation parameters
    goal_radius: float
    reward_goal_reached: float
    reward_default: float

    # Advanced configuration parameters with defaults
    distance_calculation_method: str = field(
        default=DEFAULT_DISTANCE_CALCULATION_METHOD
    )
    distance_precision: float = field(default=DISTANCE_CALCULATION_PRECISION)
    enable_performance_monitoring: bool = field(default=True)
    enable_caching: bool = field(default=True)
    custom_parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate goal_radius is non-negative float with reasonable bounds for goal detection feasibility
        if not isinstance(self.goal_radius, (int, float)) or self.goal_radius < 0:
            raise ValidationError(
                "goal_radius must be non-negative numeric value",
                parameter_name="goal_radius",
                parameter_value=self.goal_radius,
            )

        if not math.isfinite(self.goal_radius):
            raise ValidationError(
                "goal_radius must be finite (not NaN or infinity)",
                parameter_name="goal_radius",
                parameter_value=self.goal_radius,
            )

        # Store reward_goal_reached value for successful goal achievement signaling
        if not isinstance(self.reward_goal_reached, (int, float)):
            raise ValidationError(
                "reward_goal_reached must be numeric value",
                parameter_name="reward_goal_reached",
                parameter_value=self.reward_goal_reached,
            )

        if not math.isfinite(self.reward_goal_reached):
            raise ValidationError(
                "reward_goal_reached must be finite (not NaN or infinity)",
                parameter_name="reward_goal_reached",
                parameter_value=self.reward_goal_reached,
            )

        # Store reward_default value for standard sparse reward structure implementation
        if not isinstance(self.reward_default, (int, float)):
            raise ValidationError(
                "reward_default must be numeric value",
                parameter_name="reward_default",
                parameter_value=self.reward_default,
            )

        if not math.isfinite(self.reward_default):
            raise ValidationError(
                "reward_default must be finite (not NaN or infinity)",
                parameter_name="reward_default",
                parameter_value=self.reward_default,
            )

        # Validate distance_calculation_method is supported and mathematically sound
        valid_methods = ["euclidean", "manhattan", "chebyshev"]
        if self.distance_calculation_method not in valid_methods:
            self.distance_calculation_method = DEFAULT_DISTANCE_CALCULATION_METHOD

        # Ensure distance_precision enables accurate goal detection without numerical instability
        if (
            not isinstance(self.distance_precision, (int, float))
            or self.distance_precision <= 0
        ):
            self.distance_precision = DISTANCE_CALCULATION_PRECISION

        # Initialize custom_parameters dictionary for extensibility and configuration flexibility
        if not isinstance(self.custom_parameters, dict):
            self.custom_parameters = {}

    def validate(
        self, strict_mode: bool = False, validation_context: Optional[Dict] = None
    ) -> bool:
        # Validate goal_radius is non-negative and within reasonable bounds for mathematical precision
        if self.goal_radius < 0 or self.goal_radius > 1000:
            raise ValidationError(
                "goal_radius must be between 0 and 1000 for practical goal detection",
                parameter_name="goal_radius",
                parameter_value=self.goal_radius,
                parameter_constraints={"min": 0, "max": 1000},
            )

        # Check reward values are proper floats with reward_goal_reached != reward_default for learning signal
        if abs(self.reward_goal_reached - self.reward_default) < DISTANCE_PRECISION:
            raise ValidationError(
                "reward_goal_reached must differ from reward_default for learning signal",
                parameter_name="reward_values",
                parameter_value=(self.reward_goal_reached, self.reward_default),
            )

        # Validate distance_calculation_method is supported and mathematically sound
        valid_methods = ["euclidean", "manhattan", "chebyshev"]
        if self.distance_calculation_method not in valid_methods:
            raise ValidationError(
                "distance_calculation_method must be supported calculation method",
                parameter_name="distance_calculation_method",
                parameter_value=self.distance_calculation_method,
                parameter_constraints={"valid_methods": valid_methods},
            )

        # Ensure distance_precision enables accurate goal detection without numerical instability
        if self.distance_precision <= 0 or self.distance_precision > 1e-3:
            raise ValidationError(
                "distance_precision must be positive and sufficiently small",
                parameter_name="distance_precision",
                parameter_value=self.distance_precision,
                parameter_constraints={"min": 1e-15, "max": 1e-3},
            )

        # Apply strict validation rules including edge case analysis if strict_mode enabled
        if strict_mode:
            # Additional validation for high-precision requirements
            if (
                self.goal_radius > 0
                and self.goal_radius < self.distance_precision * 100
            ):
                raise ValidationError(
                    "goal_radius too small relative to distance_precision",
                    parameter_name="goal_radius_precision_ratio",
                    parameter_value=self.goal_radius / self.distance_precision,
                )

            # Validate custom_parameters for mathematical consistency
            if self.custom_parameters:
                for param_name, param_value in self.custom_parameters.items():
                    if isinstance(param_value, (int, float)) and not math.isfinite(
                        param_value
                    ):
                        raise ValidationError(
                            f"custom parameter {param_name} must be finite",
                            parameter_name=f"custom_parameters.{param_name}",
                            parameter_value=param_value,
                        )

        # Cross-validate all parameters for mathematical coherence and numerical stability
        # Validate computational requirements against performance constraints and system resources
        return True

    def estimate_performance_impact(
        self, expected_episode_length: int, include_caching_benefits: bool = True
    ) -> Dict[str, Any]:
        # Calculate base reward calculation time based on distance computation complexity
        base_calculation_time_ms = 0.01  # Base calculation overhead

        if self.distance_calculation_method == "euclidean":
            base_calculation_time_ms += 0.02  # Square root calculation
        elif self.distance_calculation_method == "manhattan":
            base_calculation_time_ms += 0.01  # Simple addition
        elif self.distance_calculation_method == "chebyshev":
            base_calculation_time_ms += 0.015  # Max operation

        # Estimate caching benefits if enable_caching is True and include_caching_benefits requested
        caching_benefit_factor = 1.0
        if self.enable_caching and include_caching_benefits:
            # Assume 70% cache hit rate for typical navigation patterns
            caching_benefit_factor = 0.3 + (0.7 * 0.1)  # 70% hits take 10% of time

        # Project total reward calculation overhead for expected_episode_length steps
        per_step_time_ms = base_calculation_time_ms * caching_benefit_factor
        total_episode_overhead_ms = per_step_time_ms * expected_episode_length

        # Analyze precision impact on computation time with distance_precision settings
        precision_overhead_factor = 1.0
        if self.distance_precision < 1e-10:
            precision_overhead_factor = (
                1.2  # Higher precision requires more computation
            )

        return {
            "per_step_calculation_ms": per_step_time_ms * precision_overhead_factor,
            "total_episode_overhead_ms": total_episode_overhead_ms
            * precision_overhead_factor,
            "caching_benefit_factor": caching_benefit_factor,
            "performance_target_compliance": per_step_time_ms
            < PERFORMANCE_TARGET_STEP_LATENCY_MS / 2,
            "optimization_recommendations": [
                (
                    "Enable caching for repeated calculations"
                    if not self.enable_caching
                    else "Caching already enabled"
                ),
                (
                    "Consider reducing precision if performance critical"
                    if self.distance_precision < 1e-12
                    else "Precision appropriate"
                ),
                (
                    "Use euclidean distance for best balance"
                    if self.distance_calculation_method != "euclidean"
                    else "Optimal method selected"
                ),
            ],
        }

    def clone(
        self, overrides: Optional[Dict] = None, preserve_custom_parameters: bool = True
    ) -> "RewardCalculatorConfig":
        # Create deep copy of current configuration for parameter isolation
        cloned_config = RewardCalculatorConfig(
            goal_radius=self.goal_radius,
            reward_goal_reached=self.reward_goal_reached,
            reward_default=self.reward_default,
            distance_calculation_method=self.distance_calculation_method,
            distance_precision=self.distance_precision,
            enable_performance_monitoring=self.enable_performance_monitoring,
            enable_caching=self.enable_caching,
            custom_parameters=(
                copy.deepcopy(self.custom_parameters)
                if preserve_custom_parameters
                else {}
            ),
        )

        # Apply parameter overrides if provided with type validation and consistency checking
        if overrides:
            for param_name, param_value in overrides.items():
                if hasattr(cloned_config, param_name):
                    value = (
                        copy.deepcopy(param_value)
                        if isinstance(param_value, (dict, list, set, tuple))
                        else param_value
                    )
                    setattr(cloned_config, param_name, value)
                else:
                    # Add to custom_parameters if not a standard parameter
                    cloned_config.custom_parameters[param_name] = copy.deepcopy(
                        param_value
                    )

        cloned_config.validate()

        return cloned_config

    def to_dict(self, include_performance_estimates: bool = False) -> Dict[str, Any]:
        config_dict = {
            "goal_radius": self.goal_radius,
            "reward_goal_reached": self.reward_goal_reached,
            "reward_default": self.reward_default,
            "distance_calculation_method": self.distance_calculation_method,
            "distance_precision": self.distance_precision,
            "enable_performance_monitoring": self.enable_performance_monitoring,
            "enable_caching": self.enable_caching,
            "custom_parameters": self.custom_parameters.copy(),
            "version": REWARD_CALCULATOR_VERSION,
        }

        # Add performance estimates if include_performance_estimates is True
        if include_performance_estimates:
            config_dict["performance_estimates"] = self.estimate_performance_impact(
                1000, True
            )

        return config_dict


@dataclass
class RewardResult:
    # Core reward calculation results
    reward: float
    goal_reached: bool
    distance_to_goal: float

    # Performance and debugging information
    calculation_time_ms: float = field(default=0.0)
    calculation_details: Dict[str, Any] = field(default_factory=dict)
    goal_achievement_reason: Optional[str] = field(default=None)

    def __post_init__(self):
        # Validate reward value is numeric
        if not isinstance(self.reward, (int, float)) or not math.isfinite(self.reward):
            raise ValidationError(
                "reward must be finite numeric value",
                parameter_name="reward",
                parameter_value=self.reward,
            )

        # Validate goal_reached is boolean
        if not isinstance(self.goal_reached, bool):
            raise ValidationError(
                "goal_reached must be boolean value",
                parameter_name="goal_reached",
                parameter_value=self.goal_reached,
            )

        # Validate distance_to_goal is non-negative numeric
        if (
            not isinstance(self.distance_to_goal, (int, float))
            or self.distance_to_goal < 0
        ):
            raise ValidationError(
                "distance_to_goal must be non-negative numeric value",
                parameter_name="distance_to_goal",
                parameter_value=self.distance_to_goal,
            )

    def set_performance_metrics(
        self, calculation_time_ms: float, calculation_details: Optional[Dict] = None
    ) -> None:
        # Store calculation_time_ms for performance monitoring and optimization
        if isinstance(calculation_time_ms, (int, float)) and calculation_time_ms >= 0:
            self.calculation_time_ms = float(calculation_time_ms)

        # Store calculation_details if provided with computational context
        if calculation_details and isinstance(calculation_details, dict):
            self.calculation_details.update(calculation_details)

        # Validate performance metrics against performance targets
        if self.calculation_time_ms > PERFORMANCE_TARGET_STEP_LATENCY_MS:
            self.calculation_details["performance_warning"] = "Exceeded target latency"

    def to_dict(self, include_performance_data: bool = False) -> Dict[str, Any]:
        result_dict = {
            "reward": self.reward,
            "goal_reached": self.goal_reached,
            "distance_to_goal": self.distance_to_goal,
        }

        # Include goal_achievement_reason if available for debugging context
        if self.goal_achievement_reason:
            result_dict["goal_achievement_reason"] = self.goal_achievement_reason

        # Add performance data if include_performance_data is True
        if include_performance_data:
            result_dict["calculation_time_ms"] = self.calculation_time_ms
            result_dict["calculation_details"] = self.calculation_details

        return result_dict


@dataclass
class TerminationResult:
    # Core termination flags for Gymnasium API compliance
    terminated: bool
    truncated: bool
    termination_reason: str

    # Additional episode completion information
    final_distance: Optional[float] = field(default=None)
    final_step_count: Optional[int] = field(default=None)
    termination_details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate terminated and truncated flags are boolean and mutually exclusive when appropriate
        if not isinstance(self.terminated, bool):
            raise ValidationError(
                "terminated must be boolean value",
                parameter_name="terminated",
                parameter_value=self.terminated,
            )

        if not isinstance(self.truncated, bool):
            raise ValidationError(
                "truncated must be boolean value",
                parameter_name="truncated",
                parameter_value=self.truncated,
            )

        # Validate termination_reason is provided for episode analysis
        if not self.termination_reason or not isinstance(self.termination_reason, str):
            raise ValidationError(
                "termination_reason must be non-empty string",
                parameter_name="termination_reason",
                parameter_value=self.termination_reason,
            )

    def set_final_state(
        self,
        final_distance: float,
        final_step_count: int,
        additional_details: Optional[Dict] = None,
    ) -> None:
        # Store final_distance for episode performance analysis
        if isinstance(final_distance, (int, float)):
            self.final_distance = float(final_distance)
            if self.final_distance < 0:
                self.final_distance = 0.0

        # Store final_step_count for episode statistics and optimization
        if isinstance(final_step_count, int):
            self.final_step_count = max(0, final_step_count)

        # Add additional_details to termination_details if provided
        if additional_details and isinstance(additional_details, dict):
            self.termination_details.update(additional_details)

    def get_summary(self) -> str:
        # Format termination status with terminated/truncated flags
        status_parts = []
        if self.terminated:
            status_parts.append("TERMINATED")
        if self.truncated:
            status_parts.append("TRUNCATED")

        status_str = " | ".join(status_parts) if status_parts else "ONGOING"

        # Include termination_reason and episode completion context
        summary_parts = [f"Status: {status_str}", f"Reason: {self.termination_reason}"]

        # Add final distance and step count if available
        if self.final_distance is not None:
            summary_parts.append(f"Final Distance: {self.final_distance:.3f}")

        if self.final_step_count is not None:
            summary_parts.append(f"Steps: {self.final_step_count}")

        return " | ".join(summary_parts)


class RewardCalculator:
    def __init__(
        self,
        config: RewardCalculatorConfig,
    ):
        if not isinstance(config, RewardCalculatorConfig):
            raise ValidationError(
                "config must be RewardCalculatorConfig instance",
                parameter_name="config",
                parameter_value=type(config),
            )

        config.validate()

        # Store configuration for reward calculation parameter access and validation reference
        self.config = config

        # Store performance timings locally (simple list-based aggregation)
        self.performance_metrics: Dict[str, List[float]] = {}

        # Create distance_cache dictionary for caching repeated distance calculations
        self.distance_cache: Dict[str, float] = {}

        # Create component logger for reward calculation operations and debugging
        self.logger = get_component_logger("reward_calculator")

        # Initialize calculation_count to 0 for operation tracking and statistics
        self.calculation_count = 0

        # Initialize goal_achievement_stats dictionary for episode success analysis
        self.goal_achievement_stats = {
            "total_calculations": 0,
            "goals_achieved": 0,
            "average_distance": 0.0,
            "performance_violations": 0,
        }

        self.logger.info(
            f"RewardCalculator initialized with config: goal_radius={config.goal_radius}, "
            f"reward_goal={config.reward_goal_reached}, reward_default={config.reward_default}"
        )

    @monitor_performance(
        "reward_calculation", PERFORMANCE_TARGET_STEP_LATENCY_MS, False
    )
    def calculate_reward(
        self,
        agent_position: Coordinates,
        source_location: Coordinates,
        step_count: Optional[int] = None,
        calculation_context: Optional[Dict] = None,
    ) -> RewardResult:
        start_time = time.perf_counter()

        try:
            # Validate agent_position and source_location using coordinate validation
            validate_coordinates(agent_position)
            validate_coordinates(source_location)

            # Calculate distance from agent to source using optimized distance calculation method
            distance = self.get_distance_to_goal(
                agent_position, source_location, use_cache=self.config.enable_caching
            )

            # Determine goal achievement using distance <= config.goal_radius comparison
            goal_reached = bool(distance <= self.config.goal_radius)

            # Set reward to config.reward_goal_reached if goal achieved, config.reward_default otherwise
            reward = (
                self.config.reward_goal_reached
                if goal_reached
                else self.config.reward_default
            )

            # Create RewardResult with reward, goal_reached flag, and distance information
            result = RewardResult(
                reward=reward, goal_reached=goal_reached, distance_to_goal=distance
            )

            # Set goal achievement reason for debugging and analysis
            if goal_reached:
                result.goal_achievement_reason = f"Agent within goal radius: {distance:.6f} <= {self.config.goal_radius}"

            # Record performance timing if performance_metrics enabled
            end_time = time.perf_counter()
            calculation_time_ms = (end_time - start_time) * 1000

            if self.config.enable_performance_monitoring:
                result.set_performance_metrics(
                    calculation_time_ms,
                    {
                        "distance_calculation_method": self.config.distance_calculation_method,
                        "cache_enabled": self.config.enable_caching,
                        "goal_radius": self.config.goal_radius,
                    },
                )
                self.performance_metrics.setdefault("reward_calculation", []).append(
                    float(calculation_time_ms)
                )

            # Update calculation_count and goal_achievement_stats for statistics tracking
            self.calculation_count += 1
            self._update_statistics(distance, goal_reached, calculation_time_ms)

            # Log reward calculation with context and performance metrics if debug enabled
            self.logger.debug(
                f"Reward calculated: reward={reward}, goal_reached={goal_reached}, "
                f"distance={distance:.6f}, time={calculation_time_ms:.3f}ms"
            )

            return result

        except ValidationError:
            # Re-raise validation errors directly without wrapping
            raise
        except Exception as e:
            # Handle calculation errors and provide fallback result
            self.logger.error(f"Reward calculation failed: {e}")
            self.goal_achievement_stats["performance_violations"] += 1
            raise ComponentError(
                f"Reward calculation failed: {e}",
                component_name="RewardCalculator",
                operation_name="calculate_reward",
                underlying_error=e,
            ) from e

    def check_termination(
        self,
        agent_state: AgentState,
        source_location: Coordinates,
        max_steps: int,
        termination_context: Optional[Dict] = None,
    ) -> TerminationResult:
        try:
            # Validate agent_state contains position, step_count, and goal_reached information
            if not hasattr(agent_state, "position") or not isinstance(
                agent_state.position, Coordinates
            ):
                raise ValidationError(
                    "agent_state must have valid position coordinates",
                    parameter_name="agent_state.position",
                    parameter_value=getattr(agent_state, "position", None),
                )

            # Calculate current distance from agent position to source location
            current_distance = self.get_distance_to_goal(
                agent_state.position, source_location
            )

            # Check goal achievement using distance <= config.goal_radius for termination
            goal_achieved = bool(current_distance <= self.config.goal_radius)

            # Check step limit truncation using agent_state.step_count >= max_steps
            step_limit_reached = agent_state.step_count >= max_steps

            # Determine termination reason based on goal achievement or step limit conditions
            terminated = goal_achieved
            truncated = step_limit_reached and not goal_achieved

            if goal_achieved:
                termination_reason = f"Goal achieved: distance {current_distance:.6f} <= radius {self.config.goal_radius}"
            elif step_limit_reached:
                termination_reason = (
                    f"Step limit reached: {agent_state.step_count} >= {max_steps}"
                )
            else:
                termination_reason = "Episode ongoing"

            # Create TerminationResult with appropriate terminated/truncated flags
            result = TerminationResult(
                terminated=terminated,
                truncated=truncated,
                termination_reason=termination_reason,
            )

            # Set final state information including distance and step count
            result.set_final_state(
                current_distance,
                agent_state.step_count,
                {
                    "goal_radius": self.config.goal_radius,
                    "max_steps": max_steps,
                    "termination_context": termination_context,
                },
            )

            # Update goal_achievement_stats with termination outcome for analysis
            if terminated:
                self.goal_achievement_stats["goals_achieved"] += 1

            # Log termination check results with goal detection status and episode analysis
            self.logger.info(
                f"Termination check: terminated={terminated}, truncated={truncated}, "
                f"distance={current_distance:.6f}, steps={agent_state.step_count}"
            )

            return result

        except ValidationError:
            # Re-raise validation errors directly without wrapping
            raise
        except Exception as e:
            self.logger.error(f"Termination check failed: {e}")
            raise ComponentError(
                f"Termination check failed: {e}",
                component_name="RewardCalculator",
                operation_name="check_termination",
                underlying_error=e,
            ) from e

    def update_agent_reward(
        self,
        agent_state: AgentState,
        reward_result: RewardResult,
        update_goal_status: bool = True,
    ) -> None:
        try:
            # Validate agent_state and reward_result for proper state coordination
            if not hasattr(agent_state, "add_reward"):
                raise ValidationError(
                    "agent_state must have add_reward method",
                    parameter_name="agent_state",
                    parameter_value=type(agent_state),
                )

            if not isinstance(reward_result, RewardResult):
                raise ValidationError(
                    "reward_result must be RewardResult instance",
                    parameter_name="reward_result",
                    parameter_value=type(reward_result),
                )

            # Add reward to agent state using agent_state.add_reward method
            agent_state.add_reward(reward_result.reward)

            # Update goal_reached flag if update_goal_status enabled and goal achieved
            if update_goal_status and hasattr(agent_state, "goal_reached"):
                agent_state.goal_reached = reward_result.goal_reached

            # Log reward update with agent state and reward information if debug enabled
            self.logger.debug(
                f"Agent reward updated: reward={reward_result.reward}, "
                f"total_reward={getattr(agent_state, 'total_reward', 'N/A')}, "
                f"goal_reached={reward_result.goal_reached}"
            )

        except ValidationError:
            # Re-raise validation errors directly without wrapping
            raise
        except Exception as e:
            self.logger.error(f"Agent reward update failed: {e}")
            raise ComponentError(
                f"Agent reward update failed: {e}",
                component_name="RewardCalculator",
                operation_name="update_agent_reward",
                underlying_error=e,
            ) from e

    def get_distance_to_goal(
        self,
        agent_position: Coordinates,
        goal_position: Coordinates,
        use_cache: bool = True,
        high_precision: bool = False,
    ) -> float:
        # Generate cache key from agent_position and goal_position coordinates
        cache_key = f"{agent_position.x},{agent_position.y}->{goal_position.x},{goal_position.y}"

        # Check distance_cache for existing calculation if use_cache enabled
        if use_cache and cache_key in self.distance_cache:
            return self.distance_cache[cache_key]

        # Calculate distance using calculate_euclidean_distance with precision settings
        if self.config.distance_calculation_method == "euclidean":
            # Use standard helper to preserve established performance characteristics
            distance = calculate_euclidean_distance(agent_position, goal_position)
        elif self.config.distance_calculation_method == "manhattan":
            distance = abs(agent_position.x - goal_position.x) + abs(
                agent_position.y - goal_position.y
            )
        elif self.config.distance_calculation_method == "chebyshev":
            distance = max(
                abs(agent_position.x - goal_position.x),
                abs(agent_position.y - goal_position.y),
            )
        else:
            # Fallback to euclidean distance
            distance = calculate_euclidean_distance(agent_position, goal_position)

        # Apply high precision calculations if high_precision enabled
        if high_precision:
            # Use numpy for higher precision floating point operations
            distance = float(
                np.sqrt(
                    np.power(float(agent_position.x - goal_position.x), 2)
                    + np.power(float(agent_position.y - goal_position.y), 2)
                )
            )

        # Cache calculated distance if use_cache enabled and cache not full
        if use_cache and len(self.distance_cache) < REWARD_CALCULATION_CACHE_SIZE:
            self.distance_cache[cache_key] = distance

        return distance

    def get_reward_statistics(
        self,
        include_performance_analysis: bool = True,
        include_cache_statistics: bool = True,
    ) -> Dict[str, Any]:
        stats = {
            "total_calculations": self.calculation_count,
            "goals_achieved": self.goal_achievement_stats["goals_achieved"],
            "goal_achievement_rate": 0.0,
            "average_distance": self.goal_achievement_stats["average_distance"],
            "performance_violations": self.goal_achievement_stats[
                "performance_violations"
            ],
            "configuration": {
                "goal_radius": self.config.goal_radius,
                "reward_goal_reached": self.config.reward_goal_reached,
                "reward_default": self.config.reward_default,
                "distance_method": self.config.distance_calculation_method,
                "caching_enabled": self.config.enable_caching,
            },
        }

        # Calculate goal achievement rate and episode success patterns
        if self.calculation_count > 0:
            stats["goal_achievement_rate"] = (
                self.goal_achievement_stats["goals_achieved"] / self.calculation_count
            )

        # Include cache hit/miss ratios and efficiency if include_cache_statistics True
        if include_cache_statistics and self.config.enable_caching:
            cache_size = len(self.distance_cache)
            cache_capacity = REWARD_CALCULATION_CACHE_SIZE
            stats["cache_statistics"] = {
                "cache_size": cache_size,
                "cache_capacity": cache_capacity,
                "cache_utilization": (
                    cache_size / cache_capacity if cache_capacity > 0 else 0.0
                ),
            }

        # Include performance analysis with timing metrics if include_performance_analysis True
        if include_performance_analysis:
            performance_compliance_rate = 1.0
            if self.calculation_count > 0:
                performance_compliance_rate = 1.0 - (
                    self.goal_achievement_stats["performance_violations"]
                    / self.calculation_count
                )

            stats["performance_analysis"] = {
                "performance_compliance_rate": performance_compliance_rate,
                "target_latency_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS / 2,
                "monitoring_enabled": self.config.enable_performance_monitoring,
            }

        return stats

    def clear_cache(self) -> Dict[str, Any]:
        # Calculate current cache size and memory usage before clearing
        cache_size_before = len(self.distance_cache)

        # Clear distance_cache dictionary and reset cache-related statistics
        self.distance_cache.clear()

        # Log cache clearing operation with memory freed and performance reset
        self.logger.info(f"Distance cache cleared: {cache_size_before} entries removed")

        return {
            "cache_entries_cleared": cache_size_before,
            "cache_size_after": len(self.distance_cache),
            "memory_freed_estimate_kb": cache_size_before * 0.1,  # Rough estimate
        }

    def validate_calculation_parameters(
        self,
        agent_position: Coordinates,
        source_location: Coordinates,
        validation_context: Optional[Dict] = None,
    ) -> bool:
        try:
            # Validate agent_position using validate_coordinates with bounds checking
            validate_coordinates(agent_position)

            # Validate source_location coordinates for mathematical consistency
            validate_coordinates(source_location)

            # Check coordinate separation for numerical precision requirements
            distance = abs(agent_position.x - source_location.x) + abs(
                agent_position.y - source_location.y
            )
            if distance < self.config.distance_precision:
                self.logger.warning(
                    f"Agent and source very close: distance={distance}, precision={self.config.distance_precision}"
                )

            # Apply validation context rules if provided with additional constraints
            if validation_context:
                max_distance = validation_context.get("max_distance")
                if max_distance and distance > max_distance:
                    raise ValidationError(
                        f"Distance {distance} exceeds maximum {max_distance}",
                        parameter_name="coordinate_separation",
                        parameter_value=distance,
                    )

            return True

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                f"Parameter validation failed: {e}",
                parameter_name="validation_parameters",
                parameter_value=(agent_position, source_location),
            ) from e

    def _update_statistics(
        self, distance: float, goal_reached: bool, calculation_time_ms: float
    ) -> None:
        self.goal_achievement_stats["total_calculations"] += 1

        # Update running average distance
        current_avg = self.goal_achievement_stats["average_distance"]
        current_count = self.goal_achievement_stats["total_calculations"]
        self.goal_achievement_stats["average_distance"] = (
            current_avg * (current_count - 1) + distance
        ) / current_count

        # Track goal achievements and success rate
        if goal_reached:
            self.goal_achievement_stats["goals_achieved"] += 1

        calculations = self.goal_achievement_stats["total_calculations"]
        if calculations > 0:
            self.goal_achievement_stats["goal_achievement_rate"] = (
                self.goal_achievement_stats["goals_achieved"] / calculations
            )

        # Track performance violations
        if calculation_time_ms > PERFORMANCE_TARGET_STEP_LATENCY_MS:
            self.goal_achievement_stats["performance_violations"] += 1


def create_reward_calculator(
    config: Optional[RewardCalculatorConfig] = None,
    enable_performance_monitoring: bool = True,
    enable_validation: bool = True,
) -> RewardCalculator:
    # Create default RewardCalculatorConfig if none provided with system constants and parameter validation
    if config is None:
        config = RewardCalculatorConfig(
            goal_radius=DEFAULT_GOAL_RADIUS,
            reward_goal_reached=REWARD_GOAL_REACHED,
            reward_default=REWARD_DEFAULT,
        )

    if enable_validation:
        config.validate(strict_mode=True)

    # Enable performance monitoring if requested with timing collection and analysis setup
    config.enable_performance_monitoring = enable_performance_monitoring

    # Initialize RewardCalculator with validated configuration and component dependencies
    reward_calculator = RewardCalculator(config)

    # Validate complete reward calculator setup and mathematical consistency
    if enable_validation:
        test_agent_pos = Coordinates(0, 0)
        test_source_pos = Coordinates(1, 1)
        reward_calculator.validate_calculation_parameters(
            test_agent_pos, test_source_pos
        )

    return reward_calculator


def validate_reward_config(  # noqa: C901
    config: RewardCalculatorConfig,
    strict_validation: bool = False,
    validation_context: Optional[Dict] = None,
) -> Tuple[bool, Dict[str, Any]]:
    validation_report = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "recommendations": [],
        "parameter_analysis": {},
    }

    try:
        # Validate goal_radius is non-negative float enabling reachable goal conditions
        if not isinstance(config.goal_radius, (int, float)) or config.goal_radius < 0:
            validation_report["is_valid"] = False
            validation_report["errors"].append(
                "goal_radius must be non-negative numeric value"
            )
        elif config.goal_radius > 100:
            validation_report["warnings"].append(
                "Large goal_radius may affect learning convergence"
            )

        # Validate reward values are proper floats with consistency checking
        if not isinstance(config.reward_goal_reached, (int, float)):
            validation_report["is_valid"] = False
            validation_report["errors"].append(
                "reward_goal_reached must be numeric value"
            )

        if not isinstance(config.reward_default, (int, float)):
            validation_report["is_valid"] = False
            validation_report["errors"].append("reward_default must be numeric value")

        # Check mathematical consistency between goal radius and distance calculation precision
        if (
            config.goal_radius > 0
            and config.goal_radius < config.distance_precision * 100
        ):
            validation_report["warnings"].append(
                "goal_radius may be too small relative to distance_precision"
            )

        # Validate performance requirements and computational feasibility
        performance_estimate = config.estimate_performance_impact(1000, True)
        if not performance_estimate["performance_target_compliance"]:
            validation_report["warnings"].append(
                "Configuration may not meet performance targets"
            )

        # Apply strict validation rules if strict_validation enabled
        if strict_validation:
            if (
                abs(config.reward_goal_reached - config.reward_default)
                < DISTANCE_PRECISION
            ):
                validation_report["is_valid"] = False
                validation_report["errors"].append(
                    "Reward values too similar - insufficient learning signal"
                )

        # Generate optimization recommendations
        validation_report["recommendations"].extend(
            [
                (
                    "Enable caching for better performance"
                    if not config.enable_caching
                    else "Caching enabled"
                ),
                (
                    "Consider euclidean distance for optimal balance"
                    if config.distance_calculation_method != "euclidean"
                    else "Optimal distance method"
                ),
                (
                    "Performance monitoring recommended"
                    if not config.enable_performance_monitoring
                    else "Performance monitoring enabled"
                ),
            ]
        )

        validation_report["parameter_analysis"] = {
            "goal_radius": config.goal_radius,
            "reward_spread": abs(config.reward_goal_reached - config.reward_default),
            "distance_method": config.distance_calculation_method,
            "performance_estimate": performance_estimate,
        }

    except Exception as e:
        validation_report["is_valid"] = False
        validation_report["errors"].append(f"Validation failed: {e}")

    return validation_report["is_valid"], validation_report
