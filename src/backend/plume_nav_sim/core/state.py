"""
Core state management types for the plume navigation simulation.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .constants import DEFAULT_MAX_STEPS, PERFORMANCE_TRACKING_ENABLED
from .geometry import Coordinates


@dataclass
class AgentState:
    """Mutable data class for tracking agent state."""

    position: Coordinates
    step_count: int = 0
    total_reward: float = 0.0
    movement_history: List[Coordinates] = field(default_factory=list)
    goal_reached: bool = False
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        from ..utils.exceptions import ValidationError

        if not isinstance(self.position, Coordinates):
            raise ValidationError(
                f"Agent position must be Coordinates instance, got {type(self.position).__name__}"
            )

        # Contract: core_types.md - Invariants I2, I3
        if self.step_count < 0:
            raise ValidationError(
                f"step_count must be non-negative, got {self.step_count}"
            )

        if self.total_reward < 0:
            raise ValidationError(
                f"total_reward must be non-negative, got {self.total_reward}"
            )

        if PERFORMANCE_TRACKING_ENABLED:
            self.performance_metrics.setdefault("state_updates", 0)
            self.performance_metrics.setdefault("position_changes", 0)
            self.performance_metrics.setdefault("reward_changes", 0)

    def update_position(
        self, new_position: Coordinates, record_history: bool = True
    ) -> None:
        """Update agent position."""
        from ..utils.exceptions import ValidationError

        if not isinstance(new_position, Coordinates):
            raise ValidationError(
                f"New position must be Coordinates instance, got {type(new_position).__name__}"
            )

        if record_history:
            self.movement_history.append(self.position)

        self.position = new_position

        if PERFORMANCE_TRACKING_ENABLED:
            self.performance_metrics["position_changes"] += 1
            self.performance_metrics["state_updates"] += 1

    def add_reward(self, reward: float, validate_reward: bool = True) -> None:
        """Add reward to the total.

        Contract: core_types.md - Precondition P1 for add_reward()
        """
        from ..utils.exceptions import ValidationError

        if validate_reward:
            if not isinstance(reward, (int, float)):
                raise ValidationError(
                    f"Reward must be numeric, got {type(reward).__name__}"
                )
            # Contract: Cannot add negative reward
            if reward < 0:
                raise ValidationError(f"Cannot add negative reward, got {reward}")

        self.total_reward += float(reward)

        if PERFORMANCE_TRACKING_ENABLED:
            self.performance_metrics["reward_changes"] += 1
            self.performance_metrics["state_updates"] += 1

        if reward > 0:
            self.goal_reached = True

    def increment_step(self) -> None:
        """Increment the step count."""
        from ..utils.exceptions import StateError

        self.step_count += 1

        if self.step_count > DEFAULT_MAX_STEPS * 10:
            raise StateError(f"Step count {self.step_count} exceeds reasonable limits")

        if PERFORMANCE_TRACKING_ENABLED:
            self.performance_metrics["state_updates"] += 1

    def mark_goal_reached(self) -> None:
        """Mark goal as reached (idempotent).

        Contract: core_types.md - AgentState.mark_goal_reached()
        Postcondition: goal_reached = True
        Idempotent: Safe to call multiple times
        Constraint: Cannot un-reach goal (no method to set False)
        """
        self.goal_reached = True

    def reset(
        self,
        new_position: Optional[Coordinates] = None,
        preserve_performance_metrics: bool = False,
    ) -> None:
        """Reset the agent state for a new episode."""
        from ..utils.exceptions import ValidationError

        if new_position is not None:
            if not isinstance(new_position, Coordinates):
                raise ValidationError(
                    f"New position must be Coordinates instance, got {type(new_position).__name__}"
                )
            self.position = new_position

        self.step_count = 0
        self.total_reward = 0.0
        self.movement_history.clear()
        self.goal_reached = False

        if not preserve_performance_metrics:
            self.performance_metrics.clear()
        elif PERFORMANCE_TRACKING_ENABLED:
            self.performance_metrics["state_updates"] = 0
            self.performance_metrics["position_changes"] = 0
            self.performance_metrics["reward_changes"] = 0

    def get_trajectory(
        self, include_current_position: bool = True
    ) -> List[Coordinates]:
        """Get the agent's movement trajectory."""
        trajectory = self.movement_history.copy()
        if include_current_position:
            trajectory.append(self.position)
        return trajectory

    def calculate_trajectory_length(self) -> float:
        """Calculate the total distance traveled."""
        if len(self.movement_history) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(len(self.movement_history) - 1):
            total_distance += self.movement_history[i].distance_to(
                self.movement_history[i + 1]
            )

        if self.movement_history:
            total_distance += self.movement_history[-1].distance_to(self.position)

        return total_distance

    def to_dict(
        self, include_history: bool = False, include_performance_metrics: bool = False
    ) -> Dict[str, Any]:
        """Convert agent state to a dictionary."""
        state_dict = {
            "position": self.position.to_tuple(),
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "goal_reached": self.goal_reached,
        }
        if include_history:
            state_dict["movement_history"] = [
                pos.to_tuple() for pos in self.movement_history
            ]
        if include_performance_metrics:
            state_dict["performance_metrics"] = self.performance_metrics.copy()
        return state_dict


@dataclass
class EpisodeState:
    """Data class for managing the state of an episode."""

    agent_state: AgentState
    terminated: bool = False
    truncated: bool = False
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    episode_summary: Dict[str, Any] = field(default_factory=dict)
    state_history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        from ..utils.exceptions import ValidationError

        if not isinstance(self.agent_state, AgentState):
            raise ValidationError(
                f"Agent state must be an AgentState instance, got {type(self.agent_state).__name__}"
            )
        if not isinstance(self.terminated, bool) or not isinstance(
            self.truncated, bool
        ):
            raise ValidationError("Termination flags must be boolean")
        if self.terminated and self.truncated:
            raise ValidationError("Episode cannot be both terminated and truncated")

    def is_done(self) -> bool:
        """Check if the episode is complete."""
        return self.terminated or self.truncated

    def set_termination(
        self, terminated: bool, truncated: bool, reason: Optional[str] = None
    ) -> None:
        """Set the episode's termination status."""
        from ..utils.exceptions import ValidationError

        if terminated and truncated:
            raise ValidationError("Episode cannot be both terminated and truncated")

        self.terminated = terminated
        self.truncated = truncated

        if self.is_done() and self.end_time is None:
            self.end_time = time.time()

        if reason:
            self.episode_summary["termination_reason"] = reason

        if terminated:
            self.agent_state.goal_reached = True

    def get_episode_duration(self) -> float:
        """Calculate the episode duration."""
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time

    def get_episode_summary(
        self,
        include_trajectory_analysis: bool = False,
        include_performance_metrics: bool = False,
    ) -> Dict[str, Any]:
        """Generate a summary of the episode."""
        summary = {
            "episode_id": self.episode_id,
            "duration_seconds": self.get_episode_duration(),
            "step_count": self.agent_state.step_count,
            "total_reward": self.agent_state.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "goal_reached": self.agent_state.goal_reached,
            "final_position": self.agent_state.position.to_tuple(),
        }
        if include_trajectory_analysis:
            trajectory = self.agent_state.get_trajectory()
            summary["trajectory_analysis"] = {
                "total_distance": self.agent_state.calculate_trajectory_length(),
                "position_count": len(trajectory),
                "start_position": trajectory[0].to_tuple() if trajectory else None,
                "end_position": trajectory[-1].to_tuple() if trajectory else None,
            }
        if include_performance_metrics:
            summary["performance_metrics"] = self.agent_state.performance_metrics.copy()

        summary.update(self.episode_summary)
        return summary

    def record_state(self, additional_context: Optional[Dict[str, Any]] = None) -> None:
        """Record the current state in the history."""
        snapshot = {
            "timestamp": time.time(),
            "step_count": self.agent_state.step_count,
            "agent_position": self.agent_state.position.to_tuple(),
            "total_reward": self.agent_state.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }
        if additional_context:
            snapshot.update(additional_context)
        self.state_history.append(snapshot)

    def reset(
        self, new_agent_state: AgentState, preserve_episode_id: bool = False
    ) -> None:
        """Reset the episode state."""
        from ..utils.exceptions import ValidationError

        if not isinstance(new_agent_state, AgentState):
            raise ValidationError(
                f"New agent state must be an AgentState instance, got {type(new_agent_state).__name__}"
            )

        self.agent_state = new_agent_state
        self.terminated = False
        self.truncated = False
        self.start_time = time.time()
        self.end_time = None
        self.episode_summary.clear()
        self.state_history.clear()
        if not preserve_episode_id:
            self.episode_id = str(uuid.uuid4())
