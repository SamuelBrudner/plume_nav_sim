"""
Core snapshot types for the plume navigation simulation.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict

from .models import PlumeModel
from .state import AgentState, EpisodeState


@dataclass(frozen=True)
class StateSnapshot:
    """Data class for capturing a snapshot of the simulation state."""

    timestamp: float
    agent_state: AgentState
    episode_state: EpisodeState
    plume_model: PlumeModel
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        from ..utils.exceptions import ValidationError

        if not isinstance(self.agent_state, AgentState):
            raise ValidationError(
                f"Agent state must be an AgentState instance, got {type(self.agent_state).__name__}"
            )
        if not isinstance(self.episode_state, EpisodeState):
            raise ValidationError(
                f"Episode state must be an EpisodeState instance, got {type(self.episode_state).__name__}"
            )
        if not isinstance(self.plume_model, PlumeModel):
            raise ValidationError(
                f"Plume model must be a PlumeModel instance, got {type(self.plume_model).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the state snapshot to a dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "agent_state": self.agent_state.to_dict(),
            "episode_state": self.episode_state.get_episode_summary(),
            "plume_model": self.plume_model.to_dict(),
            "metadata": self.metadata,
        }

    # Lightweight consistency checker used by StateManager when requested.
    def validate_consistency(self) -> Dict[str, Any]:
        errors = []
        warnings = []
        try:
            # Agent invariants
            if self.agent_state.step_count < 0:
                errors.append("negative step_count")
            # Episode invariants
            if self.episode_state.terminated and self.episode_state.truncated:
                errors.append("terminated and truncated both True")
        except Exception as exc:  # Defensive: never raise from validation
            warnings.append(f"validation_exception: {exc}")

        return {
            "is_consistent": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }
