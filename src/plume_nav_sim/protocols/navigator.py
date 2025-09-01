"""Navigation protocol definitions."""

from __future__ import annotations

from typing import Protocol, Dict, Any, Optional, runtime_checkable
import numpy as np


@runtime_checkable
class NavigatorProtocol(Protocol):
    """Structural interface for navigation controllers.

    Implementations manage agent motion and internal state updates during a
    simulation. Concrete navigators should provide deterministic behavior
    and expose their state for inspection.
    """

    # --- Required state properties ---
    positions: np.ndarray
    orientations: np.ndarray
    speeds: np.ndarray
    max_speeds: np.ndarray
    angular_velocities: np.ndarray
    num_agents: int

    # --- Core control API ---
    def reset(self, *args, **kwargs) -> None:
        """Reset the navigator to its initial state with optional parameters."""

    def step(self, env_array: np.ndarray, dt: float = 1.0) -> None:
        """Advance the navigator state using the provided environment array."""

    def sample_odor(self, env_array: np.ndarray) -> np.ndarray | float:
        """Sample odor values at the agent positions."""

    def sample_multiple_sensors(
        self,
        env_array: np.ndarray,
        sensor_distance: float = ...,
        sensor_angle: float = ...,
        num_sensors: int = ...,
        layout_name: Optional[str] = ...,
    ) -> np.ndarray:
        """Sample odor using a multi-sensor configuration."""

    # --- Optional observation and memory hooks ---
    def observe(self, sensor_output: Any) -> Dict[str, Any]:
        """Process raw sensor output into a normalized observation.

        Implementations may simply return an empty dictionary if no
        observation processing is required. Invalid input types should
        raise ``TypeError`` to prevent silent failures.
        """

    def load_memory(self, memory_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Load internal memory state for cognitive navigation strategies.

        The default expectation is a no-op returning ``None`` when memory
        is disabled. Implementations should raise ``TypeError`` when
        ``memory_data`` is provided with an incompatible type.
        """

    def save_memory(self) -> Optional[Dict[str, Any]]:
        """Return a serialisable snapshot of the current memory state.

        When memory is disabled, implementations should return ``None``.
        Invalid internal memory representations should trigger a
        ``TypeError`` to surface misconfigurations loudly.
        """

    # --- Extensibility hooks ---
    def compute_additional_obs(self, base_obs: dict) -> dict:
        """Compute additional observation components."""

    def compute_extra_reward(self, base_reward: float, info: dict) -> float:
        """Compute additional reward components for shaping."""

    def on_episode_end(self, final_info: dict) -> None:
        """Handle episode completion events."""

    def get_observation_space_info(self) -> Dict[str, Any]:
        """Return metadata describing the observation space."""


__all__ = ["NavigatorProtocol"]
