"""Sensor protocol definitions."""

from __future__ import annotations

from typing import Protocol, Sequence, Any, runtime_checkable


@runtime_checkable
class SensorProtocol(Protocol):
    """Interface for plume sensing components.

    Sensors provide detection or measurement capabilities for agents based on
    the underlying plume model state.
    """

    def detect(self, plume_state: Any, positions: Sequence[Sequence[float]]) -> Sequence[bool]:
        """Return detection events for agents at ``positions``."""

    def measure(self, plume_state: Any, positions: Sequence[Sequence[float]]) -> Sequence[float]:
        """Return scalar measurements for agents at ``positions``."""

    def compute_gradient(
        self, plume_state: Any, positions: Sequence[Sequence[float]]
    ) -> Sequence[Sequence[float]]:
        """Estimate spatial concentration gradients at agent positions.

        Args:
            plume_state: Arbitrary plume model state used to compute gradients.
            positions: Collection of agent coordinates for sampling the field.

        Returns:
            A sequence of 2-D gradient vectors aligned with ``positions``. Each
            vector represents the concentration change along the x and y axes
            at the corresponding agent position.

        Notes:
            Implementations should raise a descriptive error if gradient
            computation is unsupported for the provided ``plume_state`` or if
            the ``positions`` input is malformed.
        """

    def configure(self, **kwargs: Any) -> None:
        """Update internal sensor parameters at runtime.

        Implementations may support a variety of configuration options such as
        detection thresholds, dynamic ranges, or spatial resolution. Unknown
        parameters should raise a ``TypeError`` to avoid silent failures.

        Args:
            **kwargs: Implementation-specific keyword arguments.
        """


__all__ = ["SensorProtocol"]
