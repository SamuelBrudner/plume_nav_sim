"""Shared environment state machine enum."""

from __future__ import annotations

from enum import Enum

__all__ = ["EnvironmentState"]


class EnvironmentState(Enum):
    """Formal states for environment lifecycle."""

    CREATED = "created"  # After __init__(), must reset() before step()
    READY = "ready"  # After reset(), can step()
    TERMINATED = "terminated"  # Episode ended (goal reached)
    TRUNCATED = "truncated"  # Episode timeout
    CLOSED = "closed"  # Resources released, terminal state
