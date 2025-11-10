from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ActionInfo:
    """Debugger-facing description of an action space.

    Minimal for now: a list of human-readable names matching action indices.
    """

    names: List[str]


@dataclass(frozen=True)
class ObservationInfo:
    """Metadata about the policy observation for UI hints.

    This is intentionally lightweight to avoid UI coupling.
    """

    kind: str  # e.g., "vector", "image", "scalar"
    label: Optional[str] = None


@dataclass(frozen=True)
class PipelineInfo:
    """Names of components/wrappers contributing to the observation pipeline."""

    names: List[str]
