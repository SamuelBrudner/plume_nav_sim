"""
Deprecated compatibility shim. Import from `plume_nav_sim.core.types` instead.

This module re-exports selected symbols from `plume_nav_sim.core.types` to
provide a deprecation window while callers migrate imports. It emits a
DeprecationWarning on import and avoids wildcard re-exports to satisfy linting.
"""

from __future__ import annotations

import warnings as _warnings

from . import types as _types

_warnings.warn(
    "plume_nav_sim.core.typing is deprecated; use plume_nav_sim.core.types instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export names explicitly via assignments to avoid F401/F403 lint issues
RGBArray = _types.RGBArray
ActionType = _types.ActionType
MovementVector = _types.MovementVector
CoordinateType = _types.CoordinateType
ObservationType = _types.ObservationType
RewardType = _types.RewardType
InfoType = _types.InfoType
PerformanceMetrics = _types.PerformanceMetrics

__all__ = [
    "RGBArray",
    "ActionType",
    "MovementVector",
    "CoordinateType",
    "ObservationType",
    "RewardType",
    "InfoType",
    "PerformanceMetrics",
]
