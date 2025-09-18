"""Fallback stubs for the plume search environment in minimal builds.

The original project exposes a richly featured :mod:`plume_nav_sim.envs.plume_search_env`
module that provides the :class:`PlumeSearchEnv` Gymnasium environment alongside
factory and validation helpers.  The educational copy of the repository that backs
this kata does not bundle the full environment implementation, yet a large portion
of the codebase (including pytest fixtures) still imports the public symbols.  The
absence of the module causes import-time failures that mask the behaviour we want to
exercise in the refactor under test.

To keep the public API stable while avoiding the heavy runtime dependency, this file
provides lightweight stand-ins for the canonical objects.  They intentionally offer
only the minimal surface required for type annotations and attribute access.  Any
attempt to *use* the environment—resetting, stepping, or rendering—raises a clear
:class:`ComponentError` so callers understand that the concrete implementation is not
available in this trimmed environment.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ..utils.exceptions import ComponentError

__all__ = [
    "PlumeSearchEnv",
    "create_plume_search_env",
    "validate_plume_search_config",
]


class PlumeSearchEnv:
    """Minimal stub of the real PlumeSearchEnv implementation.

    The class intentionally accepts any initialisation arguments so existing import
    sites continue to work.  Operational methods raise :class:`ComponentError` to
    communicate that the full simulator is not part of this pared-down build.
    """

    #: Metadata attribute expected by Gymnasium environments.
    metadata: Dict[str, Any] = {"render_modes": ["human", "rgb_array"]}
    #: Action space placeholder exposed for type compatibility.
    action_space: Any = None
    #: Observation space placeholder exposed for type compatibility.
    observation_space: Any = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._init_args = args
        self._init_kwargs = kwargs

    def reset(self, *args: Any, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        """Resetting the stub environment is unsupported."""

        raise ComponentError(
            "The PlumeSearchEnv simulation is unavailable in this test fixture.",
            component_name="PlumeSearchEnv",
            operation_name="reset",
        )

    def step(self, *args: Any, **kwargs: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Advancing the stub environment is unsupported."""

        raise ComponentError(
            "The PlumeSearchEnv simulation is unavailable in this test fixture.",
            component_name="PlumeSearchEnv",
            operation_name="step",
        )

    def render(self, *args: Any, **kwargs: Any) -> Any:
        """Rendering is not implemented in the stub environment."""

        raise ComponentError(
            "The PlumeSearchEnv renderer is not bundled in this pared-down build.",
            component_name="PlumeSearchEnv",
            operation_name="render",
        )

    def close(self) -> None:
        """Closing the stub is a no-op."""

        return None


def create_plume_search_env(*args: Any, **kwargs: Any) -> PlumeSearchEnv:
    """Factory function mirroring the public API of the full package."""

    return PlumeSearchEnv(*args, **kwargs)


def validate_plume_search_config(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Placeholder validation helper.

    The real implementation performs extensive configuration checks.  The stub keeps
    the signature compatible and simply returns the collected parameters so callers
    can inspect what would have been validated.
    """

    return {"args": args, "kwargs": kwargs}
