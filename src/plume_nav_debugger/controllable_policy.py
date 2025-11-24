from __future__ import annotations

from typing import Any, Optional

from plume_nav_sim.core.types import ActionType, ObservationType


class ControllablePolicy:
    """Adapter that allows manual action override while delegating to a base policy.

    Contract expectations for base policy:
    - Prefer select_action(observation, explore=False)
    - Or callable: policy(observation) -> action
    - Optional reset(seed=...)
    - Optional action_space attribute (forwarded when present)
    """

    def __init__(self, base_policy: Any) -> None:
        self._base = base_policy
        self._next_action: Optional[ActionType] = None
        self._sticky_action: Optional[ActionType] = None

    # Policy protocol ---------------------------------------------------------
    @property
    def action_space(self):  # type: ignore[override]
        try:
            return getattr(self._base, "action_space")
        except Exception:
            return None

    def reset(self, *, seed: int | None = None) -> None:  # type: ignore[override]
        try:
            self._base.reset(seed=seed)  # type: ignore[attr-defined]
        except Exception:
            # Best-effort reset; ignore if base doesn't support it
            pass

    def select_action(
        self, observation: ObservationType, *, explore: bool = False
    ) -> ActionType:  # type: ignore[override]
        # Highest priority: sticky override
        if self._sticky_action is not None:
            return self._sticky_action

        # Next priority: one-shot override
        if self._next_action is not None:
            a = self._next_action
            self._next_action = None
            return a

        # Delegate to base policy
        if hasattr(self._base, "select_action"):
            try:
                return self._base.select_action(observation, explore=explore)  # type: ignore[attr-defined]
            except TypeError:
                return self._base.select_action(observation)  # type: ignore[misc]

        if callable(self._base):
            return self._base(observation)

        raise TypeError("Base policy must implement select_action() or be callable")

    # Controls ----------------------------------------------------------------
    def set_next_action(self, action: ActionType, *, sticky: bool = False) -> None:
        if sticky:
            self._sticky_action = action
            self._next_action = None
        else:
            self._next_action = action

    def clear_sticky(self) -> None:
        self._sticky_action = None
