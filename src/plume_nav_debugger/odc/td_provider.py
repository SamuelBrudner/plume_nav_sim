from __future__ import annotations

from typing import Any, Optional

from plume_nav_sim.policies._concentration_extractor import extract_concentration

from .models import ActionInfo, ObservationInfo
from .provider import DebuggerProvider

_ORIENTED_ACTION_NAMES = ["FORWARD", "TURN_LEFT", "TURN_RIGHT"]


def _action_count(env: Any) -> int:
    try:
        space = getattr(env, "action_space", None)
        return int(getattr(space, "n", 0) or 0)
    except Exception:
        return 0


def _extract_concentration(policy: Any, observation: Any) -> Optional[float]:
    try:
        concentration_key = getattr(policy, "concentration_key", None)
        modality_index = int(getattr(policy, "modality_index", 0) or 0)
        sensor_index = getattr(policy, "sensor_index", None)
        return float(
            extract_concentration(
                observation,
                policy_name="TemporalDerivativePolicy",
                concentration_key=concentration_key,
                modality_index=modality_index,
                sensor_index=sensor_index,
            )
        )
    except Exception:
        return None


def _one_hot(action: int, *, size: int = 3) -> list[float]:
    idx = int(action)
    if idx < 0 or idx >= size:
        return [0.0 for _ in range(size)]
    out = [0.0 for _ in range(size)]
    out[idx] = 1.0
    return out


class TemporalDerivativeProvider(DebuggerProvider):
    """ODC provider for built-in temporal-derivative policies."""

    def __init__(self, *, mode: str) -> None:
        self._mode = str(mode)

    def get_action_info(self, env: Any) -> Optional[ActionInfo]:
        n = _action_count(env)
        if n not in (0, len(_ORIENTED_ACTION_NAMES)):
            return None
        return ActionInfo(names=list(_ORIENTED_ACTION_NAMES))

    def describe_observation(
        self, observation: Any, *, context: Optional[dict] = None  # noqa: ARG002
    ) -> Optional[ObservationInfo]:
        try:
            if observation is None:
                return None
            return ObservationInfo(kind="scalar", label="concentration")
        except Exception:
            return None

    def policy_distribution(self, policy: Any, observation: Any) -> Optional[dict]:
        c = _extract_concentration(policy, observation)
        if c is None:
            return None

        prev_c = getattr(policy, "_prev_c", None)
        if prev_c is None:
            prev_c = getattr(policy, "_last_c", None)
        prev_action = getattr(policy, "_prev_action", None)
        if prev_action is None:
            prev_action = getattr(policy, "_last_action", None)
        if prev_c is None:
            return {"probs": _one_hot(0)}

        threshold = float(getattr(policy, "threshold", 1e-6))
        dc = c - float(prev_c)

        if self._mode == "deterministic":
            action = self._deterministic_action(policy, prev_action, dc, threshold)
            return {"probs": _one_hot(action)}

        if prev_action in (1, 2):
            return {"probs": _one_hot(0)}
        if dc >= threshold:
            return {"probs": _one_hot(0)}

        if bool(getattr(policy, "uniform_random_on_non_increase", False)):
            p = 1.0 / 3.0
            return {"probs": [p, p, p]}
        return {"probs": [0.0, 0.5, 0.5]}

    def _deterministic_action(
        self, policy: Any, prev_action: Any, dc: float, threshold: float
    ) -> int:
        if prev_action in (1, 2):
            return 0
        if dc >= threshold:
            return 0
        if not bool(getattr(policy, "alternate_cast", True)):
            return 2
        cast_right_next = getattr(
            policy, "_prev_cast_right_next", getattr(policy, "_cast_right_next", True)
        )
        return 2 if cast_right_next else 1


__all__ = ["TemporalDerivativeProvider"]
