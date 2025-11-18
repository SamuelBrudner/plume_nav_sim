from __future__ import annotations

"""ODC provider for the plug-and-play run–tumble demo policy.

This provider enables the debugger to display action labels (RUN/TUMBLE),
an action distribution preview derived from the observation (side-effect
free), and a lightweight pipeline description by introspecting the wrapper
chain when possible.
"""

from typing import Any, List, Optional

import numpy as np

from plume_nav_debugger.odc.models import ActionInfo, ObservationInfo, PipelineInfo
from plume_nav_debugger.odc.provider import DebuggerProvider


class RunTumbleDemoProvider(DebuggerProvider):
    """Provider for a 2-action run–tumble policy.

    - Actions: index 0 → RUN, index 1 → TUMBLE
    - Distribution: peaked on the greedy choice derived from observation
      without calling `select_action` or introducing randomness.
    - Observation: treated as a vector with a concise label when recognizable.
    - Pipeline: best-effort wrapper chain discovery (outermost → core).
    """

    def __init__(self, *, threshold: float | None = None):
        # Optional threshold hint to mirror the policy decision boundary.
        self._threshold = float(threshold) if threshold is not None else None

    # Actions --------------------------------------------------------------
    def get_action_info(self, env: Any) -> Optional[ActionInfo]:  # noqa: ARG002
        return ActionInfo(names=["RUN", "TUMBLE"])

    # Observation ----------------------------------------------------------
    def describe_observation(
        self, observation: Any, *, context: Optional[dict] = None  # noqa: ARG002
    ) -> Optional[ObservationInfo]:
        # Heuristic: treat numeric array-likes as vectors; label common form.
        try:
            arr = np.asarray(observation)
            if arr.ndim == 0 or arr.size == 1:
                return ObservationInfo(kind="scalar", label="concentration")
            if arr.ndim == 1:
                label = "[c_prev, c_now] or [c_now, c_prev, dc]"
                return ObservationInfo(kind="vector", label=label)
            return ObservationInfo(kind="image", label="observation")
        except Exception:
            return ObservationInfo(kind="unknown", label="observation")

    # Policy distribution --------------------------------------------------
    def policy_distribution(self, policy: Any, observation: Any) -> Optional[dict]:
        # Side-effect free approximation of the demo policy's greedy action.
        # Compute dc either from explicit third component or as last - prev.
        try:
            arr = np.asarray(observation, dtype=float).reshape(-1)
        except Exception:
            return None

        dc: float
        if arr.size >= 3:
            dc = float(arr[2])
        elif arr.size >= 2:
            dc = float(arr[-1]) - float(arr[-2])
        else:
            # Not enough information to infer a direction; decline.
            return None

        # Mirror the policy threshold when available; otherwise use a sane default.
        thr = (
            float(getattr(policy, "threshold", 1e-6))
            if self._threshold is None
            else self._threshold
        )
        a = 0 if dc >= thr else 1  # 0=RUN, 1=TUMBLE
        probs = [0.0, 0.0]
        probs[int(a)] = 1.0
        return {"probs": probs}

    # Pipeline -------------------------------------------------------------
    def get_pipeline(self, env: Any) -> Optional[PipelineInfo]:
        # Best-effort: walk common Gym wrapper chains via `.env`.
        names: List[str] = []
        seen = set()
        cur = env
        max_hops = 16  # safety
        try:
            for _ in range(max_hops):
                if cur is None or id(cur) in seen:
                    break
                seen.add(id(cur))
                cls = type(cur).__name__
                # Include wrapper params when helpful (e.g., n-back history)
                if hasattr(cur, "n") and cls == "ConcentrationNBackWrapper":
                    cls = f"{cls}(n={getattr(cur, 'n', '?')})"
                names.append(cls)
                next_env = getattr(cur, "env", None)
                if next_env is None or next_env is cur:
                    break
                cur = next_env
        except Exception:
            # On failure, omit pipeline detail (optional capability)
            return None

        if not names:
            return None
        return PipelineInfo(names=names)


__all__ = ["RunTumbleDemoProvider"]
