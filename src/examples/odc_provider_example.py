from __future__ import annotations

from typing import Any

import numpy as np

from plume_nav_debugger.odc.models import ActionInfo, ObservationInfo, PipelineInfo
from plume_nav_debugger.odc.provider import DebuggerProvider


class ExampleDebuggerProvider(DebuggerProvider):
    """Minimal example provider for oriented 3-action control.

    - Names: FORWARD, TURN_LEFT, TURN_RIGHT
    - Distribution: simple centered peak at action 0 (for demo only)
    - Pipeline: synthetic illustrative names
    """

    def get_action_info(self, env: Any):
        return ActionInfo(names=["FORWARD", "TURN_LEFT", "TURN_RIGHT"])

    def policy_distribution(self, policy: Any, observation: Any):
        # Demo-only: peaked distribution on action 0
        n = getattr(getattr(env := policy, "action_space", None), "n", 3)  # type: ignore
        if not isinstance(n, (int, np.integer)):
            n = 3
        probs = np.zeros(int(n), dtype=float)
        probs[0] = 1.0
        return {"probs": probs.tolist()}

    def describe_observation(self, observation: Any):
        try:
            arr = np.asarray(observation)
            if arr.ndim == 0 or arr.size == 1:
                kind = "scalar"
            elif arr.ndim == 1:
                kind = "vector"
            elif arr.ndim >= 2:
                kind = "image"
            else:
                kind = "unknown"
        except Exception:
            kind = "unknown"
        return ObservationInfo(kind=kind, label="example")

    def get_pipeline(self, env: Any):
        return PipelineInfo(names=[type(env).__name__, "ExampleWrapper", "CoreEnv"])


def provider_factory(env: Any, policy: Any) -> DebuggerProvider:
    return ExampleDebuggerProvider()
