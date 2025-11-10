from __future__ import annotations

from typing import Any, Optional

from .models import ActionInfo, ObservationInfo, PipelineInfo


class DebuggerProvider:
    """Opinionated Debugger Contract (ODC) provider interface.

    Implementers expose debugger-facing metadata without changing debugger code.
    All methods are optional and MUST be side-effect free and deterministic for
    a given input. When a method returns None, the debugger treats it as "no
    opinion" and may hide that detail (or use heuristics when strict mode is off).
    See SPEC.md for full semantics and invariants.
    """

    # Action space -----------------------------------------------------------
    def get_action_info(
        self, env: Any
    ) -> Optional[ActionInfo]:  # pragma: no cover - interface
        """Return action presentation info.

        - names: list[str], length must equal env.action_space.n (when known)
        - stable across a run; index-aligned 0..n-1
        """
        return None

    # Observation hints ------------------------------------------------------
    def describe_observation(
        self, observation: Any, *, context: Optional[dict] = None
    ) -> Optional[ObservationInfo]:  # pragma: no cover - interface
        """Describe the policy observation for presentation.

        - kind: one of {"vector","image","scalar","unknown"}
        - label: optional concise label for the UI
        - context: reserved for future (e.g., pipeline stage info)
        """
        return None

    # Policy distribution ----------------------------------------------------
    def policy_distribution(
        self, policy: Any, observation: Any
    ) -> Optional[dict]:  # pragma: no cover - interface
        """Return action distribution preview without side effects.

        Exactly one of the following keys must be returned:
        - "probs": list[float] (1D, length == action count)
        - "q_values": list[float] (1D, length == action count)
        - "logits": list[float] (1D, length == action count)

        The debugger normalizes "probs" defensively and applies softmax to
        "q_values" and "logits" to produce a probability preview.
        """
        return None

    # Pipeline ---------------------------------------------------------------
    def get_pipeline(
        self, env: Any
    ) -> Optional[PipelineInfo]:  # pragma: no cover - interface
        """Return ordered names of wrappers/components from outermost to core.

        Example: ["PlumeSearchEnv", "ConcentrationNBackWrapper(n=3)", "CoreEnv"]
        """
        return None


__all__ = [
    "DebuggerProvider",
    "ActionInfo",
    "ObservationInfo",
    "PipelineInfo",
]
