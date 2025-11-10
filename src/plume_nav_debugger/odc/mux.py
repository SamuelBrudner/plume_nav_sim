from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from ..inspector.introspection import get_env_chain_names
from ..inspector.models import _softmax  # reuse stable softmax impl
from .discovery import find_provider
from .models import ActionInfo, PipelineInfo
from .provider import DebuggerProvider


class ProviderMux:
    """Merge provider capabilities with debugger heuristics.

    Provider takes precedence; debugger falls back to heuristics otherwise.
    Results are cached per-instance where prudent.
    """

    def __init__(
        self, env: Any, policy: Any, provider: Optional[DebuggerProvider] = None
    ) -> None:
        self._env = env
        self._policy = policy
        self._provider = (
            provider if provider is not None else find_provider(env, policy)
        )
        self._action_names: Optional[List[str]] = None
        self._pipeline: Optional[List[str]] = None

    # Actions ----------------------------------------------------------------
    def get_action_names(self) -> List[str]:
        if self._action_names is not None:
            return list(self._action_names)
        # Provider
        if self._provider is not None:
            try:
                info = self._provider.get_action_info(self._env)
                if isinstance(info, ActionInfo) and info.names:
                    names = list(info.names)
                    # Validate length when action count is known
                    n = _action_count(self._env)
                    if n == 0 or n == len(names):
                        self._action_names = names
                        return list(self._action_names)
            except Exception:
                pass
        # Heuristic from env.action_space metadata
        names: Optional[List[str]] = None
        try:
            space = getattr(self._env, "action_space", None)
            if space is not None:
                meta = getattr(space, "_metadata", {}) or {}
                cand = meta.get("action_names")
                if isinstance(cand, (list, tuple)) and all(
                    isinstance(x, str) for x in cand
                ):
                    names = list(cand)
        except Exception:
            names = None
        if names is None:
            try:
                n = getattr(getattr(self._env, "action_space", None), "n", None)
                if isinstance(n, (int, np.integer)):
                    n = int(n)
                    if n == 3:
                        names = ["FORWARD", "TURN_LEFT", "TURN_RIGHT"]
                    elif n == 4:
                        names = ["UP", "RIGHT", "DOWN", "LEFT"]
            except Exception:
                names = None
        if names is None:
            try:
                n = getattr(getattr(self._env, "action_space", None), "n", 0) or 0
                n = int(n) if isinstance(n, (int, np.integer)) else 0
            except Exception:
                n = 0
            names = [str(i) for i in range(max(0, n))]
        self._action_names = names
        return list(self._action_names)

    # Distribution ------------------------------------------------------------
    def get_policy_distribution(self, observation: np.ndarray) -> Optional[List[float]]:
        # Provider
        if self._provider is not None:
            tried_provider = False
            try:
                res = self._provider.policy_distribution(self._policy, observation)
                if isinstance(res, dict):
                    tried_provider = True
                    n = _action_count(self._env)
                    if "probs" in res and _is_1d(res["probs"]):
                        p = _normalize_1d(res["probs"]).ravel()
                        if n == 0 or len(p) == n:
                            return p.tolist()
                    if "q_values" in res and _is_1d(res["q_values"]):
                        q = np.asarray(res["q_values"], dtype=float).ravel()
                        if n == 0 or len(q) == n:
                            return _softmax(q).tolist()
                    if "logits" in res and _is_1d(res["logits"]):
                        l = np.asarray(res["logits"], dtype=float).ravel()
                        if n == 0 or len(l) == n:
                            return _softmax(l).tolist()
            except Exception:
                tried_provider = True
            # If provider responded but was invalid, do not fallback
            if tried_provider:
                return None
        # Heuristic probing (side-effect free)
        n = _action_count(self._env)
        try:
            if hasattr(self._policy, "action_probabilities"):
                probs = np.asarray(self._policy.action_probabilities(observation))  # type: ignore[attr-defined]
                if probs.ndim == 1 and probs.size:
                    if n == 0 or probs.size == n:
                        return _normalize_1d(probs).tolist()
        except Exception:
            pass
        try:
            if hasattr(self._policy, "q_values"):
                q = np.asarray(self._policy.q_values(observation))  # type: ignore[attr-defined]
                if q.ndim == 1 and q.size:
                    if n == 0 or q.size == n:
                        return _softmax(q).tolist()
        except Exception:
            pass
        try:
            if hasattr(self._policy, "logits"):
                l = np.asarray(self._policy.logits(observation))  # type: ignore[attr-defined]
                if l.ndim == 1 and l.size:
                    if n == 0 or l.size == n:
                        return _softmax(l).tolist()
        except Exception:
            pass
        try:
            if hasattr(self._policy, "action_distribution"):
                d = np.asarray(self._policy.action_distribution(observation))  # type: ignore[attr-defined]
                if d.ndim == 1 and d.size:
                    if n == 0 or d.size == n:
                        return _normalize_1d(d).tolist()
        except Exception:
            pass
        return None

    # Pipeline ----------------------------------------------------------------
    def get_pipeline(self) -> List[str]:
        if self._pipeline is not None:
            return list(self._pipeline)
        if self._provider is not None:
            try:
                p = self._provider.get_pipeline(self._env)
                if isinstance(p, PipelineInfo) and p.names:
                    self._pipeline = list(p.names)
                    return list(self._pipeline)
            except Exception:
                pass
        try:
            self._pipeline = get_env_chain_names(self._env)
        except Exception:
            self._pipeline = []
        return list(self._pipeline)


def _is_1d(x: Any) -> bool:
    try:
        a = np.asarray(x)
        return a.ndim == 1 and a.size > 0
    except Exception:
        return False


def _normalize_1d(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    s = float(a.sum())
    if s <= 0 or not np.isfinite(s):
        if a.size == 0:
            return a
        return np.full(a.shape, 1.0 / a.size)
    return a / s


def _action_count(env: Any) -> int:
    try:
        n = getattr(getattr(env, "action_space", None), "n", None)
        if isinstance(n, (int, np.integer)):
            return int(n)
    except Exception:
        return 0
    return 0
