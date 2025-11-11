from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from ..inspector.introspection import get_env_chain_names
from ..inspector.models import _softmax  # reuse stable softmax impl
from .discovery import find_provider
from .models import ActionInfo, PipelineInfo
from .provider import DebuggerProvider


class ProviderMux:
    """Provider-only multiplexer for debugger capabilities.

    - No heuristic fallbacks. If no provider is available, methods return
      empty/None values suitable for a provider-required UI.
    - Results are cached per-instance where prudent.
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
        # Provider-only
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
        # No provider → no labels
        self._action_names = []
        return []

    # Distribution ------------------------------------------------------------
    def get_policy_distribution(self, observation: np.ndarray) -> Optional[List[float]]:
        # Provider-only
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
        # No provider → no pipeline
        self._pipeline = []
        return []

    def has_provider(self) -> bool:
        return self._provider is not None


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
