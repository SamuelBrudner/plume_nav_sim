from __future__ import annotations

import warnings as _warnings
from typing import Any, List, Optional

import numpy as np

from ..inspector.introspection import get_env_chain_names
from ..inspector.models import _softmax  # reuse stable softmax impl
from .discovery import find_provider
from .models import ActionInfo, ObservationInfo, PipelineInfo
from .provider import DebuggerProvider
from .safe import readonly_observation, safe_env_proxy, safe_policy_proxy


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
        # Read-only proxies to enforce side-effect freedom in provider calls
        try:
            self._env_ro = safe_env_proxy(env)
        except Exception:  # pragma: no cover - defensive
            self._env_ro = env
        try:
            self._policy_ro = safe_policy_proxy(policy)
        except Exception:  # pragma: no cover - defensive
            self._policy_ro = policy
        self._action_names: Optional[List[str]] = None
        self._pipeline: Optional[List[str]] = None
        self._obs_kind_cache: Optional[str] = None
        self._obs_label_cache: Optional[str] = None

    # Actions ----------------------------------------------------------------
    def get_action_names(self) -> List[str]:
        if self._action_names is not None:
            return list(self._action_names)
        # Provider-only
        if self._provider is not None:
            try:
                # Determinism: call twice with read-only env and compare
                info1 = self._provider.get_action_info(self._env_ro)
                info2 = self._provider.get_action_info(self._env_ro)
                if isinstance(info1, ActionInfo) and info1.names:
                    names1 = list(info1.names)
                    names2 = (
                        list(info2.names) if isinstance(info2, ActionInfo) else names1
                    )
                    if names1 != names2:
                        _warnings.warn(
                            "plume_nav_debugger:provider_nondeterministic get_action_info returned different results",
                            RuntimeWarning,
                        )
                    else:
                        # Validate length when action count is known
                        n = _action_count(self._env)
                        if n == 0 or n == len(names1):
                            self._action_names = names1
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
                # Prevent in-place mutation and enforce determinism
                obs_ro = readonly_observation(observation)
                res1 = self._provider.policy_distribution(self._policy_ro, obs_ro)
                res2 = self._provider.policy_distribution(self._policy_ro, obs_ro)
                if isinstance(res1, dict):
                    tried_provider = True
                    keys1 = {k for k in ("probs", "q_values", "logits") if k in res1}
                    keys2 = (
                        {k for k in ("probs", "q_values", "logits") if k in res2}
                        if isinstance(res2, dict)
                        else keys1
                    )
                    if len(keys1) != 1 or keys1 != keys2:
                        _warnings.warn(
                            "plume_nav_debugger:provider_invalid_distribution_keys provider must return exactly one of probs|q_values|logits consistently",
                            RuntimeWarning,
                        )
                        return None
                    key = next(iter(keys1))
                    n = _action_count(self._env)

                    def _to_probs(d: dict[str, Any], k: str) -> Optional[np.ndarray]:
                        if k == "probs" and _is_1d(d.get("probs")):
                            p = _normalize_1d(d["probs"]).ravel()
                            return p if (n == 0 or len(p) == n) else None
                        if k == "q_values" and _is_1d(d.get("q_values")):
                            q = np.asarray(d["q_values"], dtype=float).ravel()
                            return _softmax(q) if (n == 0 or len(q) == n) else None
                        if k == "logits" and _is_1d(d.get("logits")):
                            l = np.asarray(d["logits"], dtype=float).ravel()
                            return _softmax(l) if (n == 0 or len(l) == n) else None
                        return None

                    p1 = _to_probs(res1, key)
                    p2 = _to_probs(res2 if isinstance(res2, dict) else res1, key)
                    if p1 is None or p2 is None:
                        return None
                    if not np.allclose(p1, p2, rtol=1e-8, atol=1e-12):
                        _warnings.warn(
                            "plume_nav_debugger:provider_nondeterministic policy_distribution returned different results for same input",
                            RuntimeWarning,
                        )
                        return None
                    return p1.tolist()
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
                # Determinism: double-call and compare
                p1 = self._provider.get_pipeline(self._env_ro)
                p2 = self._provider.get_pipeline(self._env_ro)
                if isinstance(p1, PipelineInfo) and p1.names:
                    names1 = list(p1.names)
                    names2 = list(p2.names) if isinstance(p2, PipelineInfo) else names1
                    if names1 != names2:
                        _warnings.warn(
                            "plume_nav_debugger:provider_nondeterministic get_pipeline returned different results",
                            RuntimeWarning,
                        )
                    else:
                        self._pipeline = names1
                        return list(self._pipeline)
            except Exception:
                pass
        # No provider → no pipeline
        self._pipeline = []
        return []

    def has_provider(self) -> bool:
        return self._provider is not None

    # Observation metadata -----------------------------------------------------
    def describe_observation(self, observation: Any) -> Optional[ObservationInfo]:
        """Return provider-supplied observation metadata for inspector.

        Provider-only; no heuristics. Caches last valid response shape-agnostically
        to avoid excessive provider calls on unchanged kinds/labels.
        """
        if self._provider is None:
            return None
        try:
            # Enforce no mutation of provided observation; use read-only copy
            obs_ro = readonly_observation(observation)
            res = self._provider.describe_observation(obs_ro)
            # Determinism: double-check
            res2 = self._provider.describe_observation(obs_ro)
        except Exception:
            return None
        # Accept either ObservationInfo or duck-typed dict with keys
        kind: Optional[str] = None
        label: Optional[str] = None
        if isinstance(res, ObservationInfo):
            kind = res.kind
            label = res.label
        elif isinstance(res, dict):  # defensive duck-typing
            k = res.get("kind")  # type: ignore[assignment]
            if isinstance(k, str):
                kind = k
            v = res.get("label")  # type: ignore[assignment]
            if isinstance(v, str) or v is None:
                label = v
        else:
            return None

        # If second response is structured, compare for determinism
        if isinstance(res2, ObservationInfo):
            if (res2.kind != kind) or (res2.label != label):
                _warnings.warn(
                    "plume_nav_debugger:provider_nondeterministic describe_observation returned different results",
                    RuntimeWarning,
                )
        elif isinstance(res2, dict):
            k2 = res2.get("kind")
            l2 = res2.get("label")
            if (isinstance(k2, str) and k2 != kind) or (
                (isinstance(l2, str) or l2 is None) and l2 != label
            ):
                _warnings.warn(
                    "plume_nav_debugger:provider_nondeterministic describe_observation returned different results",
                    RuntimeWarning,
                )

        # Normalize kind
        allowed = {"vector", "image", "scalar", "unknown"}
        if not isinstance(kind, str) or kind.lower() not in allowed:
            kind = "unknown"

        # Cache and return
        self._obs_kind_cache = kind
        self._obs_label_cache = label if isinstance(label, str) else None
        return ObservationInfo(kind=kind, label=self._obs_label_cache)


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
