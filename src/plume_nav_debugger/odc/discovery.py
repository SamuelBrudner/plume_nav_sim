from __future__ import annotations

from typing import Any, Optional

from .provider import DebuggerProvider


def _load_entry_point_provider() -> Optional[DebuggerProvider]:
    """Search for entry-point based providers and return the first created provider.

    Entry point group: 'plume_nav_sim.debugger_plugins'
    Expected entry point object: callable factory (env, policy) -> DebuggerProvider
    Fallback: class with no-arg constructor returning DebuggerProvider
    """
    try:
        import importlib.metadata as _im

        eps = _im.entry_points()
        group = None
        # Python 3.10+: entry_points().select
        if hasattr(eps, "select"):
            group = list(eps.select(group="plume_nav_sim.debugger_plugins"))
        else:  # pragma: no cover - legacy API
            group = list(  # type: ignore[attr-defined]
                eps.get("plume_nav_sim.debugger_plugins", [])
            )
        for ep in group or []:
            try:
                obj = ep.load()
                # If object is a class, instantiate with no args
                if isinstance(obj, type):
                    inst = obj()  # type: ignore[call-arg]
                else:
                    inst = obj  # likely a factory, leave callable to the dispatcher
                # Return a wrapper that later create with env/policy
                return inst  # type: ignore[return-value]
            except Exception:
                continue
    except Exception:
        return None
    return None


def find_provider(env: Any, policy: Any) -> Optional[DebuggerProvider]:
    """Find a provider via entry points (priority) then reflection on env/policy.

    - Entry point factory preferred: factory(env, policy) -> provider
    - Reflection fallbacks checked after entry points
    """
    # Entry points
    ep_obj = _load_entry_point_provider()
    if ep_obj is not None:
        try:
            # Factory pattern
            if callable(ep_obj):
                prov = ep_obj(env, policy)  # type: ignore[misc]
                if prov is not None:
                    return prov  # duck-typed provider
            # Direct provider instance/object
            return ep_obj  # duck-typed provider
        except Exception:
            pass

    # Reflection on env/policy
    try:
        if hasattr(env, "get_debugger_provider"):
            prov = env.get_debugger_provider()
            if prov is not None:
                return prov
    except Exception:
        pass
    for attr in ("__debugger_provider__", "debugger_provider"):
        try:
            prov = getattr(env, attr, None)
            if prov is not None:
                return prov
        except Exception:
            pass
    try:
        if hasattr(policy, "get_debugger_provider"):
            prov = policy.get_debugger_provider()
            if prov is not None:
                return prov
    except Exception:
        pass
    for attr in ("__debugger_provider__", "debugger_provider"):
        try:
            prov = getattr(policy, attr, None)
            if prov is not None:
                return prov
        except Exception:
            pass
    # Built-in policy support (fallback)
    try:
        from plume_nav_sim.policies.temporal_derivative import TemporalDerivativePolicy
        from plume_nav_sim.policies.temporal_derivative_deterministic import (
            TemporalDerivativeDeterministicPolicy,
        )

        from .td_provider import TemporalDerivativeProvider

        if isinstance(policy, TemporalDerivativePolicy):
            return TemporalDerivativeProvider(mode="stochastic")
        if isinstance(policy, TemporalDerivativeDeterministicPolicy):
            return TemporalDerivativeProvider(mode="deterministic")
    except Exception:
        pass
    return None
