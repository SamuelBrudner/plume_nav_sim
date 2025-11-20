from __future__ import annotations

from typing import Any, List, Set

_SKIP_NAMES: Set[str] = {
    "OrderEnforcing",
    "TimeLimit",
}


def get_env_chain_names(root: Any, *, include_wrapped: bool = True) -> List[str]:
    """Return class-name chain following common `env`/`unwrapped` attributes.

    Stops on cycles. Skips some generic wrappers by default.
    """
    names: List[str] = []
    seen: Set[int] = set()
    cur = root
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        name = type(cur).__name__
        if name not in _SKIP_NAMES:
            names.append(name)
        nxt = None
        if include_wrapped and hasattr(cur, "env"):
            try:
                nxt = getattr(cur, "env")
            except Exception:
                nxt = None
        if nxt is None and hasattr(cur, "unwrapped"):
            try:
                nxt = getattr(cur, "unwrapped")
            except Exception:
                nxt = None
        cur = nxt
    return names


def format_pipeline(names: List[str]) -> str:
    if not names:
        return "(unknown)"
    return " -> ".join(names)  # placeholder; UI may format differently
