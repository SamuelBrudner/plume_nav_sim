from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any, Optional

# Lightweight policy loader usable by UIs and CLIs.
# Contract supported by the runner:
# - Prefer objects with select_action(observation, explore=False)
# - Accept plain callables: policy(observation) -> action
# - Optionally call reset(seed=...) if present


@dataclass
class LoadedPolicy:
    obj: Any
    spec: str


def _resolve_attr(mod: ModuleType, attr_path: str) -> Any:
    cur: Any = mod
    for part in attr_path.split("."):
        if not hasattr(cur, part):
            raise AttributeError(
                f"Attribute '{part}' not found while resolving '{attr_path}'"
            )
        cur = getattr(cur, part)
    return cur


def _import_module(module_name: str) -> ModuleType:
    return import_module(module_name)


def _import_longest_prefix(dotted: str) -> tuple[ModuleType, Optional[str]]:
    parts = dotted.split(".")
    # Try to import the longest importable prefix
    for i in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:i])
        try:
            mod = _import_module(mod_name)
            remainder = ".".join(parts[i:]) if i < len(parts) else None
            return mod, (remainder if remainder else None)
        except ModuleNotFoundError:
            continue
    # Fall back to plain import error
    raise ModuleNotFoundError(f"No importable module prefix found in '{dotted}'")


def load_policy(spec: str, *, kwargs: Optional[dict] = None) -> LoadedPolicy:
    """Load a policy from a spec string.

    Accepted forms:
    - "package.module:ClassOrCallable"
    - "package.module.ClassOrCallable" (last dot splits module vs. attribute)
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("Policy spec must be a non-empty string")

    module_name: str
    attr_path: Optional[str]

    if ":" in spec:
        module_name, attr_path = spec.split(":", 1)
        if not attr_path:
            raise ValueError("Policy spec missing attribute after ':'")
        # Import the specified module directly for module:Attr form
        mod = _import_module(module_name)
    else:
        # For dotted form without ':', import the longest importable prefix as module
        # and treat the remainder as an attribute path.
        mod, attr_path = _import_longest_prefix(spec)

    target: Any
    if attr_path:
        target = _resolve_attr(mod, attr_path)
    else:
        target = mod

    # If target is a class, instantiate (passing kwargs if provided); otherwise return as-is
    try:
        if isinstance(target, type):
            if kwargs:
                if not isinstance(kwargs, dict):
                    raise TypeError(
                        "kwargs must be a dict when instantiating a class policy"
                    )
                obj = target(**kwargs)
            else:
                obj = target()
        else:
            obj = target
    except Exception as e:
        raise TypeError(f"Failed to instantiate policy '{spec}': {e}") from e

    return LoadedPolicy(obj=obj, spec=spec)


def reset_policy_if_possible(obj: Any, *, seed: Optional[int]) -> None:
    try:
        obj.reset(seed=seed)  # type: ignore[attr-defined]
    except Exception:
        pass
