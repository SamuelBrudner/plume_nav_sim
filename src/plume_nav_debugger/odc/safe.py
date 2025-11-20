from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Tuple

import numpy as np

_IMMUTABLE_TYPES: Tuple[type, ...] = (
    int,
    float,
    str,
    bytes,
    tuple,
    frozenset,
    type(None),
)


def _is_immutable(value: Any) -> bool:
    return isinstance(value, _IMMUTABLE_TYPES)


class _BlockedMethod:
    def __init__(self, owner_name: str, attr_name: str) -> None:
        self._owner_name = owner_name
        self._attr_name = attr_name

    def __call__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
        raise RuntimeError(
            f"ODC provider must be side-effect free: calling '{self._owner_name}.{self._attr_name}()' is not allowed"
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<_BlockedMethod {self._owner_name}.{self._attr_name}>"


class ReadOnlyProxy:
    """Read-only proxy that blocks obvious mutating methods and attribute writes.

    - Denies attribute set/delete on the proxied object
    - Blocks common mutators by exact name and by prefix
    - Recursively wraps nested objects so deep writes are also blocked
    - Makes any ndarray attributes read-only copies
    """

    __slots__ = ("_target", "_owner_name", "_blocked", "_blocked_prefixes")

    def __init__(
        self,
        target: Any,
        *,
        owner_name: Optional[str] = None,
        blocked: Optional[Iterable[str]] = None,
        blocked_prefixes: Optional[Iterable[str]] = None,
    ) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_owner_name", owner_name or type(target).__name__)
        object.__setattr__(self, "_blocked", set(blocked or ()))
        object.__setattr__(
            self,
            "_blocked_prefixes",
            tuple(
                blocked_prefixes
                or (
                    "set_",
                    "update_",
                    "append",
                    "extend",
                    "pop",
                    "clear",
                    "remove",
                    "add",
                    "fit",
                    "train",
                )
            ),
        )

    # Attribute access -----------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        if name in {"_target", "_owner_name", "_blocked", "_blocked_prefixes"}:
            return object.__getattribute__(self, name)

        # Deny blocked names immediately (method or attribute)
        if name in self._blocked or any(
            name.startswith(p) for p in self._blocked_prefixes
        ):
            return _BlockedMethod(self._owner_name, name)

        target = object.__getattribute__(self, "_target")
        value = getattr(target, name)
        return self._wrap_value(name, value)

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover - simple
        # Prevent writes to proxied object
        if name in {"_target", "_owner_name", "_blocked", "_blocked_prefixes"}:
            object.__setattr__(self, name, value)
        else:
            raise RuntimeError(
                f"ODC provider must be side-effect free: setting '{self._owner_name}.{name}' is not allowed"
            )

    def __delattr__(self, name: str) -> None:  # pragma: no cover - simple
        raise RuntimeError(
            f"ODC provider must be side-effect free: deleting '{self._owner_name}.{name}' is not allowed"
        )

    # Helpers --------------------------------------------------------------
    def _wrap_value(self, name: str, value: Any) -> Any:
        # Keep primitive immutables as-is
        if _is_immutable(value):
            return value
        # Make ndarray attributes read-only copies
        if isinstance(value, np.ndarray):
            arr = np.array(value, copy=True)
            try:
                arr.setflags(write=False)
            except Exception:
                pass
            return arr
        # Wrap callables that are blocked only by name
        if callable(value) and (
            name in self._blocked
            or any(name.startswith(p) for p in self._blocked_prefixes)
        ):
            return _BlockedMethod(self._owner_name, name)
        # Recursively wrap nested objects to keep deep writes blocked
        try:
            return ReadOnlyProxy(
                value,
                owner_name=f"{self._owner_name}.{name}",
                blocked=self._blocked,
                blocked_prefixes=self._blocked_prefixes,
            )
        except Exception:
            # Fallback: return raw value
            return value

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<ReadOnlyProxy of {self._owner_name}>"


def safe_policy_proxy(policy: Any) -> ReadOnlyProxy:
    """Return a read-only proxy for policies that blocks obvious mutators.

    Blocks: select_action, reset, seed, train/eval/fit, and common mutator prefixes.
    """
    blocked = {
        "select_action",
        "reset",
        "seed",
        "train",
        "eval",
        "fit",
        "step",
    }
    return ReadOnlyProxy(policy, owner_name="policy", blocked=blocked)


def safe_env_proxy(env: Any) -> ReadOnlyProxy:
    """Return a read-only proxy for environments that blocks obvious mutators.

    Blocks: step, reset, seed, close, and common mutator prefixes.
    """
    blocked = {"step", "reset", "seed", "close"}
    return ReadOnlyProxy(env, owner_name="env", blocked=blocked)


def readonly_observation(observation: Any) -> np.ndarray:
    """Make a defensive, read-only ndarray copy of an observation-like input."""
    arr = np.asarray(observation, dtype=float)
    arr = np.array(arr, copy=True)
    try:
        arr.setflags(write=False)
    except Exception:
        pass
    return arr


__all__ = [
    "ReadOnlyProxy",
    "safe_policy_proxy",
    "safe_env_proxy",
    "readonly_observation",
]
