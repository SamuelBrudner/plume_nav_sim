"""
Minimal local shim of the Gymnasium API surface used by this repository's tests.

Provides:
- register() and make() for environment registration and instantiation
- envs.registry with env_specs mapping
- spaces module with Box, Discrete, Dict classes
- wrappers.common.OrderEnforcing wrapper

This is intentionally lightweight and only implements what the test suite touches.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

# Re-export spaces submodule
from . import spaces  # noqa: F401
from .spaces import Space as Space  # top-level alias for typing/annotations


# Error module for compatibility
class _ErrorModule:
    class Error(Exception):
        """Base exception for Gymnasium errors."""
        pass


error = _ErrorModule()


class _Registry:
    def __init__(self) -> None:
        # Mapping[str, EnvSpec]
        self.env_specs: Dict[str, "EnvSpec"] = {}


class _EnvsModule:
    def __init__(self) -> None:
        self.registry = _Registry()


envs = _EnvsModule()
__version__ = "0.0.local"


@dataclass
class EnvSpec:
    id: str
    entry_point: str
    kwargs: Dict[str, Any]
    max_episode_steps: Optional[int] = None
    reward_threshold: Optional[float] = None


def register(
    *,
    id: str,
    entry_point: str,
    kwargs: Optional[Dict[str, Any]] = None,
    max_episode_steps: Optional[int] = None,
    disable_env_checker: bool = True,
    additional_wrappers: tuple = (),
) -> None:
    spec = EnvSpec(
        id=id,
        entry_point=entry_point,
        kwargs=dict(kwargs or {}),
        max_episode_steps=max_episode_steps,
    )
    envs.registry.env_specs[id] = spec


def make(id: str, **override_kwargs: Any):  # noqa: A002 - match gym API
    spec = envs.registry.env_specs.get(id)
    if spec is None:
        raise error.Error(f"Environment id '{id}' is not registered")
    module_name, _, attr_name = spec.entry_point.partition(":")
    if not module_name or not attr_name:
        raise ImportError(f"Invalid entry point '{spec.entry_point}'")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    all_kwargs = dict(spec.kwargs)
    all_kwargs.update(override_kwargs)
    return factory(**all_kwargs) if callable(factory) else factory


# wrappers.common.OrderEnforcing shim
class _OrderEnforcing:
    def __init__(self, env: Any):
        self.env = env

    def __getattr__(self, name: str) -> Any:
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)


class wrappers:  # type: ignore
    class common:  # type: ignore
        OrderEnforcing = _OrderEnforcing


__all__ = [
    "envs",
    "register",
    "make",
    "spaces",
    "wrappers",
    "Env",
    "Space",
]


# Minimal base Env class matching Gymnasium's base interface for subclassing in tests
class Env:  # pragma: no cover - simple interface shim
    metadata: Dict[str, Any] = {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        raise NotImplementedError

    def step(self, action: Any):
        raise NotImplementedError

    def render(self, mode: str = "human"):
        return None

    def close(self) -> None:
        pass
