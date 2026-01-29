from __future__ import annotations

import importlib
from typing import Dict, Optional

import gymnasium

from ..constants import DEFAULT_MAX_STEPS, ENVIRONMENT_ID

ENV_ID = ENVIRONMENT_ID
ENTRY_POINT = "plume_nav_sim.envs.plume_env:create_plume_env"
MAX_EPISODE_STEPS = DEFAULT_MAX_STEPS

__all__ = [
    "register_env",
    "unregister_env",
    "is_registered",
    "ensure_registered",
    "get_registration_status",
    "ENV_ID",
    "ENTRY_POINT",
]


def _registry_dict() -> Dict[str, object]:
    reg = getattr(gymnasium.envs, "registry", None)
    if isinstance(reg, dict):
        return reg
    env_specs = getattr(reg, "env_specs", None)
    if isinstance(env_specs, dict):
        return env_specs
    reg2 = getattr(getattr(gymnasium.envs, "registration", None), "registry", None)
    if isinstance(reg2, dict):
        return reg2
    raise RuntimeError("Unable to access Gymnasium registry")


def _validate_entry_point(entry_point: str) -> None:
    if ":" not in entry_point:
        raise ValueError("entry_point must be in 'module:object' form")
    module_path, attr = entry_point.split(":", 1)
    if not module_path or not attr:
        raise ValueError("entry_point must specify module and object")
    module = importlib.import_module(module_path)
    if not hasattr(module, attr):
        raise ImportError(f"entry_point object '{attr}' not found in {module_path}")


def register_env(
    env_id: Optional[str] = None,
    *,
    entry_point: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    kwargs: Optional[Dict[str, object]] = None,
    force_reregister: bool = False,
) -> str:
    effective_env_id = env_id or ENV_ID
    if not isinstance(effective_env_id, str) or not effective_env_id.endswith("-v0"):
        raise ValueError("env_id must be a string ending with '-v0'")
    effective_entry_point = ENTRY_POINT if entry_point is None else entry_point
    effective_max_steps = MAX_EPISODE_STEPS if max_episode_steps is None else int(
        max_episode_steps
    )
    if effective_max_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    if kwargs is not None and not isinstance(kwargs, dict):
        raise ValueError("kwargs must be a dict when provided")

    _validate_entry_point(effective_entry_point)

    if is_registered(effective_env_id):
        if not force_reregister:
            raise ValueError(f"Environment '{effective_env_id}' already registered")
        unregister_env(effective_env_id)

    gymnasium.register(
        id=effective_env_id,
        entry_point=effective_entry_point,
        max_episode_steps=effective_max_steps,
        disable_env_checker=True,
        kwargs=kwargs or {},
        additional_wrappers=(),
    )
    try:
        spec = gymnasium.spec(effective_env_id)
        if spec is not None:
            if hasattr(spec, "order_enforcing"):
                spec.order_enforcing = False
            if hasattr(spec, "max_episode_steps"):
                spec.max_episode_steps = None
    except Exception:
        pass

    return effective_env_id


def unregister_env(env_id: Optional[str] = None) -> bool:
    effective_env_id = env_id or ENV_ID
    registry = _registry_dict()
    if effective_env_id in registry:
        del registry[effective_env_id]
        return True
    return False


def is_registered(env_id: Optional[str] = None) -> bool:
    effective_env_id = env_id or ENV_ID
    try:
        return effective_env_id in _registry_dict()
    except Exception:
        return False


def ensure_registered() -> bool:
    if is_registered():
        return True
    register_env()
    return True


def get_registration_status() -> Dict[str, object]:
    return {
        "env_id": ENV_ID,
        "entry_point": ENTRY_POINT,
        "registered": is_registered(),
        "gymnasium_version": getattr(gymnasium, "__version__", "unknown"),
    }
