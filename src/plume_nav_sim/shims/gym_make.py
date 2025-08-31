"""Legacy-compatible environment creation shim for Gymnasium."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import gymnasium as gym

from plume_nav_sim import warn_if_legacy_env_usage
from plume_nav_sim.envs.compat import (
    CompatibilityMode,
    detect_api_version,
    wrap_environment,
)

logger = logging.getLogger(__name__)


def gym_make(env_id: str, **kwargs: Any):
    """Create a Gymnasium environment with legacy gym compatibility.

    This shim normalizes environment identifiers, emits a deprecation warning,
    and wraps the returned environment so that legacy callers receive 4-tuple
    step/reset semantics when necessary.
    """
    if not isinstance(env_id, str):
        raise TypeError("env_id must be a string")

    warnings.warn(
        "Using gym_make is deprecated and will be removed in v1.0. "
        "Use gymnasium.make() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.info("gym_make called", extra={"env_id": env_id})

    normalized_id = warn_if_legacy_env_usage(env_id)

    detection = detect_api_version()
    mode = CompatibilityMode(use_legacy_api=detection.is_legacy, detection=detection)

    env = gym.make(normalized_id, **kwargs)
    env = wrap_environment(env, mode)
    return env


__all__ = ["gym_make"]
