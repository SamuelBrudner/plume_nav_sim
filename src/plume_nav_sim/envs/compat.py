"""Compatibility utilities for dual Gym/Gymnasium API support."""

from __future__ import annotations

import inspect
import logging
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

APIVersionResult = namedtuple(
    "APIVersionResult", ["is_legacy", "confidence", "detection_method"]
)


@dataclass
class CompatibilityMode:
    """Configuration for wrapping environments to a specific API format."""

    use_legacy_api: bool
    detection: APIVersionResult
    measure_timing: bool = False
    correlation_id: Optional[str] = None


def detect_api_version() -> APIVersionResult:
    """Detect whether the caller expects the legacy Gym API.

    This uses a simple stack inspection strategy: if any frame in the call stack
    defines a ``gym`` module that appears to be the legacy package (``__name__`` is
    ``"gym"``), the caller is treated as legacy. The result of this detection is
    logged for diagnostic purposes.
    """

    legacy_detected = False
    stack = inspect.stack()
    for frame_info in stack[1:]:
        frame = frame_info.frame
        gym_obj = None
        if "gym" in frame.f_globals:
            gym_obj = frame.f_globals["gym"]
        elif "gym" in frame.f_locals:
            gym_obj = frame.f_locals["gym"]
        if gym_obj is None:
            continue
        name = getattr(gym_obj, "__name__", "")
        if name == "gym":
            legacy_detected = True
            break
    # Confidence is high if we detected legacy usage
    confidence = 0.9 if legacy_detected else 0.5
    result = APIVersionResult(legacy_detected, confidence, "stack_inspection")
    logger.info(
        "API version detection",
        extra={
            "is_legacy": result.is_legacy,
            "confidence": result.confidence,
            "method": result.detection_method,
        },
    )
    return result


def wrap_environment(env: Any, mode: CompatibilityMode):
    """Wrap an environment to convert between legacy and modern API formats.

    The wrapper inspects ``mode.use_legacy_api`` to determine the desired output
    format. Conversions between 4-tuple (legacy) and 5-tuple (modern) step results
    as well as between ``reset`` return styles are logged.
    """

    if env is None:
        raise ValueError("Environment to wrap cannot be None")

    class EnvWrapper:
        def __init__(self, inner):
            self._env = inner

        def __getattr__(self, item):
            return getattr(self._env, item)

        def reset(self, *args, **kwargs):
            result = self._env.reset(*args, **kwargs)
            if mode.use_legacy_api:
                if isinstance(result, tuple):
                    if len(result) == 2:
                        obs, info = result
                        logger.debug("Converting reset output from 2-tuple to obs for legacy API")
                        return obs
                    elif len(result) == 1:
                        return result[0]
                    else:
                        raise ValueError("Unexpected reset return length for legacy API")
                logger.debug("Reset output already legacy compatible")
                return result
            else:
                if isinstance(result, tuple):
                    if len(result) == 2:
                        return result
                    elif len(result) == 1:
                        obs = result[0]
                        logger.debug("Converting reset output from obs to (obs, {}) for modern API")
                        return obs, {}
                    else:
                        raise ValueError("Unexpected reset return length for modern API")
                logger.debug("Converting reset output from obs to (obs, {}) for modern API")
                return result, {}

        def step(self, action, *args, **kwargs):
            result = self._env.step(action, *args, **kwargs)
            if not isinstance(result, tuple):
                raise ValueError("Environment step must return a tuple")
            if mode.use_legacy_api:
                if len(result) == 5:
                    obs, reward, terminated, truncated, info = result
                    done = bool(terminated) or bool(truncated)
                    logger.debug("Converting step output from 5-tuple to 4-tuple for legacy API")
                    return obs, reward, done, info
                if len(result) == 4:
                    logger.debug("Step output already legacy compatible")
                    return result
                raise ValueError("Unexpected step return length for legacy API")
            else:
                if len(result) == 4:
                    obs, reward, done, info = result
                    logger.debug("Converting step output from 4-tuple to 5-tuple for modern API")
                    return obs, reward, bool(done), False, info
                if len(result) == 5:
                    logger.debug("Step output already modern compatible")
                    return result
                raise ValueError("Unexpected step return length for modern API")

    return EnvWrapper(env)

__all__ = [
    "APIVersionResult",
    "CompatibilityMode",
    "detect_api_version",
    "wrap_environment",
]
