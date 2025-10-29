from __future__ import annotations

from typing import Any


def reset_then_step(env: Any, action: Any, *, seed: int | None = None):
    _ = env.reset(seed=seed)
    return env.step(action)


def step_once_safely(env: Any, action: Any, *, seed: int | None = None):
    try:
        return env.step(action)
    except Exception as e:
        msg = str(e).lower()
        if "must call reset" in msg or "must be in ready" in msg:
            _ = env.reset(seed=seed)
            return env.step(action)
        raise
