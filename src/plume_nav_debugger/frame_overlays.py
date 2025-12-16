from __future__ import annotations

from typing import Any, Optional


def _iter_env_chain(env: Any):
    cur = env
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        nxt = None
        for attr in ("env", "_env", "_core_env"):
            try:
                candidate = getattr(cur, attr, None)
            except Exception:
                candidate = None
            if candidate is not None and candidate is not cur:
                nxt = candidate
                break
        if nxt is None:
            try:
                candidate = getattr(cur, "unwrapped", None)
            except Exception:
                candidate = None
            if candidate is not None and candidate is not cur:
                nxt = candidate
        cur = nxt


def _coerce_xy(value: Any) -> Optional[tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, dict):
        try:
            if "x" in value and "y" in value:
                return int(value["x"]), int(value["y"])
        except Exception:
            return None
    try:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return int(value[0]), int(value[1])
    except Exception:
        return None
    # Coordinates-like object with x/y
    try:
        x = getattr(value, "x", None)
        y = getattr(value, "y", None)
        if x is not None and y is not None:
            return int(x), int(y)
    except Exception:
        return None
    return None


def _try_get_agent_state(env: Any) -> Any:
    for cur in _iter_env_chain(env):
        for attr in ("_agent_state", "agent_state"):
            try:
                st = getattr(cur, attr, None)
            except Exception:
                st = None
            if st is not None:
                return st
    return None


def _try_get_agent_xy(env: Any) -> Optional[tuple[int, int]]:
    st = _try_get_agent_state(env)
    if st is not None:
        try:
            pos = getattr(st, "position", None)
        except Exception:
            pos = None
        xy = _coerce_xy(pos)
        if xy is not None:
            return xy
        xy = _coerce_xy(st)
        if xy is not None:
            return xy
    # Fallback: base env style agent_pos
    for cur in _iter_env_chain(env):
        for attr in ("agent_pos", "agent_position", "_agent_pos", "_agent_position"):
            try:
                val = getattr(cur, attr, None)
            except Exception:
                val = None
            xy = _coerce_xy(val)
            if xy is not None:
                return xy
    return None


def _try_get_agent_orientation(env: Any) -> Optional[float]:
    st = _try_get_agent_state(env)
    if st is None:
        return None
    try:
        ori = getattr(st, "orientation", None)
    except Exception:
        return None
    if ori is None:
        return None
    try:
        return float(ori) % 360.0
    except Exception:
        return None


def _try_get_source_xy(env: Any) -> Optional[tuple[int, int]]:
    for cur in _iter_env_chain(env):
        for attr in ("source_location", "goal_location"):
            try:
                val = getattr(cur, attr, None)
            except Exception:
                val = None
            xy = _coerce_xy(val)
            if xy is not None:
                return xy
    return None


def _try_get_goal_radius(env: Any) -> Optional[float]:
    for cur in _iter_env_chain(env):
        try:
            val = getattr(cur, "goal_radius", None)
        except Exception:
            val = None
        if val is None:
            continue
        try:
            return float(val)
        except Exception:
            continue
    return None


def augment_info_for_overlays(info: Any, env: Any) -> dict:
    base = dict(info) if isinstance(info, dict) else {}

    pos = base.get("agent_position") or base.get("agent_xy")
    xy = _coerce_xy(pos)
    if xy is None:
        xy = _try_get_agent_xy(env)
    if xy is not None:
        base.setdefault("agent_xy", xy)

    ori = base.get("agent_orientation")
    if ori is None:
        ori = _try_get_agent_orientation(env)
    else:
        try:
            ori = float(ori) % 360.0
        except Exception:
            ori = None
    if ori is not None:
        base.setdefault("agent_orientation", float(ori))

    src = base.get("source_xy")
    src_xy = _coerce_xy(src)
    if src_xy is None:
        src_xy = _try_get_source_xy(env)
    if src_xy is not None:
        base.setdefault("source_xy", src_xy)

    gr = base.get("goal_radius")
    if gr is None:
        gr = _try_get_goal_radius(env)
    else:
        try:
            gr = float(gr)
        except Exception:
            gr = None
    if gr is not None:
        base.setdefault("goal_radius", float(gr))

    return base


class OverlayInfoWrapper:
    """Augment env.reset/env.step info with overlay-friendly fields.

    Purely adds metadata to the returned info dict; does not mutate env state.
    """

    def __init__(self, env: Any) -> None:
        self.env = env

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, augment_info_for_overlays(info, self.env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            obs,
            reward,
            terminated,
            truncated,
            augment_info_for_overlays(info, self.env),
        )

    def render(self, *args: Any, **kwargs: Any) -> Any:
        return self.env.render(*args, **kwargs)

    def close(self) -> None:
        try:
            if hasattr(self.env, "close"):
                self.env.close()
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        # Forward everything else (action_space, observation_space, etc)
        return getattr(self.env, name)
