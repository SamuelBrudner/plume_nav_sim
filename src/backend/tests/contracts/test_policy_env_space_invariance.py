from __future__ import annotations

from typing import get_args

import pytest

import plume_nav_sim as pns
from plume_nav_sim.compose.builders import build_policy
from plume_nav_sim.compose.specs import BuiltinPolicyName, PolicySpec


def _make_env(action_type: str):
    return pns.make_env(
        grid_size=(8, 8),
        source_location=(4, 4),
        max_steps=5,
        action_type=action_type,
        observation_type="concentration",
        reward_type="step_penalty",
        render_mode=None,
    )


def _spaces_identical(env_space, pol_space) -> bool:
    # For now we only use Discrete action spaces; compare n when types match
    try:
        from gymnasium.spaces import Discrete  # type: ignore

        return (
            isinstance(env_space, Discrete)
            and isinstance(pol_space, Discrete)
            and (int(env_space.n) == int(pol_space.n))
        )
    except Exception:  # pragma: no cover
        return False


@pytest.mark.parametrize("builtin", [b for b in get_args(BuiltinPolicyName)])
def test_builtin_policy_has_matching_env_action_space(builtin: str):
    # Skip policies that intentionally have no declared action space (e.g., 'random')
    if builtin == "random":
        pytest.skip("random policy has no intrinsic action_space")

    candidates = ("discrete", "oriented", "run_tumble")
    found = False
    last_error = None
    for action_type in candidates:
        env = _make_env(action_type)
        try:
            pol = build_policy(PolicySpec(builtin=builtin), env=env)
            env_space = getattr(env, "action_space", None)
            pol_space = getattr(pol, "action_space", None)
            if env_space is None or pol_space is None:
                last_error = f"Missing action_space: env={env_space} pol={pol_space}"
                continue
            if _spaces_identical(env_space, pol_space):
                found = True
                break
        finally:
            env.close()
    assert (
        found
    ), f"No environment action_type produced identical action space for builtin '{builtin}' ({last_error})"
