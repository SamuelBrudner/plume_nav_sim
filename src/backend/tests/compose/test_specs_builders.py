from __future__ import annotations

import sys

import numpy as np


def _ensure_src_on_path():
    import pathlib

    # Repository root is 4 levels up from this file: src/backend/tests/compose
    root = pathlib.Path(__file__).resolve().parents[4]
    src = root / "src"
    p = str(src)
    if p not in sys.path:
        sys.path.insert(0, p)


_ensure_src_on_path()

from plume_nav_sim.compose.builders import (  # noqa: E402
    build_env,
    build_policy,
    prepare,
)
from plume_nav_sim.compose.specs import PolicySpec, SimulationSpec  # noqa: E402
from plume_nav_sim.runner import runner as r  # noqa: E402


def test_prepare_with_builtin_deterministic_policy():
    sim = SimulationSpec(
        grid_size=(16, 16),
        max_steps=50,
        render=False,
        policy=PolicySpec(builtin="deterministic_td"),
        seed=123,
    )

    env, policy = prepare(sim)
    # Stream at least one step; ensure deterministic across two builds with same seed
    seq1 = [(ev.action, ev.reward) for ev in r.stream(env, policy, seed=sim.seed)]

    env2, policy2 = prepare(sim)
    seq2 = [(ev.action, ev.reward) for ev in r.stream(env2, policy2, seed=sim.seed)]

    assert seq1 == seq2 and len(seq1) > 0


def test_build_policy_custom_dotted_class_with_kwargs():
    # Create a fake module with a policy requiring kwargs at init
    import types

    mod = types.ModuleType("fake_policies")

    class NeedsKwargs:
        def __init__(self, gain: float = 1.0):
            self.gain = gain

        def select_action(self, observation, *, explore=False):  # noqa: ARG002
            # Return 0 (FORWARD) if positive scaled obs, else 1
            val = float(observation[0]) * self.gain
            return 0 if val >= 0.0 else 1

    mod.NeedsKwargs = NeedsKwargs
    sys.modules[mod.__name__] = mod

    ps = PolicySpec(spec="fake_policies:NeedsKwargs", kwargs={"gain": 0.0})
    # build_policy should instantiate the class with kwargs
    policy = build_policy(ps)
    a = policy.select_action(np.array([0.5]), explore=False)
    assert a in (0, 1)
