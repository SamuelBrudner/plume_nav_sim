from __future__ import annotations

import sys
from types import ModuleType

import numpy as np


def _ensure_src_on_path():
    # Make sure 'src' is importable when running tests from repo root
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    src = root / "src"
    p = str(src)
    if p not in sys.path:
        sys.path.insert(0, p)


_ensure_src_on_path()

from plume_nav_sim.compose.policy_loader import (  # noqa: E402
    load_policy,
    reset_policy_if_possible,
)


def make_fake_module(name: str) -> ModuleType:
    mod = ModuleType(name)

    class DummyPolicy:
        def __init__(self):
            self._reset_called = False

        def reset(self, *, seed=None):  # noqa: ARG002 - test double
            self._reset_called = True

        def select_action(self, observation, *, explore=False):  # noqa: ARG002
            return 0

    def dummy_callable(observation):  # noqa: ARG001
        return 1

    class Sub:
        class NestedPolicy(DummyPolicy):
            pass

    mod.DummyPolicy = DummyPolicy
    mod.dummy_callable = dummy_callable
    mod.sub = Sub
    return mod


def test_load_policy_class_and_reset():
    mod = make_fake_module("custom_policy_mod")
    sys.modules[mod.__name__] = mod

    loaded = load_policy("custom_policy_mod:DummyPolicy")
    assert loaded.spec == "custom_policy_mod:DummyPolicy"

    policy = loaded.obj
    # sanity: object has select_action
    assert callable(getattr(policy, "select_action", None))

    # reset should not raise and should flip the flag
    reset_policy_if_possible(policy, seed=123)
    assert getattr(policy, "_reset_called", False) is True

    # runner contract: select_action returns an action (we expect int)
    a = policy.select_action(np.array([0.5]), explore=False)
    assert isinstance(a, int)


def test_load_policy_callable_function():
    mod = make_fake_module("custom_policy_mod2")
    sys.modules[mod.__name__] = mod

    loaded = load_policy("custom_policy_mod2:dummy_callable")
    policy_fn = loaded.obj
    assert callable(policy_fn)
    a = policy_fn(np.array([0.1]))
    assert isinstance(a, int)


def test_load_policy_with_dotted_attribute():
    mod = make_fake_module("custom_policy_mod3")
    sys.modules[mod.__name__] = mod

    # form: module:attr
    loaded1 = load_policy("custom_policy_mod3:sub.NestedPolicy")
    assert type(loaded1.obj).__name__ == "NestedPolicy"

    # form: module.attr (last dot split)
    loaded2 = load_policy("custom_policy_mod3.sub.NestedPolicy")
    assert type(loaded2.obj).__name__ == "NestedPolicy"
