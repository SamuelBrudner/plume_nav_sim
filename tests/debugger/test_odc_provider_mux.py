import numpy as np
import pytest

pytest.importorskip(
    "plume_nav_debugger",
    reason="Debugger package not importable; ensure PYTHONPATH=src for local runs",
)

from plume_nav_debugger.odc.models import ActionInfo, PipelineInfo
from plume_nav_debugger.odc.mux import ProviderMux


class _Provider:
    def __init__(self):
        self.calls = {"action": 0, "dist": 0, "pipe": 0}

    def get_action_info(self, env):
        self.calls["action"] += 1
        return ActionInfo(names=["ZERO", "ONE", "TWO"])  # provider wins

    def policy_distribution(self, policy, observation):
        self.calls["dist"] += 1
        # Provide probs directly
        return {"probs": [0.25, 0.5, 0.25]}

    def get_pipeline(self, env):
        self.calls["pipe"] += 1
        return PipelineInfo(names=["TopEnv", "MyWrapper", "CoreEnv"])  # provider wins


class _Env:
    class _AS:
        def __init__(self):
            self._metadata = {"action_names": ["A", "B", "C"]}
            self.n = 3

    def __init__(self):
        self.action_space = self._AS()


class _PolicyHeuristic:
    def __init__(self):
        self.called_select = 0

    def select_action(self, obs, explore=False):
        self.called_select += 1
        return 0

    def q_values(self, obs):
        return np.array([1.0, 0.0, -1.0])


def test_provider_mux_prefers_provider_for_actions_distribution_pipeline():
    env = _Env()
    pol = _PolicyHeuristic()
    mux = ProviderMux(env, pol, provider=_Provider())

    # Actions: provider wins over action_space metadata
    names = mux.get_action_names()
    assert names == ["ZERO", "ONE", "TWO"]

    # Distribution: provider wins; no select_action side effects
    obs = np.array([0.5])
    p = np.array(mux.get_policy_distribution(obs))
    assert abs(float(p.sum()) - 1.0) < 1e-6
    assert pol.called_select == 0

    # Pipeline: provider wins over heuristic chain
    chain = mux.get_pipeline()
    assert chain[:2] == ["TopEnv", "MyWrapper"]


def test_provider_mux_falls_back_when_provider_absent():
    env = _Env()
    pol = _PolicyHeuristic()
    mux = ProviderMux(env, pol, provider=None)

    # Strict mode: no provider means no labels or distribution
    assert mux.get_action_names() == []

    obs = np.array([0.0])
    assert mux.get_policy_distribution(obs) is None


def test_provider_invalid_action_names_length_falls_back_to_metadata():
    class _BadProvider(_Provider):
        def get_action_info(self, env):  # type: ignore[override]
            # Wrong length: 2 vs env.action_space.n = 3
            return ActionInfo(names=["X", "Y"])  # invalid

    env = _Env()
    pol = _PolicyHeuristic()
    mux = ProviderMux(env, pol, provider=_BadProvider())
    # Strict mode: invalid provider response yields empty labels, no heuristics
    assert mux.get_action_names() == []


def test_provider_invalid_distribution_length_returns_none():
    class _BadProvider(_Provider):
        def policy_distribution(self, policy, observation):  # type: ignore[override]
            # Wrong length: 2 vs env.action_space.n = 3
            return {"probs": [0.5, 0.5]}

    env = _Env()
    pol = _PolicyHeuristic()
    mux = ProviderMux(env, pol, provider=_BadProvider())
    obs = np.array([0.0])
    assert mux.get_policy_distribution(obs) is None
