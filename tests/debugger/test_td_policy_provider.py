import numpy as np
import pytest

pytest.importorskip(
    "plume_nav_debugger",
    reason="Debugger package not importable; ensure PYTHONPATH=src for local runs",
)


def test_td_policy_provider_distribution() -> None:
    from plume_nav_debugger.odc.mux import ProviderMux
    from plume_nav_sim.policies import TemporalDerivativePolicy

    class _AS:
        n = 3

    class _Env:
        action_space = _AS()

    policy = TemporalDerivativePolicy(eps=0.05)
    obs1 = np.array([0.2], dtype=float)
    obs2 = np.array([0.1], dtype=float)
    policy.select_action(obs1, explore=False)
    policy.select_action(obs2, explore=False)

    mux = ProviderMux(_Env(), policy)
    dist = mux.get_policy_distribution(obs2)

    assert dist is not None
    assert len(dist) == 3
    assert dist[0] == 0.0
    assert dist[1] == pytest.approx(0.5, rel=1e-6, abs=1e-6)
    assert dist[2] == pytest.approx(0.5, rel=1e-6, abs=1e-6)
