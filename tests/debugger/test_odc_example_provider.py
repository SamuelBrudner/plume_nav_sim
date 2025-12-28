import numpy as np
import pytest

pytest.importorskip(
    "plume_nav_debugger",
    reason="Debugger package not importable; ensure PYTHONPATH=src for local runs",
)

from plume_nav_debugger.odc.mux import ProviderMux  # noqa: E402
from plume_nav_debugger.odc.provider import DebuggerProvider  # noqa: E402


def test_example_provider_reflection_on_env():
    # Use the example provider via env.get_debugger_provider()
    from examples.odc_provider_example import ExampleDebuggerProvider

    class _AS:
        n = 3

    class _Env:
        def __init__(self):
            self.action_space = _AS()

        def get_debugger_provider(self) -> DebuggerProvider:
            return ExampleDebuggerProvider()

    class _Policy:
        pass

    env = _Env()
    pol = _Policy()
    mux = ProviderMux(env, pol)
    # Provider names should be used
    assert mux.get_action_names() == ["FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    # Provider probs should be used
    p = np.array(mux.get_policy_distribution(np.array([0.5])))
    assert p.shape == (3,)
    assert abs(float(p.sum()) - 1.0) < 1e-6
    assert int(np.argmax(p)) == 0
