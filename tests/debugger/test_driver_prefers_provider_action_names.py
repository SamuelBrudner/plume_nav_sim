import importlib
import sys

import numpy as np
import pytest

pytest.importorskip(
    "plume_nav_debugger",
    reason="Debugger package not importable; ensure PYTHONPATH=src for local runs",
)


def _pyqt5_present() -> bool:
    try:
        if "PyQt5" in sys.modules:
            return True
        return importlib.util.find_spec("PyQt5") is not None
    except Exception:
        return False


@pytest.mark.skipif(
    _pyqt5_present(),
    reason="PySide6 not available or PyQt5 present (binding conflict)",
)
def test_driver_prefers_provider_action_names_strict_mode():
    try:
        from plume_nav_debugger.env_driver import DebuggerConfig, EnvDriver
    except RuntimeError as exc:
        pytest.skip(str(exc))
    from plume_nav_debugger.odc.models import ActionInfo
    from plume_nav_debugger.odc.mux import ProviderMux

    class _Provider:
        def get_action_info(self, env):
            # Provider offers names that should override env metadata
            return ActionInfo(names=["ZERO", "ONE", "TWO"])

    class _AS:
        n = 3
        _metadata = {"action_names": ["A", "B", "C"]}

    class _Env:
        def __init__(self):
            self.action_space = _AS()

        def reset(self, seed=None):
            return np.array([0.0], dtype=float), {"agent_xy": (0, 0)}

        def render(self, *a, **k):
            return np.zeros((2, 2, 3), dtype="uint8")

        def close(self):
            pass

    class _Policy:
        pass

    class _Driver(EnvDriver):
        def initialize(self) -> None:  # type: ignore[override]
            # Inject fakes and explicit provider-backed mux
            self._env = _Env()
            self._policy = _Policy()
            self._episode_seed = self.config.seed
            _obs0, info0 = self._env.reset(seed=self._episode_seed)
            if hasattr(self, "_update_start_from_info"):
                self._update_start_from_info(self._episode_seed or -1, info0)  # type: ignore[attr-defined]
            self._mux = ProviderMux(self._env, self._policy, provider=_Provider())

    # Strict mode is always enforced; names should come from provider (no heuristics)
    driver = _Driver(DebuggerConfig())
    driver.initialize()
    assert driver.get_action_names() == ["ZERO", "ONE", "TWO"]
