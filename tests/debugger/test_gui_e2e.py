import importlib
import sys

import pytest


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
def test_provider_mux_integration_sets_source_and_action_names(monkeypatch):
    import numpy as np

    QtWidgets = pytest.importorskip(
        "PySide6.QtWidgets",
        reason="PySide6 not available or Qt stack incomplete",
    )

    # Lazy import after Qt available
    try:
        from plume_nav_debugger.app import DebuggerConfig, EnvDriver, InspectorWidget
    except RuntimeError as exc:
        pytest.skip(str(exc))
    from plume_nav_debugger.odc.models import ActionInfo, PipelineInfo

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    class _Provider:
        def get_action_info(self, env):
            return ActionInfo(names=["ZERO", "ONE", "TWO"])  # provider wins

        def policy_distribution(self, policy, observation):
            return {"probs": [0.2, 0.6, 0.2]}

        def get_pipeline(self, env):
            return PipelineInfo(names=["TopEnv", "MyWrapper", "CoreEnv"])

    class _AS:
        n = 3
        _metadata = {"action_names": ["A", "B", "C"]}

    class _Env:
        def __init__(self):
            self.action_space = _AS()
            self._called = 0

        def reset(self, seed=None):
            self._called += 1
            # Minimal info for start xy
            return np.array([0.5], dtype=float), {"agent_xy": (1, 2)}

        def render(self, *args, **kwargs):
            # minimal rgb array
            import numpy as np

            return np.zeros((8, 8, 3), dtype="uint8")

        def close(self):
            pass

        def get_debugger_provider(self):
            return _Provider()

    class _Policy:
        pass

    # Subclass driver to inject fakes without runner.stream
    class _Driver(EnvDriver):
        def initialize(self) -> None:  # type: ignore[override]
            self._env = _Env()
            self._policy = _Policy()
            self._episode_seed = self.config.seed
            # No controller/stream needed to test signals & mux
            # Emit run meta
            _obs0, info0 = self._env.reset(seed=self._episode_seed)
            # use private helper if present
            if hasattr(self, "_update_start_from_info"):
                self._update_start_from_info(self._episode_seed or -1, info0)  # type: ignore[attr-defined]
            # Emit provider mux
            from plume_nav_debugger.odc.mux import ProviderMux

            self._mux = ProviderMux(self._env, self._policy)
            self.provider_mux_changed.emit(self._mux)
            # Emit action names
            self._emit_action_space_changed()

    driver = _Driver(DebuggerConfig())
    insp = InspectorWidget()
    # Ensure strict provider-only is enabled to avoid fallbacks
    insp.set_strict_provider_only(True)
    # Wire driver → inspector
    driver.provider_mux_changed.connect(insp.on_mux_changed)
    driver.action_space_changed.connect(insp.on_action_names)

    # Run initialize to trigger mux + names
    driver.initialize()

    # Inspector should show provider source
    assert "provider" in insp.action_panel.source_label.text().lower()
    # Driver should reflect provider names via mux
    assert driver.get_action_names() == ["ZERO", "ONE", "TWO"]


@pytest.mark.skipif(
    _pyqt5_present(),
    reason="PySide6 not available or PyQt5 present (binding conflict)",
)
def test_strict_mode_no_fallbacks(monkeypatch):
    import numpy as np

    QtWidgets = pytest.importorskip(
        "PySide6.QtWidgets",
        reason="PySide6 not available or Qt stack incomplete",
    )

    try:
        from plume_nav_debugger.app import DebuggerConfig, EnvDriver, InspectorWidget
    except RuntimeError as exc:
        pytest.skip(str(exc))

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    class _AS:
        n = 3

    class _Env:
        def __init__(self):
            self.action_space = _AS()

        def reset(self, seed=None):
            return np.array([0.5], dtype=float), {"agent_xy": (0, 0)}

        def render(self, *a, **k):
            return np.zeros((4, 4, 3), dtype="uint8")

        def close(self):
            pass

    class _Policy:
        pass

    class _Driver(EnvDriver):
        def initialize(self) -> None:  # type: ignore[override]
            self._env = _Env()
            self._policy = _Policy()
            self._episode_seed = self.config.seed
            _obs0, info0 = self._env.reset(seed=self._episode_seed)
            if hasattr(self, "_update_start_from_info"):
                self._update_start_from_info(self._episode_seed or -1, info0)  # type: ignore[attr-defined]
            # Do not emit provider mux (simulate no provider)
            self._emit_action_space_changed()

    driver = _Driver(DebuggerConfig())
    insp = InspectorWidget()
    insp.set_strict_provider_only(True)
    driver.provider_mux_changed.connect(insp.on_mux_changed)
    driver.action_space_changed.connect(insp.on_action_names)
    driver.initialize()
    # In strict mode with no provider, action names should be empty (no fallback)
    assert driver.get_action_names() == []
    # Inspector should show provider-required banner
    assert getattr(insp, "info_label", None) is not None
    assert insp.info_label.isVisible()
    assert "provider" in insp.info_label.text().lower()
