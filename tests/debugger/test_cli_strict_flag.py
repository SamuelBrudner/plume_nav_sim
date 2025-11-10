import importlib
import sys

import numpy as np
import pytest


def _pyqt5_present() -> bool:
    try:
        if "PyQt5" in sys.modules:
            return True
        return importlib.util.find_spec("PyQt5") is not None
    except Exception:
        return False


@pytest.mark.skipif(
    (
        "PySide6" not in __import__("sys").modules
        and __import__("importlib").util.find_spec("PySide6") is None
    )
    or _pyqt5_present(),
    reason="PySide6 not available or PyQt5 present (binding conflict)",
)
def test_cli_no_strict_provider_only_smoke(monkeypatch):
    # Import after skip check to avoid raising when PySide6 unavailable
    import plume_nav_debugger.app as app_mod

    # Stub QApplication to avoid launching a real event loop
    class _App:
        def __init__(self, argv):
            self._argv = list(argv)

        def exec(self):
            return 0

    monkeypatch.setattr(app_mod, "QtWidgets", type("_QtW", (), {"QApplication": _App}))

    # Replace MainWindow with a minimal stub that records applied flags
    class _StubPrefs:
        def __init__(self):
            self.strict_provider_only = True

    class _StubDriverCfg:
        def __init__(self):
            self.strict_provider_only = True

    class _StubDriver:
        def __init__(self):
            self.config = _StubDriverCfg()

    class _StubInspector:
        def __init__(self, sink):
            self._sink = sink

        def set_strict_provider_only(self, flag: bool):
            self._sink["inspector_flag"] = bool(flag)

    sink = {"win": None, "inspector_flag": None}

    class _StubWindow:
        def __init__(self):
            sink["win"] = self
            self.prefs = _StubPrefs()
            self.driver = _StubDriver()
            self.inspector = _StubInspector(sink)

        def show(self):
            pass

    monkeypatch.setattr(app_mod, "MainWindow", _StubWindow)

    # Simulate CLI: disable strict mode via flag
    monkeypatch.setattr(app_mod, "sys", sys)
    monkeypatch.setenv("QT_MAC_WANTS_LAYER", "1")  # avoid platform issues on macOS
    argv = ["-m", "plume_nav_debugger.app", "--no-strict-provider-only"]
    monkeypatch.setattr(sys, "argv", argv, raising=False)

    # Avoid exiting the test process
    def _noexit(_code=0):
        raise SystemExit(_code)

    monkeypatch.setattr(sys, "exit", _noexit, raising=False)

    with pytest.raises(SystemExit) as excinfo:
        app_mod.main()
    assert int(getattr(excinfo.value, "code", 1)) == 0

    # Assert the override propagated to prefs, driver, and inspector
    win = sink["win"]
    assert win is not None
    assert win.prefs.strict_provider_only is False
    assert win.driver.config.strict_provider_only is False
    assert sink["inspector_flag"] is False
