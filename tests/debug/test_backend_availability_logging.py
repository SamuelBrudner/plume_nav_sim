import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest


class _DummyLogger(logging.Logger):
    """Logger that ignores reserved fields in extra."""
    def makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None):
        if extra:
            extra = {k: v for k, v in extra.items() if k not in logging.LogRecord.__dict__}
            extra.pop("module", None)
        return super().makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra, sinfo)


@pytest.fixture
def debug_module(monkeypatch):
    """Load debug.__init__ with stubbed dependencies."""
    # Stub logging setup
    logging_setup = types.ModuleType("plume_nav_sim.utils.logging_setup")
    def get_logger(name):
        logging.setLoggerClass(_DummyLogger)
        return logging.getLogger(name)
    logging_setup.get_logger = get_logger

    # Stub gui and cli modules
    gui_stub = types.ModuleType("plume_nav_sim.debug.gui")
    for attr in ["DebugGUI", "DebugSession", "DebugConfig", "plot_initial_state", "launch_viewer"]:
        setattr(gui_stub, attr, object)
    cli_stub = types.ModuleType("plume_nav_sim.debug.cli")
    cli_stub.debug_group = lambda: None

    monkeypatch.setitem(sys.modules, "plume_nav_sim.utils.logging_setup", logging_setup)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.debug.gui", gui_stub)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.debug.cli", cli_stub)
    monkeypatch.setitem(sys.modules, "plume_nav_sim", types.ModuleType("plume_nav_sim"))

    # Patch backend detection to simulate PySide6 available only
    class DummySpec:
        pass
    def fake_find_spec(name):
        if name == "PySide6":
            return DummySpec()
        return None
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    path = Path("src/plume_nav_sim/debug/__init__.py")
    spec = importlib.util.spec_from_file_location("plume_nav_sim.debug", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_backend_detection_logs(debug_module, caplog):
    caplog.set_level(logging.DEBUG)
    debug_module._detect_backend_availability()
    assert any("PySide6 backend available" in r.message for r in caplog.records)
    assert any("Streamlit backend not available" in r.message for r in caplog.records)
