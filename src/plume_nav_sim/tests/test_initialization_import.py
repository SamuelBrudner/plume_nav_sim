import importlib
import logging
import builtins
import sys

import pytest

import pandas


def test_agent_initializer_protocol_import_logs_success(caplog):
    caplog.set_level(logging.INFO)
    import plume_nav_sim.core.initialization as init
    importlib.reload(init)
    assert "AgentInitializerProtocol import succeeded" in caplog.text


def test_missing_protocols_raises_import_error(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        target = "plume_nav_sim.core.protocols"
        pkg = globals.get("__package__") if globals else None
        if name == target or (level and name == "protocols" and pkg == "plume_nav_sim.core"):
            raise ModuleNotFoundError(f"No module named '{target}'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("plume_nav_sim.core.protocols", None)
    sys.modules.pop("plume_nav_sim.core.initialization", None)
    core_pkg = sys.modules.get("plume_nav_sim.core")
    if core_pkg and hasattr(core_pkg, "protocols"):
        delattr(core_pkg, "protocols")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("plume_nav_sim.core.initialization")
