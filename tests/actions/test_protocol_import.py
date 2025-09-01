import importlib
import sys
import builtins
import pytest


def test_action_interface_protocol_import(monkeypatch):
    """ActionInterfaceProtocol should import and fail loudly when protocols module is missing."""
    # successful import should expose ActionInterfaceProtocol from protocols
    import plume_nav_sim.core.actions as actions
    assert actions.ActionInterfaceProtocol.__name__ == "ActionInterfaceProtocol"

    # simulate missing protocols module
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith(".protocols"):
            raise ImportError("protocols module missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "plume_nav_sim.core.actions", raising=False)

    with pytest.raises(ImportError):
        importlib.import_module("plume_nav_sim.core.actions")
