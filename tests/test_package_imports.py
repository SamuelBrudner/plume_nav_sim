import builtins
import sys
import pytest


def test_import_fails_when_core_unavailable(monkeypatch):
    for mod in list(sys.modules):
        if mod.startswith("plume_nav_sim"):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "plume_nav_sim.core":
            raise ImportError("core modules missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        import plume_nav_sim  # noqa: F401
