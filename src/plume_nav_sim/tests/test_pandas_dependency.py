import builtins
import importlib
import pytest


def test_initialization_requires_pandas(monkeypatch):
    """Ensure pandas is a hard dependency for initialization tests."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("pandas missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(pytest, "importorskip", lambda *a, **k: object())

    with pytest.raises(ImportError):
        import plume_nav_sim.tests.test_initialization_import as ti
        importlib.reload(ti)
