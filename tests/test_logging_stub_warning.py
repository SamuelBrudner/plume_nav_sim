import sys
import types
import importlib.metadata
import pytest


def _install_loguru_stub(monkeypatch):
    """Install a loguru stub lacking required features."""
    loguru_stub = types.ModuleType("loguru")

    class StubLogger:
        """Minimal logger without configure support."""

    loguru_stub.logger = StubLogger()
    monkeypatch.setitem(sys.modules, "loguru", loguru_stub)


def _patch_distribution(monkeypatch):
    real_dist = importlib.metadata.distribution

    class DummyDist:
        def locate_file(self, path):
            return path

    def fake_distribution(name):
        if name == "plume_nav_sim":
            return DummyDist()
        return real_dist(name)

    monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)


def test_import_fails_with_loguru_stub(monkeypatch):
    """Importing with a loguru stub should raise ImportError."""
    _install_loguru_stub(monkeypatch)
    _patch_distribution(monkeypatch)
    sys.modules.pop("plume_nav_sim", None)

    with pytest.raises(ImportError, match="loguru.*configure"):
        import plume_nav_sim  # noqa: F401
