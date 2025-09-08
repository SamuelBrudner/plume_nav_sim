import importlib
import builtins
import sys
import importlib.metadata as importlib_metadata

import pytest

MODULE = "plume_nav_sim.envs"


def _patch_distribution(monkeypatch):
    class _FakeDist:
        def locate_file(self, path):
            return path
    monkeypatch.setattr(importlib_metadata, "distribution", lambda name: _FakeDist())


def test_import_requires_video_plume(monkeypatch):
    _patch_distribution(monkeypatch)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    monkeypatch.delitem(sys.modules, f"{MODULE}.video_plume", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == f"{MODULE}.video_plume":
            raise ImportError("VideoPlume not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        importlib.import_module(MODULE)


def test_import_requires_plume_navigation_env(monkeypatch):
    _patch_distribution(monkeypatch)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    monkeypatch.delitem(sys.modules, f"{MODULE}.plume_navigation_env", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == f"{MODULE}.plume_navigation_env":
            raise ImportError("PlumeNavigationEnv not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        importlib.import_module(MODULE)
