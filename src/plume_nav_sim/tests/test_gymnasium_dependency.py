"""Tests that PlumeNavigationEnv hard depends on gymnasium."""

import importlib
import sys

import pytest


def test_import_fails_without_gymnasium(monkeypatch):
    """Importing PlumeNavigationEnv should raise ImportError when gymnasium is missing."""
    # Remove gymnasium modules to simulate the dependency being absent
    for mod in [m for m in list(sys.modules) if m.startswith("gymnasium")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setitem(sys.modules, "gymnasium", None)
    monkeypatch.delitem(sys.modules, "plume_nav_sim.envs.plume_navigation_env", raising=False)

    with pytest.raises(ImportError):
        importlib.import_module("plume_nav_sim.envs.plume_navigation_env")
