"""Tests for ensuring visualization module fails fast when dependencies are missing."""

import importlib
import builtins
import sys

import pytest

MODULE_PATH = "plume_nav_sim.utils.visualization"

@pytest.mark.parametrize("missing_module", [
    "plume_nav_sim.utils.logging_setup",
    "PySide6",
    "streamlit",
    "hydra.core.config_store",
])
def test_import_fails_when_dependency_missing(monkeypatch, missing_module):
    """Visualization imports should raise ImportError when dependencies are missing."""

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == missing_module or name.startswith(missing_module + "."):
            raise ImportError(f"No module named {missing_module}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop(MODULE_PATH, None)

    with pytest.raises(ImportError):
        importlib.import_module(MODULE_PATH)
