import importlib
import builtins
import sys
from pathlib import Path

import pytest


def _simulate_missing_modules(monkeypatch, *module_names):
    """Remove modules and make future imports fail for specified names."""
    # Ensure source path is available for import
    src_path = Path(__file__).resolve().parents[2] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Remove modules from cache
    for name in list(sys.modules):
        if any(name == m or name.startswith(m + ".") for m in module_names):
            monkeypatch.delitem(sys.modules, name, raising=False)

    # Make future imports raise ImportError
    original_import = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if any(name == m or name.startswith(m + ".") for m in module_names):
            raise ImportError(f"No module named {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)


def test_missing_numba_raises_import_error(monkeypatch):
    _simulate_missing_modules(monkeypatch, "numba")
    with pytest.raises(ImportError):
        importlib.import_module("plume_nav_sim.models.wind.turbulent_wind")


def test_missing_hydra_raises_import_error(monkeypatch):
    _simulate_missing_modules(monkeypatch, "hydra", "omegaconf")
    with pytest.raises(ImportError):
        importlib.import_module("plume_nav_sim.models.wind.turbulent_wind")
