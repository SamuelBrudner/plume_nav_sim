import importlib
import sys
import types
from pathlib import Path

import pytest

PACKAGE_PATH = Path("src/plume_nav_sim")


def _import_seed_manager(monkeypatch):
    package = types.ModuleType("plume_nav_sim")
    package.__path__ = [str(PACKAGE_PATH)]
    monkeypatch.setitem(sys.modules, "plume_nav_sim", package)

    utils_pkg = types.ModuleType("plume_nav_sim.utils")
    utils_pkg.__path__ = [str(PACKAGE_PATH / "utils")]
    monkeypatch.setitem(sys.modules, "plume_nav_sim.utils", utils_pkg)

    sys.modules.pop("plume_nav_sim.utils.seed_manager", None)
    return importlib.import_module("plume_nav_sim.utils.seed_manager")


def test_import_fails_without_logging_setup(monkeypatch):
    monkeypatch.setitem(sys.modules, "plume_nav_sim.utils.logging_setup", None)
    with pytest.raises(ImportError):
        _import_seed_manager(monkeypatch)


def test_import_fails_without_hydra(monkeypatch):
    for name in [
        "omegaconf",
        "hydra",
        "hydra.core",
        "hydra.core.config_store",
    ]:
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError):
        _import_seed_manager(monkeypatch)
