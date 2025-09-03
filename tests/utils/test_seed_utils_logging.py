import importlib.util
import pathlib
import sys
import types

import pytest


def test_importerror_when_loguru_missing(monkeypatch):
    """Seed utils should raise if loguru is unavailable."""
    monkeypatch.setitem(sys.modules, "loguru", None)

    opn_utils = types.ModuleType("odor_plume_nav.utils")
    opn_utils.__path__ = []
    monkeypatch.setitem(sys.modules, "odor_plume_nav", types.ModuleType("odor_plume_nav"))
    monkeypatch.setitem(sys.modules, "odor_plume_nav.utils", opn_utils)

    stub_seed_manager = types.ModuleType("seed_manager")
    stub_seed_manager.SeedManager = object
    stub_seed_manager.get_current_seed = lambda: 0
    monkeypatch.setitem(sys.modules, "odor_plume_nav.utils.seed_manager", stub_seed_manager)

    stub_seed_utils = types.ModuleType("seed_utils")
    def _cm(*args, **kwargs):
        yield
    stub_seed_utils.seed_context_manager = _cm
    monkeypatch.setitem(sys.modules, "odor_plume_nav.utils.seed_utils", stub_seed_utils)

    seed_utils_path = pathlib.Path(__file__).resolve().parents[2] / "src/plume_nav_sim/utils/seed_utils.py"
    spec = importlib.util.spec_from_file_location("plume_nav_sim.utils.seed_utils", seed_utils_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["plume_nav_sim.utils.seed_utils"] = module

    with pytest.raises(ImportError):
        spec.loader.exec_module(module)
