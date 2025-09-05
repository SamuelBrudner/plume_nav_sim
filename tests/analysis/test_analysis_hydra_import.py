import importlib.util
import builtins
import pathlib
import sys
import types
import pytest


package_root = pathlib.Path(__file__).resolve().parents[2] / "src/plume_nav_sim"
module_path = package_root / "analysis" / "__init__.py"


def test_import_requires_hydra(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith('hydra') or name.startswith('omegaconf'):
            raise ImportError('hydra missing')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)

    pkg = types.ModuleType("plume_nav_sim")
    pkg.__path__ = [str(package_root)]
    sys.modules.setdefault("plume_nav_sim", pkg)

    sys.modules.setdefault("plume_nav_sim.recording", types.ModuleType("plume_nav_sim.recording")).RecorderFactory = object
    core_mod = types.ModuleType("plume_nav_sim.core")
    protocols_mod = types.ModuleType("plume_nav_sim.core.protocols")
    protocols_mod.StatsAggregatorProtocol = object
    core_mod.protocols = protocols_mod
    sys.modules.setdefault("plume_nav_sim.core", core_mod)
    sys.modules.setdefault("plume_nav_sim.core.protocols", protocols_mod)

    stats_mod = types.ModuleType("plume_nav_sim.analysis.stats")
    stats_mod.StatsAggregator = object
    stats_mod.StatsAggregatorConfig = object
    stats_mod.calculate_basic_stats = object
    stats_mod.calculate_advanced_stats = object
    stats_mod.create_stats_aggregator = object
    stats_mod.generate_summary_report = object
    sys.modules.setdefault("plume_nav_sim.analysis.stats", stats_mod)

    spec = importlib.util.spec_from_file_location("plume_nav_sim.analysis", module_path)
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(ImportError):
        spec.loader.exec_module(module)


def test_hydra_flag_removed():
    source = module_path.read_text()
    assert 'HYDRA_AVAILABLE' not in source
