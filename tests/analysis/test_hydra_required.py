import importlib.util
import pathlib
import sys
import types
import pytest


def test_hydra_required():
    package_root = pathlib.Path(__file__).resolve().parents[2] / "src/plume_nav_sim"
    pkg = types.ModuleType("plume_nav_sim")
    pkg.__path__ = [str(package_root)]
    sys.modules.setdefault("plume_nav_sim", pkg)

    # stub dependencies required by analysis module
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

    spec = importlib.util.spec_from_file_location(
        "plume_nav_sim.analysis", package_root / "analysis" / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(ImportError) as excinfo:
        spec.loader.exec_module(module)
    assert "hydra" in str(excinfo.value).lower()
