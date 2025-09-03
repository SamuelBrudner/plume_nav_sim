import builtins
import importlib
import sys
import types

import pytest

CORE_SUBMODULES = [
    "plume_nav_sim.core.sources",
    "plume_nav_sim.core.boundaries",
    "plume_nav_sim.core.actions",
    "plume_nav_sim.core.controllers",
    "plume_nav_sim.core.simulation",
    "plume_nav_sim.recording",
    "plume_nav_sim.analysis",
    "plume_nav_sim.debug",
]


@pytest.mark.parametrize("missing", CORE_SUBMODULES)
def test_core_import_raises_on_missing_submodule(monkeypatch, missing):
    # Provide stub SpacesFactory to allow core import baseline
    dummy_spaces = types.ModuleType("spaces")
    dummy_spaces.SpacesFactory = object
    monkeypatch.setitem(sys.modules, "plume_nav_sim.envs.spaces", dummy_spaces)

    # Stub debug module to avoid importing heavy dependencies
    dummy_debug = types.ModuleType("debug")
    dummy_debug.DebugGUI = object
    dummy_debug.plot_initial_state = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "plume_nav_sim.debug", dummy_debug)

    # Stub recording and analysis modules
    dummy_recording = types.ModuleType("recording")
    dummy_recording.RecorderFactory = object
    dummy_recording.RecorderManager = object
    dummy_recording.BaseRecorder = object
    monkeypatch.setitem(sys.modules, "plume_nav_sim.recording", dummy_recording)

    dummy_analysis = types.ModuleType("analysis")
    dummy_analysis.StatsAggregator = object
    dummy_analysis.generate_summary = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "plume_nav_sim.analysis", dummy_analysis)

    dummy_sensors = types.ModuleType("sensors")
    dummy_sensors.create_sensor_from_config = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "plume_nav_sim.core.sensors", dummy_sensors)

    # Stub utils package to avoid Hydra requirements
    dummy_utils = types.ModuleType("utils")
    dummy_utils.visualization = types.ModuleType("visualization")
    dummy_utils.visualization.SimulationVisualization = object
    dummy_utils.visualization.visualize_trajectory = lambda *a, **k: None
    dummy_utils.logging_setup = types.ModuleType("logging_setup")
    dummy_utils.logging_setup.correlation_context = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "plume_nav_sim.utils", dummy_utils)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.utils.visualization", dummy_utils.visualization)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.utils.logging_setup", dummy_utils.logging_setup)

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == missing or name.startswith(f"{missing}."):
            raise ImportError(f"No module named '{missing}'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("plume_nav_sim.core", None)

    with pytest.raises(ImportError):
        importlib.import_module("plume_nav_sim.core")
