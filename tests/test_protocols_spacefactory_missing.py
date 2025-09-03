import importlib.util
import sys
import types
from pathlib import Path
import pytest


def test_protocols_handle_missing_spacefactory(monkeypatch):
    package_path = Path(__file__).resolve().parents[1] / "src" / "plume_nav_sim"

    fake_pkg = types.ModuleType("plume_nav_sim")
    fake_pkg.__path__ = [str(package_path)]
    monkeypatch.setitem(sys.modules, "plume_nav_sim", fake_pkg)

    fake_core = types.ModuleType("plume_nav_sim.core")
    fake_core.__path__ = [str(package_path / "core")]
    monkeypatch.setitem(sys.modules, "plume_nav_sim.core", fake_core)

    fake_envs = types.ModuleType("plume_nav_sim.envs")
    fake_envs.__path__ = []  # empty path to simulate package without modules
    monkeypatch.setitem(sys.modules, "plume_nav_sim.envs", fake_envs)

    schemas = types.ModuleType("plume_nav_sim.config.schemas")
    class NavigatorConfig: ...
    class SingleAgentConfig: ...
    class MultiAgentConfig: ...
    schemas.NavigatorConfig = NavigatorConfig
    schemas.SingleAgentConfig = SingleAgentConfig
    schemas.MultiAgentConfig = MultiAgentConfig
    monkeypatch.setitem(sys.modules, "plume_nav_sim.config.schemas", schemas)

    module_path = package_path / "core" / "protocols.py"
    spec = importlib.util.spec_from_file_location("plume_nav_sim.core.protocols", module_path)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class DummyNavigator:
        num_agents = 1

    with pytest.raises(ImportError):
        module.NavigatorFactory.create_observation_space(DummyNavigator())

    with pytest.raises(ImportError):
        module.NavigatorFactory.create_action_space(DummyNavigator())
