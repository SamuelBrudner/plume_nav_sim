import importlib.util
from loguru import logger
import sys
import types
from pathlib import Path


def test_protocols_module_logs_dependencies(caplog, monkeypatch):
    package_path = Path(__file__).resolve().parents[1] / "src" / "plume_nav_sim"

    fake_pkg = types.ModuleType("plume_nav_sim")
    fake_pkg.__path__ = [str(package_path)]
    monkeypatch.setitem(sys.modules, "plume_nav_sim", fake_pkg)

    fake_core = types.ModuleType("plume_nav_sim.core")
    fake_core.__path__ = [str(package_path / "core")]
    monkeypatch.setitem(sys.modules, "plume_nav_sim.core", fake_core)

    schemas = types.ModuleType("plume_nav_sim.config.schemas")
    class NavigatorConfig: ...
    class SingleAgentConfig: ...
    class MultiAgentConfig: ...
    schemas.NavigatorConfig = NavigatorConfig
    schemas.SingleAgentConfig = SingleAgentConfig
    schemas.MultiAgentConfig = MultiAgentConfig
    monkeypatch.setitem(sys.modules, "plume_nav_sim.config.schemas", schemas)

    env_spaces = types.ModuleType("plume_nav_sim.envs.spaces")
    class SpaceFactory: ...
    env_spaces.SpaceFactory = SpaceFactory
    monkeypatch.setitem(sys.modules, "plume_nav_sim.envs.spaces", env_spaces)

    module_path = package_path / "core" / "protocols.py"
    spec = importlib.util.spec_from_file_location("plume_nav_sim.core.protocols", module_path)
    module = importlib.util.module_from_spec(spec)

    with caplog.at_level(logger.INFO):
        spec.loader.exec_module(module)

    assert "NavigatorConfig" in caplog.text
    assert "SpaceFactory" in caplog.text
