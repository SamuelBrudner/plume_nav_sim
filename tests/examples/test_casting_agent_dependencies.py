import builtins
import importlib.util
import pathlib
import sys
import types

import pytest


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "plume_nav_sim"
    / "examples"
    / "agents"
    / "casting_agent.py"
)


def _setup_stubs(monkeypatch, exclude=None):
    exclude = set() if exclude is None else set(exclude)
    modules = {}
    pkg_plume = types.ModuleType("plume_nav_sim"); pkg_plume.__path__ = []
    modules["plume_nav_sim"] = pkg_plume
    pkg_examples = types.ModuleType("plume_nav_sim.examples"); pkg_examples.__path__ = []
    modules["plume_nav_sim.examples"] = pkg_examples
    pkg_agents = types.ModuleType("plume_nav_sim.examples.agents"); pkg_agents.__path__ = []
    modules["plume_nav_sim.examples.agents"] = pkg_agents
    pkg_protocols = types.ModuleType("plume_nav_sim.protocols"); pkg_protocols.__path__ = []
    modules["plume_nav_sim.protocols"] = pkg_protocols
    pkg_protocols_nav = types.ModuleType("plume_nav_sim.protocols.navigator"); pkg_protocols_nav.__path__ = []
    class NavigatorProtocol: ...
    pkg_protocols_nav.NavigatorProtocol = NavigatorProtocol
    modules["plume_nav_sim.protocols.navigator"] = pkg_protocols_nav
    pkg_core = types.ModuleType("plume_nav_sim.core"); pkg_core.__path__ = []
    modules["plume_nav_sim.core"] = pkg_core
    pkg_core_protocols = types.ModuleType("plume_nav_sim.core.protocols"); pkg_core_protocols.__path__ = []
    class SensorProtocol: ...
    pkg_core_protocols.SensorProtocol = SensorProtocol
    modules["plume_nav_sim.core.protocols"] = pkg_core_protocols
    pkg_core_controllers = types.ModuleType("plume_nav_sim.core.controllers"); pkg_core_controllers.__path__ = []
    class SingleAgentController: ...
    pkg_core_controllers.SingleAgentController = SingleAgentController
    modules["plume_nav_sim.core.controllers"] = pkg_core_controllers
    pkg_config = types.ModuleType("plume_nav_sim.config"); pkg_config.__path__ = []
    modules["plume_nav_sim.config"] = pkg_config
    if "plume_nav_sim.config.schemas" not in exclude:
        pkg_schemas = types.ModuleType("plume_nav_sim.config.schemas"); pkg_schemas.__path__ = []
        class NavigatorConfig: ...
        pkg_schemas.NavigatorConfig = NavigatorConfig
        modules["plume_nav_sim.config.schemas"] = pkg_schemas
    else:
        monkeypatch.delitem(sys.modules, "plume_nav_sim.config.schemas", raising=False)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _load_casting_agent(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "plume_nav_sim.examples.agents.casting_agent", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)


def test_casting_agent_requires_loguru(monkeypatch):
    _setup_stubs(monkeypatch)
    real_import = builtins.__import__
    monkeypatch.delitem(sys.modules, "loguru", raising=False)

    def fake_import(name, *args, **kwargs):
        if name == "loguru":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="loguru.*CastingAgent|CastingAgent.*loguru"):
        _load_casting_agent(monkeypatch)
    sys.modules.pop("loguru", None)


def test_casting_agent_requires_hydra(monkeypatch):
    _setup_stubs(monkeypatch)
    real_import = builtins.__import__
    monkeypatch.delitem(sys.modules, "omegaconf", raising=False)
    monkeypatch.delitem(sys.modules, "hydra.core.config_store", raising=False)

    def fake_import(name, *args, **kwargs):
        if name in {"omegaconf", "hydra.core.config_store"}:
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="CastingAgent.*Hydra"):
        _load_casting_agent(monkeypatch)
    sys.modules.pop("omegaconf", None)
    sys.modules.pop("hydra.core.config_store", None)


def test_casting_agent_requires_config_schema(monkeypatch):
    _setup_stubs(monkeypatch, exclude={"plume_nav_sim.config.schemas"})
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "plume_nav_sim.config.schemas":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="CastingAgent.*NavigatorConfig"):
        _load_casting_agent(monkeypatch)
    sys.modules.pop("plume_nav_sim.config.schemas", None)
