import importlib
import sys
import types

import pytest

MODULE_PATH = "plume_nav_sim.examples.agents.infotaxis_agent"


def _setup_minimal_modules(include_sensors: bool) -> None:
    """Register minimal stub modules required to import infotaxis_agent."""
    # Base package hierarchy
    plume_pkg = types.ModuleType("plume_nav_sim")
    plume_pkg.__path__ = ["src/plume_nav_sim"]
    sys.modules.setdefault("plume_nav_sim", plume_pkg)

    examples_pkg = types.ModuleType("plume_nav_sim.examples")
    examples_pkg.__path__ = ["src/plume_nav_sim/examples"]
    sys.modules.setdefault("plume_nav_sim.examples", examples_pkg)

    agents_pkg = types.ModuleType("plume_nav_sim.examples.agents")
    agents_pkg.__path__ = ["src/plume_nav_sim/examples/agents"]
    sys.modules.setdefault("plume_nav_sim.examples.agents", agents_pkg)

    core_pkg = types.ModuleType("plume_nav_sim.core")
    core_pkg.__path__ = []
    sys.modules.setdefault("plume_nav_sim.core", core_pkg)

    protocols_pkg = types.ModuleType("plume_nav_sim.protocols")
    protocols_pkg.__path__ = []
    sys.modules.setdefault("plume_nav_sim.protocols", protocols_pkg)

    # NavigatorProtocol stub
    navigator_mod = types.ModuleType("plume_nav_sim.protocols.navigator")
    class NavigatorProtocol:  # pragma: no cover - simple stub
        pass
    navigator_mod.NavigatorProtocol = NavigatorProtocol
    sys.modules.setdefault("plume_nav_sim.protocols.navigator", navigator_mod)

    # Core protocols stub
    core_protocols = types.ModuleType("plume_nav_sim.core.protocols")
    core_protocols.PositionType = object
    core_protocols.ConfigType = dict
    sys.modules.setdefault("plume_nav_sim.core.protocols", core_protocols)

    # Controllers stub
    controllers_mod = types.ModuleType("plume_nav_sim.core.controllers")
    class SingleAgentController:  # pragma: no cover - simple stub
        pass
    controllers_mod.SingleAgentController = SingleAgentController
    sys.modules.setdefault("plume_nav_sim.core.controllers", controllers_mod)

    if include_sensors:
        sensors_mod = types.ModuleType("plume_nav_sim.core.sensors")

        class _ConcentrationSensor:  # pragma: no cover - simple stub
            def measure(self, *args, **kwargs):
                return [0.0]

        class _BinarySensor:  # pragma: no cover - simple stub
            def detect(self, *args, **kwargs):
                return [False]

        def create_sensor_from_config(config):  # pragma: no cover - simple stub
            return _ConcentrationSensor()

        sensors_mod.ConcentrationSensor = _ConcentrationSensor
        sensors_mod.BinarySensor = _BinarySensor
        sensors_mod.create_sensor_from_config = create_sensor_from_config
        sys.modules.setdefault("plume_nav_sim.core.sensors", sensors_mod)
    else:
        sys.modules.pop("plume_nav_sim.core.sensors", None)


def test_missing_sensor_dependency_raises():
    _setup_minimal_modules(include_sensors=False)
    with pytest.raises(ImportError):
        importlib.import_module(MODULE_PATH)


def test_sensor_initialization_logs(monkeypatch):
    _setup_minimal_modules(include_sensors=True)
    agent_module = importlib.import_module(MODULE_PATH)

    class LoggerStub:
        def __init__(self):
            self.messages = []

        def bind(self, **kwargs):  # pragma: no cover - simple stub
            return self

        def info(self, message, *args, **kwargs):  # pragma: no cover - simple stub
            self.messages.append(message)

    logger_stub = LoggerStub()
    monkeypatch.setattr(agent_module, "logger", logger_stub)

    agent_module.InfotaxisAgent(environment_bounds=(10.0, 10.0))
    assert any("Initialized" in msg for msg in logger_stub.messages)
