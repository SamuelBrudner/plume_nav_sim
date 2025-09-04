import importlib
import sys
import builtins
from typing import Callable
from pathlib import Path

import pytest
from loguru import logger

MODULE_PATH = "plume_nav_sim.examples.agents.reactive_agent"


def _bootstrap_minimal_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Create minimal package structure to import reactive_agent in isolation."""
    import types
    import numpy as np

    # Root package and subpackages
    root = types.ModuleType("plume_nav_sim")
    monkeypatch.setitem(sys.modules, "plume_nav_sim", root)

    examples_pkg = types.ModuleType("plume_nav_sim.examples")
    agents_pkg = types.ModuleType("plume_nav_sim.examples.agents")
    monkeypatch.setitem(sys.modules, "plume_nav_sim.examples", examples_pkg)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.examples.agents", agents_pkg)

    # Protocols
    protocols_pkg = types.ModuleType("plume_nav_sim.protocols")
    navigator_mod = types.ModuleType("plume_nav_sim.protocols.navigator")

    class NavigatorProtocol:
        pass

    navigator_mod.NavigatorProtocol = NavigatorProtocol
    protocols_pkg.navigator = navigator_mod
    monkeypatch.setitem(sys.modules, "plume_nav_sim.protocols", protocols_pkg)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.protocols.navigator", navigator_mod)

    # Core controllers
    core_pkg = types.ModuleType("plume_nav_sim.core")
    controllers_mod = types.ModuleType("plume_nav_sim.core.controllers")

    class SingleAgentController:
        def __init__(
            self,
            position=None,
            orientation=0.0,
            speed=0.0,
            max_speed=1.0,
            angular_velocity=0.0,
            **kwargs,
        ):
            self.positions = np.array([position or (0.0, 0.0)], dtype=float)
            self.orientations = np.array([orientation], dtype=float)
            self.speeds = np.array([speed], dtype=float)
            self.max_speeds = np.array([max_speed], dtype=float)
            self.angular_velocities = np.array([angular_velocity], dtype=float)

        def reset(self, speed=None, angular_velocity=None, **kwargs):
            if speed is not None:
                self.speeds[0] = speed
            if angular_velocity is not None:
                self.angular_velocities[0] = angular_velocity

        def step(self, env_array, dt):
            pass

        def sample_odor(self, env_array):
            return 0.0

    controllers_mod.SingleAgentController = SingleAgentController
    core_pkg.controllers = controllers_mod
    monkeypatch.setitem(sys.modules, "plume_nav_sim.core", core_pkg)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.core.controllers", controllers_mod)

    # Core sensors
    sensors_mod = types.ModuleType("plume_nav_sim.core.sensors")

    class GradientSensor:
        pass

    class SensorProtocol:
        pass

    def create_sensor_from_config(config):
        return GradientSensor()

    def validate_sensor_config(config):
        return None

    sensors_mod.GradientSensor = GradientSensor
    sensors_mod.SensorProtocol = SensorProtocol
    sensors_mod.create_sensor_from_config = create_sensor_from_config
    sensors_mod.validate_sensor_config = validate_sensor_config
    core_pkg.sensors = sensors_mod
    monkeypatch.setitem(sys.modules, "plume_nav_sim.core.sensors", sensors_mod)


def _import_with_missing(monkeypatch: pytest.MonkeyPatch, missing_prefix: str) -> Callable[[], None]:
    """Return a function that imports the reactive agent with a missing dependency."""

    def importer():
        if MODULE_PATH in sys.modules:
            del sys.modules[MODULE_PATH]

        _bootstrap_minimal_packages(monkeypatch)

        original_import = builtins.__import__

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith(missing_prefix):
                raise ImportError(f"mocked missing {missing_prefix}")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        file_path = Path(__file__).resolve().parents[2] / "src/plume_nav_sim/examples/agents/reactive_agent.py"
        spec = importlib.util.spec_from_file_location(MODULE_PATH, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE_PATH] = module
        spec.loader.exec_module(module)  # type: ignore

    return importer


def _load_reactive_agent(monkeypatch: pytest.MonkeyPatch):
    """Load reactive_agent with all dependencies present (stubbed)."""
    if MODULE_PATH in sys.modules:
        del sys.modules[MODULE_PATH]

    _bootstrap_minimal_packages(monkeypatch)

    file_path = Path(__file__).resolve().parents[2] / "src/plume_nav_sim/examples/agents/reactive_agent.py"
    spec = importlib.util.spec_from_file_location(MODULE_PATH, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_PATH] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


def test_import_requires_hydra(monkeypatch):
    importer = _import_with_missing(monkeypatch, "hydra")
    with pytest.raises(ImportError):
        importer()


def test_import_requires_loguru(monkeypatch):
    importer = _import_with_missing(monkeypatch, "loguru")
    with pytest.raises(ImportError):
        importer()


def test_no_fallback_gradient_sensor(monkeypatch):
    _load_reactive_agent(monkeypatch)
    import plume_nav_sim.examples.agents.reactive_agent as reactive_agent

    def mock_create_sensor_from_config(config):
        raise ImportError("mocked sensor creation failure")

    monkeypatch.setattr(
        reactive_agent, "create_sensor_from_config", mock_create_sensor_from_config
    )
    with pytest.raises(ImportError):
        reactive_agent.ReactiveAgent()


def test_initialization_logs_success(monkeypatch):
    _load_reactive_agent(monkeypatch)
    from plume_nav_sim.examples.agents.reactive_agent import ReactiveAgent

    class DummySensor:
        pass

    messages = []
    handler_id = logger.add(lambda msg: messages.append(msg), level="INFO")
    try:
        ReactiveAgent(gradient_sensor=DummySensor())
    finally:
        logger.remove(handler_id)
    assert any("initialized with gradient-following behavior" in m for m in messages)
