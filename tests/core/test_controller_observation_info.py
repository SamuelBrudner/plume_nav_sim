import sys
import types
import pytest

def _stub_gui_deps():
    PySide6 = types.ModuleType("PySide6")
    sys.modules.setdefault("PySide6", PySide6)
    sys.modules.setdefault("PySide6.QtWidgets", types.ModuleType("PySide6.QtWidgets"))
    sys.modules.setdefault("PySide6.QtCore", types.ModuleType("PySide6.QtCore"))
    sys.modules.setdefault("PySide6.QtGui", types.ModuleType("PySide6.QtGui"))


def test_missing_num_agents_raises():
    _stub_gui_deps()
    from plume_nav_sim.core.controllers import BaseController
    controller = BaseController()
    with pytest.raises(AttributeError):
        controller.get_observation_space_info()


def test_observation_info_present():
    _stub_gui_deps()
    from plume_nav_sim.core.controllers import BaseController, DirectOdorSensor

    class ConcreteController(BaseController):
        def __init__(self):
            super().__init__(sensors=[DirectOdorSensor()])
            self.num_agents = 1

    controller = ConcreteController()
    info = controller.get_observation_space_info()
    assert info["num_agents"] == 1
    assert info["sensors"] == ["DirectOdorSensor"]
