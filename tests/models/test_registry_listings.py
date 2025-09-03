import importlib.util
import sys
import types
from pathlib import Path


def load_models_module():
    pkg_name = "plume_nav_sim"
    models_name = f"{pkg_name}.models"

    # Remove existing modules
    for name in list(sys.modules):
        if name.startswith(pkg_name):
            del sys.modules[name]

    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "src/plume_nav_sim")]
    sys.modules[pkg_name] = pkg

    # Stub protocols module to satisfy imports
    core = types.ModuleType(f"{pkg_name}.core")
    protocols = types.ModuleType(f"{pkg_name}.core.protocols")
    class _P:
        pass
    protocols.PlumeModelProtocol = _P
    protocols.WindFieldProtocol = _P
    protocols.SensorProtocol = _P
    protocols.ComponentConfigType = dict
    sys.modules[f"{pkg_name}.core"] = core
    sys.modules[f"{pkg_name}.core.protocols"] = protocols

    path = Path(__file__).resolve().parents[2] / "src/plume_nav_sim/models/__init__.py"
    spec = importlib.util.spec_from_file_location(models_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[models_name] = module
    return module


def test_listing_functions_only_show_registered_components():
    mod = load_models_module()

    assert mod.list_available_plume_models() == []
    assert mod.list_available_wind_fields() == []
    assert mod.list_available_sensors() == []

    class DummyPlume:
        pass

    class DummyWind:
        pass

    class DummySensor:
        pass

    mod.register_plume_model('DummyPlume', DummyPlume, 'dummy plume')
    mod.register_wind_field('DummyWind', DummyWind, 'dummy wind')
    mod.register_sensor('DummySensor', DummySensor, 'dummy sensor')

    assert mod.list_available_plume_models() == ['DummyPlume']
    assert mod.list_available_wind_fields() == ['DummyWind']
    assert mod.list_available_sensors() == ['DummySensor']
