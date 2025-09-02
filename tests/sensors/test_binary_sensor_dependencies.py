import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "plume_nav_sim" / "core" / "sensors" / "binary_sensor.py"


def run_missing_dependency_script(package: str) -> int:
    script = textwrap.dedent(f"""
    import builtins, sys, types, importlib.util

    # stub minimal package structure
    navigator = types.ModuleType('plume_nav_sim.protocols.navigator')
    class NavigatorProtocol: ...
    navigator.NavigatorProtocol = NavigatorProtocol
    sensor = types.ModuleType('plume_nav_sim.protocols.sensor')
    class SensorProtocol: ...
    sensor.SensorProtocol = SensorProtocol
    protocols = types.ModuleType('plume_nav_sim.protocols')
    protocols.navigator = navigator
    protocols.sensor = sensor
    sys.modules['plume_nav_sim'] = types.ModuleType('plume_nav_sim')
    sys.modules['plume_nav_sim.protocols'] = protocols
    sys.modules['plume_nav_sim.protocols.navigator'] = navigator
    sys.modules['plume_nav_sim.protocols.sensor'] = sensor

    real_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name.startswith('{package}'):
            raise ImportError('No module named {package}')
        return real_import(name, *args, **kwargs)
    builtins.__import__ = mock_import

    spec = importlib.util.spec_from_file_location('binary_sensor', r"{MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules['binary_sensor'] = module
    try:
        spec.loader.exec_module(module)
    except ImportError:
        sys.exit(0)
    else:
        sys.exit(1)
    """)
    proc = subprocess.run([sys.executable, "-c", script])
    return proc.returncode


@pytest.mark.parametrize("missing", ["hydra", "loguru", "psutil"])
def test_import_error_when_dependency_missing(missing):
    assert run_missing_dependency_script(missing) == 0
