import importlib.util
import sys
import types
from pathlib import Path


def _load_io_module():
    module_name = "plume_nav_sim.utils.io"
    module_path = Path(__file__).resolve().parents[1] / "src/plume_nav_sim/utils/io.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("plume_nav_sim", types.ModuleType("plume_nav_sim"))
    pkg_utils = types.ModuleType("plume_nav_sim.utils")
    pkg_utils.__path__ = []
    sys.modules.setdefault("plume_nav_sim.utils", pkg_utils)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_legacy_io_utils_import_removed():
    _load_io_module()
    assert "plume_nav_sim.io_utils" not in sys.modules
