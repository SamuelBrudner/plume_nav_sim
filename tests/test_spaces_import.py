import importlib.util
import pathlib


def test_spaces_does_not_define_gymnasium_available():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "src" / "odor_plume_nav" / "environments" / "spaces.py"
    spec = importlib.util.spec_from_file_location("spaces", module_path)
    spaces = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spaces)
    assert not hasattr(spaces, "GYMNASIUM_AVAILABLE")
