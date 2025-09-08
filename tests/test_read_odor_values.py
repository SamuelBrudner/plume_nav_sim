import numpy as np
import pytest
import ast
from pathlib import Path

class _DummyLogger:
    def error(self, *args, **kwargs):
        pass

# Extract _read_odor_values function from controllers.py without importing package
source_path = Path(__file__).resolve().parents[1] / "src" / "plume_nav_sim" / "core" / "controllers.py"
source = source_path.read_text()
module = ast.parse(source)
for node in module.body:
    if isinstance(node, ast.FunctionDef) and node.name == "_read_odor_values":
        func_module = ast.Module([node], [])
        ast.fix_missing_locations(func_module)
        namespace: dict = {}
        exec(
            compile(func_module, filename=str(source_path), mode="exec"),
            {"np": np, "logger": _DummyLogger()},
            namespace,
        )
        _read_odor_values = namespace["_read_odor_values"]
        break
else:
    raise RuntimeError("_read_odor_values not found")


def test_invalid_env_array_raises():
    positions = np.array([[0, 0]])
    with pytest.raises(ValueError):
        _read_odor_values(object(), positions)
