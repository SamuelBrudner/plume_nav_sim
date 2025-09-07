from pathlib import Path
import ast


def test_no_native_logging_in_core():
    core_dir = Path(__file__).resolve().parents[2] / "src" / "plume_nav_sim" / "core"
    for file in core_dir.rglob("*.py"):
        text = file.read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name != "logging" for alias in node.names), f"{file} uses native logging"
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "logging", f"{file} uses native logging"
