from pathlib import Path
import importlib.util
import types
import sys
from loguru import logger

import pytest

# Stub external dependencies to avoid heavy imports
hydra_stub = types.ModuleType("hydra")
sys.modules["hydra"] = hydra_stub

omegaconf_stub = types.ModuleType("omegaconf")
class DictConfig(dict):
    pass
omegaconf_stub.DictConfig = DictConfig
sys.modules["omegaconf"] = omegaconf_stub

rich_stub = types.ModuleType("rich")
rich_stub.print = print
sys.modules["rich"] = rich_stub

console_stub = types.ModuleType("rich.console")
class Console:  # minimal stub
    def print(self, *args, **kwargs):
        pass
console_stub.Console = Console
sys.modules["rich.console"] = console_stub

progress_stub = types.ModuleType("rich.progress")
class Progress:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        pass
    def add_task(self, *args, **kwargs):
        return 0
    def update(self, *args, **kwargs):
        pass
progress_stub.Progress = Progress
progress_stub.SpinnerColumn = object
progress_stub.TextColumn = object
progress_stub.BarColumn = object
progress_stub.TimeRemainingColumn = object
sys.modules["rich.progress"] = progress_stub

rich_table = types.ModuleType("rich.table")
class Table:
    def __init__(self, *args, **kwargs):
        pass
    def add_column(self, *args, **kwargs):
        pass
    def add_row(self, *args, **kwargs):
        pass
rich_table.Table = Table
sys.modules["rich.table"] = rich_table

rich_text = types.ModuleType("rich.text")
class Text:
    def __init__(self, *args, **kwargs):
        pass
rich_text.Text = Text
sys.modules["rich.text"] = rich_text

logging_setup_stub = types.ModuleType("plume_nav_sim.utils.logging_setup")
def get_logger(name):
    return logger

def debug_command_timer(*args, **kwargs):
    class Dummy:
        def __enter__(self):
            return {}
        def __exit__(self, exc_type, exc, tb):
            pass
    return Dummy()

def log_debug_command_correlation(*args, **kwargs):
    pass

logging_setup_stub.get_logger = get_logger
logging_setup_stub.debug_command_timer = debug_command_timer
logging_setup_stub.log_debug_command_correlation = log_debug_command_correlation
sys.modules["plume_nav_sim.utils.logging_setup"] = logging_setup_stub

# Stub DebugSession
debug_gui_stub = types.ModuleType("plume_nav_sim.debug.gui")
class DebugSession:
    def __init__(self):
        self.session_id = "dummy"
    def configure(self, **kwargs):
        pass
    def start(self):
        pass
    def get_session_info(self):
        return {}
debug_gui_stub.DebugSession = DebugSession
sys.modules["plume_nav_sim.debug.gui"] = debug_gui_stub

# Load CLI module directly
CLI_PATH = Path(__file__).resolve().parents[2] / "src" / "plume_nav_sim" / "debug" / "cli.py"
spec = importlib.util.spec_from_file_location("debug_cli", CLI_PATH)
debug_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(debug_cli)


def test_launch_viewer_requires_backend(mock_cli_runner):
    """Launching viewer without specifying --backend should error."""
    runner, env = mock_cli_runner
    if runner is None:
        pytest.skip("CliRunner not available")

    with runner.isolated_filesystem():
        Path("results.json").write_text("{}")
        result = runner.invoke(
            debug_cli.debug_group,
            ["launch-viewer", "--results", "results.json"],
            env=env,
        )

    assert result.exit_code != 0
    assert "Missing option '--backend'" in result.output


@pytest.mark.parametrize("backend, hint", [
    ("qt", "pip install pyside6"),
    ("streamlit", "pip install streamlit"),
])
def test_launch_viewer_backend_unavailable(monkeypatch, mock_cli_runner, backend, hint):
    """Explicit backend request should raise when backend unavailable."""
    runner, env = mock_cli_runner
    if runner is None:
        pytest.skip("CliRunner not available")

    monkeypatch.setattr(debug_cli, "detect_available_backends", lambda: set())

    with runner.isolated_filesystem():
        Path("results.json").write_text("{}")
        result = runner.invoke(
            debug_cli.debug_group,
            ["launch-viewer", "--backend", backend, "--results", "results.json"],
            env=env,
        )

    assert result.exit_code != 0
    assert f"Backend '{backend}' is not available" in result.output
    assert hint in result.output
