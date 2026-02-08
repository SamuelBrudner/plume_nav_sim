import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_make(target: str) -> str:
    result = subprocess.run(
        ["make", "-n", target],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"make -n {target} failed with code {result.returncode}:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return result.stdout


def test_help_is_default_target():
    output = run_make("help")
    assert "awk" in output or "Available" in output


def test_install_runs_pip():
    output = run_make("install")
    assert "pip install" in output


def test_test_runs_pytest():
    output = run_make("test")
    assert "pytest" in output


def test_test_debugger_runs_pytest():
    output = run_make("test-debugger")
    assert "pytest" in output


def test_lint_runs_ruff():
    output = run_make("lint")
    assert "ruff" in output


def test_clean_removes_artifacts():
    output = run_make("clean")
    assert "rm" in output
