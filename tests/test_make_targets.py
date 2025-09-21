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


def test_setup_invokes_setup_env_script_without_dev_flag():
    output = run_make("setup")
    assert "./setup_env.sh" in output
    assert "--dev" not in output


def test_setup_dev_uses_dev_flag():
    output = run_make("setup-dev")
    assert "./setup_env.sh" in output
    assert "--dev" in output


def test_maintain_uses_update_flag():
    output = run_make("maintain")
    assert "./setup_env.sh" in output
    assert "--update" in output
