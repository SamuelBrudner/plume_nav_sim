import subprocess
import pathlib
import os


def test_setup_env_dev_mode():
    script = pathlib.Path(__file__).resolve().parent.parent / "setup_env.sh"
    env = {**os.environ, "PLUMENAV_SKIP_INSTALL": "1"}
    result = subprocess.run(["bash", str(script), "--dev"], capture_output=True, text=True, check=True, env=env)
    assert "Installing development dependencies" in result.stdout
