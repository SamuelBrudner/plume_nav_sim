import subprocess
import pathlib


def test_setup_env_help_shows_dev_flag():
    script = pathlib.Path('setup_env.sh')
    assert script.exists(), 'setup_env.sh script missing'
    result = subprocess.run(['bash', str(script), '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert '--dev' in result.stdout, 'Expected --dev flag documented in help output'
