import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from omegaconf import DictConfig

from plume_nav_sim.cli.main import cli, set_cli_config


def _basic_cfg():
    return DictConfig({
        'navigator': {'position': [0, 0]},
        'video_plume': {'video_path': 'test.mp4'},
        'simulation': {'num_steps': 10}
    })


def test_run_command_uses_api_functions():
    cfg = _basic_cfg()
    set_cli_config(cfg)
    runner = CliRunner()
    with patch('plume_nav_sim.cli.main._validate_hydra_availability'), \
         patch('plume_nav_sim.cli.main.validate_configuration', return_value=True), \
         patch('plume_nav_sim.cli.main.create_navigator', return_value='nav') as mock_nav, \
         patch('plume_nav_sim.cli.main.create_video_plume', return_value='plume') as mock_plume, \
         patch('plume_nav_sim.cli.main.run_plume_simulation', return_value={}) as mock_run:
        result = runner.invoke(cli, ['run'])
    assert result.exit_code == 0
    mock_nav.assert_called_once_with(cfg.navigator)
    mock_plume.assert_called_once_with(cfg.video_plume)
    mock_run.assert_called_once_with('nav', 'plume', cfg.simulation)


def test_run_command_dry_run_skips_simulation():
    cfg = _basic_cfg()
    set_cli_config(cfg)
    runner = CliRunner()
    with patch('plume_nav_sim.cli.main._validate_hydra_availability'), \
         patch('plume_nav_sim.cli.main.validate_configuration', return_value=True), \
         patch('plume_nav_sim.cli.main.create_navigator', return_value='nav') as mock_nav, \
         patch('plume_nav_sim.cli.main.create_video_plume', return_value='plume') as mock_plume, \
         patch('plume_nav_sim.cli.main.run_plume_simulation') as mock_run:
        result = runner.invoke(cli, ['run', '--dry-run'])
    assert result.exit_code == 0
    mock_nav.assert_called_once_with(cfg.navigator)
    mock_plume.assert_called_once_with(cfg.video_plume)
    mock_run.assert_not_called()
