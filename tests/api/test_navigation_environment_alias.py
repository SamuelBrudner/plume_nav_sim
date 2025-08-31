from unittest.mock import patch
import logging
from plume_nav_sim.api.navigation import create_gymnasium_environment


def test_create_gymnasium_environment_normalizes_id(caplog):
    with patch('gymnasium.make') as mock_make:
        with caplog.at_level(logging.INFO):
            create_gymnasium_environment(environment_id='PlumeNavSim_v0')
        mock_make.assert_called_once()
        called_id = mock_make.call_args[0][0]
        assert called_id == 'plume_nav_sim_v0'
    assert 'environment alias' in caplog.text
