from unittest.mock import Mock, patch

from plume_nav_sim.core.simulation import run_simulation as run_plume_simulation
from odor_plume_nav.core.simulation import run_simulation as run_odor_simulation
import numpy as np


class DummySpace:
    def sample(self):
        return 0


class DummyEnv:
    action_space = DummySpace()
    observation_space = None

    def reset(self, seed=None, options=None):
        return 0, {}

    def step(self, action):
        return 0, 0.0, False, False, {}

    def close(self):
        pass


def test_run_simulation_records_step_time_plume_nav_sim():
    env = DummyEnv()
    with patch('plume_nav_sim.core.simulation.PerformanceMonitor') as MockMonitor:
        mock_monitor = Mock()
        mock_monitor.record_step_time = Mock()
        MockMonitor.return_value = mock_monitor
        run_plume_simulation(env, num_steps=1)
        mock_monitor.record_step_time.assert_called()


def test_run_simulation_records_step_time_odor_plume_nav():
    class DummyNavigator:
        num_agents = 1

        def __init__(self):
            self.positions = np.zeros((1, 2))
            self.orientations = np.zeros(1)

        def step(self, frame, dt: float):
            pass

        def sample_odor(self, frame):
            return 0.0

    class DummyVideoPlume:
        frame_count = 2

        def get_frame(self, idx):
            return None

    navigator = DummyNavigator()
    video_plume = DummyVideoPlume()
    with patch('odor_plume_nav.core.simulation.PerformanceMonitor') as MockMonitor:
        mock_monitor = Mock()
        mock_monitor.record_step_time = Mock()
        MockMonitor.return_value = mock_monitor
        run_odor_simulation(
            navigator,
            video_plume,
            num_steps=1,
            record_trajectories=False,
        )
        mock_monitor.record_step_time.assert_called()
