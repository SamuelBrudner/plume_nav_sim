import logging
from unittest.mock import Mock
import pytest

from plume_nav_sim.core import simulation

run_simulation = simulation.run_simulation
PerformanceMonitor = simulation.PerformanceMonitor


class DummyEnv:
    def __init__(self):
        self.action_space = Mock()
        self.action_space.sample.return_value = 0
        self.observation_space = Mock()

    def reset(self, *, seed=None, options=None):
        return 0, {}

    def step(self, action):
        return 0, 0.0, False, False, {}

    def close(self):
        pass


def test_run_simulation_sets_results_fields_and_logs(monkeypatch, caplog):
    sentinel = {"metric": "value"}

    class DummyMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def record_step(self, *args, **kwargs):
            pass

        def get_metrics(self):
            return {"old": True}

        def get_summary(self):
            return sentinel

    monkeypatch.setattr(simulation, "PerformanceMonitor", DummyMonitor)

    env = DummyEnv()
    results = run_simulation(env, num_steps=5)

    assert results.step_count == 5
    assert results.success is True
    assert results.performance_metrics == sentinel
