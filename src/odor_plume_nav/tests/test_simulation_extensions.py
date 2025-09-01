import numpy as np
import pytest

from odor_plume_nav.core.controllers import SingleAgentController
from odor_plume_nav.core import run_simulation, simulation
from odor_plume_nav.core.simulation import PerformanceMonitor


def test_performance_monitor_new_methods_and_aliases():
    monitor = PerformanceMonitor(target_fps=10.0)
    monitor.record_step_time(0.01)
    monitor.record_step(0.02)
    summary = monitor.get_summary()
    assert summary["total_steps"] == 2
    metrics = monitor.get_metrics()
    assert metrics["total_steps"] == 2


def test_run_simulation_populates_results_and_uses_summary(monkeypatch):
    class DummyMonitor:
        instances = []

        def __init__(self, *args, **kwargs):
            self.total_steps = 0
            self.summary_called = False
            DummyMonitor.instances.append(self)

        def record_step_time(self, step_duration: float) -> None:
            self.total_steps += 1

        def record_step(self, step_duration: float) -> None:  # legacy alias should not be used
            raise AssertionError("record_step should not be used")

        def get_summary(self):
            self.summary_called = True
            return {"total_steps": self.total_steps, "sentinel": True}

        def get_metrics(self):
            raise AssertionError("legacy get_metrics should not be called")

    monkeypatch.setattr(simulation, "PerformanceMonitor", DummyMonitor)

    navigator = SingleAgentController()

    class DummyVideoPlume:
        frame_count = 10
        width = 10
        height = 10

        def get_frame(self, idx):
            return np.zeros((10, 10), dtype=np.float32)

    results = run_simulation(navigator, DummyVideoPlume(), num_steps=3, dt=1.0)

    assert results.step_count == 3
    assert results.success is True
    monitor = DummyMonitor.instances[0]
    assert monitor.summary_called is True
    assert results.performance_metrics["sentinel"] is True
