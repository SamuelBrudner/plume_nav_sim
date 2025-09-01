import logging

import pytest

from plume_nav_sim.core.simulation import PerformanceMonitor


def test_record_step_time_and_get_summary(caplog):
    monitor = PerformanceMonitor()
    with caplog.at_level(logging.INFO):
        monitor.record_step_time(0.005)
    assert "Recorded step duration" in caplog.text
    summary = monitor.get_summary()
    assert summary["avg_step_time_ms"] == pytest.approx(5.0)
    assert summary["max_step_time_ms"] == pytest.approx(5.0)
    assert summary["total_steps"] == 1
    assert summary["performance_target_met"]


def test_record_step_delegates_to_record_step_time(caplog):
    monitor = PerformanceMonitor()
    with caplog.at_level(logging.INFO):
        monitor.record_step(5.0)
    assert "Recorded step duration" in caplog.text
    summary = monitor.get_summary()
    assert summary["avg_step_time_ms"] == pytest.approx(5.0)
    assert summary["total_steps"] == 1


def test_record_step_time_invalid_duration():
    monitor = PerformanceMonitor()
    with pytest.raises(ValueError):
        monitor.record_step_time(-0.001)
