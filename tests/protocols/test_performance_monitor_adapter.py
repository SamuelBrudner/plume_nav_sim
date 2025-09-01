import logging

import pytest

from plume_nav_sim.core.simulation import PerformanceMonitor


def test_record_step_logs_and_exports(caplog):
    monitor = PerformanceMonitor()
    with caplog.at_level(logging.INFO):
        monitor.record_step(5.0, label="step")
    assert "step" in caplog.text
    exported = monitor.export()
    assert exported["step"] == pytest.approx(5.0)


def test_record_step_time_and_get_summary(caplog):
    monitor = PerformanceMonitor()
    with caplog.at_level(logging.INFO):
        monitor.record_step_time(0.005)
    assert "step" in caplog.text
    summary = monitor.get_summary()
    assert summary["step"] == pytest.approx(0.005)


def test_record_step_invalid_duration():
    monitor = PerformanceMonitor()
    with pytest.raises(ValueError):
        monitor.record_step(-1.0, label="bad")
