import logging
import pytest
from plume_nav_sim.core.simulation import SimulationContext, PerformanceMonitor


def test_run_simulation_sets_results(caplog):
    ctx = SimulationContext.create()
    ctx.config.performance_monitoring = False
    with ctx:
        ctx.performance_monitor = PerformanceMonitor()
        with caplog.at_level(logging.INFO):
            results = ctx.run_simulation(num_steps=5)
    assert results.step_count == 5
    assert results.success is True
    if ctx.performance_monitor is not None:
        assert results.performance_metrics == ctx.performance_monitor.get_summary()
