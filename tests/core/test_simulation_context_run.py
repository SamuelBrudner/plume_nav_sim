from loguru import logger
import pytest
from plume_nav_sim.core.simulation import SimulationContext, PerformanceMonitor


def test_run_simulation_sets_results(caplog):
    ctx = SimulationContext.create()
    ctx.config.performance_monitoring = False
    with ctx:
        ctx.performance_monitor = PerformanceMonitor()
        with caplog.at_level(logger.INFO):
            results = ctx.run_simulation(num_steps=5)
    assert results.step_count == 5
    assert results.success is True
    if ctx.performance_monitor is not None:
        assert results.performance_metrics == ctx.performance_monitor.get_summary()

def test_run_simulation_logs_summary(caplog):
    ctx = SimulationContext.create()
    ctx.performance_monitor = PerformanceMonitor()
    with ctx:
        with caplog.at_level(logger.INFO):
            ctx.run_simulation(num_steps=3)
    assert any(
        "Simulation completed summary" in record.message for record in caplog.records
    )
