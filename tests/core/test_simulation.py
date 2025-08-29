import pytest

from plume_nav_sim.core.simulation import SimulationBuilder


def test_execute_simulation_not_implemented():
    builder = SimulationBuilder()
    config = builder.build_config()
    with pytest.raises(NotImplementedError):
        builder._execute_simulation(config)
