import pytest

from plume_nav_sim.core.simulation import SimulationContext
from plume_nav_sim.protocols.plume_model import PlumeModelProtocol
from plume_nav_sim.protocols.wind_field import WindFieldProtocol


class TestSimulationContextFactory:
    def test_add_plume_model_instantiates_real_model(self):
        ctx = SimulationContext.create()
        ctx.add_plume_model("GaussianPlumeModel")
        component = next(iter(ctx.components.values()))
        plume = component["instance"]
        assert isinstance(plume, PlumeModelProtocol)
        assert type(plume).__name__ == "GaussianPlumeModel"

    def test_add_wind_field_invalid_type_raises(self):
        ctx = SimulationContext.create()
        with pytest.raises(Exception):
            ctx.add_wind_field("NonexistentWindField")
