import pytest

from plume_nav_sim.core.simulation import SimulationContext
from plume_nav_sim.core.protocols import (
    PlumeModelProtocol,
    WindFieldProtocol,
    SensorProtocol,
)


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

    def test_add_sensor_instantiates_real_sensor(self):
        ctx = SimulationContext.create()
        ctx.add_sensor("BinarySensor", threshold=0.1)
        component = next(iter(ctx.components.values()))
        sensor = component["instance"]
        assert isinstance(sensor, SensorProtocol)
        assert type(sensor).__name__ == "BinarySensor"
