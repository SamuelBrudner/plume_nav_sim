import pytest

from plume_nav_sim.core.simulation import SimulationContext
from plume_nav_sim.models.plume.gaussian_plume import GaussianPlumeModel
from plume_nav_sim.models.wind.constant_wind import ConstantWindField


def test_add_plume_model_instantiates_registry_class():
    ctx = SimulationContext.create()
    ctx.add_plume_model("gaussian", source_strength=1000.0)
    plume_component = next(iter(ctx.components.values()))
    plume_model = plume_component["instance"]
    assert isinstance(plume_model, GaussianPlumeModel)


def test_add_wind_field_instantiates_registry_class():
    ctx = SimulationContext.create()
    ctx.add_wind_field("constant", velocity=(1.0, 0.0))
    wind_component = next(iter(ctx.components.values()))
    wind_field = wind_component["instance"]
    assert isinstance(wind_field, ConstantWindField)


def test_add_plume_model_unknown_type_raises():
    ctx = SimulationContext.create()
    with pytest.raises(Exception):
        ctx.add_plume_model("unknown_type")
