import logging
import pytest
from plume_nav_sim.models.plume.gaussian_plume import GaussianPlumeModel, create_gaussian_plume_model
from plume_nav_sim.models.plume import create_plume_model, AVAILABLE_PLUME_MODELS
from plume_nav_sim.protocols.plume_model import PlumeModelProtocol


def test_gaussian_plume_requires_contract_methods():
    model = GaussianPlumeModel()
    with pytest.raises(TypeError):
        model.step()
    with pytest.raises(TypeError):
        model.reset("extra")
    with pytest.raises(TypeError):
        model.reset(source_position=(0.0, 0.0))


def test_factory_logs_protocol_compliance(caplog):
    with caplog.at_level(logging.DEBUG):
        model = create_gaussian_plume_model({})
    assert isinstance(model, PlumeModelProtocol)
    assert "GaussianPlumeModel complies with PlumeModelProtocol" in caplog.text


class _IncompletePlume:
    def concentration_at(self, positions):
        return [0.0]
    def step(self, dt: float) -> None:
        pass
    # reset missing


def test_create_plume_model_validates_protocol(monkeypatch, caplog):
    dummy_info = {
        'class': _IncompletePlume,
        'config_class': None,
        'factory_function': None,
        'description': '',
        'features': [],
        'performance': {},
        'use_cases': [],
        'available': True,
    }
    monkeypatch.setitem(AVAILABLE_PLUME_MODELS, 'IncompletePlume', dummy_info)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(RuntimeError):
            create_plume_model({'type': 'IncompletePlume'})
    assert "IncompletePlume does not implement PlumeModelProtocol" in caplog.text
    del AVAILABLE_PLUME_MODELS['IncompletePlume']
