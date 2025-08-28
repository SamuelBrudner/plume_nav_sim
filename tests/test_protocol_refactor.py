import numpy as np
import pytest

from plume_nav_sim.api.navigation import from_legacy
from plume_nav_sim.core import protocols as core_protocols
from plume_nav_sim.protocols import navigator as navigator_mod


def test_core_protocols_navigator_protocol_is_global():
    assert core_protocols.NavigatorProtocol is navigator_mod.NavigatorProtocol


class MissingStepNavigator:
    positions = np.zeros((1, 2))
    orientations = np.zeros(1)

    def reset(self):
        pass


class DummyPlume:
    pass


def test_legacy_adapter_missing_step(caplog):
    nav = MissingStepNavigator()
    plume = DummyPlume()
    adapter = from_legacy(nav, plume, max_episode_steps=10, render_mode=None)

    with caplog.at_level("DEBUG"):
        with pytest.raises(NotImplementedError):
            adapter.step(np.zeros(2))

    assert "missing step method" in caplog.text.lower()
