import pytest

from plume_nav_sim.core.controllers import BaseController


def test_base_controller_rejects_unknown_kwargs():
    with pytest.raises(TypeError) as exc_info:
        BaseController(unexpected=123)
    assert "unexpected" in str(exc_info.value)
