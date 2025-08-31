import numpy as np
from plume_nav_sim.core.controllers import MultiAgentController


def test_speeds_assignment_creates_copy():
    positions = np.zeros((2, 2))
    controller = MultiAgentController(
        positions=positions,
        orientations=np.zeros(2),
        speeds=np.zeros(2),
        max_speeds=np.full(2, 10.0),
    )
    new_speeds = np.array([1.0, 2.0])
    controller.speeds = new_speeds
    new_speeds[0] = 5.0
    assert controller.speeds[0] == 1.0
