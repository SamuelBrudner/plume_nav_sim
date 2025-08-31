import random
import numpy as np
from odor_plume_nav.utils.seed_manager import SeedManager

from plume_nav_sim.utils.seed_utils import seed_context_manager


def test_seed_context_manager_is_deterministic():
    """Ensure seed_context_manager applies deterministic seeding for RNGs."""
    manager = SeedManager()
    manager.set_seed(999)
    with seed_context_manager(123):
        inside_random1 = random.random()
        inside_numpy1 = np.random.rand()

    manager.set_seed(999)
    with seed_context_manager(123):
        inside_random2 = random.random()
        inside_numpy2 = np.random.rand()

    assert inside_random1 == inside_random2
    assert inside_numpy1 == inside_numpy2
