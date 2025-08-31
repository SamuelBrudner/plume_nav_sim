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


def test_seed_context_manager_restores_state_and_logs():
    """Seed context manager restores RNG state and logs lifecycle events."""
    from loguru import logger

    # Baseline sequence without context manager
    random.seed(7)
    np.random.seed(7)
    random.random()
    np.random.rand()
    expected_random = random.random()
    expected_numpy = np.random.rand()

    # Use context manager and ensure state restoration
    random.seed(7)
    np.random.seed(7)
    random.random()
    np.random.rand()

    messages = []
    sink_id = logger.add(messages.append, format="{message}")
    with seed_context_manager(123):
        random.random()
        np.random.rand()
    logger.remove(sink_id)

    after_random = random.random()
    after_numpy = np.random.rand()

    assert after_random == expected_random
    assert after_numpy == expected_numpy

    log_text = " ".join(messages).lower()
    assert "establishing seed context" in log_text
    assert "restored rng state" in log_text
