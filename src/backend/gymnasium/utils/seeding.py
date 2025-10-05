"""Minimal Gymnasium-compatible seeding utilities used by the repo's seeding helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def np_random(seed: Optional[int] = None) -> Tuple[np.random.Generator, int]:
    """
    Return a NumPy Generator and the integer seed used.

    If seed is None, a seed is generated from system entropy.
    """
    if seed is None:
        # Generate a 32-bit seed from a SeedSequence
        ss = np.random.SeedSequence()
        # Combine the first 32 bits of the mutated state as an integer seed
        seed = int(ss.generate_state(1, dtype=np.uint32)[0])
    else:
        seed = int(seed)

    rng = np.random.default_rng(seed)
    return rng, seed
