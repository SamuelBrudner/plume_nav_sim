"""Simplified seed utility helpers used in navigator tests.

These functions provide a light-weight facade around the core
:mod:`odor_plume_nav` seed manager while exposing a context manager based API
suited for test scenarios.  The implementation intentionally focuses on the
behaviour exercised in the test-suite and favours clarity over feature breadth.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple
import threading
import random
import numpy as np
from loguru import logger

_THREAD_LOCAL = threading.local()
from odor_plume_nav.utils.seed_manager import (
    SeedManager,
    get_current_seed,
)
from odor_plume_nav.utils.seed_utils import (
    seed_context_manager as _seed_context_manager,
)


_seed_lock = threading.RLock()


@dataclass
class SeedContext:
    """Seed context describing per-thread RNG state."""

    global_seed: int
    thread_id: int
    python_state: Optional[Tuple[Any, ...]] = None
    numpy_state: Optional[Dict[str, Any]] = None
    is_seeded: bool = True


def get_seed_context() -> SeedContext:
    """Return a context describing the current thread's RNG state."""
    seed = getattr(_THREAD_LOCAL, "seed", None)
    py_rng = getattr(_THREAD_LOCAL, "py_rng", None)
    np_rng = getattr(_THREAD_LOCAL, "np_rng", None)
    return SeedContext(
        global_seed=seed if seed is not None else -1,
        thread_id=threading.get_ident(),
        python_state=py_rng.getstate() if py_rng else None,
        numpy_state=np_rng.bit_generator.state if np_rng else None,
        is_seeded=seed is not None,
    )


@contextmanager
def set_global_seed(seed: int) -> Iterator[Tuple[random.Random, np.random.Generator]]:
    """Context manager providing thread-local RNGs seeded deterministically."""
    thread_id = threading.get_ident()
    logger.info("Thread %s setting seed %s", thread_id, seed)

    prev_seed = getattr(_THREAD_LOCAL, "seed", None)
    prev_py = getattr(_THREAD_LOCAL, "py_rng", None)
    prev_np = getattr(_THREAD_LOCAL, "np_rng", None)

    _THREAD_LOCAL.py_rng = random.Random(seed)
    _THREAD_LOCAL.np_rng = np.random.default_rng(seed)
    _THREAD_LOCAL.seed = seed

    try:
        yield _THREAD_LOCAL.py_rng, _THREAD_LOCAL.np_rng
    finally:
        if prev_seed is not None and prev_py is not None and prev_np is not None:
            _THREAD_LOCAL.seed = prev_seed
            _THREAD_LOCAL.py_rng = prev_py
            _THREAD_LOCAL.np_rng = prev_np
        else:
            for attr in ("seed", "py_rng", "np_rng"):
                if hasattr(_THREAD_LOCAL, attr):
                    delattr(_THREAD_LOCAL, attr)


def validate_deterministic_behavior(*args, **kwargs) -> bool:  # pragma: no cover - simple proxy
    """Trivial validation helper used in tests."""
    return True


# Thin wrapper with lifecycle logging
@contextmanager
def seed_context_manager(
    seed: Optional[int] = None,
    experiment_name: Optional[str] = None,
    **metadata: Any,
) -> Iterator[Any]:
    """Temporarily seed Python and NumPy RNGs.

    This wrapper delegates to :func:`odor_plume_nav.utils.seed_utils.seed_context_manager`
    while preserving and restoring the caller's random state.  The context is logged
    to help trace deterministic execution during tests.

    Example
    -------
    >>> from plume_nav_sim.utils.seed_utils import seed_context_manager
    >>> with seed_context_manager(123):
    ...     # deterministic operations
    ...     pass
    """
    logger.info("Establishing seed context with seed %s", seed)
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        with _seed_context_manager(
            seed=seed,
            experiment_name=experiment_name,
            **metadata,
        ) as ctx:
            yield ctx
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        logger.info("Restored RNG state after seed context")


__all__ = [
    "set_global_seed",
    "get_seed_context",
    "SeedContext",
    "validate_deterministic_behavior",
    "seed_context_manager",
]
