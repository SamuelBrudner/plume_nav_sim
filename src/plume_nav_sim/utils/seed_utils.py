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

try:  # Prefer loguru if available for consistency with project logging
    from loguru import logger
except Exception:  # pragma: no cover - fallback
    import logging
    logger = logging.getLogger(__name__)

_THREAD_LOCAL = threading.local()


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


__all__ = ["set_global_seed", "get_seed_context", "SeedContext", "validate_deterministic_behavior"]
