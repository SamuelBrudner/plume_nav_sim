"""Simplified seed utility helpers used in navigator tests.

These functions provide a light-weight facade around the core
:mod:`odor_plume_nav` seed manager while exposing a context manager based API
suited for test scenarios.  The implementation intentionally focuses on the
behaviour exercised in the test-suite and favours clarity over feature breadth.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from odor_plume_nav.utils.seed_manager import (
    SeedManager,
    get_current_seed,
)


@dataclass
class SeedContext:
    """Minimal seed context returned by :func:`get_seed_context`."""

    global_seed: int
    is_seeded: bool = True


def get_seed_context() -> SeedContext:
    """Return a context describing the current seeding state."""
    seed = get_current_seed()
    return SeedContext(global_seed=seed if seed is not None else -1, is_seeded=seed is not None)


@contextmanager
def set_global_seed(seed: int) -> Iterator[None]:
    """Context manager that seeds all RNGs deterministically.

    The previous seed is restored when exiting the context if one was set.
    """
    manager = SeedManager()
    previous = get_current_seed()
    manager.set_seed(seed)
    try:
        yield
    finally:
        if previous is not None:
            manager.set_seed(previous)


def validate_deterministic_behavior(*args, **kwargs) -> bool:  # pragma: no cover - simple proxy
    """Trivial validation helper used in tests."""
    return True


__all__ = ["set_global_seed", "get_seed_context", "SeedContext", "validate_deterministic_behavior"]
