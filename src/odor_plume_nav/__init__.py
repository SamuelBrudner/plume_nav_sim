"""Lightweight package initialisation for tests.

The original project initialised a large number of optional submodules on
import which pulled in heavy third‑party dependencies such as OpenCV.  The
minimal stub below keeps the package importable in constrained test
environments while still exposing the version identifier.
"""

__all__ = ["__version__"]
__version__ = "0.2.0"
