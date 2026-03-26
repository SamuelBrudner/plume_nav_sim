"""Backward-compatible wrapper for constants."""

from .. import constants as _constants

__all__ = list(_constants.__all__)

globals().update({name: getattr(_constants, name) for name in __all__})
