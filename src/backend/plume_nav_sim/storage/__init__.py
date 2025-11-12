"""Storage utilities and contracts for dataset formats.

This package hosts helpers and policies for on-disk representations used by
plume_nav_sim, such as Zarr chunking and compression choices.
"""

from .zarr_policies import (
    CHUNKS_TYX,
    DEFAULT_BLOSC_CLEVEL,
    DEFAULT_BLOSC_CNAME,
    DEFAULT_BLOSC_SHUFFLE,
    create_blosc_compressor,
    create_zarr_array,
)

__all__ = [
    "CHUNKS_TYX",
    "DEFAULT_BLOSC_CNAME",
    "DEFAULT_BLOSC_CLEVEL",
    "DEFAULT_BLOSC_SHUFFLE",
    "create_blosc_compressor",
    "create_zarr_array",
]
