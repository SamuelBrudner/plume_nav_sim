"""Zarr chunking and compression policy.

Contract summary (bead plume_nav_sim-197):
- Chunking: CHUNKS_TYX = (8, 64, 64) for time, Y, X layouts
- Compressor: Blosc with Zstandard (zstd) at clevel=5 when available
- Fallback: If zstd isn't available in the runtime's numcodecs/blosc build,
  fall back to Blosc with LZ4, emitting a warning, while keeping the same API
  surface and recording the chosen compressor in Zarr attrs.

Rationale (abridged):
- Small temporal chunks (T=8) keep write amplification and partial reads low
  for short clips while enabling bounded buffering.
- Spatial chunks (64x64) align well with common cache lines and typical image
  operations while keeping per-chunk memory modest for CI.
- Zstd generally yields good compression ratios with solid speed; clevel=5
  balances CPU cost with CI throughput. LZ4 is the compatibility fallback.
  Note: Zstd availability depends on the c-blosc build bundled with
  numcodecs; many environments provide only LZ4. If you require Zstd, ensure
  your numcodecs/c-blosc installation includes Zstd support for your platform.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

# Public policy constants
CHUNKS_TYX: Tuple[int, int, int] = (8, 64, 64)
DEFAULT_BLOSC_CNAME: str = "zstd"
DEFAULT_BLOSC_CLEVEL: int = 5
# Blosc shuffle filter: 0=NOSHUFFLE, 1=SHUFFLE, 2=BITSHUFFLE
DEFAULT_BLOSC_SHUFFLE: int = 1

# Internal guard to avoid spamming tests/logs when zstd is unavailable.
# We emit at most one downgraded warning per process for the zstd->lz4 fallback.
_ZSTD_FALLBACK_WARNED: bool = False


@dataclass(frozen=True)
class CompressorInfo:
    """Summary of the compressor used for a Zarr array."""

    kind: str  # e.g., "blosc"
    name: str  # e.g., "zstd" or "lz4"
    clevel: int
    shuffle: int

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.name}"


def _probe_blosc_support(cname: str, clevel: int, shuffle: int) -> bool:
    """Return True if a working blosc compressor with `cname` can encode bytes.

    We avoid importing numcodecs at module import time to keep the import graph
    light; this function is only called within creators.
    """
    try:
        from numcodecs import blosc as _blosc  # type: ignore

        comp = _blosc.Blosc(cname=cname, clevel=clevel, shuffle=shuffle)
        # Encode a tiny payload to verify the codec is actually usable.
        _ = comp.encode(b"plume-nav-sim")
        return True
    except Exception:
        return False


def create_blosc_compressor(
    preferred: str = DEFAULT_BLOSC_CNAME,
    clevel: int = DEFAULT_BLOSC_CLEVEL,
    shuffle: int = DEFAULT_BLOSC_SHUFFLE,
):
    """Create a numcodecs Blosc compressor honoring policy and fallbacks.

    Attempts to create a compressor for `preferred` (default: "zstd"). If that
    compressor cannot be used in the current environment, falls back to "lz4"
    with a warning, preserving the same interface.

    Returns the numcodecs compressor instance. Raises ImportError if numcodecs
    is not available.
    """
    try:
        from numcodecs import blosc as _blosc  # type: ignore
    except Exception as e:  # pragma: no cover - exercised in environments w/o numcodecs
        raise ImportError(
            "numcodecs is required for Zarr compression. Install with 'pip install numcodecs zarr'."
        ) from e

    if _probe_blosc_support(preferred, clevel, shuffle):
        return _blosc.Blosc(cname=preferred, clevel=clevel, shuffle=shuffle)

    # Fallback path: prefer lz4 if zstd isn't available
    fallback = "lz4"
    if not _probe_blosc_support(fallback, clevel, shuffle):  # extremely unlikely
        # Last resort: no compression
        warnings.warn(
            "Blosc zstd and lz4 unavailable; proceeding without compression.",
            RuntimeWarning,
        )
        return None  # type: ignore[return-value]

    # Downgrade and emit at most once per process to avoid noisy test output.
    # Note: Zstd support depends on the c-blosc build bundled with numcodecs;
    # many environments only include lz4. To enable zstd, install a build of
    # numcodecs/c-blosc with zstd enabled on your platform.
    global _ZSTD_FALLBACK_WARNED
    if not _ZSTD_FALLBACK_WARNED:
        warnings.warn(
            "Blosc zstd not available; falling back to lz4.",
            UserWarning,
            stacklevel=2,
        )
        _ZSTD_FALLBACK_WARNED = True
    return _blosc.Blosc(cname=fallback, clevel=clevel, shuffle=shuffle)


def _infer_compressor_info(comp) -> CompressorInfo:
    if comp is None:
        return CompressorInfo(kind="none", name="none", clevel=0, shuffle=0)
    # numcodecs.blosc.Blosc exposes cname/clevel/shuffle attributes
    name = getattr(comp, "cname", "unknown")
    clevel = int(getattr(comp, "clevel", 0))
    shuffle = int(getattr(comp, "shuffle", 0))
    return CompressorInfo(kind="blosc", name=str(name), clevel=clevel, shuffle=shuffle)


def create_zarr_array(
    store_path: "str | bytes | os.PathLike[str]",
    name: str,
    shape: Sequence[int],
    dtype: "str | object",
    *,
    chunks: Optional[Sequence[int]] = None,
    compressor=None,
    overwrite: bool = False,
    extra_attrs: Optional[dict] = None,
):
    """Create (or require) a Zarr array following the default policy.

    - Uses CHUNKS_TYX unless overridden
    - Uses Blosc Zstd clevel=5 when available, otherwise lz4
    - Records the chosen compressor metadata in attrs

    Returns the created or existing Zarr array.
    """
    try:
        import zarr as _zarr  # type: ignore
    except Exception as e:  # pragma: no cover - exercised in environments w/o zarr
        raise ImportError(
            "zarr is required to create Zarr arrays. Install with 'pip install zarr numcodecs'."
        ) from e

    if chunks is None:
        if len(shape) == 3:
            chunks = CHUNKS_TYX
        else:
            # Simple heuristic: put CHUNKS_TYX on the last 3 dims, leave leading dims unchunked (None)
            # Example: (C, T, Y, X) -> (1, 8, 64, 64) if C=1 else (None, 8, 64, 64)
            lead = len(shape) - 3
            chunks = tuple([1] * lead + list(CHUNKS_TYX))  # type: ignore[assignment]

    if compressor is None:
        compressor = create_blosc_compressor()

    comp_info = _infer_compressor_info(compressor)

    grp = _zarr.open_group(store_path, mode="a")
    if overwrite and name in grp:
        del grp[name]

    arr = grp.require_dataset(
        name,
        shape=tuple(shape),
        dtype=dtype,
        chunks=tuple(chunks),
        compressor=compressor,
    )

    # Record policy metadata
    attrs = {
        "plume_nav_sim:chunks": list(arr.chunks),
        "plume_nav_sim:compressor": comp_info.label,
        "plume_nav_sim:compressor_clevel": comp_info.clevel,
        "plume_nav_sim:compressor_shuffle": comp_info.shuffle,
    }
    if extra_attrs:
        attrs.update(extra_attrs)
    arr.attrs.update(attrs)
    return arr
