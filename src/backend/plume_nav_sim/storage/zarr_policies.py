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
    try:
        from numcodecs import blosc as _blosc

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
    try:
        from numcodecs import blosc as _blosc
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
    try:
        import zarr as _zarr
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
