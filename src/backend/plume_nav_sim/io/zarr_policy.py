from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

# Canonical Zarr chunks for movie plume concentration fields: (t, y, x)
CHUNKS_TYX: tuple[int, int, int] = (8, 64, 64)

# Default Blosc compression level; team‑approved baseline for CI and docs.
DEFAULT_BLOSC_CLEVEL: int = 5


@dataclass
class _BloscFallback:
    cname: str
    clevel: int
    shuffle: int = 1
    blocksize: int = 0

    def get_config(self) -> dict:
        return {
            "id": "blosc",
            "cname": self.cname,
            "clevel": int(self.clevel),
            "shuffle": int(self.shuffle),
            "blocksize": int(self.blocksize),
        }


def make_blosc_compressor(
    *,
    preferred_codecs: Sequence[str] | None = ("zstd", "lz4"),
    clevel: int = DEFAULT_BLOSC_CLEVEL,
    shuffle: int = 1,
) -> object:
    candidates: Iterable[str] = preferred_codecs or ("zstd", "lz4")

    try:
        from numcodecs.blosc import Blosc
    except Exception:
        warnings.warn(
            "numcodecs not available; using Blosc fallback shim (lz4)",
            RuntimeWarning,
            stacklevel=2,
        )
        # Choose a fast default for config recording; cannot actually compress.
        return _BloscFallback(cname="lz4", clevel=int(clevel), shuffle=int(shuffle))

    for cname in candidates:
        try:
            return Blosc(cname=str(cname), clevel=int(clevel), shuffle=int(shuffle))
        except Exception:  # pragma: no cover - exercised only when codec missing
            # Keep trying next candidate; capture is unnecessary
            continue

    warnings.warn(
        f"Requested Blosc codecs {list(candidates)} unavailable; using lz4 shim",
        RuntimeWarning,
        stacklevel=2,
    )
    return _BloscFallback(cname="lz4", clevel=int(clevel), shuffle=int(shuffle))


def compressor_config(compressor: object) -> dict:
    get_cfg = getattr(compressor, "get_config", None)
    if callable(get_cfg):
        return dict(get_cfg())
    # Best‑effort introspection
    cfg = {
        "id": "blosc",
        "cname": getattr(compressor, "cname", "unknown"),
        "clevel": int(getattr(compressor, "clevel", DEFAULT_BLOSC_CLEVEL)),
        "shuffle": int(getattr(compressor, "shuffle", 1)),
        "blocksize": int(getattr(compressor, "blocksize", 0)),
    }
    return cfg


def default_encoding() -> dict:
    return {
        "chunks": CHUNKS_TYX,
        "compressor": make_blosc_compressor(),
    }


__all__ = [
    "CHUNKS_TYX",
    "DEFAULT_BLOSC_CLEVEL",
    "make_blosc_compressor",
    "compressor_config",
    "default_encoding",
]
