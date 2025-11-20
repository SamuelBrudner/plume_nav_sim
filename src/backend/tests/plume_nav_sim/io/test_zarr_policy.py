from __future__ import annotations

import types

from plume_nav_sim.io.zarr_policy import (
    CHUNKS_TYX,
    DEFAULT_BLOSC_CLEVEL,
    compressor_config,
    default_encoding,
    make_blosc_compressor,
)


def test_chunks_constant():
    assert isinstance(CHUNKS_TYX, tuple)
    assert CHUNKS_TYX == (8, 64, 64)


def test_make_compressor_and_config_contract():
    comp = make_blosc_compressor()
    cfg = compressor_config(comp)
    assert isinstance(cfg, dict)
    assert cfg.get("id") == "blosc"
    assert cfg.get("cname") in {"zstd", "lz4"}
    assert int(cfg.get("clevel", -1)) == DEFAULT_BLOSC_CLEVEL
    # Required keys present for provenance
    for key in ("shuffle", "blocksize"):
        assert key in cfg


def test_default_encoding():
    enc = default_encoding()
    assert enc["chunks"] == CHUNKS_TYX
    cfg = compressor_config(enc["compressor"])  # type: ignore[index]
    assert cfg["id"] == "blosc"
    assert cfg["cname"] in {"zstd", "lz4"}
