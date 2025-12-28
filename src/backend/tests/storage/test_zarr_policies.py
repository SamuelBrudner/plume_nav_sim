import importlib

import pytest

zarr = pytest.importorskip("zarr")
numcodecs = pytest.importorskip("numcodecs")


def test_create_zarr_array_defaults(tmp_path):
    mod = importlib.import_module("plume_nav_sim.storage.zarr_policies")

    store = tmp_path / "video.zarr"
    arr = mod.create_zarr_array(
        store_path=str(store),
        name="frames",
        shape=(8, 64, 64),
        dtype="uint8",
        overwrite=True,
    )

    assert tuple(arr.shape) == (8, 64, 64)
    assert tuple(arr.chunks) == mod.CHUNKS_TYX

    # Compressor expectations: prefer zstd but accept lz4 fallback
    comp = arr.compressor
    # Some environments may lack zstd; allow lz4
    cname = getattr(comp, "cname", "none") if comp is not None else "none"
    assert cname in {"zstd", "lz4"}

    # Policy metadata recorded in attrs
    assert arr.attrs["plume_nav_sim:chunks"] == list(mod.CHUNKS_TYX)
    label = arr.attrs.get("plume_nav_sim:compressor")
    assert label in {"blosc:zstd", "blosc:lz4", "none:none"}
    assert isinstance(arr.attrs.get("plume_nav_sim:compressor_clevel"), int)


def test_blosc_zstd_fallback(tmp_path, monkeypatch: pytest.MonkeyPatch):
    mod = importlib.import_module("plume_nav_sim.storage.zarr_policies")

    # Force probe to fail for zstd and succeed for lz4
    def fake_probe(cname: str, clevel: int, shuffle: int) -> bool:
        return cname == "lz4"

    monkeypatch.setattr(mod, "_probe_blosc_support", fake_probe)

    comp = mod.create_blosc_compressor()
    # When zstd probe fails, we fall back to lz4
    assert getattr(comp, "cname", None) == "lz4"

    store = tmp_path / "video.zarr"
    arr = mod.create_zarr_array(
        store_path=str(store),
        name="frames",
        shape=(8, 64, 64),
        dtype="uint8",
        overwrite=True,
    )
    assert arr.attrs.get("plume_nav_sim:compressor") == "blosc:lz4"
