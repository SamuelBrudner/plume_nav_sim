from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from plume_nav_sim.io.zarr_policy import CHUNKS_TYX
from plume_nav_sim.utils.time_mapping import (
    DEFAULT_FRAME_MAPPING_POLICY,
    map_step_to_frame,
)
from plume_nav_sim.video.schema import SCHEMA_VERSION, validate_attrs


def _build_synthetic_frames(T: int, Y: int, X: int) -> np.ndarray:
    # Deterministic synthetic content: frame t filled with t/(T-1)
    # Use float32 to match expected dataset dtype
    if T <= 1:
        scale = 0.0
    else:
        scale = 1.0 / float(T - 1)
    frames = np.empty((T, Y, X), dtype=np.float32)
    for t in range(T):
        frames[t, :, :] = t * scale
    return frames


@pytest.mark.skipif(
    os.environ.get("PLUME_SKIP_ZARR_TESTS") == "1", reason="Zarr tests disabled by env"
)
def test_zarr_roundtrip_small_dataset(tmp_path: Path):
    # Hard skip when optional deps missing
    zarr = pytest.importorskip("zarr")
    numcodecs = pytest.importorskip("numcodecs")  # noqa: F401

    from plume_nav_sim.storage.zarr_policies import create_zarr_array

    root = tmp_path / "video_plume.zarr"
    arr = create_zarr_array(
        store_path=root,
        name="concentration",
        shape=(10, 32, 32),  # (t, y, x)
        dtype="f4",
        overwrite=True,
    )

    # Write deterministic synthetic frames
    frames = _build_synthetic_frames(10, 32, 32)
    arr[:] = frames

    # Root-level attrs required by schema
    grp = zarr.open_group(root, mode="a")
    grp.attrs.update(
        {
            "schema_version": SCHEMA_VERSION,
            "fps": 12.0,
            "source_dtype": "uint8",
            "pixel_to_grid": (1.0, 1.0),
            "origin": (0.0, 0.0),
            "extent": (32.0, 32.0),
        }
    )

    # Validate attrs via the contract
    model = validate_attrs(grp.attrs)
    assert model.fps == 12.0

    # Verify array policy persisted
    assert tuple(arr.chunks) == CHUNKS_TYX
    comp_label = arr.attrs.get("plume_nav_sim:compressor")
    assert comp_label is not None and isinstance(comp_label, str)
    assert comp_label.startswith("blosc:") or comp_label == "none:none"


def test_deterministic_frame_selection_with_time_mapping():
    # Determinism check across two passes using mapping helper
    T = 7
    fps = 24.0
    N = 25
    seq1 = [map_step_to_frame(k, total_frames=T, fps=fps) for k in range(N)]
    seq2 = [map_step_to_frame(k, total_frames=T, fps=fps) for k in range(N)]
    assert seq1 == seq2
    # Policy is wrap by default
    assert DEFAULT_FRAME_MAPPING_POLICY.value == "wrap"


@pytest.mark.slow
def test_perf_sanity_small_128x128(tmp_path: Path):
    # Lightweight perf sanity; marked slow so it can be skipped in fast CI
    zarr = pytest.importorskip("zarr")
    numcodecs = pytest.importorskip("numcodecs")  # noqa: F401

    from plume_nav_sim.storage.zarr_policies import create_zarr_array

    T, Y, X = 16, 128, 128
    root = tmp_path / "video_plume_128.zarr"
    arr = create_zarr_array(root, "concentration", (T, Y, X), "f4", overwrite=True)
    arr[:] = _build_synthetic_frames(T, Y, X)

    # Iterate a modest number of mapped indices to exercise chunked access
    idxs = [map_step_to_frame(k, total_frames=T, fps=30.0) for k in range(200)]
    # Read a small slice per index to avoid heavy IO on CI
    s = (slice(None), slice(0, 8), slice(0, 8))
    for k in idxs[:50]:  # keep runtime tight
        _ = arr[k][s[1], s[2]]
