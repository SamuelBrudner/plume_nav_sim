"""Minimal smoke test for a tiny movie-plume Zarr dataset.

Runs without plotting; suitable for CI. It will:
- Generate a small (t,y,x) float32 array with a moving Gaussian
- If xarray is available, write it to Zarr at results/movie_plume_demo.zarr
- Load (or use the in-memory array) and print basic stats

Exit code 0 on success; non-zero on unexpected exceptions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _make_moving_gaussian_frames(T=6, H=16, W=16, sigma=4.0) -> np.ndarray:
    ys = np.arange(H)[:, None]
    xs = np.arange(W)[None, :]
    frames = []
    for t in range(T):
        cx = int((t / max(1, T - 1)) * (W - 1))
        cy = int((t / max(1, T - 1)) * (H - 1))
        g = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma**2))).astype(
            np.float32
        )
        frames.append(g)
    arr = np.stack(frames, axis=0)  # (t, y, x)
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    return arr.astype(np.float32)


def main() -> int:
    out = Path("results/movie_plume_demo.zarr")
    frames = _make_moving_gaussian_frames()

    # Attempt to persist as Zarr if xarray is available
    wrote = False
    try:
        import xarray as xr  # type: ignore

        if out.exists():
            import shutil

            shutil.rmtree(out)
        da = xr.DataArray(
            frames,
            dims=("t", "y", "x"),
            name="concentration",
            attrs={
                "schema_version": "v0",
                "fps": 5.0,
                "timebase": "seconds",
                "pixel_to_grid": 1.0,
                "origin": (0, 0),
                "source_dtype": "float32",
                "provenance": {"source": "synthetic", "note": "demo"},
            },
        )
        xr.Dataset({"concentration": da}).to_zarr(out, mode="w")
        wrote = True
    except Exception:
        pass

    # Load and check
    try:
        if wrote:
            import xarray as xr  # type: ignore

            ds = xr.open_zarr(out)
            da = ds["concentration"]
            shape = tuple(int(x) for x in da.shape)
            mn = float(da.min())
            mx = float(da.max())
            print("zarr_ok", shape, f"min={mn:.3f}", f"max={mx:.3f}")
        else:
            shape = frames.shape
            mn = float(frames.min())
            mx = float(frames.max())
            print("mem_ok", shape, f"min={mn:.3f}", f"max={mx:.3f}")
    except Exception as e:
        print("error:", e, file=sys.stderr)
        return 2

    # Basic invariants
    if shape[0] < 1 or shape[1] < 4 or shape[2] < 4:
        print("error: unexpected tiny shape", shape, file=sys.stderr)
        return 3
    if not (0.0 <= mn <= mx <= 1.0):
        print("error: values not in [0,1]", mn, mx, file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
