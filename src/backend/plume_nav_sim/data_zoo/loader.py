"""Convenience loader for registry-backed plume datasets with normalization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, MutableMapping, cast

from ..media.schema import DIMS_TYX, VARIABLE_CONCENTRATION
from .download import ensure_dataset_available
from .registry import DEFAULT_CACHE_ROOT, describe_dataset
from .stats import NormalizationMethod, load_stats_from_zarr, normalize_array

if TYPE_CHECKING:
    import xarray as xr

LOG = logging.getLogger(__name__)


def _normalize_arg(normalize: NormalizationMethod | str | None) -> NormalizationMethod:
    if normalize is None:
        return None
    norm = normalize.lower() if isinstance(normalize, str) else normalize
    if norm not in ("minmax", "robust", "zscore"):
        raise ValueError(
            f"normalize must be one of None, 'minmax', 'robust', or 'zscore'; got {normalize!r}"
        )
    return cast(NormalizationMethod, norm)


def _open_zarr_dataset(path: Path, *, chunks: Any) -> "xr.Dataset":
    try:
        import xarray as xr  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "load_plume requires optional dependency 'xarray'. "
            "Install media extras: pip install plume-nav-sim[media]"
        ) from exc

    def _open(consolidated: bool) -> "xr.Dataset":
        return xr.open_zarr(  # type: ignore[attr-defined]
            str(path), consolidated=consolidated, chunks=chunks
        )

    first_error: Exception | None = None
    try:
        return _open(consolidated=True)
    except Exception as exc:  # pragma: no cover - exercised in fallback path
        first_error = exc
        LOG.debug("Falling back to unconsolidated Zarr for %s: %s", path, exc)
    try:
        return _open(consolidated=False)
    except Exception as exc:
        message = str(exc).lower()
        if (
            isinstance(exc, ValueError)
            and chunks == "auto"
            and ("dask" in message or "chunk" in message)
        ):
            raise ImportError(
                "load_plume uses dask-backed chunking by default. "
                "Install dask or call load_plume(..., chunks=None) to disable chunking."
            ) from exc
        if first_error:
            raise RuntimeError(
                f"Failed to open Zarr store at {path}: {first_error} // {exc}"
            ) from exc
        raise


def _standardize_dims(var: "xr.DataArray") -> "xr.DataArray":
    dims = tuple(var.dims)
    if len(dims) != 3:
        raise ValueError(
            f"Variable '{VARIABLE_CONCENTRATION}' must have 3 dims; got {dims}"
        )
    if dims != DIMS_TYX:
        rename = {old: new for old, new in zip(dims, DIMS_TYX)}
        var = var.rename(rename)
    if "_ARRAY_DIMENSIONS" not in var.attrs:
        var = var.assign_attrs({**var.attrs, "_ARRAY_DIMENSIONS": list(DIMS_TYX)})
    return var.transpose(*DIMS_TYX)


def _coerce_coord(
    ds: "xr.Dataset", name: str, expected_len: int
) -> "xr.DataArray | None":
    if name not in ds:
        return None
    coord = ds[name]
    if coord.ndim != 1 or coord.sizes.get(coord.dims[0]) != expected_len:
        return None
    if coord.dims != (name,):
        coord = coord.rename({coord.dims[0]: name})
    return coord


def _attach_coords(var: "xr.DataArray", ds: "xr.Dataset") -> "xr.DataArray":
    coords: Dict[str, "xr.DataArray"] = {}
    x_coord = _coerce_coord(ds, "x", int(var.sizes["x"]))
    y_coord = _coerce_coord(ds, "y", int(var.sizes["y"]))
    if x_coord is not None:
        coords["x"] = x_coord
    if y_coord is not None:
        coords["y"] = y_coord
    return var.assign_coords(coords) if coords else var


def _apply_normalization(
    var: "xr.DataArray",
    stats: Mapping[str, Any] | None,
    method: NormalizationMethod,
) -> "xr.DataArray":
    if stats is None:
        raise ValueError(
            "Normalization requested but concentration_stats are missing. "
            "Compute stats with scripts/compute_zarr_stats.py before loading."
        )

    try:
        import xarray as xr  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "load_plume normalization requires optional dependency 'xarray'. "
            "Install media extras: pip install plume-nav-sim[media]"
        ) from exc

    has_chunks = bool(getattr(var, "chunks", None))
    normalized = xr.apply_ufunc(
        lambda block: normalize_array(block, stats, method=method),
        var,
        dask="parallelized" if has_chunks else None,
        output_dtypes=[var.dtype],
        keep_attrs=True,
    )
    attrs: MutableMapping[str, Any] = dict(normalized.attrs)
    attrs["normalized"] = method
    attrs["concentration_stats"] = dict(stats)
    normalized = normalized.assign_attrs(attrs)
    normalized.name = VARIABLE_CONCENTRATION
    return normalized


def load_plume(
    dataset_id: str,
    *,
    normalize: NormalizationMethod | str | None = None,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    auto_download: bool = False,
    chunks: Any = "auto",
) -> "xr.DataArray":
    """Load a registry dataset as an xarray.DataArray with optional normalization.

    Args:
        dataset_id: Registry identifier (e.g., "colorado_jet_v1").
        normalize: Normalization method ("minmax", "robust", "zscore", or None).
        cache_root: Base cache directory for registry data.
        auto_download: Allow downloads when the dataset is missing from cache.
        chunks: Chunking strategy passed to ``xr.open_zarr`` (default: "auto" for dask).
    """

    method = _normalize_arg(normalize)
    entry = describe_dataset(dataset_id)
    layout = entry.ingest.output_layout if entry.ingest else entry.artifact.layout
    if layout.lower() != "zarr":
        raise ValueError(
            f"Dataset '{dataset_id}' has layout '{layout}', expected a Zarr store"
        )

    dataset_path = ensure_dataset_available(
        dataset_id,
        cache_root=cache_root,
        auto_download=auto_download,
    )

    ds = _open_zarr_dataset(Path(dataset_path), chunks=chunks)
    if VARIABLE_CONCENTRATION not in ds:
        raise KeyError(
            f"Dataset '{dataset_id}' is missing variable '{VARIABLE_CONCENTRATION}'"
        )

    var = _standardize_dims(ds[VARIABLE_CONCENTRATION])
    var = _attach_coords(var, ds)
    stats = ds.attrs.get("concentration_stats") or load_stats_from_zarr(
        Path(dataset_path)
    )

    result = _apply_normalization(var, stats, method) if method is not None else var
    merged_attrs: Dict[str, Any] = {**ds.attrs, **result.attrs}
    if stats is not None:
        merged_attrs.setdefault("concentration_stats", stats)
    result = result.assign_attrs(merged_attrs)
    result.name = VARIABLE_CONCENTRATION
    return result


__all__ = ["load_plume"]
