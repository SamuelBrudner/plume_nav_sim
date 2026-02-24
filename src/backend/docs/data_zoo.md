# Data Zoo

The Data Zoo is a curated registry of plume datasets for navigation RL research.
It provides a single Python function to load any registered dataset as a
standardized `xr.DataArray`, plus a CLI for browsing, downloading, and
inspecting datasets.

## Quick Start

```python
from plume_nav_sim.data_zoo import load_plume

plume = load_plume("colorado_jet_v1", auto_download=True)
```

`load_plume` returns an `xr.DataArray` with dimensions `(time, y, x)` and a
`concentration` data variable.

**Install requirements:**

- Registry browsing, CLI, and download: `pip install plume-nav-sim`
- Loading datasets (xarray, zarr, dask): `pip install plume-nav-sim[media]`

If `auto_download=True` is not set and the dataset is not already cached,
`load_plume` raises an error.

## Available Datasets

| ID | Title | Source | License | Size |
|----|-------|--------|---------|------|
| `colorado_jet_v1` | CU Boulder PLIF odor plume (a0004 near-bed, 10 cm/s) | Zenodo 4971113 | CC-BY-4.0 | ~46 MB HDF5 |
| `rigolli_dns_nose_v1` | Rigolli DNS turbulent plume -- nose level | Zenodo 15469831 | CC-BY-4.0 | ~6.8 GB .mat |
| `rigolli_dns_ground_v1` | Rigolli DNS turbulent plume -- ground level | Zenodo 15469831 | CC-BY-4.0 | ~6.8 GB .mat |
| `emonet_smoke_v1` | Emonet smoke plume video (full) | Dryad 4j0zpc87z | CC0-1.0 | ~29 GB .mat |
| `emonet_smoke_trimmed_v1` | Emonet smoke plume (tail-trimmed) | Dryad 4j0zpc87z | CC0-1.0 | ~29 GB .mat |

### colorado_jet_v1

150-frame acetone plume captured via planar laser-induced fluorescence (PLIF)
at 15 FPS, 406x216 px. Near-bed slice at 10 cm/s crossflow.

Citation: Connor et al. 2018, *Experiments in Fluids*.

### rigolli_dns_nose_v1

2D time-series extracted at nose height (~50 cm) from a 3D direct numerical
simulation (DNS) of turbulent channel flow with a scalar source.

Citation: Rigolli et al. 2022, *eLife*.

### rigolli_dns_ground_v1

Same DNS simulation as `rigolli_dns_nose_v1`, extracted at ground level (z=0).

Citation: Rigolli et al. 2022, *eLife*.

### emonet_smoke_v1

90 Hz smoke plume video in a 300x180 mm arena, background-subtracted.

**Warning:** This is a large download (~29 GB).

Citation: Demir et al. 2020, *eLife*.

### emonet_smoke_trimmed_v1

Same source as `emonet_smoke_v1` but with auto-trimmed low-intensity tail
frames.

**Warning:** This is a large download (~29 GB).

Citation: Demir et al. 2020, *eLife*.

## Normalization

Pass `normalize=` to `load_plume` to apply one of three normalization methods:

| Method | Formula | Use case |
|--------|---------|----------|
| `"minmax"` | `(x - min) / (max - min)` | Scale to [0, 1] |
| `"robust"` | `clip((x - q05) / (q95 - q05), 0, 1)` | Robust to outliers |
| `"zscore"` | `(x - mean) / std` | Zero mean, unit variance |

Example:

```python
plume = load_plume("colorado_jet_v1", normalize="robust", auto_download=True)
```

## CLI Usage

Installed via `pip install plume-nav-sim`. The CLI entry point is
`plume-nav-data-zoo`.

```bash
plume-nav-data-zoo list                          # list all datasets
plume-nav-data-zoo describe colorado_jet_v1      # detailed metadata
plume-nav-data-zoo download colorado_jet_v1      # download to cache
plume-nav-data-zoo download colorado_jet_v1 --force  # re-download
plume-nav-data-zoo cache-status                  # show what's cached
plume-nav-data-zoo validate                      # check registry integrity
```

Also available as:

```bash
python -m plume_nav_sim.data_zoo <command>
```

## Cache Management

- **Default location:** `~/.cache/plume_nav_sim/data_zoo/`
- **Override with env var:** `PLUME_DATA_ZOO_CACHE=/path/to/cache`
- **Override in code:** pass `cache_root=` to `load_plume()` or `ensure_dataset_available()`
- **Cache layout:** `<cache_root>/<cache_subdir>/<version>/<expected_root>`

To clear the cache for a single dataset, delete its directory under the cache
root. To clear everything:

```bash
rm -rf ~/.cache/plume_nav_sim/data_zoo
```

## Python API Reference

### load_plume

```python
load_plume(
    dataset_id: str,
    *,
    normalize: str | None = None,
    cache_root: Path | str | None = ...,
    auto_download: bool = False,
    chunks: str | dict = "auto",
) -> xr.DataArray
```

Load a registered dataset as a standardized DataArray with dims `(time, y, x)`.

### ensure_dataset_available

```python
ensure_dataset_available(
    dataset_id: str,
    *,
    cache_root: Path | None = None,
    auto_download: bool = False,
    force_download: bool = False,
    verify_checksum: bool = True,
) -> Path
```

Download and verify a dataset if not already cached. Returns the path to the
dataset on disk.

### describe_dataset

```python
describe_dataset(dataset_id: str) -> DatasetRegistryEntry
```

Return the full registry entry (metadata, artifact info, ingest spec) for a
dataset.

### get_dataset_registry

```python
get_dataset_registry() -> dict[str, DatasetRegistryEntry]
```

Return the complete dataset registry as a dict keyed by dataset ID.

### validate_registry

```python
validate_registry() -> None
```

Validate all registry entries. Raises `RegistryValidationError` on failure.

## Adding a New Dataset

Steps for contributors:

1. Add a `DatasetRegistryEntry` to `DATASET_REGISTRY` in `registry.py`.
2. Provide a `DatasetArtifact` with the download URL, checksum (MD5 or
   SHA256), `archive_type`, and layout.
3. Provide `DatasetMetadata` with title, creators, DOI, license, description,
   and citation.
4. If the source format is not Zarr, add an ingest spec. Subclass one of the
   existing ingest classes (`CrimaldiFluorescenceIngest`,
   `RigolliDNSIngest`, `EmonetSmokeIngest`) or create a new one for
   unsupported formats.
5. Run `plume-nav-data-zoo validate` to check the new entry.
6. Run `plume-nav-data-zoo download <new_id>` to test the full download and
   ingest pipeline.
7. Add a test to `tests/plume_nav_sim/data_zoo/test_smoke.py`.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `PLUME_DATA_ZOO_CACHE` | Override default cache root |
| `PLUME_DATA_ZOO_USER_AGENT` | Custom HTTP User-Agent for downloads |
| `PLUME_DRYAD_BEARER_TOKEN` | Bearer token for Dryad API authentication |
| `PLUME_DATA_ZOO_URL_OVERRIDE` | Override download URL for all datasets |
| `PLUME_DATA_ZOO_URL_OVERRIDE_{DATASET_ID}` | Override download URL for a specific dataset (e.g., `PLUME_DATA_ZOO_URL_OVERRIDE_COLORADO_JET_V1`) |

## Requirements

- **Core** (`pip install plume-nav-sim`): registry browsing, CLI, download.
- **Loading** (`pip install plume-nav-sim[media]`): adds xarray, zarr, and dask
  for dataset loading.

All datasets are converted to a standardized Zarr store on ingest, with a
`concentration` variable and dimensions `(time, y, x)`.
