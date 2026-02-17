# Zarr Chunking and Compression Policy

Status: Alpha (living doc).

Scope: movie-based plume concentration fields stored as `concentration: (t, y, x) float32`.

## Defaults

Chunks (T, Y, X): `(8, 64, 64)`

- Small `T=8` enables responsive time-indexed reads without loading long runs.
- `64x64` spatial chunks keep fixtures small and CI-friendly.

Compressor: Blosc/Zstd at `clevel=5`

- Fallback policy: if Zstd is unavailable in `numcodecs`, use Blosc/LZ4 and record the choice in dataset attrs.

## Reference Implementation

Reference implementation lives in `plume_nav_sim.io.zarr_policy`:

- `CHUNKS_TYX = (8, 64, 64)`
- `make_blosc_compressor(preferred_codecs=("zstd", "lz4"), clevel=5)`
- `compressor_config(compressor)` to serialize into `attrs` for provenance
- `default_encoding()` returns `{chunks, compressor}` for xarray/zarr

## Recommended Tests

- Encoding adopts the chunk shape.
- Compressor config resolves to `id=blosc` with `cname` in `{zstd, lz4}`.
- Writers persist compressor config in dataset attributes.
