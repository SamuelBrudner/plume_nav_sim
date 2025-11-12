# Zarr Chunking and Compression Policy

Scope: Movie-based plume concentration fields `concentration: (t, y, x) float32`.

- Chunks (T,Y,X): `(8, 64, 64)`
  - Small `T=8` enables responsive time-indexed reads and deterministic stepping
    without loading long temporal runs.
  - Spatial `64x64` balances memory locality, CPU cache friendliness, and
    bounded chunk sizes for CI stability on tiny fixtures (e.g., 128×128).

- Compressor: Blosc/Zstd at `clevel=5`
  - Strong compression with good throughput on float32 frames.
  - Fallback policy: if Zstd is unavailable in `numcodecs`, use Blosc/LZ4 and
    record that choice in dataset attrs. This maintains performance and avoids
    hard failures on constrained environments.

Reference implementation lives in `plume_nav_sim.io.zarr_policy`:

- `CHUNKS_TYX = (8, 64, 64)`
- `make_blosc_compressor(preferred_codecs=("zstd","lz4"), clevel=5)`
- `compressor_config(compressor)` to serialize into `attrs` for provenance
- `default_encoding()` returns `{chunks, compressor}` for xarray/zarr

Notes

- Tests should validate that encodings adopt the chunk shape and that the
  compressor config resolves to `id=blosc` with `cname` in `{zstd,lz4}`.
- Writers should persist the chosen compressor config in dataset attributes to
  support reproducibility and environment parity checks.
