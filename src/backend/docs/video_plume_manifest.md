# Video Plume Provenance Manifest

This manifest is written alongside video‐derived plume datasets (e.g., Zarr roots) to capture reproducibility metadata and invocation details.

- Location: `root/manifest.json`
- Model: `plume_nav_sim.media.ProvenanceManifest`

Example:

```
{
  "created_at": "2025-01-01T12:00:00Z",
  "git_sha": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
  "package_version": "0.0.0",
  "config_hash": "abc123cfg",
  "cli_args": [
    "video_ingest",
    "--input",
    "sample.mp4",
    "--output",
    "data/zarr/sample.zarr",
    "--fps",
    "12"
  ],
  "source_dtype": "uint8",
  "env": {
    "hostname": null,
    "platform": null,
    "python_version": null
  }
}
```

Notes:
- Include exact CLI args as invoked (ordering preserved).
- Avoid secrets; environment metadata is intentionally minimal.
- Optional: record a simple "git dirty" flag in your own pipeline if you need to distinguish uncommitted changes when producing datasets. The core manifest focuses on stable identifiers (`git_sha`, `package_version`).
- DVC: track the dataset directory as an `outs` and parametrize the ingest stage so `cli_args` reflect the same arguments used by `dvc repro`.
- Movie metadata (fps, spatial calibration, and HDF5 dataset selection) for movie‑backed plume fields is modeled separately in the per‑movie YAML sidecar; see `src/backend/SEMANTIC_MODEL.md` and `src/backend/docs/contracts/video_plume_dataset.md` for the sidecar schema and its mapping into the dataset attrs.
