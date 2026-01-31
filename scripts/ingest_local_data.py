#!/usr/bin/env python
"""Ingest local datasets into Zarr stores for Zenodo upload.

Usage:
    python scripts/ingest_local_data.py --dataset colorado_jet
    python scripts/ingest_local_data.py --dataset emonet_smoke
    python scripts/ingest_local_data.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

from plume_nav_sim.data_zoo.download import (
    _generate_provenance_for_zarr,
    _ingest_emonet_to_zarr,
    _ingest_hdf5_to_zarr,
    _ingest_mat_to_zarr,
)
from plume_nav_sim.data_zoo.registry import DATASET_REGISTRY

# Local source paths on KINGSTON
LOCAL_SOURCES = {
    "colorado_jet_v1": Path(
        "/Volumes/KINGSTON/o2a_plumes/" "10302017_10cms_bounded_2.h5"
    ),
    "emonet_smoke_v1": Path(
        "/Volumes/KINGSTON/downloaded/" "2018_09_12_NA_3_3ds_5do_IS_1-frames.mat"
    ),
    "rigolli_dns_nose_v1": Path(
        "/Volumes/KINGSTON/downloaded/data_zoo/zenodo_15469831/1.0.0/nose_data.mat"
    ),
    "rigolli_dns_ground_v1": Path(
        "/Volumes/KINGSTON/downloaded/data_zoo/zenodo_15469831/1.0.0/ground_data.mat"
    ),
}

OUTPUT_ROOT = Path("/Volumes/KINGSTON/downloaded/data_zoo_zarr")


def ingest_dataset(dataset_id: str) -> Path:
    """Ingest a dataset from local source to Zarr."""
    if dataset_id not in LOCAL_SOURCES:
        raise ValueError(f"No local source for {dataset_id}")

    source = LOCAL_SOURCES[dataset_id]
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    entry = DATASET_REGISTRY[dataset_id]
    output = OUTPUT_ROOT / dataset_id / entry.expected_root

    print(f"\n{'='*60}")
    print(f"Ingesting: {dataset_id}")
    print(f"  Source: {source} ({source.stat().st_size / 1e9:.1f} GB)")
    print(f"  Output: {output}")
    print(f"  Spec: {type(entry.ingest).__name__}")
    print(f"{'='*60}\n")

    output.parent.mkdir(parents=True, exist_ok=True)

    # Choose ingest function based on spec type
    spec = entry.ingest
    if dataset_id == "colorado_jet_v1":
        result = _ingest_hdf5_to_zarr(spec, source, output)
    elif dataset_id == "emonet_smoke_v1":
        result = _ingest_emonet_to_zarr(spec, source, output)
    elif dataset_id.startswith("rigolli_dns_"):
        # Rigolli DNS uses mat_to_zarr (coords file downloaded automatically)
        result = _ingest_mat_to_zarr(spec, source, output)
    else:
        raise ValueError(f"No ingest handler for {dataset_id}")

    # Generate provenance sidecar
    _generate_provenance_for_zarr(entry, source, result)

    print(f"\n✓ Done: {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest local datasets to Zarr")
    parser.add_argument(
        "--dataset",
        choices=list(LOCAL_SOURCES.keys()),
        help="Dataset to ingest",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all available local datasets",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets",
    )

    args = parser.parse_args()

    if args.list:
        print("Available local datasets:")
        for ds_id, path in LOCAL_SOURCES.items():
            exists = "✓" if path.exists() else "✗"
            size = f"{path.stat().st_size / 1e9:.1f} GB" if path.exists() else "missing"
            print(f"  {exists} {ds_id}: {size}")
        return 0

    if args.all:
        datasets = list(LOCAL_SOURCES.keys())
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.print_help()
        return 1

    for ds_id in datasets:
        try:
            ingest_dataset(ds_id)
        except Exception as e:
            print(f"✗ Failed {ds_id}: {e}")
            import traceback

            traceback.print_exc()

    return 0


if __name__ == "__main__":
    sys.exit(main())
