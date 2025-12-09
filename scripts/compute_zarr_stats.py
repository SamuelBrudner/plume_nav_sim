#!/usr/bin/env python
"""Compute and store concentration statistics for existing zarr datasets.

Usage:
    python scripts/compute_zarr_stats.py --dataset colorado_jet_v1
    python scripts/compute_zarr_stats.py --all --skip emonet_smoke_v1
    python scripts/compute_zarr_stats.py --list
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

from plume_nav_sim.data_zoo.stats import (
    compute_concentration_stats,
    load_stats_from_zarr,
    store_stats_in_zarr,
)

ZARR_ROOT = Path("/Volumes/KINGSTON/downloaded/data_zoo_zarr")

# Dataset configurations
DATASETS = {
    "colorado_jet_v1": {
        "zarr": "a0004_nearbed_10cm_s.zarr",
        "sample_frames": None,  # Small enough to process all
    },
    "emonet_smoke_v1": {
        "zarr": "emonet_smoke.zarr",
        "sample_frames": 500,  # Large dataset, sample for quantiles
    },
    "rigolli_dns_nose_v1": {
        "zarr": "rigolli_nose.zarr",
        "sample_frames": None,
    },
    "rigolli_dns_ground_v1": {
        "zarr": "rigolli_ground.zarr",
        "sample_frames": None,
    },
}


def compute_stats_for_dataset(
    dataset_id: str,
    force: bool = False,
) -> None:
    """Compute and store stats for a single dataset."""
    if dataset_id not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_id}")

    config = DATASETS[dataset_id]
    zarr_path = ZARR_ROOT / dataset_id / config["zarr"]

    if not zarr_path.exists():
        print(f"✗ {dataset_id}: zarr not found at {zarr_path}")
        return

    # Check if stats already exist
    existing = load_stats_from_zarr(zarr_path)
    if existing and not force:
        print(f"⊘ {dataset_id}: stats already exist (use --force to recompute)")
        return

    print(f"\n{'='*60}")
    print(f"Computing stats for: {dataset_id}")
    print(f"  Zarr: {zarr_path}")
    print(f"  Sample frames: {config['sample_frames'] or 'all'}")
    print(f"{'='*60}\n")

    # Compute stats
    stats = compute_concentration_stats(
        zarr_path,
        sample_frames=config["sample_frames"],
    )

    # Check provenance for original range (if pre-normalized)
    provenance_path = zarr_path.with_suffix(".zarr.provenance.json")
    if provenance_path.exists():
        with open(provenance_path) as f:
            prov = json.load(f)
        # Check if this was normalized during ingest
        ingest_params = prov.get("ingest_params", {})
        if ingest_params.get("normalize", False):
            stats["normalized_during_ingest"] = True
            # For Emonet, we know the original range from the logs
            if dataset_id == "emonet_smoke_v1":
                stats["original_min"] = 10.0
                stats["original_max"] = 255.0

    # Store stats
    store_stats_in_zarr(zarr_path, stats)

    print(f"\n✓ Stats stored for {dataset_id}:")
    print(f"  min={stats['min']:.4f}, max={stats['max']:.4f}")
    print(f"  mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    print(
        f"  quantiles: q05={stats['quantiles']['q05']:.4f}, "
        f"q50={stats['quantiles']['q50']:.4f}, "
        f"q95={stats['quantiles']['q95']:.4f}"
    )
    print(f"  nonzero_fraction={stats['nonzero_fraction']:.4f}")


def list_datasets() -> None:
    """List available datasets and their stats status."""
    print("Available datasets:\n")
    for dataset_id, config in DATASETS.items():
        zarr_path = ZARR_ROOT / dataset_id / config["zarr"]
        if zarr_path.exists():
            existing = load_stats_from_zarr(zarr_path)
            status = "✓ has stats" if existing else "○ no stats"
        else:
            status = "✗ not found"
        print(f"  {status}  {dataset_id}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute concentration stats for zarr datasets"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        help="Dataset to compute stats for",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compute stats for all datasets",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Datasets to skip (use with --all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute stats even if they exist",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets",
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return 0

    if args.all:
        datasets = [d for d in DATASETS.keys() if d not in args.skip]
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.print_help()
        return 1

    for ds_id in datasets:
        try:
            compute_stats_for_dataset(ds_id, force=args.force)
        except Exception as e:
            print(f"✗ Failed {ds_id}: {e}")
            import traceback

            traceback.print_exc()

    return 0


if __name__ == "__main__":
    sys.exit(main())
