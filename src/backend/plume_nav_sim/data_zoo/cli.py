#!/usr/bin/env python
"""CLI for downloading and managing Data Zoo datasets.

Usage:
    # Download all datasets to default cache
    python -m plume_nav_sim.data_zoo.cli

    # Download to external storage
    python -m plume_nav_sim.data_zoo.cli --cache-root /Volumes/KINGSTON/downloaded

    # Download specific dataset
    python -m plume_nav_sim.data_zoo.cli --dataset emonet_smoke_v1

    # List available datasets
    python -m plume_nav_sim.data_zoo.cli --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .download import ensure_dataset_available
from .registry import DATASET_REGISTRY, DEFAULT_CACHE_ROOT


def main() -> int:
    """Download Data Zoo datasets to local cache."""
    parser = argparse.ArgumentParser(
        description="Download and manage Data Zoo datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help=f"Cache directory for downloads (default: {DEFAULT_CACHE_ROOT})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Download specific dataset ID (default: all datasets)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_datasets",
        help="List available datasets and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )

    args = parser.parse_args()

    # List mode
    if args.list_datasets:
        print("Available datasets:")
        for dataset_id, entry in DATASET_REGISTRY.items():
            title = entry.metadata.title if entry.metadata else "No title"
            print(f"  {dataset_id}: {title}")
        return 0

    # Determine which datasets to download
    if args.dataset:
        if args.dataset not in DATASET_REGISTRY:
            print(f"Error: Unknown dataset '{args.dataset}'", file=sys.stderr)
            print(f"Available: {', '.join(DATASET_REGISTRY.keys())}", file=sys.stderr)
            return 1
        dataset_ids = [args.dataset]
    else:
        dataset_ids = list(DATASET_REGISTRY.keys())

    print(f"Cache root: {args.cache_root}")
    print(f"Datasets to download: {len(dataset_ids)}")
    print()

    # Dry run mode
    if args.dry_run:
        for dataset_id in dataset_ids:
            entry = DATASET_REGISTRY[dataset_id]
            cache_dir = entry.cache_path(args.cache_root)
            print(f"Would download: {dataset_id}")
            print(f"  URL: {entry.artifact.url}")
            print(f"  Target: {cache_dir / entry.expected_root}")
        return 0

    # Download datasets
    success = 0
    failed = 0

    for dataset_id in dataset_ids:
        print(f"Downloading {dataset_id}...")
        try:
            path = ensure_dataset_available(
                dataset_id,
                cache_root=args.cache_root,
                auto_download=True,
            )
            print(f"  ✓ {path}")
            success += 1
        except Exception as e:
            print(f"  ✗ {e}", file=sys.stderr)
            failed += 1

    print()
    print(f"Done: {success} succeeded, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
