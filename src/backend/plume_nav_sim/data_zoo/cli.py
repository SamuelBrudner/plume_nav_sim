"""Command-line interface for the Data Zoo.

Usage::

    python -m plume_nav_sim.data_zoo list
    python -m plume_nav_sim.data_zoo describe colorado_jet_v1
    python -m plume_nav_sim.data_zoo download colorado_jet_v1
    python -m plume_nav_sim.data_zoo cache-status
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path
from typing import Optional, Sequence

from .registry import (
    DEFAULT_CACHE_ROOT,
    DATASET_REGISTRY,
    DatasetRegistryEntry,
    describe_dataset,
    validate_registry,
)

logger = logging.getLogger("plume_nav_sim.data_zoo.cli")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logger(level: str) -> None:
    root = logging.getLogger("plume_nav_sim.data_zoo")
    if not root.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(levelname)s %(message)s")
        handler.setFormatter(fmt)
        root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def _truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return text[: width - 1] + "\u2026"


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a simple aligned table to stdout."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in col_widths)))
    for row in rows:
        print(fmt.format(*row))


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _cmd_list(args: argparse.Namespace) -> int:  # noqa: ARG001
    """List all registered datasets."""
    validate_registry()
    rows: list[list[str]] = []
    for did, entry in sorted(DATASET_REGISTRY.items()):
        rows.append(
            [
                did,
                entry.version,
                _truncate(entry.metadata.title, 52),
                entry.metadata.license or "-",
            ]
        )
    _print_table(["DATASET_ID", "VERSION", "TITLE", "LICENSE"], rows)
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    """Show detailed metadata for a single dataset."""
    try:
        entry: DatasetRegistryEntry = describe_dataset(args.dataset_id)
    except KeyError:
        print(f"error: unknown dataset id '{args.dataset_id}'", file=sys.stderr)
        available = ", ".join(sorted(DATASET_REGISTRY))
        print(f"available: {available}", file=sys.stderr)
        return 1

    m = entry.metadata
    lines = [
        f"Dataset:     {entry.dataset_id} v{entry.version}",
        f"Title:       {m.title}",
    ]

    if m.creators:
        for c in m.creators:
            orcid = f"  ({c.orcid})" if c.orcid else ""
            affil = f", {c.affiliation}" if c.affiliation else ""
            lines.append(f"Creator:     {c.name}{affil}{orcid}")

    if m.doi:
        lines.append(f"DOI:         https://doi.org/{m.doi}")
    if m.license:
        lines.append(f"License:     {m.license}")
    if m.publisher:
        lines.append(f"Publisher:   {m.publisher}")
    if m.publication_year:
        lines.append(f"Year:        {m.publication_year}")

    lines.append("")
    if m.description:
        lines.append(textwrap.fill(m.description, width=78))
        lines.append("")

    if m.citation:
        lines.append("Citation:")
        lines.append(textwrap.fill(f"  {m.citation}", width=78))
        lines.append("")

    lines.append(f"Artifact:    {entry.artifact.url}")
    lines.append(f"Checksum:    {entry.artifact.checksum_type}:{entry.artifact.checksum}")
    lines.append(f"Cache path:  {entry.cache_path(Path(args.cache_root))}")

    print("\n".join(lines))
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    """Download (and ingest) a dataset into the local cache."""
    from .download import DatasetDownloadError, ensure_dataset_available

    dataset_id: str = args.dataset_id
    try:
        describe_dataset(dataset_id)
    except KeyError:
        print(f"error: unknown dataset id '{dataset_id}'", file=sys.stderr)
        return 1

    cache_root = Path(args.cache_root)
    try:
        path = ensure_dataset_available(
            dataset_id,
            cache_root=cache_root,
            auto_download=True,
            force_download=bool(args.force),
            verify_checksum=True,
        )
    except DatasetDownloadError as exc:
        logger.error("download failed: %s", exc)
        return 2
    except Exception as exc:
        logger.error("unexpected error: %s", exc)
        return 2

    print(path)
    return 0


def _cmd_cache_status(args: argparse.Namespace) -> int:
    """Show cache status for all registered datasets."""
    cache_root = Path(args.cache_root)
    rows: list[list[str]] = []
    for did, entry in sorted(DATASET_REGISTRY.items()):
        cp = entry.cache_path(cache_root)
        expected = cp / entry.expected_root
        cached = "yes" if expected.exists() else "no"
        rows.append([did, cached, str(expected)])
    _print_table(["DATASET_ID", "CACHED", "PATH"], rows)
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    p = argparse.ArgumentParser(
        prog="plume-nav-data-zoo",
        description="Manage plume-nav-sim Data Zoo datasets",
    )
    p.add_argument(
        "--cache-root",
        default=str(DEFAULT_CACHE_ROOT),
        help=f"Cache directory (default: {DEFAULT_CACHE_ROOT})",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    sub = p.add_subparsers(dest="command")

    # list
    sub.add_parser("list", help="List all registered datasets")

    # describe
    desc_p = sub.add_parser("describe", help="Show detailed dataset metadata")
    desc_p.add_argument("dataset_id", help="Dataset identifier")

    # download
    dl_p = sub.add_parser("download", help="Download and ingest a dataset")
    dl_p.add_argument("dataset_id", help="Dataset identifier")
    dl_p.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached",
    )

    # cache-status
    sub.add_parser("cache-status", help="Show cache status for all datasets")

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _setup_logger(args.log_level)

    if args.command is None:
        parser.print_help()
        return 0

    dispatch = {
        "list": _cmd_list,
        "describe": _cmd_describe,
        "download": _cmd_download,
        "cache-status": _cmd_cache_status,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)
