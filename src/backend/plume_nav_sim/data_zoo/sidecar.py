from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .registry import DatasetRegistryEntry

LOG = logging.getLogger("plume_nav_sim.data_zoo.download")


def _format_yaml_string(value: str) -> str:
    if "\n" in value or ":" in value or '"' in value:
        return f'"{value}"'
    return value


def _format_yaml_list(values: list[Any], indent: int) -> str:
    if not values:
        return "[]"
    prefix = "  " * indent
    return "".join(
        f"\n{prefix}  - {_format_yaml_value(item, indent + 1)}" for item in values
    )


def _format_yaml_dict(values: dict[str, Any], indent: int) -> str:
    if not values:
        return "{}"
    prefix = "  " * indent
    return "".join(
        f"\n{prefix}  {key}: {_format_yaml_value(val, indent + 1)}"
        for key, val in values.items()
    )


def _format_yaml_value(value: Any, indent: int = 0) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return _format_yaml_string(value)
    if isinstance(value, list):
        return _format_yaml_list(value, indent)
    if isinstance(value, dict):
        return _format_yaml_dict(value, indent)
    return str(value)


try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore[misc,assignment]

    def Field(**kwargs: Any) -> Any:  # type: ignore[misc]  # noqa: N802
        """Stub for pydantic.Field when pydantic is not available."""
        return kwargs.get("default")


class ProvenanceSidecar(BaseModel if PYDANTIC_AVAILABLE else object):  # type: ignore[misc]
    # Dataset identification
    dataset_id: str
    version: str
    processed_at: str  # ISO 8601 timestamp

    # Processing info
    processor: str = "plume_nav_sim"
    processor_version: str

    # Source information
    source_url: str
    source_doi: Optional[str] = None
    source_filename: str
    source_checksum_md5: str
    source_size_bytes: Optional[int] = None

    # Ingest parameters (serialized from spec)
    ingest_spec_type: str
    ingest_params: Dict[str, Any] = Field(default_factory=dict)

    # Output information
    output_format: str
    output_shape: List[int] = Field(default_factory=list)
    output_dtype: str = "float32"
    output_checksum_md5: Optional[str] = None

    # Citation
    citation: Optional[str] = None
    citation_doi: Optional[str] = None

    if PYDANTIC_AVAILABLE:
        model_config = {"extra": "forbid"}

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON format."""
        if PYDANTIC_AVAILABLE:
            return self.model_dump_json(indent=indent)
        return json.dumps(self.__dict__, indent=indent)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        if PYDANTIC_AVAILABLE:
            return self.model_dump()
        return dict(self.__dict__)

    def write(self, output_path: Path, format: str = "json") -> Path:
        sidecar_path = output_path.parent / f"{output_path.name}.provenance.{format}"

        if format == "json":
            content = self.to_json(indent=2)
        else:
            # Simple YAML-like output for human readability
            content = self._to_yaml()

        sidecar_path.write_text(content)
        LOG.info("Wrote provenance sidecar: %s", sidecar_path)
        return sidecar_path

    def _to_yaml(self) -> str:
        """Simple YAML serialization without pyyaml dependency."""
        lines = ["# Provenance sidecar for plume_nav_sim Data Zoo", ""]
        data = self.to_dict()

        for key, value in data.items():
            formatted = _format_yaml_value(value)
            lines.append(
                f"{key}:{formatted}" if "\n" in formatted else f"{key}: {formatted}"
            )

        return "\n".join(lines) + "\n"

    @classmethod
    def json_schema(cls) -> Dict[str, Any]:
        """Get JSON Schema for validation."""
        if PYDANTIC_AVAILABLE:
            return cls.model_json_schema()
        return {
            "type": "object",
            "description": "Provenance sidecar (schema unavailable)",
        }


def _get_version() -> str:
    """Get plume_nav_sim version string."""
    try:
        from plume_nav_sim import __version__

        return __version__
    except (ImportError, AttributeError):
        return "unknown"


def _compute_zarr_checksum(zarr_path: Path) -> str:
    """Compute a checksum for a Zarr directory by hashing .zarray and .zattrs files."""
    hasher = hashlib.md5(usedforsecurity=False)
    # Hash metadata files in sorted order for reproducibility
    for meta_file in sorted(zarr_path.rglob(".z*")):
        if meta_file.is_file():
            hasher.update(meta_file.read_bytes())
    return hasher.hexdigest()


def generate_provenance(
    entry: DatasetRegistryEntry,
    source_path: Path,
    output_path: Path,
    output_shape: List[int],
    output_dtype: str = "float32",
) -> ProvenanceSidecar:
    # Get source file size
    source_size = source_path.stat().st_size if source_path.exists() else None

    # Serialize ingest params
    ingest_params: Dict[str, Any] = {}
    ingest_type = "none"
    if entry.ingest:
        ingest_type = type(entry.ingest).__name__
        ingest_params = asdict(entry.ingest)

    sidecar = ProvenanceSidecar(
        dataset_id=entry.dataset_id,
        version=entry.version,
        processed_at=datetime.now(timezone.utc).isoformat(),
        processor="plume_nav_sim",
        processor_version=_get_version(),
        source_url=entry.artifact.url,
        source_doi=entry.metadata.doi if entry.metadata.doi else None,
        source_filename=source_path.name,
        source_checksum_md5=entry.artifact.checksum,
        source_size_bytes=source_size,
        ingest_spec_type=ingest_type,
        ingest_params=ingest_params,
        output_format=(
            entry.ingest.output_layout if entry.ingest else entry.artifact.layout
        ),
        output_shape=output_shape,
        output_dtype=output_dtype,
        output_checksum_md5=(
            _compute_zarr_checksum(output_path) if output_path.is_dir() else None
        ),
        citation=entry.metadata.citation if entry.metadata else None,
        citation_doi=entry.metadata.doi if entry.metadata else None,
    )

    # Write sidecar (JSON preferred for Pydantic schema compatibility)
    sidecar.write(output_path, format="json")

    return sidecar


def _generate_provenance_for_zarr(
    entry: DatasetRegistryEntry, source_path: Path, output_path: Path
) -> None:
    """Generate provenance sidecar for an ingested Zarr dataset."""
    try:
        import zarr

        store = zarr.open(str(output_path), mode="r")
        # Get shape from the main data array (concentration or similar)
        if "concentration" in store:
            shape = list(store["concentration"].shape)
            dtype = str(store["concentration"].dtype)
        else:
            # Fallback: use first array found
            for key in store.keys():
                arr = store[key]
                if hasattr(arr, "shape") and len(arr.shape) >= 2:
                    shape = list(arr.shape)
                    dtype = str(arr.dtype)
                    break
            else:
                shape = []
                dtype = "unknown"

        generate_provenance(entry, source_path, output_path, shape, dtype)

    except Exception as exc:
        # Don't fail the ingest if provenance generation fails
        LOG.warning("Failed to generate provenance sidecar: %s", exc)

