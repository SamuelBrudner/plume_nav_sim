from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, model_validator

try:  # pragma: no cover - exercised when PyYAML is unavailable
    import yaml
except Exception:  # pragma: no cover - exercised when PyYAML is unavailable
    yaml = None


class MovieMetadataSidecar(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(default=1)
    path: Optional[str] = None
    fps: PositiveFloat
    spatial_unit: str
    pixels_per_unit: Optional[Tuple[float, float]] = None
    h5_dataset: Optional[str] = None

    @model_validator(mode="after")
    def _validate_spatial(self) -> "MovieMetadataSidecar":
        unit = self.spatial_unit
        ppu = self.pixels_per_unit

        if unit == "pixel":
            if ppu is not None:
                raise ValueError(
                    "pixels_per_unit must be omitted when spatial_unit='pixel'"
                )
        else:
            if ppu is None:
                raise ValueError(
                    "pixels_per_unit must be provided when spatial_unit is not 'pixel'"
                )
            py, px = ppu
            if py <= 0.0 or px <= 0.0:
                raise ValueError("pixels_per_unit entries must be positive")

        return self


def get_default_sidecar_path(movie_path: Path | str) -> Path:
    path = Path(movie_path)
    return path.with_suffix(path.suffix + ".plume-movie.yaml")


def load_movie_sidecar(movie_path: Path | str) -> MovieMetadataSidecar:
    path = Path(movie_path)
    sidecar_path = get_default_sidecar_path(path)

    if not sidecar_path.exists():
        raise FileNotFoundError(f"movie metadata sidecar not found: {sidecar_path}")

    if yaml is None:
        raise ImportError(
            "PyYAML is required to read movie metadata sidecar files. Install with 'pip install pyyaml'."
        )

    raw = sidecar_path.read_text()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(
            f"movie metadata sidecar must be a mapping, got {type(data).__name__}"
        )

    sidecar = MovieMetadataSidecar(**data)

    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"} and sidecar.h5_dataset is None:
        raise ValueError(
            "h5_dataset must be provided in movie sidecar for HDF5 sources"
        )
    if suffix not in {".h5", ".hdf5"} and sidecar.h5_dataset is not None:
        raise ValueError(
            "h5_dataset must not be set in movie sidecar for non-HDF5 sources"
        )

    return sidecar


__all__ = [
    "MovieMetadataSidecar",
    "get_default_sidecar_path",
    "load_movie_sidecar",
]
