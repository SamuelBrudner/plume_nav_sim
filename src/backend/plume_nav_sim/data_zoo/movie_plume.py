"""Convenience wrapper for registry-backed movie plume datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .download import ensure_dataset_available
from .loader import load_plume
from .registry import DEFAULT_CACHE_ROOT
from .stats import NormalizationMethod


@dataclass
class MoviePlume:
    """Registry-backed movie plume descriptor with normalized loading support."""

    dataset_id: str
    normalize: NormalizationMethod | str | None = None
    cache_root: Path = DEFAULT_CACHE_ROOT
    auto_download: bool = False
    chunks: Any = "auto"
    data_array: Any | None = None
    dataset_path: Path | None = None

    @classmethod
    def from_registry(
        cls,
        dataset_id: str,
        *,
        normalize: NormalizationMethod | str | None = None,
        cache_root: Path = DEFAULT_CACHE_ROOT,
        auto_download: bool = False,
        chunks: Any = "auto",
    ) -> MoviePlume:
        """Load a registry dataset (optionally normalized) and return a descriptor."""

        cache_root = Path(cache_root)
        dataset_path = ensure_dataset_available(
            dataset_id,
            cache_root=cache_root,
            auto_download=auto_download,
        )
        data_array = load_plume(
            dataset_id,
            normalize=normalize,
            cache_root=cache_root,
            auto_download=auto_download,
            chunks=chunks,
        )
        return cls(
            dataset_id=dataset_id,
            normalize=normalize,
            cache_root=cache_root,
            auto_download=auto_download,
            chunks=chunks,
            data_array=data_array,
            dataset_path=Path(dataset_path),
        )

    def load(self):
        """Ensure the underlying DataArray is loaded (lazy/dask-backed by default)."""

        if self.data_array is None:
            self.data_array = load_plume(
                self.dataset_id,
                normalize=self.normalize,
                cache_root=self.cache_root,
                auto_download=self.auto_download,
                chunks=self.chunks,
            )
        return self.data_array

    def env_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Build kwargs for ``plume_nav_sim.make_env`` using this dataset."""

        kwargs: dict[str, Any] = {
            "plume": "movie",
            "movie_dataset_id": self.dataset_id,
            "movie_auto_download": bool(self.auto_download),
            "movie_cache_root": self.cache_root,
            "movie_path": str(self.dataset_path) if self.dataset_path else None,
            "movie_normalize": self.normalize,
            "movie_chunks": self.chunks,
        }
        if self.data_array is not None:
            kwargs["movie_data"] = self.data_array
        kwargs.update(overrides)
        return kwargs

    def make_env(self, **overrides: Any):
        """Create an environment using the cached loader output."""

        import plume_nav_sim as pns

        return pns.make_env(**self.env_kwargs(**overrides))


__all__ = ["MoviePlume"]
