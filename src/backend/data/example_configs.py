"""
Minimal example configuration stubs used by benchmark and scenario utilities.

These stubs provide just enough structure for test collection/imports without
pulling in heavy configuration systems.
"""
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class QuickStartConfig:
    grid_size: tuple = (32, 32)
    source_location: tuple = (16, 16)
    plume_sigma: float = 12.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'grid_size': self.grid_size,
            'source_location': self.source_location,
            'plume_sigma': self.plume_sigma,
        }


class ExampleConfigCollection:
    def __init__(self) -> None:
        self._configs: List[QuickStartConfig] = []

    def add(self, cfg: QuickStartConfig) -> None:
        self._configs.append(cfg)

    def __len__(self) -> int:
        return len(self._configs)


def get_quick_start_config(*, complexity_level: str | None = None,
                           include_documentation: bool | None = None,
                           validate_configuration: bool | None = None) -> QuickStartConfig:
    return QuickStartConfig()
