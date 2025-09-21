"""Lightweight configuration helpers for the trimmed plume_nav_sim test suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

REPRODUCIBILITY_SEEDS: Iterable[int] = (0, 1, 2, 3)


def _base_config(**overrides: Any) -> Dict[str, Any]:
    config = {
        "grid_size": (16, 16),
        "source_location": (8, 8),
        "max_steps": 100,
        "goal_radius": 0.5,
        "render_mode": "rgb_array",
    }
    config.update(overrides)
    return config


def create_unit_test_config(**overrides: Any) -> Dict[str, Any]:
    return _base_config(test_profile="unit", **overrides)


def create_integration_test_config(**overrides: Any) -> Dict[str, Any]:
    return _base_config(test_profile="integration", **overrides)


def create_performance_test_config(**overrides: Any) -> Dict[str, Any]:
    return _base_config(test_profile="performance", **overrides)


def create_reproducibility_test_config(**overrides: Any) -> Dict[str, Any]:
    seeds = tuple(overrides.pop("seeds", REPRODUCIBILITY_SEEDS))
    config = _base_config(test_profile="reproducibility", **overrides)
    config["seeds"] = seeds
    return config


def create_edge_case_test_config(**overrides: Any) -> Dict[str, Any]:
    return _base_config(test_profile="edge_case", **overrides)


@dataclass
class TestConfigFactory:
    """Simplified stand-in for the production configuration factory."""

    auto_optimize: bool = False
    _system_capabilities: Dict[str, Any] = None
    _auto_optimize: bool = False
    _configuration_cache: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self._system_capabilities is None:
            self._system_capabilities = {}
        if self._configuration_cache is None:
            self._configuration_cache = {}
        self._auto_optimize = self.auto_optimize

    def detect_system_capabilities(self, force_refresh: bool = False) -> Dict[str, Any]:
        if force_refresh or not self._system_capabilities:
            self._system_capabilities = {"memory_gb": 4, "cpu_count": 2}
        return self._system_capabilities

    def create_config(self, profile: str, **overrides: Any) -> Dict[str, Any]:
        factory_map = {
            "unit": create_unit_test_config,
            "integration": create_integration_test_config,
            "performance": create_performance_test_config,
            "reproducibility": create_reproducibility_test_config,
            "edge_case": create_edge_case_test_config,
        }
        try:
            builder = factory_map[profile]
        except KeyError as exc:
            raise ValueError(f"Unknown test configuration profile: {profile}") from exc
        return builder(**overrides)
