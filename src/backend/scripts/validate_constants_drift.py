"""
Constants drift validator for plume_nav_sim.

Checks that YAML-configured constants (conf/constants.yaml) match the
corresponding Python constants in plume_nav_sim.core.constants and that
expected keys are present. Exits with code 1 on mismatch.

Usage:
  python -m scripts.validate_constants_drift
  or
  plume-nav-constants-check
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Tuple

import yaml

from plume_nav_sim.core import constants as C


def load_yaml() -> Dict[str, Any]:
    try:
        with C.CONFIG_PATH.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}


def compare(
    name: str, py_val: Any, yaml_val: Any, mismatches: list[Tuple[str, Any, Any]]
):
    if py_val != yaml_val:
        mismatches.append((name, py_val, yaml_val))


def main() -> int:
    data = load_yaml()
    mismatches: list[Tuple[str, Any, Any]] = []
    missing: list[str] = []

    # Expected sections/keys in YAML
    expected = {
        "package": ["name", "version", "environment_id"],
        "performance": [
            "tracking_enabled",
            "target_step_latency_ms",
            "target_rgb_render_ms",
            "target_human_render_ms",
            "target_episode_reset_ms",
            "target_plume_generation_ms",
            "boundary_enforcement_ms",
        ],
        "testing": ["default_seeds"],
    }

    for section, keys in expected.items():
        sec = data.get(section, {})
        for key in keys:
            if key not in sec:
                missing.append(f"{section}.{key}")

    # Compare mapped values
    pkg = data.get("package", {})
    compare("package.name", C.PACKAGE_NAME, pkg.get("name"), mismatches)
    compare("package.version", C.PACKAGE_VERSION, pkg.get("version"), mismatches)
    compare(
        "package.environment_id",
        C.ENVIRONMENT_ID,
        pkg.get("environment_id"),
        mismatches,
    )

    perf = data.get("performance", {})
    compare(
        "performance.tracking_enabled",
        C.PERFORMANCE_TRACKING_ENABLED,
        perf.get("tracking_enabled"),
        mismatches,
    )
    compare(
        "performance.target_step_latency_ms",
        C.PERFORMANCE_TARGET_STEP_LATENCY_MS,
        perf.get("target_step_latency_ms"),
        mismatches,
    )
    compare(
        "performance.target_rgb_render_ms",
        C.PERFORMANCE_TARGET_RGB_RENDER_MS,
        perf.get("target_rgb_render_ms"),
        mismatches,
    )
    compare(
        "performance.target_human_render_ms",
        C.PERFORMANCE_TARGET_HUMAN_RENDER_MS,
        perf.get("target_human_render_ms"),
        mismatches,
    )
    compare(
        "performance.target_episode_reset_ms",
        C.PERFORMANCE_TARGET_EPISODE_RESET_MS,
        perf.get("target_episode_reset_ms"),
        mismatches,
    )
    compare(
        "performance.target_plume_generation_ms",
        C.PERFORMANCE_TARGET_PLUME_GENERATION_MS,
        perf.get("target_plume_generation_ms"),
        mismatches,
    )
    compare(
        "performance.boundary_enforcement_ms",
        C.BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS,
        perf.get("boundary_enforcement_ms"),
        mismatches,
    )

    test = data.get("testing", {})
    compare(
        "testing.default_seeds",
        list(C.DEFAULT_TEST_SEEDS),
        list(test.get("default_seeds", []) or []),
        mismatches,
    )

    ok = True
    if missing:
        print("Missing YAML keys:")
        for k in missing:
            print(f"  - {k}")
        ok = False

    if mismatches:
        print("Constant drift detected (Python vs YAML):")
        for name, py_val, y_val in mismatches:
            print(f"  - {name}: python={py_val!r} yaml={y_val!r}")
        ok = False

    # Warn on extra YAML keys not used by code (best-effort)
    extra_keys: list[str] = []

    def collect_extras(section: str, allowed: list[str]):
        sec = data.get(section, {})
        for k in sec.keys():
            if k not in allowed:
                extra_keys.append(f"{section}.{k}")

    for section, keys in expected.items():
        collect_extras(section, keys)

    if extra_keys:
        print("Notice: Extra YAML keys present (not mapped in code):")
        for k in extra_keys:
            print(f"  - {k}")

    if ok:
        print("Constants are in sync.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
