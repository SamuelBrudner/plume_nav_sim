"""Config package entrypoint.

Run as a module to interact with the configuration system without executing
package internals directly:

    python -m config --status
    python -m config --dump-default

This ensures relative imports in the package resolve correctly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from typing import Any


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        except Exception:
            pass
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m config",
        description=(
            "Package-safe entrypoint for the plume_nav_sim configuration system."
        ),
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print configuration system status as JSON",
    )
    parser.add_argument(
        "--dump-default",
        action="store_true",
        help="Print the default CompleteConfig as JSON",
    )

    args = parser.parse_args(argv)

    if args.dump_default:
        from . import get_complete_default_config

        cfg = get_complete_default_config()
        print(json.dumps(_to_jsonable(cfg), indent=2, sort_keys=True))
        return 0

    # Default action (or --status): print system status
    from . import get_configuration_system_status

    status = get_configuration_system_status()
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - convenience runtime entry
    raise SystemExit(main())
