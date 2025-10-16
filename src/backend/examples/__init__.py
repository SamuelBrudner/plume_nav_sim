"""Curated example entry points for Phase 3 documentation."""

from .custom_components import main as custom_components_main
from .custom_configuration import main as custom_configuration_main
from .quickstart import main as quickstart_main
from .reproducibility import main as reproducibility_main

__all__ = [
    "custom_components_main",
    "custom_configuration_main",
    "quickstart_main",
    "reproducibility_main",
]
