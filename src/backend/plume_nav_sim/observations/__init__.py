"""
Observation model implementations for plume navigation simulation.

This module provides concrete implementations of the ObservationModel protocol,
enabling diverse sensor configurations without environment modification.

Contract: src/backend/contracts/observation_model_interface.md

Available Sensors:
- ConcentrationSensor: Single odor sensor at agent position
- AntennaeArraySensor: Multiple sensors with orientation-relative positioning
"""

from .antennae_array import AntennaeArraySensor
from .concentration import ConcentrationSensor
from .wind import WindVectorSensor

__all__ = [
    "ConcentrationSensor",
    "AntennaeArraySensor",
    "WindVectorSensor",
]

__version__ = "1.0.0"
