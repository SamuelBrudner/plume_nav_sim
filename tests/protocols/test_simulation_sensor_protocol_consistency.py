"""Tests for sensor protocol consistency in simulation module."""

from plume_nav_sim.core import simulation
from plume_nav_sim.protocols.sensor import SensorProtocol


def test_simulation_uses_main_sensor_protocol():
    """Simulation should reference the central SensorProtocol definition."""
    assert simulation.SensorProtocol is SensorProtocol
