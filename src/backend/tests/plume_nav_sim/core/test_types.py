"""Contractual and functional tests for the canonical core type exports."""

import importlib

import pytest

core_types = importlib.import_module("plume_nav_sim.core.types")
geometry = importlib.import_module("plume_nav_sim.core.geometry")
state = importlib.import_module("plume_nav_sim.core.state")


class TestCoreTypeContracts:
    """Verify that the public type aliases point to the canonical implementations."""

    def test_coordinates_alias(self):
        assert core_types.Coordinates is geometry.Coordinates

    def test_grid_size_alias(self):
        assert core_types.GridSize is geometry.GridSize

    def test_agent_state_alias(self):
        assert core_types.AgentState is state.AgentState

    def test_all_contains_expected_exports(self):
        expected_names = {
            "Action",
            "RenderMode",
            "Coordinates",
            "GridSize",
            "AgentState",
            "create_coordinates",
            "create_grid_size",
            "validate_action",
            "get_movement_vector",
        }
        assert expected_names.issubset(set(core_types.__all__))


class TestCoreTypeFactories:
    """Functional expectations for the helper factories defined in core.types."""

    def test_create_coordinates_from_tuple(self):
        coord = core_types.create_coordinates((3, 4))
        assert isinstance(coord, geometry.Coordinates)
        assert coord.to_tuple() == (3, 4)

    def test_create_coordinates_invalid_sequence_raises(self):
        with pytest.raises(TypeError):
            core_types.create_coordinates((1,))

    def test_create_grid_size_from_tuple(self):
        grid = core_types.create_grid_size((5, 6))
        assert isinstance(grid, geometry.GridSize)
        assert grid.to_tuple() == (5, 6)
