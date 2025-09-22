"""Contractual and functional tests for the canonical core type exports."""

import importlib

import pytest

core_types = importlib.import_module("plume_nav_sim.core.types")
geometry = importlib.import_module("plume_nav_sim.core.geometry")
state = importlib.import_module("plume_nav_sim.core.state")
models = importlib.import_module("plume_nav_sim.core.models")


class TestCoreTypeContracts:
    """Verify that the public type aliases point to the canonical implementations."""

    def test_coordinates_alias(self):
        assert core_types.Coordinates is geometry.Coordinates

    def test_grid_size_alias(self):
        assert core_types.GridSize is geometry.GridSize

    def test_agent_state_alias(self):
        assert core_types.AgentState is state.AgentState

    def test_episode_state_alias(self):
        assert core_types.EpisodeState is state.EpisodeState

    def test_plume_parameters_alias(self):
        assert core_types.PlumeParameters is models.PlumeModel

    def test_all_contains_expected_exports(self):
        expected_names = {
            "Action",
            "RenderMode",
            "Coordinates",
            "GridSize",
            "AgentState",
            "EpisodeState",
            "PlumeParameters",
            "EnvironmentConfig",
            "create_coordinates",
            "create_grid_size",
            "create_agent_state",
        }
        assert expected_names.issubset(set(core_types.__all__))

    def test_validation_error_alias(self):
        utils_exceptions = importlib.import_module("plume_nav_sim.utils.exceptions")
        assert core_types.ValidationError is utils_exceptions.ValidationError


class TestCoreTypeFactories:
    """Functional expectations for the helper factories defined in core.types."""

    def test_create_coordinates_from_tuple(self):
        coord = core_types.create_coordinates((3, 4))
        assert isinstance(coord, geometry.Coordinates)
        assert coord.to_tuple() == (3, 4)

    def test_create_coordinates_invalid_sequence_raises(self):
        with pytest.raises(core_types.ValidationError):
            core_types.create_coordinates((1,))

    def test_create_grid_size_from_tuple(self):
        grid = core_types.create_grid_size((5, 6))
        assert isinstance(grid, geometry.GridSize)
        assert grid.to_tuple() == (5, 6)

    def test_create_agent_state_from_tuple(self):
        agent_state = core_types.create_agent_state((2, 2))
        assert isinstance(agent_state, state.AgentState)
        assert agent_state.position.to_tuple() == (2, 2)


class TestEnvironmentConfigContract:
    """Ensure EnvironmentConfig validates and normalizes its inputs."""

    def test_environment_config_normalizes_inputs(self):
        config = core_types.EnvironmentConfig(
            grid_size=(7, 8),
            source_location=(1, 2),
            max_steps=100,
            goal_radius=0.5,
        )
        assert isinstance(config.grid_size, geometry.GridSize)
        assert isinstance(config.source_location, geometry.Coordinates)
        assert config.grid_size.to_tuple() == (7, 8)
        assert config.source_location.to_tuple() == (1, 2)

    def test_environment_config_validation(self):
        with pytest.raises(core_types.ValidationError):
            core_types.EnvironmentConfig(
                grid_size=(5, 5),
                source_location=(10, 10),
                max_steps=-1,
            )
