from plume_nav_sim.registration import register as reg


def test_env_id_suffix_required():
    is_valid, report = reg.validate_registration_config(
        env_id="CustomEnv",
        entry_point="m.n:C",
        max_episode_steps=100,
        kwargs={
            "grid_size": (16, 16),
            "source_location": (1, 1),
            "goal_radius": 1.0,
        },
        strict_validation=True,
    )
    assert is_valid is False
    assert any("must end with '-v0'" in e for e in report["errors"])


def test_entry_point_missing_colon_error():
    is_valid, report = reg.validate_registration_config(
        env_id="CustomEnv-v0",
        entry_point="module_without_colon",
        max_episode_steps=100,
        kwargs={"grid_size": (16, 16), "source_location": (1, 1), "goal_radius": 1.0},
        strict_validation=True,
    )
    assert is_valid is False
    assert any("must contain ':'" in e for e in report["errors"])


def test_entry_point_invalid_identifier_warning_strict():
    is_valid, report = reg.validate_registration_config(
        env_id="AnotherEnv-v0",
        entry_point="mod:123Bad",
        max_episode_steps=100,
        kwargs={"grid_size": (16, 16), "source_location": (1, 1), "goal_radius": 1.0},
        strict_validation=True,
    )
    # Not necessarily invalid overall, but should warn about identifier
    assert any("class name may not be valid" in w for w in report["warnings"])


def test_performance_analysis_present_when_grid_size_provided():
    is_valid, report = reg.validate_registration_config(
        env_id="PerfEnv-v0",
        entry_point="m.n:C",
        max_episode_steps=None,
        kwargs={"grid_size": (32, 64), "source_location": (10, 20), "goal_radius": 1.0},
        strict_validation=False,
    )
    assert "grid_cells" in report["performance_analysis"]


def test_missing_params_recommendation_in_strict():
    is_valid, report = reg.validate_registration_config(
        env_id="RecsEnv-v0",
        entry_point="m.n:C",
        max_episode_steps=200,
        kwargs={"grid_size": (16, 16)},
        strict_validation=True,
    )
    assert any("Consider specifying parameters" in r for r in report["recommendations"])


def test_cross_validate_source_location_error():
    is_valid, report = reg.validate_registration_config(
        env_id="CrossEnv-v0",
        entry_point="m.n:C",
        max_episode_steps=200,
        kwargs={"grid_size": (10, 10), "source_location": (10, 9), "goal_radius": 1.0},
        strict_validation=True,
    )
    assert any("within grid_size bounds" in e for e in report["errors"])


def test_goal_radius_edge_warning():
    width, height = 10, 10
    sx, sy = 1, 1
    mde = min(sx, sy, width - sx - 1, height - sy - 1)
    radius = float(mde + 1)

    is_valid, report = reg.validate_registration_config(
        env_id="EdgeEnv-v0",
        entry_point="m.n:C",
        max_episode_steps=200,
        kwargs={
            "grid_size": (width, height),
            "source_location": (sx, sy),
            "goal_radius": radius,
        },
        strict_validation=True,
    )
    assert any(
        "Goal radius extends beyond grid boundaries" in w for w in report["warnings"]
    )


def test_integration_valid_config_true():
    is_valid, report = reg.validate_registration_config(
        env_id="ValidEnv-v0",
        entry_point="a.b:C",
        max_episode_steps=200,
        kwargs={"grid_size": (32, 32), "source_location": (10, 20), "goal_radius": 1.0},
        strict_validation=True,
    )
    assert is_valid is True
    assert report["errors"] == []


def test_integration_invalid_max_episode_steps():
    is_valid, report = reg.validate_registration_config(
        env_id="MaxEnv-v0",
        entry_point="a.b:C",
        max_episode_steps="100",  # invalid type
        kwargs={"grid_size": (32, 32), "source_location": (10, 20), "goal_radius": 1.0},
        strict_validation=True,
    )
    assert is_valid is False
    assert any("must be an integer" in e for e in report["errors"])
