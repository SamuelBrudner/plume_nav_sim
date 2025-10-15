import pytest

from plume_nav_sim.registration.register import (
    _validate_registration_config as validate_registration_config,
)
from plume_nav_sim.utils.exceptions import ValidationError


def _new_report(strict: bool = True):
    return {
        "timestamp": 0.0,
        "strict_validation": strict,
        "errors": [],
        "warnings": [],
        "recommendations": [],
        "performance_analysis": {},
        "compatibility_check": {},
    }


class TestValidateValueHelpers:
    def test_validate_grid_size_value_ok(self):
        report = _new_report(strict=True)
        ok = reg._validate_grid_size_value((32, 32), report, strict_validation=True)
        assert ok is True
        assert report["errors"] == []

    def test_validate_grid_size_value_type_error(self):
        report = _new_report()
        ok = reg._validate_grid_size_value("bad", report, strict_validation=True)
        assert ok is False
        assert any("grid_size must be a tuple/list" in e for e in report["errors"])

    def test_validate_grid_size_value_large_warns_when_strict(self):
        report = _new_report(strict=True)
        ok = reg._validate_grid_size_value((2048, 2048), report, strict_validation=True)
        assert ok is True
        assert any("Large grid_size" in w for w in report["warnings"])

    def test_validate_source_location_value_ok(self):
        report = _new_report()
        ok = reg._validate_source_location_value((10, 20), report)
        assert ok is True
        assert report["errors"] == []

    def test_validate_source_location_value_len_error(self):
        report = _new_report()
        ok = reg._validate_source_location_value([1], report)
        assert ok is False
        assert any(
            "source_location must be a tuple/list" in e for e in report["errors"]
        )

    def test_validate_goal_radius_value_ok(self):
        report = _new_report()
        ok = reg._validate_goal_radius_value(3, report)
        assert ok is True

    def test_validate_goal_radius_value_negative(self):
        report = _new_report()
        ok = reg._validate_goal_radius_value(-1, report)
        assert ok is False
        assert any("non-negative" in e for e in report["errors"])

    def test_validate_max_episode_steps_none_warns(self):
        report = _new_report()
        ok = reg._validate_max_episode_steps(None, report)
        assert ok is True
        assert any("not set" in w for w in report["warnings"])

    def test_validate_max_episode_steps_invalid(self):
        report = _new_report()
        assert reg._validate_max_episode_steps("100", report) is False
        assert any("integer" in e for e in report["errors"])

        report = _new_report()
        assert reg._validate_max_episode_steps(0, report) is False
        assert any("positive" in e for e in report["errors"])

        report = _new_report()
        assert reg._validate_max_episode_steps(100001, report) is False
        assert any("exceeds" in e for e in report["errors"])

        report = _new_report()
        assert reg._validate_max_episode_steps(50, report) is True
        assert any("quite low" in w for w in report["warnings"])

    def test_cross_validate_params_bounds(self):
        report = _new_report()
        reg._cross_validate_params(
            {"grid_size": (10, 10), "source_location": (11, 5)}, report
        )
        assert any("within grid_size bounds" in e for e in report["errors"])

        report = _new_report()
        reg._cross_validate_params(
            {"grid_size": (10, 10), "source_location": (2, 3)}, report
        )
        assert report["errors"] == []


class TestAssertHelpers:
    def test_assert_grid_size_or_raise(self):
        width, height = reg._assert_grid_size_or_raise((10, 12))
        assert (width, height) == (10, 12)
        with pytest.raises(ValidationError):
            reg._assert_grid_size_or_raise((10,))

    def test_assert_source_location_or_raise(self):
        x, y = reg._assert_source_location_or_raise((2, 3), 10, 10)
        assert (x, y) == (2.0, 3.0)
        with pytest.raises(ValidationError):
            reg._assert_source_location_or_raise((11, 0), 10, 10)

    def test_assert_max_steps_or_raise(self):
        assert reg._assert_max_steps_or_raise(100) == 100
        with pytest.raises(ValidationError):
            reg._assert_max_steps_or_raise(0)
        with pytest.raises(ValidationError):
            reg._assert_max_steps_or_raise(200000)

    def test_assert_goal_radius_or_raise(self):
        assert reg._assert_goal_radius_or_raise(4, 10, 10) == 4.0
        with pytest.raises(ValidationError):
            reg._assert_goal_radius_or_raise(20, 10, 10)


class TestValidateRegistrationConfig:
    def test_validate_registration_config_happy_path(self):
        is_valid, report = reg.validate_registration_config(
            env_id="CustomEnv-v0",
            entry_point="a.b:C",
            max_episode_steps=None,
            kwargs={
                "grid_size": (32, 32),
                "source_location": (10, 20),
                "goal_radius": 1.0,
            },
            strict_validation=True,
        )
        assert is_valid is True
        assert report["errors"] == []
        assert any(
            "TimeLimit" in w or "not set" in w for w in report["warnings"]
        )  # from max_episode_steps None

    def test_validate_registration_config_strict_name_warning(self):
        _, report = reg.validate_registration_config(
            env_id="gymMyEnv-v0",
            entry_point="x.y:Z",
            max_episode_steps=200,
            kwargs={
                "grid_size": (32, 32),
                "source_location": (5, 5),
                "goal_radius": 1.0,
            },
            strict_validation=True,
        )
        assert any("starting with 'gym'" in w for w in report["warnings"])
