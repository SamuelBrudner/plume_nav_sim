"""Contract and functional tests for the configuration package exports."""

import importlib

import plume_nav_sim.config as config
import plume_nav_sim.core.types as core_types


class TestConfigurationExports:
    """Ensure the configuration package exposes canonical core types."""

    def test_environment_config_is_canonical(self) -> None:
        """The config package must expose the core EnvironmentConfig without wrapping."""
        assert config.EnvironmentConfig is core_types.EnvironmentConfig
        assert config.EnvironmentConfig.__module__ == core_types.EnvironmentConfig.__module__

    def test_fallback_controls_are_not_present(self) -> None:
        """Fail loud: the package should not offer silent fallback toggles."""
        assert not hasattr(config, "DEFAULT_CONFIG_FALLBACK_ENABLED")
        assert not hasattr(config, "CONFIGURATION_WARNINGS_ENABLED")

    def test_environment_config_exported_publicly(self) -> None:
        """Ensure __all__ contains the canonical type name exactly once."""
        exports = [name for name in getattr(config, "__all__", []) if name == "EnvironmentConfig"]
        assert exports == ["EnvironmentConfig"], "EnvironmentConfig should appear once in __all__"


class TestConfigurationFactories:
    """Functional coverage for the default configuration factory helpers."""

    def test_get_default_environment_config_returns_canonical_type(self) -> None:
        """The factory should return the canonical EnvironmentConfig instance."""
        env_config = config.get_default_environment_config()
        assert isinstance(env_config, core_types.EnvironmentConfig)

    def test_module_reload_preserves_canonical_exports(self) -> None:
        """Reloading the module must not reintroduce fallback placeholders."""
        reloaded = importlib.reload(config)
        assert reloaded.EnvironmentConfig is core_types.EnvironmentConfig
        assert not hasattr(reloaded, "DEFAULT_CONFIG_FALLBACK_ENABLED")
