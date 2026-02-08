from __future__ import annotations

from pathlib import Path

import pytest

from plume_nav_sim.cli import capture


def test_locate_hydra_config_path_uses_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "conf"
    assert capture._locate_hydra_config_path(str(explicit)) == explicit


def test_locate_hydra_config_path_uses_resolver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved = tmp_path / "resolved-conf"
    monkeypatch.setattr(capture, "_resolve_config_dir", lambda _: str(resolved))
    assert capture._locate_hydra_config_path(None) == resolved


def test_build_hydra_overrides_returns_empty_for_none() -> None:
    assert capture._build_hydra_overrides(None) == []


def test_build_hydra_overrides_copies_input_list() -> None:
    overrides = ["seed=42", "env.plume=movie"]
    built = capture._build_hydra_overrides(overrides)
    assert built == overrides
    assert built is not overrides


def test_map_hydra_cfg_to_run_config_accepts_mapping() -> None:
    raw = {"output": "results", "env": {"grid_size": [64, 64]}}
    mapped = capture._map_hydra_cfg_to_run_config(raw, cfg_is_mapping=True)
    assert mapped == raw
    assert mapped is not raw


def test_map_hydra_cfg_to_run_config_uses_omegaconf_for_dictconfig(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOmegaConf:
        @staticmethod
        def to_container(cfg: object, resolve: bool = False) -> dict[str, object]:
            assert cfg == "dictconfig"
            assert resolve is True
            return {"experiment": "default"}

    monkeypatch.setattr(capture, "_require_omegaconf", lambda: FakeOmegaConf)
    mapped = capture._map_hydra_cfg_to_run_config("dictconfig")
    assert mapped == {"experiment": "default"}


def test_map_hydra_cfg_to_run_config_rejects_non_mapping() -> None:
    with pytest.raises(SystemExit, match="must resolve to a mapping"):
        capture._map_hydra_cfg_to_run_config(["not", "a", "mapping"], cfg_is_mapping=True)
