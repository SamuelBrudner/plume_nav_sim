from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_RUNTIME_PACKAGE = REPO_ROOT / "src" / "backend" / "plume_nav_sim"
BACKEND_TESTS = REPO_ROOT / "src" / "backend" / "tests"


def _active_truth_surface() -> list[Path]:
    return [
        REPO_ROOT / "README.md",
        REPO_ROOT / ".github" / "workflows" / "ci-examples.yml",
        REPO_ROOT / ".github" / "workflows" / "ci-lint.yml",
        REPO_ROOT / ".github" / "workflows" / "ci-tests.yml",
        REPO_ROOT / ".github" / "workflows" / "publish.yml",
        REPO_ROOT / "src" / "backend" / "pyproject.toml",
        REPO_ROOT / "src" / "backend" / "tox.ini",
        REPO_ROOT / "src" / "backend" / ".pre-commit-config.yaml",
        REPO_ROOT / "src" / "backend" / "TESTING_GUIDE.md",
        REPO_ROOT / "src" / "backend" / "CONTRACTS.md",
        REPO_ROOT / "src" / "backend" / "SEMANTIC_MODEL.md",
        REPO_ROOT / "src" / "backend" / "docs" / "plume_types.md",
    ]


REMOVED_REFERENCES = {
    "requirements-test.txt": "backend requirements-test.txt no longer exists",
    "scripts/validate_installation.py": "validate_installation.py was archived",
    "docs/user_guide.py": "that documentation path does not exist",
    "docs/developer_guide.py": "that documentation path does not exist",
    "tree/main/examples": "examples live under src/backend/examples",
    "src/backend/tests/data_capture/": "that backend test directory does not exist",
    "make setup-dev": "the Makefile has no setup-dev target",
    "make install-qt": "the Makefile has no install-qt target",
    "make debugger": "the Makefile target is demo-debugger",
    "local_scripts/emonet_mean_intensity.py": "Emonet helper scripts live under repo-root scripts/",
    "tests/debugger/test_replay_loader_engine.py": "replay coverage lives in the current debugger test modules",
    "tests/debugger/test_replay_driver.py": "replay coverage lives in the current debugger test modules",
    "src/backend/plume_nav_sim/envs/base_env.py": "base_env.py was removed",
    "src/backend/plume_nav_sim/core/episode_manager.py": "episode_manager.py was removed",
    "src/plume_nav_sim/envs/plume_search_env.py": "plume_search_env.py was removed",
}


def test_active_docs_workflows_and_configs_avoid_known_dead_references():
    failures: list[str] = []

    for path in _active_truth_surface():
        text = path.read_text(encoding="utf-8")
        for needle, reason in REMOVED_REFERENCES.items():
            if needle in text:
                failures.append(f"{path}: found stale reference {needle!r} ({reason})")

    assert not failures, "\n".join(failures)


def test_backend_tree_has_no_orphan_scenarios_package():
    assert not (
        REPO_ROOT / "src" / "backend" / "scenarios"
    ).exists(), "Keep scenario helpers in active packages/tests, not as an orphan top-level backend package"


def test_backend_tree_has_no_envs_compat_module():
    assert not (
        REPO_ROOT / "src" / "backend" / "plume_nav_sim" / "envs" / "compat.py"
    ).exists(), "Keep the env construction surface in plume_nav_sim.envs, not a legacy compat module"


def test_runtime_package_has_no_test_named_modules():
    offenders = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in BACKEND_RUNTIME_PACKAGE.rglob("test_*.py")
    )
    assert not offenders, (
        "Runtime package modules should not look like pytest test modules: "
        + ", ".join(offenders)
    )


def test_backend_test_tree_has_no_zero_byte_test_files():
    offenders = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in BACKEND_TESTS.rglob("test_*.py")
        if path.stat().st_size == 0
    )
    assert not offenders, "Delete or implement empty backend tests: " + ", ".join(
        offenders
    )
