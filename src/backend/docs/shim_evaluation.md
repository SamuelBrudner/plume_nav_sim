# Shim Evaluation (plume_nav_sim-84)

Date: 2026-02-08

## Existing shim-like modules
- `src/backend/vendor/__init__.py`: namespace docstring only; no `vendor.psutil` or `vendor.gymnasium_vendored` module files exist.
- `src/backend/tests/conftest.py`: test-only runtime shim for `gymnasium.utils.env_checker` via `sys.modules`; optional `psutil` guard (`_require_psutil`, `MemoryMonitor`).

## Import search results
- `gymnasium_vendored` / `vendor.gymnasium_vendored`:
  - No code imports. Mentions only in docs/comments:
  - `src/backend/vendor/__init__.py`
  - `src/backend/CONTRIBUTING.md`
- `vendor.psutil`:
  - No code imports. Mentioned only in docs/comments above.
- `psutil`:
  - Test-only imports in `src/backend/tests/conftest.py` and archived tests (`src/backend/tests/archived/test_episode_manager_legacy_core.py`).
  - No imports in `src/backend/plume_nav_sim/` production package.

## Production vs test usage
- Production code (`src/backend/plume_nav_sim/`): no vendored shim usage for gymnasium/psutil.
- Tests: optional `psutil` and env-checker shim are already located in `tests/conftest.py`.

## Recommendation
- `gymnasium_vendored`: remove stale documentation references unless a real shim is reintroduced.
- `psutil` shim: keep current test-only pattern (optional import + guard in `tests/conftest.py`); no relocation needed.
- Optional extras/import guards for production shims: not needed now, because no production shim modules exist.
