# Contributing to plume-nav-sim (backend)

Status: Alpha.

This repository is still in an early stage. The process below is intentionally pragmatic: enough rigor to keep behavior stable and reproducible, without heavyweight ceremony.

## Goals for Contributions

- Keep the environment Gymnasium-compatible.
- Keep results reproducible across runs when seeds are fixed.
- Keep components swappable (`action_model`, `sensor_model`, `reward_fn`, `plume`).
- Keep diffs small and reviewable.

## Quick Start

From repository root:

```bash
cd src/backend
python scripts/setup_dev_env.py --verbose
source plume-nav-env/bin/activate
pre-commit run --all-files
pytest -q
```

If all commands pass, you are ready to contribute.

## Prerequisites

- Python 3.10+
- macOS or Linux recommended
- Git

Windows is accepted on a best-effort basis.

## Development Environment

### Recommended Setup Script

```bash
cd src/backend
python scripts/setup_dev_env.py --verbose
source plume-nav-env/bin/activate
```

The setup script installs development dependencies and prepares local tooling.

### Manual Setup

```bash
cd src/backend
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pre-commit install --install-hooks
```

### Validate Setup

```bash
cd src/backend
python -c "import gymnasium as gym; import plume_nav_sim; env = gym.make('PlumeNav-StaticGaussian-v0'); obs, info = env.reset(seed=0); print(type(obs), type(info))"
pytest -q
```

## Repository Layout (backend)

- `src/backend/plume_nav_sim/`: installable package
- `src/backend/tests/`: test suite
- `src/backend/conf/`: experiment/config files
- `src/backend/contracts/`: concise design contracts and invariants
- `src/backend/docs/`: user and developer guides
- `src/backend/scripts/`: utility scripts

## Work Tracking (beads)

This project uses `bd` for issue tracking.

Common commands:

```bash
bd ready
bd show <id>
bd update <id> --status in_progress
bd close <id>
bd sync
```

Guidelines:

- Claim one bead at a time unless coordinated.
- Keep scope aligned with bead title/description.
- File follow-up beads instead of inflating one PR.

## Branching and Commit Conventions

Suggested branch naming:

- `bead/<id>-<short-slug>`

Commit guidance:

- Prefer small logical commits.
- Use imperative subject lines.
- Include context in the body when behavior changes.

Examples:

```text
trim backend contract docs for alpha stage
fix action validation for numpy integer actions
add regression test for step-before-reset state error
```

## Typical Contribution Workflow

1. Read bead details and existing tests.
2. Add or update tests first when changing behavior.
3. Implement the smallest change that satisfies the bead.
4. Run quality gates locally.
5. Update docs/contracts if semantics changed.
6. Open PR with concise rationale and test evidence.

## Quality Gates

Run before opening or updating a PR:

```bash
cd src/backend
pre-commit run --all-files
pytest -q
```

Useful variants:

```bash
pytest -q -m "not slow"
pytest -q -m slow
pytest -q tests/contracts
```

If you modify performance-sensitive code, include before/after measurements or a brief benchmark note.

## Testing Expectations

Focus on behavior and invariants, not implementation details.

Add tests for:

- New public behavior.
- Bug regressions.
- Boundary conditions and error paths.
- Determinism when seed behavior is relevant.

Avoid:

- Over-mocking core environment behavior.
- Snapshotting large structures that are likely to churn without value.

## Coding Conventions

- Prefer explicit types on public interfaces.
- Keep functions small and single-purpose.
- Avoid hidden global state.
- Fail fast on invalid input.

Coordinate conventions:

- `Coordinates` are `(x, y)`.
- NumPy grid indexing is `[y, x]`.
- Use helper methods when converting between coordinate and array indices.

## Determinism and Reproducibility

Deterministic behavior is a first-class requirement.

Rules:

- Seed entry point is `env.reset(seed=...)`.
- Component randomness must be controlled by explicit RNG/seed plumbing.
- Do not use uncontrolled module-level random state in core logic.

When relevant, add tests that verify fixed-seed reproducibility.

## Component Extension Guidelines

Most extensibility work targets one of these protocols:

- `plume_nav_sim.interfaces.action.ActionProcessor`
- `plume_nav_sim.interfaces.observation.ObservationModel`
- `plume_nav_sim.interfaces.reward.RewardFunction`

When implementing new components:

- Keep spaces stable for each instance.
- Ensure returned values are contained by declared spaces.
- Keep `get_metadata()` JSON-serializable.
- Document required `env_state` keys for observation models.

## Contracts and Documentation

If behavior/invariants change, update docs in the same PR:

- Relevant file in `src/backend/contracts/`
- User-facing docs in `src/backend/docs/` or `src/backend/README.md`
- `src/backend/CHANGELOG.md` for user-visible changes

Contract docs should stay concise: define intent, invariants, and edge-case behavior. Avoid long tutorials and redundant examples.

## Pull Request Expectations

A good PR includes:

- What changed
- Why it changed
- How it was validated
- Any migration or compatibility notes

Suggested PR checklist:

- [ ] Scope matches bead
- [ ] Tests added/updated where needed
- [ ] `pre-commit run --all-files` passes
- [ ] `pytest -q` passes
- [ ] Contracts/docs updated when semantics changed
- [ ] Changelog updated if user-visible

## Review Criteria

Reviewers prioritize:

- Correctness and invariants
- Regression risk
- Test quality
- API clarity
- Determinism/reproducibility implications

Large PRs are harder to review and slower to merge. Split independent changes.

## Bug Reports

Please include:

- Environment/version info
- Minimal reproduction steps
- Seed used
- Expected behavior
- Actual behavior
- Relevant logs/tracebacks

Minimal template:

```text
Environment: plume-nav-sim backend <commit or version>
Seed: <value>
Repro steps:
1. ...
2. ...
Expected:
Actual:
Traceback/logs:
```

## Performance Changes

When changing core loops or heavy array operations:

- Mention complexity or allocation impact.
- Provide a small benchmark or timing note.
- Prefer vectorized NumPy paths in hotspots.

## Compatibility Notes (alpha)

Because the project is alpha-stage:

- Some APIs may change rapidly.
- We still try to avoid unnecessary breakage.
- Any intentional behavior change should be explicit in PR description and changelog.

## Security and Safety

- Do not commit secrets, tokens, or private keys.
- Keep external downloads/checksums explicit and validated.
- Avoid unsafe archive extraction patterns.

## Getting Help

If blocked:

- Comment on the bead with concrete blocker details.
- Link code locations and failed commands.
- Propose one or two actionable paths forward.

Clear, specific unblock notes are more useful than long status text.

## Final Notes

The fastest way to contribute effectively in this stage:

- Keep scope tight.
- Write or update tests with each semantic change.
- Keep contracts concise and aligned with code.
- Prefer clarity over cleverness.
