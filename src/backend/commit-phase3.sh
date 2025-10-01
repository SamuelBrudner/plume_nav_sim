#!/bin/bash
# Phase 3 Commit Script - Run from src/backend directory

cd "$(dirname "$0")"

echo "=== Staging Phase 3 Files ==="

# Stage Phase 3 core deliverables
git add plume_nav_sim/observations/
git add plume_nav_sim/interfaces/
git add tests/unit/observations/
git add tests/strategies.py
git add tests/contracts/test_observation_model_interface.py

# Stage contract files
git add contracts/observation_model_interface.md
git add contracts/component_interfaces.md
git add contracts/action_processor_interface.md
git add contracts/reward_function_interface.md
git add contracts/CONTRACT_VERIFICATION.md
git add contracts/OBSERVATION_DESIGN_SUMMARY.md

# Stage test infrastructure
git add tests/contracts/test_action_processor_interface.py
git add tests/contracts/test_reward_function_interface.py

# Stage related updates from earlier phases
git add contracts/core_types.md
git add contracts/environment_state_machine.md
git add contracts/gymnasium_api.md
git add contracts/reward_function.md
git add plume_nav_sim/core/state.py
git add plume_nav_sim/core/types.py
git add plume_nav_sim/plume/concentration_field.py

echo ""
echo "=== Staging Documentation (project root) ==="

# Stage documentation at project root
cd ../..
git add PHASE_3_COMPLETION_SUMMARY.md
git add REFACTORING_PROGRESS.md
git add PHASE_0_COMPLETION_SUMMARY.md
git add PHASE_1_COMPLETION_SUMMARY.md
git add PHASE_2_COMPLETION_SUMMARY.md
git add PHASE_1_2_TEST_RESULTS.md

cd src/backend

echo ""
echo "=== Git Status ==="
git status

echo ""
echo "=== Creating Commit ==="

cat > .commit-msg.txt << 'EOF'
feat(observations): implement observation models with TDD (Phase 3)

Implement two complete observation models using Test-Driven Development:
- ConcentrationSensor: single-point odor sensor at agent position
- AntennaeArraySensor: multi-sensor array with orientation-aware positioning

Implementation:
- Protocol-based design (no inheritance required)
- Full contract compliance with ObservationModel interface
- 40 tests passing (19 ConcentrationSensor + 21 AntennaeArraySensor)
- Property-based testing with Hypothesis
- Orientation-relative sensor positioning (math convention: 0°=East, 90°=North)

Test Infrastructure:
- Abstract test suite for universal properties (13 tests per model)
- Hypothesis strategies for env_state generation
- Fixed health check warnings (function_scoped_fixture, differing_executors)

Documentation:
- PHASE_3_COMPLETION_SUMMARY.md with TDD workflow and design decisions
- Contract verification and usage examples
- Updated REFACTORING_PROGRESS.md (Phase 3 complete)

Files:
- plume_nav_sim/observations/{__init__.py,concentration.py,antennae_array.py}
- plume_nav_sim/interfaces/observation_model.py
- tests/unit/observations/test_{concentration,antennae_array}_sensor.py
- tests/contracts/test_observation_model_interface.py
- tests/strategies.py
- contracts/observation_model_interface.md
- PHASE_3_COMPLETION_SUMMARY.md

Related:
- Phase 0: Contract fixes
- Phase 1: Core types and protocols
- Phase 2: Test infrastructure

Next: Phase 4 - Implement reward functions using same TDD approach
EOF

git commit -F .commit-msg.txt
rm .commit-msg.txt

echo ""
echo "=== Commit Complete ==="
echo "Run 'git push' to push to remote"
