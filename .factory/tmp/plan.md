# Development Plan – utils test fixes

1. Create feature branch:  
   `git checkout -b droid/utils-tests`

2. Remove local stub that conflicts with real plugin:  
   `git rm pytest_benchmark.py`

3. Set up isolated environment:  
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

4. Execute targeted test suite:  
   `.venv/bin/pytest tests/utils -q`

5. If failures occur, apply **minimal fixes only** under `src/plume_nav_sim/utils/` and rerun tests until green.

6. Commit logically grouped changes with clear messages.

7. After user review, push branch and open PR summarizing fixes and test results.
