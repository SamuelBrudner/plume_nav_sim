name: Bug Report
description: Report a bug or unexpected behavior in plume-nav-sim
title: "[BUG] Brief description of the issue"
labels: ["bug", "needs-triage"]
assignees: []
body:
  - type: markdown
    attributes:
      value: |
        ## Bug Description

        **Clear and concise description of the bug**
        <!-- Describe what went wrong and what you expected to happen -->

  - type: dropdown
    id: bug-category
    attributes:
      label: Bug Category
      description: Select the category that best describes the bug.
      options:
        - Environment API (reset, step, render, close methods)
        - Action/Observation Space Issues
        - Rendering Problems (rgb_array or human mode)
        - Seeding/Reproducibility Issues
        - Performance Problems
        - Installation/Dependencies
        - Documentation
        - Other (please specify)
    validations:
      required: true

  - type: input
    id: system-configuration
    attributes:
      label: System Configuration
      description: e.g., OS, Python version, plume-nav-sim version
      placeholder: "Ubuntu 22.04, Python 3.11.5, plume-nav-sim 0.0.1"
    validations:
      required: true

  - type: input
    id: dependencies
    attributes:
      label: Dependencies
      description: e.g., gymnasium version, numpy version, matplotlib version
      placeholder: "gymnasium==0.29.1, numpy==2.1.0, matplotlib==3.9.0"
    validations:
      required: true

  - type: input
    id: environment-setup
    attributes:
      label: Environment Setup
      description: Installation method, virtual environment, display available
      placeholder: "pip install -e ., venv environment, headless server"
    validations:
      required: true

  - type: textarea
    id: reproduction-steps
    attributes:
      label: Reproduction Steps
      description: Minimal code to reproduce the bug
      placeholder: |
        ```python
        import gymnasium as gym
        from plume_nav_sim.registration import register_env, ENV_ID
        
        register_env()
        env = gym.make(ENV_ID, render_mode="rgb_array")
        obs, info = env.reset(seed=42)
        # Add your reproduction steps here
        ```
      render: python
    validations:
      required: true

  - type: input
    id: expected-behavior
    attributes:
      label: Expected Behavior
      description: What should happen
      placeholder: "Environment should reset successfully and return observation tuple"
    validations:
      required: true

  - type: input
    id: actual-behavior
    attributes:
      label: Actual Behavior
      description: What actually happens
      placeholder: "Environment throws ValueError when attempting to reset"
    validations:
      required: true

  - type: textarea
    id: error-message
    attributes:
      label: Full error message/traceback
      description: Paste the complete error message and traceback here
      placeholder: |
        Traceback (most recent call last):
          File "example.py", line 5, in <module>
            obs, info = env.reset(seed=42)
          File "plume_nav_sim/envs/static_gaussian.py", line 123, in reset
            raise ValueError("Invalid seed value")
        ValueError: Invalid seed value
      render: shell

  - type: input
    id: error-context
    attributes:
      label: Error Context
      description: When does the error occur? Is it reproducible? Does it happen with default parameters?
      placeholder: "Error occurs on first reset call, reproducible with any seed value, happens with default parameters"

  - type: input
    id: environment-parameters
    attributes:
      label: Environment Parameters
      description: grid_size, source_location, max_steps, goal_radius, render_mode
      placeholder: "grid_size=(128, 128), source_location=(64, 64), max_steps=1000, goal_radius=0, render_mode='rgb_array'"

  - type: input
    id: seeding-issues
    attributes:
      label: For Seeding Issues
      description: Seed value used, expected deterministic behavior, multiple runs produce different results
      placeholder: "seed=42, expected identical episodes, different start positions across runs"

  - type: input
    id: performance-info
    attributes:
      label: Performance Information
      description: Step latency, memory usage, grid size, number of steps before issue
      placeholder: "Step latency >5ms, memory usage 200MB, grid_size=(256,256), issue after 500 steps"

  - type: textarea
    id: benchmarking
    attributes:
      label: Benchmarking
      description: If you measured performance, include timing code
      placeholder: |
        ```python
        import time
        env = gym.make(ENV_ID)
        obs, info = env.reset(seed=42)
        
        start_time = time.perf_counter()
        for i in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        end_time = time.perf_counter()
        
        print(f"Average step time: {(end_time - start_time) / 1000:.4f}s")
        ```
      render: python

  - type: input
    id: rendering-issues
    attributes:
      label: Rendering Issues
      description: Render mode affected, matplotlib backend, headless environment, error occurs on first or subsequent renders
      placeholder: "human mode fails, TkAgg backend, headless environment, error on first render() call"

  - type: checkboxes
    id: testing-info
    attributes:
      label: Have you run the test suite?
      options:
        - label: Yes, tests pass
        - label: Yes, tests fail (include failed test names)
        - label: No, haven't run tests
        - label: Cannot run tests (explain why)

  - type: input
    id: test-command
    attributes:
      label: Test command used
      description: e.g., pytest -v tests/test_environment_api.py
      placeholder: "pytest -q tests/"

  - type: textarea
    id: additional-context
    attributes:
      label: Additional Context
      description: Workarounds found, related issues, logs/debug information, screenshots/visual output
      placeholder: |
        - Workaround: Using rgb_array mode instead of human mode
        - Related to issue #123
        - Debug logs show matplotlib backend error
        - Screenshot of rendering output (if applicable)

  - type: textarea
    id: proposed-solution
    attributes:
      label: Proposed Solution (Optional)
      description: If you have ideas for how to fix this bug, please share them
      placeholder: |
        Possible fixes:
        1. Add backend detection in rendering pipeline
        2. Implement graceful fallback for headless environments
        3. Update matplotlib version requirements

  - type: markdown
    attributes:
      value: |
        ## For Maintainers

        **Triage Checklist:**
        - [ ] Bug reproduced on development environment
        - [ ] Component responsible identified
        - [ ] Severity level assigned (critical/high/medium/low)
        - [ ] Performance regression checked (if applicable)
        - [ ] Cross-platform impact assessed
        - [ ] Test coverage gap identified (if applicable)

        **Debug Priority:**
        - [ ] API Compliance Issue (high priority)
        - [ ] Performance Regression (high priority)
        - [ ] Rendering Compatibility (medium priority)
        - [ ] Documentation/Usability (low priority)
