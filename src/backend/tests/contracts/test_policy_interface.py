"""
Universal test suite for Policy protocol implementations.

Contract: src/backend/contracts/policy_interface.md

All policy implementations MUST pass these tests.
Concrete test classes should inherit from TestPolicyInterface
and provide a `policy` fixture.
"""

import numpy as np
import pytest

from plume_nav_sim.interfaces import Policy as PolicyProtocol


class TestPolicyInterface:
    """Universal test suite for Policy implementations.

    Contract: policy_interface.md

    All implementations must pass these tests to be considered valid.
    Concrete test classes should inherit this and provide `policy` fixture.
    """

    __test__ = False

    # ==================================================================
    # Fixtures (Override in concrete test classes)
    # ==================================================================

    @pytest.fixture
    def policy(self) -> PolicyProtocol:
        """Override this fixture to provide the policy to test.

        Returns:
            Policy implementation to test

        Raises:
            NotImplementedError: If not overridden in subclass
        """
        raise NotImplementedError("Concrete test classes must override policy fixture")

    # ==================================================================
    # Property 1: Action-Space Containment (UNIVERSAL)
    # ==================================================================

    def test_action_in_space(self, policy: PolicyProtocol):
        """Selected actions must be in the defined action space.

        Contract: policy_interface.md - Property 1
        """
        policy.reset(seed=123)
        # Simple 1D concentration observations typical for default sensors
        observations = [
            np.array([c], dtype=np.float32)
            for c in [0.1, 0.2, 0.2, 0.05, 0.06, 0.07, 0.01, 0.5, 0.49, 0.5]
        ]
        for obs in observations:
            action = policy.select_action(obs, explore=True)
            assert policy.action_space.contains(
                action
            ), f"Action {action} not in space {policy.action_space}"

    # ==================================================================
    # Property 2: Seed Determinism (UNIVERSAL)
    # ==================================================================

    def test_seed_determinism_single_instance(self, policy: PolicyProtocol):
        """Resetting with the same seed reproduces the action sequence.

        Contract: policy_interface.md - Property 2
        """
        observations = [
            np.array([c], dtype=np.float32)
            for c in [0.2, 0.2, 0.1, 0.3, 0.29, 0.31, 0.0, 1.0, 0.99, 1.0]
        ]

        policy.reset(seed=777)
        seq1 = [policy.select_action(obs, explore=True) for obs in observations]

        policy.reset(seed=777)
        seq2 = [policy.select_action(obs, explore=True) for obs in observations]

        assert (
            seq1 == seq2
        ), "Policy must produce identical actions for same seed and observations"

    def test_seed_determinism_across_instances(self, policy: PolicyProtocol):
        """Two instances with the same seed behave identically.

        Contract: policy_interface.md - Property 2
        """
        # Obtain a new instance of the same policy via type() to avoid coupling
        cls = type(policy)
        p1 = policy
        p2 = cls()

        observations = [
            np.array([c], dtype=np.float32)
            for c in [0.0, 0.01, 0.0, 0.02, 0.02, 0.018, 0.5, 0.49, 0.51]
        ]

        p1.reset(seed=42)
        p2.reset(seed=42)

        a1 = [p1.select_action(obs, explore=True) for obs in observations]
        a2 = [p2.select_action(obs, explore=True) for obs in observations]

        assert (
            a1 == a2
        ), "Different instances with same seed must produce identical actions"

    # ==================================================================
    # Property 3: Input Purity (UNIVERSAL)
    # ==================================================================

    def test_does_not_mutate_observation(self, policy: PolicyProtocol):
        """Policy must not mutate the provided observation.

        Contract: policy_interface.md - Property 3
        """
        obs = np.array([0.123], dtype=np.float32)
        obs_copy = obs.copy()
        _ = policy.select_action(obs, explore=True)
        np.testing.assert_array_equal(obs, obs_copy)

    # ==================================================================
    # Property 4: Space Immutability (UNIVERSAL)
    # ==================================================================

    def test_action_space_immutable(self, policy: PolicyProtocol):
        """action_space property must return a stable object.

        Contract: policy_interface.md - Property 4
        """
        assert (
            policy.action_space is policy.action_space
        ), "action_space must be immutable (same instance)"

    # ==================================================================
    # Protocol Conformance (Runtime)
    # ==================================================================

    def test_runtime_protocol_conformance(self, policy: PolicyProtocol):
        """Ensure the policy object conforms to the Policy Protocol at runtime."""
        assert isinstance(policy, PolicyProtocol)
