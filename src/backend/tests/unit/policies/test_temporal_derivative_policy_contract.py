"""Contract compliance tests for TemporalDerivativePolicy.

Contract: src/backend/contracts/policy_interface.md

This test suite reuses the universal Policy interface tests by providing
the concrete `TemporalDerivativePolicy` implementation via a fixture.
"""

import pytest

from plume_nav_sim.policies import TemporalDerivativePolicy
from tests.contracts.test_policy_interface import TestPolicyInterface


class TestTemporalDerivativePolicyContract(TestPolicyInterface):
    __test__ = True

    @pytest.fixture
    def policy(self):
        # Use default parameters; tests explicitly seed via reset()
        return TemporalDerivativePolicy()
