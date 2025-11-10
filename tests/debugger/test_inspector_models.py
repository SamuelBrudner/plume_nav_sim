import numpy as np

from plume_nav_debugger.inspector.models import ActionPanelModel, ObservationPanelModel
from plume_nav_debugger.inspector.plots import normalize_series_to_polyline


def test_observation_summary_scalar_like():
    m = ObservationPanelModel()
    m.update(np.array([0.3], dtype=np.float32))
    assert m.summary is not None
    assert m.summary.shape == (1,)
    assert abs(m.summary.vmin - 0.3) < 1e-6
    assert abs(m.summary.vmean - 0.3) < 1e-6
    assert abs(m.summary.vmax - 0.3) < 1e-6


def test_observation_summary_array():
    m = ObservationPanelModel()
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    m.update(arr)
    assert m.summary is not None
    assert m.summary.shape == (2, 2)
    assert m.summary.vmin == 1.0
    assert abs(m.summary.vmean - 2.5) < 1e-6
    assert m.summary.vmax == 4.0


class _PolicyProbs:
    def action_probabilities(self, observation):
        return np.array([0.1, 0.7, 0.2])


class _PolicyQ:
    def q_values(self, observation):
        return np.array([1.0, 2.0, -1.0])


class _PolicyLogits:
    def logits(self, observation):
        return np.array([0.0, 1.0, -2.0])


def test_action_panel_update_and_label():
    m = ActionPanelModel()
    m.set_action_names(["FORWARD", "TURN_LEFT", "TURN_RIGHT"])
    m.update_event(2)
    assert m.state.action_index == 2
    assert m.state.action_label.startswith("TURN_RIGHT")


def test_action_panel_distribution_probs():
    m = ActionPanelModel()
    m.set_action_names(["a", "b", "c"])
    obs = np.array([0.5])
    m.probe_distribution(_PolicyProbs(), obs)
    assert m.state.distribution_source == "probs"
    assert m.state.distribution is not None
    assert abs(sum(m.state.distribution) - 1.0) < 1e-6
    assert len(m.state.distribution) == 3


def test_action_panel_distribution_q_values_softmax():
    m = ActionPanelModel()
    m.set_action_names(["a", "b", "c"])
    obs = np.array([0.5])
    m.probe_distribution(_PolicyQ(), obs)
    assert m.state.distribution_source == "q_values"
    assert m.state.distribution is not None
    p = np.array(m.state.distribution)
    assert p.shape == (3,)
    assert abs(float(p.sum()) - 1.0) < 1e-6
    # The middle entry (corresponding to Q=2.0) should be the largest
    assert int(np.argmax(p)) == 1


def test_action_panel_distribution_logits_softmax():
    m = ActionPanelModel()
    m.set_action_names(["a", "b", "c"])
    obs = np.array([0.5])
    m.probe_distribution(_PolicyLogits(), obs)
    assert m.state.distribution_source == "logits"
    p = np.array(m.state.distribution)
    assert abs(float(p.sum()) - 1.0) < 1e-6
    assert int(np.argmax(p)) == 1


def test_distribution_is_normalized_and_does_not_call_select_action():
    class _Policy:
        def __init__(self):
            self.called_select = 0
            self.called_q = 0

        def select_action(self, obs, explore=False):
            self.called_select += 1
            return 0

        def q_values(self, obs):
            self.called_q += 1
            return np.array([0.0, 0.0, 0.0])  # uniform fallback path

    m = ActionPanelModel()
    obs = np.array([0.25])
    pol = _Policy()
    m.probe_distribution(pol, obs)
    # Ensure q_values used, select_action not called
    assert pol.called_q == 1
    assert pol.called_select == 0
    # Distribution normalized and uniform due to zero-sum
    p = np.array(m.state.distribution)
    assert abs(float(p.sum()) - 1.0) < 1e-6
    assert np.allclose(p, np.full(3, 1 / 3))


def test_distribution_depends_on_observation_and_is_deterministic():
    class _PolicyObsQ:
        def q_values(self, obs):
            # Make distribution depend on observation value
            v = float(obs[0])
            return np.array([v, 0.0, -v])

    m = ActionPanelModel()
    obs1 = np.array([0.1])
    obs2 = np.array([0.9])
    m.probe_distribution(_PolicyObsQ(), obs1)
    p1 = np.array(m.state.distribution)
    m.probe_distribution(_PolicyObsQ(), obs2)
    p2 = np.array(m.state.distribution)
    assert not np.allclose(p1, p2)  # depends on observation
    # Deterministic for same input
    m.probe_distribution(_PolicyObsQ(), obs1)
    p1_again = np.array(m.state.distribution)
    assert np.allclose(p1, p1_again)


def test_action_label_mapping_with_names():
    m = ActionPanelModel()
    m.set_action_names(["FORWARD", "LEFT", "RIGHT"])
    m.update_event(1)
    assert m.state.action_label.startswith("LEFT")


def test_normalize_series_to_polyline_basic():
    pts = normalize_series_to_polyline(np.array([0.0, 0.5, 1.0]), 100, 30, pad=2)
    assert len(pts) == 3
    # X should be increasing
    xs = [p[0] for p in pts]
    assert xs[0] < xs[1] < xs[2]
    # Y should be decreasing as value increases (top-left origin)
    ys = [p[1] for p in pts]
    assert ys[0] > ys[1] > ys[2]


def test_normalize_series_to_polyline_flat_series_centered():
    pts = normalize_series_to_polyline(np.array([1.0, 1.0, 1.0]), 50, 20, pad=2)
    assert len(pts) == 3
    ys = [p[1] for p in pts]
    assert len(set(ys)) == 1  # all equal
