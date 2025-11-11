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


def test_action_panel_update_and_label():
    m = ActionPanelModel()
    m.set_action_names(["FORWARD", "TURN_LEFT", "TURN_RIGHT"])
    m.update_event(2)
    assert m.state.action_index == 2
    assert m.state.action_label.startswith("TURN_RIGHT")


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
