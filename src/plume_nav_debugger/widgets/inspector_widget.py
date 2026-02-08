from __future__ import annotations

import logging
from typing import Optional

import numpy as np

try:
    from PySide6 import QtCore, QtWidgets
except Exception as e:  # pragma: no cover - GUI import guard
    raise RuntimeError(
        "PySide6 is required for the debugger app. Install it in your env (e.g., conda run -n plume-nav-sim pip install PySide6)."
    ) from e

from plume_nav_debugger.inspector.introspection import format_pipeline
from plume_nav_debugger.inspector.models import ActionPanelModel, ObservationPanelModel
from plume_nav_debugger.odc.mux import ProviderMux
from plume_nav_debugger.widgets.action_panel_widget import ActionPanelWidget
from plume_nav_debugger.widgets.observation_panel_widget import ObservationPanelWidget

logger = logging.getLogger(__name__)


class InspectorWidget(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        vbox = QtWidgets.QVBoxLayout(self)
        # Strict-mode banner (hidden by default)
        self.info_label = QtWidgets.QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #a60; padding: 4px;")
        self.info_label.setTextFormat(QtCore.Qt.RichText)
        self.info_label.setOpenExternalLinks(True)
        self.info_label.setVisible(False)
        self.tabs = QtWidgets.QTabWidget()
        self.action_panel = ActionPanelWidget()
        self.obs_panel = ObservationPanelWidget()
        self.tabs.addTab(self.action_panel, "Action")
        self.tabs.addTab(self.obs_panel, "Observation")
        vbox.addWidget(self.info_label)
        vbox.addWidget(self.tabs)

        # No control signals: Inspector is information-only

        # Models for TDD-friendly logic
        self._obs_model = ObservationPanelModel()
        self._act_model = ActionPanelModel()
        self._policy_for_probe = None
        self._mux: Optional[ProviderMux] = None
        self._pipeline_text: str = ""
        self._strict_provider_only: bool = True

    @QtCore.Slot(object)
    def on_step_event(self, ev: object) -> None:
        try:
            obs = getattr(ev, "obs", None)
            action = getattr(ev, "action", None)

            # Update models
            if isinstance(obs, np.ndarray):
                self._obs_model.update(obs)
            else:
                self._obs_model.summary = None

            self._act_model.update_event(
                action if isinstance(action, (int, np.integer)) else None
            )

            # Probe distribution (provider-only)
            if isinstance(obs, np.ndarray):
                dist = None
                if self._mux is not None:
                    try:
                        dist = self._mux.get_policy_distribution(obs)
                    except Exception:
                        dist = None
                if dist is not None:
                    self._act_model.state.distribution = [float(x) for x in dist]
                    self._act_model.state.distribution_source = "provider"
                else:
                    self._act_model.state.distribution = None
                    self._act_model.state.distribution_source = None

            # Update UI: observation
            if self._obs_model.summary is not None:
                s = self._obs_model.summary
                self.obs_panel.obs_shape.setText(
                    f"shape: {'x'.join(str(d) for d in s.shape)}"
                )
                self.obs_panel.obs_stats.setText(
                    f"min/mean/max: {s.vmin:.3f}/{s.vmean:.3f}/{s.vmax:.3f}"
                )
                # Provider observation metadata (preferred in preview area)
                meta_shown = False
                try:
                    if self._mux is not None and hasattr(
                        self._mux, "describe_observation"
                    ):
                        info = self._mux.describe_observation(obs)  # type: ignore[attr-defined]
                        if info is not None:
                            kind = getattr(info, "kind", None)
                            label = getattr(info, "label", None)
                            parts = []
                            if isinstance(kind, str) and kind:
                                parts.append(f"kind={kind}")
                            if isinstance(label, str) and label:
                                parts.append(f"label={label}")
                            if parts:
                                self.obs_panel.preview_label.setText("; ".join(parts))
                                meta_shown = True
                except Exception:
                    meta_shown = False

                # Small preview for tiny vectors (only if no provider meta)
                try:
                    if not meta_shown and isinstance(obs, np.ndarray) and obs.size <= 8:
                        vals = ", ".join(f"{float(v):.3f}" for v in obs.ravel())
                        self.obs_panel.preview_label.setText(f"values: [{vals}]")
                    else:
                        self.obs_panel.preview_label.setText("")
                    # Sparkline for vector observations
                    if isinstance(obs, np.ndarray) and obs.size > 1:
                        self.obs_panel.sparkline.set_values(obs.astype(float))
                    else:
                        self.obs_panel.sparkline.set_values(None)
                except Exception:
                    self.obs_panel.preview_label.setText("")
                    self.obs_panel.sparkline.set_values(None)
            else:
                self.obs_panel.obs_shape.setText("shape: -")
                self.obs_panel.obs_stats.setText("min/mean/max: -/-/-")
                self.obs_panel.preview_label.setText("")
                self.obs_panel.sparkline.set_values(None)
            # Pipeline label if known
            if self._pipeline_text:
                if hasattr(self.obs_panel, "pipeline_label"):
                    self.obs_panel.pipeline_label.setText(self._pipeline_text)

            # Update UI: action
            action_label = self._act_model.state.action_label
            tie_label = None
            if self._act_model.state.distribution is not None:
                tie_label = self._distribution_tie(self._act_model.state.distribution)
            if tie_label:
                self.action_panel.expected_action_label.setText(
                    f"action taken: {action_label} (policy tie: {tie_label})"
                )
            else:
                self.action_panel.expected_action_label.setText(
                    f"action taken: {action_label}"
                )
            if self._act_model.state.distribution is not None:
                src = self._act_model.state.distribution_source or "probs"
                preview = self._format_distribution(self._act_model.state.distribution)
                self.action_panel.distribution_label.setText(
                    f"distribution ({src}): {preview}"
                )
            else:
                self.action_panel.distribution_label.setText("distribution: N/A")
        except Exception:
            logger.debug("InspectorWidget.on_step_event failed", exc_info=True)

    @QtCore.Slot(list)
    def on_action_names(self, names: list[str]) -> None:
        self._act_model.set_action_names(names)
        self.action_panel.on_action_names(names)
        self._update_strict_banner()

    def set_grid_size(self, w: int, h: int) -> None:
        self.action_panel.set_grid_size(w, h)

    @QtCore.Slot(object)
    def on_policy_changed(self, policy: object) -> None:
        self._policy_for_probe = policy

    def set_observation_pipeline_from_env(self, env: object) -> None:
        # Provider-only UI: pipeline derived only via ProviderMux
        self._pipeline_text = ""

    @QtCore.Slot(object)
    def on_mux_changed(self, mux: object) -> None:
        try:
            self._mux = mux  # type: ignore[assignment]
            # Update pipeline with provider if available
            if hasattr(self._mux, "get_pipeline"):
                try:
                    names = self._mux.get_pipeline()  # type: ignore[attr-defined]
                    self._pipeline_text = f"Pipeline: {format_pipeline(list(names))}"
                except Exception:
                    logger.debug("Pipeline extraction from mux failed", exc_info=True)
            # Update source label to indicate provider
            self.action_panel.source_label.setText("source: provider")
            self._update_strict_banner()
        except Exception:
            logger.debug("on_mux_changed failed", exc_info=True)
            self._mux = None
            self.action_panel.source_label.setText("source: none")
            self._update_strict_banner()

    def set_strict_provider_only(self, flag: bool) -> None:
        self._strict_provider_only = bool(flag)
        self._update_strict_banner()

    # Inspector display preferences passthrough
    def set_show_pipeline(self, flag: bool) -> None:
        try:
            self.obs_panel.set_show_pipeline(flag)
        except Exception:
            logger.debug("set_show_pipeline failed", exc_info=True)

    def set_show_preview(self, flag: bool) -> None:
        try:
            self.obs_panel.set_show_preview(flag)
        except Exception:
            logger.debug("set_show_preview failed", exc_info=True)

    def set_show_sparkline(self, flag: bool) -> None:
        try:
            self.obs_panel.set_show_sparkline(flag)
        except Exception:
            logger.debug("set_show_sparkline failed", exc_info=True)

    def _label_for_action_index(self, idx: int, *, include_index: bool = True) -> str:
        name = None
        try:
            if 0 <= idx < len(self._act_model.action_names):
                name = self._act_model.action_names[idx]
        except Exception:
            name = None
        if name:
            return f"{name}({idx})" if include_index else str(name)
        return str(idx)

    def _format_distribution(self, probs: list[float], *, max_items: int = 6) -> str:
        parts = []
        for idx, val in enumerate(probs):
            label = self._label_for_action_index(idx, include_index=True)
            parts.append(f"{label}={val:.2f}")
        if len(parts) > max_items:
            return ", ".join(parts[:max_items]) + ", ..."
        return ", ".join(parts)

    def _distribution_tie(self, probs: list[float], *, tol: float = 1e-3) -> str | None:
        if not probs:
            return None
        arr = np.asarray(probs, dtype=float).ravel()
        if arr.size == 0:
            return None
        max_val = float(np.max(arr))
        if not np.isfinite(max_val):
            return None
        top = [i for i, v in enumerate(arr) if abs(v - max_val) <= tol]
        if len(top) <= 1:
            return None
        labels = [self._label_for_action_index(i, include_index=False) for i in top]
        return "/".join(labels)

    def _update_strict_banner(self) -> None:
        try:
            show = bool(self._strict_provider_only) and (self._mux is None)
            if show:
                self.info_label.setText(
                    "Strict mode: no DebuggerProvider detected. Inspector shows limited information. "
                    "Implement a provider (ODC) to enable action labels, distributions, and pipeline details. "
                    '<a href="https://plume-nav-sim.dev/odc" style="color:#06c;">Read ODC docs</a>.'
                )
            self.info_label.setVisible(show)
            if show and self.parent() is None and not self.isVisible():
                # Ensure the banner registers as visible for standalone widgets.
                self.show()
        except Exception:
            self.info_label.setVisible(False)

