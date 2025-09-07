import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _stub_visualization_dependencies(monkeypatch):
    """Stub heavy dependencies for visualization module."""
from loguru import logger

    pkg = types.ModuleType("plume_nav_sim")
    utils_pkg = types.ModuleType("plume_nav_sim.utils")
    logging_setup = types.ModuleType("plume_nav_sim.utils.logging_setup")
    logging_setup.get_module_logger = lambda name: logger
    core_pkg = types.ModuleType("plume_nav_sim.core")
    protocols = types.ModuleType("plume_nav_sim.core.protocols")
    class SourceProtocol:  # minimal placeholder
        pass
    protocols.SourceProtocol = SourceProtocol

    monkeypatch.setitem(sys.modules, "plume_nav_sim", pkg)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.utils.logging_setup", logging_setup)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.core", core_pkg)
    monkeypatch.setitem(sys.modules, "plume_nav_sim.core.protocols", protocols)

    # Stub PySide6 modules
    PySide6 = types.ModuleType("PySide6")
    QtWidgets = types.ModuleType("PySide6.QtWidgets")
    QtCore = types.ModuleType("PySide6.QtCore")
    QtGui = types.ModuleType("PySide6.QtGui")
    class Dummy:  # generic placeholder class
        pass
    QtWidgets.QApplication = Dummy
    QtWidgets.QMainWindow = Dummy
    QtWidgets.QWidget = Dummy
    QtWidgets.QVBoxLayout = Dummy
    QtWidgets.QHBoxLayout = Dummy
    QtCore.QTimer = Dummy
    QtCore.Signal = Dummy
    QtCore.QThread = Dummy
    QtGui.QPixmap = Dummy
    QtGui.QPainter = Dummy
    PySide6.QtWidgets = QtWidgets
    PySide6.QtCore = QtCore
    PySide6.QtGui = QtGui
    monkeypatch.setitem(sys.modules, "PySide6", PySide6)
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", QtWidgets)
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", QtCore)
    monkeypatch.setitem(sys.modules, "PySide6.QtGui", QtGui)

    # Stub streamlit
    monkeypatch.setitem(sys.modules, "streamlit", types.ModuleType("streamlit"))

    # Stub hydra and omegaconf
    hydra = types.ModuleType("hydra")
    core = types.ModuleType("hydra.core")
    config_store = types.ModuleType("hydra.core.config_store")
    class ConfigStore:
        @classmethod
        def instance(cls):
            return cls()
        def store(self, *args, **kwargs):
            pass
    config_store.ConfigStore = ConfigStore
    core.config_store = config_store
    hydra.core = core
    monkeypatch.setitem(sys.modules, "hydra", hydra)
    monkeypatch.setitem(sys.modules, "hydra.core", core)
    monkeypatch.setitem(sys.modules, "hydra.core.config_store", config_store)

    omegaconf = types.ModuleType("omegaconf")
    class DictConfig(dict):
        pass
    class OmegaConf:
        pass
    omegaconf.DictConfig = DictConfig
    omegaconf.OmegaConf = OmegaConf
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf)


def test_console_backend_removed(monkeypatch):
    _stub_visualization_dependencies(monkeypatch)
    path = Path("src/plume_nav_sim/utils/visualization.py")
    spec = importlib.util.spec_from_file_location("plume_nav_sim.utils.visualization", path)
    vis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vis)

    with pytest.raises(ImportError):
        vis.create_debug_visualizer(backend="console")
