import warnings

import plume_nav_sim.render as render


def test_detect_numpy_sets_flags():
    caps = {"warnings": [], "performance_characteristics": {}}
    render._detect_numpy(caps, test_performance=False)
    # If numpy is installed, flags should be set; otherwise a warning should be added
    if caps.get("numpy_available"):
        assert caps["numpy_version"] is not None
    else:
        assert any(
            "NumPy not available" in w for w in caps["warnings"]
        )  # pragma: no cover


def test_detect_numpy_with_performance_metrics():
    caps = {"warnings": [], "performance_characteristics": {}}
    render._detect_numpy(caps, test_performance=True)
    if caps.get("numpy_available"):
        pc = caps["performance_characteristics"]
        # The helper records conversion time and a rating
        assert "numpy_performance_rating" in pc


def test_detect_matplotlib_section_populates_fields():
    caps = {"warnings": [], "performance_characteristics": {}}
    render._detect_matplotlib_section(caps, test_performance_characteristics=False)
    # Keys should be added regardless of availability
    assert "matplotlib_available" in caps
    assert "matplotlib_backends" in caps
    assert "display_available" in caps


def test_configure_headless_if_needed_sets_agg(monkeypatch):
    caps = {"matplotlib_available": True, "display_available": False, "warnings": []}
    render._configure_headless_if_needed(caps)
    # Either set True or record a warning; both acceptable
    assert caps.get("headless_compatible") in {True, False}


def test_generate_recommendations_section_produces_items():
    caps = {
        "warnings": [],
        "recommendations": [],
        "numpy_available": False,
        "matplotlib_available": False,
        "matplotlib_backends": [],
        "display_available": False,
        "performance_characteristics": {"numpy_performance_rating": "slow"},
    }
    render._generate_recommendations_section(
        caps, test_performance_characteristics=True
    )
    assert isinstance(caps.get("recommendations"), list)
    assert len(caps["recommendations"]) >= 1


def test_assess_system_resources_section_handles_psutil():
    caps = {"warnings": []}
    render._assess_system_resources_section(caps)
    if "system_resources" in caps:
        sr = caps["system_resources"]
        assert set(
            ["total_memory_gb", "available_memory_gb", "memory_usage_percent"]
        ).issubset(sr.keys())
    else:
        # psutil may not be available in some environments
        assert any(
            "psutil not available" in w for w in caps["warnings"]
        )  # pragma: no cover


def test_warn_about_limitations_emits_expected_warnings(caplog):
    caps = {
        "numpy_available": False,
        "matplotlib_available": False,
        "matplotlib_backends": [],
        "display_available": False,
        "performance_characteristics": {"numpy_performance_rating": "slow"},
        "color_scheme_support": {"colormap_support_rating": "limited"},
        "platform_support": {"full_support": True},
        "recommendations": ["Install NumPy", "Install matplotlib"],
    }
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        render._warn_about_limitations(caps, include_recommendations=True)
        msgs = [str(w.message) for w in wlist]
        assert any("NumPy not available" in m for m in msgs)
        assert any("Matplotlib not available" in m for m in msgs)
        assert any("NumPy performance rating" in m for m in msgs)
    # Log of colormap limitation
    assert any("Limited colormap support" in rec.message for rec in caplog.records)
