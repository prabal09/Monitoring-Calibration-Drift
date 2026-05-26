import math

import pytest

from calibration_drift.drift.detector import DriftDetector


def test_window_size_must_be_at_least_two():
    with pytest.raises(ValueError):
        DriftDetector(window_size=1)


def test_constant_errors_do_not_trigger_drift():
    d = DriftDetector(window_size=10, slope_threshold=0.1)
    for _ in range(20):
        d.update(2.0)
    assert math.isclose(d.slope(), 0.0, abs_tol=1e-9)
    assert d.is_drifting() is False


def test_increasing_errors_trigger_drift():
    d = DriftDetector(window_size=10, slope_threshold=0.1)
    for i in range(10):
        d.update(2.0 + 0.5 * i)   # slope = 0.5 px/frame, well above threshold
    assert d.slope() > 0.1
    assert d.is_drifting() is True


def test_decreasing_errors_do_not_trigger():
    d = DriftDetector(window_size=10, slope_threshold=0.1)
    for i in range(10):
        d.update(10.0 - 0.5 * i)   # slope = -0.5 px/frame
    assert d.slope() < 0
    assert d.is_drifting() is False


def test_partial_window_never_flags():
    d = DriftDetector(window_size=10, slope_threshold=0.1)
    for i in range(5):
        d.update(2.0 + 5.0 * i)   # huge slope, but window not full yet
    assert d.is_drifting() is False


def test_nan_values_are_dropped():
    d = DriftDetector(window_size=5, slope_threshold=0.1)
    d.update(1.0)
    d.update(float("nan"))
    d.update(2.0)
    assert d.errors == [1.0, 2.0]


def test_window_eviction_keeps_only_recent_samples():
    d = DriftDetector(window_size=3, slope_threshold=0.1)
    for v in [10.0, 9.0, 8.0, 1.0, 2.0, 3.0]:
        d.update(v)
    assert d.errors == [1.0, 2.0, 3.0]
