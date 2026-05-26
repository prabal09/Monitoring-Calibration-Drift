import math

import numpy as np

from calibration_drift.drift.error_metrics import mean_reprojection_error


def _vertical_edge_image(width=100, height=100, x=50):
    edges = np.zeros((height, width), dtype=np.uint8)
    edges[:, x] = 255
    return edges


def test_pixels_on_edges_yield_zero_error():
    edges = _vertical_edge_image(x=50)
    pixels = np.array([[50.0, 10.0], [50.0, 50.0], [50.0, 90.0]])
    err = mean_reprojection_error(edges, pixels)
    assert err == 0.0


def test_pixels_off_edges_yield_their_offset():
    edges = _vertical_edge_image(x=50)
    pixels = np.array([[55.0, 50.0], [45.0, 50.0]])   # 5 px to either side of edge
    err = mean_reprojection_error(edges, pixels)
    assert math.isclose(err, 5.0, abs_tol=0.1)


def test_drift_increases_error():
    edges = _vertical_edge_image(x=50)
    aligned = np.array([[50.0, y] for y in range(20, 80)])
    drifted = aligned + np.array([3.0, 0.0])

    err_aligned = mean_reprojection_error(edges, aligned)
    err_drifted = mean_reprojection_error(edges, drifted)

    assert err_aligned < err_drifted
    assert math.isclose(err_drifted, 3.0, abs_tol=0.1)


def test_no_edges_returns_nan():
    edges = np.zeros((50, 50), dtype=np.uint8)
    pixels = np.array([[25.0, 25.0]])
    err = mean_reprojection_error(edges, pixels)
    assert math.isnan(err)


def test_no_pixels_returns_nan():
    edges = _vertical_edge_image()
    pixels = np.zeros((0, 2))
    err = mean_reprojection_error(edges, pixels)
    assert math.isnan(err)
