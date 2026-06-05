import numpy as np

from calibration_drift.features.lidar_features import extract_depth_edges


def test_uniform_depth_has_no_edges():
    pixels = np.array([[10.0, 10.0], [11.0, 10.0], [10.0, 11.0], [11.0, 11.0]])
    depths = np.array([10.0, 10.0, 10.0, 10.0])
    mask = extract_depth_edges(pixels, depths, k=3, depth_threshold=0.5)
    assert not mask.any()


def test_isolated_far_point_among_close_points_is_an_edge():
    # First point at depth 20m, neighbors at depth 5m → big gap → edge.
    pixels = np.array([
        [100.0, 100.0],
        [101.0, 100.0],
        [99.0, 100.0],
        [100.0, 101.0],
        [100.0, 99.0],
    ])
    depths = np.array([20.0, 5.0, 5.0, 5.0, 5.0])
    mask = extract_depth_edges(pixels, depths, k=4, depth_threshold=0.5)
    assert mask[0]            # the isolated far point sees close neighbors → edge
    assert mask[1:].all()     # each close point also sees the far one as a neighbor → edge


def test_two_separate_clusters_have_no_internal_edges():
    # Two tight clusters far apart in pixel space, each at its own depth.
    # With k=3, each point's neighbors are all in its own cluster (same depth),
    # so no point qualifies as an edge.
    near = np.array([[10.0, 10.0], [11.0, 10.0], [10.0, 11.0], [11.0, 11.0]])
    far = np.array([[500.0, 500.0], [501.0, 500.0], [500.0, 501.0], [501.0, 501.0]])
    pixels = np.vstack([near, far])
    depths = np.array([5.0] * 4 + [30.0] * 4)

    mask = extract_depth_edges(pixels, depths, k=3, depth_threshold=0.5)
    assert not mask.any()


def test_depth_threshold_gates_edge_classification():
    pixels = np.array([[10.0, 10.0], [11.0, 10.0]])
    depths = np.array([5.0, 5.3])   # 0.3 m gap

    # Threshold above the gap → no edges.
    assert not extract_depth_edges(pixels, depths, k=1, depth_threshold=0.5).any()
    # Threshold below the gap → both points see each other → both edges.
    assert extract_depth_edges(pixels, depths, k=1, depth_threshold=0.2).all()


def test_empty_input_returns_empty_mask():
    mask = extract_depth_edges(np.zeros((0, 2)), np.zeros((0,)))
    assert mask.shape == (0,)
    assert mask.dtype == bool


def test_single_point_cannot_be_an_edge():
    mask = extract_depth_edges(np.array([[50.0, 50.0]]), np.array([10.0]))
    assert mask.shape == (1,)
    assert not mask[0]
