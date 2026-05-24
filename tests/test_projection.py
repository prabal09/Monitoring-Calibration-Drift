import numpy as np
import pytest

from calibration_drift.projection.lidar_to_image import project_points


def _intrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0):
    return np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]]
    )


def test_point_on_optical_axis_projects_to_principal_point():
    K = _intrinsics(cx=640.0, cy=360.0)
    T = np.eye(4)
    points = np.array([[0.0, 0.0, 5.0]])

    pixels, depths, keep = project_points(points, T, K)

    assert keep.tolist() == [True]
    np.testing.assert_allclose(pixels, [[640.0, 360.0]])
    np.testing.assert_allclose(depths, [5.0])


def test_pinhole_projection_with_known_offsets():
    # A point at (x=1, y=2, z=10) with fx=fy=1000, cx=640, cy=360
    # should project to (640 + 1000 * 1/10, 360 + 1000 * 2/10) = (740, 560).
    K = _intrinsics()
    T = np.eye(4)
    points = np.array([[1.0, 2.0, 10.0]])

    pixels, _, _ = project_points(points, T, K)

    np.testing.assert_allclose(pixels, [[740.0, 560.0]])


def test_points_behind_camera_are_filtered():
    K = _intrinsics()
    T = np.eye(4)
    points = np.array([
        [0.0, 0.0, 5.0],    # in front
        [0.0, 0.0, -5.0],   # behind
        [0.0, 0.0, 0.0],    # at the camera (z < min_depth)
    ])

    pixels, depths, keep = project_points(points, T, K)

    assert keep.tolist() == [True, False, False]
    assert pixels.shape == (1, 2)
    np.testing.assert_allclose(depths, [5.0])


def test_image_bounds_filter():
    K = _intrinsics(cx=640.0, cy=360.0)
    T = np.eye(4)
    # First point projects to (640, 360) — inside a 1280×720 image.
    # Second point at (10, 0, 1) projects to (640 + 10000, 360) — outside.
    points = np.array([
        [0.0, 0.0, 5.0],
        [10.0, 0.0, 1.0],
    ])

    pixels, _, keep = project_points(points, T, K, image_size=(1280, 720))

    assert keep.tolist() == [True, False]
    assert pixels.shape == (1, 2)


def test_translation_extrinsics_shifts_projection():
    # T_cam_lidar translates LiDAR origin by +1m along camera x.
    # A LiDAR point at (0, 0, 5) becomes (1, 0, 5) in camera frame, projecting
    # to (cx + fx * 1/5, cy) = (640 + 200, 360) = (840, 360).
    K = _intrinsics()
    T = np.eye(4)
    T[0, 3] = 1.0
    points = np.array([[0.0, 0.0, 5.0]])

    pixels, _, _ = project_points(points, T, K)

    np.testing.assert_allclose(pixels, [[840.0, 360.0]])


def test_empty_input_returns_empty_outputs():
    K = _intrinsics()
    T = np.eye(4)
    points = np.zeros((0, 3))

    pixels, depths, keep = project_points(points, T, K)

    assert pixels.shape == (0, 2)
    assert depths.shape == (0,)
    assert keep.shape == (0,)


def test_invalid_input_shape_raises():
    K = _intrinsics()
    T = np.eye(4)
    with pytest.raises(ValueError):
        project_points(np.zeros((5, 4)), T, K)
    with pytest.raises(ValueError):
        project_points(np.zeros((5, 3)), np.eye(3), K)
    with pytest.raises(ValueError):
        project_points(np.zeros((5, 3)), T, np.eye(4))
