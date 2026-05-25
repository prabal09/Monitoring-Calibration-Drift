import numpy as np

from calibration_drift.viz.overlay import draw_overlay


def test_empty_pixels_returns_image_unchanged():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    out = draw_overlay(image, np.zeros((0, 2)), np.zeros((0,)))
    np.testing.assert_array_equal(out, image)


def test_dots_drawn_at_projection_sites():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    pixels = np.array([[50.0, 50.0], [150.0, 80.0]])
    depths = np.array([10.0, 30.0])

    out = draw_overlay(image, pixels, depths, radius=3)

    # Pixels near each projection site should have been written.
    assert out[48:53, 48:53].sum() > 0
    assert out[78:83, 148:153].sum() > 0
    # Far-away region should be untouched (still all zeros).
    assert out[0:10, 0:10].sum() == 0


def test_depth_changes_color():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    pixels = np.array([[25.0, 25.0]])

    near = draw_overlay(image, pixels, np.array([1.0]))
    far = draw_overlay(image, pixels, np.array([49.0]))

    # Same pixel position, different depths → different colors.
    assert not np.array_equal(near[25, 25], far[25, 25])
