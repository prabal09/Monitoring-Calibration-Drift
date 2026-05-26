import numpy as np

from calibration_drift.features.image_features import extract_edges


def test_canny_finds_square_boundary():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[30:70, 30:70] = 255

    edges = extract_edges(image, canny_low=50, canny_high=150)

    # Canny should fire along the square's outline.
    assert edges.shape == image.shape
    assert edges.dtype == np.uint8
    assert edges[30, 30:70].sum() > 0           # top edge of the square
    assert edges[69, 30:70].sum() > 0           # bottom edge
    # Interior of the square should have no edges (uniform region).
    assert edges[40:60, 40:60].sum() == 0


def test_handles_rgb_input():
    rgb = np.zeros((50, 50, 3), dtype=np.uint8)
    rgb[10:40, 10:40] = 255
    edges = extract_edges(rgb)
    assert edges.shape == (50, 50)
    assert edges.sum() > 0


def test_uniform_image_has_no_edges():
    image = np.full((50, 50), 128, dtype=np.uint8)
    edges = extract_edges(image)
    assert edges.sum() == 0
