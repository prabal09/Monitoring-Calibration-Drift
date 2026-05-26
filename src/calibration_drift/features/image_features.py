"""2D visual feature extraction (edges, semantic boundaries) via OpenCV."""

from __future__ import annotations

import cv2
import numpy as np


def extract_edges(image: np.ndarray, canny_low: int = 50, canny_high: int = 150) -> np.ndarray:
    """Return a binary edge mask of an image via Canny edge detection.

    Args:
        image: (H, W, 3) RGB or (H, W) grayscale uint8 image.
        canny_low: lower hysteresis threshold for Canny.
        canny_high: upper hysteresis threshold for Canny.

    Returns:
        (H, W) uint8 mask with 255 at edge pixels and 0 elsewhere.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return cv2.Canny(gray, canny_low, canny_high)
