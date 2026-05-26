"""Reprojection-error metrics between projected LiDAR points and image edges."""

from __future__ import annotations

import cv2
import numpy as np


def mean_reprojection_error(image_edges: np.ndarray, projected_pixels: np.ndarray) -> float:
    """Mean pixel-distance from projected LiDAR points to the nearest image edge.

    The signal: if calibration is correct, projected LiDAR points (especially
    depth discontinuities like object boundaries) should land on or near image
    edges. As calibration drifts, the mean distance grows. We use the distance
    transform of the edge mask so each lookup is O(1).

    Args:
        image_edges: (H, W) uint8 edge mask (255 at edges, 0 elsewhere) —
            output of `extract_edges`.
        projected_pixels: (N, 2) (u, v) projected LiDAR pixel coordinates.

    Returns:
        Mean distance in pixels. NaN if there are no edges in the image OR no
        projected pixels (no signal to measure).
    """
    if image_edges.sum() == 0 or len(projected_pixels) == 0:
        return float("nan")

    # cv2.distanceTransform expects 0=background, non-zero=foreground.
    # We want distance from each pixel TO the nearest edge, so invert the mask.
    non_edge = (image_edges == 0).astype(np.uint8)
    dist_to_edge = cv2.distanceTransform(non_edge, cv2.DIST_L2, 5)

    h, w = image_edges.shape
    u = np.clip(projected_pixels[:, 0].astype(int), 0, w - 1)
    v = np.clip(projected_pixels[:, 1].astype(int), 0, h - 1)

    return float(dist_to_edge[v, u].mean())
