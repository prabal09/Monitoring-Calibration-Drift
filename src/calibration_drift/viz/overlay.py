"""Render projected LiDAR points over camera images for visual sanity checks."""

from __future__ import annotations

import cv2
import numpy as np


_CMAPS = {
    "jet": cv2.COLORMAP_JET,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "turbo": cv2.COLORMAP_TURBO,
}


def draw_overlay(
    image: np.ndarray,
    pixels: np.ndarray,
    depths: np.ndarray,
    *,
    radius: int = 2,
    cmap: str = "jet",
    max_depth: float = 50.0,
) -> np.ndarray:
    """Draw projected LiDAR points on a camera image, colored by depth.

    Args:
        image: (H, W, 3) RGB camera image. Not modified.
        pixels: (N, 2) projected (u, v) pixel coordinates.
        depths: (N,) camera-frame depth (metres) for each pixel; used to
            color the dots.
        radius: dot radius in pixels.
        cmap: one of "jet", "viridis", "turbo".
        max_depth: depths are clipped to [0, max_depth] before colormapping,
            so points closer than 0 m or farther than this become the colormap
            endpoints rather than washing out the gradient.

    Returns:
        (H, W, 3) RGB image with dots drawn at each (u, v).
    """
    out = image.copy()
    if len(pixels) == 0:
        return out

    norm = np.clip(depths / max_depth, 0.0, 1.0)
    norm_uint8 = (norm * 255).astype(np.uint8).reshape(-1, 1)
    colors_bgr = cv2.applyColorMap(norm_uint8, _CMAPS[cmap]).reshape(-1, 3)
    colors_rgb = colors_bgr[:, ::-1]

    for (u, v), color in zip(pixels.astype(int), colors_rgb):
        cv2.circle(out, (int(u), int(v)), radius, tuple(int(c) for c in color), -1)

    return out
