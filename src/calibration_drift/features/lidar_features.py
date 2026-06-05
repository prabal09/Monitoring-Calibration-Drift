"""3D geometric feature extraction (depth discontinuities, surface normals) via Open3D."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def extract_depth_edges(
    pixels: np.ndarray,
    depths: np.ndarray,
    k: int = 5,
    depth_threshold: float = 0.5,
) -> np.ndarray:
    """Boolean mask marking projected LiDAR points at depth discontinuities.

    For each projected point, looks up its `k` nearest neighbors in pixel space
    and flags the point as a depth edge if the maximum absolute depth
    difference to any neighbor exceeds `depth_threshold` metres. These are
    the points that *should* coincide with image edges if calibration is
    correct — silhouettes of cars against road, building edges against sky,
    etc. — so they're the only ones whose distance-to-image-edge encodes
    real calibration error.

    Args:
        pixels: (N, 2) projected (u, v) pixel coordinates.
        depths: (N,) camera-frame z in metres for each pixel.
        k: number of nearest neighbors to compare against in pixel space.
        depth_threshold: minimum max-neighbor depth gap (metres) to qualify
            as a depth edge.

    Returns:
        (N,) bool mask. mask[i] == True means pixels[i] is at a depth
        discontinuity.
    """
    n = len(pixels)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if n == 1:
        # A single point has no neighbors — it can't be classified as an edge.
        return np.zeros(1, dtype=bool)

    # k+1 because the closest neighbor of each point is itself.
    k_eff = min(k + 1, n)
    tree = cKDTree(pixels)
    _, idx = tree.query(pixels, k=k_eff)

    # idx[:, 0] is the point itself; idx[:, 1:] are the actual neighbors.
    neighbor_idx = idx[:, 1:]
    neighbor_depths = depths[neighbor_idx]
    own_depths = depths[:, np.newaxis]

    max_gap = np.max(np.abs(neighbor_depths - own_depths), axis=1)
    return max_gap > depth_threshold
