"""Project 3D LiDAR points onto the 2D image plane using known extrinsics + intrinsics."""

from __future__ import annotations

import numpy as np


def project_points(
    points_lidar: np.ndarray,
    T_cam_lidar: np.ndarray,
    K: np.ndarray,
    image_size: tuple[int, int] | None = None,
    min_depth: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D LiDAR points into the image plane via a pinhole camera model.

    Args:
        points_lidar: (N, 3) XYZ points in the LiDAR coordinate frame.
        T_cam_lidar: (4, 4) homogeneous SE(3) transform from LiDAR to camera frame.
        K: (3, 3) camera intrinsic matrix.
        image_size: optional (width, height); if given, points projecting outside
            image bounds are also filtered out.
        min_depth: minimum camera-frame z (metres) to keep a point. Points with
            depth below this (including behind the camera) are dropped.

    Returns:
        pixels:    (M, 2) (u, v) pixel coordinates of kept points.
        depths:    (M,) camera-frame z (metres) for each kept point.
        keep_mask: (N,) bool mask aligning kept points to the input.
                   `pixels[i]` corresponds to `points_lidar[keep_mask][i]`.
    """
    if points_lidar.ndim != 2 or points_lidar.shape[1] != 3:
        raise ValueError(f"points_lidar must be (N, 3), got {points_lidar.shape}")
    if T_cam_lidar.shape != (4, 4):
        raise ValueError(f"T_cam_lidar must be (4, 4), got {T_cam_lidar.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"K must be (3, 3), got {K.shape}")

    n = points_lidar.shape[0]
    homog = np.hstack([points_lidar, np.ones((n, 1))])
    points_cam = (T_cam_lidar @ homog.T).T[:, :3]

    depths = points_cam[:, 2]
    in_front = depths >= min_depth

    # Pinhole projection. Division is safe because we mask out points where
    # z is small/negative via `in_front` before consuming `pixels`.
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels_homog = (K @ points_cam.T).T
        pixels = pixels_homog[:, :2] / pixels_homog[:, 2:3]

    keep = in_front
    if image_size is not None:
        w, h = image_size
        in_bounds = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < w)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < h)
        )
        keep = keep & in_bounds

    return pixels[keep], depths[keep], keep
