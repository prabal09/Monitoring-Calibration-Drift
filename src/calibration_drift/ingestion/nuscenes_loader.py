"""Load synchronized LiDAR / camera samples from the NuScenes dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


@dataclass
class FrameBundle:
    """Single time-aligned (LiDAR, camera) frame ready for projection."""

    lidar_points: np.ndarray   # (N, 3) XYZ in the LiDAR frame
    image: np.ndarray          # (H, W, 3) RGB camera image
    K: np.ndarray              # (3, 3) camera intrinsic matrix
    T_cam_lidar: np.ndarray    # (4, 4) SE(3) LiDAR → camera, ego-motion corrected
    timestamp: int             # microseconds (NuScenes convention)
    sample_token: str          # NuScenes sample token, for traceability


def _se3_from_record(record: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Quaternion(record["rotation"]).rotation_matrix
    T[:3, 3] = record["translation"]
    return T


def load_frame(
    nusc: NuScenes,
    sample_token: str,
    camera_channel: str = "CAM_FRONT",
) -> FrameBundle:
    """Load one (LiDAR, camera) bundle from a NuScenes sample.

    Args:
        nusc: An initialized NuScenes instance (`dataroot` points at the data).
        sample_token: A keyframe sample token.
        camera_channel: One of CAM_FRONT, CAM_FRONT_LEFT, ..., CAM_BACK_RIGHT.

    Returns:
        FrameBundle with LiDAR points in the LiDAR frame, the camera image,
        the camera intrinsics, and an SE(3) transform from LiDAR to camera
        that accounts for ego-vehicle motion between the LiDAR sweep and the
        camera exposure (they are captured at slightly different instants).
    """
    sample = nusc.get("sample", sample_token)
    lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    cam_data = nusc.get("sample_data", sample["data"][camera_channel])

    # LiDAR .pcd.bin is float32 with 5 channels: x, y, z, intensity, ring.
    lidar_path = Path(nusc.dataroot) / lidar_data["filename"]
    points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 5)
    lidar_points = points[:, :3].copy()

    cam_path = Path(nusc.dataroot) / cam_data["filename"]
    image_bgr = cv2.imread(str(cam_path))
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    cam_calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    K = np.array(cam_calib["camera_intrinsic"])

    # Chain: LiDAR -> ego(t_lidar) -> world -> ego(t_cam) -> camera.
    # The two ego_pose records differ because the vehicle moves between the
    # LiDAR sweep and the camera exposure.
    T_ego_lidar = _se3_from_record(
        nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    )
    T_world_ego_at_lidar = _se3_from_record(
        nusc.get("ego_pose", lidar_data["ego_pose_token"])
    )
    T_world_ego_at_cam = _se3_from_record(
        nusc.get("ego_pose", cam_data["ego_pose_token"])
    )
    T_ego_cam = _se3_from_record(cam_calib)

    T_cam_lidar = (
        np.linalg.inv(T_ego_cam)
        @ np.linalg.inv(T_world_ego_at_cam)
        @ T_world_ego_at_lidar
        @ T_ego_lidar
    )

    return FrameBundle(
        lidar_points=lidar_points,
        image=image,
        K=K,
        T_cam_lidar=T_cam_lidar,
        timestamp=sample["timestamp"],
        sample_token=sample_token,
    )
