"""CLI: walk a NuScenes scene, compute reprojection error per frame, detect drift."""

import argparse

import numpy as np
import yaml
from nuscenes.nuscenes import NuScenes

from calibration_drift.drift.detector import DriftDetector
from calibration_drift.drift.error_metrics import mean_reprojection_error
from calibration_drift.features.image_features import extract_edges
from calibration_drift.features.lidar_features import extract_depth_edges
from calibration_drift.ingestion.nuscenes_loader import load_frame
from calibration_drift.projection.lidar_to_image import project_points


def yaw_perturbation(yaw_deg: float) -> np.ndarray:
    """Build a 4x4 SE(3) rotation around the camera's y-axis (yaw)."""
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c,  0.0, s,   0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c,   0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="/data/nuscenes-mini")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene", type=int, default=0, help="Scene index (0-9 in mini).")
    parser.add_argument("--camera", default="CAM_FRONT")
    parser.add_argument("--config", default="configs/drift_thresholds.yaml")
    parser.add_argument("--perturb-yaw-deg", type=float, default=0.0,
                        help="Constant yaw offset (degrees) applied every frame — simulates a step-impact drift event.")
    parser.add_argument("--perturb-yaw-deg-per-frame", type=float, default=0.0,
                        help="Additional yaw drift accumulated per frame — simulates gradual mechanical degradation.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    scene = nusc.scene[args.scene]

    detector = DriftDetector(
        window_size=config["temporal_window"]["num_frames"],
        slope_threshold=config["temporal_window"]["slope_threshold"],
    )

    print(f"Scene: {scene['name']} ({scene['nbr_samples']} samples), camera={args.camera}")
    if args.perturb_yaw_deg or args.perturb_yaw_deg_per_frame:
        print(f"Injecting yaw drift: static={args.perturb_yaw_deg}°, "
              f"per-frame=+{args.perturb_yaw_deg_per_frame}°/frame")
    print()

    token = scene["first_sample_token"]
    frame_idx = 0
    while token:
        bundle = load_frame(nusc, token, camera_channel=args.camera)

        # Total yaw drift at this frame = static offset + cumulative per-frame.
        total_yaw_deg = args.perturb_yaw_deg + frame_idx * args.perturb_yaw_deg_per_frame
        if total_yaw_deg:
            T = yaw_perturbation(total_yaw_deg) @ bundle.T_cam_lidar
        else:
            T = bundle.T_cam_lidar

        h, w = bundle.image.shape[:2]
        pixels, depths, _ = project_points(bundle.lidar_points, T, bundle.K, image_size=(w, h))

        # C1: restrict the error metric to projected LiDAR points at depth
        # discontinuities (object silhouettes) — these are the points that
        # *should* coincide with image edges if calibration is correct.
        # Surface-hit points are excluded because their distance-to-edge
        # encodes scene geometry, not calibration quality.
        depth_edge_mask = extract_depth_edges(
            pixels, depths,
            k=config["edge_matching"].get("lidar_depth_edge_k", 5),
            depth_threshold=config["edge_matching"]["lidar_depth_grad_threshold"],
        )
        edge_pixels = pixels[depth_edge_mask]

        edges = extract_edges(
            bundle.image,
            canny_low=config["edge_matching"]["canny_low"],
            canny_high=config["edge_matching"]["canny_high"],
        )
        error = mean_reprojection_error(edges, edge_pixels)
        detector.update(error)

        flag = "DRIFT" if detector.is_drifting() else "ok   "
        print(f"[{frame_idx:3d}] error={error:5.2f}px  slope={detector.slope():+.3f} px/frame  "
              f"yaw={total_yaw_deg:+5.2f}°  n_edge_pts={depth_edge_mask.sum():4d}  {flag}")

        token = nusc.get("sample", token)["next"]
        frame_idx += 1

    print()
    print(f"Final state: {'DRIFTING' if detector.is_drifting() else 'OK'}  "
          f"(slope={detector.slope():+.3f} px/frame, threshold={detector.slope_threshold})")


if __name__ == "__main__":
    main()
