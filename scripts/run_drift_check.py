"""CLI: walk a NuScenes scene, compute reprojection error per frame, detect drift."""

import argparse

import numpy as np
import yaml
from nuscenes.nuscenes import NuScenes

from calibration_drift.drift.detector import DriftDetector
from calibration_drift.drift.error_metrics import mean_reprojection_error
from calibration_drift.features.image_features import extract_edges
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
                        help="Inject this much yaw drift (degrees) into T_cam_lidar to simulate calibration drift.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    scene = nusc.scene[args.scene]

    detector = DriftDetector(
        window_size=config["temporal_window"]["num_frames"],
        slope_threshold=config["temporal_window"]["slope_threshold"],
    )

    perturb = yaw_perturbation(args.perturb_yaw_deg) if args.perturb_yaw_deg else None

    print(f"Scene: {scene['name']} ({scene['nbr_samples']} samples), camera={args.camera}")
    if perturb is not None:
        print(f"Injecting {args.perturb_yaw_deg}° yaw drift into T_cam_lidar")
    print()

    token = scene["first_sample_token"]
    frame_idx = 0
    while token:
        bundle = load_frame(nusc, token, camera_channel=args.camera)

        T = perturb @ bundle.T_cam_lidar if perturb is not None else bundle.T_cam_lidar

        h, w = bundle.image.shape[:2]
        pixels, _, _ = project_points(bundle.lidar_points, T, bundle.K, image_size=(w, h))

        edges = extract_edges(
            bundle.image,
            canny_low=config["edge_matching"]["canny_low"],
            canny_high=config["edge_matching"]["canny_high"],
        )
        error = mean_reprojection_error(edges, pixels)
        detector.update(error)

        flag = "DRIFT" if detector.is_drifting() else "ok   "
        print(f"[{frame_idx:3d}] error={error:5.2f}px  slope={detector.slope():+.3f} px/frame  {flag}")

        token = nusc.get("sample", token)["next"]
        frame_idx += 1

    print()
    print(f"Final state: {'DRIFTING' if detector.is_drifting() else 'OK'}  "
          f"(slope={detector.slope():+.3f} px/frame, threshold={detector.slope_threshold})")


if __name__ == "__main__":
    main()
