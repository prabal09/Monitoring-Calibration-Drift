"""CLI: render a LiDAR-on-image overlay for a single NuScenes sample."""

import argparse

import cv2
from nuscenes.nuscenes import NuScenes

from calibration_drift.ingestion.nuscenes_loader import load_frame
from calibration_drift.projection.lidar_to_image import project_points
from calibration_drift.viz.overlay import draw_overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="/data/nuscenes-mini")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--sample-token", default=None,
                        help="Sample token to render. If omitted, uses the first sample in the dataset.")
    parser.add_argument("--camera", default="CAM_FRONT")
    parser.add_argument("--out", default="/overlays/overlay.png")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--max-depth", type=float, default=50.0)
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    sample_token = args.sample_token or nusc.sample[0]["token"]

    bundle = load_frame(nusc, sample_token, camera_channel=args.camera)
    h, w = bundle.image.shape[:2]

    pixels, depths, _ = project_points(
        bundle.lidar_points,
        bundle.T_cam_lidar,
        bundle.K,
        image_size=(w, h),
    )

    overlay = draw_overlay(bundle.image, pixels, depths,
                           radius=args.radius, max_depth=args.max_depth)

    cv2.imwrite(args.out, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"sample={sample_token} camera={args.camera} "
          f"projected={len(pixels)}/{len(bundle.lidar_points)} points  →  {args.out}")


if __name__ == "__main__":
    main()
