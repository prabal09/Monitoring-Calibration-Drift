from pathlib import Path

import numpy as np
import pytest

DATAROOT = "/data/nuscenes-mini"
HAS_DATA = Path(DATAROOT, "v1.0-mini").is_dir()


def test_se3_from_record_identity():
    from calibration_drift.ingestion.nuscenes_loader import _se3_from_record

    T = _se3_from_record({
        "rotation": [1.0, 0.0, 0.0, 0.0],   # identity quaternion (w, x, y, z)
        "translation": [0.0, 0.0, 0.0],
    })
    np.testing.assert_allclose(T, np.eye(4), atol=1e-12)


def test_se3_from_record_translation_only():
    from calibration_drift.ingestion.nuscenes_loader import _se3_from_record

    T = _se3_from_record({
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "translation": [1.0, 2.0, 3.0],
    })
    expected = np.eye(4)
    expected[:3, 3] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(T, expected, atol=1e-12)


@pytest.mark.skipif(not HAS_DATA, reason=f"no nuscenes-mini at {DATAROOT}")
def test_load_frame_returns_well_formed_bundle():
    from nuscenes.nuscenes import NuScenes

    from calibration_drift.ingestion.nuscenes_loader import FrameBundle, load_frame

    nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=False)
    sample_token = nusc.sample[0]["token"]

    bundle = load_frame(nusc, sample_token)

    assert isinstance(bundle, FrameBundle)
    assert bundle.lidar_points.ndim == 2 and bundle.lidar_points.shape[1] == 3
    assert bundle.lidar_points.shape[0] > 1000   # LiDAR sweeps are dense
    assert bundle.image.ndim == 3 and bundle.image.shape[2] == 3
    assert bundle.K.shape == (3, 3)
    assert bundle.T_cam_lidar.shape == (4, 4)
    np.testing.assert_allclose(bundle.T_cam_lidar[3, :], [0, 0, 0, 1], atol=1e-6)
    assert bundle.timestamp > 0
    assert bundle.sample_token == sample_token


@pytest.mark.skipif(not HAS_DATA, reason=f"no nuscenes-mini at {DATAROOT}")
def test_load_frame_feeds_projection_pipeline():
    from nuscenes.nuscenes import NuScenes

    from calibration_drift.ingestion.nuscenes_loader import load_frame
    from calibration_drift.projection.lidar_to_image import project_points

    nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=False)
    sample_token = nusc.sample[0]["token"]

    bundle = load_frame(nusc, sample_token)
    h, w = bundle.image.shape[:2]

    pixels, depths, keep = project_points(
        bundle.lidar_points,
        bundle.T_cam_lidar,
        bundle.K,
        image_size=(w, h),
    )

    # A NuScenes front camera frame should have hundreds to thousands of LiDAR
    # points projecting into the image with positive depth.
    assert keep.sum() > 100
    assert (pixels[:, 0] >= 0).all() and (pixels[:, 0] < w).all()
    assert (pixels[:, 1] >= 0).all() and (pixels[:, 1] < h).all()
    assert (depths > 0).all()
