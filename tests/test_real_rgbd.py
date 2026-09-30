"""Independent geometry and failure tests for the real-data adapter."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from roomgraph.mapping import depth_points
from roomgraph.real_rgbd import (
    ARKitTrajectory,
    geometry_metrics,
    perturb,
    scaled_intrinsics,
    upright_record,
    voxel_fuse,
)


class RealRGBDTests(unittest.TestCase):
    def test_trajectory_inverts_vendor_world_to_camera(self):
        rotation = Rotation.from_euler("z", 90, degrees=True)
        rows = [[t, *rotation.as_rotvec(), 1, 2, 3] for t in (0, 0.1)]
        actual = ARKitTrajectory(rows).pose(0.05)
        expected = np.eye(4)
        expected[:3, :3] = rotation.as_matrix()
        expected[:3, 3] = [1, 2, 3]
        np.testing.assert_allclose(actual @ expected, np.eye(4), atol=1e-12)

    def test_interpolation_rotation_is_rigid_and_center_is_linear(self):
        rows = [[0, 0, 0, 0, 0, 0, 0], [0.1, 0, 0, np.pi / 2, 2, 0, 0]]
        pose = ARKitTrajectory(rows).pose(0.05)
        np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(pose[:3, 3], [0, 1, 0], atol=1e-12)

    def test_extrapolation_and_missing_pose_intervals_fail(self):
        trajectory = ARKitTrajectory([[0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]])
        with self.assertRaisesRegex(ValueError, "extrapolation"):
            trajectory.pose(-0.1)
        with self.assertRaisesRegex(ValueError, "gap"):
            trajectory.pose(0.5)

    def test_unsorted_or_nonfinite_trajectory_rejected(self):
        for times in ((1, 0), (0, 0), (0, np.nan)):
            with self.assertRaises(ValueError):
                ARKitTrajectory([[t, 0, 0, 0, 0, 0, 0] for t in times])

    def test_intrinsics_resize_preserves_camera_rays(self):
        k = np.array([[200, 0, 128], [0, 200, 96], [0, 0, 1]])
        target = scaled_intrinsics(k, (256, 192), (384, 256))
        pixel = np.array([36, 60, 1])
        resized = pixel * [1.5, 256 / 192, 1]
        np.testing.assert_allclose(np.linalg.inv(k) @ pixel, np.linalg.inv(target) @ resized)

    def test_backprojection_reprojection_matches_pixels(self):
        k = np.array([[200, 0, 2], [0, 190, 2], [0, 0, 1]])
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.1, 0.6]).as_matrix()
        pose[:3, 3] = [1, -3, 2]
        points, pixels = depth_points(
            np.full((4, 4), 2.5), {"intrinsics": k, "camera_to_world": pose}
        )
        camera = (points - pose[:3, 3]) @ pose[:3, :3]
        projected = camera @ k.T
        np.testing.assert_allclose(projected[:, :2] / projected[:, 2:], pixels, atol=1e-12)

    def test_all_upright_rotations_preserve_world_points(self):
        record = {
            "intrinsics": [[2, 0, 1], [0, 3, 0.5], [0, 0, 1]],
            "camera_to_world": np.eye(4).tolist(),
        }
        depth = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], float)
        original, _ = depth_points(depth, record)
        for turns in range(4):
            calibrated = upright_record(record, (4, 3), turns)
            points, _ = depth_points(np.rot90(depth, -turns), calibrated)
            expected = np.rot90(original.reshape(3, 4, 3), -turns).reshape(-1, 3)
            np.testing.assert_allclose(points, expected, atol=1e-12)

    def test_noise_is_reproducible_and_does_not_mutate_nominal(self):
        record = {"intrinsics": np.eye(3).tolist(), "camera_to_world": np.eye(4).tolist()}
        spec = {"translation_std_m": 0.02, "rotation_std_deg": 1}
        one = perturb(record, spec, 3, 10, 11)
        self.assertEqual(one, perturb(record, spec, 3, 10, 11))
        self.assertNotEqual(one, perturb(record, spec, 3, 10, 23))
        np.testing.assert_array_equal(record["camera_to_world"], np.eye(4))

    def test_drift_and_calibration_have_declared_endpoint(self):
        record = {"intrinsics": np.diag([100, 100, 1]), "camera_to_world": np.eye(4)}
        result = perturb(
            record, {"drift_translation_m": 0.1, "focal_bias_fraction": 0.02}, 9, 10, 0
        )
        self.assertAlmostEqual(result["camera_to_world"][0][3], 0.1)
        self.assertAlmostEqual(result["intrinsics"][0][0], 102)

    def test_pixels_in_one_frame_do_not_count_as_multiple_views(self):
        points = np.tile([0.011, 0.012, 0.013], (100, 1))
        self.assertEqual(len(voxel_fuse([points], min_views=2)["points"]), 0)
        result = voxel_fuse([points, points[:1]], min_views=2)
        np.testing.assert_array_equal(result["support"], [2])

    def test_fusion_does_not_assume_floor_height(self):
        points = np.array([[0, 0, -5], [1, 0, 10]])
        np.testing.assert_allclose(voxel_fuse([points])["points"], points)

    def test_precision_and_recall_distinguish_partial_coverage(self):
        reference = np.array([[0, 0, 0], [1, 0, 0]])
        result = geometry_metrics(reference[:1], reference)
        self.assertEqual(result["precision_10cm"], 1)
        self.assertEqual(result["recall_10cm"], 0.5)
        self.assertEqual(geometry_metrics(np.empty((0, 3)), reference)["f1_10cm"], 0)


if __name__ == "__main__":
    unittest.main()
