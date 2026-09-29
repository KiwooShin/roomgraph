"""Leakage, sparse-boundary scoring, model gradients, and calibrated reconstruction."""

import importlib.util
import unittest

import numpy as np

from roomgraph.geometry import Edge, depth_edge_masks, look_at, scaled_room
from roomgraph.headcam import robot_pose

ML_AVAILABLE = all(
    importlib.util.find_spec(name) for name in ["torch", "scipy", "skimage", "torchvision"]
)


class HeadCameraTests(unittest.TestCase):
    def test_body_mount_and_scaled_shell(self):
        camera = look_at([1, 2, 1.585], [-1, 1, 1.0])
        body = robot_pose(camera)
        local = np.linalg.inv(body) @ np.r_[camera[:3, 3], 1]
        np.testing.assert_allclose(local[:3], [0, 0.135, 1.585], atol=1e-8)
        self.assertAlmostEqual(body[2, 3], 0)
        _, edges = scaled_room([7.2, 6, 3.3])
        points = np.array([e.start for e in edges])
        np.testing.assert_allclose(points.min(0), [-3.6, -3, 0])
        np.testing.assert_allclose(points.max(0), [3.6, 3, 3.3])


@unittest.skipUnless(ML_AVAILABLE, "Install the learning dependencies to run model tests")
class LearningTests(unittest.TestCase):
    def test_group_leakage_rejected(self):
        from roomgraph.learning.data import validate_groups

        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_groups(
                [
                    {"building_id": "same_room", "split": "train"},
                    {"building_id": "same_room", "split": "test"},
                ]
            )

    def test_boundary_metrics_empty_and_displaced(self):
        from roomgraph.learning.metrics import boundary_counts, summarize_counts

        target = np.zeros((32, 32), bool)
        target[5:25, 10] = True
        close = np.roll(target, 2, axis=1)
        far = np.roll(target, 3, axis=1)
        self.assertEqual(summarize_counts(boundary_counts(close, target, 2))["f1"], 1)
        self.assertEqual(summarize_counts(boundary_counts(far, target, 2))["f1"], 0)
        self.assertEqual(
            summarize_counts(boundary_counts(np.zeros_like(target), target))["recall"], 0
        )
        self.assertEqual(
            summarize_counts(boundary_counts(target, np.zeros_like(target)))["precision"], 0
        )

    def test_reconstruction_requires_multiple_views(self):
        from roomgraph.learning.reconstruction import fit_room

        with self.assertRaisesRegex(ValueError, "at least three"):
            fit_room(np.zeros((1, 6, 32, 32)), [{}])

    def test_model_output_and_gradient(self):
        import torch

        from roomgraph.learning.model import EdgeNet, edge_loss

        torch.set_num_threads(2)
        model = EdgeNet(pretrained=False)
        output = model(torch.rand(2, 3, 64, 96))
        self.assertEqual(tuple(output.shape), (2, 6, 64, 96))
        target = torch.zeros_like(output)
        target[:, :, :, 30] = 1
        loss = edge_loss(output, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0)

    def test_oracle_reconstruction_without_depth(self):
        from roomgraph.learning.reconstruction import cuboid_edges, fit_room, reconstruction_metrics

        bounds = [-3.2, 3.1, -2.7, 2.4, 0, 3.2]
        segments, kinds = cuboid_edges(bounds)
        records = []
        probabilities = []
        intrinsic = np.array([[85, 0, 96], [0, 85, 64], [0, 0, 1]])
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            pose = look_at([1.9 * np.cos(angle), 1.5 * np.sin(angle), 1.585], [0, 0, 1.2])
            record = {"intrinsics": intrinsic.tolist(), "camera_to_world": pose.tolist()}
            records.append(record)
            maps = np.zeros((6, 128, 192), np.float32)
            for channel in [1, 2, 3]:
                edges = [
                    Edge(str(i), tuple(e[0]), tuple(e[1]), "test")
                    for i, (e, k) in enumerate(zip(segments, kinds, strict=True))
                    if k == channel
                ]
                _, mask = depth_edge_masks(edges, intrinsic, pose, np.full((128, 192), np.inf))
                maps[channel] = mask > 0
            probabilities.append(maps)
        estimate = fit_room(np.stack(probabilities), records, maxiter=65)
        metrics = reconstruction_metrics(estimate["bounds_m"], bounds)
        self.assertLess(metrics["dimension_mae_m"], 0.08)
        self.assertLess(metrics["edge_chamfer_m"], 0.06)


if __name__ == "__main__":
    unittest.main()
