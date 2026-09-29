"""Analytical checks for camera conventions and architectural visibility."""

import unittest

import numpy as np

from roomgraph.geometry import Box, Edge, demo_room, edge_masks, look_at, project, ray_depth


class GeometryTests(unittest.TestCase):
    def test_camera_basis_and_projection(self):
        pose = look_at(np.array([0, 0, 1]), np.array([0, 1, 1]))
        np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3))
        self.assertAlmostEqual(np.linalg.det(pose[:3, :3]), 1)
        intrinsics = np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]])
        pixels, depth = project(np.array([[0, 2, 1], [1, 2, 2], [0, -1, 1]]), intrinsics, pose)
        np.testing.assert_allclose(pixels[:2], [[50, 40], [100, -10]])
        self.assertTrue(np.isnan(pixels[2]).all())
        np.testing.assert_allclose(depth, [2, 2, -1])

    def test_parallel_rays_and_inside_origin(self):
        box = Box("test", (1, -1, -1), (2, 1, 1), (1, 1, 1))
        depths = ray_depth([0, 0, 0], [[1, 0, 0], [0, 1, 0], [-1, 0, 0]], [box])
        np.testing.assert_allclose(depths, [1, np.inf, np.inf])
        self.assertAlmostEqual(float(ray_depth([1.5, 0, 0], [1, 0, 0], [box])), 0.5)

    def test_door_and_window_are_open(self):
        boxes, edges = demo_room()
        self.assertEqual(len(edges), 20)
        origin = np.array([0, 0, 1.5])
        rays = np.array([[-1.5, 2.5, 0], [3, 0, 0], [0, 2.5, 0], [0, 0, -1]])
        np.testing.assert_allclose(ray_depth(origin, rays, boxes), [np.inf, np.inf, 1, 1.5])

    def test_invalid_look_at(self):
        with self.assertRaises(ValueError):
            look_at(np.zeros(3), np.zeros(3))
        with self.assertRaises(ValueError):
            look_at(np.zeros(3), np.array([0, 0, 1]))

    def test_occluder_hides_edge_but_keeps_full_projection(self):
        pose = look_at(np.array([0, 0, 1]), np.array([0, 1, 1]))
        intrinsics = np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]])
        edge = Edge("test", (-0.5, 2, 1), (0.5, 2, 1), "wall_wall")
        occluder = Box("cabinet", (-1, 0.5, 0), (1, 1, 2), (1, 1, 1))
        visible, full = edge_masks([edge], [occluder], intrinsics, pose, 100, 80)
        self.assertEqual(np.count_nonzero(visible), 0)
        self.assertGreater(np.count_nonzero(full), 40)
        visible, clear_full = edge_masks([edge], [], intrinsics, pose, 100, 80)
        np.testing.assert_array_equal(visible, full)
        np.testing.assert_array_equal(clear_full, full)


if __name__ == "__main__":
    unittest.main()
