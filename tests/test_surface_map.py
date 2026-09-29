"""Observed RGB-D geometry must respect depth semantics and missing data."""

import unittest

import numpy as np

from roomgraph.surface_map import SurfaceMap


class SurfaceMapTests(unittest.TestCase):
    def test_optical_depth_backprojects_off_axis_without_ray_normalization(self):
        record = {"intrinsics": np.eye(3), "camera_to_world": np.eye(4)}
        depth = np.array([[2.0, 2.0]])
        surface = SurfaceMap(0.01)
        surface.integrate(np.zeros((1, 2, 3)), depth, record, stride=1)
        np.testing.assert_allclose(surface.arrays()["points"], [[0, 0, 2], [2, 0, 2]])

    def test_invalid_depth_cannot_create_geometry(self):
        record = {"intrinsics": np.eye(3), "camera_to_world": np.eye(4)}
        surface = SurfaceMap()
        surface.integrate(np.zeros((1, 4, 3)), np.array([[0, np.inf, np.nan, -1]]), record)
        self.assertEqual(len(surface.arrays()["points"]), 0)

    def test_repeated_pixels_in_one_voxel_count_as_one_view(self):
        record = {"intrinsics": np.diag([1000, 1000, 1]), "camera_to_world": np.eye(4)}
        surface = SurfaceMap(0.1)
        for _ in range(2):
            surface.integrate(np.zeros((1, 2, 3)), np.array([[2.0, 2.0]]), record, stride=1)
        np.testing.assert_array_equal(surface.arrays()["support"], [2])


if __name__ == "__main__":
    unittest.main()
