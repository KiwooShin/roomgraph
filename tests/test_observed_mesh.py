"""Observed wall patches preserve unsupported openings and evidence boundaries."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from roomgraph.observed_mesh import observed_wall_patches, write_wall_obj


def wall_points(with_door=True):
    y, z = np.meshgrid(np.arange(-2, 2, 0.05) + 0.025, np.arange(0, 3, 0.05) + 0.025)
    valid = ~((np.abs(y) < 0.55) & (z < 2.1)) if with_door else np.ones(y.shape, bool)
    return np.column_stack((np.ones(valid.sum()), y[valid], z[valid]))


class ObservedMeshTests(unittest.TestCase):
    def test_open_door_grid_cells_remain_absent(self):
        points = wall_points()
        mesh = observed_wall_patches(points, np.full(len(points), 3))
        self.assertGreater(len(mesh["faces"]), 50)
        centers = mesh["vertices"][mesh["faces"]].mean(axis=1)
        opening = (np.abs(centers[:, 1]) < 0.4) & (centers[:, 2] < 1.9)
        self.assertFalse(np.any(opening))
        self.assertTrue(np.any((np.abs(centers[:, 1]) < 0.4) & (centers[:, 2] > 2.2)))
        self.assertTrue(all(plane["normal_axis"] == "x" for plane in mesh["planes"]))

    def test_unknown_patch_is_not_interpolated_and_single_view_is_unsupported(self):
        points = wall_points(False)
        hole = (
            (points[:, 1] >= 0.6)
            & (points[:, 1] < 0.75)
            & (points[:, 2] >= 1.2)
            & (points[:, 2] < 1.35)
        )
        mesh = observed_wall_patches(points[~hole], np.full((~hole).sum(), 2))
        centers = mesh["vertices"][mesh["faces"]].mean(axis=1)
        self.assertFalse(np.any(np.linalg.norm(centers - [1, 0.675, 1.275], axis=1) < 0.01))
        single = observed_wall_patches(points, np.ones(len(points)))
        self.assertEqual(single["faces"].shape, (0, 4))

    def test_short_furniture_and_floor_are_not_tall_wall_candidates(self):
        points = wall_points(False)
        short = points[points[:, 2] < 1.0]
        floor = points[:, [2, 1, 0]].copy()
        floor[:, 2] = 0
        points = np.vstack((short, floor))
        mesh = observed_wall_patches(points, np.full(len(points), 4))
        self.assertEqual(len(mesh["faces"]), 0)

    def test_plane_evidence_and_obj_indices(self):
        points = wall_points()
        mesh = observed_wall_patches(points, np.full(len(points), 3))
        self.assertEqual(len(mesh["planes"]), 1)
        plane = mesh["planes"][0]
        self.assertAlmostEqual(plane["offset_m"], 1)
        self.assertEqual(plane["mean_view_support"], 3)
        self.assertGreater(plane["evidence_score"], 0)
        self.assertLessEqual(plane["evidence_score"], 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observed.obj"
            write_wall_obj(path, mesh)
            lines = path.read_text().splitlines()
            self.assertEqual(sum(line.startswith("f ") for line in lines), len(mesh["faces"]))
            self.assertTrue(
                all(min(map(int, line.split()[1:])) >= 1 for line in lines if line.startswith("f "))
            )

    def test_invalid_points_and_empty_arrays(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            observed_wall_patches(np.full((3, 3), np.nan))
        mesh = observed_wall_patches(np.empty((0, 3)))
        self.assertEqual(mesh["vertices"].shape, (0, 3))
        self.assertEqual(mesh["faces"].shape, (0, 4))


if __name__ == "__main__":
    unittest.main()
