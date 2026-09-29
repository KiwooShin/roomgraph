"""Scene reproducibility, mesh geometry, visibility, and camera-map conventions."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from roomgraph.geometry import Edge, depth_edge_masks, look_at
from roomgraph.meshes import rounded_box
from roomgraph.scene_config import load_scene
from roomgraph.visualization import camera_triangle, floorplan_pixel

ROOT = Path(__file__).resolve().parents[1]


class FurnishedTests(unittest.TestCase):
    def test_scene_configs_are_reproducible(self):
        for path in (ROOT / "configs/scenes").glob("*.json"):
            with self.subTest(scene=path.stem):
                config, first = load_scene(path)
                _, second = load_scene(path)
                self.assertEqual(first, second)
                self.assertGreaterEqual(len({p.object_id for p in first}), 15)
                self.assertEqual(len(config["cameras"]), 3)
                self.assertTrue(all(p.center[2] > 0 for p in first))

    def test_unknown_recipe_rejected(self):
        config = json.loads((ROOT / "configs/scenes/bedroom.json").read_text())
        config["furnishings"][0]["recipe"] = "unrecognized"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "Unknown furnishing recipe"):
                load_scene(path)

    def test_rounded_box_bounds_normals_and_winding(self):
        points, normals, uv, indices = rounded_box((2, 1, 0.4), 0.08)
        self.assertTrue(np.all(np.abs(points) <= np.array([1, 0.5, 0.2]) + 1e-8))
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)
        self.assertEqual(uv.shape, (len(points), 2))
        faces = points[indices.reshape(-1, 4)]
        cross = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        self.assertTrue(np.all(np.sum(cross * faces.mean(axis=1), axis=1) > 0))

    def test_camera_triangle_points_toward_target(self):
        origin, direction, vertices = camera_triangle([1, 2, 1.5], [2, 3, 1])
        np.testing.assert_allclose(origin, [1, 2])
        np.testing.assert_allclose(direction, [1 / np.sqrt(2), 1 / np.sqrt(2)])
        self.assertGreater(np.dot(vertices[0] - origin, direction), 0)
        self.assertLess(np.dot(vertices[1] - origin, direction), 0)
        self.assertLess(floorplan_pixel([0, 1], 800, 600, 8)[1], 300)
        np.testing.assert_allclose(floorplan_pixel([0, 0], 800, 600, 8), [400, 300])
        with self.assertRaises(ValueError):
            camera_triangle([0, 0, 1], [0, 0, 0])

    def test_depth_visibility_with_foreground_occluder(self):
        edge = Edge("wall", (-0.5, 2, 1), (0.5, 2, 1), "wall_wall")
        pose = look_at(np.array([0, 0, 1]), np.array([0, 1, 1]))
        intrinsic = np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]])
        depth = np.full((80, 100), 2.0)
        visible, full = depth_edge_masks([edge], intrinsic, pose, depth)
        np.testing.assert_array_equal(visible, full)
        depth[:, 40:60] = 1.0
        occluded, projected = depth_edge_masks([edge], intrinsic, pose, depth)
        self.assertEqual(np.count_nonzero(occluded[:, 40:60]), 0)
        self.assertGreater(np.count_nonzero(occluded), 0)
        np.testing.assert_array_equal(projected, full)


if __name__ == "__main__":
    unittest.main()
