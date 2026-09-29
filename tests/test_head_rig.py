"""Articulated head extrinsics and compatibility without a rendering dependency."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from roomgraph.geometry import look_at
from roomgraph.headcam import (
    CAMERA_OFFSET_M,
    HEAD_PART_NAMES,
    NECK_PIVOT_M,
    camera_rig_pose,
    head_camera_pose,
    proxy_parts,
    resolve_camera_pose,
    robot_pose,
)
from roomgraph.scene_config import load_scene

ROOT = Path(__file__).resolve().parents[1]


class HeadRigTests(unittest.TestCase):
    def test_neutral_pose_agrees_with_legacy_camera_and_body(self):
        rig = head_camera_pose([0.3, -0.4, 0], base_yaw_deg=34)
        camera = rig.camera_to_world
        np.testing.assert_allclose(
            camera, look_at(camera[:3, 3], camera[:3, 3] + camera[:3, 2]), atol=1e-12
        )
        np.testing.assert_allclose(rig.base_to_world, robot_pose(camera), atol=1e-12)
        np.testing.assert_allclose(rig.head_to_base, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(camera[2, 3], 1.585)

    def test_positive_yaw_turns_left_and_positive_pitch_looks_up(self):
        left = head_camera_pose([0, 0, 0], head_yaw_deg=90)
        up = head_camera_pose([0, 0, 0], head_pitch_deg=30)
        np.testing.assert_allclose(left.camera_to_world[:3, 2], [-1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(up.camera_to_world[:3, 2], [0, np.sqrt(3) / 2, 0.5])
        combined = head_camera_pose([0, 0, 0], base_yaw_deg=90, head_yaw_deg=-90)
        np.testing.assert_allclose(combined.camera_to_world[:3, 2], [0, 1, 0], atol=1e-12)

    def test_camera_moves_around_neck_and_stays_attached_to_head(self):
        rig = head_camera_pose([0, 0, 0], head_pitch_deg=30)
        pose = rig.camera_to_world
        c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
        expected = [0, 0.135 * c - 0.155 * s, 1.43 + 0.135 * s + 0.155 * c]
        np.testing.assert_allclose(pose[:3, 3], expected)
        np.testing.assert_allclose(
            rig.head_to_base @ np.r_[NECK_PIVOT_M, 1], np.r_[NECK_PIVOT_M, 1]
        )
        mounted = rig.base_to_world @ rig.head_to_base @ np.r_[NECK_PIVOT_M + CAMERA_OFFSET_M, 1]
        np.testing.assert_allclose(mounted[:3], pose[:3, 3])

    def test_only_head_geometry_articulates_with_fixed_body(self):
        neutral = head_camera_pose([0.7, -0.2, 0], base_yaw_deg=17)
        turned = head_camera_pose(
            [0.7, -0.2, 0], base_yaw_deg=17, head_yaw_deg=55, head_pitch_deg=-25
        )
        np.testing.assert_array_equal(neutral.base_to_world, turned.base_to_world)
        moved, fixed = set(), set()
        for part in proxy_parts():
            local = np.r_[part.center, 1]
            if part.name in HEAD_PART_NAMES:
                first = neutral.base_to_world @ neutral.head_to_base @ local
                second = turned.base_to_world @ turned.head_to_base @ local
            else:
                first = neutral.base_to_world @ local
                second = turned.base_to_world @ local
            (fixed if np.allclose(first, second) else moved).add(part.name)
        self.assertEqual(moved, HEAD_PART_NAMES)
        self.assertTrue({"torso", "neck", "left_forearm", "right_foot"}.issubset(fixed))

    def test_extreme_orientations_keep_valid_cv_extrinsics(self):
        for yaw, pitch in [(0, 90), (130, -90), (-75, 37), (360, 0)]:
            with self.subTest(yaw=yaw, pitch=pitch):
                pose = head_camera_pose([1, 2, 0.1], 21, yaw, pitch).camera_to_world
                rotation = pose[:3, :3]
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(rotation), 1)
                # A world point along the optical axis projects to the principal point.
                world_point = pose @ [0, 0, 2.5, 1]
                np.testing.assert_allclose(
                    np.linalg.inv(pose) @ world_point, [0, 0, 2.5, 1], atol=1e-12
                )

    def test_rig_fields_are_authoritative_and_legacy_views_unchanged(self):
        config = json.loads((ROOT / "configs/demos/headcam_mirror.json").read_text())
        original = copy.deepcopy(config["cameras"][0])
        self.assertIsNone(camera_rig_pose(original))
        np.testing.assert_array_equal(
            resolve_camera_pose(original), look_at(original["position"], original["target"])
        )
        view = {
            "id": "head_left_up",
            "focal_length_mm": 14,
            "robot_base": {"position": [0, -1, 0], "yaw_deg": 20},
            "head": {"yaw_deg": 35, "pitch_deg": 15},
            # These stale fields must not override the rig used for rendering.
            "position": [100, 100, 100],
            "target": [0, 0, 0],
        }
        config["cameras"] = [view]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            path.write_text(json.dumps(config))
            loaded, _ = load_scene(path)
        resolved = loaded["cameras"][0]
        pose = resolve_camera_pose(view)
        np.testing.assert_allclose(resolved["position"], pose[:3, 3])
        np.testing.assert_allclose(resolved["target"], pose[:3, 3] + pose[:3, 2])

    def test_invalid_rigs_are_rejected(self):
        for base, yaw, pitch in [([0, 0], 0, 0), ([0, np.nan, 0], 0, 0), ([0, 0, 0], np.inf, 0)]:
            with self.subTest(base=base, yaw=yaw, pitch=pitch), self.assertRaises(ValueError):
                head_camera_pose(base, head_yaw_deg=yaw, head_pitch_deg=pitch)
        with self.assertRaisesRegex(ValueError, "both robot_base and head"):
            camera_rig_pose({"head": {"yaw_deg": 15}})
        config = json.loads((ROOT / "configs/demos/headcam_mirror.json").read_text())
        config["headcam"]["enabled"] = False
        config["cameras"] = [
            {
                "id": "disabled",
                "focal_length_mm": 14,
                "robot_base": {"position": [0, 0, 0]},
                "head": {},
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "headcam.enabled"):
                load_scene(path)


if __name__ == "__main__":
    unittest.main()
