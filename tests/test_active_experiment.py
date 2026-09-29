"""CPU checks for replay evaluation and shared baseline acquisition constraints."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from roomgraph.active_view import ActiveHeadPlanner, HeadView
from roomgraph.headcam import head_camera_pose

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_active_head.py"
ML_AVAILABLE = all(
    importlib.util.find_spec(name) for name in ["torch", "torchvision", "scipy", "skimage", "cv2"]
)
if ML_AVAILABLE:
    SPEC = importlib.util.spec_from_file_location("active_head_experiment", RUNNER_PATH)
    RUNNER = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(RUNNER)


def camera(identifier="center", position=(0, 0, 0)):
    pose = np.eye(4)
    pose[:3, 3] = position
    return {
        "id": identifier,
        "intrinsics": [[10, 0, 5], [0, 10, 5], [0, 0, 1]],
        "camera_to_world": pose.tolist(),
    }


@unittest.skipUnless(ML_AVAILABLE, "Install learning dependencies to test the replay runner")
class VisibilityEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.geometry = Path(self.temporary.name) / "geometry.npz"
        self.edges = [{"start": [-0.2, 0, 1], "end": [0.2, 0, 1]}]

    def observe(self, evaluator, depth, view=None):
        np.savez(self.geometry, depth=np.full((10, 10), depth, dtype=float))
        return evaluator.observe(view or camera(), self.geometry)

    def test_finite_depth_distinguishes_direct_visibility_and_occlusion(self):
        occluded = RUNNER.VisibilityEvaluator(self.edges)
        self.assertEqual(self.observe(occluded, 0.5)["visible_edge_coverage"], 0)
        self.assertAlmostEqual(self.observe(occluded, 1.0)["visible_edge_coverage"], 1)
        # Coverage records observations already acquired and never decreases.
        self.assertAlmostEqual(self.observe(occluded, 0.5)["visible_edge_coverage"], 1)

    def test_invalid_depth_never_establishes_visible_edge_coverage(self):
        for depth in [np.inf, -np.inf, np.nan, 0.0, -1.0]:
            with self.subTest(depth=depth):
                scores = self.observe(RUNNER.VisibilityEvaluator(self.edges), depth)
                self.assertEqual(scores["visible_edge_coverage"], 0)
                self.assertEqual(scores["multi_view_edge_coverage"], 0)

    def test_multiple_views_require_translated_centers_and_shared_visibility(self):
        evaluator = RUNNER.VisibilityEvaluator(self.edges)
        self.observe(evaluator, 1.0)
        self.assertEqual(self.observe(evaluator, 1.0)["multi_view_edge_coverage"], 0)
        nearby = camera(position=(0.1, 0, 0))
        self.assertEqual(self.observe(evaluator, 1.0, nearby)["multi_view_edge_coverage"], 0)
        translated = camera(position=(0.3, 0, 0))
        self.assertEqual(self.observe(evaluator, 0.5, translated)["multi_view_edge_coverage"], 0)
        self.assertAlmostEqual(
            self.observe(evaluator, 1.0, translated)["multi_view_edge_coverage"], 1
        )


@unittest.skipUnless(ML_AVAILABLE, "Install learning dependencies to test the replay runner")
class CameraRecordTests(unittest.TestCase):
    def test_resize_scales_intrinsic_rows_without_changing_pose_or_source(self):
        view = camera("original", (0.5, 0.2, 1.5))
        view["intrinsics"] = [[200, 4, 100], [0, 150, 50], [0, 0, 1]]
        original = copy.deepcopy(view)
        result = RUNNER.camera_record(view, (200, 100), (100, 40), "resized")
        np.testing.assert_allclose(result["intrinsics"], [[100, 2, 50], [0, 60, 20], [0, 0, 1]])
        self.assertEqual(result["camera_to_world"], original["camera_to_world"])
        self.assertEqual(result["id"], "resized")
        self.assertEqual(view, original)
        self.assertEqual(RUNNER.camera_record(view, (200, 100), (100, 40))["id"], "original")


@unittest.skipUnless(ML_AVAILABLE, "Install learning dependencies to test the replay runner")
class BaselineSelectionTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            HeadView.from_record(str(yaw), yaw, 0, camera(str(yaw)), (10, 10))
            for yaw in (-40, 0, 40)
        ]
        self.planner = ActiveHeadPlanner(self.candidates)

    def choose(self, name, remaining, acquired=frozenset({"0"}), seed=42):
        return RUNNER.choose_baseline(
            name,
            self.candidates,
            acquired,
            (0, 0),
            self.planner,
            remaining,
            np.random.default_rng(seed),
        )

    def test_both_baselines_exclude_acquired_views_and_respect_motion_budget(self):
        duration = self.planner.action_time((0, 0), self.candidates[0])
        for name in ("raster", "random"):
            with self.subTest(policy=name):
                self.assertIsNone(self.choose(name, duration - 0.001))
                action = self.choose(name, duration)
                self.assertNotEqual(action["view_id"], "0")
                self.assertLessEqual(action["duration_s"], duration)
                self.assertIsNone(self.choose(name, 10, {"-40", "0", "40"}))
        self.assertEqual(self.planner.acquired_ids, set())

    def test_raster_has_fixed_order_and_random_baseline_is_seeded(self):
        self.assertEqual(self.choose("raster", 3)["view_id"], "-40")
        self.assertEqual(self.choose("raster", 3, {"-40", "0"})["view_id"], "40")
        self.assertEqual(self.choose("random", 3, seed=7), self.choose("random", 3, seed=7))


@unittest.skipUnless(ML_AVAILABLE, "Install learning dependencies to test the replay runner")
class CaptureValidationTests(unittest.TestCase):
    def setUp(self):
        base = {"position": [0, 0, 0], "yaw_deg": 0}
        self.config = {
            "robot_base": base,
            "yaw_grid_deg": [0, 20],
            "pitch_grid_deg": [0],
            "focal_length_mm": 12,
            "render": {"width": 20, "height": 10},
        }
        scene = {
            "room": {"dimensions_m": [4, 4, 3]},
            "furnishings": [{"asset": "chair"}],
            "asset_overrides": {},
            "lighting": {"intensity": 10},
            "headcam": {"enabled": True},
            "seed": 42,
            "dataset": {"split": "val", "building_id": "fixture"},
            "render": {"width": 20, "height": 10},
        }
        self.bootstrap = {
            "scene_config": scene,
            "structural_edges": [{"start": [0, 0, 0], "end": [0, 0, 3]}],
        }
        self.capture = copy.deepcopy(self.bootstrap)
        self.capture["views"] = [
            {
                "id": str(yaw),
                "robot_base": copy.deepcopy(base),
                "head": {"yaw_deg": yaw, "pitch_deg": 0},
                "camera_to_world": head_camera_pose([0, 0, 0], 0, yaw, 0).camera_to_world.tolist(),
                "intrinsics": [[10, 0, 10], [0, 10, 5], [0, 0, 1]],
            }
            for yaw in [0, 20]
        ]

    def test_matching_capture_is_accepted_and_stale_geometry_or_clutter_is_rejected(self):
        RUNNER.validate_capture(self.config, self.bootstrap, self.capture)
        geometry = copy.deepcopy(self.capture)
        geometry["structural_edges"][0]["end"][2] = 4
        with self.assertRaisesRegex(ValueError, "same room geometry"):
            RUNNER.validate_capture(self.config, self.bootstrap, geometry)
        clutter = copy.deepcopy(self.capture)
        clutter["scene_config"]["furnishings"] = [{"asset": "sofa"}]
        with self.assertRaisesRegex(ValueError, "furnishings"):
            RUNNER.validate_capture(self.config, self.bootstrap, clutter)

    def test_changed_grid_or_calibration_cannot_reuse_stale_capture(self):
        config = copy.deepcopy(self.config)
        config["yaw_grid_deg"].append(40)
        with self.assertRaisesRegex(ValueError, "orientation grid"):
            RUNNER.validate_capture(config, self.bootstrap, self.capture)
        for key, row, column, message in [
            ("camera_to_world", 0, 3, "rig metadata"),
            ("intrinsics", 0, 0, "requested camera"),
        ]:
            with self.subTest(field=key):
                capture = copy.deepcopy(self.capture)
                capture["views"][0][key][row][column] += 1
                with self.assertRaisesRegex(ValueError, message):
                    RUNNER.validate_capture(self.config, self.bootstrap, capture)

    def test_held_out_test_capture_is_rejected(self):
        self.capture["scene_config"]["dataset"]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "held-out test"):
            RUNNER.validate_capture(self.config, self.bootstrap, self.capture)


if __name__ == "__main__":
    unittest.main()
