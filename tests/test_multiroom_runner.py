"""Causal movement/sensor checks and evaluator-only metric regression tests."""

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from roomgraph.headcam import head_camera_pose
from roomgraph.mapping import Frontier

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
ML_AVAILABLE = all(
    importlib.util.find_spec(name) for name in ["torch", "torchvision", "scipy", "skimage", "cv2"]
)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
        spec.loader.exec_module(module)
    return module


if ML_AVAILABLE:
    RUNNER = load_script("run_multiroom")
    EVALUATOR = load_script("evaluate_multiroom")
    REPLAY = load_script("check_multiroom_replay")


def frontier(goal, gain=20, distance=2):
    return Frontier(tuple(goal), tuple(goal), np.asarray([[0, 0], goal]), gain, distance, 0.0)


@unittest.skipUnless(ML_AVAILABLE, "Install learning dependencies")
class MovementTests(unittest.TestCase):
    def test_prefix_preserves_corners_and_truncates_by_path_length(self):
        path = np.array([[0, 0], [1, 0], [1, 1]], dtype=float)
        result = RUNNER.path_prefix(path, 1.5)
        np.testing.assert_allclose(result, [[0, 0], [1, 0], [1, 0.5]])
        self.assertAlmostEqual(np.linalg.norm(np.diff(result, axis=0), axis=1).sum(), 1.5)
        np.testing.assert_array_equal(path, [[0, 0], [1, 0], [1, 1]])

    def test_zero_budget_and_duplicate_points_do_not_create_motion(self):
        path = [[0, 0], [0, 0], [0, 1]]
        result = RUNNER.path_prefix(path, 0)
        self.assertTrue(np.all(result == 0))
        np.testing.assert_allclose(RUNNER.path_prefix(path, 5), path)
        for budget in (-1, np.nan, np.inf):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                RUNNER.path_prefix(path, budget)
        with self.assertRaises(ValueError):
            RUNNER.path_prefix([[0, np.nan]], 1)

    def test_frontier_selection_discounts_revisits_but_allows_backtracking(self):
        previous = frontier([2, 0], gain=25)
        fresh = frontier([0, 2], gain=20)
        self.assertIs(RUNNER.choose_frontier([previous, fresh], [], []), previous)
        self.assertIs(RUNNER.choose_frontier([previous, fresh], [[2, 0]], []), fresh)
        self.assertIs(RUNNER.choose_frontier([previous], [[2, 0]], []), previous)

    def test_unproductive_and_too_near_frontiers_are_not_reselected(self):
        near = frontier([0.2, 0], gain=100, distance=0.2)
        attempted = frontier([1, 1], gain=100)
        eligible = frontier([2, 0])
        self.assertIs(RUNNER.choose_frontier([near, attempted, eligible], [], [[1, 1]]), eligible)
        self.assertIsNone(RUNNER.choose_frontier([near, attempted], [], [[1, 1]]))


@unittest.skipUnless(ML_AVAILABLE, "Install learning dependencies")
class FileCameraTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.capture = self.root / "capture"
        self.capture.mkdir()
        self.camera = RUNNER.FileCamera(self.capture, self.root / "requests", timeout_s=0.05)
        self.position, self.body, self.head, self.pitch = [-1, 2, 0], 30, 20, -25
        Image.fromarray(np.zeros((8, 12, 3), np.uint8)).save(self.capture / "rgb.png")
        np.savez(self.capture / "depth.npz", depth=np.ones((8, 12)))
        self.response = {
            "status": "complete",
            "view": {
                "id": "F0000",
                "rgb": "rgb.png",
                "geometry": "depth.npz",
                "intrinsics": [[7, 0, 6], [0, 7, 4], [0, 0, 1]],
                "camera_to_world": head_camera_pose(
                    self.position, self.body, self.head, self.pitch
                ).camera_to_world.tolist(),
                "secret_scene_rooms": ["this must not enter policy"],
            },
        }

    def acquire(self, response=None):
        response = self.response if response is None else response

        def adapter(request, path):
            path.write_text(json.dumps(request))
            path.with_name(path.name.replace("request_", "response_")).write_text(
                json.dumps(response)
            )

        with patch.object(RUNNER, "atomic_write_json", side_effect=adapter):
            return self.camera.acquire(self.position, self.body, self.head, self.pitch, 14)

    def test_only_requested_calibration_and_sensor_paths_cross_the_boundary(self):
        observed = self.acquire()
        self.assertEqual(observed["id"], "F0000")
        self.assertNotIn("secret_scene_rooms", observed)
        self.assertEqual(observed["rgb"], (self.capture / "rgb.png").resolve())
        self.assertEqual(observed["request"]["head"], {"yaw_deg": 20, "pitch_deg": -25})

    def test_mismatched_frame_id_and_pose_are_rejected(self):
        response = copy.deepcopy(self.response)
        response["view"]["id"] = "other"
        with self.assertRaisesRegex(ValueError, "different requested frame"):
            self.acquire(response)
        self.camera.count = 1
        response = copy.deepcopy(self.response)
        response["view"]["id"] = "F0001"
        response["view"]["camera_to_world"][0][3] += 1
        with self.assertRaisesRegex(ValueError, "articulated pose"):
            self.acquire(response)

    def test_bad_intrinsics_and_depth_dimensions_are_rejected(self):
        response = copy.deepcopy(self.response)
        response["view"]["intrinsics"][0][0] *= 2
        with self.assertRaisesRegex(ValueError, "intrinsics"):
            self.acquire(response)
        self.camera.count = 1
        response = copy.deepcopy(self.response)
        response["view"]["id"] = "F0001"
        np.savez(self.capture / "depth.npz", depth=np.ones((4, 6)))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.acquire(response)

    def test_external_sensor_paths_and_stale_requests_are_rejected(self):
        response = copy.deepcopy(self.response)
        response["view"]["rgb"] = "../outside.png"
        with self.assertRaisesRegex(ValueError, "inside its capture"):
            self.acquire(response)
        self.camera.count = 0
        with self.assertRaisesRegex(ValueError, "fresh capture"):
            self.acquire()


@unittest.skipUnless(ML_AVAILABLE, "Install learning dependencies")
class FrozenReplayTests(unittest.TestCase):
    def test_replay_is_exact_by_default_and_tolerance_is_explicit(self):
        expected = np.array([1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "values differ"):
            REPLAY.require_equal(expected + 1e-8, expected, "fixture")
        REPLAY.require_equal(expected + 1e-8, expected, "fixture", atol=1e-7)
        with self.assertRaisesRegex(ValueError, "shape differs"):
            REPLAY.require_equal([[1.0, 2.0]], expected, "fixture")

    def test_changed_frozen_input_is_rejected_before_it_can_be_used(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sensor.bin"
            path.write_bytes(b"acquired frame")
            expected = REPLAY.file_hash(path)
            REPLAY.require_hash(path, expected, "sensor")
            path.write_bytes(b"changed frame")
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                REPLAY.require_hash(path, expected, "sensor")
        with self.assertRaisesRegex(ValueError, "completed nonempty"):
            REPLAY.replay({"status": "running"})


@unittest.skipUnless(ML_AVAILABLE, "Install learning dependencies")
class MultiroomEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.edges = [
            {"name": "room/long", "start": [0, 0, 1], "end": [9, 0, 1], "kind": "wall_wall"},
            {"name": "room/short", "start": [0, 2, 1], "end": [1, 2, 1], "kind": "door"},
        ]

    def test_completeness_is_length_weighted_and_type_errors_are_separate(self):
        samples = EVALUATOR.sample_reference(self.edges)
        points = samples["points"][samples["channels"] == 1]
        scores = EVALUATOR.edge_scores(
            points,
            np.full(len(points), 4),
            self.edges,
            samples,
            np.ones(len(samples["points"]), bool),
        )
        self.assertEqual(scores["edge_precision_5cm"], 1)
        self.assertAlmostEqual(scores["edge_completeness_5cm"], 0.9)
        self.assertEqual(scores["typed_edge_precision_5cm"], 0)
        self.assertEqual(scores["typed_edge_completeness_5cm"], 0)

    def test_missing_predictions_are_incomplete_and_never_produce_nonfinite_json(self):
        samples = EVALUATOR.sample_reference(self.edges)
        scores = EVALUATOR.edge_scores(
            np.empty((0, 3)),
            np.zeros(0),
            self.edges,
            samples,
            np.zeros(len(samples["points"]), bool),
        )
        self.assertEqual(scores["edge_completeness_10cm"], 0)
        self.assertEqual(scores["edge_precision_10cm"], 0)
        self.assertIsNone(scores["observed_edge_completeness_10cm"])
        self.assertIsNone(scores["symmetric_edge_chamfer_m"])
        json.dumps(scores, allow_nan=False)

    def test_invalid_depth_never_counts_as_observed_structure(self):
        record = {"intrinsics": [[1, 0, 1], [0, 1, 1], [0, 0, 1]], "camera_to_world": np.eye(4)}
        for value in (np.inf, np.nan, 0, -1, 0.5):
            with self.subTest(value=value):
                self.assertFalse(
                    EVALUATOR.visible_samples(
                        np.array([[0, 0, 1]]), record, np.full((3, 3), value)
                    )[0]
                )
        self.assertTrue(
            EVALUATOR.visible_samples(np.array([[0, 0, 1]]), record, np.ones((3, 3)))[0]
        )

    def test_dense_path_audit_finds_crossed_walls_between_waypoints(self):
        points = EVALUATOR.densify_path([[-1, 0], [1, 0]])
        box = {"lower": [-0.1, -1, 0], "upper": [0.1, 1, 2]}
        result = EVALUATOR._box_clearance(points, [box], 0.25)
        self.assertLess(result["minimum_footprint_clearance_m"], 0)
        self.assertGreater(result["intersecting_samples"], 0)
        clear = EVALUATOR._box_clearance(points + [0, 2], [box], 0.25)
        self.assertEqual(clear["intersecting_samples"], 0)
        sparse = EVALUATOR._box_clearance(np.array([[-1, 0], [1, 0]]), [box], 0.25)
        self.assertGreater(sparse["minimum_footprint_clearance_m"], 0)
        self.assertEqual(sparse["minimum_continuous_footprint_clearance_m"], -0.25)
        self.assertEqual(sparse["intersecting_path_segments"], 1)

    def test_incomplete_experiment_cannot_be_scored(self):
        with self.assertRaisesRegex(ValueError, "completed nonempty"):
            EVALUATOR.evaluate({"status": "running"}, {})

    def test_evaluator_scores_completed_run_and_rejects_changed_acquired_depth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry, snapshot = root / "depth.npz", root / "snapshot.npz"
            np.savez(geometry, depth=np.ones((10, 10)))
            edges = [{"name": "room/edge", "start": [0, 0, 1], "end": [0.2, 0, 1], "kind": "door"}]
            points = EVALUATOR.sample_reference(edges)["points"]
            np.savez(
                snapshot,
                states=np.zeros((3, 3)),
                origin=[-1.5, -1.5],
                resolution_m=1,
                edge_points=points,
                edge_channels=np.full(len(points), 4),
            )
            frame = {
                "id": "F0000",
                "request": {"robot_base": {"position": [0, 0, 0]}},
                "intrinsics": [[10, 0, 5], [0, 10, 5], [0, 0, 1]],
                "camera_to_world": np.eye(4),
                "source_geometry": str(geometry),
                "source_geometry_sha256": EVALUATOR.sha256(geometry),
                "action_seconds": 1,
                "path_length_m": 0,
                "inference_seconds": 0.002,
                "mapping_seconds": 0.001,
            }
            results = {
                "status": "complete",
                "trace": [frame],
                "stations": [
                    {
                        "station": 0,
                        "view_ids": ["F0000"],
                        "snapshot": str(snapshot),
                        "planning_seconds": 0.001,
                    }
                ],
                "path": [[0, 0]],
                "path_length_m": 0,
                "action_seconds": 1,
                "wall_seconds": 2,
                "stop_reason": "station_budget",
                "config": {"sensor_assumptions": ["ideal depth"], "footprint_radius_m": 0.25},
            }
            reference = {
                "edges": edges,
                "rooms": [{"id": "room", "kind": "room", "bounds_xy": [[-1, -1], [1, 1]]}],
                "config": {"walls": []},
                "portals": [],
                "boxes": [],
                "parts": [],
            }
            report = EVALUATOR.evaluate(results, reference)
            self.assertEqual(report["final"]["edge_precision_5cm"], 1)
            self.assertEqual(report["final"]["visible_edge_coverage"], 1)
            self.assertEqual(report["visits"]["region_entry_order"], ["room"])
            self.assertNotIn(str(root), json.dumps(report, allow_nan=False))
            np.savez(geometry, depth=np.full((10, 10), 2))
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                EVALUATOR.evaluate(results, reference)

    def test_portal_crossing_requires_changing_sides_inside_the_aperture(self):
        reference = {
            "rooms": [],
            "config": {"walls": [{"id": "wall", "start": [-2, 0], "end": [2, 0]}]},
            "portals": [
                {"id": "door", "wall": "wall", "offset_m": 2, "width_m": 1, "connects": ["a", "b"]}
            ],
        }
        crossing = EVALUATOR.reference_visits([], [[0, -1], [0, 0], [0, 1]], reference)
        self.assertEqual(crossing["unique_portals_crossed"], 1)
        touch = EVALUATOR.reference_visits([], [[0, -1], [0, 0], [0, -1]], reference)
        self.assertEqual(touch["unique_portals_crossed"], 0)
        wall = EVALUATOR.reference_visits([], [[1, -1], [1, 1]], reference)
        self.assertEqual(wall["unique_portals_crossed"], 0)


if __name__ == "__main__":
    unittest.main()
