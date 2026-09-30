"""Versioned policy isolation and tamper rejection without sensors or a GPU."""

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from roomgraph import policy_runtime as runtime
from roomgraph.headcam import head_camera_pose

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
        spec.loader.exec_module(module)
    return module


SUITE = load_script("run_space_suite_v2")
ML_AVAILABLE = all(
    importlib.util.find_spec(name) for name in ["torch", "torchvision", "scipy", "skimage", "cv2"]
)


class FakePolicy:
    """Small stateful stand-in to isolate runtime hooks from policy algorithm tests."""

    def __init__(self):
        self.decisions = []
        self.observation = None

    def observe_grid(self, grid, position):
        self.observation = grid, position

    def choose(self, frontiers, visited, attempted):
        self.decisions.append({"candidate_count": len(frontiers)})
        return frontiers[0] if frontiers else None


class PolicyRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in runtime.VERSIONED_SOURCE_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"frozen {name}\n")
        self.writes = []
        self.module = types.SimpleNamespace(
            scan_targets=lambda grid, position, first: [(first, "unchanged scan")],
            choose_frontier=lambda frontiers, visited, attempted: "baseline",
            atomic_write_json=lambda value, path: self.writes.append((value, path)),
        )
        self.originals = self.module.__dict__.copy()

    def result(self):
        return {
            "status": "complete",
            "source_sha256": {"baseline.py": "baseline hash"},
            "stations": [{"station": 0}],
        }

    def test_scoped_hooks_preserve_scans_and_restore_after_success_or_failure(self):
        for failure in (False, True):
            with self.subTest(failure=failure):
                try:
                    with (
                        patch.object(runtime, "FrontierCommitmentPolicy", FakePolicy),
                        runtime.policy_bindings(self.module) as policy,
                    ):
                        grid, position = object(), [1, 2, 0]
                        self.assertEqual(
                            self.module.scan_targets(grid, position, True),
                            [(True, "unchanged scan")],
                        )
                        self.assertIs(policy.observation[0], grid)
                        self.assertIs(self.module.choose_frontier([grid], [], []), grid)
                        if failure:
                            raise RuntimeError("fixture failure")
                except RuntimeError as error:
                    self.assertEqual(str(error), "fixture failure")
                for name, value in self.originals.items():
                    self.assertIs(getattr(self.module, name), value)

    def test_final_write_has_startup_provenance_and_an_independent_decision_trace(self):
        result = self.result()
        with (
            patch.object(runtime, "FrontierCommitmentPolicy", FakePolicy),
            runtime.policy_bindings(self.module, capture_writes=True, root=self.root) as policy,
        ):
            self.module.choose_frontier([], [], [])
            self.module.atomic_write_json(result, self.root / "experiment.json")
            policy.decisions[0]["candidate_count"] = 100
        stored = self.writes[0][0]
        self.assertEqual(stored["policy_decisions"], [{"candidate_count": 0}])
        self.assertEqual(stored["exploration_policy"], runtime.policy_specification())
        self.assertEqual(stored["source_sha256"]["baseline.py"], "baseline hash")
        runtime.validate_replay_manifest(stored, self.root)
        self.assertNotIn("exploration_policy", result)
        self.assertIs(self.module.atomic_write_json, self.originals["atomic_write_json"])

    def test_source_mutation_prevents_a_completed_write_and_restores_writer(self):
        with patch.object(runtime, "FrontierCommitmentPolicy", FakePolicy):
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                with runtime.policy_bindings(self.module, capture_writes=True, root=self.root):
                    self.module.choose_frontier([], [], [])
                    (self.root / runtime.VERSIONED_SOURCE_FILES[0]).write_text("changed policy")
                    self.module.atomic_write_json(self.result(), self.root / "experiment.json")
        self.assertFalse(self.writes)
        self.assertIs(self.module.atomic_write_json, self.originals["atomic_write_json"])

    def test_manifest_rejects_unknown_policy_missing_sources_and_changed_decisions(self):
        result = {
            **self.result(),
            "source_sha256": runtime.freeze_sources(self.root),
            "exploration_policy": runtime.policy_specification(),
            "policy_decisions": [{"candidate_count": 1}],
        }
        runtime.validate_replay_manifest(result, self.root)
        for field, value in (("id", "another policy"), ("settings", {})):
            wrong = copy.deepcopy(result)
            wrong["exploration_policy"][field] = value
            with (
                patch.object(runtime, "verify_sources") as verify,
                self.assertRaisesRegex(ValueError, "ID/settings"),
            ):
                runtime.validate_replay_manifest(wrong, self.root)
            verify.assert_not_called()
        missing = copy.deepcopy(result)
        missing["source_sha256"].pop(runtime.VERSIONED_SOURCE_FILES[0])
        with self.assertRaisesRegex(ValueError, "hash is missing"):
            runtime.validate_replay_manifest(missing, self.root)
        with self.assertRaisesRegex(ValueError, "commitment decisions differ"):
            runtime.require_decision_trace(FakePolicy(), result)
        result["policy_decisions"] = []
        with self.assertRaisesRegex(ValueError, "one entry per"):
            runtime.validate_replay_manifest(result, self.root)


class VersionedSuiteTests(unittest.TestCase):
    def test_dispatch_only_changes_exact_python_entrypoints(self):
        original = [sys.executable, "scripts/run_multiroom.py", "--config", "same.json"]
        self.assertEqual(
            SUITE.adapt_command(original),
            [sys.executable, "scripts/run_multiroom_v2.py", "--config", "same.json"],
        )
        self.assertEqual(original[1], "scripts/run_multiroom.py")
        for command in (
            [sys.executable, "scripts/evaluate_multiroom.py", "scripts/run_multiroom.py"],
            ["./installation/run_python.sh", "scripts/render_building.py"],
            ["ffmpeg", "-i", "scripts/run_multiroom.py"],
            [sys.executable, "./scripts/run_multiroom.py"],
        ):
            self.assertEqual(SUITE.adapt_command(command), command)

    def test_suite_keeps_budget_cases_checkpoint_and_controller_config_identical(self):
        baseline = json.loads((ROOT / "configs/experiments/space_suite_v1.json").read_text())
        versioned = json.loads((ROOT / SUITE.DEFAULT_SUITE).read_text())
        SUITE.validate_suite(versioned, baseline)
        for key in ("base_experiment", "checkpoint", "shared_budget", "cases"):
            changed = copy.deepcopy(versioned)
            changed[key] = None
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                SUITE.validate_suite(changed, baseline)
        base = json.loads((ROOT / baseline["base_experiment"]).read_text())
        for case in versioned["cases"]:
            actual = SUITE.baseline.make_case_config(base, case, versioned["shared_budget"])
            self.assertNotIn("exploration_policy", actual)
            self.assertNotIn("building", actual)
            self.assertEqual(actual["max_frames"], 300)
            self.assertEqual(actual["max_stations"], 32)

    def test_suite_restores_provenance_and_command_bindings_after_failure(self):
        module = SUITE.baseline
        before = module.run_command, module.build_provenance, module.PIPELINE_FILES
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with SUITE.suite_bindings():
                self.assertTrue(set(runtime.VERSIONED_SOURCE_FILES) <= set(module.PIPELINE_FILES))
                raise RuntimeError("fixture failure")
        after = module.run_command, module.build_provenance, module.PIPELINE_FILES
        for actual, expected in zip(after, before, strict=True):
            self.assertIs(actual, expected)


@unittest.skipUnless(ML_AVAILABLE, "Install learning dependencies")
class RunnerReplayIntegrationTests(unittest.TestCase):
    def test_real_mapping_loop_and_policy_replay_match_with_synthetic_sensor_frames(self):
        """Exercise actual scan/motion/map loops while replacing only expensive I/O."""
        runner = load_script("run_multiroom_v2")
        replay = load_script("check_multiroom_replay_v2")

        class Camera:
            def __init__(self, capture, requests, prefix=""):
                self.directory, self.count = capture, 0
                capture.mkdir(parents=True)

            def acquire(self, position, body_yaw, head_yaw, pitch, focal_length_mm):
                identifier = f"F{self.count:04d}"
                self.count += 1
                rgb, geometry = (
                    self.directory / f"{identifier}.png",
                    self.directory / f"{identifier}.npz",
                )
                Image.fromarray(np.full((24, 32, 3), self.count, np.uint8)).save(rgb)
                np.savez(geometry, depth=np.full((24, 32), 2.5, np.float32))
                focal = 32 * focal_length_mm / 24
                return {
                    "id": identifier,
                    "rgb": rgb,
                    "geometry": geometry,
                    "intrinsics": [[focal, 0, 16], [0, focal, 12], [0, 0, 1]],
                    "camera_to_world": head_camera_pose(
                        position, body_yaw, head_yaw, pitch
                    ).camera_to_world.tolist(),
                    "capture_seconds": 0,
                    "request": {
                        "id": identifier,
                        "robot_base": {"position": position.tolist(), "yaw_deg": body_yaw},
                        "head": {"yaw_deg": head_yaw, "pitch_deg": pitch},
                        "focal_length_mm": focal_length_mm,
                    },
                }

        class Observer:
            def __init__(self, checkpoint, network_size, output, threshold):
                self.output = output
                output.mkdir()

            def acquire(self, rgb, identifier):
                probabilities = np.zeros((6, 24, 32), np.float32)
                target = self.output / f"{identifier}.npz"
                np.savez(target, probabilities=probabilities)
                return {
                    "probabilities": probabilities,
                    "prediction_path": str(target),
                    "prediction_sha256": hashlib.sha256(probabilities.tobytes()).hexdigest(),
                    "source_rgb_sha256": runtime.file_hash(rgb),
                    "inference_seconds": 0,
                }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"fake observer does not load this checkpoint")
            config = json.loads((ROOT / "configs/experiments/multiroom_v1.json").read_text())
            config.update(
                checkpoint=str(checkpoint),
                network_size=[32, 24],
                map_bounds_xy=[-4, 4, -4, 4],
                start_position=[0, 0, 0],
                max_frames=22,
                max_stations=2,
            )
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            args = types.SimpleNamespace(
                config=config_path,
                capture=root / "capture",
                requests=root / "requests",
                output=root / "run",
                request_prefix="",
            )
            with (
                patch.object(runner.baseline, "FileCamera", Camera),
                patch.object(runner.baseline, "RGBObserver", Observer),
                runtime.policy_bindings(runner.baseline, capture_writes=True),
            ):
                runner.baseline.run(args)
            results = json.loads((args.output / "experiment.json").read_text())
            self.assertEqual(len(results["stations"]), 2)
            checked = replay.replay(results)
            self.assertEqual(checked["status"], "pass")
            self.assertEqual(checked["frames"], 22)
            self.assertEqual(checked["exploration_policy"]["decisions_checked"], 2)
            results["policy_decisions"][0]["action"] = "tampered"
            with self.assertRaisesRegex(ValueError, "commitment decisions differ"):
                replay.replay(results)


if __name__ == "__main__":
    unittest.main()
