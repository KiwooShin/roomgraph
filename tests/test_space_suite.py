"""Suite isolation, frozen-input checks, and cleanup without rendering or a GPU."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_space_suite", ROOT / "scripts/run_space_suite.py"
)
SUITE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUITE)


class SpaceSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.base = json.loads((ROOT / "configs/experiments/multiroom_v1.json").read_text())
        self.case = {
            "id": "sample",
            "building": "secret_building.json",
            "split": "development",
            "start_position": [0, 0, 0],
            "initial_body_yaw_deg": 20,
        }
        self.budget = {"max_frames": 300, "max_stations": 32}

    def test_same_policy_and_budget_without_building_truth(self):
        first = SUITE.make_case_config(self.base, self.case, self.budget)
        another = {**self.case, "id": "another", "start_position": [3, -1, 0]}
        second = SUITE.make_case_config(self.base, another, self.budget)
        self.assertEqual(first["max_frames"], 300)
        self.assertEqual(first["max_stations"], 32)
        self.assertEqual(first["checkpoint"], self.base["checkpoint"])
        for config in (first, second):
            self.assertNotIn("building", config)
            self.assertNotIn("split", config)
            for key in ("name", "start_position", "initial_body_yaw_deg"):
                config.pop(key)
        self.assertEqual(first, second)
        self.assertEqual(self.base["max_frames"], 420)

    def test_geometry_leaks_per_case_tuning_and_invalid_budgets_are_rejected(self):
        for base, case, budget in (
            ({**self.base, "rooms": ["secret"]}, self.case, self.budget),
            (self.base, {**self.case, "threshold": 0.3}, self.budget),
            (self.base, {**self.case, "reference": "secret"}, self.budget),
            (self.base, {**self.case, "split": "test"}, self.budget),
            (self.base, self.case, {**self.budget, "max_frames": True}),
            (self.base, self.case, {**self.budget, "max_stations": 0}),
        ):
            with self.subTest(case=case, budget=budget), self.assertRaises(ValueError):
                SUITE.make_case_config(base, case, budget)

    def test_frozen_input_change_stops_suite(self):
        source = self.directory / "controller.py"
        source.write_text("frozen controller")
        provenance = {"input_sha256": {"controller.py": SUITE.sha256(source)}}
        SUITE.verify_inputs(provenance, self.directory)
        source.write_text("changed controller")
        with self.assertRaisesRegex(ValueError, "Frozen suite input changed"):
            SUITE.verify_inputs(provenance, self.directory)

    def test_fresh_guard_and_matching_resume_preserve_results(self):
        output = self.directory / "suite"
        provenance = {"configuration": {"max_frames": 300}, "input_sha256": {"a": "123"}}
        SUITE.prepare_output(output, provenance)
        sentinel = output / "untouched.txt"
        sentinel.write_text("old result")
        with self.assertRaises(FileExistsError):
            SUITE.prepare_output(output, provenance)
        SUITE.prepare_output(output, provenance, resume=True)
        changed = copy.deepcopy(provenance)
        changed["input_sha256"]["a"] = "456"
        with self.assertRaisesRegex(ValueError, "matching frozen"):
            SUITE.prepare_output(output, changed, resume=True)
        self.assertEqual(sentinel.read_text(), "old result")

    def test_resume_checks_outputs_and_preserves_partial_case(self):
        case_dir = self.directory / "case"
        case_dir.mkdir()
        SUITE.atomic_json(case_dir / "status.json", {"status": "failed"})
        with self.assertRaisesRegex(FileExistsError, "Partial/failed case preserved"):
            SUITE.verify_completed(case_dir)
        result = case_dir / "experiment.json"
        result.write_text('{"status":"complete"}')
        SUITE.atomic_json(case_dir / "status.json", {"status": "complete"})
        SUITE.atomic_json(
            case_dir / "receipt.json",
            {
                "output_sha256": {"experiment.json": SUITE.sha256(result)},
            },
        )
        SUITE.verify_completed(case_dir)
        result.write_text("modified result")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            SUITE.verify_completed(case_dir)

    def test_startup_failure_still_requests_stop_and_closes_log(self):
        renderer = SUITE.Renderer(self.directory, "building.json", 1000, stop_timeout_s=0.01)
        process = MagicMock()
        process.poll.return_value = 9
        process.returncode = 9
        with (
            patch.object(SUITE, "local_path", side_effect=str),
            patch.object(SUITE.subprocess, "Popen", return_value=process),
            self.assertRaisesRegex(RuntimeError, "startup"),
        ):
            renderer.__enter__()
        self.assertTrue((self.directory / "requests/STOP").exists())
        self.assertTrue(renderer.log.closed)
        self.assertEqual(SUITE.read_json(self.directory / "renderer_stop.json")["return_code"], 9)

    def test_hung_renderer_cleanup_targets_only_its_recorded_container(self):
        renderer = SUITE.Renderer(self.directory, "building.json", 1000, stop_timeout_s=0.01)
        renderer.process = MagicMock()
        renderer.process.wait.side_effect = subprocess.TimeoutExpired("renderer", 0.01)
        renderer.process.returncode = -15
        cid = "a" * 64
        renderer.cidfile.write_text(cid)
        inspected = subprocess.CompletedProcess([], 0, "true\n", "")
        stopped = subprocess.CompletedProcess([], 0, cid, "")
        with (
            patch.object(SUITE.subprocess, "run", side_effect=[inspected, stopped]) as command,
            patch.object(SUITE, "terminate_group") as terminate,
        ):
            renderer.close()
        self.assertEqual(command.call_args_list[0].args[0][-1], cid)
        self.assertEqual(command.call_args_list[1].args[0], ["docker", "stop", "--time", "10", cid])
        terminate.assert_called_once_with(renderer.process)
        self.assertTrue(SUITE.read_json(self.directory / "renderer_stop.json")["forced_cleanup"])

    def test_acquisition_failure_closes_renderer_and_never_runs_evaluator(self):
        case_dir = self.directory / "case"
        renderer = MagicMock()
        config = SUITE.make_case_config(self.base, self.case, self.budget)
        with (
            patch.object(SUITE, "Renderer", return_value=renderer),
            patch.object(SUITE, "local_path", side_effect=str),
            patch.object(SUITE, "verify_inputs"),
            patch.object(
                SUITE, "run_command", side_effect=RuntimeError("acquisition failed")
            ) as run,
            self.assertRaisesRegex(RuntimeError, "acquisition failed"),
        ):
            SUITE.run_case(self.case, config, case_dir, {})
        renderer.__exit__.assert_called_once()
        run.assert_called_once()
        status = SUITE.read_json(case_dir / "status.json")
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["stage"], "acquisition")
        self.assertTrue((case_dir / "experiment_config.json").exists())

    def test_independent_case_failure_keeps_other_cases_running(self):
        cases = [self.case, {**self.case, "id": "second"}]
        provenance = {
            "suite": {"cases": cases},
            "case_configs": {case["id"]: {} for case in cases},
        }
        with (
            patch.object(SUITE, "verify_inputs"),
            patch.object(SUITE, "suite_status"),
            patch.object(SUITE, "run_case", side_effect=[RuntimeError("bad capture"), None]) as run,
        ):
            failures = SUITE.execute_cases(provenance, self.directory, {"sample", "second"})
        self.assertEqual(failures, ["sample"])
        self.assertEqual(run.call_count, 2)

    def test_cleanup_or_provenance_uncertainty_halts_pending_cases(self):
        cases = [self.case, {**self.case, "id": "second"}]
        provenance = {
            "suite": {"cases": cases},
            "case_configs": {case["id"]: {} for case in cases},
        }
        for error_type in (SUITE.FrozenInputError, SUITE.RendererCleanupError, KeyboardInterrupt):
            with (
                self.subTest(error_type=error_type),
                patch.object(SUITE, "verify_inputs"),
                patch.object(SUITE, "run_case", side_effect=error_type("stop")) as run,
                self.assertRaises(error_type),
            ):
                SUITE.execute_cases(provenance, self.directory, {"sample", "second"})
            self.assertEqual(run.call_count, 1)

    def test_command_timeout_reaps_actual_child(self):
        children = []
        real_popen = subprocess.Popen

        def remember(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child

        with (
            patch.object(SUITE.subprocess, "Popen", side_effect=remember),
            self.assertRaises(subprocess.TimeoutExpired),
        ):
            SUITE.run_command(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                self.directory / "command.log",
                timeout_s=0.03,
            )
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())

    def test_docker_wrapper_preserves_optional_cid_and_spaced_arguments(self):
        fake_docker = self.directory / "docker"
        fake_docker.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['CAPTURE_ARGV'], 'w') as stream:\n"
            "    json.dump(sys.argv[1:], stream)\n"
        )
        fake_docker.chmod(0o755)
        args_path = self.directory / "argv.json"
        cid = self.directory / "renderer.cid"
        env = {
            **os.environ,
            "PATH": f"{self.directory}:{os.environ['PATH']}",
            "ROOMGRAPH_CONTAINER_CIDFILE": str(cid),
            "CAPTURE_ARGV": str(args_path),
        }
        subprocess.run(
            [str(ROOT / "installation/run_python.sh"), "script.py", "--name", "spaced value"],
            env=env,
            check=True,
            capture_output=True,
        )
        arguments = SUITE.read_json(args_path)
        index = arguments.index("--cidfile")
        self.assertEqual(arguments[index + 1], str(cid))
        self.assertEqual(arguments[-3:], ["script.py", "--name", "spaced value"])


if __name__ == "__main__":
    unittest.main()
