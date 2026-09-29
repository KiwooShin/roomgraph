"""Check that efficiency reports do not silently compare different training workloads."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "Learning dependencies are optional")
class TrainingComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts" / "compare_training.py"
        spec = importlib.util.spec_from_file_location("compare_training", path)
        cls.comparison = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.comparison)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.dataset = self.directory / "dataset"
        self.dataset.mkdir()
        index = {"splits": {"train": {"images": 17}, "val": {"images": 4, "rooms": 2}}}
        self.index = self.dataset / "index.json"
        self.index.write_text(json.dumps(index))
        self.run = {
            "config": {"dataset": str(self.dataset), "batch_size": 8, "learning_rate": 0.001},
            "dataset_sha256": hashlib.sha256(self.index.read_bytes()).hexdigest(),
            "total_seconds": 4,
        }
        self.history = [
            {"epoch": 0, "seconds": 1, "images_per_second": 17, "lr": 0.001},
            {"epoch": 1, "seconds": 1, "images_per_second": 17, "lr": 0.0001},
        ]
        self.write_run()

    def write_run(self):
        (self.directory / "run.json").write_text(json.dumps(self.run))
        (self.directory / "history.json").write_text(json.dumps(self.history))

    def test_work_counts_include_partial_batches_without_loading_image_caches(self):
        run, _ = self.comparison.describe_run("baseline", self.directory)
        self.assertEqual(run["optimizer_steps"], 6)
        self.assertEqual(run["training_images_seen"], 34)
        self.assertEqual(run["train_loop_images_per_second"], 17)
        self.assertIsNone(run["process_seconds"])
        self.assertIsNone(run["peak_allocated_mb"])
        self.assertFalse((self.dataset / "test.npz").exists())

    def test_dataset_tampering_and_impossible_step_counts_are_rejected(self):
        self.index.write_text(self.index.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "dataset index differs"):
            self.comparison.describe_run("baseline", self.directory)
        self.run["dataset_sha256"] = hashlib.sha256(self.index.read_bytes()).hexdigest()
        self.history[-1]["optimizer_steps"] = 5
        self.write_run()
        with self.assertRaisesRegex(ValueError, "optimizer step count contradicts"):
            self.comparison.describe_run("baseline", self.directory)

    def test_equal_epochs_with_different_batch_or_schedule_are_not_equal_work(self):
        run, _ = self.comparison.describe_run("baseline", self.directory)
        fast = copy.deepcopy(run)
        fast["name"] = "fast"
        fast["config"]["batch_size"] = 16
        fast["optimizer_steps"] = 4
        fast["learning_rate_history"] = [0.001, 0.001]
        result = self.comparison.compare_work([run, fast])
        self.assertFalse(result["matched_training_work"])
        self.assertTrue(any("batch_size" in item for item in result["work_mismatches"]))
        self.assertTrue(any("optimizer_steps" in item for item in result["work_mismatches"]))
        self.assertTrue(any("learning_rate_history" in item for item in result["work_mismatches"]))

    def test_throughput_optimizations_can_preserve_equal_work(self):
        run, _ = self.comparison.describe_run("baseline", self.directory)
        fast = copy.deepcopy(run)
        fast["name"] = "fast"
        fast["config"].update(workers=0, compile_loss=True)
        self.assertTrue(self.comparison.compare_work([run, fast])["matched_training_work"])
        self.run["diagnostic_overfit"] = True
        self.write_run()
        with self.assertRaisesRegex(ValueError, "diagnostic overfit"):
            self.comparison.describe_run("diagnostic", self.directory)

    def test_resume_allows_runtime_config_changes_but_flags_timing(self):
        previous = {"learning_rate": 0.001, "workers": 2, "checkpoint_every": 1}
        resumed = {"learning_rate": 0.001, "workers": 0, "checkpoint_every": 5}
        self.assertEqual(
            self.comparison.semantic_config(previous), self.comparison.semantic_config(resumed)
        )
        resumed["learning_rate"] = 0.01
        self.assertNotEqual(
            self.comparison.semantic_config(previous), self.comparison.semantic_config(resumed)
        )
        run, _ = self.comparison.describe_run("baseline", self.directory)
        resumed_run = copy.deepcopy(run)
        resumed_run.update(name="resumed", resumed=True)
        comparison = self.comparison.compare_work([run, resumed_run])
        self.assertTrue(comparison["matched_training_work"])
        self.assertFalse(comparison["comparable_fresh_run_timing"])
        self.assertIn("latest session", comparison["timing_caveats"][0])

    def test_time_to_quality_uses_first_scheduled_crossing_without_estimation(self):
        history = [
            {"epoch": 0, "val_loss": 0.6, "elapsed_seconds": 2},
            {"epoch": 1, "elapsed_seconds": 4},
            {"epoch": 2, "val_loss": 0.49, "elapsed_seconds": 6},
            {"epoch": 3, "val_loss": 0.4, "elapsed_seconds": 8},
        ]
        result = self.comparison.time_to_quality(history, 0.5)
        self.assertTrue(result["attained"])
        self.assertEqual(result["epoch"], 2)
        self.assertEqual(result["train_session_elapsed_seconds"], 6)
        self.assertFalse(self.comparison.time_to_quality(history, 0.3)["attained"])
        legacy = self.comparison.time_to_quality([{"epoch": 0, "val_loss": 0.4}], 0.5)
        self.assertTrue(legacy["attained"])
        self.assertIsNone(legacy["train_session_elapsed_seconds"])


if __name__ == "__main__":
    unittest.main()
