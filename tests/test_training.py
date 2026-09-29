"""Epoch-boundary resume order and atomic checkpoint failure guarantees."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TRAINER_AVAILABLE = all(
    importlib.util.find_spec(name) for name in ["torch", "torchvision", "tensorboard", "cv2"]
)


@unittest.skipUnless(TORCH_AVAILABLE, "Install torch to run training utility tests")
class TrainingUtilityTests(unittest.TestCase):
    def test_epoch_order_survives_resume_with_persistent_workers(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from roomgraph.learning.training import EpochRandomSampler

        data = TensorDataset(torch.arange(37))

        def make_loader(workers):
            sampler = EpochRandomSampler(data, seed=17)
            loader = DataLoader(
                data,
                batch_size=8,
                sampler=sampler,
                num_workers=workers,
                persistent_workers=workers > 0,
                multiprocessing_context="spawn" if workers else None,
                generator=torch.Generator().manual_seed(731),
            )
            return sampler, loader

        def order(loader):
            return torch.cat([batch[0] for batch in loader]).tolist()

        for workers in [0, 2]:
            with self.subTest(workers=workers):
                sampler, loader = make_loader(workers)
                resumed_sampler, resumed_loader = make_loader(workers)
                try:
                    before = torch.get_rng_state().clone()
                    first = order(loader)
                    sampler.set_epoch(1)
                    uninterrupted = order(loader)
                    resumed_sampler.set_epoch(1)
                    resumed = order(resumed_loader)
                    self.assertEqual(uninterrupted, resumed)
                    self.assertNotEqual(first, uninterrupted)
                    self.assertEqual(sorted(uninterrupted), list(range(len(data))))
                    self.assertTrue(torch.equal(before, torch.get_rng_state()))
                finally:
                    # Explicitly close persistent workers before leaving this test.
                    for item in [loader, resumed_loader]:
                        if item._iterator is not None:
                            item._iterator._shutdown_workers()

    def test_sampler_rejects_negative_epoch(self):
        from roomgraph.learning.training import EpochRandomSampler

        sampler = EpochRandomSampler(range(3))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            sampler.set_epoch(-1)

    def test_atomic_checkpoint_preserves_existing_file_after_partial_write(self):
        import torch

        from roomgraph.learning.training import atomic_torch_save

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            atomic_torch_save({"epoch": 3, "weights": torch.arange(4)}, path)
            original = path.read_bytes()

            def failing_save(payload, stream):
                stream.write(b"partial checkpoint")
                raise OSError("Simulated storage failure")

            with (
                patch("roomgraph.learning.training.torch.save", side_effect=failing_save),
                self.assertRaisesRegex(OSError, "storage failure"),
            ):
                atomic_torch_save({"epoch": 4}, path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(torch.load(path, weights_only=True)["epoch"], 3)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            atomic_torch_save({"epoch": 4}, path)
            self.assertEqual(torch.load(path, weights_only=True)["epoch"], 4)

    def test_atomic_json_preserves_existing_file_on_replace_failure(self):
        from roomgraph.learning.training import atomic_write_json

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            atomic_write_json({"epoch": 3}, path)
            original = path.read_bytes()
            with (
                patch("roomgraph.learning.training.os.replace", side_effect=OSError("disk error")),
                self.assertRaisesRegex(OSError, "disk error"),
            ):
                atomic_write_json({"epoch": 4}, path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            atomic_write_json({"epoch": 4}, path)
            self.assertEqual(json.loads(path.read_text()), {"epoch": 4})
            with self.assertRaises(ValueError):
                atomic_write_json({"loss": float("nan")}, path)
            self.assertEqual(json.loads(path.read_text()), {"epoch": 4})

    def test_best_checkpoint_recovers_committed_improvement(self):
        import torch

        from roomgraph.learning.training import atomic_torch_save, recover_best_checkpoint

        state = {
            "epoch": 4,
            "best_epoch": 4,
            "model": {"weights": torch.arange(3)},
            "config": {"seed": 42},
            "provenance": {"dataset_sha256": "dataset-a"},
        }
        for previous in [None, 2, 7]:
            with self.subTest(previous=previous), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "best.pt"
                if previous is not None:
                    atomic_torch_save({"epoch": previous}, path)
                self.assertTrue(recover_best_checkpoint(state, directory))
                repaired = torch.load(path, weights_only=True)
                self.assertEqual(repaired["epoch"], 4)
                self.assertEqual(repaired["config"], state["config"])
                self.assertEqual(repaired["provenance"], state["provenance"])
                self.assertTrue(torch.equal(repaired["model"]["weights"], torch.arange(3)))
                with patch("roomgraph.learning.training.atomic_torch_save") as save:
                    self.assertFalse(recover_best_checkpoint(state, directory))
                    self.assertFalse(recover_best_checkpoint(dict(state, epoch=6), directory))
                    save.assert_not_called()

    def test_best_checkpoint_mismatch_rejected_without_best_model(self):
        import torch

        from roomgraph.learning.training import atomic_torch_save, recover_best_checkpoint

        state = {
            "epoch": 6,
            "best_epoch": 4,
            "model": {"weights": torch.arange(3)},
            "config": {"seed": 42},
            "provenance": {"dataset_sha256": "dataset-a"},
        }
        previous_states = [
            None,
            {"epoch": 2},
            {"epoch": 7},
            {"epoch": 4, "model": {}, "provenance": {"dataset_sha256": "dataset-b"}},
        ]
        for previous in previous_states:
            with self.subTest(previous=previous), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "best.pt"
                if previous is not None:
                    atomic_torch_save(previous, path)
                before = path.read_bytes() if path.exists() else None
                with self.assertRaisesRegex(ValueError, "Cannot recover best.pt"):
                    recover_best_checkpoint(state, directory)
                self.assertEqual(path.read_bytes() if path.exists() else None, before)

    def test_best_checkpoint_dataset_mismatch_repaired_from_best_epoch(self):
        import torch

        from roomgraph.learning.training import atomic_torch_save, recover_best_checkpoint

        state = {
            "epoch": 4,
            "best_epoch": 4,
            "model": {"weights": torch.arange(3)},
            "config": {"seed": 42},
            "provenance": {"dataset_sha256": "dataset-a"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            atomic_torch_save(
                {"epoch": 4, "model": {}, "provenance": {"dataset_sha256": "dataset-b"}}, path
            )
            self.assertTrue(recover_best_checkpoint(state, directory))
            self.assertEqual(
                torch.load(path, weights_only=True)["provenance"]["dataset_sha256"], "dataset-a"
            )


@unittest.skipUnless(TRAINER_AVAILABLE, "Install the learning dependencies to test resume guards")
class ResumeGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = Path(__file__).resolve().parents[1] / "scripts" / "train_edges.py"
        spec = importlib.util.spec_from_file_location("roomgraph_train_edges_tests", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.check_resume = staticmethod(module.check_resume)

    def setUp(self):
        self.config = {
            "dataset": "artifacts/datasets/example",
            "run_dir": "artifacts/runs/example",
            "workers": 2,
            "checkpoint_every": 1,
            "seed": 42,
            "epochs": 60,
            "batch_size": 16,
            "compile_loss": False,
        }
        self.state = {
            "config": self.config.copy(),
            "provenance": {"dataset_sha256": "dataset-a", "diagnostic_overfit": False},
        }

    def test_same_experiment_allows_runtime_changes(self):
        self.check_resume(self.state, self.config, "dataset-a", False)
        self.check_resume(
            self.state,
            dict(self.config, workers=0, checkpoint_every=5),
            "dataset-a",
            False,
        )

    def test_recipe_changes_rejected(self):
        changes = {"seed": 123, "epochs": 80, "batch_size": 32, "compile_loss": True}
        for key, value in changes.items():
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "configuration differs"),
            ):
                self.check_resume(self.state, dict(self.config, **{key: value}), "dataset-a", False)

    def test_changed_dataset_content_rejected(self):
        with self.assertRaisesRegex(ValueError, "dataset identity differs"):
            self.check_resume(self.state, self.config, "dataset-b", False)

    def test_diagnostic_and_held_out_runs_cannot_mix(self):
        with self.assertRaisesRegex(ValueError, "diagnostic and held-out"):
            self.check_resume(self.state, self.config, "dataset-a", True)

    def test_missing_resume_provenance_rejected(self):
        for missing in ["config", "provenance"]:
            state = {key: value for key, value in self.state.items() if key != missing}
            with (
                self.subTest(missing=missing),
                self.assertRaisesRegex(ValueError, "Legacy checkpoint"),
            ):
                self.check_resume(state, self.config, "dataset-a", False)


if __name__ == "__main__":
    unittest.main()
