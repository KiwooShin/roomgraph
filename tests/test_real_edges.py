"""Protect evaluation coordinates from resize and padding mistakes."""

import unittest

import numpy as np

from roomgraph.real_edges import prepare_image, rasterize_annotation, restore_probabilities


class RealEdgesTests(unittest.TestCase):
    def test_padding_is_removed_before_evaluation(self):
        image = np.zeros((256, 192, 3), np.uint8)
        padded, rect = prepare_image(image, "letterbox")
        self.assertEqual(padded.shape, (256, 384, 3))
        self.assertEqual(rect, (96, 0, 192, 256))
        prediction = np.ones((6, 256, 384), np.float32)
        prediction[:, :, 96:288] = 0
        restored = restore_probabilities(prediction, rect, (192, 256))
        self.assertFalse(restored.any())

    def test_coordinate_round_trip(self):
        image = np.zeros((256, 192, 3), np.uint8)
        for mode in ("stretch", "portrait", "portrait_large", "letterbox"):
            network, rect = prepare_image(image, mode)
            prediction = np.zeros((1, *network.shape[:2]), np.float32)
            x, y, w, h = rect
            prediction[0, y : y + h, x + w // 2] = 1
            restored = restore_probabilities(prediction, rect, (192, 256))
            self.assertLessEqual(abs(int(restored[0, 128].argmax()) - 96), 1)

    def test_unreviewed_pixels_are_ignored(self):
        annotation = {
            "review_regions": [[2, 2, 8, 8]],
            "polylines": [{"points": [[5, 0], [5, 9]]}],
        }
        target, mask = rasterize_annotation(annotation, (10, 10))
        self.assertEqual(mask.sum(), 36)
        self.assertEqual(target.sum(), 6)
        annotation["review_regions"] = [[-1, 0, 8, 8]]
        with self.assertRaises(ValueError):
            rasterize_annotation(annotation, (10, 10))


class AnnotationGateTests(unittest.TestCase):
    def test_unreviewed_annotations_cannot_produce_accuracy(self):
        import importlib.util
        import json
        import tempfile
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "evaluation", "scripts/evaluate_real_edges.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text(json.dumps({"records": [{"id": "a"}]}))
            with self.assertRaisesRegex(ValueError, "human review"):
                module.score(root, {"frames": [{"id": "a", "status": "unreviewed"}]})

    def test_perfect_reviewed_prediction_scores_one_and_detects_tampering(self):
        import hashlib
        import importlib.util
        import json
        import tempfile
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "evaluation", "scripts/evaluate_real_edges.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "portrait").mkdir()
            path = root / "portrait/a.npz"
            probabilities = np.zeros((6, 256, 192), np.float32)
            probabilities[0, 20:81, 50] = 1
            np.savez(path, probabilities=probabilities)
            summary = {
                "threshold": 0.7,
                "methods": {"portrait": {}},
                "records": [
                    {
                        "id": "a",
                        "vga_exact_timestamp": True,
                        "portrait": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                    }
                ],
            }
            (root / "summary.json").write_text(json.dumps(summary))
            labels = {
                "split": "development_review",
                "frames": [
                    {
                        "id": "a",
                        "size": [192, 256],
                        "status": "human_reviewed",
                        "reviewer": "test",
                        "review_regions": [[0, 0, 192, 256]],
                        "polylines": [{"type": "wall_wall", "points": [[50, 20], [50, 80]]}],
                    }
                ],
            }
            self.assertEqual(module.score(root, labels)["methods"]["portrait"]["f1"], 1)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "cache changed"):
                module.score(root, labels)
