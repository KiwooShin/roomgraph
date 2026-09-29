"""Metric denominator and frozen-source checks for the development suite report."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "suite_report", ROOT / "scripts/make_space_suite_report.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def measured(name, length, voxels, value):
    return {
        "id": name,
        "evaluated": True,
        "reference_edge_length_m": length,
        "predicted_edge_voxels": voxels,
        "metrics": dict.fromkeys(report.METRICS, value),
        "all_reference_rooms_entered": False,
        "visits": {"rooms_entered": 1, "reference_room_count": 3},
        "frames": 10,
        "path_length_m": 2,
        "wall_seconds": 3,
    }


class SpaceSuiteReportTests(unittest.TestCase):
    def test_length_and_prediction_denominators_are_distinct(self):
        summary = report.aggregate([measured("small", 10, 90, 0.2), measured("large", 90, 10, 0.8)])
        self.assertAlmostEqual(summary["macro"]["visible_edge_coverage"]["value"], 0.5)
        self.assertAlmostEqual(summary["weighted"]["visible_edge_coverage"]["value"], 0.74)
        self.assertAlmostEqual(summary["weighted"]["edge_precision_10cm"]["value"], 0.26)
        self.assertEqual(summary["weighted"]["edge_completeness_10cm"]["denominator_value"], 100)

    def test_failure_is_retained_without_inventing_measurements(self):
        summary = report.aggregate(
            [measured("partial_coverage", 10, 5, 0.2), {"id": "failed", "evaluated": False}]
        )
        self.assertEqual(summary["declared_buildings"], 2)
        self.assertEqual(summary["evaluated_buildings"], 1)
        self.assertEqual(summary["unevaluated_case_ids"], ["failed"])
        self.assertEqual(summary["all_reference_rooms_entered_buildings"], 0)
        self.assertEqual(summary["macro"]["visible_edge_coverage"]["denominator_buildings"], 1)

    def test_no_predictions_has_no_pooled_precision_denominator(self):
        summary = report.aggregate([measured("empty", 10, 0, 0)])
        self.assertEqual(summary["weighted"]["edge_completeness_10cm"]["value"], 0)
        self.assertIsNone(summary["weighted"]["edge_precision_10cm"]["value"])
        self.assertEqual(summary["macro"]["edge_precision_10cm"]["value"], 0)

    def test_missing_case_loads_as_explicit_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = report.load_case({"id": "missing", "title": "Missing case"}, Path(temporary))
        self.assertEqual(case["status"], "not_run")
        self.assertFalse(case["evaluated"])
        self.assertNotIn("metrics", case)

    def test_display_declaration_must_match_frozen_suite_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite = root / "config.json"
            suite.write_text(json.dumps({"cases": []}))
            (root / "suite.json").write_text(
                json.dumps(
                    {
                        "suite": {"cases": []},
                        "input_sha256": {"config.json": report.sha256(suite)},
                    }
                )
            )
            with patch.object(report, "ROOT", root):
                declaration, _ = report.frozen_suite(suite, root)
                self.assertEqual(declaration, {"cases": []})
                suite.write_text('{"cases": []}\n')
                with self.assertRaisesRegex(ValueError, "bytes differ"):
                    report.frozen_suite(suite, root)

    def test_completed_case_rejects_changed_artifact_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "case"
            directory.mkdir()
            (directory / "status.json").write_text('{"status": "complete"}')
            artifact = directory / "artifact.json"
            artifact.write_text("original")
            (directory / "receipt.json").write_text(
                json.dumps({"output_sha256": {"artifact.json": report.sha256(artifact)}})
            )
            artifact.write_text("changed")
            with self.assertRaisesRegex(ValueError, "differs from its frozen receipt"):
                report.load_case({"id": "case", "title": "Changed case"}, root)


if __name__ == "__main__":
    unittest.main()
