"""Paired policy reports reject confounding inputs and retain adverse outcomes."""

import copy
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "policy_comparison", ROOT / "scripts/compare_space_policies.py"
)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def bundle(candidate=False):
    case = {
        "id": "example",
        "title": "Example",
        "split": "development",
        "status": "complete",
        "evaluated": True,
        "visits": {"rooms_entered": 3 if candidate else 4, "reference_room_count": 4},
        "metrics": {
            "visible_edge_coverage": 0.6 if candidate else 0.8,
            "edge_precision_10cm": 0.7,
            "edge_completeness_10cm": 0.4,
            "typed_edge_precision_10cm": 0.5,
            "typed_edge_completeness_10cm": 0.3,
        },
        "frames": 300,
        "path_length_m": 24,
        "action_seconds": 900,
        "wall_seconds": 300,
        "full_pipeline_wall_seconds": 370,
        "unentered_reference_rooms": ["study"] if candidate else [],
    }
    config = {
        "name": "candidate" if candidate else "baseline",
        "max_frames": 300,
        "start_position": [0, 0, 0],
        "footprint_radius_m": 0.25,
    }
    experiment = {"stations": []}
    if candidate:
        experiment["exploration_policy"] = {"id": "frontier_commitment_v2", "settings": {}}
    return {
        "summary": {"frozen_suite_sha256": "frozen"},
        "summary_sha256": "summary",
        "cases": {"example": case},
        "experiments": {"example": experiment},
        "manifests": {
            "example": {"renderer": "Isaac", "sensor": {"width": 384}, "samples_per_pixel": 16}
        },
        "frozen": {
            "suite": {
                "exploration_policy": {"id": "frontier_commitment_v2", "settings": {}}
                if candidate
                else None,
                "checkpoint": "model",
                "shared_budget": {"max_frames": 300},
                "cases": [{"id": "example", "building": "layout", "split": "development"}],
            },
            "input_sha256": {
                "model": "checkpoint",
                "layout": "layout_hash",
                "src/roomgraph/mapping.py": "frozen_source",
            },
            "case_configs": {"example": config},
        },
    }


class PolicyComparisonTests(unittest.TestCase):
    def test_regressions_and_unentered_rooms_remain(self):
        result = comparison.build_comparison(bundle(), bundle(True))
        case = result["cases"][0]
        self.assertEqual(case["delta"]["rooms_entered"], -1)
        self.assertAlmostEqual(case["delta"]["visible_edge_coverage"], -0.2)
        self.assertEqual(case["candidate"]["unentered_reference_rooms"], ["study"])
        self.assertEqual(result["macro"]["rooms_entered"]["denominator_buildings"], 1)
        self.assertIn("not held-out", result["qualification"])

    def test_confounds_rejected(self):
        mutations = (
            (lambda x: x["frozen"]["input_sha256"].update(layout="changed"), "Layout mismatch"),
            (lambda x: x["frozen"]["input_sha256"].update(model="changed"), "checkpoint"),
            (
                lambda x: x["frozen"]["case_configs"]["example"].update(start_position=[1, 0, 0]),
                "settings or start",
            ),
            (
                lambda x: x["frozen"]["case_configs"]["example"].update(footprint_radius_m=0.1),
                "settings or start",
            ),
            (lambda x: x["frozen"]["suite"]["shared_budget"].update(max_frames=500), "budgets"),
            (lambda x: x["manifests"]["example"].update(samples_per_pixel=8), "samples_per_pixel"),
            (
                lambda x: x["frozen"]["input_sha256"].update({"src/roomgraph/mapping.py": "new"}),
                "Shared implementation",
            ),
        )
        for mutation, message in mutations:
            with self.subTest(message=message):
                candidate = bundle(True)
                mutation(candidate)
                with self.assertRaisesRegex(ValueError, message):
                    comparison.validate_pair(bundle(), candidate)

    def test_policy_must_be_declared(self):
        candidate = bundle(True)
        candidate["experiments"]["example"].pop("exploration_policy")
        with self.assertRaisesRegex(ValueError, "explicitly declare"):
            comparison.validate_pair(bundle(), candidate)

    def test_missing_evaluation_is_not_invented_zero(self):
        candidate = bundle(True)
        candidate["cases"]["example"] = {
            "id": "example",
            "title": "Example",
            "status": "failed",
            "evaluated": False,
        }
        candidate["experiments"]["example"] = None
        result = comparison.build_comparison(bundle(), candidate)
        self.assertIsNone(result["cases"][0]["delta"]["rooms_entered"])
        self.assertIsNone(result["macro"]["rooms_entered"]["candidate"])
        self.assertEqual(result["macro"]["rooms_entered"]["denominator_buildings"], 0)
        self.assertEqual(result["cases"][0]["candidate"]["status"], "failed")

    def test_policy_must_match_frozen_declaration(self):
        candidate = bundle(True)
        candidate["frozen"]["suite"]["exploration_policy"]["settings"] = {"new": 1}
        with self.assertRaisesRegex(ValueError, "frozen suite declaration"):
            comparison.validate_pair(bundle(), candidate)

    def test_failed_replay_retained_as_unavailable_pair(self):
        candidate = bundle(True)
        candidate["cases"]["example"]["_replay_qualified"] = False
        candidate["cases"]["example"]["status"] = "failed"
        result = comparison.build_comparison(bundle(), candidate)
        case = result["cases"][0]
        self.assertTrue(case["candidate"]["source_evaluated"])
        self.assertFalse(case["candidate"]["evaluated"])
        self.assertEqual(case["candidate"]["availability_reason"], "missing_or_failed_replay")
        self.assertIsNone(case["delta"]["frames"])
        self.assertIsNone(case["candidate"]["rooms_entered"])
        self.assertEqual(result["macro"]["rooms_entered"]["denominator_buildings"], 0)

    def test_stale_curves_room_names_counts_and_stop_reason_rejected(self):
        case = bundle()["cases"]["example"]
        names = ["one", "two", "three", "four"]
        case["visits"]["region_entry_order"] = names
        case["visibility_curve"] = [{"visible_edge_coverage": 0.8}]
        case["stop_reason"] = "frame_budget"
        evaluation = {
            "final": case["metrics"],
            "visits": copy.deepcopy(case["visits"]),
            "visibility_curve": copy.deepcopy(case["visibility_curve"]),
            "stop_reason": "frame_budget",
        }
        reference = {"rooms": [{"id": name, "kind": "room"} for name in names]}
        experiment = {**case, "trace": list(range(300))}
        comparison.validate_summary_measurements(case, evaluation, reference, experiment)
        mutations = (
            lambda x: x.update(visibility_curve=[{"visible_edge_coverage": 0.999}]),
            lambda x: x["visits"].update(reference_room_count=99),
            lambda x: x.update(unentered_reference_rooms=["invented"]),
            lambda x: x.update(stop_reason="invented"),
        )
        for mutate in mutations:
            changed = copy.deepcopy(case)
            mutate(changed)
            with self.assertRaises(ValueError):
                comparison.validate_summary_measurements(changed, evaluation, reference, experiment)

    def test_dropped_case_rejected(self):
        candidate = bundle(True)
        candidate["cases"] = {}
        with self.assertRaisesRegex(ValueError, "same declared cases"):
            comparison.validate_pair(bundle(), candidate)

    def test_diagnostics_do_not_count_adjacent_stops_as_revisit(self):
        stations = []
        for index, (xy, gain) in enumerate([([0, 0], 5000), ([0.1, 0], 20), ([0, 0], 0)]):
            stations.append(
                {
                    "position": [*xy, 0],
                    "view_ids": list(range(10)),
                    "new_surface_voxels": gain,
                    "selected_goal": [index + 2, 0],
                }
            )
        result = comparison.diagnostics({"stations": stations})
        self.assertEqual(result["low_surface_gain_frames"], 20)
        self.assertEqual(result["revisited_pose_frames"], 10)
        self.assertEqual(result["exact_revisited_pose_frames"], 10)

    def test_summary_has_no_raw_observations_or_local_paths(self):
        baseline = bundle()
        baseline["cases"]["example"]["local_directory"] = "/private/workspace"
        baseline["experiments"]["example"]["trace"] = [{"source_rgb": "/private/raw.png"}]
        result = comparison.build_comparison(baseline, copy.deepcopy(bundle(True)))
        self.assertNotIn("/private", str(result))


if __name__ == "__main__":
    unittest.main()
