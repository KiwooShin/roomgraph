"""Head motion constraints, causal coverage, and non-oracle evidence bookkeeping."""

import unittest

import numpy as np

from roomgraph.active_view import ActiveHeadPlanner, HeadMotionLimits, HeadView
from roomgraph.geometry import look_at

SHAPE = (64, 96)
BOUNDS = [-2, 2, -2, 2, 0, 3]


def record(yaw=0, pitch=0, center=(0, 0, 1.5), identifier=None):
    yaw, pitch = np.radians([yaw, pitch])
    direction = [np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)]
    result = {
        "intrinsics": [[40, 0, 48], [0, 40, 32], [0, 0, 1]],
        "camera_to_world": look_at(center, np.asarray(center) + direction).tolist(),
    }
    if identifier is not None:
        result["id"] = identifier
    return result


def views():
    return [
        HeadView.from_record(str(yaw), yaw, 0, record(yaw), SHAPE) for yaw in (-80, -40, 0, 40, 80)
    ]


class ActiveViewTests(unittest.TestCase):
    def test_motion_accounts_for_acceleration_and_settling(self):
        limits = HeadMotionLimits()
        self.assertAlmostEqual(limits.duration((0, 0), (0, 0)), 0.15)
        self.assertAlmostEqual(limits.duration((0, 0), (15, 0)), 2 * np.sqrt(15 / 120) + 0.15)
        self.assertAlmostEqual(limits.duration((0, 0), (60, 0)), 1.65)
        self.assertAlmostEqual(
            limits.duration((0, 0), (60, 30)),
            max(limits.duration((0, 0), (60, 0)), limits.duration((0, 0), (0, 30))),
        )
        with self.assertRaisesRegex(ValueError, "outside joint limits"):
            limits.duration((0, 0), (90, 0))
        with self.assertRaises(ValueError):
            HeadMotionLimits(settle_s=0)

    def test_looked_directions_are_not_reconstructed_surfaces(self):
        planner = ActiveHeadPlanner(views(), BOUNDS)
        blank = np.zeros((6, *SHAPE))
        planner.observe(blank, record(identifier="0"))
        diagnostics = planner.diagnostics()
        self.assertGreater(diagnostics["looked_direction_fraction"], 0)
        self.assertGreater(diagnostics["shell_looked_unsupported"], 0)
        self.assertGreater(diagnostics["shell_unseen"], 0)
        self.assertEqual(diagnostics["shell_single_center_support"], 0)
        self.assertEqual(diagnostics["shell_multiple_center_support"], 0)
        self.assertIn("looked_unsupported", planner.shell_states)
        choice = planner.choose()
        self.assertNotEqual(choice["view_id"], "0")
        self.assertGreater(choice["new_direction_cells"], 0)

    def test_visible_and_typed_agreement_are_both_required(self):
        planner = ActiveHeadPlanner(views(), BOUNDS)
        amodal = np.ones((6, *SHAPE))
        amodal[0] = 0
        planner.observe(amodal, record())
        self.assertEqual(planner.diagnostics()["shell_single_center_support"], 0)
        visible_only = np.zeros((6, *SHAPE))
        visible_only[0] = 1
        planner.observe(visible_only, record())
        self.assertEqual(planner.diagnostics()["shell_single_center_support"], 0)
        planner.observe(np.ones((6, *SHAPE)), record())
        self.assertGreater(planner.diagnostics()["shell_single_center_support"], 0)

    def test_rotations_and_tiny_head_offsets_do_not_supply_translation_baseline(self):
        planner = ActiveHeadPlanner(views(), BOUNDS)
        predictions = np.ones((6, *SHAPE))
        planner.observe(predictions, record())
        planner.observe(predictions, record(yaw=20))
        planner.observe(predictions, record(yaw=-20, center=(0.05, 0, 1.5)))
        self.assertEqual(planner.diagnostics()["shell_multiple_center_support"], 0)
        planner.observe(predictions, record(center=(0.3, 0, 1.5)))
        self.assertGreater(planner.diagnostics()["shell_multiple_center_support"], 0)

    def test_plan_is_hypothetical_unique_budgeted_and_does_not_mutate_evidence(self):
        planner = ActiveHeadPlanner(views(), BOUNDS)
        planner.observe(np.zeros((6, *SHAPE)), record(identifier="0"))
        before = planner.diagnostics()
        state_before = planner.shell_state()
        plan = planner.plan(horizon_s=8)
        self.assertGreater(len(plan), 1)
        self.assertLessEqual(sum(action["duration_s"] for action in plan), 8)
        self.assertEqual(len(plan), len({action["view_id"] for action in plan}))
        self.assertNotIn("0", [action["view_id"] for action in plan])
        self.assertTrue(all(action["hypothetical"] for action in plan))
        self.assertEqual(before, planner.diagnostics())
        self.assertEqual(state_before, planner.shell_state())
        self.assertEqual(planner.acquired_ids, {"0"})

    def test_exhaustion_and_small_time_budget_return_no_action(self):
        planner = ActiveHeadPlanner(views())
        blank = np.zeros((6, *SHAPE))
        for view in planner.candidates:
            planner.observe(blank, record(view.yaw_deg), view_id=view.view_id)
        self.assertIsNone(planner.choose())
        self.assertEqual(planner.plan(), [])
        self.assertIsNone(ActiveHeadPlanner(views()).choose(remaining_s=0.1))

    def test_candidate_and_acquired_calibration_are_validated(self):
        invalid = record()
        invalid["camera_to_world"][0][0] = 100
        with self.assertRaisesRegex(ValueError, "rigid"):
            HeadView.from_record("bad", 0, 0, invalid, SHAPE)
        with self.assertRaisesRegex(ValueError, "unique"):
            ActiveHeadPlanner([views()[0], views()[0]])
        planner = ActiveHeadPlanner(views())
        with self.assertRaisesRegex(ValueError, "finite"):
            planner.observe(np.full((6, *SHAPE), np.nan), record())
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            planner.choose(remaining_s=-1)
        with self.assertRaisesRegex(ValueError, "maximum"):
            ActiveHeadPlanner(views(), [1, 0, 0, 1, 0, 1])

    def test_equal_area_direction_grid_and_candidate_cache_need_no_images(self):
        planner = ActiveHeadPlanner(iter(views()))
        snapshot = planner.directional_state()
        np.testing.assert_allclose(np.linalg.norm(snapshot["directions_world"], axis=1), 1)
        self.assertFalse(any(snapshot["looked"]))
        self.assertTrue(any(snapshot["reachable"]))
        self.assertIsNotNone(planner.choose())


if __name__ == "__main__":
    unittest.main()
