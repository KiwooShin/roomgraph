"""The rendered replay obeys the same simulated joint limits as action selection."""

import copy
import importlib.util
import unittest
from pathlib import Path

import numpy as np

from roomgraph.active_view import HeadMotionLimits

SPEC = importlib.util.spec_from_file_location(
    "prepare_head_motion", Path(__file__).parents[1] / "scripts" / "prepare_head_motion.py"
)
motion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(motion)


class HeadMotionTests(unittest.TestCase):
    def test_profiles_obey_speed_acceleration_and_end_at_rest(self):
        limits = HeadMotionLimits()
        for start, target in [((0, 0), (15, 0)), ((-80, -30), (80, 30)), ((0, 0), (-40, 5))]:
            duration = limits.duration(start, target)
            times = np.linspace(0, duration, 15001)
            position = motion.interpolate_head(start, target, times, limits)
            velocity = np.diff(position, axis=0) / (times[1] - times[0])
            acceleration = np.diff(velocity, axis=0) / (times[1] - times[0])
            np.testing.assert_allclose(position[0], start)
            np.testing.assert_allclose(position[-1], target)
            self.assertTrue(np.all(np.abs(velocity).max(0) <= np.array(limits.speed_deg_s) + 1e-5))
            self.assertTrue(
                np.all(np.abs(acceleration).max(0) <= np.array(limits.accel_deg_s2) + 1e-4)
            )
            np.testing.assert_allclose(velocity[-1], 0, atol=1e-8)
            settled = position[times >= duration - limits.settle_s]
            np.testing.assert_allclose(settled, np.broadcast_to(target, settled.shape))

    def test_faster_axis_holds_while_other_axis_continues(self):
        limits = HeadMotionLimits()
        intermediate = motion.interpolate_head((0, 0), (80, 5), 1, limits)
        self.assertLess(intermediate[0], 80)
        self.assertAlmostEqual(intermediate[1], 5)
        np.testing.assert_allclose(motion.interpolate_head((0, 0), (80, 5), 9, limits), (80, 5))

    def test_dense_timeline_covers_entire_budget_without_future_observation_updates(self):
        limits = HeadMotionLimits()
        trace = [
            {"t_s": 0, "view_id": "A", "yaw_deg": 0, "pitch_deg": 0},
            {
                "t_s": limits.duration((0, 0), (40, 30)),
                "view_id": "B",
                "yaw_deg": 40,
                "pitch_deg": 30,
            },
        ]
        frames = motion.build_timeline(trace, 3.1, 5, limits)
        self.assertEqual(frames[0]["time_s"], 0)
        self.assertEqual(frames[-1]["time_s"], 3.1)
        self.assertEqual(frames[-1]["phase"], "holding")
        for frame in frames:
            expected = "A" if frame["time_s"] < trace[1]["t_s"] else "B"
            self.assertEqual(frame["last_acquired_view_id"], expected)
            self.assertTrue(frame["visualization_only"])
        invalid = copy.deepcopy(trace)
        invalid[1]["t_s"] = 0.1
        with self.assertRaisesRegex(ValueError, "motion limits"):
            motion.build_timeline(invalid, 3, 5, limits)

    def test_incomplete_run_and_invalid_times_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "completed"):
            motion.prepare({"status": "running"}, {})
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            motion.interpolate_head((0, 0), (1, 1), -0.1, HeadMotionLimits())

    def test_scene_preserves_furnishing_and_marks_extra_frames_visualization_only(self):
        config = {
            "capture_name": "office_04_active_head",
            "action_budget_s": 1,
            "motion_limits": {},
            "render": {"width": 576, "height": 384, "samples_per_pixel": 16},
            "focal_length_mm": 14,
            "robot_base": {"position": [1.2, 0, 0], "yaw_deg": 90},
        }
        results = {
            "status": "complete",
            "config": config,
            "policies": [
                {
                    "name": "active",
                    "trace": [{"t_s": 0, "view_id": "H12", "yaw_deg": 0, "pitch_deg": 0}],
                }
            ],
        }
        manifest = {
            "scene_config": {
                "name": config["capture_name"],
                "furnishings": [{"recipe": "chair", "parameters": {"xy": [1, 2]}}],
                "headcam": {"enabled": True, "arm_reach_m": 0.52},
                "render": config["render"].copy(),
            }
        }
        original = copy.deepcopy(manifest)
        scene = motion.prepare(results, manifest)
        self.assertEqual(scene["name"], "office_04_active_motion")
        self.assertEqual(scene["furnishings"], manifest["scene_config"]["furnishings"])
        self.assertEqual(scene["headcam"], manifest["scene_config"]["headcam"])
        self.assertTrue(scene["head_motion_visualization"]["excluded_from_policy_and_metrics"])
        self.assertEqual(len(scene["cameras"]), 6)
        self.assertEqual(scene["cameras"][0]["robot_base"], config["robot_base"])
        self.assertEqual(manifest, original)


if __name__ == "__main__":
    unittest.main()
