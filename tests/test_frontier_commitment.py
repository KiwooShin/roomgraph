"""Safety and bounded-state regressions for the acquired-map commitment policy."""

import ast
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

from roomgraph.frontier_commitment import DEFAULT_SETTINGS, FrontierCommitmentPolicy
from roomgraph.mapping import Frontier, OccupancyGrid

SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


def free_grid():
    grid = OccupancyGrid((0, 6, 0, 6), resolution_m=0.1, footprint_radius_m=0.2)
    grid.observed[:] = True
    grid.log_odds[:] = -4
    return grid


def frontier(grid, start, goal=(4.55, 1.55), gain=100, look=None):
    path = grid.path(start, goal)
    if path is None:
        raise AssertionError("Test frontier must be footprint-safe and reachable")
    distance = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    return Frontier(
        tuple(goal),
        tuple(look if look is not None else np.asarray(goal) + [0.4, 0]),
        path,
        gain,
        distance,
        gain / (1 + distance),
    )


def baseline_chooser():
    """Load the frozen pure chooser without importing the learning/runtime stack."""
    path = Path(__file__).resolve().parents[1] / "scripts/run_multiroom.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if getattr(node, "name", "") == "choose_frontier")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["choose_frontier"]


@unittest.skipUnless(SCIPY_AVAILABLE, "Install learning dependencies")
class FrontierCommitmentTests(unittest.TestCase):
    def setUp(self):
        self.grid = free_grid()
        self.start = np.array([1.55, 1.55])
        self.policy = FrontierCommitmentPolicy()
        self.policy.observe_grid(self.grid, self.start)

    def choose(self, position, candidates, visited=(), attempted=()):
        self.policy.observe_grid(self.grid, position)
        return self.policy.choose(candidates, visited, attempted)

    def test_initial_selection_matches_frozen_score_filters_and_tie_order(self):
        baseline = baseline_chooser()
        rng = np.random.default_rng(810)
        for _ in range(30):
            candidates = [
                frontier(self.grid, self.start, goal, int(rng.integers(1, 100)))
                for goal in ((1.75, 1.55), (4.55, 1.55), (1.55, 4.55))
            ]
            visited = [candidates[int(rng.integers(3))].goal_xy] if rng.random() < 0.5 else []
            attempted = [candidates[int(rng.integers(3))].goal_xy] if rng.random() < 0.5 else []
            policy = FrontierCommitmentPolicy()
            policy.observe_grid(self.grid, self.start)
            self.assertIs(
                policy.choose(candidates, visited, attempted),
                baseline(candidates, visited, attempted),
            )
        candidates = [
            frontier(self.grid, self.start, goal) for goal in ((4.55, 1.55), (1.55, 4.55))
        ]
        self.assertIs(self.policy.choose(candidates, [], []), candidates[0])

    def test_retains_exact_goal_despite_new_high_score_and_rebuilds_path(self):
        initial = frontier(self.grid, self.start)
        self.policy.choose([initial], [], [])
        position = [2.35, 1.55]
        shifted = frontier(self.grid, position, (4.65, 1.55), gain=10)
        tempting = frontier(self.grid, position, (1.55, 4.55), gain=10000)
        chosen = self.choose(position, [tempting, shifted])
        self.assertEqual(chosen.goal_xy, initial.goal_xy)
        np.testing.assert_allclose(chosen.path_xy[0], position)
        np.testing.assert_allclose(chosen.path_xy[-1], initial.goal_xy)
        self.assertTrue(
            np.all(self.grid.safe_mask()[tuple(self.grid.world_to_cell(chosen.path_xy).T)])
        )
        self.assertEqual(self.policy.last_decision["action"], "retained")
        self.assertEqual(self.policy.last_decision["baseline_goal_xy"], list(tempting.goal_xy))

    def test_new_obstacle_releases_goal_and_cannot_preserve_stale_path(self):
        initial = frontier(self.grid, self.start)
        self.policy.choose([initial], [], [])
        # The observed grid reference changes after binding, as during a head sweep.
        self.grid.log_odds[:, 30] = 4
        alternate = frontier(self.grid, self.start, (1.55, 4.55))
        chosen = self.policy.choose([alternate], [], [])
        self.assertIs(chosen, alternate)
        self.assertEqual(self.policy.last_decision["release_reason"], "blocked")

    def test_unknown_barrier_is_not_treated_as_free_for_commitment(self):
        self.policy.choose([frontier(self.grid, self.start)], [], [])
        self.grid.observed[:, 30] = False
        alternate = frontier(self.grid, self.start, (1.55, 4.55))
        self.assertIs(self.policy.choose([alternate], [], []), alternate)
        self.assertEqual(self.policy.last_decision["release_reason"], "blocked")

    def test_resolved_frontier_releases_and_empty_candidates_stop(self):
        self.policy.choose([frontier(self.grid, self.start)], [], [])
        alternate = frontier(self.grid, self.start, (1.55, 4.55))
        self.assertIs(self.policy.choose([alternate], [], []), alternate)
        self.assertEqual(self.policy.last_decision["release_reason"], "resolved")
        self.assertIsNone(self.policy.choose([], [], []))
        self.assertEqual(self.policy.last_decision["release_reason"], "resolved")

    def test_both_goal_and_look_anchor_must_match(self):
        self.policy.choose([frontier(self.grid, self.start)], [], [])
        different_boundary = frontier(self.grid, self.start, look=(4.55, 3.55))
        self.policy.choose([different_boundary], [], [])
        self.assertEqual(self.policy.last_decision["release_reason"], "resolved")

    def test_reached_goal_and_attempted_goal_release(self):
        initial = frontier(self.grid, self.start)
        self.policy.choose([initial], [], [])
        position = [4.35, 1.55]
        alternate = frontier(self.grid, position, (1.55, 4.55))
        self.assertIs(self.choose(position, [alternate]), alternate)
        self.assertEqual(self.policy.last_decision["release_reason"], "reached")
        self.assertIsNone(self.policy.choose([alternate], [], [alternate.goal_xy]))
        self.assertEqual(self.policy.last_decision["release_reason"], "attempted")

    def test_fixed_cap_releases_priority_to_baseline(self):
        self.policy = FrontierCommitmentPolicy({"max_commit_steps": 2})
        initial = frontier(self.grid, self.start)
        self.choose(self.start, [initial])
        for index, x in enumerate((2.15, 2.75)):
            position = [x, 1.55]
            matching = frontier(self.grid, position)
            tempting = frontier(self.grid, position, (1.55, 4.55), gain=10000)
            chosen = self.choose(position, [tempting, matching])
            if index == 0:
                self.assertEqual(chosen.goal_xy, initial.goal_xy)
                self.assertEqual(self.policy.last_decision["commit_steps"], 2)
            else:
                self.assertIs(chosen, tempting)
                self.assertEqual(self.policy.last_decision["release_reason"], "step_cap")

    def test_stagnation_releases_after_two_station_checks(self):
        matching = frontier(self.grid, self.start)
        self.policy.choose([matching], [], [])
        tempting = frontier(self.grid, self.start, (1.55, 4.55), gain=10000)
        chosen = self.policy.choose([tempting, matching], [], [])
        self.assertEqual(chosen.goal_xy, matching.goal_xy)
        self.assertEqual(self.policy.last_decision["stagnation_steps"], 1)
        self.assertIs(self.policy.choose([tempting, matching], [], []), tempting)
        self.assertEqual(self.policy.last_decision["release_reason"], "stalled")

    def test_progress_uses_path_distance_around_obstacle(self):
        self.grid.log_odds[:35, 30] = 4
        initial = frontier(self.grid, self.start)
        self.policy.choose([initial], [], [])
        # Walk the safe path until its first upward segment: this can move away
        # in straight-line distance while making real progress around the wall.
        position = initial.path_xy[8]
        self.assertGreater(
            np.linalg.norm(position - initial.goal_xy),
            np.linalg.norm(self.start - initial.goal_xy),
        )
        matching = frontier(self.grid, position)
        self.choose(position, [matching])
        self.assertGreater(self.policy.last_decision["route_progress_m"], 0.7)
        self.assertEqual(self.policy.last_decision["stagnation_steps"], 0)

    def test_deterministic_trace_is_json_serializable_and_input_position_is_copied(self):
        first = frontier(self.grid, self.start)
        self.policy.choose([first], [], [])
        self.start[:] = 0
        self.policy.choose([first], [], [])
        other = FrontierCommitmentPolicy()
        other.observe_grid(self.grid, [1.55, 1.55, 0])
        other.choose([first], [], [])
        other.choose([first], [], [])
        self.assertEqual(
            json.dumps(self.policy.decisions, allow_nan=False), json.dumps(other.decisions)
        )


class CommitmentValidationTests(unittest.TestCase):
    def test_invalid_settings_fail_without_mutating_defaults(self):
        before = DEFAULT_SETTINGS.copy()
        invalid = [
            [],
            {"extra": 1},
            {"goal_tolerance_m": None},
            {"goal_tolerance_m": True},
            {"goal_tolerance_m": float("nan")},
            {"minimum_progress_m": -1},
            {"frontier_match_radius_m": 0.1},
            {"max_commit_steps": 1.5},
            {"max_commit_steps": True},
            {"stagnation_steps": 0},
        ]
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                FrontierCommitmentPolicy(settings)
        self.assertEqual(DEFAULT_SETTINGS, before)

    def test_grid_and_pose_must_be_bound_before_choice(self):
        policy = FrontierCommitmentPolicy()
        with self.assertRaisesRegex(RuntimeError, "observe_grid"):
            policy.choose([], [], [])
        with self.assertRaises(ValueError):
            policy.observe_grid(None, [0, 0])
        for position in ([0], [0, 0, 0, 0], [np.nan, 0]):
            with self.subTest(position=position), self.assertRaises(ValueError):
                policy.observe_grid(free_grid(), position)


if __name__ == "__main__":
    unittest.main()
