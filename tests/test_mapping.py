"""Causal ideal-depth navigation and visible structural-edge fusion checks."""

import unittest

import numpy as np

from roomgraph.geometry import look_at
from roomgraph.headcam import head_camera_pose
from roomgraph.mapping import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    OccupancyGrid,
    StructuralEdgeMap,
    depth_points,
    mask_robot_depth,
    robot_self_mask,
)


def calibration(yaw_deg=0, pitch_deg=-20, center=(0, 0, 1.55)):
    yaw, pitch = np.radians([yaw_deg, pitch_deg])
    forward = [np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)]
    return {
        "intrinsics": [[72, 0, 63.5], [0, 72, 47.5], [0, 0, 1]],
        "camera_to_world": look_at(center, np.asarray(center) + forward).tolist(),
    }


def synthetic_depth(record, box=None):
    """Independent analytic depth sensor for a 4x4x3 m room with optional box."""
    intrinsic = np.asarray(record["intrinsics"])
    pose = np.asarray(record["camera_to_world"])
    yy, xx = np.indices((96, 128))
    rays = np.stack((xx, yy, np.ones_like(xx)), axis=-1) @ np.linalg.inv(intrinsic).T
    rays = rays @ pose[:3, :3].T
    origin = pose[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        intersections = np.stack(
            [
                (value - origin[axis]) / rays[..., axis]
                for axis, value in ((0, -2), (0, 2), (1, -2), (1, 2), (2, 0), (2, 3))
            ]
        )
        intersections[intersections <= 0] = np.inf
        depth = np.min(intersections, axis=0)
        if box is not None:
            lower, upper = np.asarray(box[0]), np.asarray(box[1])
            a, b = (lower - origin) / rays, (upper - origin) / rays
            enter = np.max(np.minimum(a, b), axis=-1)
            leave = np.min(np.maximum(a, b), axis=-1)
            hit = (enter > 0) & (leave >= enter)
            depth = np.minimum(depth, np.where(hit, enter, np.inf))
    return depth.astype(np.float32)


def set_free(grid, rows, columns):
    grid.observed[rows, columns] = True
    grid.log_odds[rows, columns] = -4


class DepthMappingTests(unittest.TestCase):
    def test_known_robot_mask_removes_extended_hand_but_preserves_nearby_wall(self):
        request = {
            "robot_base": {"position": [2, -1, 0], "yaw_deg": 45},
            "head": {"yaw_deg": -30, "pitch_deg": -20},
        }
        rig = head_camera_pose([2, -1, 0], 45, -30, -20)
        local = np.array([[0.24, 0.65, 1.1], [0.50, 0.20, 1.1], [0, 0, 1.14]])
        points = local @ rig.base_to_world[:3, :3].T + rig.base_to_world[:3, 3]
        np.testing.assert_array_equal(robot_self_mask(points, request), [True, False, True])
        # Head geometry follows neck articulation independently of the fixed torso.
        head_point = np.array([0, 0, 1.55])
        articulated = head_point @ rig.head_to_base[:3, :3].T + rig.head_to_base[:3, 3]
        world = articulated @ rig.base_to_world[:3, :3].T + rig.base_to_world[:3, 3]
        self.assertTrue(robot_self_mask(world[None], request)[0])

    def test_depth_self_mask_preserves_raw_depth_and_leaves_occlusion_unknown(self):
        request = {
            "robot_base": {"position": [0, 0, 0], "yaw_deg": 0},
            "head": {"yaw_deg": 0, "pitch_deg": -25},
        }
        rig = head_camera_pose([0, 0, 0], 0, 0, -25)
        world = np.array([0.24, 0.65, 1.1])
        camera = (world - rig.camera_to_world[:3, 3]) @ rig.camera_to_world[:3, :3]
        # Place this known hand return at pixel 0,0 through its principal point.
        record = {
            "intrinsics": [
                [1, 0, -camera[0] / camera[2]],
                [0, 1, -camera[1] / camera[2]],
                [0, 0, 1],
            ],
            "camera_to_world": rig.camera_to_world.tolist(),
        }
        depth = np.array([[camera[2]]], dtype=np.float32)
        filtered, mask = mask_robot_depth(depth, record, request)
        self.assertTrue(np.isfinite(depth).all())
        self.assertTrue(mask[0, 0])
        self.assertTrue(np.isnan(filtered[0, 0]))
        grid = OccupancyGrid()
        grid.integrate_depth(filtered, record)
        self.assertTrue(np.all(grid.states() == UNKNOWN))
        record["camera_to_world"] = np.eye(4).tolist()
        with self.assertRaisesRegex(ValueError, "extrinsics"):
            mask_robot_depth(depth, record, request)

    def test_optical_z_depth_backprojection_and_invalid_samples(self):
        record = {
            "intrinsics": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
            "camera_to_world": np.eye(4).tolist(),
        }
        depth = np.array([[2, 2, np.inf], [0, np.nan, -1]])
        points, pixels = depth_points(depth, record)
        np.testing.assert_array_equal(pixels, [[0, 0], [1, 0]])
        np.testing.assert_allclose(points, [[0, 0, 2], [1, 0, 2]])
        with self.assertRaises(ValueError):
            depth_points(depth, record, stride=0)

    def test_scan_endpoint_blocks_and_unseen_is_not_free(self):
        grid = OccupancyGrid((-3, 3, -3, 3), resolution_m=0.1, footprint_radius_m=0.2)
        grid.integrate_scan((0, 0), [[0, 2]], [True])
        states = grid.states()
        self.assertEqual(states[tuple(grid.world_to_cell((0, 1)))], FREE)
        self.assertEqual(states[tuple(grid.world_to_cell((0, 2)))], OCCUPIED)
        self.assertEqual(states[tuple(grid.world_to_cell((0, 2.5)))], UNKNOWN)
        # Multiple identical pixels must not count as independent observations.
        other = OccupancyGrid((-3, 3, -3, 3), resolution_m=0.1, footprint_radius_m=0.2)
        other.integrate_scan((0, 0), [[0, 2]] * 30, [True] * 30)
        np.testing.assert_array_equal(grid.log_odds, other.log_odds)

    def test_obstacle_immediately_blocks_previously_free_cell(self):
        grid = OccupancyGrid((-3, 3, -3, 3))
        for _ in range(10):
            grid.integrate_scan((0, 0), [[0, 2]], [False])
        grid.integrate_scan((0, 0), [[0, 2]], [True])
        self.assertEqual(grid.states()[tuple(grid.world_to_cell((0, 2)))], OCCUPIED)

    def test_invalid_depth_does_not_clear_and_seed_only_marks_stance(self):
        grid = OccupancyGrid((-3, 3, -3, 3))
        grid.seed_robot_pose((0, 0))
        before = grid.states()
        grid.integrate_depth(np.full((96, 128), np.nan), calibration())
        np.testing.assert_array_equal(grid.states(), before)
        self.assertLess(np.count_nonzero(before == FREE) * grid.resolution_m**2, 0.3)
        self.assertEqual(before[tuple(grid.world_to_cell((1, 0)))], UNKNOWN)

    def test_room_sensor_creates_reachable_frontier_without_scene_input(self):
        grid = OccupancyGrid((-15, 15, -15, 15), resolution_m=0.1, footprint_radius_m=0.25)
        grid.seed_robot_pose((0, 0))
        record = calibration()
        grid.integrate_depth(synthetic_depth(record), record, robot_base_xy=(0, 0))
        self.assertGreater(np.count_nonzero(grid.states() == FREE), 100)
        # Initial panoramic head sensing makes the current stance footprint safe.
        for yaw in (90, 180, 270):
            record = calibration(yaw)
            grid.integrate_depth(synthetic_depth(record), record, robot_base_xy=(0, 0))
        path = grid.path((0, 0), (0, 1.2))
        self.assertIsNotNone(path)
        safe = grid.safe_mask()
        self.assertTrue(np.all(safe[tuple(grid.world_to_cell(path).T)]))
        self.assertGreater(grid.snapshot()["free_area_m2"], 8)

    def test_near_body_height_obstacle_prevents_clearing_behind_it(self):
        grid = OccupancyGrid((-3, 3, -3, 3))
        record = calibration(pitch_deg=-25)
        depth = synthetic_depth(record, box=((-0.45, 0.8, 0), (0.45, 1.1, 1.3)))
        grid.integrate_depth(depth, record, stride=1)
        states = grid.states()
        self.assertTrue(np.any(states[grid.world_to_cell((0, 0.8))[0], :] == OCCUPIED))
        self.assertEqual(states[tuple(grid.world_to_cell((0, 1.5)))], UNKNOWN)

    def test_paths_block_unknown_and_insufficient_door_clearance(self):
        grid = OccupancyGrid((0, 4, 0, 4), resolution_m=0.1, footprint_radius_m=0.21)
        set_free(grid, slice(None), slice(None))
        grid.observed[:, 20] = False
        self.assertIsNone(grid.path((1, 2), (3, 2)))
        grid.observed[:, 20] = True
        grid.log_odds[:, 20] = 4
        grid.log_odds[18:22, 20] = -4
        self.assertIsNone(grid.path((1, 2), (3, 2)))
        grid.log_odds[16:24, 20] = -4
        path = grid.path((1, 2), (3, 2))
        self.assertIsNotNone(path)
        self.assertTrue(np.all(grid.safe_mask()[tuple(grid.world_to_cell(path).T)]))
        self.assertTrue(
            np.all(np.sum(np.abs(np.diff(grid.world_to_cell(path), axis=0)), axis=1) == 1)
        )

    def test_frontier_goals_stand_off_unknown_and_are_deterministic(self):
        grid = OccupancyGrid((0, 5, 0, 5), resolution_m=0.1, footprint_radius_m=0.25)
        set_free(grid, slice(10, 40), slice(10, 40))
        frontiers = grid.frontiers((2.5, 2.5))
        self.assertGreater(len(frontiers), 0)
        repeated = grid.frontiers((2.5, 2.5))
        self.assertEqual([item.goal_xy for item in frontiers], [item.goal_xy for item in repeated])
        for frontier in frontiers:
            self.assertGreater(frontier.gain_cells, 0)
            self.assertTrue(np.all(grid.safe_mask()[tuple(grid.world_to_cell(frontier.path_xy).T)]))
            self.assertGreater(
                np.linalg.norm(np.asarray(frontier.goal_xy) - frontier.look_at_xy), 0.1
            )
            length = np.linalg.norm(np.diff(frontier.path_xy, axis=0), axis=1).sum()
            self.assertAlmostEqual(frontier.distance_m, length)

    def test_clearance_routing_crosses_opening_near_its_center(self):
        grid = OccupancyGrid((0, 7, 0, 5), resolution_m=0.1, footprint_radius_m=0.2)
        set_free(grid, slice(None), slice(None))
        grid.log_odds[:, 35] = 4
        grid.log_odds[19:31, 35] = -4
        path = grid.path((1.0, 1.7), (6.0, 1.7))
        self.assertIsNotNone(path)
        cells = grid.world_to_cell(path)
        crossing = cells[cells[:, 1] == 35]
        self.assertEqual(len(crossing), 1)
        self.assertLessEqual(abs(crossing[0, 0] - 24.5), 1.5)
        self.assertTrue(np.all(grid.safe_mask()[tuple(cells.T)]))

    def test_diagonal_visibility_boundary_keeps_a_reachable_frontier(self):
        grid = OccupancyGrid((0, 6, 0, 6), resolution_m=0.1, footprint_radius_m=0.15)
        # A depth frustum's diagonal unknown boundary is a staircase of cells.
        # Its axis-aligned sides abut observed obstacles, not unknown space.
        for row in range(10, 41):
            set_free(grid, row, slice(10, row + 1))
        grid.observed[:, 9] = True
        grid.log_odds[:, 9] = 4
        grid.observed[9, :] = True
        grid.log_odds[9, :] = 4
        grid.observed[41, :] = True
        grid.log_odds[41, :] = 4
        candidates = grid.frontiers((1.45, 3.55), min_cells=3, minimum_distance_m=0.65)
        self.assertEqual(len(candidates), 1)
        self.assertGreater(candidates[0].gain_cells, 20)
        cells = grid.world_to_cell(candidates[0].path_xy)
        self.assertTrue(np.all(np.abs(np.diff(cells, axis=0)).sum(axis=1) == 1))
        self.assertTrue(np.all(grid.safe_mask()[tuple(cells.T)]))

    def test_clearance_includes_blocked_cell_area_not_only_its_center(self):
        grid = OccupancyGrid((0, 3, 0, 3), resolution_m=0.1, footprint_radius_m=0.26)
        set_free(grid, slice(None), slice(None))
        grid.log_odds[15, 18] = 4
        # Obstacle cell center is 0.3 m away but its near face is only 0.25 m.
        self.assertFalse(grid.safe_mask()[15, 15])
        self.assertTrue(grid.safe_mask()[15, 14])


class StructuralEdgeMappingTests(unittest.TestCase):
    def test_requires_visible_and_typed_prediction_with_valid_depth(self):
        record = {"intrinsics": np.eye(3).tolist(), "camera_to_world": np.eye(4).tolist()}
        depth = np.ones((2, 3))
        depth[0, 2] = np.inf
        prediction = np.ones((6, 2, 3))
        prediction[0, 1] = 0
        prediction[1:, 0, 1] = 0
        edges = StructuralEdgeMap()
        edges.integrate(prediction, depth, record, "one")
        arrays = edges.arrays()
        self.assertEqual(len(arrays["points"]), 5)
        np.testing.assert_allclose(arrays["points"], [[0, 0, 1]] * 5)
        np.testing.assert_array_equal(arrays["channels"], [1, 2, 3, 4, 5])
        with self.assertRaisesRegex(ValueError, "twice"):
            edges.integrate(prediction, depth, record, "one")

    def test_support_counts_views_and_translated_centers_separately(self):
        record = {"intrinsics": np.eye(3).tolist(), "camera_to_world": np.eye(4).tolist()}
        prediction = np.ones((2, 1, 1))
        edges = StructuralEdgeMap(voxel_size_m=1)
        edges.integrate(prediction, np.ones((1, 1)), record, "one")
        edges.integrate(prediction, np.ones((1, 1)), record, "same_center")
        self.assertEqual(edges.arrays()["support"].tolist(), [2])
        self.assertEqual(edges.arrays()["camera_support"].tolist(), [1])
        moved = np.eye(4)
        moved[0, 3] = 0.3
        record["camera_to_world"] = moved.tolist()
        edges.integrate(prediction, np.ones((1, 1)), record, "translated")
        self.assertEqual(edges.arrays()["support"].tolist(), [3])
        self.assertEqual(edges.arrays()["camera_support"].tolist(), [2])

    def test_empty_result_shapes_and_bad_prediction_rejection(self):
        edges = StructuralEdgeMap()
        self.assertEqual(edges.arrays()["points"].shape, (0, 3))
        record = {"intrinsics": np.eye(3).tolist(), "camera_to_world": np.eye(4).tolist()}
        with self.assertRaisesRegex(ValueError, "probabilities"):
            edges.integrate(np.full((2, 2, 2), np.nan), np.ones((2, 2)), record, "invalid")
        zero_threshold = StructuralEdgeMap(threshold=0)
        zero_threshold.integrate(np.zeros((2, 2, 2)), np.ones((2, 2)), record, "zero")
        self.assertEqual(len(zero_threshold.arrays()["points"]), 0)


if __name__ == "__main__":
    unittest.main()
