"""Observed-map topology must preserve unknown space and physical disconnections."""

import json
import unittest

import numpy as np

from roomgraph.topology import extract_topology


def three_rooms_and_corridor():
    """Synthetic observed-free mask; no room labels are provided to the extractor."""
    states = np.ones((120, 140), dtype=np.int8)
    states[10:110, 60:80] = 0
    states[10:50, 10:59] = 0
    states[70:110, 10:59] = 0
    states[40:80, 81:130] = 0
    states[27:39, 59:61] = 0
    states[87:99, 59:61] = 0
    states[54:66, 79:81] = 0
    return states


class ObservedTopologyTests(unittest.TestCase):
    def test_three_rooms_and_corridor_have_observed_neck_connectivity(self):
        result = extract_topology(three_rooms_and_corridor(), [0, 0], 0.1)
        self.assertEqual(len(result["regions"]), 4)
        self.assertEqual(len(result["portals"]), 3)
        corridors = [r for r in result["regions"] if r["candidate_type"] == "corridor_candidate"]
        self.assertEqual(len(corridors), 1)
        corridor = corridors[0]["id"]
        self.assertTrue(all(corridor in portal["regions"] for portal in result["portals"]))
        self.assertTrue(all(not portal["verified_doorway"] for portal in result["portals"]))
        self.assertTrue(all(r["reconstruction_complete"] is None for r in result["regions"]))

    def test_unknown_doorway_does_not_create_an_invented_connection(self):
        states = three_rooms_and_corridor()
        states[27:39, 59:61] = -1
        result = extract_topology(states, [0, 0], 0.1)
        labels = np.asarray(result["region_labels"])
        first_room = f"region_{labels[30, 30]:03d}"
        self.assertEqual(len(result["portals"]), 2)
        self.assertTrue(all(first_room not in portal["regions"] for portal in result["portals"]))
        self.assertTrue(np.all(labels[states == -1] == 0))
        room = next(r for r in result["regions"] if r["id"] == first_room)
        self.assertTrue(room["touches_unknown"])

    def test_all_unknown_input_produces_no_regions_or_portals(self):
        result = extract_topology(np.full((20, 30), -1), [-2, 4], 0.2, poses=[[0, 5]])
        self.assertEqual(result["regions"], [])
        self.assertEqual(result["portals"], [])
        self.assertEqual(result["pose_region_ids"], [None])
        self.assertEqual(result["unknown_cell_count"], 600)

    def test_thin_component_is_retained_without_claiming_an_interior(self):
        states = np.full((10, 10), -1)
        states[4:6, 3:7] = 0
        result = extract_topology(states, [10, -3], 0.1)
        self.assertEqual(len(result["regions"]), 1)
        region = result["regions"][0]
        self.assertEqual(region["candidate_type"], "free_space_fragment")
        self.assertTrue(region["touches_unknown"])
        np.testing.assert_allclose(region["centroid_xy_m"], [10.5, -2.5])
        np.testing.assert_allclose(region["bounds_xy_m"], [[10.3, -2.6], [10.7, -2.4]])
        self.assertAlmostEqual(region["observed_free_area_m2"], 0.08)

    def test_pose_visits_and_crossings_follow_observed_free_segments(self):
        states = three_rooms_and_corridor()
        poses = np.array([[30, 30], [70, 30], [70, 60], [110, 60], [70, 60], [70, 90], [30, 90]])
        result = extract_topology(states, [0, 0], 0.1, poses=(poses + 0.5) * 0.1)
        self.assertEqual(len(set(result["pose_region_ids"])), 4)
        self.assertEqual(len(result["observed_region_crossings"]), 4)
        self.assertEqual(sum(r["observation_visits"] for r in result["regions"]), len(poses))

    def test_teleport_across_wall_or_unknown_does_not_count_a_crossing(self):
        states = three_rooms_and_corridor()
        result = extract_topology(states, [0, 0], 0.1, poses=[[3.05, 3.05], [11.05, 6.05]])
        self.assertNotEqual(*result["pose_region_ids"])
        self.assertEqual(result["observed_region_crossings"], [])

    def test_map_edges_and_missing_observations_never_claim_completeness(self):
        result = extract_topology(np.zeros((20, 20)), [0, 0], 0.1)
        self.assertEqual(len(result["regions"]), 1)
        self.assertTrue(result["regions"][0]["touches_unknown"])
        self.assertIsNone(result["regions"][0]["reconstruction_complete"])

    def test_output_is_deterministic_json_and_does_not_mutate_input(self):
        states = three_rooms_and_corridor()
        original = states.copy()
        first = extract_topology(states, [0, 0], 0.1)
        second = extract_topology(states, [0, 0], 0.1)
        self.assertEqual(json.dumps(first, allow_nan=False), json.dumps(second, allow_nan=False))
        np.testing.assert_array_equal(states, original)

    def test_invalid_grid_calibration_or_pose_is_rejected(self):
        for kwargs in [
            {"states": [[2]]},
            {"states": np.empty((0, 2))},
            {"origin": [1, float("nan")]},
            {"resolution": 0},
            {"portal_width_m": -1},
            {"minimum_region_area_m2": float("inf")},
            {"poses": [[1]]},
        ]:
            params = {"states": [[0]], "origin": [0, 0], "resolution": 0.1, **kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                extract_topology(**params)


if __name__ == "__main__":
    unittest.main()
