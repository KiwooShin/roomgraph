"""Connected-space scene validity and frozen development-suite protocol."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from roomgraph.building import load_building
from roomgraph.geometry import ray_depth
from roomgraph.space_cases import validate_space_case

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs/experiments/space_suite_v1.json"


class SpaceCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = json.loads(SUITE.read_text())
        cls.paths = {case["id"]: ROOT / case["building"] for case in cls.suite["cases"]}
        cls.loaded = {name: load_building(path) for name, path in cls.paths.items()}

    def altered(self, mutate, name="residential_branch"):
        config = copy.deepcopy(self.loaded[name][0])
        mutate(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            path.write_text(json.dumps(config))
            return validate_space_case(path)

    def test_all_cases_share_frozen_budget_model_and_development_split(self):
        self.assertEqual(self.suite["shared_budget"], {"max_frames": 300, "max_stations": 32})
        base = json.loads((ROOT / self.suite["base_experiment"]).read_text())
        self.assertEqual(self.suite["checkpoint"], base["checkpoint"])
        self.assertEqual(len(self.paths), 4)
        for case in self.suite["cases"]:
            with self.subTest(case=case["id"]):
                config = self.loaded[case["id"]][0]
                self.assertEqual(case["split"], "development")
                self.assertEqual(config["dataset"]["split"], "development")
                self.assertEqual(case["start_position"], config["start"]["position"])
                self.assertEqual(case["initial_body_yaw_deg"], config["start"]["yaw_deg"])
                self.assertEqual(
                    config["render"],
                    {
                        "width": 384,
                        "height": 256,
                        "samples_per_pixel": 16,
                        "mode": "PathTracing",
                        "focal_length_mm": base["focal_length_mm"],
                    },
                )
                self.assertEqual(config["headcam"]["arm_reach_m"], base["robot_arm_reach_m"])
                bounds = np.asarray(config["bounds_xy"])
                self.assertTrue((bounds[0] > -15).all() and (bounds[1] < 15).all())

    def test_distinct_layouts_are_connected_and_physically_traversable(self):
        audit = {name: validate_space_case(path) for name, path in self.paths.items()}
        self.assertEqual({a["floor_area_m2"] for a in audit.values()}, {56, 72, 80, 90})
        self.assertEqual(len({a["building_sha256"] for a in audit.values()}), 4)
        self.assertEqual(len({self.loaded[name][0]["seed"] for name in self.paths}), 4)
        for item in audit.values():
            self.assertGreaterEqual(item["rooms"], 3)
            self.assertGreaterEqual(item["furniture_objects"], 25)
            self.assertGreater(item["primitive_clearance_reachable_area_m2"], 10)
            self.assertGreaterEqual(item["minimum_portal_width_m"], 1.4)
        self.assertEqual(audit["residential_branch"]["max_region_degree"], 4)
        self.assertEqual(audit["loop_workspaces"]["graph_cycle_rank"], 1)
        self.assertEqual(audit["compact_apartment"]["minimum_portal_width_m"], 1.4)
        main = next(r for r in self.loaded["open_office"][0]["regions"] if r["id"] == "open_office")
        bounds = np.asarray(main["bounds_xy"])
        self.assertEqual(float(np.prod(bounds[1] - bounds[0])), 54)

    def test_door_openings_are_physical_and_headers_remain_solid(self):
        for name, (config, boxes, _, _) in self.loaded.items():
            walls = {wall["id"]: wall for wall in config["walls"]}
            for portal in config["portals"]:
                with self.subTest(case=name, portal=portal["id"]):
                    wall = walls[portal["wall"]]
                    start, end = np.asarray(wall["start"]), np.asarray(wall["end"])
                    axis = int(np.argmax(end - start))
                    cross = 1 - axis
                    origin = np.r_[start, 1.5].astype(float)
                    origin[axis] += portal["offset_m"]
                    origin[cross] -= 0.4
                    direction = np.zeros(3)
                    direction[cross] = 1
                    self.assertGreater(float(ray_depth(origin, direction, boxes)), 0.75)
                    origin[2] = 2.7
                    self.assertAlmostEqual(float(ray_depth(origin, direction, boxes)), 0.32)

    def test_reference_floor_edges_do_not_span_open_doorways(self):
        for name, (config, _, _, edges) in self.loaded.items():
            walls = {wall["id"]: wall for wall in config["walls"]}
            for portal in config["portals"]:
                wall = walls[portal["wall"]]
                start, end = np.asarray(wall["start"]), np.asarray(wall["end"])
                axis = int(np.argmax(end - start))
                cross = 1 - axis
                center = start[axis] + portal["offset_m"]
                for edge in edges:
                    if (
                        edge.kind != "wall_floor"
                        or edge.name.split("/")[0] not in portal["connects"]
                    ):
                        continue
                    if not np.isclose(edge.start[cross], edge.end[cross]):
                        continue
                    if not np.isclose(abs(edge.start[cross] - start[cross]), 0.08):
                        continue
                    low, high = sorted([edge.start[axis], edge.end[axis]])
                    self.assertFalse(low < center < high, (name, portal["id"], edge.name))

    def test_missing_partition_wall_cannot_invent_region_reference_edges(self):
        def mutate(config):
            config["walls"] = [w for w in config["walls"] if w["id"] != "left_rooms"]

        with self.assertRaisesRegex(ValueError, "physical wall"):
            self.altered(mutate)

    def test_portal_must_connect_its_actual_adjacent_regions(self):
        with self.assertRaisesRegex(ValueError, "shared boundary"):
            self.altered(lambda c: c["portals"][0].update(connects=["living", "study"]))

    def test_disconnected_room_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "connect every region"):
            self.altered(lambda c: c["portals"].pop())

    def test_overlapping_or_uncovered_region_partition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "tile"):
            self.altered(lambda c: c["regions"][0]["bounds_xy"][0].__setitem__(1, 0.1))

    def test_start_inside_rotated_furniture_is_rejected(self):
        # The study bookshelf is rotated -90 degrees and physically blocks this stance.
        with self.assertRaisesRegex(ValueError, "Start footprint intersects actual primitive"):
            self.altered(lambda c: c["start"].update(position=[4.62, -2.5, 0]))

    def test_furniture_cannot_seal_a_physically_open_portal(self):
        def mutate(config):
            corridor = next(r for r in config["regions"] if r["id"] == "corridor")
            corridor["furnishings"].append(
                {
                    "recipe": "cabinet",
                    "parameters": {
                        "name": "blocked_passage",
                        "xy": [0, 2],
                        "size": [1.75, 0.9, 1.2],
                    },
                }
            )

        with self.assertRaisesRegex(ValueError, "Portal approaches are unreachable"):
            self.altered(mutate)

    def test_clearance_validation_rejects_invalid_footprint_radius(self):
        for radius in (0, -0.1, float("nan")):
            with self.subTest(radius=radius), self.assertRaisesRegex(ValueError, "radius"):
                validate_space_case(self.paths["loop_workspaces"], radius)


if __name__ == "__main__":
    unittest.main()
