"""Connected architectural geometry, apertures and evaluator-only references."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from roomgraph.building import architectural_parts, load_building
from roomgraph.geometry import ray_depth

CONFIG = Path(__file__).resolve().parents[1] / "configs/buildings/connected_three_rooms.json"


class BuildingTests(unittest.TestCase):
    def setUp(self):
        self.config, self.boxes, self.furniture, self.edges = load_building(CONFIG)

    def altered(self, mutate):
        data = copy.deepcopy(self.config)
        mutate(data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            path.write_text(json.dumps(data))
            return load_building(path)

    def test_three_furnished_rooms_connect_through_a_corridor(self):
        rooms = [r for r in self.config["regions"] if r["kind"] == "room"]
        self.assertEqual(len(rooms), 3)
        for room in rooms:
            self.assertGreater(len(room["furnishings"]), 5)
            self.assertTrue(
                any(p["connects"] == [room["id"], "corridor"] for p in self.config["portals"])
            )
        self.assertEqual(len({p.name for p in self.furniture}), len(self.furniture))

    def test_doorway_is_empty_and_lintel_still_blocks_rays(self):
        for x in (-4, 0, 4):
            # Start inside a room and look through its open doorway to the
            # corridor's south wall, rather than a fake 2D door texture.
            distance = ray_depth([x, 2, 1.5], [0, -1, 0], self.boxes)
            self.assertAlmostEqual(float(distance), 2.92)
            lintel = ray_depth([x, 2, 2.6], [0, -1, 0], self.boxes)
            self.assertAlmostEqual(float(lintel), 0.92)
            jamb = ray_depth([x + 0.9, 2, 1.5], [0, -1, 0], self.boxes)
            self.assertAlmostEqual(float(jamb), 0.92)

    def test_room_partition_blocks_shortcut_and_corridor_is_connected(self):
        self.assertAlmostEqual(float(ray_depth([-4, 3, 1.5], [1, 0, 0], self.boxes)), 1.92)
        self.assertAlmostEqual(float(ray_depth([-4, 0, 1.5], [1, 0, 0], self.boxes)), 9.92)

    def test_reference_edges_tag_regions_and_do_not_close_thresholds(self):
        doors = [edge for edge in self.edges if edge.kind == "door"]
        self.assertEqual(len(doors), 18)  # Three jamb/header sets on both wall faces.
        regions = {r["id"] for r in self.config["regions"]}
        self.assertEqual({e.name.split("/")[0] for e in self.edges}, regions)
        for edge in self.edges:
            if edge.kind != "wall_floor":
                continue
            if abs(edge.start[1] - 1) > 0.09 or edge.start[1] != edge.end[1]:
                continue
            low, high = sorted([edge.start[0], edge.end[0]])
            self.assertFalse(any(low < x < high for x in (-4, 0, 4)))

    def test_renderer_geometry_matches_reference_solids(self):
        for box, part in zip(self.boxes, architectural_parts(self.boxes), strict=True):
            np.testing.assert_allclose(
                np.asarray(part.center) - np.asarray(part.size) / 2, box.lower
            )
            np.testing.assert_allclose(
                np.asarray(part.center) + np.asarray(part.size) / 2, box.upper
            )

    def test_overlapping_or_out_of_wall_portals_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.altered(lambda c: c["portals"][1].update(offset_m=2.1))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.altered(lambda c: c["portals"][0].update(offset_m=0.1))

    def test_malformed_geometry_and_region_references_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "axis-aligned"):
            self.altered(lambda c: c["walls"][0].update(end=[-5, 6]))
        with self.assertRaisesRegex(ValueError, "known regions"):
            self.altered(lambda c: c["portals"][0].update(connects=["office", "outside"]))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.altered(lambda c: c.update(height_m=float("nan")))


if __name__ == "__main__":
    unittest.main()
