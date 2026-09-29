"""Renderer-side connected building geometry, never a navigation policy input.

Rooms and portals are explicit editable scene descriptions. The controller must
receive only acquired sensor frames; this module's boxes, regions and edges are
reference geometry for rendering and evaluation.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from roomgraph.furnishings import Furnisher, Part
from roomgraph.geometry import Box, Edge
from roomgraph.scene_config import RECIPES


def _finite(values, shape, label):
    values = np.asarray(values, dtype=float)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"{label} must have shape {shape} and finite values")
    return values


def load_building(path):
    """Compile a metre/Z-up building into config, solids, furniture and edges."""
    config = json.loads(Path(path).read_text())
    if config.get("schema_version") != 1 or config.get("kind") != "connected_building":
        raise ValueError("Expected connected_building schema_version 1")
    if config.get("units") != "metres" or config.get("up_axis") != "Z":
        raise ValueError("Buildings require metres and Z up")
    bounds = _finite(config["bounds_xy"], (2, 2), "bounds_xy")
    height, thickness = config["height_m"], config["wall_thickness_m"]
    if not np.isfinite([height, thickness]).all() or height <= 1 or thickness <= 0:
        raise ValueError("Height and wall thickness must be finite and positive")
    if np.any(bounds[1] <= bounds[0]):
        raise ValueError("Building bounds must be increasing")
    for key in ("width", "height", "samples_per_pixel"):
        if not isinstance(config["render"][key], int) or config["render"][key] <= 0:
            raise ValueError(f"render.{key} must be a positive integer")
    regions = {region["id"]: region for region in config["regions"]}
    if len(regions) != len(config["regions"]):
        raise ValueError("Region IDs must be unique")
    for region in regions.values():
        region_bounds = _finite(region["bounds_xy"], (2, 2), "region bounds")
        if np.any(region_bounds[1] <= region_bounds[0]):
            raise ValueError("Region bounds must be increasing")
        if np.any(region_bounds[0] < bounds[0]) or np.any(region_bounds[1] > bounds[1]):
            raise ValueError("Region lies outside building bounds")
    wall_ids = [wall["id"] for wall in config["walls"]]
    if len(wall_ids) != len(set(wall_ids)):
        raise ValueError("Wall IDs must be unique")
    portals = config["portals"]
    if len({p["id"] for p in portals}) != len(portals):
        raise ValueError("Portal IDs must be unique")
    for portal in portals:
        if portal["wall"] not in wall_ids:
            raise ValueError("Portal references an unknown wall")
        if len(portal["connects"]) != 2 or any(r not in regions for r in portal["connects"]):
            raise ValueError("Portal must connect two known regions")
        numbers = [portal["offset_m"], portal["width_m"], portal["height_m"]]
        if not np.isfinite(numbers).all() or not 0 < numbers[2] < height or numbers[1] <= 0:
            raise ValueError("Invalid portal dimensions")

    boxes = [
        Box("floor", (*bounds[0], -0.12), (*bounds[1], 0.0), (0.6, 0.4, 0.2)),
        Box("ceiling", (*bounds[0], height), (*bounds[1], height + 0.12), (0.84, 0.84, 0.80)),
    ]
    for wall in config["walls"]:
        start = _finite(wall["start"], (2,), "wall start")
        end = _finite(wall["end"], (2,), "wall end")
        delta = end - start
        if np.count_nonzero(np.abs(delta) > 1e-8) != 1 or np.any(delta < 0):
            raise ValueError("Walls must be increasing axis-aligned segments")
        axis = int(np.argmax(delta))
        cross = 1 - axis
        length = delta[axis]
        openings = sorted(
            (p for p in portals if p["wall"] == wall["id"]), key=lambda p: p["offset_m"]
        )
        cursor = 0.0

        def solid(name, low, high, zlow=0.0, start=start, axis=axis, cross=cross):
            if high - low <= 1e-9:
                return
            lo, hi = start.copy(), start.copy()
            lo[axis], hi[axis] = start[axis] + low, start[axis] + high
            lo[cross], hi[cross] = start[cross] - thickness / 2, start[cross] + thickness / 2
            boxes.append(Box(name, (*lo, zlow), (*hi, height), (0.78, 0.76, 0.69)))

        for i, portal in enumerate(openings):
            low = portal["offset_m"] - portal["width_m"] / 2
            high = portal["offset_m"] + portal["width_m"] / 2
            if low < cursor or high > length:
                raise ValueError("Portal overlaps another opening or exceeds its wall")
            solid(f"{wall['id']}_segment_{i}", cursor, low)
            solid(f"{wall['id']}_{portal['id']}_lintel", low, high, portal["height_m"])
            cursor = high
        solid(f"{wall['id']}_segment_{len(openings)}", cursor, length)

    edges = []
    for region in regions.values():
        low, high = np.asarray(region["bounds_xy"], float)
        low, high = low + thickness / 2, high - thickness / 2
        footprint = [low, [high[0], low[1]], high, [low[0], high[1]]]
        for i, first in enumerate(footprint):
            last = footprint[(i + 1) % 4]
            prefix = f"{region['id']}/side_{i}"
            edges.append(Edge(prefix + "_corner", (*first, 0), (*first, height), "wall_wall"))
            edges.append(Edge(prefix + "_top", (*first, height), (*last, height), "wall_ceiling"))
            axis = 0 if abs(last[0] - first[0]) > 1e-9 else 1
            cross = 1 - axis
            wall_coordinate = first[cross]
            a, b = sorted([first[axis], last[axis]])
            openings = []
            for portal in portals:
                if region["id"] not in portal["connects"]:
                    continue
                wall = next(w for w in config["walls"] if w["id"] == portal["wall"])
                wall_axis = 0 if wall["start"][0] != wall["end"][0] else 1
                if wall_axis != axis:
                    continue
                if not np.isclose(abs(wall_coordinate - wall["start"][cross]), thickness / 2):
                    continue
                center = wall["start"][axis] + portal["offset_m"]
                opening = (center - portal["width_m"] / 2, center + portal["width_m"] / 2)
                if opening[0] >= a and opening[1] <= b:
                    openings.append((*opening, portal))
            cursor = a

            def point(coordinate, z, first=first, axis=axis):
                xy = list(first)
                xy[axis] = coordinate
                return (*xy, z)

            for j, (opening_low, opening_high, portal) in enumerate(sorted(openings)):
                edges.append(
                    Edge(
                        prefix + f"_bottom_{j}",
                        point(cursor, 0),
                        point(opening_low, 0),
                        "wall_floor",
                    )
                )
                door = f"{region['id']}/{portal['id']}"
                z = portal["height_m"]
                edges.extend(
                    [
                        Edge(door + "_left", point(opening_low, 0), point(opening_low, z), "door"),
                        Edge(door + "_top", point(opening_low, z), point(opening_high, z), "door"),
                        Edge(
                            door + "_right", point(opening_high, z), point(opening_high, 0), "door"
                        ),
                    ]
                )
                cursor = opening_high
            edges.append(Edge(prefix + "_bottom_last", point(cursor, 0), point(b, 0), "wall_floor"))

    furniture = []
    for i, region in enumerate(config["regions"]):
        furnisher = Furnisher(config["seed"] + i)
        for item in region.get("furnishings", []):
            recipe = item["recipe"]
            if recipe not in RECIPES or recipe == "kitchen_suite":
                raise ValueError(f"Unsupported building furniture recipe: {recipe}")
            getattr(furnisher, recipe)(**item["parameters"])
        for part in furnisher.parts:
            furniture.append(
                replace(
                    part,
                    name=region["id"] + "/" + part.name,
                    object_id=region["id"] + "/" + part.object_id,
                )
            )
    names = [p.name for p in furniture]
    if len(names) != len(set(names)):
        raise ValueError("Furniture part IDs must be unique within each region")
    for part in furniture:
        if not np.isfinite([part.center, part.size, part.rotation]).all() or min(part.size) <= 0:
            raise ValueError(f"Invalid furniture geometry: {part.name}")
    return config, boxes, furniture, edges


def architectural_parts(boxes):
    """Convert reference solids to renderer primitives without changing geometry."""
    return [
        Part(
            box.name,
            box.name,
            "architecture",
            "rounded_box",
            tuple((np.asarray(box.lower) + box.upper) / 2),
            tuple(np.asarray(box.upper) - box.lower),
            "floor" if box.name == "floor" else "wall",
            radius=0.001,
        )
        for box in boxes
    ]
