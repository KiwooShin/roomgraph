"""Renderer-side preflight for connected-space development layouts.

Reference walls, furniture and region labels are used only to reject invalid
scene descriptions. They must never enter the observation-driven planner.
"""

import hashlib
from collections import deque
from itertools import combinations
from pathlib import Path

import numpy as np

from roomgraph.building import load_building


def _part_bounds(part):
    """Conservative world AABB of the actual rotated primitive dimensions."""
    x, y, z = np.radians(part.rotation)
    sx, sy, sz = np.sin([x, y, z])
    cx, cy, cz = np.cos([x, y, z])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    extent = np.abs(rz @ ry @ rx) @ (np.asarray(part.size) / 2)
    return np.asarray(part.center) - extent, np.asarray(part.center) + extent


def _covers(intervals, low, high):
    cursor = low
    for first, last in sorted(intervals):
        if first > cursor + 1e-7:
            return False
        if last >= cursor:
            cursor = max(cursor, last)
        if cursor >= high - 1e-7:
            return True
    return False


def _inside_circle(points, bounds, radius):
    low, high = bounds
    nearest = np.clip(points, low[:2], high[:2])
    return np.sum((points - nearest) ** 2, axis=-1) <= radius**2


def _reachable_targets(bounds, obstacle_bounds, start, targets, radius, spacing=0.1):
    low, high = np.asarray(bounds)
    shape = np.ceil((high - low) / spacing).astype(int)
    xx, yy = np.meshgrid(
        low[0] + (np.arange(shape[0]) + 0.5) * spacing,
        low[1] + (np.arange(shape[1]) + 0.5) * spacing,
    )
    points = np.stack([xx, yy], axis=-1)
    blocked = np.zeros(xx.shape, dtype=bool)
    # Inflate by half a grid diagonal so a free cell is a conservative stance.
    for box in obstacle_bounds:
        blocked |= _inside_circle(points, box, radius + spacing / np.sqrt(2))

    def cell(point):
        index = np.floor((np.asarray(point)[:2] - low) / spacing).astype(int)
        return int(index[1]), int(index[0])

    first = cell(start)
    if blocked[first]:
        raise ValueError("Start stance has insufficient conservative grid clearance")
    visited = {first}
    queue = deque([first])
    while queue:
        row, col = queue.popleft()
        for key in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if (
                0 <= key[0] < blocked.shape[0]
                and 0 <= key[1] < blocked.shape[1]
                and key not in visited
                and not blocked[key]
            ):
                visited.add(key)
                queue.append(key)
    missing = [name for name, point in targets if cell(point) not in visited]
    if missing:
        raise ValueError(f"Portal approaches are unreachable from start: {missing}")
    return float(len(visited) * spacing**2)


def validate_space_case(path, footprint_radius_m=0.25):
    """Reject false room boundaries, disconnected layouts and blocked starts.

    Uses conservative primitive AABBs for renderer-side authoring validation,
    not a humanoid collision/dynamics certificate. Decorative imported assets
    remain subject to the runtime depth-based clearance checks.
    """
    config, boxes, parts, _ = load_building(path)
    radius = float(footprint_radius_m)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Footprint radius must be finite and positive")
    bounds = np.asarray(config["bounds_xy"], float)
    regions = {r["id"]: r for r in config["regions"]}
    region_bounds = {key: np.asarray(r["bounds_xy"], float) for key, r in regions.items()}
    area = float(np.prod(bounds[1] - bounds[0]))
    if not np.isclose(sum(np.prod(v[1] - v[0]) for v in region_bounds.values()), area):
        raise ValueError("Regions must tile the complete rectangular building footprint")
    for a, b in combinations(region_bounds.values(), 2):
        if np.all(np.minimum(a[1], b[1]) - np.maximum(a[0], b[0]) > 1e-7):
            raise ValueError("Region interiors must not overlap")

    walls = {wall["id"]: wall for wall in config["walls"]}
    for region in region_bounds.values():
        for axis in (0, 1):
            cross = 1 - axis
            for coordinate in region[:, cross]:
                supporting = []
                for wall in walls.values():
                    start, end = np.asarray(wall["start"]), np.asarray(wall["end"])
                    if np.isclose(start[cross], coordinate) and np.isclose(end[cross], coordinate):
                        supporting.append((start[axis], end[axis]))
                if not _covers(supporting, region[0, axis], region[1, axis]):
                    raise ValueError("Every region boundary must have a physical wall")

    graph = {key: set() for key in regions}
    targets = []
    margin = config["wall_thickness_m"] / 2
    for portal in config["portals"]:
        first, second = portal["connects"]
        if first == second:
            raise ValueError("A portal must connect distinct regions")
        a, b = region_bounds[first], region_bounds[second]
        wall = walls[portal["wall"]]
        start, end = np.asarray(wall["start"]), np.asarray(wall["end"])
        axis = int(np.argmax(end - start))
        cross = 1 - axis
        coordinate = start[cross]
        shared = (np.isclose(a[1, cross], coordinate) and np.isclose(b[0, cross], coordinate)) or (
            np.isclose(b[1, cross], coordinate) and np.isclose(a[0, cross], coordinate)
        )
        center = start.copy().astype(float)
        center[axis] += portal["offset_m"]
        low, high = center[axis] + np.array([-1, 1]) * portal["width_m"] / 2
        shared_low, shared_high = max(a[0, axis], b[0, axis]), min(a[1, axis], b[1, axis])
        if not shared or low < shared_low + margin or high > shared_high - margin:
            raise ValueError(f"Portal {portal['id']} does not lie on its regions' shared boundary")
        graph[first].add(second)
        graph[second].add(first)
        for sign in (-1, 1):
            point = center.copy()
            point[cross] += sign * (radius + margin + 0.18)
            targets.append((f"{portal['id']}:{sign:+d}", point))
    visited = set()
    pending = [next(iter(regions))]
    while pending:
        current = pending.pop()
        if current not in visited:
            visited.add(current)
            pending.extend(graph[current] - visited)
    if visited != set(regions):
        raise ValueError("Portal graph must connect every region")

    start = np.asarray(config["start"]["position"], float)
    if start.shape != (3,) or not np.isfinite(start).all() or not np.isclose(start[2], 0):
        raise ValueError("Start must be a finite floor-level 3D position")
    if not np.isfinite(config["start"]["yaw_deg"]):
        raise ValueError("Start body yaw must be finite")
    if not np.all((start[:2] > bounds[0]) & (start[:2] < bounds[1])):
        raise ValueError("Start must be inside the building")
    obstacles = []
    for name, low, high in [(b.name, np.asarray(b.lower), np.asarray(b.upper)) for b in boxes] + [
        (p.name, *_part_bounds(p)) for p in parts
    ]:
        if high[2] <= 0.12 or low[2] >= 1.85:
            continue
        obstacles.append((low, high))
        if _inside_circle(start[:2], (low, high), radius):
            raise ValueError(f"Start footprint intersects actual primitive: {name}")
    reachable = _reachable_targets(bounds, obstacles, start, targets, radius)
    return {
        "building_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "split": config.get("dataset", {}).get("split"),
        "floor_area_m2": area,
        "regions": len(regions),
        "rooms": sum(r["kind"] == "room" for r in regions.values()),
        "portals": len(config["portals"]),
        "graph_cycle_rank": len(config["portals"]) - len(regions) + 1,
        "max_region_degree": max(map(len, graph.values())),
        "minimum_portal_width_m": min(p["width_m"] for p in config["portals"]),
        "primitive_clearance_reachable_area_m2": reachable,
        "furniture_objects": len({p.object_id for p in parts}),
        "furniture_parts": len(parts),
        "validation_scope": "reference partition and conservative procedural-primitive clearance",
    }
