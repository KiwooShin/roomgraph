"""Metric room geometry and pinhole camera operations, independent of Isaac Sim."""

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Box:
    """Axis-aligned opaque architectural solid in a Z-up world, measured in metres."""

    name: str
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    color: tuple[float, float, float]


@dataclass(frozen=True)
class Edge:
    """An architectural boundary, excluding triangulation and material seams."""

    name: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    kind: str


def demo_room() -> tuple[list[Box], list[Edge]]:
    """A 6 × 5 × 3 m room with an open door and an unglazed window opening."""
    neutral = (0.72, 0.75, 0.78)
    boxes = [
        Box("floor", (-3.2, -2.7, -0.2), (3.2, 2.7, 0), (0.28, 0.20, 0.13)),
        Box("ceiling", (-3.2, -2.7, 3), (3.2, 2.7, 3.2), (0.85, 0.85, 0.82)),
        Box("west_wall", (-3.2, -2.5, 0), (-3, 2.5, 3), neutral),
        Box("south_wall", (-3, -2.7, 0), (3, -2.5, 3), neutral),
        Box("door_left", (-3, 2.5, 0), (-2, 2.7, 3), (0.56, 0.65, 0.69)),
        Box("door_right", (-1, 2.5, 0), (3, 2.7, 3), (0.56, 0.65, 0.69)),
        Box("door_header", (-2, 2.5, 2.2), (-1, 2.7, 3), (0.56, 0.65, 0.69)),
        Box("window_south", (3, -2.5, 0), (3.2, -1, 3), neutral),
        Box("window_north", (3, 1, 0), (3.2, 2.5, 3), neutral),
        Box("window_sill", (3, -1, 0), (3.2, 1, 1), neutral),
        Box("window_header", (3, -1, 2.2), (3.2, 1, 3), neutral),
    ]
    edges = []
    footprint = [(-3, -2.5), (3, -2.5), (3, 2.5), (-3, 2.5)]
    for index, (x, y) in enumerate(footprint):
        edges.append(Edge(f"corner_{index}", (x, y, 0), (x, y, 3), "wall_wall"))
        nx, ny = footprint[(index + 1) % 4]
        edges.append(Edge(f"top_{index}", (x, y, 3), (nx, ny, 3), "wall_ceiling"))
        if index != 2:
            edges.append(Edge(f"bottom_{index}", (x, y, 0), (nx, ny, 0), "wall_floor"))
    edges.extend(
        [
            Edge("bottom_door_left", (-3, 2.5, 0), (-2, 2.5, 0), "wall_floor"),
            Edge("bottom_door_right", (-1, 2.5, 0), (3, 2.5, 0), "wall_floor"),
            Edge("door_left", (-2, 2.5, 0), (-2, 2.5, 2.2), "door"),
            Edge("door_top", (-2, 2.5, 2.2), (-1, 2.5, 2.2), "door"),
            Edge("door_right", (-1, 2.5, 2.2), (-1, 2.5, 0), "door"),
        ]
    )
    window = [(3, -1, 1), (3, 1, 1), (3, 1, 2.2), (3, -1, 2.2)]
    for index, point in enumerate(window):
        edges.append(Edge(f"window_{index}", point, window[(index + 1) % 4], "window"))
    return boxes, edges


def scene_metadata(boxes: list[Box], edges: list[Edge]) -> dict:
    """Return a JSON-serializable specification that can regenerate the scene."""
    return {
        "schema_version": "0.1",
        "world_axes": "right-handed, Z up",
        "units": "metres",
        "boxes": [asdict(box) for box in boxes],
        "edges": [asdict(edge) for edge in edges],
    }


def look_at(eye: NDArray, target: NDArray) -> NDArray:
    """Return camera-to-world: camera +X right, +Y down, +Z forward (OpenCV)."""
    eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
    forward = target - eye
    if np.linalg.norm(forward) < 1e-10:
        raise ValueError("Camera eye and target must differ")
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    if np.linalg.norm(right) < 1e-10:
        raise ValueError("Camera direction must not be parallel to world up")
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, down, forward])
    pose[:3, 3] = eye
    return pose


def project(points: NDArray, intrinsics: NDArray, camera_to_world: NDArray):
    """Project world points; pixels for non-positive camera depth are NaN."""
    camera = (np.asarray(points) - camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
    depth = camera[:, 2]
    pixels = np.full((len(camera), 2), np.nan)
    valid = depth > 0
    homogeneous = camera[valid] @ intrinsics.T
    pixels[valid] = homogeneous[:, :2] / homogeneous[:, 2:3]
    return pixels, depth


def ray_depth(origins: NDArray, directions: NDArray, boxes: list[Box]) -> NDArray:
    """First nonnegative ray/solid intersection; direction need not be normalized.

    For camera rays whose camera-space Z component is one, the parameter is
    optical-axis depth, matching Isaac's distance_to_image_plane annotator.
    """
    origins, directions = np.broadcast_arrays(
        np.asarray(origins, dtype=float), np.asarray(directions, dtype=float)
    )
    result = np.full(origins.shape[:-1], np.inf)
    parallel = np.abs(directions) < 1e-12
    safe = np.where(parallel, 1.0, directions)
    for box in boxes:
        lo, hi = np.asarray(box.lower), np.asarray(box.upper)
        t1, t2 = (lo - origins) / safe, (hi - origins) / safe
        lower = np.where(parallel, -np.inf, np.minimum(t1, t2)).max(axis=-1)
        upper = np.where(parallel, np.inf, np.maximum(t1, t2)).min(axis=-1)
        outside = np.any(parallel & ((origins < lo) | (origins > hi)), axis=-1)
        hit = (~outside) & (upper >= np.maximum(lower, 0))
        candidate = np.where(lower >= 0, lower, upper)
        result = np.minimum(result, np.where(hit, candidate, np.inf))
    return result


def edge_masks(edges, boxes, intrinsics, camera_to_world, width, height):
    """Rasterize sampled edges with analytic occlusion by opaque scene solids.

    Labels are single-pixel samples. Hidden labels include only in-frame,
    positive-depth structural projections. Surface coincidences allow 1 mm.
    """
    visible = np.zeros((height, width), dtype=np.uint8)
    full = np.zeros_like(visible)
    eye = camera_to_world[:3, 3]
    for edge in edges:
        # Dense metric sampling avoids gaps at the demo's camera distances.
        count = max(2, int(np.linalg.norm(np.subtract(edge.end, edge.start)) * 2000))
        points = np.linspace(edge.start, edge.end, count)
        pixels, depth = project(points, intrinsics, camera_to_world)
        valid = (
            (depth > 0.05)
            & np.isfinite(pixels).all(axis=1)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        points, pixels = points[valid], pixels[valid].astype(int)
        if not len(points):
            continue
        full[pixels[:, 1], pixels[:, 0]] = 255
        direction = points - eye
        first = ray_depth(eye, direction, boxes)
        distance = np.linalg.norm(direction, axis=1)
        unobstructed = (first >= 1 - 0.001 / distance) | np.isinf(first)
        pixels = pixels[unobstructed]
        visible[pixels[:, 1], pixels[:, 0]] = 255
    return visible, full


def depth_edge_masks(edges, intrinsics, camera_to_world, depth, tolerance=0.025):
    """Approximate direct visibility using renderer depth, including mesh furniture.

    Edge samples compare their optical-axis depth to the corresponding rendered
    pixel, with 2.5 cm tolerance for silhouettes and finite pixel footprints.
    Reflected architecture is excluded: mirrors are treated as first-hit surfaces.
    """
    height, width = depth.shape
    visible = np.zeros((height, width), dtype=np.uint8)
    full = np.zeros_like(visible)
    for edge in edges:
        count = max(2, int(np.linalg.norm(np.subtract(edge.end, edge.start)) * 2000))
        pixels, z = project(np.linspace(edge.start, edge.end, count), intrinsics, camera_to_world)
        valid = (
            (z > 0.05)
            & np.isfinite(pixels).all(axis=1)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        xy, z = pixels[valid].astype(int), z[valid]
        measured = depth[xy[:, 1], xy[:, 0]]
        full[xy[:, 1], xy[:, 0]] = 255
        seen = (measured > 0) & (z <= measured + tolerance)
        xy = xy[seen]
        visible[xy[:, 1], xy[:, 0]] = 255
    return visible, full
