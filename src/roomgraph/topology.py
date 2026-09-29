"""Extract provisional connectivity from observed free space, without scene labels.

Clearance cores split a free-space snapshot at narrow necks. A watershed grows
those cores only through observed-free cells. The resulting regions and necks
are geometric candidates, not semantic rooms, detected doors, or proof that a
robot can traverse them. Unknown space always remains outside the graph.
"""

import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed


def _validate(states, origin, resolution, poses, portal_width_m, minimum_region_area_m2):
    states = np.asarray(states)
    origin = np.asarray(origin, dtype=float)
    if states.ndim != 2 or 0 in states.shape or not np.isin(states, [-1, 0, 1]).all():
        raise ValueError("States must be a nonempty [y, x] grid containing -1, 0, or 1")
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("Origin must be a finite XY pair for the lower grid corner")
    scales = np.asarray([resolution, portal_width_m, minimum_region_area_m2], dtype=float)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Resolution, portal width and minimum region area must be positive")
    if poses is None or len(poses) == 0:
        poses = np.empty((0, 2))
    else:
        poses = np.asarray(poses, dtype=float)
        if poses.ndim != 2 or poses.shape[1] not in (2, 3) or not np.isfinite(poses).all():
            raise ValueError("Poses must be finite XY or XYZ positions in acquisition order")
        poses = poses[:, :2]
    return states, origin, poses


def _region_labels(free, clearance, resolution, portal_width_m, minimum_region_area_m2):
    """Seed wide components and retain thin observed components with a fallback seed."""
    cores, count = ndimage.label(free & (clearance > portal_width_m / 2))
    minimum_core_cells = max(1, int(np.ceil(minimum_region_area_m2 * 0.1 / resolution**2)))
    markers = np.zeros(free.shape, np.int32)
    label = 0
    for core in range(1, count + 1):
        mask = cores == core
        if np.count_nonzero(mask) >= minimum_core_cells:
            label += 1
            markers[mask] = label
    components, count = ndimage.label(free)
    for component in range(1, count + 1):
        mask = components == component
        if not np.any(markers[mask]):
            label += 1
            flat = np.flatnonzero(mask)
            chosen = flat[np.argmax(clearance.flat[flat])]
            markers.flat[chosen] = label
    if not label:
        return markers
    labels = watershed(-clearance, markers, mask=free, connectivity=1).astype(np.int32)
    # Number snapshots in spatial order, independent of fallback seed ordering.
    order = sorted(range(1, label + 1), key=lambda k: tuple(np.argwhere(labels == k).mean(0)))
    lookup = np.zeros(label + 1, np.int32)
    lookup[order] = np.arange(1, label + 1)
    return lookup[labels]


def _interfaces(labels, clearance, unknown, origin, resolution):
    """Record region adjacency across observed-free cell faces, never across unknown."""
    interfaces = {}
    for axis in (0, 1):
        left = labels[:-1, :] if axis == 0 else labels[:, :-1]
        right = labels[1:, :] if axis == 0 else labels[:, 1:]
        rows, cols = np.nonzero((left > 0) & (right > 0) & (left != right))
        for row, col in zip(rows, cols, strict=True):
            other = (row + (axis == 0), col + (axis == 1))
            pair = tuple(sorted((int(labels[row, col]), int(labels[other]))))
            interfaces.setdefault(pair, []).append(((row, col), other))
    near_unknown = ndimage.binary_dilation(unknown, structure=np.ones((3, 3)))
    portals = []
    for pair, faces in sorted(interfaces.items()):
        boundary = np.zeros(labels.shape, bool)
        for first, second in faces:
            boundary[first] = boundary[second] = True
        components, count = ndimage.label(boundary, structure=np.ones((3, 3)))
        for component in range(1, count + 1):
            cells = np.argwhere(components == component)
            xy = origin + (cells[:, ::-1] + 0.5) * resolution
            indices = tuple(cells.T)
            portals.append(
                {
                    "id": f"neck_{len(portals) + 1:03d}",
                    "regions": [f"region_{number:03d}" for number in pair],
                    "center_xy_m": xy.mean(0).tolist(),
                    "clearance_width_m": float(2 * clearance[indices].max()),
                    "interface_span_m": (np.ptp(xy, axis=0) + resolution).tolist(),
                    "touches_unknown": bool(np.any(near_unknown[indices])),
                    "status": "observed_free_space_neck_candidate",
                    "verified_doorway": False,
                }
            )
    return portals


def _pose_visits(labels, poses, origin, resolution, portals):
    """Count observed visits and crossings only along completely observed-free segments."""
    height, width = labels.shape

    def lookup(xy):
        cells = np.floor((xy - origin) / resolution).astype(int)
        valid = (cells[:, 0] >= 0) & (cells[:, 0] < width)
        valid &= (cells[:, 1] >= 0) & (cells[:, 1] < height)
        result = np.zeros(len(cells), int)
        result[valid] = labels[cells[valid, 1], cells[valid, 0]]
        return result

    assigned = lookup(poses)
    connected = {tuple(sorted(portal["regions"])) for portal in portals}
    crossings = []
    for index in range(1, len(poses)):
        distance = np.linalg.norm(poses[index] - poses[index - 1])
        count = max(2, int(np.ceil(distance / (resolution * 0.25))) + 1)
        traversed = lookup(np.linspace(poses[index - 1], poses[index], count))
        if np.any(traversed == 0):
            continue
        for previous, current in zip(traversed[:-1], traversed[1:], strict=True):
            if previous == current:
                continue
            pair = (f"region_{previous:03d}", f"region_{current:03d}")
            if tuple(sorted(pair)) in connected:
                crossings.append(
                    {
                        "from_region": pair[0],
                        "to_region": pair[1],
                        "from_observation_index": index - 1,
                        "to_observation_index": index,
                        "status": "pose_segment_crossed_observed_region_boundary",
                    }
                )
    return assigned, crossings


def extract_topology(
    states,
    origin,
    resolution,
    *,
    poses=None,
    portal_width_m=1.6,
    minimum_region_area_m2=1.0,
):
    """Return a JSON-safe provisional region graph from one occupancy snapshot.

    ``states[y, x]`` contains -1 unknown, 0 free, or 1 occupied. ``origin`` is
    the lower XY corner of the grid in metres. Optional ``poses`` are acquired
    XY/XYZ positions, not renderer region IDs. Region IDs are deterministic for
    this snapshot; they are not persistent tracking IDs across map updates.

    ``portal_width_m`` controls the clearance erosion used to separate wide
    interiors at narrow necks. Furnishings and missing observations can split
    real rooms, and wide doorways can merge real rooms. The graph deliberately
    records this as candidate connectivity, never completed building geometry.
    """
    states, origin, poses = _validate(
        states, origin, resolution, poses, portal_width_m, minimum_region_area_m2
    )
    free, unknown = states == 0, states == -1
    # Padding makes the finite map boundary unobserved/blocked even for all-free input.
    clearance = ndimage.distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1] * resolution
    labels = _region_labels(free, clearance, resolution, portal_width_m, minimum_region_area_m2)
    portals = _interfaces(labels, clearance, unknown, origin, resolution)
    assigned, crossings = _pose_visits(labels, poses, origin, resolution, portals)
    regions = []
    for number in range(1, int(labels.max()) + 1):
        mask = labels == number
        cells = np.argwhere(mask)
        xy = origin + (cells[:, ::-1] + 0.5) * resolution
        area = len(cells) * resolution**2
        extent = np.ptp(xy, axis=0) + resolution
        ratio = float(extent.max() / extent.min())
        # Corridor is only an aspect/width cue; it carries no semantic guarantee.
        kind = "room_candidate"
        if area < minimum_region_area_m2:
            kind = "free_space_fragment"
        elif ratio >= 2.5 and extent.min() <= 2 * portal_width_m:
            kind = "corridor_candidate"
        adjacent_unknown = ndimage.binary_dilation(mask) & unknown
        edge = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        visits = np.flatnonzero(assigned == number)
        regions.append(
            {
                "id": f"region_{number:03d}",
                "candidate_type": kind,
                "centroid_xy_m": xy.mean(0).tolist(),
                "bounds_xy_m": [
                    (xy.min(0) - resolution / 2).tolist(),
                    (xy.max(0) + resolution / 2).tolist(),
                ],
                "observed_free_area_m2": float(area),
                "maximum_clearance_m": float(clearance[mask].max()),
                "touches_unknown": bool(adjacent_unknown.any() or edge),
                "observation_visits": int(len(visits)),
                "first_observation_index": int(visits[0]) if len(visits) else None,
                "status": "partial_observed_region",
                "reconstruction_complete": None,
            }
        )
    return {
        "schema_version": "0.1",
        "source": "observed_free_space_only",
        "method": "clearance_core_watershed",
        "origin_xy_m": origin.tolist(),
        "resolution_m": float(resolution),
        "parameters": {
            "portal_width_m": float(portal_width_m),
            "minimum_region_area_m2": float(minimum_region_area_m2),
        },
        "region_labels": labels.tolist(),
        "regions": regions,
        "portals": portals,
        "pose_region_ids": [f"region_{number:03d}" if number else None for number in assigned],
        "observed_region_crossings": crossings,
        "unknown_cell_count": int(np.count_nonzero(unknown)),
        "limitations": [
            "Snapshot candidate regions, not persistent semantic room IDs",
            "Furniture and partial observations can split a room; wide openings can merge rooms",
            "Observed free-space necks are not detected doors or robot-clearance guarantees",
            "Unknown space remains unlabeled; exhausted frontiers do not prove completion",
        ],
    }
