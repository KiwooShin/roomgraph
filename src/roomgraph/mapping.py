"""Online mapping from acquired calibrated depth and RGB edge predictions.

The navigation map is an ideal-depth, flat-floor 2.5D approximation. It projects
the nearest observed body-height obstacle in each azimuth bin into a log-odds
grid; unknown cells block paths. This is not a humanoid collision guarantee or
SLAM: camera poses, depth scale and the horizontal floor plane are supplied.
There is deliberately no scene geometry, room identity or future-frame input.
"""

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush

import numpy as np

from roomgraph.active_view import _calibration
from roomgraph.headcam import HEAD_PART_NAMES, camera_rig_pose, proxy_parts

UNKNOWN = -1
FREE = 0
OCCUPIED = 1


@lru_cache(maxsize=8)
def _robot_boxes(reach):
    """Known proxy part boxes in their original body coordinates, not scene data."""
    result = []
    for part in proxy_parts(reach):
        x, y, z = np.radians(part.rotation)
        cx, cy, cz = np.cos([x, y, z])
        sx, sy, sz = np.sin([x, y, z])
        rotation_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        rotation_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rotation_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        result.append(
            (
                part.name in HEAD_PART_NAMES,
                np.asarray(part.center),
                np.asarray(part.size) / 2,
                rotation_z @ rotation_y @ rotation_x,
            )
        )
    return tuple(result)


def robot_self_mask(points_world, rig_request, reach=0.52, tolerance_m=0.02):
    """Identify acquired points inside the known articulated proxy's part boxes.

    This uses robot geometry and commanded/known joint poses only. Rounded boxes,
    ellipsoids and cylinders use conservative oriented bounding boxes expanded by
    ``tolerance_m``. It can suppress external surfaces within that small envelope;
    unlike a broad exclusion disc it preserves nearby points outside the robot.
    """
    points = np.asarray(points_world, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Self masking expects finite acquired world points [N, 3]")
    if not np.isfinite([reach, tolerance_m]).all() or reach <= 0 or tolerance_m < 0:
        raise ValueError("Robot reach must be positive and mask tolerance nonnegative")
    rig = camera_rig_pose(rig_request)
    if rig is None:
        raise ValueError("Robot self masking requires known robot_base and head poses")
    base, head = rig.base_to_world, rig.head_to_base
    body_points = (points - base[:3, 3]) @ base[:3, :3]
    head_points = (body_points - head[:3, 3]) @ head[:3, :3]
    mask = np.zeros(len(points), dtype=bool)
    for is_head, center, half_extent, rotation in _robot_boxes(float(reach)):
        local = ((head_points if is_head else body_points) - center) @ rotation
        mask |= np.all(np.abs(local) <= half_extent + tolerance_m, axis=1)
    return mask


def mask_robot_depth(depth, record, rig_request, reach=0.52, tolerance_m=0.02):
    """Return ``(filtered_depth, self_mask)`` using calibrated known robot geometry.

    Masked pixels become NaN, preserving the fact that surfaces behind the body
    were not observed. Raw RGB is unchanged for model inference. Use this same
    filtered depth for occupancy, observed surfaces and predicted edge fusion.
    """
    _, pose = _calibration(record)
    rig = camera_rig_pose(rig_request)
    if rig is None or not np.allclose(pose, rig.camera_to_world, atol=1e-6):
        raise ValueError("Sensor extrinsics must match the known articulated robot pose")
    points, pixels = depth_points(depth, record)
    own = robot_self_mask(points, rig_request, reach, tolerance_m)
    filtered = np.array(depth, dtype=np.result_type(np.asarray(depth).dtype, np.float32), copy=True)
    mask = np.zeros(filtered.shape, dtype=bool)
    mask[pixels[own, 1], pixels[own, 0]] = True
    filtered[mask] = np.nan
    return filtered, mask


def depth_points(depth, record, stride=1):
    """Return finite positive optical-z depth samples as world points and pixels.

    ``record`` calibration must correspond to the supplied depth resolution.
    Rays are deliberately not normalized: renderer depth is optical z, not range.
    """
    depth = np.asarray(depth)
    if depth.ndim != 2 or min(depth.shape) < 1:
        raise ValueError("Depth must be a nonempty [height, width] array")
    if int(stride) != stride or stride < 1:
        raise ValueError("Depth sampling stride must be a positive integer")
    intrinsic, pose = _calibration(record)
    yy, xx = np.mgrid[0 : depth.shape[0] : int(stride), 0 : depth.shape[1] : int(stride)]
    values = depth[yy, xx]
    valid = np.isfinite(values) & (values > 0)
    pixels = np.column_stack((xx[valid], yy[valid]))
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(intrinsic).T
    points = (rays * values[valid, None]) @ pose[:3, :3].T + pose[:3, 3]
    return points, pixels


@dataclass(frozen=True)
class Frontier:
    """Reachable stand-off goal facing an observed-free/unknown boundary."""

    goal_xy: tuple[float, float]
    look_at_xy: tuple[float, float]
    path_xy: np.ndarray
    gain_cells: int
    distance_m: float
    score: float


class OccupancyGrid:
    """Fixed-budget occupancy map; its bounds are not a known building outline.

    Each acquired frame contributes at most one free/occupied update per cell.
    All endpoints take precedence over free rays from that frame. Navigation
    uses four-connected cell centers with the robot footprint clear of both
    unknown and occupied cells. No diagonal corner cutting is possible.
    """

    def __init__(
        self,
        bounds_xy=(-15, 15, -15, 15),
        resolution_m=0.1,
        footprint_radius_m=0.28,
        clearance_preference_m=0.3,
    ):
        bounds = np.asarray(bounds_xy, dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all():
            raise ValueError("Expected finite [xmin, xmax, ymin, ymax] bounds")
        if bounds[1] <= bounds[0] or bounds[3] <= bounds[2]:
            raise ValueError("Grid maximum bounds must exceed minimum bounds")
        if not np.isfinite([resolution_m, footprint_radius_m, clearance_preference_m]).all():
            raise ValueError("Grid dimensions must be finite")
        if resolution_m <= 0 or footprint_radius_m < 0 or clearance_preference_m < 0:
            raise ValueError("Resolution must be positive and footprint nonnegative")
        self.bounds_xy = bounds.copy()
        self.resolution_m = float(resolution_m)
        self.footprint_radius_m = float(footprint_radius_m)
        self.clearance_preference_m = float(clearance_preference_m)
        nx, ny = np.ceil((bounds[[1, 3]] - bounds[[0, 2]]) / resolution_m).astype(int)
        self.log_odds = np.zeros((ny, nx), dtype=np.float32)
        self.observed = np.zeros((ny, nx), dtype=bool)
        self.frames = 0

    @property
    def origin(self):
        """Lower world XY corner of the map, independent of any building bounds."""
        return self.bounds_xy[[0, 2]].copy()

    def world_to_cell(self, xy):
        """Convert [..., 2] world XY to [..., 2] integer row/column indices."""
        xy = np.asarray(xy, dtype=float)
        if xy.shape[-1:] != (2,) or not np.isfinite(xy).all():
            raise ValueError("Expected finite world XY coordinates")
        indices = np.floor((xy - self.bounds_xy[[0, 2]]) / self.resolution_m).astype(int)
        return indices[..., ::-1]

    def cell_to_world(self, rc):
        """Convert [..., 2] row/column indices to world XY cell centers."""
        return (np.asarray(rc)[..., ::-1] + 0.5) * self.resolution_m + self.bounds_xy[[0, 2]]

    def _inside(self, rc):
        return np.all((rc >= 0) & (rc < self.log_odds.shape), axis=-1)

    def seed_robot_pose(self, xy):
        """Mark only the already occupied stance footprint as known free.

        This is the physical initial-placement assumption, not a free-space
        oracle. It must not be used to make an unexplored target traversable.
        """
        center = np.asarray(xy, dtype=float)
        rc = self.world_to_cell(center)
        if not self._inside(rc):
            raise ValueError("Robot stance lies outside the map budget")
        yy, xx = np.indices(self.log_odds.shape)
        xy_grid = self.cell_to_world(np.stack((yy, xx), axis=-1))
        mask = np.linalg.norm(xy_grid - center, axis=-1) <= self.footprint_radius_m
        mask[tuple(rc)] = True
        self.observed[mask] = True
        self.log_odds[mask] = -4

    def states(self):
        result = np.full(self.log_odds.shape, UNKNOWN, dtype=np.int8)
        result[self.observed & (self.log_odds <= -0.2)] = FREE
        result[self.observed & (self.log_odds >= 0.6)] = OCCUPIED
        return result

    def integrate_scan(self, origin_xy, endpoints_xy, hit_mask):
        """Integrate already reduced acquired depth rays, primarily for testing.

        ``hit_mask`` distinguishes observed obstacle endpoints from floor-limited
        free rays. Rays are sampled at less than half a grid cell, including
        endpoints. Hit endpoints never receive the same frame's free update.
        """
        origin = np.asarray(origin_xy, dtype=float)
        endpoints = np.asarray(endpoints_xy, dtype=float).reshape(-1, 2)
        hits = np.asarray(hit_mask, dtype=bool)
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("Ray origin must be finite XY")
        if hits.shape != (len(endpoints),) or not np.isfinite(endpoints).all():
            raise ValueError("Ray endpoints and hit flags must match and be finite")
        if not self._inside(self.world_to_cell(origin)):
            raise ValueError("Camera lies outside the map budget")
        free_flat, hit_flat = [], []
        _, nx = self.log_odds.shape
        for endpoint, hit in zip(endpoints, hits, strict=True):
            distance = np.linalg.norm(endpoint - origin)
            count = max(2, int(np.ceil(distance / (self.resolution_m * 0.4))) + 1)
            cells = self.world_to_cell(np.linspace(origin, endpoint, count))
            valid = self._inside(cells)
            cells = cells[valid]
            if len(cells):
                free_flat.extend((cells[:, 0] * nx + cells[:, 1]).tolist())
            cell = self.world_to_cell(endpoint)
            if hit and self._inside(cell):
                hit_flat.append(int(cell[0] * nx + cell[1]))
        free_flat = np.unique(free_flat).astype(int)
        hit_flat = np.unique(hit_flat).astype(int)
        free_flat = np.setdiff1d(free_flat, hit_flat, assume_unique=True)
        log_odds = self.log_odds.reshape(-1)
        observed = self.observed.reshape(-1)
        log_odds[free_flat] = np.maximum(-4, log_odds[free_flat] - 0.7)
        # An observed obstacle immediately blocks a formerly free cell.
        log_odds[hit_flat] = np.minimum(4, np.maximum(1.2, log_odds[hit_flat] + 1.2))
        observed[free_flat] = True
        observed[hit_flat] = True
        self.frames += 1
        return {"rays": len(endpoints), "free_cells": len(free_flat), "hit_cells": len(hit_flat)}

    def integrate_depth(
        self,
        depth,
        record,
        stride=2,
        obstacle_height_m=(0.12, 1.85),
        max_range_m=10.0,
        robot_base_xy=None,
        self_exclusion_radius_m=0.55,
        azimuth_bin_deg=0.5,
    ):
        """Reduce acquired depth to nearest-obstacle/floor-limited horizontal rays.

        The nearest body-height hit in an azimuth bin limits every free ray in
        that bin. If no obstacle was seen, the furthest observed floor sample
        limits clearing. Invalid depth does not clear space. Known robot geometry
        can be excluded by its supplied base XY and a local horizontal radius;
        excluded returns do not produce free rays. Floor z=0 is an explicit prior.
        """
        lower, upper = obstacle_height_m
        if not np.isfinite([lower, upper, max_range_m, self_exclusion_radius_m]).all():
            raise ValueError("Depth mapping limits must be finite")
        if lower < 0 or upper <= lower or max_range_m <= 0 or self_exclusion_radius_m < 0:
            raise ValueError("Invalid depth mapping limits")
        if not np.isfinite(azimuth_bin_deg) or not 0 < azimuth_bin_deg <= 5:
            raise ValueError("Azimuth bins must be in (0, 5] degrees")
        _, pose = _calibration(record)
        points, _ = depth_points(depth, record, stride)
        origin = pose[:2, 3]
        if robot_base_xy is not None:
            base = np.asarray(robot_base_xy, dtype=float)
            if base.shape != (2,) or not np.isfinite(base).all():
                raise ValueError("Robot base must be finite XY")
            points = points[np.linalg.norm(points[:, :2] - base, axis=1) >= self_exclusion_radius_m]
        delta = points[:, :2] - origin
        ranges = np.linalg.norm(delta, axis=1)
        obstacle = (points[:, 2] >= lower) & (points[:, 2] <= upper)
        floor = (points[:, 2] >= -0.05) & (points[:, 2] < lower)
        usable = (ranges > 0.05) & (obstacle | floor)
        delta, ranges = delta[usable], ranges[usable]
        obstacle, floor = obstacle[usable], floor[usable]
        angles = np.arctan2(delta[:, 1], delta[:, 0])
        bin_width = np.radians(azimuth_bin_deg)
        bins = np.floor((angles + np.pi) / bin_width).astype(int)
        endpoints, hits = [], []
        for index in np.unique(bins):
            selected = bins == index
            obstacle_ids = np.flatnonzero(selected & obstacle)
            floor_ids = np.flatnonzero(selected & floor)
            if len(obstacle_ids):
                chosen = obstacle_ids[np.argmin(ranges[obstacle_ids])]
                hit = ranges[chosen] <= max_range_m
            elif len(floor_ids):
                chosen = floor_ids[np.argmax(ranges[floor_ids])]
                hit = False
            else:
                continue
            ray_range = min(float(ranges[chosen]), max_range_m)
            endpoints.append(origin + delta[chosen] * (ray_range / ranges[chosen]))
            hits.append(hit)
        return self.integrate_scan(origin, endpoints, hits)

    def safe_mask(self):
        """Observed free cells with circular-footprint clearance from all else."""
        free = self.states() == FREE
        radius = int(np.ceil(self.footprint_radius_m / self.resolution_m + 0.5))
        padded = np.pad(free, radius, constant_values=False)
        safe = free.copy()
        height, width = free.shape
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                # Distance to the blocked cell rectangle, not just its center.
                # This includes the cell extent in the footprint clearance.
                clearance = np.hypot(max(0, abs(dx) - 0.5), max(0, abs(dy) - 0.5))
                if clearance * self.resolution_m <= self.footprint_radius_m + 1e-9:
                    safe &= padded[
                        radius + dy : radius + dy + height,
                        radius + dx : radius + dx + width,
                    ]
        return safe

    def clearance_m(self):
        """Conservative distance from cell centers to unknown/occupied cell areas.

        Used for a soft routing preference only; ``safe_mask`` remains the hard
        collision constraint. SciPy is supplied by RoomGraph's learning extras.
        """
        from scipy.ndimage import distance_transform_edt

        free = np.pad(self.states() == FREE, 1, constant_values=False)
        center_distance = distance_transform_edt(free)[1:-1, 1:-1] * self.resolution_m
        return np.maximum(0, center_distance - self.resolution_m / np.sqrt(2))

    def _reachable(self, start_xy, safe=None):
        safe = self.safe_mask() if safe is None else safe
        start = tuple(self.world_to_cell(start_xy))
        distances = np.full(safe.shape, np.inf)
        costs = np.full(safe.shape, np.inf)
        parents = {}
        if not self._inside(np.asarray(start)) or not safe[start]:
            return distances, parents
        distances[start] = 0
        costs[start] = 0
        margin = np.maximum(0, self.clearance_m() - self.footprint_radius_m)
        penalty = 1 + self.clearance_preference_m / (margin + self.resolution_m / 2)
        pending = [(0.0, *start)]
        while pending:
            cost, row, column = heappop(pending)
            if cost != costs[row, column]:
                continue
            for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                target = (row + dr, column + dc)
                if not (0 <= target[0] < safe.shape[0] and 0 <= target[1] < safe.shape[1]):
                    continue
                candidate = cost + self.resolution_m * (penalty[row, column] + penalty[target]) / 2
                if safe[target] and candidate < costs[target]:
                    costs[target] = candidate
                    # Physical distance is separate from the clearance-weighted cost.
                    distances[target] = distances[row, column] + self.resolution_m
                    parents[target] = (row, column)
                    heappush(pending, (candidate, *target))
        return distances, parents

    def _path(self, target, parents):
        cells = [target]
        while cells[-1] in parents:
            cells.append(parents[cells[-1]])
        return self.cell_to_world(np.asarray(cells[::-1]))

    def path(self, start_xy, goal_xy):
        """Return an observed-free footprint-safe path or None if unreachable."""
        goal = tuple(self.world_to_cell(goal_xy))
        if not self._inside(np.asarray(goal)):
            return None
        distances, parents = self._reachable(start_xy)
        return self._path(goal, parents) if np.isfinite(distances[goal]) else None

    def frontiers(self, start_xy, min_cells=3, minimum_distance_m=0.25):
        """Find unknown-adjacent components, choosing reachable stand-off goals.

        Goal cells remain inside footprint-safe observed free space. Components
        are not assumed to be rooms or doorways. ``look_at_xy`` faces the nearest
        actual boundary cell rather than claiming the unknown region is free.
        Boundary components use eight neighbors to preserve diagonal visibility
        edges; executable paths still use only four neighbors without corner cuts.
        """
        if min_cells < 1 or minimum_distance_m < 0:
            raise ValueError("Invalid frontier component or distance limit")
        states = self.states()
        unknown = np.pad(states == UNKNOWN, 1, constant_values=True)
        adjacent = unknown[:-2, 1:-1] | unknown[2:, 1:-1] | unknown[1:-1, :-2] | unknown[1:-1, 2:]
        boundary = (states == FREE) & adjacent
        distances, parents = self._reachable(start_xy)
        clearance = self.clearance_m()
        reachable = np.argwhere(np.isfinite(distances))
        if not len(reachable):
            return []
        seen = np.zeros(boundary.shape, bool)
        result = []
        stand_off = self.footprint_radius_m + 2.5 * self.resolution_m
        for row, column in np.argwhere(boundary):
            if seen[row, column]:
                continue
            queue = deque([(int(row), int(column))])
            seen[row, column] = True
            component = []
            while queue:
                cell = queue.popleft()
                component.append(cell)
                for dr, dc in (
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ):
                    other = (cell[0] + dr, cell[1] + dc)
                    if not (
                        0 <= other[0] < boundary.shape[0] and 0 <= other[1] < boundary.shape[1]
                    ):
                        continue
                    if boundary[other] and not seen[other]:
                        queue.append(other)
                        seen[other] = True
            if len(component) < min_cells:
                continue
            cells = np.asarray(component)
            # Consider component cells sparsely for a bounded nearest-neighbor pass.
            samples = cells[:: max(1, len(cells) // 128)]
            nearest = np.full(len(reachable), np.inf)
            facing = np.zeros(len(reachable), dtype=int)
            for index, sample in enumerate(samples):
                squared = np.sum((reachable - sample) ** 2, axis=1)
                better = squared < nearest
                nearest[better] = squared[better]
                facing[better] = index
            nearby = np.flatnonzero(np.sqrt(nearest) * self.resolution_m <= stand_off)
            if not len(nearby):
                continue
            travel = distances[tuple(reachable[nearby].T)]
            eligible = travel >= minimum_distance_m
            if not np.any(eligible):
                continue
            nearby, travel = nearby[eligible], travel[eligible]
            # Stay in the clear part of the boundary's stand-off neighborhood.
            # Only then minimize travel; tiny raster clearance ties are equivalent.
            goal_clearance = clearance[tuple(reachable[nearby].T)]
            clear_enough = goal_clearance >= goal_clearance.max() - self.resolution_m / 2
            nearby, travel = nearby[clear_enough], travel[clear_enough]
            choice = nearby[np.argmin(travel)]
            target = tuple(reachable[choice])
            goal_xy = tuple(self.cell_to_world(reachable[choice]).tolist())
            look_xy = tuple(self.cell_to_world(samples[facing[choice]]).tolist())
            gain = len(component)
            distance = float(distances[target])
            result.append(
                Frontier(
                    goal_xy,
                    look_xy,
                    self._path(target, parents),
                    gain,
                    distance,
                    gain / (1 + distance),
                )
            )
        return sorted(result, key=lambda item: (-item.score, item.goal_xy))

    def snapshot(self):
        states = self.states()
        return {
            "bounds_xy": self.bounds_xy.tolist(),
            "resolution_m": self.resolution_m,
            "footprint_radius_m": self.footprint_radius_m,
            "clearance_preference_m": self.clearance_preference_m,
            "frames": self.frames,
            "free_area_m2": float(np.count_nonzero(states == FREE) * self.resolution_m**2),
            "occupied_area_m2": float(np.count_nonzero(states == OCCUPIED) * self.resolution_m**2),
            "states": states.tolist(),
        }


class StructuralEdgeMap:
    """Voxel-fuse depth-backed visible structural predictions, without cuboid fits.

    A semantic edge requires agreement from visible channel 0 and a typed channel.
    ``support`` counts distinct acquired images; ``camera_support`` counts centers
    separated by a configured translation baseline. Neither implies triangulation
    accuracy, and amodal-only predictions never create 3D points.
    """

    def __init__(self, voxel_size_m=0.05, threshold=0.7, minimum_baseline_m=0.2):
        if not np.isfinite([voxel_size_m, threshold, minimum_baseline_m]).all():
            raise ValueError("Edge fusion parameters must be finite")
        if voxel_size_m <= 0 or minimum_baseline_m <= 0 or not 0 <= threshold <= 1:
            raise ValueError("Invalid edge fusion voxel, threshold or baseline")
        self.voxel_size_m = float(voxel_size_m)
        self.threshold = float(threshold)
        self.minimum_baseline_m = float(minimum_baseline_m)
        self._voxels = {}
        self._observations = set()

    def integrate(self, probabilities, depth, record, observation_id, max_depth_m=15.0):
        predictions = np.asarray(probabilities)
        depth = np.asarray(depth)
        if (
            predictions.ndim != 3
            or predictions.shape[0] < 2
            or predictions.shape[1:] != depth.shape
        ):
            raise ValueError("Expected probabilities [channels, height, width] matching depth")
        if not np.isfinite(predictions).all() or np.any((predictions < 0) | (predictions > 1)):
            raise ValueError("Predictions must be finite probabilities in [0, 1]")
        if not np.isfinite(max_depth_m) or max_depth_m <= 0:
            raise ValueError("Maximum depth must be finite and positive")
        if observation_id in self._observations:
            raise ValueError("An acquired observation cannot be fused twice")
        _, pose = _calibration(record)
        points, pixels = depth_points(depth, record)
        yy, xx = pixels[:, 1], pixels[:, 0]
        visible = (
            (predictions[0, yy, xx] >= self.threshold)
            & (predictions[0, yy, xx] > 0)
            & (depth[yy, xx] <= max_depth_m)
        )
        updates = 0
        for channel in range(1, predictions.shape[0]):
            selected = (
                visible
                & (predictions[channel, yy, xx] >= self.threshold)
                & (predictions[channel, yy, xx] > 0)
            )
            selected_points = points[selected]
            confidence = np.minimum(predictions[0, yy, xx], predictions[channel, yy, xx])[selected]
            quantized = np.floor(selected_points / self.voxel_size_m).astype(int)
            if not len(quantized):
                continue
            keys, inverse = np.unique(quantized, axis=0, return_inverse=True)
            totals = np.bincount(inverse, weights=confidence, minlength=len(keys))
            means = np.column_stack(
                [
                    np.bincount(inverse, weights=selected_points[:, axis] * confidence) / totals
                    for axis in range(3)
                ]
            )
            maxima = np.zeros(len(keys))
            np.maximum.at(maxima, inverse, confidence)
            for voxel, point, weight in zip(keys, means, maxima, strict=True):
                key = (channel, *voxel.tolist())
                previous = self._voxels.get(key)
                if previous is None:
                    self._voxels[key] = {
                        "sum": point * weight,
                        "weight": float(weight),
                        "support": 1,
                        "centers": [pose[:3, 3].copy()],
                    }
                else:
                    previous["sum"] += point * weight
                    previous["weight"] += float(weight)
                    previous["support"] += 1
                    if all(
                        np.linalg.norm(pose[:3, 3] - center) >= self.minimum_baseline_m
                        for center in previous["centers"]
                    ):
                        previous["centers"].append(pose[:3, 3].copy())
                updates += 1
        self._observations.add(observation_id)
        return {"updated_voxels": updates, "total_voxels": len(self._voxels)}

    def arrays(self):
        keys = sorted(self._voxels)
        values = [self._voxels[key] for key in keys]
        return {
            "points": np.asarray([value["sum"] / value["weight"] for value in values]).reshape(
                -1, 3
            ),
            "channels": np.asarray([key[0] for key in keys], dtype=np.int8),
            "support": np.asarray([value["support"] for value in values], dtype=np.int32),
            "camera_support": np.asarray(
                [len(value["centers"]) for value in values], dtype=np.int32
            ),
            "confidence": np.asarray([value["weight"] / value["support"] for value in values]),
        }
