"""Causal head-view selection from acquired images and an inferred shell.

Angular coverage means looked-at directions, not reconstructed surfaces. Shell
support is a reprojection heuristic; even agreement from translated cameras is
not a calibrated geometry estimate or evidence of navigable free space.
"""

from dataclasses import dataclass

import numpy as np


def _calibration(record):
    intrinsic = np.asarray(record["intrinsics"], dtype=float)
    pose = np.asarray(record["camera_to_world"], dtype=float)
    if intrinsic.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("Expected intrinsics [3, 3] and camera_to_world [4, 4]")
    if not np.isfinite(intrinsic).all() or not np.isfinite(pose).all():
        raise ValueError("Camera calibration must be finite")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError("Camera focal lengths must be positive")
    rotation = pose[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
        or not np.allclose(pose[3], [0, 0, 0, 1])
    ):
        raise ValueError("camera_to_world must be a rigid, right-handed transform")
    return intrinsic.copy(), pose.copy()


@dataclass(frozen=True)
class HeadView:
    """Candidate calibration and joint targets, with no candidate image data."""

    view_id: str
    yaw_deg: float
    pitch_deg: float
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    image_shape: tuple[int, int]

    @classmethod
    def from_record(cls, view_id, yaw_deg, pitch_deg, record, image_shape=(256, 384)):
        intrinsic, pose = _calibration(record)
        if len(image_shape) != 2 or min(image_shape) <= 0:
            raise ValueError("image_shape must be positive (height, width)")
        if not np.isfinite([yaw_deg, pitch_deg]).all():
            raise ValueError("Head angles must be finite")
        return cls(str(view_id), float(yaw_deg), float(pitch_deg), intrinsic, pose, image_shape)


@dataclass(frozen=True)
class HeadMotionLimits:
    """Illustrative simultaneous-axis, rest-to-rest motion limits; not robot specs."""

    yaw_bounds: tuple[float, float] = (-80, 80)
    pitch_bounds: tuple[float, float] = (-35, 30)
    speed_deg_s: tuple[float, float] = (60, 45)
    accel_deg_s2: tuple[float, float] = (120, 90)
    settle_s: float = 0.15

    def __post_init__(self):
        values = [*self.yaw_bounds, *self.pitch_bounds, *self.speed_deg_s, *self.accel_deg_s2]
        if not np.isfinite([*values, self.settle_s]).all():
            raise ValueError("Motion limits must be finite")
        if min(*self.speed_deg_s, *self.accel_deg_s2, self.settle_s) <= 0:
            raise ValueError("Speed, acceleration and settling time must be positive")
        if self.yaw_bounds[0] >= self.yaw_bounds[1] or self.pitch_bounds[0] >= self.pitch_bounds[1]:
            raise ValueError("Head joint bounds must be increasing")

    def validate(self, angles):
        if len(angles) != 2 or not np.isfinite(angles).all():
            raise ValueError("Expected finite (yaw, pitch) angles")
        if not all(
            lo <= value <= hi
            for value, (lo, hi) in zip(angles, (self.yaw_bounds, self.pitch_bounds), strict=True)
        ):
            raise ValueError("Head target is outside joint limits")

    def duration(self, start, target):
        """Seconds for a triangular/trapezoidal velocity profile plus settling."""
        self.validate(start)
        self.validate(target)
        distance = np.abs(np.asarray(target) - start)
        speed, accel = np.asarray(self.speed_deg_s), np.asarray(self.accel_deg_s2)
        seconds = np.where(
            distance <= speed**2 / accel,
            2 * np.sqrt(distance / accel),
            distance / speed + speed / accel,
        )
        return float(seconds.max() + self.settle_s)


def _shell_samples(bounds, samples):
    if bounds is None:
        return np.zeros((0, 3)), np.zeros(0, int)
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (6,) or not np.isfinite(bounds).all():
        raise ValueError("Expected six finite shell bounds")
    if np.any(bounds[[1, 3, 5]] <= bounds[[0, 2, 4]]):
        raise ValueError("Shell maximum bounds must exceed minimum bounds")
    xl, xh, yl, yh, zl, zh = bounds
    corners = [[xl, yl], [xh, yl], [xh, yh], [xl, yh]]
    points, channels = [], []
    for i, corner in enumerate(corners):
        other = corners[(i + 1) % 4]
        for channel, a, b in [
            (1, [*corner, zl], [*corner, zh]),
            (2, [*corner, zl], [*other, zl]),
            (3, [*corner, zh], [*other, zh]),
        ]:
            points.extend(np.linspace(a, b, samples))
            channels.extend([channel] * samples)
    return np.asarray(points), np.asarray(channels)


def _projection(points, intrinsic, pose, image_shape, directions=False):
    camera = (points if directions else points - pose[:3, 3]) @ pose[:3, :3]
    homogeneous = camera @ intrinsic.T
    pixels = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:3], 1e-8)
    height, width = image_shape
    valid = (
        (camera[:, 2] > 0.05)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width - 0.5)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height - 0.5)
    )
    return pixels, valid


def _near_mask(mask, pixels, tolerance):
    """Check circular pixel tolerance without a SciPy or GPU dependency."""
    height, width = mask.shape
    xy = np.rint(pixels).astype(int)
    result = np.zeros(len(pixels), bool)
    for dy in range(-tolerance, tolerance + 1):
        for dx in range(-tolerance, tolerance + 1):
            if dx * dx + dy * dy > tolerance * tolerance:
                continue
            x, y = xy[:, 0] + dx, xy[:, 1] + dy
            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            result[valid] |= mask[y[valid], x[valid]]
    return result


class ActiveHeadPlanner:
    """Greedy expected coverage per second, replanned after each acquired image.

    The directional sphere is uniform in azimuth and sine(elevation), making its
    cells equal-area. Candidate projections use a fitted shell hypothesis only.
    No future images, ground-truth geometry, labels or renderer depth are inputs.
    """

    def __init__(
        self,
        candidates,
        shell_bounds_m=None,
        limits=None,
        threshold=0.7,
        tolerance_px=3,
        minimum_baseline_m=0.2,
        shell_samples=24,
    ):
        self.candidates = tuple(candidates)
        if not self.candidates or len({v.view_id for v in self.candidates}) != len(self.candidates):
            raise ValueError("Provide nonempty candidates with unique view IDs")
        self.limits = limits or HeadMotionLimits()
        for view in self.candidates:
            self.limits.validate((view.yaw_deg, view.pitch_deg))
        if not 0 <= threshold <= 1 or tolerance_px < 0 or int(tolerance_px) != tolerance_px:
            raise ValueError("Threshold must be in [0, 1] and tolerance a nonnegative integer")
        if minimum_baseline_m <= 0 or not np.isfinite(minimum_baseline_m):
            raise ValueError("Minimum translation baseline must be finite and positive")
        if shell_samples < 2:
            raise ValueError("At least two samples per edge are required")
        self.threshold = threshold
        self.tolerance_px = int(tolerance_px)
        self.minimum_baseline_m = minimum_baseline_m
        azimuth, height = np.meshgrid(
            np.linspace(-np.pi, np.pi, 72, endpoint=False), np.linspace(-1 + 1 / 36, 1 - 1 / 36, 36)
        )
        radius = np.sqrt(1 - height**2)
        self.directions = np.stack(
            [radius * np.cos(azimuth), radius * np.sin(azimuth), height], axis=-1
        ).reshape(-1, 3)
        self.points, self.channels = _shell_samples(shell_bounds_m, shell_samples)
        self.direction_seen = np.zeros(len(self.directions), bool)
        self.inspections = np.zeros(len(self.points), int)
        self.support_count = np.zeros(len(self.points), int)
        self.multi_center_support = np.zeros(len(self.points), bool)
        self._support_centers = [[] for _ in self.points]
        self.observations = 0
        self.acquired_ids = set()
        self._candidate_directions = {}
        self._candidate_points = {}
        for view in self.candidates:
            args = (view.intrinsics, view.camera_to_world, view.image_shape)
            self._candidate_directions[view.view_id] = _projection(
                self.directions, *args, directions=True
            )[1]
            self._candidate_points[view.view_id] = _projection(self.points, *args)[1]
        self.reachable_directions = np.any(list(self._candidate_directions.values()), axis=0)

    def observe(self, probabilities, record, view_id=None):
        """Fuse predictions from one already acquired image; input calibration is scaled."""
        probability = np.asarray(probabilities)
        if probability.ndim != 3 or probability.shape[0] != 6 or min(probability.shape[1:]) <= 0:
            raise ValueError("Expected probabilities [6, height, width]")
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("Probabilities must be finite and in [0, 1]")
        intrinsic, pose = _calibration(record)
        shape = probability.shape[1:]
        self.direction_seen |= _projection(
            self.directions, intrinsic, pose, shape, directions=True
        )[1]
        pixels, valid = _projection(self.points, intrinsic, pose, shape)
        self.inspections += valid
        visible = _near_mask(probability[0] >= self.threshold, pixels[valid], self.tolerance_px)
        indices = np.flatnonzero(valid)
        for channel in (1, 2, 3):
            typed = _near_mask(
                probability[channel] >= self.threshold, pixels[valid], self.tolerance_px
            )
            supported = indices[visible & typed & (self.channels[valid] == channel)]
            for index in supported:
                center = pose[:3, 3]
                previous = self._support_centers[index]
                if previous and np.any(
                    np.linalg.norm(np.asarray(previous) - center, axis=1) >= self.minimum_baseline_m
                ):
                    self.multi_center_support[index] = True
                if not previous or np.all(
                    np.linalg.norm(np.asarray(previous) - center, axis=1) > 1e-4
                ):
                    previous.append(center.copy())
                self.support_count[index] += 1
        self.observations += 1
        identifier = view_id if view_id is not None else record.get("id")
        if identifier is not None:
            self.acquired_ids.add(str(identifier))

    def action_time(self, current, target):
        """Shared motion clock for active, random and raster comparison policies."""
        if isinstance(target, HeadView):
            target = (target.yaw_deg, target.pitch_deg)
        return self.limits.duration(current, target)

    def _choose(self, current, remaining_s, seen, inspections, excluded=None):
        if not np.isfinite(remaining_s) or remaining_s < 0:
            raise ValueError("Remaining action budget must be finite and nonnegative")
        self.limits.validate(current)
        excluded = self.acquired_ids if excluded is None else excluded
        choices = []
        for view in self.candidates:
            if view.view_id in excluded:
                continue
            duration = self.action_time(current, view)
            if duration > remaining_s + 1e-9:
                continue
            new_direction = self._candidate_directions[view.view_id] & ~seen
            unresolved = self._candidate_points[view.view_id] & ~self.multi_center_support
            direction_gain = new_direction.sum() / max(self.reachable_directions.sum(), 1)
            edge_gain = (unresolved / (1 + inspections)).sum() / max(len(self.points), 1)
            # Reinspection has diminishing value, but cannot manufacture support.
            # A zero-motion repeated frame adds no new viewpoint or directional coverage.
            if np.allclose(current, (view.yaw_deg, view.pitch_deg)) and not new_direction.any():
                continue
            score = (direction_gain + 0.65 * edge_gain) / duration
            score -= 0.01 * (duration - self.limits.settle_s)
            if score <= 0:
                continue
            choices.append(
                {
                    "view_id": view.view_id,
                    "yaw_deg": view.yaw_deg,
                    "pitch_deg": view.pitch_deg,
                    "duration_s": duration,
                    "score": float(score),
                    "new_direction_cells": int(new_direction.sum()),
                    "unresolved_shell_samples_in_fov": int(unresolved.sum()),
                    "expected_direction_gain": float(direction_gain),
                    "expected_edge_inspection_gain": float(edge_gain),
                }
            )
        return max(choices, key=lambda x: (x["score"], x["view_id"])) if choices else None

    def choose(self, current=(0, 0), remaining_s=3):
        """Choose the next achievable target; None requests another strategy or station."""
        return self._choose(current, remaining_s, self.direction_seen, self.inspections)

    def plan(self, current=(0, 0), horizon_s=3):
        """Preview a short greedy sequence; future inspections remain hypothetical.

        Execute only the first action, acquire its image, and replan. This method
        never updates observed evidence or assumes successful future reconstruction.
        """
        seen, inspections = self.direction_seen.copy(), self.inspections.copy()
        remaining, actions = horizon_s, []
        excluded = self.acquired_ids.copy()
        while len(actions) < 32:
            action = self._choose(current, remaining, seen, inspections, excluded)
            if action is None:
                break
            actions.append({**action, "hypothetical": True})
            excluded.add(action["view_id"])
            seen |= self._candidate_directions[action["view_id"]]
            inspections += self._candidate_points[action["view_id"]]
            remaining = max(0, remaining - action["duration_s"])
            current = (action["yaw_deg"], action["pitch_deg"])
        return actions

    def diagnostics(self):
        """JSON-safe evidence counts; support labels are deliberately non-probabilistic."""
        reachable = self.reachable_directions
        supported = self.support_count > 0
        return {
            "acquired_views": self.observations,
            "reachable_direction_cells": int(reachable.sum()),
            "looked_direction_cells": int((self.direction_seen & reachable).sum()),
            "looked_direction_fraction": float(
                (self.direction_seen & reachable).sum() / max(reachable.sum(), 1)
            ),
            "shell_samples": len(self.points),
            "shell_unseen": int((self.inspections == 0).sum()),
            "shell_looked_unsupported": int(((self.inspections > 0) & ~supported).sum()),
            "shell_single_center_support": int((supported & ~self.multi_center_support).sum()),
            "shell_multiple_center_support": int(self.multi_center_support.sum()),
            "support_note": (
                "Visible and typed edge agreement is a heuristic, not verified 3D geometry. "
                "Multiple-center support requires translated views. Angular coverage records "
                "looked directions only; inferred walls do not establish navigable free space."
            ),
        }

    def shell_state(self):
        """Sample-level diagnostics for inspection or uncertainty-map visualization."""
        return {
            "points_m": self.points.tolist(),
            "channel": self.channels.tolist(),
            "state": self.shell_states.tolist(),
            "inspection_count": self.inspections.tolist(),
        }

    @property
    def shell_points(self):
        return self.points.copy()

    @property
    def shell_channels(self):
        return self.channels.copy()

    @property
    def shell_states(self):
        state = np.full(len(self.points), "unseen", dtype=object)
        state[self.inspections > 0] = "looked_unsupported"
        state[self.support_count > 0] = "single_center_support"
        state[self.multi_center_support] = "multiple_center_support"
        return state

    def directional_state(self):
        """Equal-area world directions and looked flags for a directional heatmap."""
        return {
            "directions_world": self.directions.tolist(),
            "looked": self.direction_seen.tolist(),
            "reachable": self.reachable_directions.tolist(),
        }
