"""Real RGB-D calibration, controlled errors and observation-only voxel fusion."""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from roomgraph.active_view import _calibration


def timestamp(path):
    return float(Path(path).stem.rsplit("_", 1)[1])


class ARKitTrajectory:
    """Invert published world-to-camera axis-angle poses; interpolate without extrapolation.

    The convention follows Apple's tenFpsDataLoader.TrajStringToMatrix. Poses
    are estimates, not surveyed ground truth. Interpolation uses camera centers
    and SO(3), never elementwise matrix interpolation.
    """

    def __init__(self, values):
        values = np.asarray(values, float)
        if values.ndim != 2 or values.shape[1] != 7 or len(values) < 2:
            raise ValueError("Expected timestamp, rotation vector, translation rows")
        if not np.isfinite(values).all() or np.any(np.diff(values[:, 0]) <= 0):
            raise ValueError("Trajectory must be finite with strictly increasing times")
        self.times = values[:, 0]
        world_to_camera = Rotation.from_rotvec(values[:, 1:4])
        rotations = world_to_camera.inv()
        self.centers = -rotations.apply(values[:, 4:7])
        self.rotations = Slerp(self.times, rotations)

    def pose(self, time_s, max_gap_s=0.2):
        if not self.times[0] <= time_s <= self.times[-1]:
            raise ValueError("Pose extrapolation is not allowed")
        index = np.clip(np.searchsorted(self.times, time_s), 1, len(self.times) - 1)
        if self.times[index] - self.times[index - 1] > max_gap_s:
            raise ValueError("Trajectory gap is too large")
        pose = np.eye(4)
        pose[:3, :3] = self.rotations(time_s).as_matrix()
        pose[:3, 3] = [np.interp(time_s, self.times, axis) for axis in self.centers.T]
        return pose


def read_intrinsics(path):
    values = np.loadtxt(path)
    if values.shape != (6,) or not np.isfinite(values).all() or np.any(values[:4] <= 0):
        raise ValueError("Expected finite width height fx fy cx cy")
    width, height, fx, fy, cx, cy = values
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), (int(width), int(height))


def scaled_intrinsics(intrinsic, source_size, target_size):
    result = np.array(intrinsic, float, copy=True)
    result[0] *= target_size[0] / source_size[0]
    result[1] *= target_size[1] / source_size[1]
    return result


def upright_record(record, size, clockwise_turns):
    """Rotate pixel coordinates and camera basis together, preserving world rays."""
    if clockwise_turns not in (0, 1, 2, 3):
        raise ValueError("Expected zero to three clockwise quarter turns")
    intrinsic, pose = _calibration(record)
    width, height = size
    raw_from_upright = np.eye(3)
    axis_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    for _ in range(clockwise_turns):
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        intrinsic = np.array([[fy, 0, height - 1 - cy], [0, fx, cx], [0, 0, 1]])
        pose[:3, :3] = pose[:3, :3] @ axis_rotation.T
        raw_from_upright = raw_from_upright @ axis_rotation.T
        width, height = height, width
    return {
        **record,
        "intrinsics": intrinsic.tolist(),
        "camera_to_world": pose.tolist(),
        "size": [width, height],
        "raw_from_upright": raw_from_upright.tolist(),
    }


def perturb(record, specification, index, count, seed):
    """Inject declared camera-center jitter or drift, never change reference poses.

    Gaussian scales are per axis. Rotational errors are local camera rotations;
    drift is a fixed world-X translation and local-Y rotation ramp. Intrinsics
    bias is shared by RGB and registered depth. These are sensitivity tests,
    not a model of every physical tracking or calibration failure.
    """
    intrinsic, pose = _calibration(record)
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    pose[:3, 3] += rng.normal(0, specification.get("translation_std_m", 0), 3)
    rotation = rng.normal(0, np.radians(specification.get("rotation_std_deg", 0)), 3)
    fraction = index / max(1, count - 1)
    pose[0, 3] += fraction * specification.get("drift_translation_m", 0)
    rotation[1] += fraction * np.radians(specification.get("drift_rotation_deg", 0))
    pose[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec(rotation).as_matrix()
    intrinsic[0, 0] *= 1 + specification.get("focal_bias_fraction", 0)
    intrinsic[1, 1] *= 1 + specification.get("focal_bias_fraction", 0)
    intrinsic[:2, 2] += specification.get("principal_bias_px", 0)
    return {"intrinsics": intrinsic.tolist(), "camera_to_world": pose.tolist()}


def voxel_fuse(point_sets, color_sets=None, voxel_size=0.05, min_views=1):
    """One contribution per voxel per frame, with no floor or Manhattan assumptions.

    Requiring multiple frames is a conservative support filter, not pose-error
    correction or calibrated uncertainty. Persistent biased measurements survive.
    """
    if voxel_size <= 0 or min_views < 1:
        raise ValueError("Positive voxel size and view support required")
    contributions, colors = [], []
    for index, points in enumerate(point_sets):
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Finite XYZ points required")
        if not len(points):
            continue
        _, inverse = np.unique(
            np.floor(points / voxel_size).astype(int), axis=0, return_inverse=True
        )
        counts = np.bincount(inverse)
        means = np.column_stack([np.bincount(inverse, weights=a) / counts for a in points.T])
        contributions.append(means)
        if color_sets is not None:
            color = np.asarray(color_sets[index], float)
            if color.shape != points.shape:
                raise ValueError("Color and point shapes differ")
            colors.append(
                np.column_stack([np.bincount(inverse, weights=a) / counts for a in color.T])
            )
    if not contributions:
        return {"points": np.empty((0, 3)), "colors": np.empty((0, 3)), "support": np.zeros(0, int)}
    points = np.concatenate(contributions)
    _, inverse = np.unique(np.floor(points / voxel_size).astype(int), axis=0, return_inverse=True)
    support = np.bincount(inverse)
    means = np.column_stack([np.bincount(inverse, weights=a) / support for a in points.T])
    rgb = np.full_like(means, 150)
    if colors:
        rgb = np.column_stack(
            [np.bincount(inverse, weights=a) / support for a in np.concatenate(colors).T]
        )
    keep = support >= min_views
    return {"points": means[keep], "colors": rgb[keep], "support": support[keep]}


def geometry_metrics(predicted, reference):
    """Nearest-surface agreement to the available reference subset, in its supplied frame."""
    if not len(reference):
        raise ValueError("A nonempty evaluation reference is required")
    result = {"predicted_points": len(predicted), "reference_points": len(reference)}
    if not len(predicted):
        return {
            **result,
            "precision_10cm": 0.0,
            "recall_10cm": 0.0,
            "f1_10cm": 0.0,
            "precision_5cm": 0.0,
            "recall_5cm": 0.0,
            "f1_5cm": 0.0,
            "accuracy_mean_m": None,
            "completeness_mean_m": None,
        }
    forward = cKDTree(reference).query(predicted, workers=1)[0]
    reverse = cKDTree(predicted).query(reference, workers=1)[0]
    for cm in (5, 10):
        precision, recall = float(np.mean(forward <= cm / 100)), float(np.mean(reverse <= cm / 100))
        result.update(
            {
                f"precision_{cm}cm": precision,
                f"recall_{cm}cm": recall,
                f"f1_{cm}cm": 2 * precision * recall / max(precision + recall, 1e-12),
            }
        )
    return {
        **result,
        "accuracy_mean_m": float(forward.mean()),
        "completeness_mean_m": float(reverse.mean()),
    }
