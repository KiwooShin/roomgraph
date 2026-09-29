"""Fit a Manhattan room shell from typed edges and metric camera calibration.

Inference consumes probability maps and cameras only. No scene dimensions,
renderer depth, furniture geometry, or ground-truth edge positions are inputs.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree

from roomgraph.learning.metrics import thin_prediction


def cuboid_edges(bounds):
    """Return 12 shell segments and their learned channel IDs."""
    xl, xh, yl, yh, zl, zh = np.asarray(bounds)
    corners = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]])
    edges = []
    channels = []
    for i, corner in enumerate(corners):
        other = corners[(i + 1) % 4]
        edges.extend(
            [
                [[*corner, zl], [*corner, zh]],
                [[*corner, zl], [*other, zl]],
                [[*corner, zh], [*other, zh]],
            ]
        )
        channels.extend([1, 2, 3])
    return np.asarray(edges), np.asarray(channels)


def sample_edges(bounds, samples=48):
    edges, channels = cuboid_edges(bounds)
    alpha = np.linspace(0, 1, samples)
    points = (
        edges[:, 0, None, :] * (1 - alpha)[None, :, None]
        + edges[:, 1, None, :] * alpha[None, :, None]
    )
    return points, channels


def _project_all(points, intrinsics, poses):
    world = points.reshape(-1, 3)
    camera = np.einsum("vnj,vjk->vnk", world[None] - poses[:, :3, 3][:, None], poses[:, :3, :3])
    projected = np.einsum("vnj,vkj->vnk", camera, intrinsics)
    depth = camera[:, :, 2]
    pixels = projected[:, :, :2] / np.maximum(depth[:, :, None], 1e-6)
    return pixels, depth


def fit_room(probabilities, records, threshold=0.5, seed=42, maxiter=90):
    """Estimate six axis-aligned planes, with a non-calibrated support diagnostic."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 4 or probabilities.shape[1] != 6:
        raise ValueError("Expected probability maps shaped [views, 6, height, width]")
    if len(records) != len(probabilities) or len(records) < 3:
        raise ValueError("Reconstruction needs at least three calibrated views")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probability maps must be finite")
    intrinsics = np.asarray([r["intrinsics"] for r in records])
    poses = np.asarray([r["camera_to_world"] for r in records])
    if intrinsics.shape != (len(records), 3, 3) or poses.shape != (len(records), 4, 4):
        raise ValueError("Invalid camera matrix shapes")
    if not np.isfinite(intrinsics).all() or not np.isfinite(poses).all():
        raise ValueError("Camera matrices must be finite")
    eyes = poses[:, :3, 3]
    if np.linalg.norm(np.ptp(eyes, axis=0)) < 0.2:
        raise ValueError("Metric reconstruction requires translated camera viewpoints")
    bounds = [
        (eyes[:, 0].min() - 3.5, eyes[:, 0].min() - 0.15),
        (eyes[:, 0].max() + 0.15, eyes[:, 0].max() + 3.5),
        (eyes[:, 1].min() - 3.5, eyes[:, 1].min() - 0.15),
        (eyes[:, 1].max() + 0.15, eyes[:, 1].max() + 3.5),
        (eyes[:, 2].min() - 2.6, eyes[:, 2].min() - 0.3),
        (eyes[:, 2].max() + 0.15, eyes[:, 2].max() + 2.8),
    ]
    count, _, height, width = probabilities.shape
    fields = np.zeros((count, 3, height, width), np.float32)
    for v in range(count):
        for k in range(3):
            mask = thin_prediction(probabilities[v, k + 1], threshold)
            fields[v, k] = np.minimum(distance_transform_edt(~mask), 20) if mask.any() else 20
    samples = 48
    channels = np.repeat(cuboid_edges([0, 1, 0, 1, 0, 1])[1] - 1, samples)
    views = np.arange(count)[:, None]

    def objective(candidate, diagnostic=False):
        points, _ = sample_edges(candidate, samples)
        pixels, depth = _project_all(points, intrinsics, poses)
        x, y = pixels[:, :, 0], pixels[:, :, 1]
        valid = (depth > 0.1) & (x >= 0) & (x < width - 1) & (y >= 0) & (y < height - 1)
        x = np.clip(x, 0, width - 1.001)
        y = np.clip(y, 0, height - 1.001)
        xi, yi = x.astype(int), y.astype(int)
        dx, dy = x - xi, y - yi
        values = (
            fields[views, channels, yi, xi] * (1 - dx) * (1 - dy)
            + fields[views, channels, yi, xi + 1] * dx * (1 - dy)
            + fields[views, channels, yi + 1, xi] * (1 - dx) * dy
            + fields[views, channels, yi + 1, xi + 1] * dx * dy
        )
        valid = valid.reshape(count, 12, samples)
        values = values.reshape(count, 12, samples)
        observations = valid.sum((0, 2))
        errors = (values * valid).sum((0, 2)) / np.maximum(observations, 1)
        errors = np.where(observations >= 8, errors, 20)
        if diagnostic:
            support = ((values <= 2) & valid).sum((0, 2)) / np.maximum(observations, 1)
            view_support = (((values <= 2) & valid).sum(2) >= 5).sum(0)
            return errors, support, view_support
        return float(np.mean(errors))

    result = differential_evolution(
        objective, bounds, seed=seed, maxiter=maxiter, popsize=10, tol=0.001, polish=False
    )
    refined = minimize(
        objective,
        result.x,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 60, "xtol": 1e-4, "ftol": 1e-4},
    )
    estimate = refined.x if refined.fun < result.fun else result.x
    errors, support, view_support = objective(estimate, True)
    segments, kinds = cuboid_edges(estimate)
    return {
        "bounds_m": estimate.tolist(),
        "dimensions_m": (estimate[[1, 3, 5]] - estimate[[0, 2, 4]]).tolist(),
        "objective_pixels": objective(estimate),
        "camera_views": len(records),
        "assumptions": [
            "known metric camera poses",
            "known gravity and Manhattan axes",
            "single rectangular room",
        ],
        "edges": [
            {
                "start": e[0].tolist(),
                "end": e[1].tolist(),
                "channel": int(k),
                "reprojection_error_px": float(error),
                "support_fraction": float(s),
                "supporting_views": int(v),
                "status": "supported" if s > 0.5 and v >= 2 else "inferred",
            }
            for e, k, error, s, v in zip(
                segments, kinds, errors, support, view_support, strict=True
            )
        ],
        "uncertainty_note": (
            "Support is a heuristic, not a calibrated posterior. "
            "Inferred walls do not establish navigable free space."
        ),
        "support_source": "Amodal edge predictions, not verified physical observations",
        "openings_reconstructed": False,
    }


def reconstruction_metrics(prediction, truth):
    predicted = sample_edges(prediction, 200)[0].reshape(-1, 3)
    target = sample_edges(truth, 200)[0].reshape(-1, 3)
    forward = cKDTree(target).query(predicted)[0]
    reverse = cKDTree(predicted).query(target)[0]

    def dims(b):
        return np.asarray(b)[[1, 3, 5]] - np.asarray(b)[[0, 2, 4]]

    error = np.abs(dims(prediction) - dims(truth))
    return {
        "dimension_absolute_error_m": error.tolist(),
        "dimension_mae_m": float(error.mean()),
        "plane_mae_m": float(np.abs(np.asarray(prediction) - truth).mean()),
        "edge_chamfer_m": float((forward.mean() + reverse.mean()) / 2),
        "edge_accuracy_10cm": float(np.mean(forward < 0.1)),
        "edge_completeness_10cm": float(np.mean(reverse < 0.1)),
    }


def write_room_obj(bounds, path):
    """Write the fitted closed shell for inspection; openings are not inferred."""
    xl, xh, yl, yh, zl, zh = bounds
    vertices = [(x, y, z) for z in (zl, zh) for x, y in [(xl, yl), (xh, yl), (xh, yh), (xl, yh)]]
    faces = [(1, 4, 3, 2), (5, 6, 7, 8), (1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7), (4, 1, 5, 8)]
    text = "# Inferred Manhattan room shell; no door/window openings\n"
    text += "".join("v " + " ".join(f"{v:.6f}" for v in point) + "\n" for point in vertices)
    text += "".join("f " + " ".join(map(str, face)) + "\n" for face in faces)
    path.write_text(text)
