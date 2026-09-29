"""Small Manhattan surface patches inferred only from acquired RGB-D points.

These are tall planar surface candidates, not guaranteed semantic walls or a
watertight building model. Each face covers one supported observed grid cell.
No faces bridge empty cells, fill door apertures, or extrapolate whole planes.
Known vertical/Manhattan axes and floor z=0 are explicit pilot assumptions.
"""

from pathlib import Path

import numpy as np


def observed_wall_patches(
    points,
    support=None,
    *,
    patch_size_m=0.15,
    plane_tolerance_m=0.035,
    minimum_view_support=2,
    minimum_points_per_patch=2,
    minimum_plane_points=80,
    minimum_height_span_m=1.6,
    minimum_width_m=0.6,
    minimum_observed_fill=0.2,
):
    """Fit axis-aligned planes, exporting only their supported observed patches.

    Input ``support`` counts acquired views per fused surface voxel. Geometric
    density, vertical extent and observed fill reject small furniture faces and
    orthogonal-plane intersection strips, but tall cabinets can still be accepted.
    Per-plane ``evidence_score`` is a descriptive heuristic, not a probability.
    Patch boundaries can extend up to one patch width beyond an observed sample;
    absence of a patch means unobserved/unsupported, not a proven opening.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite acquired surface points [N, 3]")
    support = np.ones(len(points)) if support is None else np.asarray(support, dtype=float)
    if support.shape != (len(points),) or not np.isfinite(support).all() or np.any(support < 1):
        raise ValueError("Surface support must be positive and match points")
    positive = [patch_size_m, plane_tolerance_m, minimum_height_span_m, minimum_width_m]
    counts = [minimum_view_support, minimum_points_per_patch, minimum_plane_points]
    if not np.isfinite([*positive, *counts, minimum_observed_fill]).all():
        raise ValueError("Patch extraction parameters must be finite")
    if min(positive) <= 0 or any(int(value) != value or value < 1 for value in counts):
        raise ValueError("Patch dimensions and integer evidence counts must be positive")
    if not 0 < minimum_observed_fill <= 1:
        raise ValueError("Minimum observed fill must be in (0, 1]")
    keep = (support >= minimum_view_support) & (points[:, 2] >= 0.15)
    points, support = points[keep], support[keep]
    vertices, faces, face_planes, planes = [], [], [], []
    vertex_lookup = {}
    bin_width = plane_tolerance_m * 2
    for axis in (0, 1):
        tangent = 1 - axis
        bins = np.rint(points[:, axis] / bin_width).astype(np.int64)
        keys, counts = np.unique(bins, return_counts=True)
        order = np.lexsort((keys, -counts))
        accepted = []
        for bin_index in order:
            if counts[bin_index] < minimum_plane_points:
                continue
            seed = bins == keys[bin_index]
            offset = float(np.median(points[seed, axis]))
            if any(abs(offset - previous) <= 2 * plane_tolerance_m for previous in accepted):
                continue
            inlier = np.abs(points[:, axis] - offset) <= plane_tolerance_m
            observed, view_support = points[inlier], support[inlier]
            if len(observed) < minimum_plane_points:
                continue
            width, height = np.ptp(observed[:, [tangent, 2]], axis=0)
            if height < minimum_height_span_m or width < minimum_width_m:
                continue
            projected = observed[:, [tangent, 2]]
            cells, inverse, samples = np.unique(
                np.floor(projected / patch_size_m + 1e-8).astype(int),
                axis=0,
                return_inverse=True,
                return_counts=True,
            )
            supported = samples >= minimum_points_per_patch
            selected_cells = cells[supported]
            if not len(selected_cells):
                continue
            extent = np.ptp(cells, axis=0) + 1
            fill = len(selected_cells) / float(np.prod(extent))
            if fill < minimum_observed_fill:
                continue
            # Tall evidence must occur through the middle, not just ceiling/floor strips.
            z_lower, z_upper = np.quantile(observed[:, 2], [0.1, 0.9])
            middle = observed[(observed[:, 2] >= z_lower) & (observed[:, 2] <= z_upper)]
            if len(middle) < minimum_plane_points or np.ptp(middle[:, tangent]) < minimum_width_m:
                continue
            offset = float(np.median(observed[:, axis]))
            residual = float(np.sqrt(np.mean((observed[:, axis] - offset) ** 2)))
            mean_support = float(view_support.mean())
            score = float(
                fill
                * min(1, height / 2)
                * min(1, mean_support / 3)
                * np.exp(-residual / plane_tolerance_m)
            )
            identifier = len(planes)
            patch_support = np.bincount(inverse, weights=view_support) / samples
            planes.append(
                {
                    "id": identifier,
                    "normal_axis": "xy"[axis],
                    "offset_m": offset,
                    "observed_points": len(observed),
                    "patches": len(selected_cells),
                    "observed_fill_fraction": fill,
                    "height_span_m": float(height),
                    "width_span_m": float(width),
                    "rms_residual_m": residual,
                    "mean_view_support": mean_support,
                    "minimum_patch_mean_support": float(patch_support[supported].min()),
                    "evidence_score": score,
                }
            )
            accepted.append(offset)
            for u, v in selected_cells:
                quad = []
                for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    point = np.zeros(3)
                    point[axis] = offset
                    point[tangent] = (u + du) * patch_size_m
                    point[2] = (v + dv) * patch_size_m
                    key = tuple(np.round(point, 8))
                    if key not in vertex_lookup:
                        vertex_lookup[key] = len(vertices)
                        vertices.append(point)
                    quad.append(vertex_lookup[key])
                faces.append(quad)
                face_planes.append(identifier)
    return {
        "vertices": np.asarray(vertices, dtype=float).reshape(-1, 3),
        "faces": np.asarray(faces, dtype=np.int64).reshape(-1, 4),
        "face_plane_ids": np.asarray(face_planes, dtype=np.int32),
        "planes": planes,
        "patch_size_m": float(patch_size_m),
        "limitations": [
            "known Manhattan axes and floor level; no SLAM or semantic wall guarantee",
            "supported observed cells only; missing cells are not proof of open doors",
            "patch discretization can extend up to one patch width beyond observed samples",
            "tall planar furniture can be included; unseen geometry is not completed",
        ],
    }


def write_wall_obj(path, mesh):
    """Write small observed quads, grouped by their inferred plane candidate."""
    vertices = np.asarray(mesh["vertices"])
    faces = np.asarray(mesh["faces"])
    groups = np.asarray(mesh["face_plane_ids"])
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Mesh vertices must be finite [N, 3]")
    if faces.ndim != 2 or faces.shape[1] != 4 or groups.shape != (len(faces),):
        raise ValueError("Expected quad faces and one plane identifier per face")
    if len(faces) and (np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("Mesh face index outside vertices")
    with Path(path).open("w") as stream:
        stream.write("# Observed Manhattan patches; unknown cells remain absent. Units: meters.\n")
        for vertex in vertices:
            stream.write("v " + " ".join(f"{value:.6f}" for value in vertex) + "\n")
        previous = None
        for face, group in zip(faces, groups, strict=True):
            if group != previous:
                stream.write(f"g observed_plane_{group}\n")
                previous = group
            stream.write("f " + " ".join(str(int(index) + 1) for index in face) + "\n")
