"""Score a completed RGB-D exploration run; scene truth exists only in this process."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from roomgraph.geometry import project

CHANNELS = {"wall_wall": 1, "wall_floor": 2, "wall_ceiling": 3, "door": 4, "window": 5}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_reference(edges, spacing_m=0.025):
    """Length-weighted edge samples; long lines cannot lose weight to short ones."""
    points, weights, regions, channels = [], [], [], []
    for edge in edges:
        start, end = np.asarray(edge["start"], float), np.asarray(edge["end"], float)
        length = np.linalg.norm(end - start)
        if length <= 0:
            continue
        count = max(2, int(np.ceil(length / spacing_m)))
        alpha = (np.arange(count) + 0.5) / count
        points.extend(start + alpha[:, None] * (end - start))
        weights.extend([length / count] * count)
        regions.extend([edge.get("region", edge["name"].split("/")[0])] * count)
        channels.extend([CHANNELS[edge["kind"]]] * count)
    return {
        "points": np.asarray(points).reshape(-1, 3),
        "weights": np.asarray(weights),
        "regions": np.asarray(regions),
        "channels": np.asarray(channels),
    }


def visible_samples(points, record, depth, tolerance_m=0.025):
    """Evaluate direct first-hit visibility; missing or infinite depth never counts."""
    pixels, z = project(
        points, np.asarray(record["intrinsics"]), np.asarray(record["camera_to_world"])
    )
    height, width = depth.shape
    valid = (z > 0.05) & np.isfinite(pixels).all(1)
    valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
    valid &= (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    selected = np.flatnonzero(valid)
    xy = pixels[selected].astype(int)
    measured = depth[xy[:, 1], xy[:, 0]]
    visible = np.zeros(len(points), bool)
    visible[selected] = (
        np.isfinite(measured) & (measured > 0) & (z[selected] <= measured + tolerance_m)
    )
    return visible


def segment_distances(points, edges):
    """Exact nearest distance from each predicted point to reference line segments."""
    distance = np.full(len(points), np.inf)
    for edge in edges:
        start, end = np.asarray(edge["start"], float), np.asarray(edge["end"], float)
        delta = end - start
        squared = np.dot(delta, delta)
        if squared <= 0:
            continue
        fraction = np.clip((points - start) @ delta / squared, 0, 1)
        distance = np.minimum(
            distance, np.linalg.norm(points - start - fraction[:, None] * delta, axis=1)
        )
    return distance


def edge_scores(points, channels, edges, samples, visible):
    """Voxel-point precision and length-weighted reference-line completeness."""
    points, channels = np.asarray(points), np.asarray(channels)
    if points.shape != (len(channels), 3) or not np.isfinite(points).all():
        raise ValueError("Predicted edge points/channels must be aligned and finite")
    weights = samples["weights"]
    forward = segment_distances(points, edges)
    reverse = (
        cKDTree(points).query(samples["points"])[0]
        if len(points)
        else np.full(len(weights), np.inf)
    )
    typed_forward, typed_reverse = np.full(len(points), np.inf), np.full(len(weights), np.inf)
    for channel in CHANNELS.values():
        source, target = channels == channel, samples["channels"] == channel
        typed_edges = [edge for edge in edges if CHANNELS[edge["kind"]] == channel]
        if source.any():
            typed_forward[source] = segment_distances(points[source], typed_edges)
            if target.any():
                typed_reverse[target] = cKDTree(points[source]).query(samples["points"][target])[0]

    def mean_complete(distances, mask=None):
        mask = np.ones(len(weights), bool) if mask is None else mask
        if not mask.any() or not np.isfinite(distances[mask]).all():
            return None
        return float(np.average(distances[mask], weights=weights[mask]))

    def completeness(distances, threshold, mask=None):
        mask = np.ones(len(weights), bool) if mask is None else mask
        return (
            float(np.average(distances[mask] < threshold, weights=weights[mask]))
            if mask.any()
            else None
        )

    accuracy = float(forward.mean()) if len(forward) and np.isfinite(forward).all() else None
    complete = mean_complete(reverse)
    result = {
        "predicted_edge_voxels": len(points),
        "edge_accuracy_mean_m": accuracy,
        "edge_completeness_mean_m": complete,
        "symmetric_edge_chamfer_m": (accuracy + complete) / 2
        if accuracy is not None and complete is not None
        else None,
        "per_region": {},
    }
    for cm in (5, 10):
        threshold = cm / 100
        result[f"edge_precision_{cm}cm"] = (
            float(np.mean(forward < threshold)) if len(points) else 0.0
        )
        result[f"edge_completeness_{cm}cm"] = completeness(reverse, threshold)
        result[f"observed_edge_completeness_{cm}cm"] = completeness(reverse, threshold, visible)
        result[f"typed_edge_precision_{cm}cm"] = (
            float(np.mean(typed_forward < threshold)) if len(points) else 0.0
        )
        result[f"typed_edge_completeness_{cm}cm"] = completeness(typed_reverse, threshold)
    for region in sorted(set(samples["regions"])):
        mask = samples["regions"] == region
        result["per_region"][str(region)] = {
            "reference_edge_length_m": float(weights[mask].sum()),
            "visible_edge_coverage": float(np.average(visible[mask], weights=weights[mask])),
            "edge_completeness_5cm": completeness(reverse, 0.05, mask),
            "edge_completeness_10cm": completeness(reverse, 0.1, mask),
            "observed_edge_completeness_10cm": completeness(reverse, 0.1, mask & visible),
        }
    return result


def densify_path(path, spacing_m=0.025):
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2 or not np.isfinite(path).all():
        raise ValueError("Path must contain finite XY points")
    if len(path) <= 1:
        return path.copy()
    pieces = [path[:1]]
    for start, end in zip(path[:-1], path[1:], strict=True):
        count = max(1, int(np.ceil(np.linalg.norm(end - start) / spacing_m)))
        pieces.append(np.linspace(start, end, count + 1)[1:])
    return np.concatenate(pieces)


def region_at(point, rooms):
    for room in rooms:
        low, high = np.asarray(room["bounds_xy"])
        if np.all(point >= low) and np.all(point < high):
            return room["id"]
    return None


def reference_visits(trace, path, reference):
    """Evaluator-only region names and physical aperture crossings along the path."""
    rooms = reference["rooms"]
    sequence, entered = [], []
    for frame in trace:
        region = region_at(np.asarray(frame["request"]["robot_base"]["position"][:2]), rooms)
        if region and (not sequence or sequence[-1]["region"] != region):
            sequence.append(
                {
                    "region": region,
                    "frame_id": frame["id"],
                    "action_seconds": frame["action_seconds"],
                }
            )
        if region and region not in entered:
            entered.append(region)
    path = np.asarray(path)
    crossings = []
    walls = {wall["id"]: wall for wall in reference["config"]["walls"]}
    for portal in reference["portals"]:
        wall = walls[portal["wall"]]
        start, end = np.asarray(wall["start"]), np.asarray(wall["end"])
        axis = int(np.argmax(np.abs(end - start)))
        normal = 1 - axis
        center = start[axis] + portal["offset_m"]
        signed = path[:, normal] - start[normal]
        previous = None
        for index, value in enumerate(signed):
            if abs(value) <= 1e-9:
                continue
            if previous is not None and value * signed[previous] < 0:
                fraction = -signed[previous] / (value - signed[previous])
                coordinate = path[previous, axis] + fraction * (
                    path[index, axis] - path[previous, axis]
                )
                if abs(coordinate - center) <= portal["width_m"] / 2:
                    crossings.append(
                        {
                            "portal": portal["id"],
                            "connects": portal["connects"],
                            "path_index": index,
                        }
                    )
            previous = index
    return {
        "region_entry_order": entered,
        "region_transition_sequence": sequence,
        "rooms_entered": sum(room["kind"] == "room" and room["id"] in entered for room in rooms),
        "reference_room_count": sum(room["kind"] == "room" for room in rooms),
        "portal_crossings": sorted(crossings, key=lambda item: item["path_index"]),
        "unique_portals_crossed": len({item["portal"] for item in crossings}),
    }


def _box_clearance(points, boxes, footprint_radius_m):
    nearest = np.full(len(points), np.inf)
    segment_nearest = np.full(max(len(points) - 1, 0), np.inf)
    for box in boxes:
        lower, upper = np.asarray(box["lower"]), np.asarray(box["upper"])
        if upper[2] < 0.12 or lower[2] > 1.85:
            continue
        separation = np.maximum(np.maximum(lower[:2] - points, points - upper[:2]), 0)
        distance = np.linalg.norm(separation, axis=1)
        nearest = np.minimum(nearest, distance - footprint_radius_m)
        if len(points) > 1:
            start, delta = points[:-1], np.diff(points, axis=0)
            between = np.minimum(distance[:-1], distance[1:])
            squared = np.einsum("ij,ij->i", delta, delta)
            for corner in (lower[:2], upper[:2], [lower[0], upper[1]], [upper[0], lower[1]]):
                fraction = np.clip(
                    np.einsum("ij,ij->i", np.asarray(corner) - start, delta)
                    / np.maximum(squared, 1e-12),
                    0,
                    1,
                )
                between = np.minimum(
                    between,
                    np.linalg.norm(np.asarray(corner) - start - fraction[:, None] * delta, axis=1),
                )
            parallel = np.abs(delta) < 1e-12
            safe_delta = np.where(parallel, 1, delta)
            t0, t1 = (lower[:2] - start) / safe_delta, (upper[:2] - start) / safe_delta
            near = np.where(parallel, -np.inf, np.minimum(t0, t1)).max(1)
            far = np.where(parallel, np.inf, np.maximum(t0, t1)).min(1)
            outside = np.any(parallel & ((start < lower[:2]) | (start > upper[:2])), axis=1)
            intersects = ~outside & (np.maximum(near, 0) <= np.minimum(far, 1))
            between[intersects] = 0
            segment_nearest = np.minimum(segment_nearest, between - footprint_radius_m)
    return {
        "minimum_footprint_clearance_m": float(nearest.min())
        if len(nearest) and np.isfinite(nearest).any()
        else None,
        "intersecting_samples": int(np.count_nonzero(nearest < -1e-6)),
        "path_samples": len(points),
        "minimum_continuous_footprint_clearance_m": (
            float(segment_nearest.min())
            if len(segment_nearest) and np.isfinite(segment_nearest).any()
            else float(nearest.min())
            if len(nearest) and np.isfinite(nearest).any()
            else None
        ),
        "intersecting_path_segments": int(np.count_nonzero(segment_nearest < -1e-6)),
    }


def path_audit(path, stations, reference, footprint_radius_m):
    """Wall geometry audit plus explicitly approximate furniture and map audits."""
    points = densify_path(path)
    furniture_boxes = []
    for part in reference.get("parts", []):
        half = np.abs(Rotation.from_euler("xyz", part["rotation"], degrees=True).as_matrix()) @ (
            np.asarray(part["size"]) / 2
        )
        center = np.asarray(part["center"])
        furniture_boxes.append({"lower": center - half, "upper": center + half})
    station_checks = []
    for station in stations:
        if "executed_path" not in station:
            continue
        with np.load(station["snapshot"]) as archive:
            states, origin, resolution = (
                archive["states"],
                archive["origin"],
                float(archive["resolution_m"]),
            )
        radius = int(np.ceil(footprint_radius_m / resolution + 0.5))
        yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        separation_x = np.maximum(np.abs(xx) - 0.5, 0) * resolution
        separation_y = np.maximum(np.abs(yy) - 0.5, 0) * resolution
        footprint = np.hypot(separation_x, separation_y) <= footprint_radius_m + 1e-9
        safe = ndimage.binary_erosion(states == 0, structure=footprint, border_value=0)
        movement = densify_path(station["executed_path"], min(0.025, resolution / 4))
        cells = np.floor((movement - origin) / resolution).astype(int)
        valid = (cells[:, 0] >= 0) & (cells[:, 0] < states.shape[1])
        valid &= (cells[:, 1] >= 0) & (cells[:, 1] < states.shape[0])
        permitted = np.zeros(len(cells), bool)
        permitted[valid] = safe[cells[valid, 1], cells[valid, 0]]
        station_checks.append(
            {
                "station": station["station"],
                "samples": len(cells),
                "outside_observed_safe_map": int(np.count_nonzero(~permitted)),
            }
        )
    return {
        "footprint_radius_m": footprint_radius_m,
        "architecture": _box_clearance(points, reference["boxes"], footprint_radius_m),
        "furniture_recipe_aabb_proxy": _box_clearance(points, furniture_boxes, footprint_radius_m),
        "pre_move_map_checks": station_checks,
        "all_motions_in_observed_safe_map": all(
            item["outside_observed_safe_map"] == 0 for item in station_checks
        ),
        "limitations": [
            "Continuous horizontal disk/box audit between heights 0.12 and 1.85 m; no dynamics",
            "Furniture uses rotated recipe bounding boxes; downloaded replacement meshes differ",
        ],
    }


def _timing(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    return (
        {
            "count": len(values),
            "median_ms": float(np.median(values) * 1000),
            "p95_ms": float(np.percentile(values, 95) * 1000),
        }
        if len(values)
        else {"count": 0}
    )


def evaluate(results, reference):
    if (
        results.get("status") != "complete"
        or not results.get("trace")
        or not results.get("stations")
    ):
        raise ValueError("Evaluation requires a completed nonempty experiment")
    samples = sample_reference(reference["edges"])
    if not len(samples["points"]):
        raise ValueError("Reference must contain nonzero architectural edges")
    seen = np.zeros(len(samples["points"]), bool)
    stations_by_last_view = {
        station["view_ids"][-1]: station for station in results["stations"] if station["view_ids"]
    }
    curves, station_scores = [], []
    for frame in results["trace"]:
        geometry = Path(frame["source_geometry"])
        if sha256(geometry) != frame["source_geometry_sha256"]:
            raise ValueError(f"Acquired depth provenance changed: {frame['id']}")
        with np.load(geometry) as archive:
            depth = archive["depth"]
        seen |= visible_samples(samples["points"], frame, depth)
        curves.append(
            {
                "frame_id": frame["id"],
                "action_seconds": frame["action_seconds"],
                "path_length_m": frame["path_length_m"],
                "visible_edge_coverage": float(np.average(seen, weights=samples["weights"])),
            }
        )
        if frame["id"] in stations_by_last_view:
            station = stations_by_last_view[frame["id"]]
            with np.load(station["snapshot"]) as archive:
                score = edge_scores(
                    archive["edge_points"],
                    archive["edge_channels"],
                    reference["edges"],
                    samples,
                    seen,
                )
            station_scores.append(
                {
                    "station": station["station"],
                    "action_seconds": frame["action_seconds"],
                    "visible_edge_coverage": curves[-1]["visible_edge_coverage"],
                    **score,
                }
            )
    final = station_scores[-1]
    trace, config = results["trace"], results["config"]
    return {
        "status": "complete",
        "protocol": "evaluator_only_reference_geometry",
        "sensor_assumptions": config["sensor_assumptions"],
        "frames": len(trace),
        "stations": len(results["stations"]),
        "path_length_m": results["path_length_m"],
        "action_seconds": results["action_seconds"],
        "wall_seconds": results["wall_seconds"],
        "stop_reason": results["stop_reason"],
        "final": final,
        "visibility_curve": curves,
        "station_scores": station_scores,
        "visits": reference_visits(trace, results["path"], reference),
        "path_audit": path_audit(
            results["path"], results["stations"], reference, config["footprint_radius_m"]
        ),
        "timings": {
            "inference": _timing([frame["inference_seconds"] for frame in trace]),
            "mapping": _timing([frame["mapping_seconds"] for frame in trace]),
            "capture": _timing([frame.get("capture_seconds") for frame in trace]),
            "planning": _timing([station["planning_seconds"] for station in results["stations"]]),
        },
        "definitions": {
            "visible_edge_coverage": (
                "Length-weighted true architectural edge visibility in acquired RGB-D frames; "
                "not predicted reconstruction completeness"
            ),
            "edge_precision": (
                "Fraction of fused predicted edge voxels within tolerance of a reference segment"
            ),
            "edge_completeness": (
                "Length-weighted fraction of all reference edge samples within tolerance "
                "of a predicted voxel"
            ),
            "observed_edge_completeness": (
                "Completeness restricted to directly visible reference samples from acquired views"
            ),
            "typed_edge_scores": (
                "Distance requires the predicted wall/ceiling/floor/door/window channel to agree"
            ),
            "symmetric_edge_chamfer": (
                "Mean of point-weighted prediction-to-segment distance and "
                "length-weighted reference-to-prediction distance"
            ),
            "room_names": "Evaluator-only true scene identities; controller receives no room IDs",
        },
        "limitations": [
            "Single validation building; no policy comparison or held-out generalization claim",
            "Ideal depth and known metric poses; previous RGB-only shell scores are incomparable",
            "Sparse observed edge/surface maps, not a completed watertight architectural mesh",
            "Portals are physically open apertures; no moving door leaf or inferred door angle",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.results.parent / "evaluation.json"
    if output.exists():
        raise FileExistsError("Choose a fresh evaluation output to preserve measured results")
    results, reference = (
        json.loads(args.results.read_text()),
        json.loads(args.reference.read_text()),
    )
    report = evaluate(results, reference)
    report["provenance"] = {
        "experiment_sha256": sha256(args.results),
        "reference_sha256": sha256(args.reference),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(
        json.dumps(
            {"output": str(output), "visits": report["visits"], "final": report["final"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
