"""Replay frozen acquired RGB-D/predictions on CPU; no renderer, model, or reference scene."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from run_active_head import camera_record
from run_multiroom import angular_distance, choose_frontier, path_prefix, scan_targets

from roomgraph.mapping import OccupancyGrid, StructuralEdgeMap, mask_robot_depth
from roomgraph.surface_map import SurfaceMap
from roomgraph.topology import extract_topology


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_hash(path, expected, label):
    if file_hash(path) != expected:
        raise ValueError(f"Frozen {label} SHA256 differs: {Path(path).name}")


def require_equal(actual, expected, label, atol=0):
    """Exact integers, and explicit absolute-only tolerance for finite float arrays."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f"Replay shape differs for {label}: {actual.shape} != {expected.shape}")
    if np.issubdtype(actual.dtype, np.floating) or np.issubdtype(expected.dtype, np.floating):
        equal = np.allclose(actual, expected, rtol=0, atol=atol, equal_nan=True)
    else:
        equal = np.array_equal(actual, expected)
    if not equal:
        raise ValueError(f"Replay values differ for {label} (absolute tolerance {atol})")


def compare_snapshot(station, grid, surface, edges, atol):
    arrays = {
        "states": grid.states(),
        "origin": grid.origin,
        "resolution_m": np.asarray(grid.resolution_m),
        **{f"surface_{key}": value for key, value in surface.arrays().items()},
        **{f"edge_{key}": value for key, value in edges.arrays().items()},
    }
    with np.load(station["snapshot"]) as saved:
        if set(arrays) != set(saved.files):
            raise ValueError("Station snapshot array keys differ from the replay schema")
        for name, value in arrays.items():
            require_equal(value, saved[name], f"station {station['station']} {name}", atol)
    return len(arrays)


def replay(results, *, atol=0):
    """Reproduce map snapshots, decisions and clocks from acquired inputs only."""
    if results.get("status") != "complete" or not results.get("trace"):
        raise ValueError("Replay requires a completed nonempty experiment")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("Absolute replay tolerance must be finite and nonnegative")
    started = time.perf_counter()
    source_root = Path(__file__).resolve().parents[1]
    if not results.get("source_sha256"):
        raise ValueError("Replay requires frozen core source hashes")
    for filename, expected in results["source_sha256"].items():
        path = (source_root / filename).resolve()
        if not path.is_relative_to(source_root):
            raise ValueError("Source provenance must reference files inside the repository")
        require_hash(path, expected, "source")
    frames = {frame["id"]: frame for frame in results["trace"]}
    ordered_ids = [view for station in results["stations"] for view in station["view_ids"]]
    if len(frames) != len(results["trace"]) or ordered_ids != [f["id"] for f in results["trace"]]:
        raise ValueError("Station acquisition lists do not match the unique chronological trace")
    config = results["config"]
    grid = OccupancyGrid(
        bounds_xy=config["map_bounds_xy"],
        resolution_m=config["map_resolution_m"],
        footprint_radius_m=config["footprint_radius_m"],
    )
    surface = SurfaceMap(voxel_size_m=0.06)
    edges = StructuralEdgeMap(voxel_size_m=0.05, threshold=config["threshold"])
    position = np.asarray(config["start_position"], dtype=float)
    body_yaw, head_yaw, pitch = float(config["initial_body_yaw_deg"]), 0.0, 0.0
    grid.seed_robot_pose(position[:2])
    visited, attempted, trajectory = [], [], [position[:2].tolist()]
    action_seconds, distance_m, compared_arrays, checked_frames = 0.0, 0.0, 0, 0
    for station_index, station in enumerate(results["stations"]):
        if station["station"] != station_index:
            raise ValueError("Stations must be numbered chronologically from zero")
        require_equal(position, station["position"], "station body pose")
        visited.append(position[:2].copy())
        targets = scan_targets(grid, position, station_index == 0)
        if len(station["view_ids"]) > len(targets):
            raise ValueError("Station acquired more frames than its selected scan targets")
        for identifier, (_, absolute_yaw, desired_pitch) in zip(
            station["view_ids"], targets, strict=False
        ):
            frame = frames[identifier]
            desired_head = (absolute_yaw - body_yaw + 180) % 360 - 180
            if abs(desired_head) > 75:
                desired_body = (round(absolute_yaw / 90) * 90) % 360
                action_seconds += (
                    angular_distance(desired_body, body_yaw) / config["body_yaw_speed_deg_s"]
                )
                body_yaw = desired_body
                desired_head = (absolute_yaw - body_yaw + 180) % 360 - 180
            action_seconds += (
                max(
                    abs(desired_head - head_yaw) / config["head_yaw_speed_deg_s"],
                    abs(desired_pitch - pitch) / config["head_pitch_speed_deg_s"],
                )
                + 0.15
            )
            head_yaw, pitch = desired_head, desired_pitch
            request = frame["request"]
            require_equal(request["robot_base"]["position"], position, "acquired body pose")
            require_equal(
                [
                    request["robot_base"]["yaw_deg"],
                    request["head"]["yaw_deg"],
                    request["head"]["pitch_deg"],
                ],
                [body_yaw, head_yaw, pitch],
                "selected head/body angles",
            )
            require_equal(frame["action_seconds"], action_seconds, "acquisition action clock", 1e-9)
            require_equal(frame["path_length_m"], distance_m, "acquisition path length", 1e-9)
            require_hash(frame["source_rgb"], frame["source_rgb_sha256"], "acquired RGB")
            require_hash(
                frame["source_geometry"], frame["source_geometry_sha256"], "acquired depth"
            )
            with np.load(frame["prediction_path"]) as archive:
                probabilities = archive["probabilities"]
            if hashlib.sha256(probabilities.tobytes()).hexdigest() != frame["prediction_sha256"]:
                raise ValueError(f"Frozen prediction tensor differs: {identifier}")
            rgb = np.asarray(Image.open(frame["source_rgb"]).convert("RGB"))
            with np.load(frame["source_geometry"]) as archive:
                raw_depth = archive["depth"]
            depth, mask = mask_robot_depth(
                raw_depth, frame, request, reach=config["robot_arm_reach_m"]
            )
            if hashlib.sha256(mask.tobytes()).hexdigest() != frame["self_mask_sha256"]:
                raise ValueError(f"Calibrated robot self-mask differs: {identifier}")
            if int(mask.sum()) != frame["self_masked_pixels"]:
                raise ValueError("Robot self-mask count differs")
            record = camera_record(frame, (rgb.shape[1], rgb.shape[0]), config["network_size"])
            depth_small = cv2.resize(
                depth, tuple(config["network_size"]), interpolation=cv2.INTER_NEAREST
            )
            grid.integrate_depth(depth, frame)
            surface.integrate(rgb, depth, frame)
            edges.integrate(probabilities, depth_small, record, identifier)
            checked_frames += 1
        compared_arrays += compare_snapshot(station, grid, surface, edges, atol)
        frontiers = grid.frontiers(position[:2], min_cells=3, minimum_distance_m=0.65)
        serialized = [
            {
                "goal_xy": np.asarray(f.goal_xy).tolist(),
                "look_at_xy": np.asarray(f.look_at_xy).tolist(),
                "gain_cells": int(f.gain_cells),
                "distance_m": float(f.distance_m),
            }
            for f in frontiers
        ]
        if serialized != station["frontiers"]:
            raise ValueError(f"Replayed frontier candidates differ at station {station_index}")
        target = choose_frontier(frontiers, visited, attempted)
        selected = None if target is None else np.asarray(target.goal_xy).tolist()
        if selected != station["selected_goal"]:
            raise ValueError(f"Replayed frontier selection differs at station {station_index}")
        if "executed_path" in station:
            if target is None:
                raise ValueError("Recorded body motion has no selected frontier")
            movement = path_prefix(
                np.vstack([position[:2], target.path_xy]), config["step_distance_m"]
            )
            require_equal(movement, station["executed_path"], "selected body path")
            length = np.linalg.norm(np.diff(movement, axis=0), axis=1).sum()
            heading = movement[-1] - movement[0]
            desired_body = math.degrees(math.atan2(-heading[0], heading[1]))
            action_seconds += (
                angular_distance(desired_body, body_yaw) / config["body_yaw_speed_deg_s"]
            )
            action_seconds += float(length) / config["body_speed_m_s"]
            action_seconds += abs(head_yaw) / config["head_yaw_speed_deg_s"]
            body_yaw, head_yaw = desired_body, 0.0
            position[:2] = movement[-1]
            distance_m += float(length)
            trajectory.extend(movement[1:].tolist())
            grid.seed_robot_pose(position[:2])
        elif target is not None and station_index + 1 < len(results["stations"]):
            attempted.append(np.asarray(target.goal_xy))
        print(
            f"Replayed station {station_index:02d}: {checked_frames} frames; "
            "maps and decisions match",
            flush=True,
        )
    require_equal(trajectory, results["path"], "full executed trajectory")
    require_equal(action_seconds, results["action_seconds"], "final action clock", 1e-9)
    require_equal(distance_m, results["path_length_m"], "final path length", 1e-9)
    topology = extract_topology(
        grid.states(),
        grid.origin,
        grid.resolution_m,
        poses=np.asarray(trajectory),
        portal_width_m=1.6,
    )
    if topology != results["topology"]:
        raise ValueError("Replayed topology differs")
    return {
        "status": "pass",
        "frames": checked_frames,
        "stations": len(results["stations"]),
        "map_arrays_compared": compared_arrays,
        "absolute_map_tolerance": atol,
        "source_files_verified": len(results["source_sha256"]),
        "verified": [
            "RGB hashes",
            "depth hashes",
            "frozen prediction hashes",
            "robot self-mask hashes",
            "occupancy maps",
            "surface maps",
            "edge maps",
            "frontier candidates and decisions",
            "head targets",
            "executed paths",
            "simulated motion clock",
            "final topology",
        ],
        "wall_seconds": time.perf_counter() - started,
        "limitations": [
            "Reuses archived model probabilities; does not rerun network inference or rendering",
            "Same host/software numeric repeat; no cross-platform bitwise guarantee",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--atol", type=float, default=0)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError("Choose a fresh replay output to preserve validation results")
    report = replay(json.loads(args.results.read_text()), atol=args.atol)
    report["experiment_sha256"] = file_hash(args.results)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
