"""Explore from one pose using acquired RGB-D, then export a shared observed 3D map.

The renderer and evaluator own the building description. This controller receives
only requested sensor frames, calibration, and known poses; no destination list,
true floor plan, or renderer labels enters frontier selection.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from run_active_head import RGBObserver, camera_record

from roomgraph.headcam import head_camera_pose
from roomgraph.learning.training import atomic_write_json
from roomgraph.mapping import OccupancyGrid, StructuralEdgeMap, mask_robot_depth
from roomgraph.surface_map import SurfaceMap, write_points_ply
from roomgraph.topology import extract_topology


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def angular_distance(a, b):
    return abs((a - b + 180) % 360 - 180)


class FileCamera:
    """Atomic request/response boundary to the warm Isaac sensor adapter."""

    def __init__(self, capture, requests, timeout_s=180, prefix=""):
        self.capture, self.requests = Path(capture), Path(requests)
        self.timeout_s = timeout_s
        if any(not (c.isalnum() or c == "_") for c in prefix):
            raise ValueError("Request prefix must contain only letters, numbers, and underscores")
        self.prefix = prefix
        self.requests.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def acquire(self, position, body_yaw, head_yaw, pitch, focal_length_mm):
        identifier = f"{self.prefix}F{self.count:04d}"
        self.count += 1
        request = {
            "id": identifier,
            "robot_base": {"position": list(position), "yaw_deg": float(body_yaw)},
            "head": {"yaw_deg": float(head_yaw), "pitch_deg": float(pitch)},
            "focal_length_mm": focal_length_mm,
        }
        target = self.requests / f"request_{identifier}.json"
        response = self.requests / f"response_{identifier}.json"
        if target.exists() or response.exists():
            raise ValueError("Use a fresh capture/request directory for each run")
        atomic_write_json(request, target)
        started = time.monotonic()
        while not response.exists():
            if time.monotonic() - started > self.timeout_s:
                raise TimeoutError(f"Renderer did not answer {identifier}")
            time.sleep(0.05)
        result = json.loads(response.read_text())
        if result["status"] != "complete":
            raise RuntimeError(f"Renderer rejected {identifier}: {result}")
        view = result["view"]
        if view["id"] != identifier:
            raise ValueError("Received a response for a different requested frame")
        paths = {}
        for key in ("rgb", "geometry"):
            candidate = (self.capture / view[key]).resolve()
            if not candidate.is_relative_to(self.capture.resolve()):
                raise ValueError("Sensor response paths must remain inside its capture directory")
            paths[key] = candidate
        with Image.open(paths["rgb"]) as image:
            width, height = image.size
        with np.load(paths["geometry"]) as archive:
            if archive["depth"].shape != (height, width):
                raise ValueError("Depth and RGB dimensions disagree")
        focal = width * focal_length_mm / 24
        expected_k = [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]
        if not np.allclose(expected_k, view["intrinsics"], atol=1e-6):
            raise ValueError("Received intrinsics disagree with the requested camera")
        expected = head_camera_pose(position, body_yaw, head_yaw, pitch)
        if not np.allclose(expected.camera_to_world, view["camera_to_world"], atol=1e-6):
            raise ValueError(
                "Received camera calibration does not match requested articulated pose"
            )
        # Deliberately whitelist the observation; reference scene data cannot pass through.
        return {
            "id": identifier,
            "intrinsics": view["intrinsics"],
            "camera_to_world": view["camera_to_world"],
            **paths,
            "capture_seconds": view.get("capture_seconds"),
            "request": request,
        }


def choose_frontier(frontiers, visited, attempted):
    """Prefer uninspected reachable boundaries; all inputs come from the partial map."""
    choices = []
    for frontier in frontiers:
        goal = np.asarray(frontier.goal_xy)
        if frontier.distance_m < 0.45:
            continue
        if any(np.linalg.norm(goal - p) < 0.5 for p in attempted):
            continue
        # Revisiting a passage is allowed; repeatedly inspecting the same endpoint is discounted.
        revisits = sum(np.linalg.norm(goal - p) < 0.7 for p in visited)
        score = frontier.gain_cells / (1 + frontier.distance_m) / (1 + 4 * revisits)
        choices.append((score, frontier))
    return max(choices, key=lambda item: item[0])[1] if choices else None


def path_prefix(path, distance_m):
    """Execute a bounded prefix, preserving every checked intermediate waypoint."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2 or not np.isfinite(path).all():
        raise ValueError("Expected finite path points [N,2]")
    if not np.isfinite(distance_m) or distance_m < 0:
        raise ValueError("Movement distance must be finite and nonnegative")
    if len(path) < 2:
        return path.copy()
    result = [path[0]]
    remaining = distance_m
    for a, b in zip(path[:-1], path[1:], strict=True):
        length = float(np.linalg.norm(b - a))
        if length <= remaining + 1e-9:
            result.append(b)
            remaining -= length
        else:
            if remaining > 1e-8:
                result.append(a + (b - a) * remaining / length)
            break
    return np.asarray(result)


def scan_targets(grid, position, first_station):
    """Inspect unknown directions first, then cover local floor and overhead structure."""
    states = grid.states()
    yy, xx = np.indices(states.shape)
    x = grid.origin[0] + (xx + 0.5) * grid.resolution_m - position[0]
    y = grid.origin[1] + (yy + 0.5) * grid.resolution_m - position[1]
    angle = np.degrees(np.arctan2(-x, y))
    distance = np.hypot(x, y)
    targets = []
    for yaw in np.arange(0, 360, 45):
        bearing = np.abs((angle - yaw + 180) % 360 - 180)
        gain = int(((states == -1) & (distance < 5) & (bearing < 35)).sum())
        targets.append((gain, float(yaw), -25.0))
    targets.sort(key=lambda item: (-item[0], item[1]))
    # Upper views are acquired observations as well; they cannot clear unseen floor.
    upper = [(gain, yaw, 15.0) for gain, yaw, _ in targets if yaw % 90 == 0]
    return targets + (upper if first_station else upper[:2])


def save_snapshot(output, index, grid, surface, edges):
    surface_data, edge_data = surface.arrays(), edges.arrays()
    target = output / "maps" / f"station_{index:03d}.npz"
    np.savez_compressed(
        target,
        states=grid.states(),
        origin=grid.origin,
        resolution_m=grid.resolution_m,
        **{f"surface_{k}": v for k, v in surface_data.items()},
        **{f"edge_{k}": v for k, v in edge_data.items()},
    )
    return str(target.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/multiroom_v1.json")
    )
    parser.add_argument("--capture", type=Path, default=Path("vis/multiroom/capture"))
    parser.add_argument("--requests", type=Path, default=Path("vis/multiroom/requests"))
    parser.add_argument("--request-prefix", default="")
    parser.add_argument("--output", type=Path, default=Path("vis/multiroom/run_v1"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Experiment output must be fresh")
    try:
        run(args)
    except Exception as error:
        if args.output.exists():
            atomic_write_json(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                args.output / "status.json",
            )
        raise


def run(args):
    """Run a fresh online experiment; incomplete runs retain explicit failure status."""
    config = json.loads(args.config.read_text())
    source_hashes = {
        name: digest(Path(__file__).resolve().parents[1] / name)
        for name in (
            "scripts/run_multiroom.py",
            "src/roomgraph/mapping.py",
            "src/roomgraph/surface_map.py",
            "src/roomgraph/topology.py",
        )
    }
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Experiment output must be fresh")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "maps").mkdir()
    atomic_write_json({"status": "running", "config": config}, args.output / "status.json")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    observer = RGBObserver(
        Path(config["checkpoint"]),
        config["network_size"],
        args.output / "frames",
        config["threshold"],
    )
    camera = FileCamera(args.capture, args.requests, prefix=args.request_prefix)
    grid = OccupancyGrid(
        bounds_xy=config["map_bounds_xy"],
        resolution_m=config["map_resolution_m"],
        footprint_radius_m=config["footprint_radius_m"],
    )
    edge_map = StructuralEdgeMap(voxel_size_m=0.05, threshold=config["threshold"])
    surface = SurfaceMap(voxel_size_m=0.06)
    position = np.asarray(config["start_position"], dtype=float)
    body_yaw, head_yaw, pitch = float(config["initial_body_yaw_deg"]), 0.0, 0.0
    grid.seed_robot_pose(position[:2])
    trace, stations, visited, attempted = [], [], [], []
    path = [position[:2].tolist()]
    action_seconds, distance_m = 0.0, 0.0
    start = time.perf_counter()
    stop_reason = "station_budget"
    for station in range(config["max_stations"]):
        print(f"Station {station:02d}: position {position[:2].round(2).tolist()}", flush=True)
        station_started = time.perf_counter()
        visited.append(position[:2].copy())
        new_ids = []
        previous_voxels = len(surface.arrays()["points"])
        for _, absolute_yaw, desired_pitch in scan_targets(grid, position, station == 0):
            if len(trace) >= config["max_frames"]:
                stop_reason = "frame_budget"
                break
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
            observed = camera.acquire(
                position, body_yaw, head_yaw, pitch, config["focal_length_mm"]
            )
            prediction = observer.acquire(observed["rgb"], observed["id"])
            rgb = np.asarray(Image.open(observed["rgb"]).convert("RGB"))
            with np.load(observed["geometry"]) as archive:
                depth = archive["depth"].copy()
            mapping_started = time.perf_counter()
            depth, self_mask = mask_robot_depth(
                depth, observed, observed["request"], reach=config["robot_arm_reach_m"]
            )
            record = camera_record(observed, (rgb.shape[1], rgb.shape[0]), config["network_size"])
            depth_small = cv2.resize(
                depth, tuple(config["network_size"]), interpolation=cv2.INTER_NEAREST
            )
            grid.integrate_depth(depth, observed)
            surface.integrate(rgb, depth, observed)
            edge_map.integrate(prediction["probabilities"], depth_small, record, observed["id"])
            trace.append(
                {
                    "id": observed["id"],
                    "station": station,
                    "action_seconds": action_seconds,
                    "path_length_m": distance_m,
                    "request": observed["request"],
                    "intrinsics": observed["intrinsics"],
                    "camera_to_world": observed["camera_to_world"],
                    "source_rgb": str(observed["rgb"].resolve()),
                    "source_geometry": str(observed["geometry"].resolve()),
                    "source_geometry_sha256": digest(observed["geometry"]),
                    "self_masked_pixels": int(self_mask.sum()),
                    "self_mask_sha256": hashlib.sha256(self_mask.tobytes()).hexdigest(),
                    **{k: v for k, v in prediction.items() if k != "probabilities"},
                    "mapping_seconds": time.perf_counter() - mapping_started,
                    "capture_seconds": observed["capture_seconds"],
                }
            )
            new_ids.append(observed["id"])
        decision_started = time.perf_counter()
        frontiers = grid.frontiers(position[:2], min_cells=3, minimum_distance_m=0.65)
        target = choose_frontier(frontiers, visited, attempted)
        decision_seconds = time.perf_counter() - decision_started
        snapshot = save_snapshot(args.output, station, grid, surface, edge_map)
        station_data = {
            "station": station,
            "position": position.tolist(),
            "view_ids": new_ids,
            "action_seconds": action_seconds,
            "path_length_m": distance_m,
            "snapshot": snapshot,
            "frontiers": [
                {
                    "goal_xy": np.asarray(f.goal_xy).tolist(),
                    "look_at_xy": np.asarray(f.look_at_xy).tolist(),
                    "gain_cells": int(f.gain_cells),
                    "distance_m": float(f.distance_m),
                }
                for f in frontiers
            ],
            "surface_voxels": len(surface.arrays()["points"]),
            "new_surface_voxels": len(surface.arrays()["points"]) - previous_voxels,
            "planning_seconds": decision_seconds,
            "station_wall_seconds": time.perf_counter() - station_started,
            "selected_goal": None if target is None else np.asarray(target.goal_xy).tolist(),
        }
        stations.append(station_data)
        atomic_write_json(
            {
                "status": "running",
                "config": config,
                "trace": trace,
                "stations": stations,
                "path": path,
            },
            args.output / "progress.json",
        )
        print(
            f"  {len(trace)} views, {station_data['surface_voxels']} surface voxels, "
            f"{len(frontiers)} frontiers, next={station_data['selected_goal']}",
            flush=True,
        )
        if len(trace) >= config["max_frames"]:
            break
        if station + 1 == config["max_stations"]:
            stop_reason = "station_budget"
            break
        if target is None:
            stop_reason = "no_reachable_uninspected_frontier"
            break
        checked_path = np.vstack([position[:2], target.path_xy])
        movement = path_prefix(checked_path, config["step_distance_m"])
        if len(movement) < 2 or np.linalg.norm(movement[-1] - position[:2]) < 0.2:
            attempted.append(np.asarray(target.goal_xy))
            continue
        length = np.linalg.norm(np.diff(movement, axis=0), axis=1).sum()
        heading = movement[-1] - movement[0]
        desired_body = math.degrees(math.atan2(-heading[0], heading[1]))
        action_seconds += angular_distance(desired_body, body_yaw) / config["body_yaw_speed_deg_s"]
        action_seconds += float(length) / config["body_speed_m_s"]
        action_seconds += abs(head_yaw) / config["head_yaw_speed_deg_s"]
        body_yaw, head_yaw = desired_body, 0.0
        position[:2] = movement[-1]
        distance_m += float(length)
        path.extend(movement[1:].tolist())
        station_data["executed_path"] = movement.tolist()
        grid.seed_robot_pose(position[:2])
    surfaces, edges = surface.arrays(), edge_map.arrays()
    write_points_ply(args.output / "observed_surfaces.ply", surfaces["points"], surfaces["colors"])
    write_points_ply(args.output / "predicted_edges.ply", edges["points"])
    topology = extract_topology(
        grid.states(), grid.origin, grid.resolution_m, poses=np.asarray(path), portal_width_m=1.6
    )
    atomic_write_json(topology, args.output / "topology.json")
    result = {
        "status": "complete",
        "stop_reason": stop_reason,
        "config": config,
        "checkpoint_sha256": digest(config["checkpoint"]),
        "source_sha256": source_hashes,
        "wall_seconds": time.perf_counter() - start,
        "action_seconds": action_seconds,
        "path_length_m": distance_m,
        "trace": trace,
        "stations": stations,
        "path": path,
        "topology": topology,
        "limitations": [
            "ideal depth sensor and known poses; not RGB-only reconstruction or SLAM",
            "observed point-cloud geometry; unobserved surfaces remain absent",
            "heuristic 2.5D navigation; kinematic proxy, not humanoid dynamics",
            "stopping is budget/frontier exhaustion, never proof of whole-building completion",
        ],
    }
    atomic_write_json(result, args.output / "experiment.json")
    atomic_write_json(
        {"status": "complete", "stop_reason": stop_reason}, args.output / "status.json"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in {"status", "stop_reason", "wall_seconds", "action_seconds", "path_length_m"}
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
