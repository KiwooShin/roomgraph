"""Freeze and run an offline real RGB-D pilot, then replay controlled sensor errors."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from run_active_head import RGBObserver

from roomgraph.mapping import StructuralEdgeMap, depth_points
from roomgraph.real_rgbd import (
    ARKitTrajectory,
    geometry_metrics,
    perturb,
    read_intrinsics,
    scaled_intrinsics,
    timestamp,
    upright_record,
    voxel_fuse,
)
from roomgraph.surface_map import write_points_ply

SOURCE_FILES = (
    "scripts/run_arkitscenes.py",
    "scripts/run_active_head.py",
    "src/roomgraph/real_rgbd.py",
    "src/roomgraph/mapping.py",
    "src/roomgraph/active_view.py",
    "src/roomgraph/surface_map.py",
    "src/roomgraph/learning/model.py",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def frame_record(rgb_path, data, trajectory):
    intrinsic_path = data / "lowres_wide_intrinsics" / rgb_path.with_suffix(".pincam").name
    intrinsic, size = read_intrinsics(intrinsic_path)
    time_s = timestamp(rgb_path)
    pose = trajectory.pose(time_s)
    return {
        "id": rgb_path.stem,
        "timestamp_s": time_s,
        "size": list(size),
        "intrinsics": intrinsic.tolist(),
        "camera_to_world": pose.tolist(),
        "rgb": str(rgb_path),
        "depth": str(data / "lowres_depth" / rgb_path.name),
        "confidence": str(data / "confidence" / rgb_path.name),
        "intrinsic_file": str(intrinsic_path),
    }


def prepare(data, output, config, config_path):
    trajectory = ARKitTrajectory(np.loadtxt(data / "lowres_wide.traj"))
    references = sorted((data / "highres_depth").glob("*.png"), key=timestamp)
    reference_times = np.array([timestamp(p) for p in references])
    candidates = []
    for path in sorted((data / "lowres_wide").glob("*.png"), key=timestamp):
        t = timestamp(path)
        if not trajectory.times[0] <= t <= trajectory.times[-1] - 0.05:
            continue
        if (
            len(reference_times)
            and np.min(np.abs(reference_times - t)) <= config["reference_exclusion_s"]
        ):
            continue
        if not all(
            (data / folder / path.name).exists() for folder in ("lowres_depth", "confidence")
        ):
            continue
        candidates.append(frame_record(path, data, trajectory))
    if len(candidates) < config["frames"] or not references:
        raise ValueError("Insufficient distinct input frames or evaluator reference")
    indices = np.linspace(0, len(candidates) - 1, config["frames"]).round().astype(int)
    records = [candidates[i] for i in indices]
    sources = {p: digest(p) for p in SOURCE_FILES}
    source_data = {
        str(data / p): digest(data / p)
        for p in ("download_receipt.json", "lowres_wide.traj", "metadata.csv")
    }
    with (data / "metadata.csv").open() as stream:
        metadata = next(row for row in csv.DictReader(stream) if row["video_id"] == config["video"])
    turns = {"Up": 0, "Left": 1, "Down": 2, "Right": 3}[metadata["sky_direction"]]
    for record in records:
        for key in ("rgb", "depth", "confidence", "intrinsic_file"):
            source_data[record[key]] = digest(record[key])
    manifest = {
        "config": config,
        "dataset_metadata": metadata,
        "clockwise_quarter_turns": turns if config.get("upright") else 0,
        "config_sha256": digest(config_path),
        "checkpoint_sha256": digest(config["checkpoint"]),
        "source_sha256": sources,
        "input_sha256": source_data,
        "frames": records,
        "eligible_input_frames": len(candidates),
        "reference_files": [str(p) for p in references],
        "reference_sha256": {str(p): digest(p) for p in references},
        "preprocessing": (
            "Metadata-directed upright rotation of RGB/depth/confidence and camera basis. "
            "Existing RGBObserver anisotropic resize; scale K rows separately. "
            "Registered lowres RGB/depth; no proxy self-mask, no floor clipping."
        ),
    }
    write_json(output / "manifest.json", manifest)
    return manifest, trajectory


def build_reference(manifest, data, trajectory):
    """Only called after prediction caching; projected FARO depth never enters inference/fusion."""
    intrinsic_paths = sorted((data / "lowres_wide_intrinsics").glob("*.pincam"), key=timestamp)
    intrinsic_times = np.array([timestamp(p) for p in intrinsic_paths])
    points, checks, used = [], [], []
    for filename in manifest["reference_files"]:
        path = Path(filename)
        t = timestamp(path)
        if not trajectory.times[0] <= t <= trajectory.times[-1]:
            continue
        index = int(np.argmin(np.abs(intrinsic_times - t)))
        if abs(intrinsic_times[index] - t) > 0.02:
            continue
        intrinsic, size = read_intrinsics(intrinsic_paths[index])
        reference = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        k = scaled_intrinsics(intrinsic, size, (reference.shape[1], reference.shape[0]))
        record = {"intrinsics": k.tolist(), "camera_to_world": trajectory.pose(t).tolist()}
        reference[(reference < 0.15) | (reference > 8)] = np.nan
        cloud, _ = depth_points(reference, record, stride=manifest["config"]["reference_stride"])
        points.append(cloud)
        used.append(
            {
                "depth": filename,
                "intrinsics": str(intrinsic_paths[index]),
                "intrinsics_sha256": digest(intrinsic_paths[index]),
                "timestamp_s": t,
            }
        )
        sensor_path = data / "lowres_depth" / path.name
        if sensor_path.exists():
            sensor = cv2.imread(str(sensor_path), cv2.IMREAD_UNCHANGED).astype(float) / 1000
            small = cv2.resize(
                reference, (sensor.shape[1], sensor.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            valid = np.isfinite(small) & (sensor > 0.15) & (sensor < 8)
            if valid.any():
                checks.append(float(np.median(np.abs(sensor[valid] - small[valid]))))
    cloud = voxel_fuse(points, voxel_size=manifest["config"]["voxel_size_m"])["points"]
    return cloud, {
        "frames": used,
        "median_frame_sensor_reference_depth_error_m": float(np.median(checks)) if checks else None,
    }


def cache_predictions(manifest, output):
    config = manifest["config"]
    observer = RGBObserver(
        Path(config["checkpoint"]),
        config["network_size"],
        output / "predictions",
        config["threshold"],
    )
    frames, receipt = [], []
    turns = manifest["clockwise_quarter_turns"]
    (output / "upright_rgb").mkdir()
    for index, record in enumerate(manifest["frames"]):
        depth = cv2.imread(record["depth"], cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        confidence = cv2.imread(record["confidence"], cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(cv2.imread(record["rgb"]), cv2.COLOR_BGR2RGB)
        if depth.shape != confidence.shape or rgb.shape != (*depth.shape, 3):
            raise ValueError("RGB/depth/confidence grids do not match")
        if list(depth.shape[::-1]) != record["size"]:
            raise ValueError("Intrinsics image dimensions differ")
        rgb, depth, confidence = [np.rot90(a, -turns).copy() for a in (rgb, depth, confidence)]
        oriented = upright_record(record, record["size"], turns)
        image_path = output / "upright_rgb" / f"{record['id']}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        observed = observer.acquire(image_path, record["id"])
        depth[(depth < 0.15) | (depth > 8)] = np.nan
        frames.append(
            {
                "record": oriented,
                "rgb": rgb,
                "depth": depth,
                "confidence": confidence,
                "probabilities": observed["probabilities"],
            }
        )
        receipt.append({k: v for k, v in observed.items() if k != "probabilities"})
        if index % 40 == 0:
            print(f"Cached prediction {index + 1}/{len(manifest['frames'])}", flush=True)
    write_json(output / "prediction_receipt.json", receipt)
    return frames, observer.timings


def reconstruct(frames, config, specification, seed, filtered, trajectory):
    point_sets, color_sets = [], []
    edges = StructuralEdgeMap(config["voxel_size_m"], config["threshold"])
    for index, frame in enumerate(frames):
        original = frame["record"]
        nominal = dict(original)
        if specification.get("time_offset_s"):
            shifted = trajectory.pose(original["timestamp_s"] + specification["time_offset_s"])
            shifted[:3, :3] = shifted[:3, :3] @ np.asarray(original["raw_from_upright"])
            nominal["camera_to_world"] = shifted.tolist()
        record = perturb(nominal, specification, index, len(frames), seed)
        depth = frame["depth"].copy() * (1 + specification.get("depth_scale_fraction", 0))
        if filtered:
            depth[frame["confidence"] < 2] = np.nan
        points, pixels = depth_points(depth, record, stride=config["surface_stride"])
        point_sets.append(points)
        color_sets.append(frame["rgb"][pixels[:, 1], pixels[:, 0]])
        small = cv2.resize(depth, tuple(config["network_size"]), interpolation=cv2.INTER_NEAREST)
        edge_record = {
            **record,
            "intrinsics": scaled_intrinsics(
                record["intrinsics"], original["size"], config["network_size"]
            ).tolist(),
        }
        edges.integrate(frame["probabilities"], small, edge_record, original["id"], max_depth_m=8)
    surface = voxel_fuse(
        point_sets, color_sets, config["voxel_size_m"], min_views=2 if filtered else 1
    )
    edge_data = edges.arrays()
    if filtered:
        keep = (edge_data["support"] >= 2) & (edge_data["camera_support"] >= 2)
        edge_data = {key: value[keep] for key, value in edge_data.items()}
    return surface, edge_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/arkitscenes_pilot.json")
    )
    parser.add_argument(
        "--data", type=Path, default=Path("artifacts/real/arkitscenes/Validation/42445021")
    )
    parser.add_argument("--output", type=Path, default=Path("vis/arkitscenes/pilot_v2"))
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    manifest, trajectory = prepare(args.data, args.output, config, args.config)
    frames, timings = cache_predictions(manifest, args.output)
    inference_stage = time.perf_counter() - started
    print("Predictions frozen; constructing evaluator-only reference", flush=True)
    reference, reference_info = build_reference(manifest, args.data, trajectory)
    np.savez_compressed(args.output / "reference.npz", points=reference)
    write_json(args.output / "reference_info.json", reference_info)
    results = []
    for name, specification in config["variants"].items():
        seeds = config["seeds"] if "jitter" in name else config["seeds"][:1]
        for seed in seeds:
            for filtered in (False, True):
                label = "confidence_multiview" if filtered else "all_valid"
                print(f"Replaying {name} / {seed} / {label}", flush=True)
                tick = time.perf_counter()
                surface, edges = reconstruct(
                    frames, config, specification, seed, filtered, trajectory
                )
                runtime = time.perf_counter() - tick
                metrics = geometry_metrics(surface["points"], reference)
                identifier = f"{name}_{seed}_{label}"
                np.savez_compressed(
                    args.output / f"{identifier}.npz",
                    **surface,
                    **{f"edge_{k}": v for k, v in edges.items()},
                )
                if name == "nominal":
                    write_points_ply(
                        args.output / f"{identifier}.ply", surface["points"], surface["colors"]
                    )
                    write_points_ply(args.output / f"{identifier}_edges.ply", edges["points"])
                results.append(
                    {
                        "id": identifier,
                        "variant": name,
                        "seed": seed,
                        "fusion": label,
                        "surface_metrics": metrics,
                        "edge_voxels": len(edges["points"]),
                        "fusion_seconds": runtime,
                    }
                )
                write_json(args.output / "progress.json", results)
    frozen = {
        **manifest["source_sha256"],
        **manifest["input_sha256"],
        **manifest["reference_sha256"],
        str(args.config): manifest["config_sha256"],
        config["checkpoint"]: manifest["checkpoint_sha256"],
    }
    for path, expected in frozen.items():
        if digest(path) != expected:
            raise ValueError(f"Source changed during experiment: {path}")
    result = {
        "status": "complete",
        "video": config["video"],
        "frames": len(frames),
        "manifest_sha256": digest(args.output / "manifest.json"),
        "prediction_receipt_sha256": digest(args.output / "prediction_receipt.json"),
        "reference_info_sha256": digest(args.output / "reference_info.json"),
        "wall_seconds": time.perf_counter() - started,
        "prepare_and_inference_seconds": inference_stage,
        "median_inference_seconds": float(np.median(timings)),
        "results": results,
        "limitations": [
            "One development capture; no real-data retraining or threshold selection.",
            "Nominal ARKit poses are estimates, not independently measured ground truth.",
            "Reference is the union of held-out FARO-derived depth views, lifted using "
            "nominal ARKit calibration/poses. "
            "Not a separately registered full-scene laser benchmark.",
            "Surface metrics are not architectural-edge accuracy; "
            "no typed real edge labels are available.",
            "Confidence and multi-view filtering do not correct systematic "
            "pose/calibration errors.",
            "Offline replay; no active navigation, humanoid self-mask or completed room shell.",
        ],
    }
    write_json(args.output / "summary.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("results", "limitations")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
