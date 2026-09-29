"""Causal active-head replay: acquire RGB after selection; score geometry separately."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from roomgraph.active_view import ActiveHeadPlanner, HeadMotionLimits, HeadView
from roomgraph.geometry import project
from roomgraph.headcam import head_camera_pose
from roomgraph.learning.model import EdgeNet
from roomgraph.learning.reconstruction import fit_room, reconstruction_metrics, write_room_obj
from roomgraph.learning.training import atomic_write_json


def camera_record(view, source_size, network_size, identifier=None):
    intrinsic = np.asarray(view["intrinsics"], dtype=float).copy()
    intrinsic[0] *= network_size[0] / source_size[0]
    intrinsic[1] *= network_size[1] / source_size[1]
    return {
        "id": identifier or view["id"],
        "intrinsics": intrinsic.tolist(),
        "camera_to_world": view["camera_to_world"],
    }


class RGBObserver:
    """Only selected RGB paths enter the frozen model; caching avoids repeated inference."""

    def __init__(self, checkpoint, network_size, output, threshold):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EdgeNet(pretrained=False).to(self.device, memory_format=torch.channels_last)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.network_size = tuple(network_size)
        self.output = output
        self.threshold = threshold
        self.cache = {}
        self.timings = []
        output.mkdir(parents=True, exist_ok=True)

    def acquire(self, rgb_path, identifier):
        key = str(rgb_path.resolve())
        if key in self.cache:
            return self.cache[key]
        image = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.network_size, interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())[None].to(self.device)
        tensor = tensor.float().div_(255).contiguous(memory_format=torch.channels_last)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with (
            torch.inference_mode(),
            torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
            ),
        ):
            probabilities = self.model(tensor).float().sigmoid().cpu().numpy()[0]
        duration = time.perf_counter() - started
        self.timings.append(duration)
        overlay = image.copy()
        visible = probabilities[0] >= self.threshold
        overlay[visible] = [45, 255, 189]
        rgb_output = self.output / f"{identifier}_rgb.png"
        overlay_output = self.output / f"{identifier}_prediction.png"
        Image.fromarray(image).save(rgb_output)
        Image.fromarray(overlay).save(overlay_output)
        prediction_path = self.output / f"{identifier}_probabilities.npz"
        np.savez_compressed(prediction_path, probabilities=probabilities)
        result = {
            "probabilities": probabilities,
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": hashlib.sha256(probabilities.tobytes()).hexdigest(),
            "source_rgb_sha256": hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
            "inference_seconds": duration,
            "rgb_path": str(rgb_output.resolve()),
            "overlay_path": str(overlay_output.resolve()),
        }
        self.cache[key] = result
        return result


class VisibilityEvaluator:
    """Evaluator-only ground truth: never passed to the planner or reconstruction fitter."""

    def __init__(self, edges):
        points, weights = [], []
        for edge in edges:
            start, end = np.asarray(edge["start"]), np.asarray(edge["end"])
            length = np.linalg.norm(end - start)
            count = max(2, int(np.ceil(length / 0.025)))
            alpha = (np.arange(count) + 0.5) / count
            points.extend(start + alpha[:, None] * (end - start))
            weights.extend([length / count] * count)
        self.points, self.weights = np.asarray(points), np.asarray(weights)
        self.seen = np.zeros(len(points), bool)
        self.multiview = np.zeros(len(points), bool)
        self.centers = []
        self.masks = []

    def observe(self, view, geometry_path):
        with np.load(geometry_path) as archive:
            depth = archive["depth"]
        pose = np.asarray(view["camera_to_world"])
        pixels, z = project(self.points, np.asarray(view["intrinsics"]), pose)
        height, width = depth.shape
        valid = (
            (z > 0.05)
            & np.isfinite(pixels).all(axis=1)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        ids = np.flatnonzero(valid)
        xy = pixels[valid].astype(int)
        measured = depth[xy[:, 1], xy[:, 0]]
        visible = np.zeros(len(self.points), bool)
        visible[ids] = np.isfinite(measured) & (measured > 0) & (z[valid] <= measured + 0.025)
        eye = pose[:3, 3]
        for center, mask in zip(self.centers, self.masks, strict=True):
            if np.linalg.norm(eye - center) >= 0.2:
                self.multiview |= visible & mask
        self.seen |= visible
        self.centers.append(eye)
        self.masks.append(visible)
        return self.scores()

    def scores(self):
        denominator = self.weights.sum()
        return {
            "visible_edge_coverage": float(self.weights[self.seen].sum() / denominator),
            "multi_view_edge_coverage": float(self.weights[self.multiview].sum() / denominator),
        }


def choose_baseline(name, candidates, acquired, current, planner, remaining, random):
    feasible = [
        c
        for c in candidates
        if c.view_id not in acquired and planner.action_time(current, c) <= remaining + 1e-9
    ]
    if not feasible:
        return None
    if name == "raster":
        # Fixed serpentine scan: increasing pitch, alternating yaw direction.
        order = sorted(candidates, key=lambda c: (c.pitch_deg, c.yaw_deg))
        pitches = sorted({c.pitch_deg for c in order})
        rank = {}
        for row, pitch in enumerate(pitches):
            items = sorted(
                [c for c in order if c.pitch_deg == pitch],
                key=lambda c: c.yaw_deg,
                reverse=row % 2 == 1,
            )
            for candidate in items:
                rank[candidate.view_id] = len(rank)
        target = min(feasible, key=lambda c: rank[c.view_id])
    else:
        target = feasible[int(random.integers(len(feasible)))]
    return {
        "view_id": target.view_id,
        "yaw_deg": target.yaw_deg,
        "pitch_deg": target.pitch_deg,
        "duration_s": planner.action_time(current, target),
        "score": None,
    }


def validate_capture(config, bootstrap, capture):
    """Reject a stale head bank; metadata checks never enter policy scoring."""
    for manifest in (bootstrap, capture):
        if manifest["scene_config"]["dataset"]["split"] == "test":
            raise ValueError("Use development/validation captures, not held-out test rooms")
    if bootstrap["structural_edges"] != capture["structural_edges"]:
        raise ValueError("Bootstrap and head bank must use the same room geometry")
    source, bank = bootstrap["scene_config"], capture["scene_config"]
    for key in ("room", "furnishings", "asset_overrides", "lighting", "headcam", "seed", "dataset"):
        if source[key] != bank[key]:
            raise ValueError(f"Head bank differs from bootstrap scene: {key}")
    for key, value in config["render"].items():
        if bank["render"][key] != value:
            raise ValueError(f"Head bank render configuration differs: {key}")
    expected = {
        (float(y), float(p)) for y in config["yaw_grid_deg"] for p in config["pitch_grid_deg"]
    }
    actual = {(v["head"]["yaw_deg"], v["head"]["pitch_deg"]) for v in capture["views"]}
    if expected != actual or len(actual) != len(capture["views"]):
        raise ValueError("Head bank does not match the requested orientation grid")
    width, height = config["render"]["width"], config["render"]["height"]
    focal = width * config["focal_length_mm"] / 24
    intrinsic = [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]
    for view in capture["views"]:
        if view["robot_base"] != config["robot_base"]:
            raise ValueError("Head bank robot base differs from the requested station")
        base, head = view["robot_base"], view["head"]
        pose = head_camera_pose(
            base["position"], base["yaw_deg"], head["yaw_deg"], head["pitch_deg"]
        )
        if not np.allclose(pose.camera_to_world, view["camera_to_world"], atol=1e-7):
            raise ValueError("Head bank camera pose disagrees with its rig metadata")
        if not np.allclose(intrinsic, view["intrinsics"], atol=1e-7):
            raise ValueError("Head bank intrinsics disagree with the requested camera")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/active_head_v1.json")
    )
    parser.add_argument("--capture", type=Path, default=Path("vis/active_head/capture"))
    parser.add_argument("--output", type=Path, default=Path("vis/active_head/run_v1"))
    args = parser.parse_args()
    started = time.perf_counter()
    config = json.loads(args.config.read_text())
    bootstrap_path = Path(config["bootstrap_manifest"])
    bootstrap = json.loads(bootstrap_path.read_text())
    capture_path = args.capture / config["capture_name"]
    capture = json.loads((capture_path / "manifest.json").read_text())
    validate_capture(config, bootstrap, capture)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Experiment output must be empty; choose a fresh --output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json({"status": "running", "config": config}, args.output / "status.json")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    observer = RGBObserver(
        Path(config["checkpoint"]),
        config["network_size"],
        args.output / "frames",
        config["threshold"],
    )
    boot_views, boot_records, boot_predictions = [], [], []
    source_size = [bootstrap["scene_config"]["render"][k] for k in ["width", "height"]]
    for view_id in config["bootstrap_views"]:
        view = next(v for v in bootstrap["views"] if v["id"] == view_id)
        record = camera_record(view, source_size, config["network_size"], "B_" + view_id)
        result = observer.acquire(bootstrap_path.parent / (view["stem"] + "_rgb.png"), record["id"])
        boot_views.append(view)
        boot_records.append(record)
        boot_predictions.append(result["probabilities"])
    initial = fit_room(np.stack(boot_predictions), boot_records, config["threshold"], maxiter=65)
    # Read truth solely after inference, to score the hypothesis and later selected observations.
    width, depth, height = bootstrap["scene_config"]["room"]["dimensions_m"]
    truth = [-width / 2, width / 2, -depth / 2, depth / 2, 0, height]
    source_size = [capture["scene_config"]["render"][k] for k in ["width", "height"]]
    lookup, records, candidates = {}, {}, []
    for view in capture["views"]:
        identifier = view["id"]
        record = camera_record(view, source_size, config["network_size"])
        records[identifier] = record
        lookup[identifier] = view
        candidates.append(
            HeadView.from_record(
                identifier,
                view["head"]["yaw_deg"],
                view["head"]["pitch_deg"],
                record,
                tuple(reversed(config["network_size"])),
            )
        )
    center = next(c for c in candidates if c.yaw_deg == 0 and c.pitch_deg == 0)
    limits = HeadMotionLimits(**config["motion_limits"])
    policies = []
    choices = [("active", None), ("raster", None)]
    choices.extend((f"random_{seed}", seed) for seed in config["random_seeds"])
    for name, seed in choices:
        planner = ActiveHeadPlanner(candidates, initial["bounds_m"], limits, config["threshold"])
        evaluator = VisibilityEvaluator(capture["structural_edges"])
        predictions, acquired_records = list(boot_predictions), list(boot_records)
        for probability, record, view in zip(
            boot_predictions, boot_records, boot_views, strict=True
        ):
            planner.observe(probability, record)
            evaluator.observe(view, bootstrap_path.parent / (view["stem"] + "_geometry.npz"))
        trace, acquired = [], set()
        current, elapsed = (0.0, 0.0), 0.0
        random = np.random.default_rng(seed)
        action = {
            "view_id": center.view_id,
            "yaw_deg": 0,
            "pitch_deg": 0,
            "duration_s": 0,
        }
        while action is not None:
            identifier = action["view_id"]
            elapsed += action["duration_s"]
            current = (action["yaw_deg"], action["pitch_deg"])
            # Selection has finished. Only now reveal this candidate's RGB/prediction.
            view, record = lookup[identifier], records[identifier]
            observation = observer.acquire(capture_path / (view["stem"] + "_rgb.png"), identifier)
            planner.observe(observation["probabilities"], record)
            predictions.append(observation["probabilities"])
            acquired_records.append(record)
            acquired.add(identifier)
            # Ground-truth scoring is isolated from all subsequent policy inputs.
            evaluation = evaluator.observe(view, capture_path / (view["stem"] + "_geometry.npz"))
            remaining = min(config["planning_horizon_s"], config["action_budget_s"] - elapsed)
            tick = time.perf_counter()
            if name == "active":
                preview = planner.plan(current, max(0, remaining))
                following = preview[0] if preview else None
            else:
                following = choose_baseline(
                    name, candidates, acquired, current, planner, max(0, remaining), random
                )
                preview = [following] if following else []
            decision_ms = 1000 * (time.perf_counter() - tick)
            trace.append(
                {
                    "t_s": elapsed,
                    "view_id": identifier,
                    "yaw_deg": current[0],
                    "pitch_deg": current[1],
                    "decision_ms": decision_ms,
                    "diagnostics": planner.diagnostics(),
                    "evaluation": evaluation,
                    "preview_plan": preview,
                    "shell_state": planner.shell_state(),
                    "directional_state": planner.directional_state(),
                    "record": record,
                    "rgb_path": observation["rgb_path"],
                    "overlay_path": observation["overlay_path"],
                    "prediction_path": observation["prediction_path"],
                    "prediction_sha256": observation["prediction_sha256"],
                }
            )
            action = following
        reconstruction_started = time.perf_counter()
        final = fit_room(np.stack(predictions), acquired_records, config["threshold"], maxiter=65)
        reconstruction_seconds = time.perf_counter() - reconstruction_started
        # Coverage is held constant between acquired frames and through any remaining budget.
        times = np.r_[[row["t_s"] for row in trace], config["action_budget_s"]]
        coverage = np.array([row["evaluation"]["visible_edge_coverage"] for row in trace])
        auc = float(np.sum(np.diff(times) * coverage) / config["action_budget_s"])
        policy = {
            "name": name,
            "seed": seed,
            "trace": trace,
            "final_reconstruction": final,
            "final_metrics": reconstruction_metrics(final["bounds_m"], truth),
            "acquired_head_ids": [row["view_id"] for row in trace],
            "simulated_seconds": elapsed,
            "reconstruction_seconds": reconstruction_seconds,
            "coverage_auc": auc,
        }
        policies.append(policy)
        write_room_obj(final["bounds_m"], args.output / f"{name}_room.obj")
        print(
            json.dumps(
                {
                    "policy": name,
                    "views": len(trace),
                    "coverage_auc": auc,
                    **trace[-1]["evaluation"],
                    **policy["final_metrics"],
                }
            ),
            flush=True,
        )
    report = {
        "status": "complete",
        "config": config,
        "bootstrap_manifest_sha256": hashlib.sha256(bootstrap_path.read_bytes()).hexdigest(),
        "capture_manifest_sha256": hashlib.sha256(
            (capture_path / "manifest.json").read_bytes()
        ).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(Path(config["checkpoint"]).read_bytes()).hexdigest(),
        "bootstrap_camera_ids": config["bootstrap_views"],
        "initial_reconstruction": initial,
        "initial_metrics": reconstruction_metrics(initial["bounds_m"], truth),
        "truth_bounds_m": truth,
        "capture_top_down": {
            "path": str((capture_path / "top_down.png").resolve()),
            "extent_m": capture["top_down"]["horizontal_extent_m"],
        },
        "policies": policies,
        "inference_median_ms_excluding_first": float(np.median(observer.timings[1:]) * 1000),
        "wall_seconds": time.perf_counter() - started,
        "observation_cache": {
            path: {k: v for k, v in observation.items() if k != "probabilities"}
            for path, observation in observer.cache.items()
        },
        "protocol": {
            "inference": "BF16; deterministic PyTorch algorithms and fixed cuDNN kernels",
            "replay": "Selected probability tensors archived losslessly with raw-array SHA256",
            "selection_inputs": "acquired RGB predictions, calibration, fitted shell hypothesis",
            "ground_truth": "evaluator only; never supplied to planner or fitter",
            "time_budget": "simulated head motion plus settling; excludes compute time",
            "bootstrap": "same three translated RGBs and initial head view for every policy",
            "mapping": "coverage/support updates every acquired frame; geometry refit at the end",
            "support": "uncalibrated visible/typed agreement; not verified physical geometry",
            "evaluation": "one development room; no held-out test or real robot claim",
        },
    }
    atomic_write_json(report, args.output / "experiment.json")
    atomic_write_json({"status": "complete"}, args.output / "status.json")
    print("COMPLETE", args.output / "experiment.json", flush=True)


if __name__ == "__main__":
    main()
