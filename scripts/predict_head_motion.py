"""Overlay frozen-model predictions on a dense display-only head-motion replay."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from run_active_head import RGBObserver, camera_record

from roomgraph.learning.training import atomic_write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("vis/active_head/motion_predictions"))
    args = parser.parse_args()
    experiment = json.loads(args.experiment.read_text())
    manifest = json.loads(args.manifest.read_text())
    if experiment.get("status") != "complete":
        raise ValueError("Freeze and complete policy evaluation before dense visualization")
    config = experiment["config"]
    checkpoint = Path(config["checkpoint"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != experiment["checkpoint_sha256"]:
        raise ValueError("The checkpoint changed since policy evaluation")
    metadata = manifest["scene_config"].get("head_motion_visualization")
    if not metadata:
        raise ValueError("Expected a visualization-only dense head-motion capture")
    if (
        metadata.get("experiment_sha256")
        != hashlib.sha256(args.experiment.read_bytes()).hexdigest()
    ):
        raise ValueError("Motion capture was generated from a different experiment")
    if not metadata.get("excluded_from_policy_and_metrics"):
        raise ValueError("Motion capture must be display-only")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Motion predictions output must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    observer = RGBObserver(
        checkpoint, config["network_size"], args.output / "frames", config["threshold"]
    )
    size = [manifest["scene_config"]["render"][k] for k in ["width", "height"]]
    frames = []
    for view in manifest["views"]:
        observation = observer.acquire(
            args.manifest.parent / (view["stem"] + "_rgb.png"), view["id"]
        )
        frames.append(
            {
                "t_s": view["visualization"]["time_s"],
                "yaw_deg": view["head"]["yaw_deg"],
                "pitch_deg": view["head"]["pitch_deg"],
                "record": camera_record(view, size, config["network_size"]),
                "rgb_path": observation["rgb_path"],
                "overlay_path": observation["overlay_path"],
            }
        )
    result = {
        "status": "complete",
        "frames": frames,
        "experiment_sha256": hashlib.sha256(args.experiment.read_bytes()).hexdigest(),
        "capture_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "note": "Display-only dense replay; extra frames do not enter policy decisions or metrics",
    }
    atomic_write_json(result, args.output / "motion.json")
    print("COMPLETE", args.output / "motion.json", len(frames), "frames")


if __name__ == "__main__":
    main()
