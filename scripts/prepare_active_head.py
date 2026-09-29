"""Prepare a furnished validation-room head-view bank with a fixed articulated body."""

import argparse
import copy
import json
from pathlib import Path


def prepare(config):
    manifest = json.loads(Path(config["bootstrap_manifest"]).read_text())
    scene = copy.deepcopy(manifest["scene_config"])
    if scene["dataset"]["split"] == "test":
        raise ValueError("Policy development must not reuse held-out test rooms")
    scene.update(name=config["capture_name"], title="Active head-view development pilot")
    scene["description"] = config["note"]
    scene["render"].update(config["render"])
    scene["cameras"] = []
    for j, pitch in enumerate(config["pitch_grid_deg"]):
        for i, yaw in enumerate(config["yaw_grid_deg"]):
            scene["cameras"].append(
                {
                    "id": f"H{j}{i}",
                    "focal_length_mm": config["focal_length_mm"],
                    "robot_base": copy.deepcopy(config["robot_base"]),
                    "head": {"yaw_deg": yaw, "pitch_deg": pitch},
                }
            )
    scene["active_head_experiment"] = copy.deepcopy(config)
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/active_head_v1.json")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/active-head-config.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    scene = prepare(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scene, indent=2) + "\n")
    print(f"{len(scene['cameras'])} head orientations: {args.output}")


if __name__ == "__main__":
    main()
