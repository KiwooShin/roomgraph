"""Verify a frozen real pilot and reproduce selected maps from cached predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from run_arkitscenes import digest, reconstruct, write_json

from roomgraph.real_rgbd import ARKitTrajectory, upright_record


def verify_manifest(root):
    summary = json.loads((root / "summary.json").read_text())
    if summary["status"] != "complete" or summary["manifest_sha256"] != digest(
        root / "manifest.json"
    ):
        raise ValueError("Incomplete run or changed manifest")
    manifest = json.loads((root / "manifest.json").read_text())
    for group in ("source_sha256", "input_sha256", "reference_sha256"):
        for path, expected in manifest[group].items():
            if digest(path) != expected:
                raise ValueError(f"Changed frozen input: {path}")
    if digest(manifest["config"]["checkpoint"]) != manifest["checkpoint_sha256"]:
        raise ValueError("Changed model checkpoint")
    for name in ("prediction_receipt", "reference_info"):
        if digest(root / f"{name}.json") != summary[f"{name}_sha256"]:
            raise ValueError(f"Changed {name}")
    reference = json.loads((root / "reference_info.json").read_text())
    for frame in reference["frames"]:
        if digest(frame["intrinsics"]) != frame["intrinsics_sha256"]:
            raise ValueError("Changed evaluator intrinsics")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("vis/arkitscenes/pilot_v2"))
    args = parser.parse_args()
    manifest = verify_manifest(args.run)
    receipts = json.loads((args.run / "prediction_receipt.json").read_text())
    turns = manifest["clockwise_quarter_turns"]
    frames = []
    for record, receipt in zip(manifest["frames"], receipts, strict=True):
        with np.load(receipt["prediction_path"]) as archive:
            probabilities = archive["probabilities"]
        if hashlib.sha256(probabilities.tobytes()).hexdigest() != receipt["prediction_sha256"]:
            raise ValueError("Changed cached probabilities")
        if digest(args.run / "upright_rgb" / f"{record['id']}.png") != receipt["source_rgb_sha256"]:
            raise ValueError("Changed model input image")
        rgb = cv2.cvtColor(cv2.imread(record["rgb"]), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(record["depth"], cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        confidence = cv2.imread(record["confidence"], cv2.IMREAD_UNCHANGED)
        rgb, depth, confidence = [np.rot90(x, -turns).copy() for x in (rgb, depth, confidence)]
        depth[(depth < 0.15) | (depth > 8)] = np.nan
        oriented = upright_record(record, record["size"], turns)
        frames.append(
            {
                "record": oriented,
                "rgb": rgb,
                "depth": depth,
                "confidence": confidence,
                "probabilities": probabilities,
            }
        )
    trajectory_path = next(p for p in manifest["input_sha256"] if p.endswith(".traj"))
    trajectory = ARKitTrajectory(np.loadtxt(trajectory_path))
    checked, arrays = [], 0
    for name, filtered in (("nominal", False), ("nominal", True), ("jitter_5cm_3deg", True)):
        label = "confidence_multiview" if filtered else "all_valid"
        identifier = f"{name}_11_{label}"
        surface, edges = reconstruct(
            frames,
            manifest["config"],
            manifest["config"]["variants"][name],
            11,
            filtered,
            trajectory,
        )
        with np.load(args.run / f"{identifier}.npz") as archive:
            for key, value in {**surface, **{f"edge_{k}": v for k, v in edges.items()}}.items():
                np.testing.assert_array_equal(value, archive[key], err_msg=f"{identifier}/{key}")
                arrays += 1
        checked.append(identifier)
    write_json(
        args.run / "replay_check.json",
        {
            "status": "pass",
            "manifest_sha256": digest(args.run / "manifest.json"),
            "summary_sha256": digest(args.run / "summary.json"),
            "maps": checked,
            "arrays_compared": arrays,
            "tolerance": 0,
            "scope": (
                "Archived probabilities and nominal plus one severe-jitter seed; "
                "no network rerun or reference-based pose alignment."
            ),
        },
    )
    print(f"Exact replay passed: {len(checked)} maps, {arrays} arrays")


if __name__ == "__main__":
    main()
