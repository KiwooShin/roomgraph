"""Score independently reviewed visible edges, refusing unreviewed annotations."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

from roomgraph.learning.metrics import boundary_counts, summarize_counts, thin_prediction
from roomgraph.real_edges import rasterize_annotation


def score(run, annotations, tolerance=2):
    summary = json.loads((run / "summary.json").read_text())
    records = {r["id"]: r for r in summary["records"]}
    frames = annotations["frames"]
    if not frames or len({f["id"] for f in frames}) != len(frames):
        raise ValueError("Empty or duplicate annotation frames")
    for frame in frames:
        if (
            frame["status"] != "human_reviewed"
            or not frame.get("reviewer")
            or not frame["review_regions"]
        ):
            raise ValueError(f"Needs independent human review: {frame['id']}")
        if frame["id"] not in records or frame["size"] != [192, 256]:
            raise ValueError("Unknown frame or annotation grid")
    # All methods use the same exact-timestamp subset, including VGA.
    paired = [f for f in frames if records[f["id"]]["vga_exact_timestamp"]]
    if not paired:
        raise ValueError("No reviewed exact timestamp pairs")
    result = {
        "frames": [f["id"] for f in paired],
        "tolerance_native_px": tolerance,
        "threshold": summary["threshold"],
        "scope": "Reviewed regions; visible edges only",
        "split": annotations["split"],
        "methods": {},
    }
    for method in summary["methods"]:
        total = np.zeros(4, np.int64)
        for frame in paired:
            path = run / method / (frame["id"] + ".npz")
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != records[frame["id"]][method]["sha256"]
            ):
                raise ValueError("Prediction cache changed")
            with np.load(path) as archive:
                probability = archive["probabilities"][0]
            target, mask = rasterize_annotation(frame, (192, 256))
            # Ignore region borders so incomplete labels outside cannot affect matches.
            mask = binary_erosion(mask, iterations=tolerance, border_value=1)
            prediction = thin_prediction(probability, summary["threshold"]) & mask
            total += boundary_counts(prediction, target & mask, tolerance)
        result["methods"][method] = {**summarize_counts(total), "counts": total.tolist()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.run, json.loads(args.annotations.read_text()))
    result["annotations_sha256"] = hashlib.sha256(args.annotations.read_bytes()).hexdigest()
    result["summary_sha256"] = hashlib.sha256((args.run / "summary.json").read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
