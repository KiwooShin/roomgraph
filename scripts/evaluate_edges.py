"""Select thresholds on validation, then evaluate a frozen checkpoint on held-out rooms."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from roomgraph.learning.data import CHANNELS, EdgeDataset
from roomgraph.learning.metrics import boundary_counts, evaluate_boundaries, summarize_counts
from roomgraph.learning.model import EdgeNet
from roomgraph.learning.reconstruction import fit_room, reconstruction_metrics


def predict(model, dataset, device, batch_size=16):
    output = []
    with torch.inference_mode():
        for image, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            image = image.to(device).float().div_(255).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                result = model(image).float().sigmoid()
            output.append(result.cpu().numpy().astype(np.float16))
    return np.concatenate(output)


def baseline(images, targets, low):
    counts = np.zeros(4, dtype=np.int64)
    for image, target in zip(images, targets, strict=True):
        grey = cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(grey, low, 2 * low) > 0
        counts += boundary_counts(edge, target[0], 2)
    return summarize_counts(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("artifacts/runs/headcam_v1"))
    parser.add_argument("--output", type=Path, default=Path("vis/perception"))
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    checkpoint = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeNet(pretrained=False).to(device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    val = EdgeDataset(
        config["dataset"],
        "train" if args.diagnostic else "val",
        limit=8 if args.diagnostic else None,
    )
    probabilities = predict(model, val, device)
    curve = []
    for threshold in config["thresholds"]:
        metrics = evaluate_boundaries(
            probabilities, val.targets, threshold, config["tolerance_pixels"]
        )
        row = {"threshold": threshold, **metrics}
        curve.append(row)
        print(
            "VALIDATION", threshold, metrics["visible"]["f1"], metrics["amodal"]["f1"], flush=True
        )
    selected = max(curve, key=lambda r: (r["visible"]["f1"] + r["amodal"]["f1"]) / 2)
    threshold = selected["threshold"]
    if not args.diagnostic:
        (args.run / "inference.json").write_text(
            json.dumps(
                {
                    "threshold": threshold,
                    "width": 384,
                    "height": 256,
                    "channels": CHANNELS,
                    "selection": "validation only",
                },
                indent=2,
            )
            + "\n"
        )
    result = {
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": hashlib.sha256((args.run / "best.pt").read_bytes()).hexdigest(),
        "channels": CHANNELS,
        "threshold": threshold,
        "threshold_selection": "Mean visible/amodal micro F1 on validation only",
        "metric": (
            "Skeletonized boundary precision/recall/F1, symmetric 2-pixel "
            "Euclidean tolerance at 384 x 256; many-to-one matching allowed"
        ),
        "validation": selected,
        "validation_curve": curve,
    }
    np.savez_compressed(args.output / "validation_predictions.npz", probabilities=probabilities)
    if args.diagnostic:
        result["diagnostic"] = "Same eight training images; not held-out performance"
        (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        return
    canny_choices = [(low, baseline(val.images, val.targets, low)) for low in [40, 80, 120]]
    low, canny_val = max(canny_choices, key=lambda item: item[1]["f1"])
    test = EdgeDataset(config["dataset"], "test")
    prediction = predict(model, test, device)
    result["test"] = evaluate_boundaries(
        prediction, test.targets, threshold, config["tolerance_pixels"]
    )
    result["test_images"] = len(test)
    result["test_rooms"] = len({r["building_id"] for r in test.records})
    result["canny"] = {
        "low": low,
        "high": 2 * low,
        "validation": canny_val,
        "test_visible": baseline(test.images, test.targets, low),
    }
    result["per_room"] = {}
    for room in sorted({r["building_id"] for r in test.records}):
        indices = [i for i, r in enumerate(test.records) if r["building_id"] == room]
        result["per_room"][room] = evaluate_boundaries(
            prediction[indices], test.targets[indices], threshold, 2
        )
    np.savez_compressed(args.output / "test_predictions.npz", probabilities=prediction)
    (args.output / "test_records.json").write_text(json.dumps(test.records, indent=2) + "\n")
    train = EdgeDataset(config["dataset"], "train")
    train.images = train.images[::4]
    train.targets = train.targets[::4]
    train.records = train.records[::4]
    train_prediction = predict(model, train, device)
    result["train_subset"] = evaluate_boundaries(train_prediction, train.targets, threshold, 2)
    result["train_subset_images"] = len(train)
    # Forward-only timing, excluding I/O, CPU postprocessing, and reconstruction.
    image = (
        torch.from_numpy(test.images[:1])
        .to(device)
        .float()
        .div_(255)
        .contiguous(memory_format=torch.channels_last)
    )
    times = []
    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"),
    ):
        for i in range(40):
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            model(image)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= 10:
                times.append(1000 * (time.perf_counter() - started))
    result["inference"] = {
        "batch": 1,
        "width": 384,
        "height": 256,
        "precision": "BF16 autocast" if device.type == "cuda" else "FP32",
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "scope": "model forward only; 10 warmup, 30 timed iterations",
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print("TEST", json.dumps(result["test"]), flush=True)
    # Geometry reconstruction only receives predictions and calibration.
    reconstructions = []
    for room in sorted(result["per_room"]):
        indices = [i for i, r in enumerate(test.records) if r["building_id"] == room][::3]
        records = [test.records[i] for i in indices]
        started = time.perf_counter()
        estimated = fit_room(prediction[indices], records, threshold)
        estimated["seconds"] = time.perf_counter() - started
        oracle = fit_room(test.targets[indices].astype(np.float32), records, 0.5)
        # Ground-truth dimensions are opened strictly AFTER inference, for scoring.
        manifest = json.loads(Path(records[0]["manifest"]).read_text())
        w, d, h = manifest["scene_config"]["room"]["dimensions_m"]
        truth = [-w / 2, w / 2, -d / 2, d / 2, 0, h]
        prior_manifest = [json.loads(Path(r["manifest"]).read_text()) for r in train.records[::6]]
        median = np.median(
            [m["scene_config"]["room"]["dimensions_m"] for m in prior_manifest], axis=0
        )
        prior = [-median[0] / 2, median[0] / 2, -median[1] / 2, median[1] / 2, 0, median[2]]
        result_room = {
            "room": room,
            "prediction": estimated,
            "oracle": oracle,
            "truth_bounds_m": truth,
            "metrics": reconstruction_metrics(estimated["bounds_m"], truth),
            "oracle_metrics": reconstruction_metrics(oracle["bounds_m"], truth),
            "training_median_prior_metrics": reconstruction_metrics(prior, truth),
        }
        reconstructions.append(result_room)
        (args.output / "reconstruction.json").write_text(
            json.dumps(reconstructions, indent=2) + "\n"
        )
        print("RECONSTRUCTION", room, json.dumps(result_room["metrics"]), flush=True)
    from roomgraph.learning.tracking import log_evaluation

    log_evaluation(args.run, result, reconstructions)
    print("COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
