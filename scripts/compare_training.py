"""Compare training speed and fixed-threshold validation quality without opening test data."""

import argparse
import hashlib
import html
import json
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from roomgraph.learning.data import EdgeDataset
from roomgraph.learning.metrics import evaluate_boundaries, thin_prediction
from roomgraph.learning.model import EdgeNet

COLORS = ["#5be8bd", "#ffbd66", "#a993ff", "#59bbf7", "#f48da5"]
EDGE_COLORS = [(62, 245, 187), (255, 190, 90), (74, 207, 255), (211, 143, 255), (255, 115, 139)]
WORK_KEYS = (
    "seed",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "amp",
    "pretrained",
    "patience",
    "eval_every",
    "tolerance_pixels",
)
RUNTIME_KEYS = {"run_dir", "workers", "checkpoint_every"}


def semantic_config(config):
    """Match the trainer's allowed runtime-only resume changes."""
    return {key: value for key, value in config.items() if key not in RUNTIME_KEYS}


def time_to_quality(history, target):
    """Use the first scheduled validation crossing; never interpolate missing timings."""
    row = next((row for row in history if row.get("val_loss", math.inf) <= target), None)
    return {
        "target_val_loss": target,
        "attained": row is not None,
        "epoch": row["epoch"] if row else None,
        "train_session_elapsed_seconds": row.get("elapsed_seconds") if row else None,
    }


def describe_run(name, directory, target_val_loss=0.5):
    """Read provenance and measured work; missing measurements remain explicitly unknown."""
    run = json.loads((directory / "run.json").read_text())
    history = json.loads((directory / "history.json").read_text())
    if not history:
        raise ValueError(f"{name}: training history is empty")
    if run.get("diagnostic_overfit"):
        raise ValueError(f"{name}: diagnostic overfit runs are not held-out validation runs")
    config = run["config"]
    index_path = Path(config["dataset"]) / "index.json"
    dataset_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    if dataset_sha != run.get("dataset_sha256"):
        raise ValueError(f"{name}: dataset index differs from recorded training provenance")
    index = json.loads(index_path.read_text())
    train_images = index["splits"]["train"]["images"]
    images_seen = sum(row.get("train_images", train_images) for row in history)
    inferred_steps = sum(
        math.ceil(row.get("train_images", train_images) / config["batch_size"]) for row in history
    )
    steps = history[-1].get("optimizer_steps", inferred_steps)
    if steps != inferred_steps:
        raise ValueError(f"{name}: optimizer step count contradicts history and batch size")
    loop_seconds = sum(row["seconds"] for row in history)
    peak = run.get("peak_allocated_mb")
    if peak is None:
        measured = [row["peak_allocated_mb"] for row in history if "peak_allocated_mb" in row]
        peak = max(measured) if measured else None
    result = {
        "name": name,
        "directory": str(directory),
        "dataset_sha256": dataset_sha,
        "config": config,
        "gpu": run.get("gpu"),
        "torch": run.get("torch"),
        "parameters": run.get("parameters"),
        "git_commit": run.get("git_commit"),
        "git_dirty": run.get("git_dirty"),
        "sampling": run.get("sampling", "legacy shared RNG"),
        "compiler_cache": run.get("compiler_cache", "unrecorded"),
        "resumed": "resumed_from" in run,
        "resume_epoch": run.get("resumed_from", {}).get("epoch"),
        "epochs_completed": len(history),
        "best_epoch": run.get("best_epoch"),
        "best_val_loss": run.get("best_val_loss"),
        "time_to_quality": time_to_quality(history, target_val_loss),
        "optimizer_steps": steps,
        "optimizer_steps_source": "recorded" if "optimizer_steps" in history[-1] else "inferred",
        "training_images_seen": images_seen,
        "train_loop_seconds": loop_seconds,
        "train_loop_images_per_second": images_seen / loop_seconds,
        "steady_epoch_median_images_per_second": float(
            np.median([row["images_per_second"] for row in history[1:] or history])
        ),
        "train_session_seconds": run.get("total_seconds"),
        "process_seconds": run.get("process_seconds"),
        "launcher_wall_seconds": run.get("launcher_wall_seconds"),
        "peak_allocated_mb": peak,
        "validation_images": index["splits"]["val"]["images"],
        "validation_rooms": index["splits"]["val"]["rooms"],
        "learning_rate_history": [row["lr"] for row in history],
        "epoch_indices": [row["epoch"] for row in history],
    }
    return result, history


def compare_work(runs):
    """Check equal workload separately from hardware/software comparability."""
    reference = runs[0]
    mismatches = []
    environment = []
    timing = [
        f"{run['name']}: resumed run; process/wall time covers only the latest session, "
        "and cumulative session timing excludes earlier checkpoint serialization tails"
        for run in runs
        if run.get("resumed")
    ]
    for other in runs[1:]:
        for key in WORK_KEYS:
            if reference["config"].get(key) != other["config"].get(key):
                mismatches.append(f"{other['name']}: config.{key} differs from {reference['name']}")
        for key in (
            "dataset_sha256",
            "parameters",
            "epochs_completed",
            "optimizer_steps",
            "training_images_seen",
            "learning_rate_history",
            "epoch_indices",
            "sampling",
        ):
            if reference.get(key) != other.get(key):
                mismatches.append(f"{other['name']}: {key} differs from {reference['name']}")
        for key in ("gpu", "torch"):
            if reference.get(key) != other.get(key):
                environment.append(f"{other['name']}: {key} differs from {reference['name']}")
    return {
        "matched_training_work": not mismatches,
        "matched_reported_environment": not environment,
        "comparable_fresh_run_timing": not timing,
        "work_mismatches": mismatches,
        "environment_mismatches": environment,
        "timing_caveats": timing,
        "caveat": (
            "Equal seed/work does not guarantee identical augmentation draws or numerical "
            "results. This comparison does not establish multi-seed statistical significance."
        ),
    }


def predict_validation(model, dataset, device, batch_size, amp):
    """Evaluate only the explicitly supplied validation cache."""
    probabilities = []
    with torch.inference_mode():
        for image, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            image = image.to(device).float().div_(255).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                probability = model(image).float().sigmoid()
            probabilities.append(probability.cpu().numpy().astype(np.float16))
    return np.concatenate(probabilities)


def plot_comparison(runs, histories, output, target_val_loss=0.5):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": "#101820",
            "axes.facecolor": "#17232e",
            "axes.edgecolor": "#607080",
            "text.color": "#e7eff7",
            "axes.labelcolor": "#becfdf",
            "xtick.color": "#becfdf",
            "ytick.color": "#becfdf",
            "grid.color": "#334655",
            "savefig.facecolor": "#101820",
        }
    )
    walltime = all("elapsed_seconds" in row for history in histories for row in history)
    time_scope = "Training session elapsed seconds" if walltime else "Cumulative train-loop seconds"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for i, (run, history) in enumerate(zip(runs, histories, strict=True)):
        times = (
            [row["elapsed_seconds"] for row in history]
            if walltime
            else np.cumsum([row["seconds"] for row in history])
        )
        color = COLORS[i % len(COLORS)]
        axes[0].plot(times, [row["train_loss"] for row in history], color=color, label=run["name"])
        valid = [
            (second, row) for second, row in zip(times, history, strict=True) if "val_loss" in row
        ]
        axes[1].plot(
            [pair[0] for pair in valid],
            [pair[1]["val_loss"] for pair in valid],
            color=color,
            label=run["name"],
        )
    for axis, title in zip(axes, ["Augmented training loss", "Validation loss"], strict=True):
        axis.set(xlabel=time_scope, ylabel="Weighted BCE + Dice", title=title)
        axis.grid(alpha=0.4)
        axis.legend(facecolor="#17232e", labelcolor="white")
    axes[1].axhline(target_val_loss, color="#aab9c7", linestyle=":", linewidth=1)
    axes[1].text(
        0.98,
        target_val_loss,
        f"Fixed target {target_val_loss:g}",
        transform=axes[1].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=9,
    )
    fig.savefig(output / "loss_curves.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for axis, key, title, unit in zip(
        axes,
        ["train_session_seconds", "train_loop_images_per_second", "peak_allocated_mb"],
        [
            "Training session",
            "Train-loop average\n(includes compilation)",
            "Peak CUDA tensor allocation",
        ],
        [
            "Seconds · lower is faster",
            "Images / second · higher is faster",
            "MiB · lower is smaller",
        ],
        strict=True,
    ):
        values = [run[key] for run in runs]
        bars = axis.bar(
            [run["name"] for run in runs],
            [value or 0 for value in values],
            color=[COLORS[i % len(COLORS)] for i in range(len(runs))],
            width=0.6,
        )
        axis.set(title=title, ylabel=unit)
        axis.set_ylim(0, max([value for value in values if value] or [1]) * 1.18)
        axis.tick_params(axis="x", labelrotation=15)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}" if value is not None else "Unrecorded",
                ha="center",
                va="bottom",
            )
    fig.savefig(output / "speed_memory.png", dpi=180)
    plt.close(fig)
    return time_scope


def overlay(image, masks, colors):
    output = image.copy()
    for mask, color in zip(masks, colors, strict=True):
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
        output[mask] = color
    return output


def example_predictions(dataset, probabilities, threshold):
    selected = []
    for room in sorted({record["building_id"] for record in dataset.records})[:4]:
        indices = [i for i, record in enumerate(dataset.records) if record["building_id"] == room]
        index = indices[len(indices) // 2]
        selected.append(
            {
                "record": dataset.records[index],
                "image": dataset.images[index].transpose(1, 2, 0),
                "target": dataset.targets[index],
                "predicted": [thin_prediction(p, threshold) for p in probabilities[index]],
            }
        )
    return selected


def save_overlays(runs, examples, output):
    """Pair exactly matching camera IDs; mismatched datasets get separate previews."""
    by_id = {}
    for run, items in zip(runs, examples, strict=True):
        for item in items:
            key = (run["dataset_sha256"], item["record"]["id"])
            by_id.setdefault(key, []).append((run["name"], item))
    font = ImageFont.truetype("DejaVuSans.ttf", 14)
    previews = []
    for number, ((dataset_sha, record_id), variants) in enumerate(by_id.items()):
        _, reference = variants[0]
        image, target = reference["image"], reference["target"]
        panels = [
            ("Head-camera RGB", image),
            ("Reference visible", overlay(image, target[:1], [(100, 193, 255)])),
            ("Reference full structure", overlay(image, target[1:], EDGE_COLORS)),
        ]
        for name, item in variants:
            panels.extend(
                [
                    (
                        f"{name}: visible",
                        overlay(item["image"], item["predicted"][:1], [(62, 245, 187)]),
                    ),
                    (
                        f"{name}: full structure",
                        overlay(item["image"], item["predicted"][1:], EDGE_COLORS),
                    ),
                ]
            )
        height, width = image.shape[:2]
        sheet = Image.new("RGB", (width * len(panels), height + 38), "#101820")
        draw = ImageDraw.Draw(sheet)
        for column, (label, picture) in enumerate(panels):
            sheet.paste(Image.fromarray(picture), (column * width, 34))
            draw.text((column * width + 9, 9), label, fill="white", font=font)
        filename = f"validation_{number:02d}.webp"
        sheet.save(output / filename, quality=92)
        previews.append({"record_id": record_id, "dataset_sha256": dataset_sha, "image": filename})
    return previews


def write_report(result, output):
    def value(number, suffix=""):
        return "Unrecorded" if number is None else f"{number:.2f}{suffix}"

    rows = []
    for run in result["runs"]:
        metrics = run["validation"]
        quality = run["time_to_quality"]
        quality_time = (
            value(quality["train_session_elapsed_seconds"], " s")
            if quality["attained"]
            else "Not reached"
        )
        cells = [
            html.escape(run["name"]),
            str(run["optimizer_steps"]),
            value(run["train_session_seconds"], " s"),
            value(run["process_seconds"], " s"),
            value(run["launcher_wall_seconds"], " s"),
            value(run["train_loop_images_per_second"]),
            value(run["peak_allocated_mb"], " MiB"),
            quality_time,
            f"{metrics['visible']['f1']:.4f}",
            f"{metrics['amodal']['f1']:.4f}",
        ]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    comparison = result["comparison"]
    issues = (
        comparison["work_mismatches"]
        + comparison["environment_mismatches"]
        + comparison["timing_caveats"]
    )
    comparability = (
        "Matched training workload and reported hardware/software."
        if not issues
        else "Comparison caveats: " + "; ".join(issues)
    )
    cards = "".join(
        f'<figure><div class="scroll"><img class="preview" src="{item["image"]}" '
        f'alt="Validation camera {html.escape(item["record_id"])} edge comparison"></div>'
        f"<figcaption>{html.escape(item['record_id'])}</figcaption></figure>"
        for item in result["previews"]
    )
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoomGraph · training efficiency</title>
<style>
body{{background:#101820;color:#e7eff7;font:16px/1.65 system-ui;max-width:1500px;
margin:36px auto;padding:0 22px}}h1{{line-height:1.15}}a{{color:#5be8bd}}
.eyebrow{{color:#5be8bd;letter-spacing:.12em;font-size:13px;text-transform:uppercase}}
.muted,figcaption{{color:#b5c7d7}}img{{width:100%;height:auto}}figure{{margin:30px 0}}
.scroll{{overflow-x:auto}}.preview{{min-width:1400px}}table{{border-collapse:collapse;width:100%}}
th,td{{padding:13px 12px;border-bottom:1px solid #334655;text-align:left;white-space:nowrap}}
th{{font-size:13px;color:#b5c7d7}}.note{{border-left:3px solid #ffbd66;padding:8px 16px}}
</style></head><body>
<p class="eyebrow">RoomGraph / measured training efficiency</p>
<h1>Faster training, checked against validation quality</h1>
<p>Fixed threshold {result["threshold"]:.2f}; no threshold search and no test cache opened.
Visible and amodal scores are skeletonized boundary F1 with symmetric pixel tolerance
specified in each run. All checkpoints were selected by validation loss.</p>
<p class="note">{html.escape(comparability)} {html.escape(comparison["caveat"])}</p>
<p><a href="comparison.json">Measurements and provenance (JSON)</a></p>
<div class="scroll"><table><thead><tr><th>Run</th><th>Updates</th><th>Train session</th>
<th>Main process</th><th>Launcher wall</th><th>Train images/s</th><th>Peak CUDA</th>
<th>Time to val loss ≤ {result["target_val_loss"]:g}</th>
<th>Visible F1</th><th>Amodal F1</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<p class="muted">Train-loop time measures batches, augmentation, forward/backward and optimizer
steps. Train-session time also includes validation, checkpointing and logging. Main-process
time additionally includes initialization but excludes Python imports. Launcher wall time
includes interpreter startup/imports. Peak CUDA is tensor allocation, not total system memory.
Unrecorded fields are not estimated.</p>
<p class="muted">Chart throughput averages all training batches, including first-epoch
compilation. Steady throughput in the JSON is the median after the first epoch.</p>
<p class="muted">Time to quality is the first scheduled validation loss ≤
{result["target_val_loss"]:g}, measured on the training-session clock. It includes compilation
within training and validation/checkpoint overhead, but excludes process initialization.
The target is fixed before comparison; no interpolation between validation checks.</p>
<img src="speed_memory.png" alt="Measured training-session speed, throughput and GPU memory">
<img src="loss_curves.png" alt="Training and validation losses against measured elapsed time">
<p class="muted">Curve time axis: {html.escape(result["curve_time_scope"])}.
One seed per run; repeat experiments before claiming a robust speed or quality improvement.</p>
<h2>Fixed validation camera examples</h2>
<p>The middle camera in each of up to four validation rooms, chosen independently of model
quality. Scroll wide comparisons horizontally. Visible reference: blue; prediction: mint.
Full structure: wall–wall mint, wall–floor amber, wall–ceiling cyan, door violet, window coral.
Faint furniture edges and hallucinated or missing structure remain visible in these predictions.</p>
{cards}<p class="muted">These synthetic rooms share procedural scene families with training;
validation quality does not establish generalization to unseen buildings or real robots.
This report and full local visualizations remain outside Git.</p></body></html>
"""
    (output / "report.html").write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, metavar="NAME=DIRECTORY")
    parser.add_argument("--output", type=Path, default=Path("vis/training_efficiency"))
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--target-val-loss", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be strictly between 0 and 1")
    if not math.isfinite(args.target_val_loss) or args.target_val_loss <= 0:
        parser.error("--target-val-loss must be finite and positive")
    directories = []
    for item in args.runs:
        if "=" not in item or not all(item.split("=", 1)):
            parser.error("Each --runs entry must be NAME=DIRECTORY")
        name, directory = item.split("=", 1)
        directories.append((name, Path(directory)))
    if len({name for name, _ in directories}) != len(directories):
        parser.error("Run names must be unique")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    runs, histories, examples = [], [], []
    for name, directory in directories:
        run, history = describe_run(name, directory, args.target_val_loss)
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        if checkpoint["provenance"]["dataset_sha256"] != run["dataset_sha256"]:
            raise ValueError(f"{name}: best checkpoint and run dataset provenance differ")
        if semantic_config(checkpoint["config"]) != semantic_config(run["config"]):
            raise ValueError(f"{name}: best checkpoint and run configurations differ")
        run["checkpoint_config"] = checkpoint["config"]
        run["checkpoint_sha256"] = hashlib.sha256((directory / "best.pt").read_bytes()).hexdigest()
        run["checkpoint_epoch"] = checkpoint["epoch"]
        if run["checkpoint_epoch"] != run["best_epoch"]:
            raise ValueError(f"{name}: best checkpoint epoch differs from run metadata")
        model = EdgeNet(pretrained=False).to(device, memory_format=torch.channels_last)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        config = run["config"]
        dataset = EdgeDataset(config["dataset"], "val")
        probabilities = predict_validation(
            model, dataset, device, config["batch_size"], config["amp"] and device.type == "cuda"
        )
        run["validation"] = evaluate_boundaries(
            probabilities, dataset.targets, args.threshold, config["tolerance_pixels"]
        )
        examples.append(example_predictions(dataset, probabilities, args.threshold))
        runs.append(run)
        histories.append(history)
        print("VALIDATION", name, json.dumps(run["validation"]), flush=True)
        del model, probabilities, dataset, checkpoint
    result = {
        "schema_version": 1,
        "threshold": args.threshold,
        "target_val_loss": args.target_val_loss,
        "threshold_selection": "Fixed before comparison; no threshold search",
        "split": "val",
        "evaluation_device": str(device),
        "comparison": compare_work(runs),
        "runs": runs,
        "curve_time_scope": plot_comparison(runs, histories, args.output, args.target_val_loss),
        "previews": save_overlays(runs, examples, args.output),
    }
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(result, args.output)
    print("COMPLETE", args.output / "report.html", flush=True)


if __name__ == "__main__":
    main()
