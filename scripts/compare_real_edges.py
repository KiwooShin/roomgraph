"""Compare frozen real edge predictions on common native pixels; no label tuning."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from roomgraph.learning.model import EdgeNet
from roomgraph.real_edges import prepare_image, restore_probabilities
from roomgraph.real_rgbd import read_intrinsics


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("vis/arkitscenes/pilot_v2"))
    parser.add_argument(
        "--vga", type=Path, default=Path("artifacts/real/arkitscenes_vga/Validation/42445021")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = json.loads((args.pilot / "manifest.json").read_text())
    checkpoint = Path(manifest["config"]["checkpoint"])
    if digest(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from frozen pilot")
    for frame in manifest["frames"]:
        if digest(frame["rgb"]) != manifest["input_sha256"][frame["rgb"]]:
            raise ValueError("Pilot RGB changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = EdgeNet(pretrained=False).to(device, memory_format=torch.channels_last).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    modes = ["stretch", "letterbox", "portrait", "portrait_large", "vga_portrait_large"]
    records, times, fractions = [], {m: [] for m in modes}, {m: [] for m in modes}
    matched_indices = [
        i
        for i, f in enumerate(manifest["frames"])
        if (args.vga / "vga_wide" / Path(f["rgb"]).name).exists()
    ]
    if len(matched_indices) < 8:
        raise ValueError("Need at least eight exact timestamp VGA matches")
    review_indices = {
        matched_indices[i] for i in np.linspace(0, len(matched_indices) - 1, 8).round().astype(int)
    }
    reviews = []
    for index, frame in enumerate(manifest["frames"]):
        raw = cv2.cvtColor(cv2.imread(frame["rgb"]), cv2.COLOR_BGR2RGB)
        rgb = np.rot90(raw, -manifest["clockwise_quarter_turns"]).copy()
        vga_path = args.vga / "vga_wide" / Path(frame["rgb"]).name
        vga = None
        extra = {}
        if vga_path.exists():
            vk, size = read_intrinsics(
                args.vga / "vga_wide_intrinsics" / vga_path.with_suffix(".pincam").name
            )
            # Same wide camera, exact timestamp. Map its calibrated rays onto the
            # lowres camera's grid at 2.5x resolution, before upright rotation.
            k = np.asarray(frame["intrinsics"])
            scale = np.diag([size[0] / raw.shape[1], size[1] / raw.shape[0], 1])
            homography = scale @ k @ np.linalg.inv(vk)
            source = cv2.cvtColor(cv2.imread(str(vga_path)), cv2.COLOR_BGR2RGB)
            aligned = cv2.warpPerspective(source, homography, size)
            valid = cv2.warpPerspective(
                np.ones(source.shape[:2], np.uint8), homography, size, flags=cv2.INTER_NEAREST
            )
            if valid.mean() < 0.995:
                raise ValueError("VGA calibration leaves significant missing coverage")
            vga = np.rot90(aligned, -manifest["clockwise_quarter_turns"]).copy()
            extra = {
                "vga_sha256": digest(vga_path),
                "vga_homography": homography.tolist(),
                "vga_intrinsics_sha256": digest(
                    args.vga / "vga_wide_intrinsics" / vga_path.with_suffix(".pincam").name
                ),
                "vga_coverage": float(valid.mean()),
            }
        overlays = []
        row = {
            "id": frame["id"],
            "index": index,
            "vga_exact_timestamp": vga is not None,
            "input_sha256": digest(frame["rgb"]),
            **extra,
        }
        for mode in modes:
            source = vga if mode.startswith("vga_") else rgb
            if source is None:
                overlays.append(Image.new("RGB", (192, 256), "#263343"))
                continue
            network, rectangle = prepare_image(source, mode.removeprefix("vga_"))
            tensor = torch.from_numpy(network.transpose(2, 0, 1).copy())[None].to(device)
            tensor = tensor.float().div_(255).contiguous(memory_format=torch.channels_last)
            if device.type == "cuda":
                torch.cuda.synchronize()
            tick = time.perf_counter()
            with (
                torch.inference_mode(),
                torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"),
            ):
                probabilities = model(tensor).float().sigmoid().cpu().numpy()[0]
            elapsed = time.perf_counter() - tick
            times[mode].append(elapsed)
            restored = restore_probabilities(probabilities, rectangle, rgb.shape[1::-1])
            fractions[mode].append(float((restored[0] >= 0.7).mean()))
            folder = args.output / mode
            folder.mkdir(exist_ok=True)
            cache = folder / (frame["id"] + ".npz")
            np.savez_compressed(cache, probabilities=restored)
            row[mode] = {
                "sha256": digest(cache),
                "forward_transfer_s": elapsed,
                "network_size": list(network.shape[1::-1]),
                "predicted_pixel_fraction": fractions[mode][-1],
            }
            overlay = rgb.copy()
            overlay[restored[0] >= 0.7] = [45, 255, 189]
            overlays.append(Image.fromarray(overlay))
        records.append(row)
        if index in review_indices:
            review = args.output / "review"
            review.mkdir(exist_ok=True)
            Image.fromarray(rgb).save(review / (frame["id"] + ".png"))
            reviews.append(
                {
                    "id": frame["id"],
                    "size": [192, 256],
                    "status": "unreviewed",
                    "review_regions": [],
                    "polylines": [],
                }
            )
            canvas = Image.new("RGB", (192 * 6, 286), "#101820")
            draw = ImageDraw.Draw(canvas)
            for col, (name, im) in enumerate(
                zip(["Original"] + modes, [Image.fromarray(rgb)] + overlays, strict=True)
            ):
                canvas.paste(im, (192 * col, 30))
                draw.text((192 * col + 4, 4), name, fill="white")
            canvas.save(args.output / f"comparison_{index:03d}.jpg", quality=95)
        if index % 40 == 0:
            print(f"{index + 1}/{len(manifest['frames'])} frames", flush=True)
    summary = {
        "status": "complete",
        "threshold": 0.7,
        "device": str(device),
        "pilot_manifest_sha256": digest(args.pilot / "manifest.json"),
        "checkpoint_sha256": digest(checkpoint),
        "source_sha256": {
            p: digest(p)
            for p in [__file__, "src/roomgraph/real_edges.py", "src/roomgraph/learning/model.py"]
        },
        "frames": len(records),
        "vga_exact_matches": sum(r["vga_exact_timestamp"] for r in records),
        "wall_seconds": time.perf_counter() - started,
        "peak_tensor_memory_mib": torch.cuda.max_memory_allocated() / 2**20
        if device.type == "cuda"
        else None,
        "metrics_status": "No verified real edge labels. Coverage is NOT accuracy.",
        "methods": {
            m: {
                "frames": len(times[m]),
                "median_forward_transfer_ms": float(np.median(times[m]) * 1000)
                if times[m]
                else None,
                "mean_predicted_pixel_fraction": float(np.mean(fractions[m]))
                if fractions[m]
                else None,
                "paired_mean_predicted_pixel_fraction": float(
                    np.mean(
                        [
                            r[m]["predicted_pixel_fraction"]
                            for r in records
                            if r["vga_exact_timestamp"]
                        ]
                    )
                ),
            }
            for m in modes
        },
        "records": records,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    annotation = {
        "schema_version": 1,
        "video": manifest["config"]["video"],
        "visit": manifest["dataset_metadata"]["visit_id"],
        "split": "development_review",
        "provenance": "Unlabeled selection; no pseudo-GT",
        "coordinates": "upright native RGB pixels, origin top-left",
        "classes": ["wall_wall", "wall_floor", "wall_ceiling", "door", "window"],
        "frames": reviews,
    }
    (args.output / "review/annotations.json").write_text(json.dumps(annotation, indent=2) + "\n")
    shutil.copyfile("assets/tools/edge_review.html", args.output / "review/index.html")
    rows = "".join(
        f'<img alt="Same-view preprocessing comparison" src="comparison_{i:03d}.jpg">'
        for i in sorted(review_indices)
    )
    brief = {k: v for k, v in summary.items() if k != "records"}
    (args.output / "index.html").write_text(
        '<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Real edge preprocessing comparison</title><style>"
        "body{background:#101820;color:#edf2f7;font:16px system-ui;margin:24px}"
        "img{width:100%;max-width:1152px;display:block;margin:20px 0}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#67e8c3}</style>"
        "<h1>Same camera, five input strategies</h1><p>Frozen synthetic model, threshold 0.7. "
        "All overlays restored to the same 192 × 256 upright RGB grid. "
        "Blank VGA panels mean no exact timestamp match. Increased edge coverage is not "
        "improved accuracy. One development visit; no held-out real test.</p>"
        '<p><a href="review/index.html">Review structural-edge annotations</a></p>'
        + rows
        + "<h2>Measured execution</h2><pre>"
        + json.dumps(brief, indent=2)
        + "</pre>"
    )
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
