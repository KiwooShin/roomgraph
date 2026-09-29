"""Build local training, held-out prediction, and 3D reconstruction visualizations."""

# HTML templates keep complete tags together.
# ruff: noqa: E501

import argparse
import html
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from roomgraph.learning.data import EdgeDataset
from roomgraph.learning.metrics import thin_prediction
from roomgraph.learning.reconstruction import cuboid_edges, write_room_obj

COLORS = [(62, 245, 187), (255, 190, 90), (74, 207, 255), (211, 143, 255), (255, 115, 139)]


def overlay(image, masks, colors):
    output = image.copy()
    for mask, color in zip(masks, colors, strict=True):
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
        output[mask] = color
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("artifacts/runs/headcam_v1"))
    parser.add_argument("--output", type=Path, default=Path("vis/perception"))
    args = parser.parse_args()
    out = args.output
    metrics = json.loads((out / "metrics.json").read_text())
    reconstruction = json.loads((out / "reconstruction.json").read_text())
    run = json.loads((args.run / "run.json").read_text())
    history = json.loads((args.run / "history.json").read_text())
    data = EdgeDataset(run["config"]["dataset"], "test")
    with np.load(out / "test_predictions.npz") as archive:
        probabilities = archive["probabilities"]
    threshold = metrics["threshold"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    axes[0].plot(
        [r["epoch"] + 1 for r in history],
        [r["train_loss"] for r in history],
        color="#5be8bd",
        label="Training (augmented)",
    )
    valid = [r for r in history if "val_loss" in r]
    axes[0].plot(
        [r["epoch"] + 1 for r in valid],
        [r["val_loss"] for r in valid],
        color="#ffbd66",
        label="Validation",
    )
    axes[0].axvline(run["best_epoch"] + 1, ls=":", color="#9cabb7", label="Selected checkpoint")
    axes[0].set(xlabel="Epoch", ylabel="Weighted BCE + Dice", title="Measured learning curves")
    axes[0].legend(facecolor="#17232e", labelcolor="white", fontsize=9)
    axes[0].grid(alpha=0.3)
    names = ["Canny\nvisible", "EdgeNet\nvisible", "EdgeNet\namodal"]
    values = [
        metrics["canny"]["test_visible"]["f1"],
        metrics["test"]["visible"]["f1"],
        metrics["test"]["amodal"]["f1"],
    ]
    bars = axes[1].bar(names, values, color=["#718294", "#5be8bd", "#ffbd66"], width=0.6)
    axes[1].set(ylim=(0, 1), ylabel="Boundary F1", title="Held-out test rooms / 2 px tolerance")
    for bar, value in zip(bars, values, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center")
    fig.savefig(out / "training.png", dpi=180)
    plt.close(fig)
    small = ImageFont.truetype("DejaVuSans.ttf", 13)
    contact = Image.new("RGB", (1536, 4 * 310), "#101820")
    cards = []
    room_data = []
    for row, room in enumerate(sorted(metrics["per_room"])):
        indices = [i for i, r in enumerate(data.records) if r["building_id"] == room]
        folder = out / room
        folder.mkdir(exist_ok=True)
        selected = indices[::6]
        for number, index in enumerate(selected):
            image = data.images[index].transpose(1, 2, 0)
            predicted = [thin_prediction(p, threshold) for p in probabilities[index]]
            gt = overlay(image, [data.targets[index, 0]], [(100, 193, 255)])
            visible = overlay(image, [predicted[0]], [(62, 245, 187)])
            amodal = overlay(image, predicted[1:], COLORS)
            sheet = Image.new("RGB", (1536, 302), "#101820")
            draw = ImageDraw.Draw(sheet)
            for column, (picture, label) in enumerate(
                [
                    (image, "Head-camera RGB"),
                    (gt, "Reference visible edges"),
                    (visible, "Predicted visible edges"),
                    (amodal, "Predicted full structure"),
                ]
            ):
                sheet.paste(Image.fromarray(picture), (column * 384, 30))
                draw.text((column * 384 + 10, 7), label, font=small, fill="white")
            sheet.save(folder / f"comparison_{number:02d}.webp", quality=90)
            Image.fromarray(visible).save(folder / f"prediction_{number:02d}.webp", quality=90)
            Image.fromarray(image).save(folder / f"rgb_{number:02d}.webp", quality=90)
            if number == 0:
                contact.paste(sheet, (0, row * 310))
                ImageDraw.Draw(contact).text(
                    (10, row * 310 + 285), room, font=small, fill="#d8e9f4"
                )
            cards.append(
                f'<figure><img src="{room}/comparison_{number:02d}.webp" alt="{room}, camera {data.records[index]["id"]}: RGB, reference and predicted structural edges"><figcaption>{html.escape(data.records[index]["id"])}</figcaption></figure>'
            )
        item = next(r for r in reconstruction if r["room"] == room)
        write_room_obj(item["prediction"]["bounds_m"], folder / "room.obj")
        (folder / "room.json").write_text(json.dumps(item["prediction"], indent=2) + "\n")
        room_data.append(
            {
                "room": room,
                "images": [f"{room}/comparison_{i:02d}.webp" for i in range(len(selected))],
                "metrics": metrics["per_room"][room],
                "reconstruction": item,
                "camera_positions": [data.records[i]["camera_to_world"] for i in indices[::3]],
            }
        )
    contact.save(out / "predictions.jpg", quality=94)
    fig = plt.figure(figsize=(12, 9), layout="constrained")
    for i, item in enumerate(reconstruction):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        for bounds, color, style, label in [
            (item["truth_bounds_m"], "#94b3cf", "--", "Reference"),
            (item["prediction"]["bounds_m"], "#5be8bd", "-", "Predicted"),
        ]:
            segments, _ = cuboid_edges(bounds)
            for j, edge in enumerate(segments):
                ax.plot(*edge.T, color=color, ls=style, lw=1.6, label=label if j == 0 else None)
        ax.set(
            xlabel="X / m",
            ylabel="Y / m",
            zlabel="Z / m",
            title=f"{item['room']} · dimension MAE {item['metrics']['dimension_mae_m']:.3f} m",
        )
        ax.set_box_aspect((6, 5, 3))
        ax.view_init(24, -52)
        ax.legend(facecolor="#17232e", labelcolor="white", fontsize=8)
    fig.savefig(out / "reconstruction.png", dpi=160)
    plt.close(fig)
    summary = {
        "run": run,
        "metrics": metrics,
        "rooms": room_data,
        "reconstruction_mean": {
            key: float(np.mean([r["metrics"][key] for r in reconstruction]))
            for key in [
                "dimension_mae_m",
                "plane_mae_m",
                "edge_chamfer_m",
                "edge_accuracy_10cm",
                "edge_completeness_10cm",
            ]
        },
        "oracle_mean": {
            key: float(np.mean([r["oracle_metrics"][key] for r in reconstruction]))
            for key in ["dimension_mae_m", "edge_chamfer_m"]
        },
        "prior_mean": {
            key: float(np.mean([r["training_median_prior_metrics"][key] for r in reconstruction]))
            for key in ["dimension_mae_m", "edge_chamfer_m"]
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RoomGraph perception results</title><style>body{{background:#101820;color:#e7eff7;font:16px/1.6 system-ui;margin:30px auto;max-width:1500px;padding:0 20px}}img{{width:100%;height:auto}}figure{{margin:28px 0}}a{{color:#5be8bd}}code{{color:#ffbd66}}</style><h1>Head-camera RGB → structural edges → room shell</h1><p>Held-out synthetic rooms. Known metric camera poses and Manhattan axes. No depth at inference. Robot geometry is an original approximate proxy, not an official 1X model.</p><p>Visible F1: {metrics["test"]["visible"]["f1"]:.3f}; amodal F1: {metrics["test"]["amodal"]["f1"]:.3f}; mean dimension error: {summary["reconstruction_mean"]["dimension_mae_m"]:.3f} m. Threshold selected only on validation.</p><p><a href="summary.json">Full measured results</a> · TensorBoard: <code>http://127.0.0.1:6006</code></p><img src="training.png" alt="Training curves and held-out boundary F1"><img src="reconstruction.png" alt="Predicted versus ground-truth room shells"><h2>Fixed camera samples from each test room</h2><p>Visible predictions: mint. Reference: blue. Typed full structure: wall–wall mint, wall–floor amber, wall–ceiling cyan, door violet, window coral. These are predictions, including errors.</p>{"".join(cards)}<p>Single-room closed shells only; opening geometry, SLAM, free-space mapping and navigation are not implemented. Full local captures and checkpoints stay out of Git.</p></html>"""
    (out / "report.html").write_text(content)
    print(out / "report.html")


if __name__ == "__main__":
    main()
