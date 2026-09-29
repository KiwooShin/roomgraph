"""Visualize measured active-head trials without requiring a renderer or GPU."""

# Complete HTML/JavaScript template lines are kept together for readability.
# ruff: noqa: E501

import argparse
import hashlib
import html
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from roomgraph.headcam import head_camera_pose
from roomgraph.visualization import camera_triangle, floorplan_pixel

BG = "#0c1420"
PANEL = "#142131"
TEXT = "#e7f0f7"
MUTED = "#9bb0c5"
MINT = "#55e4bd"
AMBER = "#ffbe70"
STATE_COLORS = {
    "unseen": "#ef798c",
    "looked_unsupported": "#ffbe70",
    "single_center_support": "#69baff",
    "multiple_center_support": "#55e4bd",
}
STATE_LABELS = {
    "unseen": "Unseen",
    "looked_unsupported": "Looked, no support",
    "single_center_support": "One center",
    "multiple_center_support": "Translated support",
}
WIDTH, HEIGHT = 1440, 986


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default(size=size)


def image_path(value, source):
    path = Path(value)
    return path if path.is_absolute() else source.parent / path


def bounds_segments(bounds):
    x0, x1, y0, y1, z0, z1 = bounds
    corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    return [corners[[i, j]] for i in range(8) for j in range(i + 1, 8) if (i ^ j).bit_count() == 1]


def plot_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": "#466077",
            "text.color": TEXT,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": "#466077",
            "savefig.facecolor": BG,
        }
    )


def policy_label(policy):
    name = policy["name"]
    return name.replace("_", " ").capitalize()


def policy_color(policy):
    if policy["name"] == "active":
        return MINT
    if policy["name"] == "raster":
        return AMBER
    return "#8099bc"


def comparison_plot(data, output):
    """Plot every measured policy; do not smooth or fabricate intermediate evidence."""
    budget = float(data["config"]["action_budget_s"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
    for ax, key, title in zip(
        axes[:2],
        ("visible_edge_coverage", "multi_view_edge_coverage"),
        ("Reference edges seen", "Reference edges seen from translated views"),
        strict=True,
    ):
        for policy in data["policies"]:
            trace = policy["trace"]
            times = [step["t_s"] for step in trace] + [budget]
            values = [step["evaluation"][key] * 100 for step in trace]
            values.append(values[-1])
            ax.step(
                times,
                values,
                where="post",
                label=policy_label(policy),
                color=policy_color(policy),
                lw=2.5 if policy["name"] in {"active", "raster"} else 1,
                alpha=1 if policy["name"] in {"active", "raster"} else 0.65,
            )
        ax.set(
            xlabel="Simulated head-action time / s",
            ylabel="Coverage / %",
            title=title,
            xlim=(0, budget),
            ylim=(0, 100),
        )
        ax.grid(alpha=0.3)
    axes[0].legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
    names = ["Initial"] + [policy_label(p) for p in data["policies"]]
    metrics = [data["initial_metrics"]] + [p["final_metrics"] for p in data["policies"]]
    values = [m["dimension_mae_m"] * 100 for m in metrics]
    bars = axes[2].bar(
        range(len(values)), values, color=["#adc0d0"] + [policy_color(p) for p in data["policies"]]
    )
    axes[2].set(
        ylabel="Dimension MAE / cm",
        title="Final 3D error, lower is better",
        xticks=range(len(names)),
        xticklabels=names,
    )
    axes[2].tick_params(axis="x", labelrotation=55, labelsize=8)
    axes[2].set_ylim(0, max(max(values) * 1.3, 0.1))
    axes[2].grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values, strict=True):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.03,
            f"{value:.2f}",
            ha="center",
            fontsize=8,
        )
    fig.suptitle(
        f"One development room · same frozen model, bootstrap images and {budget:g}-second action budget",
        fontsize=12,
    )
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def reconstruction_plot(data, output):
    active = next(p for p in data["policies"] if p["name"] == "active")
    fig = plt.figure(figsize=(11, 5), layout="constrained")
    for index, (title, prediction, metrics) in enumerate(
        [
            ("Initial shell", data["initial_reconstruction"], data["initial_metrics"]),
            ("After active head scan", active["final_reconstruction"], active["final_metrics"]),
        ]
    ):
        ax = fig.add_subplot(1, 2, index + 1, projection="3d")
        for bounds, color, style, label in [
            (data["truth_bounds_m"], "#8eabc1", "--", "Reference (evaluation only)"),
            (prediction["bounds_m"], MINT, "-", "Predicted shell"),
        ]:
            for j, segment in enumerate(bounds_segments(bounds)):
                ax.plot(*segment.T, color=color, ls=style, lw=1.8, label=label if j == 0 else None)
        ax.set(
            xlabel="X / m",
            ylabel="Y / m",
            zlabel="Z / m",
            title=f"{title} · {metrics['dimension_mae_m'] * 100:.2f} cm MAE",
        )
        ax.set_box_aspect((6, 5, 3))
        ax.view_init(24, -53)
        ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8, loc="upper left")
    fig.savefig(output / "reconstruction.png", dpi=150)
    plt.close(fig)


def panel(canvas, box, title, subtitle=None):
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(box, radius=14, fill=PANEL)
    draw.text((box[0] + 15, box[1] + 13), title, font=font(17, True), fill=TEXT)
    if subtitle:
        draw.text((box[0] + 15, box[1] + 39), subtitle, font=font(12), fill=MUTED)


def fit_image(canvas, image, box):
    fitted = ImageOps.contain(image, (box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
    x, y = (
        box[0] + (box[2] - box[0] - fitted.width) // 2,
        box[1] + (box[3] - box[1] - fitted.height) // 2,
    )
    canvas.paste(fitted, (x, y))
    return x, y, fitted.width, fitted.height


def camera_panel(canvas, data, pose, overhead):
    box = (20, 146, 480, 510)
    panel(canvas, box, "Furnished room / top-down", "Fixed robot body · head direction in amber")
    x, y, w, h = fit_image(canvas, overhead, (32, 207, 468, 497))
    draw = ImageDraw.Draw(canvas)
    extent = data["capture_top_down"]["extent_m"]

    def pixel(point):
        px, py = floorplan_pixel(point, w, h, extent)
        return x + px, y + py

    base = data["config"]["robot_base"]["position"]
    bx, by = pixel(base)
    draw.ellipse((bx - 8, by - 8, bx + 8, by + 8), outline="white", width=2)
    eye, forward = pose[:3, 3], pose[:3, 2]
    if np.linalg.norm(forward[:2]) < 1e-8:
        forward = np.cross([0, 0, 1], pose[:3, 0])
    origin, direction, triangle = camera_triangle(eye, eye + forward, length=0.65, half_width=0.19)
    draw.line((pixel(origin), pixel(origin + direction * 1.45)), fill=AMBER, width=3)
    draw.polygon([pixel(p) for p in triangle], fill=AMBER, outline="#10151b", width=2)


def directions_panel(canvas, step, pose):
    box = (20, 528, 480, 913)
    panel(
        canvas,
        box,
        "Where has the camera looked?",
        "Direction coverage alone does not reconstruct a surface",
    )
    state = step["directional_state"]
    directions = np.asarray(state["directions_world"])
    seen, reachable = np.asarray(state["looked"]), np.asarray(state["reachable"])
    colors = np.full((36, 72, 3), [39, 53, 69], dtype=np.uint8)
    azimuth = np.arctan2(directions[:, 1], directions[:, 0])
    xs = np.minimum(((azimuth + np.pi) / (2 * np.pi) * 72 + 1e-7).astype(int), 71)
    ys = np.clip(((1 - directions[:, 2]) / 2 * 36).astype(int), 0, 35)
    colors[ys[reachable], xs[reachable]] = [107, 74, 51]
    colors[ys[seen], xs[seen]] = [58, 158, 145]
    heat = Image.fromarray(colors).resize((424, 213), Image.Resampling.NEAREST)
    canvas.paste(heat, (38, 607))
    draw = ImageDraw.Draw(canvas)
    forward = pose[:3, 2]
    az = math.atan2(forward[1], forward[0])
    cx, cy = 38 + (az + np.pi) / (2 * np.pi) * 424, 607 + (1 - forward[2]) / 2 * 213
    draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=AMBER, outline="white", width=2)
    for px, label in [(38, "−180°"), (225, "World azimuth"), (415, "+180°")]:
        draw.text((px, 829), label, font=font(11), fill=MUTED)
    draw.text((38, 586), "+90° elevation", font=font(10), fill=MUTED)
    draw.text((353, 586), "Equal-area cells", font=font(10), fill=MUTED)
    for x, color, label in [
        (38, "#3a9e91", "Looked"),
        (162, "#6b4a33", "Reachable, unseen"),
        (349, "#273545", "Outside reach"),
    ]:
        draw.rectangle((x, 870, x + 10, 880), fill=color)
        draw.text((x + 16, 867), label, font=font(10), fill=MUTED)


def shell_panel(canvas, step, bounds):
    box = (490, 528, 950, 913)
    panel(
        canvas,
        box,
        "Unconfirmed structure / edge samples",
        "Fixed initial shell · prediction agreement is a heuristic",
    )
    points = np.asarray(step["shell_state"]["points_m"])
    state = np.asarray(step["shell_state"]["state"])
    draw = ImageDraw.Draw(canvas)
    if len(points) == 0:
        draw.text((515, 680), "No shell hypothesis is available", font=font(16), fill=MUTED)
        return
    lo, hi = np.array(bounds)[[0, 2, 4]], np.array(bounds)[[1, 3, 5]]
    center = (lo + hi) / 2
    scale = 210 / max(np.max(hi - lo), 1)

    def project(point):
        q = (point - center) * scale
        return 720 + q[0] - 0.65 * q[1], 710 + 0.30 * q[0] + 0.40 * q[1] - q[2]

    for edge in bounds_segments(bounds):
        draw.line([project(point) for point in edge], fill="#384d63", width=2)
    for point, label in zip(points, state, strict=True):
        x, y = project(point)
        draw.ellipse((x - 2.5, y - 2.5, x + 2.5, y + 2.5), fill=STATE_COLORS[str(label)])
    for i, (label, color) in enumerate(STATE_COLORS.items()):
        x, y = 510 + (i % 2) * 219, 860 + (i // 2) * 23
        draw.ellipse((x, y + 2, x + 8, y + 10), fill=color)
        draw.text(
            (x + 15, y),
            f"{STATE_LABELS[label]}  {int((state == label).sum())}",
            font=font(10),
            fill=MUTED,
        )


def evidence_panel(canvas, data, policy, step):
    box = (960, 528, 1420, 913)
    panel(
        canvas,
        box,
        "Acquired evidence / evaluation",
        "Reference labels below are hidden from the planner",
    )
    draw = ImageDraw.Draw(canvas)
    entries = [
        (
            "Visible reference edge coverage",
            step["evaluation"]["visible_edge_coverage"] * 100,
            MINT,
        ),
        (
            "Translated-view reference coverage",
            step["evaluation"]["multi_view_edge_coverage"] * 100,
            "#72baff",
        ),
        (
            "Reachable directions looked at",
            step["diagnostics"].get("looked_direction_fraction", 0) * 100,
            AMBER,
        ),
    ]
    for i, (label, value, color) in enumerate(entries):
        y = 604 + i * 65
        draw.text((981, y), label, font=font(12), fill=MUTED)
        draw.text((1400, y - 4), f"{value:.1f}%", font=font(19, True), fill=color, anchor="ra")
        draw.rounded_rectangle((981, y + 25, 1400, y + 35), radius=5, fill="#293b4c")
        if value > 0:
            draw.rounded_rectangle(
                (981, y + 25, 981 + 419 * np.clip(value / 100, 0, 1), y + 35), radius=5, fill=color
            )
    preview = step.get("preview_plan", [])
    if isinstance(preview, dict):
        preview = preview.get("actions", [])
    planned = " → ".join(str(p.get("view_id", "?")) for p in preview)
    draw.text((981, 814), "Next few seconds (hypothetical, then replan)", font=font(11), fill=MUTED)
    draw.text((981, 837), planned or "No further affordable target", font=font(15, True), fill=TEXT)
    acquired_now = sum(item["t_s"] <= step["t_s"] for item in policy["trace"])
    draw.text(
        (981, 871),
        f"Decision {step.get('decision_ms', 0):.2f} ms · {acquired_now} head captures so far",
        font=font(11),
        fill=MUTED,
    )


def compose_frame(
    data, policy, index, images, overhead, marker=None, motion_frame=None, motion_images=None
):
    step = policy["trace"][index]
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 22), "ROOMGRAPH / ACTIVE HEAD PERCEPTION", font=font(13, True), fill=MINT)
    draw.text((20, 51), "Look toward what is still uncertain", font=font(34, True), fill=TEXT)
    t, yaw, pitch = (step["t_s"], step["yaw_deg"], step["pitch_deg"]) if marker is None else marker
    if motion_frame is not None:
        t, yaw, pitch = (motion_frame[key] for key in ("t_s", "yaw_deg", "pitch_deg"))
    capture_label = (
        f"dense RGB replay · evidence through {step['view_id']}"
        if motion_frame is not None
        else f"captured {step['view_id']}"
    )
    draw.text(
        (22, 105),
        f"{policy_label(policy)}  ·  {t:.2f} s  ·  yaw {yaw:+.0f}°  ·  pitch {pitch:+.0f}°  ·  {capture_label}",
        font=font(16),
        fill=MUTED,
    )
    if motion_frame is not None:
        pose = np.asarray(motion_frame["record"]["camera_to_world"])
    elif marker is None:
        pose = np.asarray(step["record"]["camera_to_world"])
    else:
        base = data["config"]["robot_base"]
        pose = head_camera_pose(
            base["position"], base.get("yaw_deg", 0), yaw, pitch
        ).camera_to_world
    camera_panel(canvas, data, pose, overhead)
    for x, title, kind in [
        (490, "Acquired head-camera RGB", "rgb"),
        (960, "Predicted visible structure", "overlay"),
    ]:
        panel(
            canvas,
            (x, 146, x + 460, 510),
            title,
            (
                "Dense rendered replay (display only)"
                if motion_frame is not None
                else "Actual rendered observation"
            )
            if kind == "rgb"
            else "Frozen model · mint indicates predicted edges",
        )
        fit_image(
            canvas,
            motion_images[kind]
            if motion_images is not None
            else images[(policy["name"], index, kind)],
            (x + 12, 207, x + 448, 497),
        )
    directions_panel(canvas, step, pose)
    shell_panel(canvas, step, data["initial_reconstruction"]["bounds_m"])
    evidence_panel(canvas, data, policy, step)
    draw.text(
        (22, 935),
        (
            "Actual dense Isaac Sim replay; intermediate images are display-only and excluded from policy measurements."
            if motion_frame is not None
            else "Illustrative head-marker motion only; RGB, overlays and evidence are held until the next captured view."
        ),
        font=font(13),
        fill=AMBER,
    )
    draw.text(
        (22, 960),
        (
            "Evidence updates only at selected policy observations · known poses + translated bootstrap · no navigation"
            if motion_frame is not None
            else "Known poses + translated bootstrap views · one development room · no navigation or verified free-space map"
        ),
        font=font(12),
        fill=MUTED,
    )
    return canvas


def public_summary(data, source, motion=None):
    """Whitelist measured values and provenance; omit paths and large evidence snapshots."""
    bootstrap_path = Path(data["config"]["bootstrap_manifest"])
    if not bootstrap_path.is_absolute():
        bootstrap_path = Path(__file__).resolve().parents[1] / bootstrap_path
    bootstrap_bytes = bootstrap_path.read_bytes()
    if hashlib.sha256(bootstrap_bytes).hexdigest() != data["bootstrap_manifest_sha256"]:
        raise ValueError("Bootstrap manifest changed after the frozen experiment")
    dataset = json.loads(bootstrap_bytes)["scene_config"]["dataset"]
    policies = []
    for policy in data["policies"]:
        decisions = [step["decision_ms"] for step in policy["trace"]]
        policies.append(
            {
                "name": policy["name"],
                "seed": policy["seed"],
                "head_captures": len(policy["acquired_head_ids"]),
                "simulated_seconds": policy["simulated_seconds"],
                **policy["trace"][-1]["evaluation"],
                "coverage_auc": policy["coverage_auc"],
                "final_metrics": policy["final_metrics"],
                "mean_decision_ms": float(np.mean(decisions)) if decisions else None,
                "median_decision_ms": float(np.median(decisions)) if decisions else None,
                "reconstruction_seconds": policy["reconstruction_seconds"],
            }
        )
    config_keys = (
        "capture_name",
        "robot_base",
        "yaw_grid_deg",
        "pitch_grid_deg",
        "focal_length_mm",
        "render",
        "network_size",
        "threshold",
        "action_budget_s",
        "planning_horizon_s",
        "motion_limits",
        "random_seeds",
        "note",
    )
    summary = {
        "schema_version": 1,
        "status": data["status"],
        "room": dataset["building_id"],
        "split": dataset["split"],
        "experiment_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "checkpoint_sha256": data["checkpoint_sha256"],
        "bootstrap_manifest_sha256": data["bootstrap_manifest_sha256"],
        "capture_manifest_sha256": data["capture_manifest_sha256"],
        "config": {key: data["config"][key] for key in config_keys},
        "bootstrap_camera_ids": data["bootstrap_camera_ids"],
        "initial_metrics": data["initial_metrics"],
        "initial_coverage": data["policies"][0]["trace"][0]["evaluation"],
        "policies": policies,
        "inference": {
            "median_ms_excluding_first": data["inference_median_ms_excluding_first"],
            "scope": "Batch-one network forward, sigmoid and probability transfer to CPU; excludes image I/O, resize, input transfer, setup and first call",
        },
        "recorded_experiment_wall_seconds": data["wall_seconds"],
        "protocol": data["protocol"],
        "animation": {
            "kind": "illustrative_marker_interpolation",
            "intermediate_images_enter_policy": False,
        },
    }
    summary["protocol"] = {
        **summary["protocol"],
        "decision_timing": "All measured head-plan calls, including the initial plan and stop decision",
    }
    if motion is not None:
        summary["animation"] = {
            "kind": "dense_isaac_replay",
            "frames": len(motion["frames"]),
            "duration_s": motion["frames"][-1]["t_s"] - motion["frames"][0]["t_s"],
            "capture_manifest_sha256": motion["capture_manifest_sha256"],
            "intermediate_images_enter_policy": False,
        }
    return summary


def build_report(source, output, motion_path=None):
    data = json.loads(source.read_text())
    if data.get("status") != "complete":
        raise ValueError("Only completed experiments can produce a final report")
    motion = None
    if motion_path is not None:
        motion = json.loads(motion_path.read_text())
        if motion.get("status") != "complete" or not motion.get("frames"):
            raise ValueError("Motion replay must be complete and contain captured frames")
        if motion.get("experiment_sha256") != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError("Motion replay must match the exact frozen experiment JSON")
        times = np.array([frame["t_s"] for frame in motion["frames"]], dtype=float)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or times[0] < 0:
            raise ValueError("Motion replay timestamps must be finite, nonnegative and increasing")
    output.mkdir(parents=True, exist_ok=True)
    (output / "frames").mkdir(exist_ok=True)
    plot_style()
    comparison_plot(data, output)
    reconstruction_plot(data, output)
    overhead = Image.open(image_path(data["capture_top_down"]["path"], source)).convert("RGB")
    images = {}
    timeline = []
    for policy in data["policies"]:
        frames = []
        for index, step in enumerate(policy["trace"]):
            for kind in ("rgb", "overlay"):
                images[(policy["name"], index, kind)] = Image.open(
                    image_path(step[kind + "_path"], source)
                ).convert("RGB")
            frame = compose_frame(data, policy, index, images, overhead)
            filename = f"frames/{policy['name']}_{index:03d}.webp"
            frame.save(output / filename, quality=88)
            frames.append({"path": filename, "t_s": step["t_s"], "view_id": step["view_id"]})
        timeline.append({"name": policy["name"], "label": policy_label(policy), "frames": frames})
    active = next(p for p in data["policies"] if p["name"] == "active")
    trace = active["trace"]
    frames, durations = [], []
    for index, step in enumerate(trace if motion is None else []):
        if index == len(trace) - 1:
            frame = compose_frame(data, active, index, images, overhead)
            frames.append(frame.resize((1080, 740), Image.Resampling.LANCZOS))
            durations.append(1600)
            break
        next_step = trace[index + 1]
        delta = next_step["t_s"] - step["t_s"]
        count = max(1, int(math.ceil(delta * 4)))
        for subframe in range(count):
            alpha = subframe / count
            marker = tuple(
                (1 - alpha) * step[key] + alpha * next_step[key]
                for key in ("t_s", "yaw_deg", "pitch_deg")
            )
            frame = compose_frame(data, active, index, images, overhead, marker=marker)
            frames.append(frame.resize((1080, 740), Image.Resampling.LANCZOS))
            durations.append(max(40, int(delta * 1000 / count)))
    if motion is not None:
        (output / "replay").mkdir(exist_ok=True)
        for index, replay in enumerate(motion["frames"]):
            trace_index = max(
                i
                for i, observation in enumerate(trace)
                if observation["t_s"] <= replay["t_s"] + 1e-8
            )
            replay_images = {
                kind: Image.open(image_path(replay[kind + "_path"], motion_path)).convert("RGB")
                for kind in ("rgb", "overlay")
            }
            frame = compose_frame(
                data,
                active,
                trace_index,
                images,
                overhead,
                motion_frame=replay,
                motion_images=replay_images,
            )
            frame.save(output / "replay" / f"frame_{index:03d}.webp", lossless=True, method=3)
            if index == 0:
                frame.save(output / "poster.jpg", quality=94)
            frames.append(frame.resize((1080, 740), Image.Resampling.LANCZOS))
            delta_ms = (
                (motion["frames"][index + 1]["t_s"] - replay["t_s"]) * 1000
                if index + 1 < len(motion["frames"])
                else 1600
            )
            durations.append(max(40, round(delta_ms)))
    frames[0].save(
        output / "active_head.gif",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    rows = []
    for policy in data["policies"]:
        last = policy["trace"][-1]
        rows.append(
            f"<tr><th>{html.escape(policy_label(policy))}</th><td>{len(policy['acquired_head_ids'])}</td><td>{policy['simulated_seconds']:.2f}</td><td>{last['evaluation']['visible_edge_coverage'] * 100:.1f}%</td><td>{last['evaluation']['multi_view_edge_coverage'] * 100:.1f}%</td><td>{policy['coverage_auc'] * 100:.1f}%</td><td>{policy['final_metrics']['dimension_mae_m'] * 100:.2f}</td><td>{policy['final_metrics']['edge_chamfer_m'] * 100:.2f}</td></tr>"
        )
    raster = next(p for p in data["policies"] if p["name"] == "raster")
    result_note = (
        f"Active final visible-edge coverage: {active['trace'][-1]['evaluation']['visible_edge_coverage'] * 100:.2f}%; "
        f"raster: {raster['trace'][-1]['evaluation']['visible_edge_coverage'] * 100:.2f}%. "
        f"Final dimension MAE: active {active['final_metrics']['dimension_mae_m'] * 100:.2f} cm; "
        f"raster {raster['final_metrics']['dimension_mae_m'] * 100:.2f} cm. "
        "The comparison measures coverage and 3D error separately; it does not establish that the active policy is generally better."
    )
    decision_times = [step["decision_ms"] for step in active["trace"]]
    decision_note = (
        f"Median active head-plan call, including the initial and stop calls: {np.median(decision_times):.2f} ms. "
        if decision_times
        else "No head movement was selected. "
    )
    runtime_note = (
        decision_note
        + f"Measured image inference, excluding the first call: {data['inference_median_ms_excluding_first']:.2f} ms. "
        f"Active final reconstruction fit: {active['reconstruction_seconds']:.2f} s. "
        "The action clock is simulated head motion; rendering, inference, decision-making and final fitting are not charged to it."
    )
    selected = timeline[[p["name"] for p in timeline].index("active")]
    options = "".join(
        f'<option value="{i}"{" selected" if p["name"] == "active" else ""}>{html.escape(p["label"])}</option>'
        for i, p in enumerate(timeline)
    )
    safe_json = json.dumps(timeline).replace("<", "\\u003c")
    animation_note = (
        "Actual dense Isaac Sim head-camera replay: every displayed RGB and overlay frame comes from its rendered camera pose. "
        "Intermediate replay images are display-only and excluded from policy observations, decisions and benchmark metrics. "
        "Evidence maps update only at the selected policy observations. The motion model remains an illustrative proxy, not validated robot control."
        if motion is not None
        else "Only the amber head marker interpolates between observations. RGB, predictions and evidence stay frozen until the next acquired view. "
        "Marker interpolation illustrates direction changes and is not a rendered continuous video or a validated motor trajectory."
    )
    content = (
        HTML.replace("__OPTIONS__", options)
        .replace("__TIMELINE__", safe_json)
        .replace("__FIRST_FRAME__", selected["frames"][0]["path"])
        .replace("__ROWS__", "".join(rows))
        .replace("__RESULT_NOTE__", html.escape(result_note))
        .replace("__RUNTIME_NOTE__", html.escape(runtime_note))
        .replace("__ANIMATION_NOTE__", html.escape(animation_note))
        .replace("__CHECKPOINT__", html.escape(data["checkpoint_sha256"]))
        .replace("__BOOTSTRAP__", html.escape(", ".join(data["bootstrap_camera_ids"])))
    )
    (output / "report.html").write_text(content)
    (output / "timeline.json").write_text(json.dumps(timeline, indent=2) + "\n")
    (output / "experiment.json").write_text(json.dumps(data, indent=2) + "\n")
    (output / "summary.json").write_text(
        json.dumps(public_summary(data, source, motion), indent=2) + "\n"
    )
    if motion is not None:
        (output / "motion.json").write_text(json.dumps(motion, indent=2) + "\n")
    print(output / "report.html")


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoomGraph · Active head perception</title>
<style>
:root{color-scheme:dark;--bg:#0c1420;--panel:#142131;--text:#e7f0f7;--muted:#9bb0c5;--mint:#55e4bd;--amber:#ffbe70}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.65 system-ui,sans-serif}main{max-width:1480px;margin:auto;padding:52px 24px 70px}a{color:var(--mint)}.eyebrow{color:var(--mint);letter-spacing:.14em;font-weight:700;font-size:12px}h1{font-size:clamp(34px,5vw,64px);line-height:1.1;letter-spacing:-.045em;max-width:950px;margin:18px 0}h2{font-size:28px;letter-spacing:-.025em;margin:0 0 14px}p{max-width:1040px;color:var(--muted)}.lead{font-size:20px;max-width:1000px}.chips{display:flex;gap:10px;flex-wrap:wrap;margin:24px 0}.chip{border:1px solid #2c4559;border-radius:30px;padding:6px 13px;font-size:12px;color:var(--muted)}section{margin-top:50px}.card{background:var(--panel);border:1px solid #294153;border-radius:18px;padding:22px;overflow:hidden}img{max-width:100%;width:100%;height:auto;display:block;border-radius:12px}figure{margin:24px 0}figcaption,.note{font-size:13px;color:var(--muted);margin-top:12px}.notice{border-left:3px solid var(--amber);padding:2px 18px;color:var(--amber)}.controls{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:18px}select,button{font:inherit;color:var(--text);background:#203348;border:1px solid #3c5871;border-radius:8px;padding:8px 13px}button{cursor:pointer}input[type=range]{flex:1;min-width:160px;accent-color:var(--mint)}output{min-width:180px;font-variant-numeric:tabular-nums;color:var(--mint)}table{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}td,th{padding:12px 13px;border-bottom:1px solid #2b4153;text-align:right}th:first-child{text-align:left}thead th{color:var(--muted);font-weight:500}.table-scroll{overflow-x:auto}.details{display:grid;grid-template-columns:1fr 1fr;gap:22px}.details p{font-size:14px}code{word-break:break-all;font-size:12px;color:var(--amber)}footer{margin-top:50px;color:var(--muted);font-size:12px}@media(max-width:700px){main{padding:30px 12px}.card{padding:12px}.details{grid-template-columns:1fr}h2{font-size:24px}.lead{font-size:17px}output{font-size:13px}}
</style></head><body><main>
<div class="eyebrow">ROOMGRAPH / EMBODIED PERCEPTION EXPERIMENT</div>
<h1>A movable head.<br>A more deliberate next view.</h1>
<p class="lead">A frozen structural-edge model observes a furnished room from a humanoid head camera. A lightweight planner selects where to look next, balancing unexplored directions, unsupported structure and head-motion time.</p>
<div class="chips"><span class="chip">One validation room · development pilot</span><span class="chip">Known metric camera poses</span><span class="chip">No new model training</span><span class="chip">Independent head yaw + pitch</span><span class="chip">Fixed robot base</span></div>
<p class="notice">Looking toward a region is different from reconstructing it. Predicted walls and edge agreement remain hypotheses; this experiment does not provide a safe navigation map.</p>
<section><h2>Watch the active head scan</h2><figure><img src="active_head.gif" alt="Animated head direction on a furnished top-down room alongside acquired RGB, edge predictions and evidence maps"><figcaption>__ANIMATION_NOTE__</figcaption></figure></section>
<section class="card"><h2>Inspect each acquired observation</h2><p>Switch policy and move through its actual captures. The initial shell stays fixed during planning; geometry is fitted again only after the scan.</p><div class="controls"><label for="policy">Policy</label><select id="policy">__OPTIONS__</select><button id="play" type="button">Play captures</button><input id="frame" aria-label="Acquired observation" type="range" min="0" value="0" step="1"><output id="frame-label" for="frame"></output></div><img id="capture" src="__FIRST_FRAME__" alt="Selected acquired view with top-down camera direction, predicted edges and accumulated evidence"></section>
<section><h2>Coverage is useful. Final 3D quality decides.</h2><p>Active, raster and seeded random policies use the same frozen checkpoint, initial observations, candidate captures and motion clock. Coverage is evaluated using reference geometry after the decision; the planner does not receive reference edges or unseen candidate images. These are development results from one room, not held-out generalization evidence.</p><p class="notice">__RESULT_NOTE__</p><figure><img src="comparison.png" alt="Measured coverage over simulated head-action time and final dimension error for all policies"><figcaption>Step plots add evidence only at acquisition times and hold the final value to the common action budget. Direction coverage is deliberately excluded from these reference-edge coverage plots.</figcaption></figure><div class="card table-scroll"><table><thead><tr><th>Policy</th><th>Head captures</th><th>Action seconds</th><th>Visible edges</th><th>Translated-view edges</th><th>Mean visible coverage / budget</th><th>Dimension MAE / cm</th><th>Edge Chamfer / cm</th></tr></thead><tbody>__ROWS__</tbody></table></div></section>
<section><h2>Before and after the head scan</h2><figure><img src="reconstruction.png" alt="Initial and final reconstructed shells compared with evaluation-only ground truth"><figcaption>Shell geometry is inferred from RGB edge probabilities and calibrated poses. The reference wireframe is displayed only for evaluation. Small camera translation around the neck cannot replace deliberate robot translation for robust triangulation.</figcaption></figure></section>
<section class="card"><h2>Computation and the action clock</h2><p>__RUNTIME_NOTE__</p><p class="note">Capture renders and model predictions are cached for policy comparison. No model retraining is required for this planner experiment. The initial fit is shared across policies.</p></section>
<section class="details"><div class="card"><h2>What the planner knows</h2><p>Images acquired so far, camera calibration, an initial fitted shell, and candidate head orientations. Its short-horizon plan is hypothetical: execute the first action, capture the image, update evidence and replan.</p><p>Shell colors distinguish unseen samples, inspected samples without prediction support, single-camera-center support and support from translated camera centers. These categories are uncalibrated evidence heuristics.</p></div><div class="card"><h2>What remains to build</h2><p>This pilot uses a fixed body, known poses, a supplied translated bootstrap and an original approximate humanoid proxy. Joint limits are simulation assumptions, not official 1X specifications.</p><p>SLAM, body-motion planning, obstacle/free-space mapping, doorway mesh reconstruction, online geometry updates and real-robot control remain future work. More rooms and repeated measurements are needed before policy comparisons generalize.</p></div></section>
<footer><p>Bootstrap cameras: __BOOTSTRAP__.<br>Frozen checkpoint SHA-256: <code>__CHECKPOINT__</code></p><p><a href="experiment.json">Full measured experiment JSON</a> · <a href="timeline.json">Capture timeline</a> · <a href="active_head.gif">Download animation</a> · All report artifacts remain local.</p></footer>
</main><script>
const policies=__TIMELINE__,select=document.getElementById('policy'),slider=document.getElementById('frame'),picture=document.getElementById('capture'),label=document.getElementById('frame-label'),button=document.getElementById('play');let timer=null;
function stop(){if(timer!==null)clearInterval(timer);timer=null;button.textContent='Play captures';}
function update(){const policy=policies[Number(select.value)];slider.max=policy.frames.length-1;slider.value=Math.min(Number(slider.value),policy.frames.length-1);const frame=policy.frames[Number(slider.value)];picture.src=frame.path;label.textContent=frame.view_id+' · '+frame.t_s.toFixed(2)+' s · '+(Number(slider.value)+1)+'/'+policy.frames.length;}
select.addEventListener('change',()=>{stop();slider.value=0;update();});slider.addEventListener('input',()=>{stop();update();});button.addEventListener('click',()=>{if(timer!==null){stop();return;}button.textContent='Pause captures';timer=setInterval(()=>{slider.value=(Number(slider.value)+1)%(Number(slider.max)+1);update();},900);});update();
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("vis/active_head/run_v1/experiment.json")
    )
    parser.add_argument("--output", type=Path, default=Path("vis/active_head/report"))
    parser.add_argument(
        "--motion",
        type=Path,
        help="Optional complete dense replay JSON; display-only, not policy evidence",
    )
    args = parser.parse_args()
    build_report(args.results, args.output, args.motion)


if __name__ == "__main__":
    main()
